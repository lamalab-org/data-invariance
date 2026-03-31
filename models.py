from __future__ import annotations

import torch
import torch.nn as nn


def _make_backbone(input_dim: int, hidden_dim: int) -> nn.Sequential:
    """Shared backbone factory used by both MLP and SplitMLP.

    Two hidden layers with ReLU activations. Factored out so both model
    classes have identical backbone architecture by construction.
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
    )


class MLP(nn.Module):
    """Single-head MLP for the ERM baseline.

    Architecture:
        input (input_dim) → Linear → ReLU → Linear → ReLU  [backbone]
                                                          → Linear → 2  [head]

    input_dim is passed explicitly (not hardcoded) so the model is not coupled
    to CMNIST's image shape. Compute it from the dataset before constructing:
        input_dim = train_dataset.images.shape[1:].numel()  # 3*28*28 = 2352

    The backbone/head split is explicit so SplitMLP can share the same
    backbone architecture without duplicating this code.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.backbone = _make_backbone(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 2)
        # Bias initialisation: for balanced classes (base rate ~50/50), log(p/(1-p)) = 0,
        # which matches PyTorch's default. If we move to imbalanced datasets (Waterbirds,
        # CelebA), initialise head.bias to log(base_rate / (1 - base_rate)) to avoid
        # wasting early epochs recovering from a miscalibrated starting point.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten(1) collapses all dims from axis 1 onward, leaving the batch dim intact.
        # (B, 3, 28, 28) → (B, 2352) → logits (B, 2)
        return self.head(self.backbone(x.flatten(1)))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities (B, 2). Used by evaluate() for model-agnostic eval."""
        return self.forward(x).softmax(dim=1)


class SplitMLP(nn.Module):
    """Two-head MLP for split-based training methods (random split, adversarial split).

    Two independent heads produce separate predictions. Training methods assign each
    example to one head via soft weights s_i ∈ [0,1] and penalise disagreement.

    separate_backbones=False (default): one shared backbone feeds both heads.
        Both heads see identical features — only their linear maps can differ.
        This is cheap but limits specialisation: the representation is shared.

    separate_backbones=True: each head has its own backbone, independently
        initialised and updated. The two pathways can learn genuinely different
        representations, which is necessary for the adversarial partition to
        cause meaningful specialisation. Costs 2× the parameters and compute.

    forward() returns (logits_a, logits_b) — raw logits from both heads.
    get_features() returns (features_a, features_b) — used by the adversarial
        training loop to inject per-head noise before the linear heads.
    predict() returns averaged probabilities for evaluation (mixture model).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, separate_backbones: bool = False) -> None:
        super().__init__()
        self.separate_backbones = separate_backbones
        if separate_backbones:
            # Independent initialisations — the two pathways start from different
            # random weights and can diverge freely under the adversarial partition.
            self.backbone_a = _make_backbone(input_dim, hidden_dim)
            self.backbone_b = _make_backbone(input_dim, hidden_dim)
        else:
            self.backbone = _make_backbone(input_dim, hidden_dim)
        self.head_a = nn.Linear(hidden_dim, 2)
        self.head_b = nn.Linear(hidden_dim, 2)

    def get_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (features_a, features_b) — the backbone outputs for each head.

        With a shared backbone both tensors are the same object (no extra compute).
        With separate backbones each is an independent forward pass through its own
        backbone. Used by the adversarial training loop to inject per-head noise
        before the linear heads without duplicating the backbone/head split logic.
        """
        flat = x.flatten(1)   # (B, input_dim)
        if self.separate_backbones:
            return self.backbone_a(flat), self.backbone_b(flat)   # independent
        f = self.backbone(flat)
        return f, f   # same tensor — no overhead vs the original forward

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features_a, features_b = self.get_features(x)   # (B, hidden_dim) each
        return self.head_a(features_a), self.head_b(features_b)   # each (B, 2)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return averaged class probabilities (B, 2). Used by evaluate()."""
        logits_a, logits_b = self.forward(x)
        return (logits_a.softmax(dim=1) + logits_b.softmax(dim=1)) / 2


class MultiHeadMLP(nn.Module):
    """K-head MLP for adversarial split with K>2 environments.

    Generalises SplitMLP from binary (sigmoid) to K-way (softmax) assignments.
    Assignment is a K-simplex vector s_i ∈ Δ^{K-1} per training example, learned
    by (N, K) logits passed through softmax.

    K=2 is a valid choice and gives the same structural setup as SplitMLP, though
    the parameterisation differs (one extra logit per example vs one).  Use SplitMLP
    for binary ablations so results are directly comparable to the existing baselines.

    separate_backbones=False: one shared backbone; heads can only differ in their
        linear maps.
    separate_backbones=True: K independent backbones, one per head; each pathway
        can learn a genuinely different representation.

    forward() returns a list of K logit tensors, each (B, 2).
    get_all_features() returns a list of K feature tensors, each (B, hidden_dim),
        used by the training loop for per-head noise injection.
    predict() returns the mean of K softmax distributions (mixture model).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        separate_backbones: bool = False,
    ) -> None:
        super().__init__()
        assert num_heads >= 2, "num_heads must be at least 2"
        self.num_heads = num_heads
        self.separate_backbones = separate_backbones

        if separate_backbones:
            self.backbones = nn.ModuleList(
                [_make_backbone(input_dim, hidden_dim) for _ in range(num_heads)]
            )
        else:
            self.backbone = _make_backbone(input_dim, hidden_dim)

        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 2) for _ in range(num_heads)]
        )

    def get_all_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return a list of num_heads feature tensors (B, hidden_dim).

        With a shared backbone all entries are the same tensor (no extra compute).
        With separate backbones each is an independent forward pass, so the K
        pathways can learn genuinely different representations.
        """
        flat = x.flatten(1)   # (B, input_dim)
        if self.separate_backbones:
            return [bb(flat) for bb in self.backbones]
        f = self.backbone(flat)
        return [f] * self.num_heads   # same object — no overhead

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return list of num_heads logit tensors, each (B, 2)."""
        features = self.get_all_features(x)
        return [head(f) for head, f in zip(self.heads, features)]

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return averaged class probabilities (B, 2). Used by evaluate()."""
        logits_list = self.forward(x)
        return sum(l.softmax(dim=1) for l in logits_list) / self.num_heads
