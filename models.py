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

    The shared backbone produces a single feature vector per example. Two independent
    heads produce separate predictions. Training methods assign each example to one head
    via soft weights s_i ∈ [0,1] and penalise disagreement between the heads.

    forward() returns (logits_a, logits_b) — raw logits from both heads on the same input.
    This lets the training loop compute task losses and the KL disagreement term in one pass.

    predict() returns averaged probabilities for evaluation, so we compare fairly with ERM's
    single head. Averaging is principled: it is the mixture model (equal weight on each head).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.backbone = _make_backbone(input_dim, hidden_dim)
        self.head_a = nn.Linear(hidden_dim, 2)
        self.head_b = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # flatten(1): (B, 3, 28, 28) → (B, input_dim)
        features = self.backbone(x.flatten(1))   # (B, hidden_dim)
        return self.head_a(features), self.head_b(features)  # each (B, 2)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return averaged class probabilities (B, 2). Used by evaluate()."""
        logits_a, logits_b = self.forward(x)
        # Average the two probability distributions — the mixture model
        return (logits_a.softmax(dim=1) + logits_b.softmax(dim=1)) / 2
