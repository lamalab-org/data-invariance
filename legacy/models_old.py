from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


def _make_mlp_backbone(input_dim: int, hidden_dim: int) -> tuple[nn.Sequential, int]:
    """MLP backbone: flatten → two hidden layers with ReLU.

    Returns (backbone_module, output_dim) so all backbone factories have the
    same interface.  The Flatten(1) is inside the backbone so that forward()
    can call self.backbone(x) without knowing what kind of backbone it is.
    """
    backbone = nn.Sequential(
        nn.Flatten(1),                        # (B, C, H, W) → (B, input_dim)
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
    )
    return backbone, hidden_dim


def make_resnet_backbone(freeze: bool = False) -> tuple[nn.Module, int]:
    """ResNet-50 pretrained on ImageNet, final classification layer removed.

    Returns (backbone_module, output_dim=2048).

    The backbone takes (B, 3, 224, 224) images normalised with ImageNet
    statistics and outputs (B, 2048) feature vectors.

    Args:
        freeze: If True, freeze all backbone weights.  Useful for the
                throw-away discovery ERM where we only need to train the
                final linear layer (much faster, 5 epochs is enough).
    """
    resnet = torchvision.models.resnet50(weights="IMAGENET1K_V1")
    # Everything except the final fc layer.
    modules = list(resnet.children())[:-1]
    backbone = nn.Sequential(*modules, nn.Flatten(1))

    if freeze:
        for param in backbone.parameters():
            param.requires_grad = False

    return backbone, 2048


class DistilBertBackbone(nn.Module):
    """DistilBERT backbone for text classification.

    Takes tokenized text (input_ids, attention_mask) and returns the [CLS]
    token representation (B, 768).  Pretrained on English text.

    Args:
        freeze: If True, freeze all DistilBERT weights (train only the head).
    """

    def __init__(self, freeze: bool = False) -> None:
        super().__init__()
        from transformers import DistilBertModel
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """x is a dict with 'input_ids' and 'attention_mask', each (B, seq_len)."""
        out = self.bert(input_ids=x["input_ids"], attention_mask=x["attention_mask"])
        return out.last_hidden_state[:, 0]  # [CLS] token, (B, 768)


def make_distilbert_backbone(freeze: bool = False) -> tuple[nn.Module, int]:
    """DistilBERT backbone. Returns (module, 768)."""
    return DistilBertBackbone(freeze=freeze), 768


class MLP(nn.Module):
    """Single-head model for ERM and discovered_split.

    Works with any backbone (MLP or ResNet).  The backbone handles its own
    input format — MLP flattens internally, ResNet expects (B, 3, 224, 224).

    Two construction modes:
        MLP(input_dim=2352, hidden_dim=256)        # MLP backbone (CMNIST)
        MLP(backbone=resnet_bb, backbone_out_dim=2048)  # ResNet backbone (Waterbirds)
    """

    def __init__(
        self,
        input_dim: int | None = None,
        hidden_dim: int = 256,
        backbone: nn.Module | None = None,
        backbone_out_dim: int | None = None,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if backbone is not None:
            self.backbone = backbone
            self.head = nn.Linear(backbone_out_dim, num_classes)
        else:
            self.backbone, out_dim = _make_mlp_backbone(input_dim, hidden_dim)
            self.head = nn.Linear(out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities (B, 2). Used by evaluate() for model-agnostic eval."""
        return self.forward(x).softmax(dim=1)


class SplitMLP(nn.Module):
    """Two-head model for split-based training (random, oracle, adversarial).

    Two independent heads produce separate predictions.  Training methods assign
    each example to one head via soft weights s_i in [0,1] and penalise disagreement.

    separate_backbones=False (default): one shared backbone feeds both heads.
    separate_backbones=True: each head has its own backbone.

    Two construction modes (same as MLP):
        SplitMLP(input_dim=2352, hidden_dim=256)
        SplitMLP(backbone=bb, backbone_out_dim=2048)
    """

    def __init__(
        self,
        input_dim: int | None = None,
        hidden_dim: int = 256,
        separate_backbones: bool = False,
        backbone: nn.Module | None = None,
        backbone_out_dim: int | None = None,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.separate_backbones = separate_backbones

        if backbone is not None:
            out_dim = backbone_out_dim
            if separate_backbones:
                raise ValueError("separate_backbones not supported with external backbone")
            self.backbone = backbone
        else:
            out_dim = hidden_dim
            if separate_backbones:
                self.backbone_a, _ = _make_mlp_backbone(input_dim, hidden_dim)
                self.backbone_b, _ = _make_mlp_backbone(input_dim, hidden_dim)
            else:
                self.backbone, _ = _make_mlp_backbone(input_dim, hidden_dim)

        self.head_a = nn.Linear(out_dim, num_classes)
        self.head_b = nn.Linear(out_dim, num_classes)

    def get_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (features_a, features_b) from the backbone(s)."""
        if self.separate_backbones:
            return self.backbone_a(x), self.backbone_b(x)
        f = self.backbone(x)
        return f, f

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features_a, features_b = self.get_features(x)
        return self.head_a(features_a), self.head_b(features_b)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return averaged class probabilities (B, 2). Used by evaluate()."""
        logits_a, logits_b = self.forward(x)
        return (logits_a.softmax(dim=1) + logits_b.softmax(dim=1)) / 2


class MultiHeadMLP(nn.Module):
    """K-head model for adversarial split with K>2 environments.

    Generalises SplitMLP from binary (sigmoid) to K-way (softmax) assignments.

    Two construction modes (same as MLP):
        MultiHeadMLP(input_dim=2352, hidden_dim=256, num_heads=4)
        MultiHeadMLP(backbone=bb, backbone_out_dim=2048, num_heads=4)
    """

    def __init__(
        self,
        input_dim: int | None = None,
        hidden_dim: int = 256,
        num_heads: int = 4,
        separate_backbones: bool = False,
        backbone: nn.Module | None = None,
        backbone_out_dim: int | None = None,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        assert num_heads >= 2, "num_heads must be at least 2"
        self.num_heads = num_heads
        self.separate_backbones = separate_backbones

        if backbone is not None:
            out_dim = backbone_out_dim
            if separate_backbones:
                raise ValueError("separate_backbones not supported with external backbone")
            self.backbone = backbone
        else:
            out_dim = hidden_dim
            if separate_backbones:
                self.backbones = nn.ModuleList(
                    [_make_mlp_backbone(input_dim, hidden_dim)[0] for _ in range(num_heads)]
                )
            else:
                self.backbone, _ = _make_mlp_backbone(input_dim, hidden_dim)

        self.heads = nn.ModuleList(
            [nn.Linear(out_dim, num_classes) for _ in range(num_heads)]
        )

    def get_all_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return a list of num_heads feature tensors."""
        if self.separate_backbones:
            return [bb(x) for bb in self.backbones]
        f = self.backbone(x)
        return [f] * self.num_heads

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return list of num_heads logit tensors, each (B, 2)."""
        features = self.get_all_features(x)
        return [head(f) for head, f in zip(self.heads, features)]

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return averaged class probabilities (B, 2). Used by evaluate()."""
        logits_list = self.forward(x)
        return sum(l.softmax(dim=1) for l in logits_list) / self.num_heads
