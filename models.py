"""Model architectures: MLP with pluggable backbones.

Backbones:
  - MLP: two hidden layers with ReLU (for tabular / CMNIST)
  - ResNet-50: pretrained on ImageNet (for Waterbirds / CelebA)
  - DistilBERT: pretrained on English text (for CivilComments)

The MLP class accepts any backbone and adds a linear classification head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


def _make_mlp_backbone(input_dim: int, hidden_dim: int) -> tuple[nn.Sequential, int]:
    """MLP backbone: flatten → two hidden layers with ReLU."""
    backbone = nn.Sequential(
        nn.Flatten(1),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
    )
    return backbone, hidden_dim


def make_resnet_backbone(freeze: bool = False) -> tuple[nn.Module, int]:
    """ResNet-50 pretrained on ImageNet, final fc layer removed.
    Returns (backbone, 2048)."""
    resnet = torchvision.models.resnet50(weights="IMAGENET1K_V1")
    modules = list(resnet.children())[:-1]
    backbone = nn.Sequential(*modules, nn.Flatten(1))
    if freeze:
        for param in backbone.parameters():
            param.requires_grad = False
    return backbone, 2048


class DistilBertBackbone(nn.Module):
    """DistilBERT backbone for text. Takes {input_ids, attention_mask} → (B, 768)."""

    def __init__(self, freeze: bool = False) -> None:
        super().__init__()
        from transformers import DistilBertModel
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.bert(input_ids=x["input_ids"], attention_mask=x["attention_mask"])
        return out.last_hidden_state[:, 0]  # [CLS] token


def make_distilbert_backbone(freeze: bool = False) -> tuple[nn.Module, int]:
    """DistilBERT backbone. Returns (module, 768)."""
    return DistilBertBackbone(freeze=freeze), 768


class ChemBertaBackbone(nn.Module):
    """ChemBERTa-77M-MTR backbone for SMILES. {input_ids, attention_mask} → (B, H)."""

    def __init__(self, model_name: str, freeze: bool = False) -> None:
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    @property
    def hidden_size(self) -> int:
        return self.encoder.config.hidden_size

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.encoder(input_ids=x["input_ids"], attention_mask=x["attention_mask"])
        return out.last_hidden_state[:, 0]  # [CLS] / <s> token


def make_chemberta_backbone(
    model_name: str = "DeepChem/ChemBERTa-77M-MTR", freeze: bool = False
) -> tuple[nn.Module, int]:
    """ChemBERTa backbone. Returns (module, hidden_size)."""
    bb = ChemBertaBackbone(model_name, freeze=freeze)
    return bb, bb.hidden_size


class GINBackbone(nn.Module):
    """Small GIN encoder for molecular graphs.

    3 GINConv layers with mean-pool readout; from-scratch (no pretrained
    chemistry features).  Takes a torch_geometric.data.Batch (forwarded
    by the dataloader as the ``image`` field) and returns (B, hidden_dim).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, n_layers: int = 3) -> None:
        super().__init__()
        from torch_geometric.nn import GINConv
        self.atom_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
        self.norm = nn.LayerNorm(hidden_dim)
        self._hidden_dim = hidden_dim

    @property
    def hidden_size(self) -> int:
        return self._hidden_dim

    def forward(self, data) -> torch.Tensor:
        from torch_geometric.utils import scatter
        x = self.atom_proj(data.x)
        for conv in self.convs:
            x = torch.relu(conv(x, data.edge_index)) + x
        x = self.norm(x)
        return scatter(x, data.batch, dim=0, reduce="mean")


def make_gin_backbone(in_dim: int, hidden_dim: int = 128) -> tuple[nn.Module, int]:
    bb = GINBackbone(in_dim=in_dim, hidden_dim=hidden_dim)
    return bb, bb.hidden_size


class MLP(nn.Module):
    """Single-head classifier with pluggable backbone.

    Two construction modes:
        MLP(input_dim=2352, hidden_dim=256)           # MLP backbone
        MLP(backbone=resnet_bb, backbone_out_dim=2048) # external backbone
    """

    def __init__(self, input_dim: int | None = None, hidden_dim: int = 256,
                 backbone: nn.Module | None = None, backbone_out_dim: int | None = None,
                 num_classes: int = 2) -> None:
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
        """Return class probabilities (B, num_classes)."""
        return self.forward(x).softmax(dim=1)
