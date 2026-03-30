from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Single-head MLP for the ERM baseline.

    Architecture:
        input (input_dim) → Linear → ReLU → Linear → ReLU  [backbone]
                                                          → Linear → 2  [head]

    input_dim is passed explicitly (not hardcoded) so the model is not coupled
    to CMNIST's image shape. Compute it from the dataset before constructing:
        input_dim = train_dataset.images.shape[1:].numel()  # 3*28*28 = 2352

    The backbone/head split is explicit so SplitMLP (Step 3) can share the
    backbone and attach two independent heads without duplicating this code.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, 2)
        # Bias initialisation: for balanced classes (base rate ~50/50), log(p/(1-p)) = 0,
        # which matches PyTorch's default. If we move to imbalanced datasets (Waterbirds,
        # CelebA), initialise head.bias to log(base_rate / (1 - base_rate)) to avoid
        # wasting early epochs recovering from a miscalibrated starting point.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten(1) collapses all dims from axis 1 onward, leaving the batch dim intact.
        # (B, 3, 28, 28) → (B, 2352) → logits (B, 2)
        return self.head(self.backbone(x.flatten(1)))
