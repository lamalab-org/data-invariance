from __future__ import annotations

import torch
import torchvision
from torch.utils.data import Dataset


class ColoredMNIST(Dataset):
    """Colored MNIST dataset for studying spurious correlations.

    Data generation pipeline (deterministic given `seed`):

    1. Load MNIST and binarise digits:
          label = 1 if digit >= 5 else 0
       This gives a balanced binary classification task independent of the
       10-class structure of raw MNIST.

    2. Apply label noise:
       With probability `label_noise`, flip each label (0→1 or 1→0).
       Implemented via XOR with a Bernoulli draw: `label ^ flip_bit`.
       - XOR on {0,1}: x^1 = 1-x (flip), x^0 = x (keep).
       - label_noise=0.25 matches the original CMNIST paper. It prevents
         the model from perfectly predicting from color even in-distribution,
         making the task genuinely non-trivial.

    3. Assign color:
       With probability `env_correlation`, color = label (spurious cue aligned).
       Otherwise color = 1 - label (misaligned).
       - train_correlation=0.9 → color is a very reliable predictor at train time.
       - test_correlation=0.1  → color is mostly anti-correlated at test time.
       - A model that memorises color will fail OOD. That is the distribution shift
         this entire project is designed to study.

    4. Build 3-channel image (shape: 3×28×28):
       - channel 0 (red)   receives grayscale values where color == 0
       - channel 1 (green) receives grayscale values where color == 1
       - channel 2 (blue)  is always zero
       Pixel values are in [0, 1] (normalised from uint8 MNIST).

    Args:
        env_correlation: P(color = label). Use 0.9 for train, 0.1 for test.
        label_noise:     P(label is flipped before color assignment). Default 0.25.
        split:           "train" or "test" — passed to torchvision to select MNIST split.
        data_dir:        Directory where MNIST will be downloaded/cached.
        seed:            Seeds a local torch.Generator for all randomness (noise draws +
                         color draws). Does not affect global RNG state.
    """

    def __init__(
        self,
        env_correlation: float,
        label_noise: float = 0.25,
        split: str = "train",
        data_dir: str = "./data",
        seed: int = 42,
    ) -> None:
        # Local generator — keeps dataset construction independent of global RNG,
        # so model weight initialisation and data randomness don't interfere.
        g = torch.Generator().manual_seed(seed)

        # train=(split == "train") converts our string arg to the bool that
        # torchvision.datasets.MNIST expects for its `train` parameter.
        mnist = torchvision.datasets.MNIST(data_dir, train=(split == "train"), download=True)
        images = mnist.data.float() / 255.0  # (N, 28, 28), values in [0, 1]
        digits = mnist.targets               # (N,), values 0–9

        N = len(digits)

        # --- Step 1: binarise ---
        # .long() produces int64, which F.cross_entropy requires for class indices.
        labels = (digits >= 5).long()        # (N,) ∈ {0, 1}

        # --- Step 2: label noise via XOR ---
        # torch.bernoulli returns a float tensor of 0.0/1.0; .bool() makes it a mask.
        # XOR on int64: label ^ 1 = 1-label (flip), label ^ 0 = label (keep).
        flip = torch.bernoulli(torch.full((N,), label_noise), generator=g).long()
        labels = labels ^ flip               # (N,) ∈ {0, 1}

        # --- Step 3: color assignment ---
        # color_matches[i]=True  → color[i] = label[i]   (spurious cue is reliable)
        # color_matches[i]=False → color[i] = 1-label[i] (spurious cue is misleading)
        color_matches = torch.bernoulli(torch.full((N,), env_correlation), generator=g).bool()
        colors = torch.where(color_matches, labels, 1 - labels)  # (N,) ∈ {0, 1}

        # --- Step 4: build 3-channel images ---
        imgs = torch.zeros(N, 3, 28, 28)
        imgs[colors == 0, 0] = images[colors == 0]  # red   channel for color=0 examples
        imgs[colors == 1, 1] = images[colors == 1]  # green channel for color=1 examples
        # blue channel (index 2) stays zero — no information there

        self.images = imgs    # (N, 3, 28, 28) float32
        self.labels = labels  # (N,) int64
        self.colors = colors  # (N,) int64

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": self.images[idx],         # (3, 28, 28) float32
            "label": self.labels[idx].item(),  # int ∈ {0, 1}
            "color": self.colors[idx].item(),  # int ∈ {0, 1}
            "index": idx,                      # int — used by adversarial assignment weights
        }

    def __len__(self) -> int:
        return len(self.labels)
