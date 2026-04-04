from __future__ import annotations

from pathlib import Path

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms


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

    @property
    def spurious(self) -> torch.Tensor:
        """Spurious attribute (color). Generic name so downstream code works for any dataset."""
        return self.colors

    @property
    def input_dim(self) -> int:
        """Flattened input dimensionality for MLP models."""
        return self.images.shape[1:].numel()  # 3*28*28 = 2352

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": self.images[idx],         # (3, 28, 28) float32
            "label": self.labels[idx].item(),  # int ∈ {0, 1}
            "color": self.colors[idx].item(),  # int ∈ {0, 1} (CMNIST-specific)
            "spurious": self.colors[idx].item(),  # generic name for spurious attribute
            "index": idx,                      # int — used by adversarial assignment weights
        }

    def __len__(self) -> int:
        return len(self.labels)


class ContinuousCMNIST(Dataset):
    """CMNIST with continuous (graded) spurious colour signal.

    Instead of binary red/green, each example gets a colour strength
    c_i in [0, 1] that determines the R/G channel blend:
        R_channel = c_i * grayscale
        G_channel = (1 - c_i) * grayscale

    c_i is correlated with the label via a Beta distribution:
        label=1 → c_i ~ Beta(α, β)  with mean ≈ env_correlation
        label=0 → c_i ~ Beta(β, α)  with mean ≈ 1 - env_correlation

    Higher env_correlation → stronger colour-label link, but it's continuous,
    not a clean binary split.  A model can't just threshold on "R > G" — it
    has to handle the full spectrum of colour strengths.

    This models real-world scenarios where the spurious feature has graded
    strength (scaffold frequency in chemistry, image quality in medical data).

    The ``.spurious`` attribute returns a binarised version (c_i > 0.5) for
    compatibility with diagnostics, but the underlying signal is continuous.
    ``.color_strength`` provides the raw continuous values.
    """

    def __init__(
        self,
        env_correlation: float,
        label_noise: float = 0.25,
        split: str = "train",
        data_dir: str = "./data",
        seed: int = 42,
        beta_concentration: float = 5.0,
    ) -> None:
        g = torch.Generator().manual_seed(seed)

        mnist = torchvision.datasets.MNIST(data_dir, train=(split == "train"), download=True)
        images = mnist.data.float() / 255.0
        digits = mnist.targets
        N = len(digits)

        # Step 1: binarise
        labels = (digits >= 5).long()

        # Step 2: label noise
        flip = torch.bernoulli(torch.full((N,), label_noise), generator=g).long()
        labels = labels ^ flip

        # Step 3: continuous colour strength via Beta distribution.
        # Beta(a, b) has mean a/(a+b).  We want mean = env_correlation for
        # label=1 and mean = 1-env_correlation for label=0.
        # With concentration k: a = k*mean, b = k*(1-mean).
        k = beta_concentration
        alpha_pos = k * env_correlation
        beta_pos = k * (1.0 - env_correlation)

        # Sample colour strength for each example.
        import numpy as np
        rng = np.random.RandomState(seed + 1)  # separate from torch generator
        color_strength = torch.zeros(N)
        for i in range(N):
            if labels[i] == 1:
                color_strength[i] = rng.beta(alpha_pos, beta_pos)
            else:
                color_strength[i] = rng.beta(beta_pos, alpha_pos)

        # Step 4: blend R and G channels by colour strength.
        # c=1 → all red, c=0 → all green, c=0.5 → equal mix.
        imgs = torch.zeros(N, 3, 28, 28)
        imgs[:, 0] = images * color_strength.unsqueeze(1).unsqueeze(2)      # R channel
        imgs[:, 1] = images * (1 - color_strength).unsqueeze(1).unsqueeze(2)  # G channel

        self.images = imgs
        self.labels = labels
        self._color_strength = color_strength  # (N,) float ∈ [0, 1]

    @property
    def spurious(self) -> torch.Tensor:
        """Binarised colour (c > 0.5) for diagnostic compatibility."""
        return (self._color_strength > 0.5).long()

    @property
    def color_strength(self) -> torch.Tensor:
        """Raw continuous colour strength in [0, 1]."""
        return self._color_strength

    @property
    def input_dim(self) -> int:
        return self.images.shape[1:].numel()

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": self.images[idx],
            "label": self.labels[idx].item(),
            "spurious": (self._color_strength[idx] > 0.5).long().item(),
            "index": idx,
        }

    def __len__(self) -> int:
        return len(self.labels)


class MultiSpuriousCMNIST(Dataset):
    """CMNIST with two independent spurious features: colour AND brightness.

    Extends ColoredMNIST with a second shortcut.  A model that relies on
    *either* colour or brightness fails OOD when both are flipped.

    Feature 1 — Colour: digit placed in red (color=0) or green (color=1) channel.
        Correlated with label at ``color_correlation``.
    Feature 2 — Brightness: pixel intensities scaled by 0.3 (dim=0) or 1.0 (bright=1).
        Correlated with label at ``brightness_correlation``.

    Both features are sampled independently per example.  This creates 4 spurious
    groups per label (colour x brightness), 8 groups total.  The hardest group has
    both features misaligned with the label.

    At test time, set both correlations low (e.g. 0.1) to flip both shortcuts.
    A model relying on either shortcut fails.

    The ``.spurious`` attribute returns colour (the primary spurious feature)
    for compatibility with diagnostics.  ``.brightness`` provides the second
    feature separately.  ``.group`` returns the full (label, colour, brightness)
    group index (0-7) for worst-group evaluation.
    """

    def __init__(
        self,
        color_correlation: float,
        brightness_correlation: float,
        label_noise: float = 0.25,
        split: str = "train",
        data_dir: str = "./data",
        seed: int = 42,
    ) -> None:
        g = torch.Generator().manual_seed(seed)

        mnist = torchvision.datasets.MNIST(data_dir, train=(split == "train"), download=True)
        images = mnist.data.float() / 255.0  # (N, 28, 28)
        digits = mnist.targets

        N = len(digits)

        # Step 1: binarise
        labels = (digits >= 5).long()

        # Step 2: label noise
        flip = torch.bernoulli(torch.full((N,), label_noise), generator=g).long()
        labels = labels ^ flip

        # Step 3a: colour assignment (same as standard CMNIST)
        color_matches = torch.bernoulli(torch.full((N,), color_correlation), generator=g).bool()
        colors = torch.where(color_matches, labels, 1 - labels)

        # Step 3b: brightness assignment (independent second spurious feature)
        # bright=1 correlates with label=1; dim=0 correlates with label=0
        bright_matches = torch.bernoulli(torch.full((N,), brightness_correlation), generator=g).bool()
        brightness = torch.where(bright_matches, labels, 1 - labels)  # (N,) ∈ {0, 1}

        # Step 4: build images — colour determines channel, brightness scales intensity
        dim_factor = 0.3
        scale = torch.where(brightness == 1, torch.tensor(1.0), torch.tensor(dim_factor))  # (N,)

        imgs = torch.zeros(N, 3, 28, 28)
        for i in range(N):
            channel = colors[i].item()
            imgs[i, channel] = images[i] * scale[i]

        self.images = imgs          # (N, 3, 28, 28)
        self.labels = labels        # (N,) int64
        self.colors = colors        # (N,) int64 — spurious feature 1
        self._brightness = brightness  # (N,) int64 — spurious feature 2

    @property
    def spurious(self) -> torch.Tensor:
        """Primary spurious attribute (colour) for diagnostic compatibility."""
        return self.colors

    @property
    def brightness(self) -> torch.Tensor:
        """Second spurious attribute (brightness)."""
        return self._brightness

    @property
    def input_dim(self) -> int:
        return self.images.shape[1:].numel()

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": self.images[idx],
            "label": self.labels[idx].item(),
            "spurious": self.colors[idx].item(),
            "index": idx,
        }

    def __len__(self) -> int:
        return len(self.labels)


# ImageNet statistics used to normalise inputs for pretrained ResNet.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _waterbirds_transform(split: str) -> transforms.Compose:
    """Standard transforms for Waterbirds following Sagawa et al. (2020).

    Train: random crop + flip for augmentation.
    Val/Test: deterministic resize + center crop for reproducible evaluation.
    Both: ImageNet normalisation (required because the ResNet backbone was
    pretrained on ImageNet-normalised inputs).
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])


class WaterbirdsDataset(Dataset):
    """Waterbirds: bird type classification with background as spurious feature.

    From Sagawa et al. (2020).  Waterbirds appear on water backgrounds 95%
    of the time in training; landbirds appear on land backgrounds 95% of the
    time.  A model that uses background as a shortcut fails on the minority
    groups (waterbird on land, landbird on water).

    Loaded via the Hugging Face ``datasets`` library from ``grodino/waterbirds``.
    The library handles downloading and caching automatically.

    Standard splits:
        train:       4,795 examples (imbalanced — rarest group has only 56)
        validation:  1,199 examples (balanced across groups)
        test:        5,794 examples (balanced across groups)

    Group composition in training:
        landbird  + land:   3,498  (73%)
        waterbird + water:  1,057  (22%)
        landbird  + water:    184  ( 4%)
        waterbird + land:      56  ( 1%)  ← hardest group

    Args:
        split:     "train", "val", or "test".
        data_dir:  Cache directory for the Hugging Face download.
        transform: Optional custom transform; if None, uses the standard
                   ImageNet-normalised crop/resize.
    """

    def __init__(
        self,
        split: str = "train",
        data_dir: str = "./data/waterbirds",
        transform: transforms.Compose | None = None,
    ) -> None:
        from datasets import load_dataset

        # Map our split names to HF split names.
        hf_split = {"train": "train", "val": "validation", "test": "test"}[split]
        ds = load_dataset("grodino/waterbirds", split=hf_split, cache_dir=data_dir)

        # Store labels and spurious attributes as tensors for the .spurious
        # and .labels interface used by discover_environments and evaluate.
        self._labels = torch.tensor(ds["label"], dtype=torch.long)
        self._spurious = torch.tensor(ds["place"], dtype=torch.long)
        self._images = ds["image"]  # list of PIL images, loaded lazily

        self.transform = transform or _waterbirds_transform(split)

    @property
    def labels(self) -> torch.Tensor:
        return self._labels

    @property
    def spurious(self) -> torch.Tensor:
        """Spurious attribute: background type (0=land, 1=water)."""
        return self._spurious

    @property
    def input_dim(self) -> int:
        # ResNet handles its own input dimensions; this is not used.
        raise NotImplementedError(
            "WaterbirdsDataset uses a ResNet backbone — input_dim is not applicable."
        )

    def __getitem__(self, idx: int) -> dict:
        img = self._images[idx]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return {
            "image": self.transform(img),         # (3, 224, 224)
            "label": self._labels[idx].item(),    # int ∈ {0, 1}
            "spurious": self._spurious[idx].item(),  # int ∈ {0, 1}
            "index": idx,
        }

    def __len__(self) -> int:
        return len(self._labels)


class TADFDataset(Dataset):
    """TADF emission wavelength classification with controlled spurious correlations.

    Binary classification: short-wavelength emitters (< median ~492 nm) vs
    long-wavelength emitters (>= median).

    Features: precomputed molecular descriptors (RDKit + Morgan FP)
    from the clever-materials-hans pipeline.  Tabular input to an MLP.

    Spurious correlations can be injected via property-based subsampling:
    a molecular property (e.g., NumHeteroatoms) is made to correlate with
    the label by keeping aligned examples and dropping most counterexamples
    from the training set.  The test set is unbiased.

    This models real-world confounding: a property (e.g., molecular weight)
    correlates with the target in the training data due to sampling bias,
    but the correlation breaks on new data.

    Args:
        parquet_path:        Path to tadf_preprocess.parquet.
        split:               "train" or "test".
        seed:                Random seed for splits.
        spurious_property:   Feature column to use as spurious signal
                             (e.g., "feat_mol_NumHeteroatoms").  None = no injection.
        spurious_correlation: Fraction of aligned (property→label) examples
                             to keep in training.  0.9 = strong shortcut.
    """

    def __init__(
        self,
        parquet_path: str,
        split: str = "train",
        seed: int = 42,
        spurious_property: str | None = None,
        spurious_correlation: float = 0.9,
    ) -> None:
        import pandas as pd
        import numpy as np

        df = pd.read_parquet(parquet_path)

        # Parse target: take first value from standard_value string.
        def _parse(x):
            if not isinstance(x, str):
                return np.nan
            return float(x.strip("[]").split(",")[0])

        df["_target_nm"] = df["standard_value"].apply(_parse)
        df = df.dropna(subset=["_target_nm"]).reset_index(drop=True)

        # Binary label at median.
        median_nm = df["_target_nm"].median()
        df["_label"] = (df["_target_nm"] >= median_nm).astype(int)

        # Feature matrix: all feat_ columns, drop constant and NaN columns.
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        feats = df[feat_cols].copy()
        feats = feats.fillna(0.0)
        non_const = feats.columns[feats.std() > 1e-10]
        feats = feats[non_const]
        feat_cols = list(non_const)

        # Standardise features.
        feat_mean = feats.mean()
        feat_std = feats.std().clip(lower=1e-8)
        feats = (feats - feat_mean) / feat_std

        # Spurious attribute: binarise the chosen property at its median.
        if spurious_property and spurious_property in df.columns:
            prop_vals = df[spurious_property].values.astype(float)
            prop_median = np.median(prop_vals)
            spurious = torch.tensor((prop_vals >= prop_median).astype(int), dtype=torch.long)
        else:
            spurious = torch.zeros(len(df), dtype=torch.long)

        # Train/test split (80/20).
        rng = np.random.RandomState(seed)
        n = len(df)
        perm = rng.permutation(n)
        n_train = int(0.8 * n)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]

        if split == "train":
            idx = train_idx
            # Inject spurious correlation by subsampling.
            # "aligned" = spurious attribute matches label (high prop → class 1).
            # Keep `spurious_correlation` fraction of aligned, and
            # `1 - spurious_correlation` of misaligned.
            if spurious_property and spurious_property in df.columns:
                labels_train = df["_label"].iloc[idx].values
                spur_train = spurious[idx].numpy()
                aligned = spur_train == labels_train
                keep = np.zeros(len(idx), dtype=bool)
                # Keep almost all aligned examples.
                aligned_idx = np.where(aligned)[0]
                n_keep_aligned = int(len(aligned_idx) * spurious_correlation)
                keep[rng.choice(aligned_idx, n_keep_aligned, replace=False)] = True
                # Keep a few misaligned examples.
                misaligned_idx = np.where(~aligned)[0]
                n_keep_mis = int(len(misaligned_idx) * (1 - spurious_correlation))
                if n_keep_mis > 0 and len(misaligned_idx) > 0:
                    keep[rng.choice(misaligned_idx, n_keep_mis, replace=False)] = True
                idx = idx[keep]
        elif split == "test":
            # Test uses all held-out data — no subsampling, unbiased.
            idx = test_idx
        else:
            raise ValueError(f"Unknown split: {split}")

        self.images = torch.tensor(feats.iloc[idx].values, dtype=torch.float32)
        self.labels = torch.tensor(df["_label"].iloc[idx].values, dtype=torch.long)
        self._spurious = spurious[idx]
        self._feat_cols = feat_cols
        self._median_nm = median_nm

    @property
    def spurious(self) -> torch.Tensor:
        """Binary: 1 = high-class1-fraction author group, 0 = rest."""
        return self._spurious

    @property
    def input_dim(self) -> int:
        return self.images.shape[1]

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": self.images[idx],  # (D,) feature vector, named "image" for compatibility
            "label": self.labels[idx].item(),
            "spurious": self._spurious[idx].item(),
            "index": idx,
        }

    def __len__(self) -> int:
        return len(self.labels)
