from __future__ import annotations

from pathlib import Path

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms


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
                aligned_idx = np.where(aligned)[0]
                n_keep_aligned = int(len(aligned_idx) * spurious_correlation)
                keep[rng.choice(aligned_idx, n_keep_aligned, replace=False)] = True
                misaligned_idx = np.where(~aligned)[0]
                n_keep_mis = int(len(misaligned_idx) * (1 - spurious_correlation))
                if n_keep_mis > 0 and len(misaligned_idx) > 0:
                    keep[rng.choice(misaligned_idx, n_keep_mis, replace=False)] = True
                idx = idx[keep]
        elif split == "test":
            idx = test_idx
        elif split == "test_misaligned":
            # Only counterexamples: spurious attribute disagrees with label.
            # This is the hardest subset — the examples the shortcut gets wrong.
            idx = test_idx
            if spurious_property and spurious_property in df.columns:
                labels_test = df["_label"].iloc[idx].values
                spur_test = spurious[idx].numpy()
                misaligned = spur_test != labels_test
                idx = idx[misaligned]
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


class ChemistryDataset(Dataset):
    """General tabular chemistry dataset from clever-materials-hans.

    Works with any parquet file that has ``feat_*`` columns (molecular/materials
    descriptors) and a numeric target column. Binary classification at the
    target median. Spurious correlations are injected by subsampling to make a
    chosen property correlate with the label.

    Currently used for:
    - **MOF thermal stability** — target: ``assigned_T_decomp`` (K).
      Spurious: ``publication_year`` or any ``feat_*`` column.
    - **Perovskite solar cells** — target: ``data.jv.default_PCE`` (%).
      Spurious: ``publication_year`` or any ``feat_*`` column.

    The same class could be used for battery, MOF solvent stability, etc.
    by changing ``target_column`` and ``parquet_path``.

    Args:
        parquet_path:         Path to the preprocessed parquet file.
        target_column:        Name of the numeric target column.
        split:                "train", "test", or "test_misaligned".
        seed:                 Random seed for splits.
        spurious_property:    Column to use as spurious signal (None = no injection).
        spurious_correlation: Fraction of aligned examples to keep in training.
    """

    def __init__(
        self,
        parquet_path: str,
        target_column: str,
        split: str = "train",
        seed: int = 42,
        spurious_property: str | None = None,
        spurious_correlation: float = 0.9,
    ) -> None:
        import pandas as pd
        import numpy as np

        df = pd.read_parquet(parquet_path)

        # Parse target — handle both numeric and string-encoded values.
        target = pd.to_numeric(df[target_column], errors="coerce")
        df["_target"] = target
        df = df.dropna(subset=["_target"]).reset_index(drop=True)

        # Binary label at median.
        median_val = df["_target"].median()
        df["_label"] = (df["_target"] >= median_val).astype(int)

        # Feature matrix: all feat_ columns.
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        feats = df[feat_cols].copy().fillna(0.0)
        non_const = feats.columns[feats.std() > 1e-10]
        feats = feats[non_const]
        feat_cols = list(non_const)

        # Standardise.
        feat_mean = feats.mean()
        feat_std = feats.std().clip(lower=1e-8)
        feats = (feats - feat_mean) / feat_std

        # Spurious attribute: binarise the chosen property at its median.
        if spurious_property and spurious_property in df.columns:
            prop_vals = pd.to_numeric(df[spurious_property], errors="coerce").fillna(0).values
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
            if spurious_property and spurious_property in df.columns:
                labels_train = df["_label"].iloc[idx].values
                spur_train = spurious[idx].numpy()
                aligned = spur_train == labels_train
                keep = np.zeros(len(idx), dtype=bool)
                aligned_idx = np.where(aligned)[0]
                n_keep_aligned = int(len(aligned_idx) * spurious_correlation)
                keep[rng.choice(aligned_idx, n_keep_aligned, replace=False)] = True
                misaligned_idx = np.where(~aligned)[0]
                n_keep_mis = int(len(misaligned_idx) * (1 - spurious_correlation))
                if n_keep_mis > 0 and len(misaligned_idx) > 0:
                    keep[rng.choice(misaligned_idx, n_keep_mis, replace=False)] = True
                idx = idx[keep]
        elif split == "test":
            idx = test_idx
        elif split == "test_misaligned":
            idx = test_idx
            if spurious_property and spurious_property in df.columns:
                labels_test = df["_label"].iloc[idx].values
                spur_test = spurious[idx].numpy()
                misaligned = spur_test != labels_test
                idx = idx[misaligned]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.images = torch.tensor(feats.iloc[idx].values, dtype=torch.float32)
        self.labels = torch.tensor(df["_label"].iloc[idx].values, dtype=torch.long)
        self._spurious = spurious[idx]
        self._feat_cols = feat_cols

    @property
    def spurious(self) -> torch.Tensor:
        return self._spurious

    @property
    def input_dim(self) -> int:
        return self.images.shape[1]

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": self.images[idx],
            "label": self.labels[idx].item(),
            "spurious": self._spurious[idx].item(),
            "index": idx,
        }

    def __len__(self) -> int:
        return len(self.labels)
