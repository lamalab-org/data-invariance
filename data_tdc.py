"""TDC (Therapeutics Data Commons) datasets, plugged into our scaffold-split protocol.

We use TDC for the SMILES + target data (it's the standard source for
drug-discovery benchmarks) but apply our own scaffold split + random
in-distribution holdout, identical to ``data_molnet.MolNetDataset``. This
keeps every chemistry dataset in the paper directly comparable.

Splits returned by `__init__` (matches MolNetDataset):
    "train"          80 % of scaffold-train (used for bootstrapping)
    "test"           20 % of scaffold-train, random split (id_test)
    "test_scaffold"  scaffold-test (ood_test, novel scaffolds)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data_molnet import (
    _scaffold_split,
    _scaffold_split_with_id_holdout,
    _smiles_to_morgan,
    _tokenize_smiles_chemberta,
    _validate_smiles,
)


# Map our short dataset name → (TDC group, TDC dataset name).
# Every entry is single-task binary classification with columns
# ('Drug_ID', 'Drug', 'Y'); 'Drug' is SMILES, 'Y' ∈ {0, 1}.
_TDC_DATASETS = {
    # ADME — absorption / distribution / metabolism / excretion
    "hia_hou":            ("ADME", "HIA_Hou"),
    "bioavailability_ma": ("ADME", "Bioavailability_Ma"),
    "pgp_broccatelli":    ("ADME", "Pgp_Broccatelli"),
    "bbb_martins":        ("ADME", "BBB_Martins"),
    # ADME — CYP P450 substrate prediction (binary; ~660-940 each)
    "cyp2c9_substrate":   ("ADME", "CYP2C9_Substrate_CarbonMangels"),
    "cyp2d6_substrate":   ("ADME", "CYP2D6_Substrate_CarbonMangels"),
    "cyp3a4_substrate":   ("ADME", "CYP3A4_Substrate_CarbonMangels"),
    # Tox
    "herg":               ("Tox",  "hERG"),
    "dili":               ("Tox",  "DILI"),
    "ames":               ("Tox",  "AMES"),
    "skin_reaction":      ("Tox",  "Skin_Reaction"),
}


def _load_tdc(name: str, data_dir: Path):
    """Return (smiles_list, labels) for the requested TDC dataset."""
    group_name, tdc_name = _TDC_DATASETS[name]
    if group_name == "ADME":
        from tdc.single_pred import ADME
        data = ADME(name=tdc_name, path=str(data_dir))
    elif group_name == "Tox":
        from tdc.single_pred import Tox
        data = Tox(name=tdc_name, path=str(data_dir))
    else:
        raise ValueError(f"unknown TDC group {group_name}")

    df = data.get_data().dropna(subset=["Drug", "Y"]).reset_index(drop=True)
    return df["Drug"].tolist(), df["Y"].astype(int).values


class TDCDataset(Dataset):
    """TDC dataset → Morgan FP + scaffold-split protocol matching MolNetDataset."""

    def __init__(
        self,
        name: str,
        split: str = "train",
        seed: int = 42,
        data_dir: str = "./data/tdc",
        n_bits: int = 2048,
    ) -> None:
        if name not in _TDC_DATASETS:
            raise ValueError(f"unknown TDC dataset {name}; "
                             f"available: {sorted(_TDC_DATASETS)}")

        cache = Path(data_dir)
        cache.mkdir(parents=True, exist_ok=True)
        smiles_list, labels = _load_tdc(name, cache)

        # Featurize.
        fps, valid_idx = [], []
        for i, smi in enumerate(smiles_list):
            fp = _smiles_to_morgan(smi, n_bits=n_bits)
            if fp is not None:
                fps.append(fp)
                valid_idx.append(i)
        features = np.stack(fps)
        labels = labels[valid_idx]
        smiles_list = [smiles_list[i] for i in valid_idx]

        # Same split logic as MolNetDataset.
        scaffold_train_idx, scaffold_test_idx = _scaffold_split(
            smiles_list, seed=seed)
        rng = np.random.RandomState(seed)
        rng.shuffle(scaffold_train_idx)
        n_id_test = max(1, int(len(scaffold_train_idx) * 0.2))
        id_test_idx = scaffold_train_idx[:n_id_test]
        train_idx   = scaffold_train_idx[n_id_test:]

        if split == "train":
            idx = train_idx
        elif split == "test":
            idx = id_test_idx
        elif split == "test_scaffold":
            idx = scaffold_test_idx
        else:
            raise ValueError(f"unknown split {split}")

        self.images = torch.tensor(features[idx], dtype=torch.float32)
        self.labels = torch.tensor(labels[idx], dtype=torch.long)
        self._spurious = torch.full((len(idx),),
                                    fill_value=int(split == "test_scaffold"),
                                    dtype=torch.long)

    @property
    def spurious(self) -> torch.Tensor:
        return self._spurious

    @property
    def input_dim(self) -> int:
        return self.images.shape[1]

    def __getitem__(self, idx: int) -> dict:
        return {
            "image":    self.images[idx],
            "label":    self.labels[idx].item(),
            "spurious": self._spurious[idx].item(),
            "index":    idx,
        }

    def __len__(self) -> int:
        return len(self.labels)


class TDCTokenDataset(Dataset):
    """TDC dataset with ChemBERTa-tokenised SMILES, mirroring TDCDataset.

    Uses the same scaffold split + 20% random holdout as Morgan TDCDataset, so
    cross-sample protocols match across featurisers.
    """

    def __init__(
        self,
        name: str,
        split: str = "train",
        seed: int = 42,
        data_dir: str = "./data/tdc",
        max_length: int = 128,
    ) -> None:
        if name not in _TDC_DATASETS:
            raise ValueError(f"unknown TDC dataset {name}; "
                             f"available: {sorted(_TDC_DATASETS)}")

        cache = Path(data_dir)
        cache.mkdir(parents=True, exist_ok=True)
        smiles_list, labels = _load_tdc(name, cache)

        valid_idx = _validate_smiles(smiles_list)
        smiles_list = [smiles_list[i] for i in valid_idx]
        labels = labels[valid_idx]

        train_idx, id_test_idx, scaffold_test_idx = _scaffold_split_with_id_holdout(
            smiles_list, seed
        )
        if split == "train":
            idx = train_idx
        elif split == "test":
            idx = id_test_idx
        elif split == "test_scaffold":
            idx = scaffold_test_idx
        else:
            raise ValueError(f"unknown split {split}")

        smiles_subset = [smiles_list[i] for i in idx]
        self._input_ids, self._attention_mask = _tokenize_smiles_chemberta(
            smiles_subset, max_length=max_length
        )
        self.labels = torch.tensor(labels[idx], dtype=torch.long)
        self._spurious = torch.full(
            (len(idx),), int(split == "test_scaffold"), dtype=torch.long
        )

    @property
    def spurious(self) -> torch.Tensor:
        return self._spurious

    @property
    def input_dim(self) -> int:
        raise NotImplementedError("ChemBERTa uses tokenized input, not input_dim.")

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": {
                "input_ids": self._input_ids[idx],
                "attention_mask": self._attention_mask[idx],
            },
            "label": self.labels[idx].item(),
            "spurious": self._spurious[idx].item(),
            "index": idx,
        }

    def __len__(self) -> int:
        return len(self.labels)
