"""MoleculeNet datasets for scaffold-split distribution shift experiments.

Scaffold split is the chemistry analog of spurious correlations: models trained
with random splits memorize scaffold patterns (core chemical substructures)
that don't generalize to structurally novel molecules. Scaffold splitting
ensures no scaffold appears in both train and test.

Available datasets: BACE, BBBP, HIV — all binary classification with
well-documented scaffold-split performance gaps (5-15% AUROC drop).

Features: Morgan fingerprints (2048 bits) via RDKit. Tabular input to MLP.
"""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from torch.utils.data import Dataset


# DeepChem data server URLs for MoleculeNet datasets
_MOLNET_URLS = {
    "bace":    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
    "bbbp":    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
    "hiv":     "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
    "clintox": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
    "tox21":   "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
    "sider":   "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
    "muv":     "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz",
    "pcba":    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pcba.csv.gz",
    "esol":    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
}

# Column names for SMILES and target in each dataset.
# For multi-task datasets, we pick a single representative task.
_MOLNET_COLS = {
    "bace":    {"smiles": "mol",    "target": "Class",     "task": "classification"},
    "bbbp":    {"smiles": "smiles", "target": "p_np",      "task": "classification"},
    "hiv":     {"smiles": "smiles", "target": "HIV_active", "task": "classification"},
    "clintox": {"smiles": "smiles", "target": "CT_TOX",    "task": "classification"},
    # Tox21: NR-AR (Nuclear Receptor - Androgen Receptor) — most-studied single task
    "tox21":   {"smiles": "smiles", "target": "NR-AR",     "task": "classification"},
    # SIDER: Hepatobiliary disorders — common side-effect category
    "sider":   {"smiles": "smiles", "target": "Hepatobiliary disorders", "task": "classification"},
    # MUV: 466 (S1P1 receptor agonists) — common single-task choice from MUV-17
    "muv":     {"smiles": "smiles", "target": "MUV-466",   "task": "classification"},
    # PCBA: PCBA-686978 — well-studied bioassay (sigma-1 receptor binding)
    "pcba":    {"smiles": "smiles", "target": "PCBA-686978", "task": "classification"},
    "esol":    {"smiles": "smiles", "target": "measured log solubility in mols per litre", "task": "regression"},
}


def _smiles_to_morgan(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    """Convert a SMILES string to a Morgan fingerprint bit vector."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


CHEMBERTA_MODEL = "DeepChem/ChemBERTa-77M-MTR"


def _validate_smiles(smiles_list: list[str]) -> list[int]:
    """Return indices of SMILES that RDKit can parse."""
    from rdkit import Chem

    valid_idx = []
    for i, smi in enumerate(smiles_list):
        if Chem.MolFromSmiles(smi) is not None:
            valid_idx.append(i)
    return valid_idx


def _tokenize_smiles_chemberta(
    smiles_list: list[str], max_length: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize SMILES with the ChemBERTa-77M-MTR tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL)
    encoded = tokenizer(
        smiles_list,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]


def _scaffold_split(smiles_list: list[str], seed: int = 42, frac_train: float = 0.8) -> tuple[list[int], list[int]]:
    """Bemis-Murcko scaffold split: no scaffold shared between train and test.

    Returns (train_indices, test_indices).
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    # Group molecules by scaffold
    scaffold_to_indices: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaffold = ""
        scaffold_to_indices.setdefault(scaffold, []).append(i)

    # Sort scaffolds by size (largest first) for deterministic splitting
    scaffolds = sorted(scaffold_to_indices.items(), key=lambda x: -len(x[1]))

    rng = np.random.RandomState(seed)
    # Shuffle scaffolds (deterministically) then assign to train/test
    scaffold_indices = list(range(len(scaffolds)))
    rng.shuffle(scaffold_indices)

    n_total = sum(len(indices) for _, indices in scaffolds)
    n_train = int(n_total * frac_train)

    train_idx, test_idx = [], []
    current_train = 0
    for si in scaffold_indices:
        _, indices = scaffolds[si]
        if current_train + len(indices) <= n_train:
            train_idx.extend(indices)
            current_train += len(indices)
        else:
            test_idx.extend(indices)

    return train_idx, test_idx


class MolNetDataset(Dataset):
    """MoleculeNet dataset with Morgan fingerprint features and scaffold split.

    Args:
        name:      Dataset name ("bace", "bbbp", "hiv", "clintox", "esol").
        split:     "train" (random subset of train scaffolds),
                   "test" (random split of train for ID eval),
                   "test_scaffold" (held-out scaffolds for OOD eval).
        seed:      Random seed for splits.
        data_dir:  Cache directory for downloads.
        n_bits:    Morgan fingerprint length.
    """

    def __init__(
        self,
        name: str = "bace",
        split: str = "train",
        seed: int = 42,
        data_dir: str = "./data/molnet",
        n_bits: int = 2048,
    ) -> None:
        import pandas as pd

        assert name in _MOLNET_URLS, f"Unknown dataset: {name}. Choose from {list(_MOLNET_URLS.keys())}"
        info = _MOLNET_COLS[name]

        # Download if needed (use the file extension from the URL)
        url = _MOLNET_URLS[name]
        ext = ".csv.gz" if url.endswith(".gz") else ".csv"
        data_path = Path(data_dir) / f"{name}{ext}"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        if not data_path.exists():
            print(f"Downloading {name} from {url}...")
            urlretrieve(url, data_path)

        df = pd.read_csv(data_path)  # pandas handles .gz transparently

        # Parse target
        smiles_col = info["smiles"]
        target_col = info["target"]

        # Drop rows with missing target or SMILES
        df = df.dropna(subset=[smiles_col, target_col]).reset_index(drop=True)
        smiles_list = df[smiles_col].tolist()

        # Featurize: Morgan fingerprints
        fps = []
        valid_idx = []
        for i, smi in enumerate(smiles_list):
            fp = _smiles_to_morgan(smi, n_bits=n_bits)
            if fp is not None:
                fps.append(fp)
                valid_idx.append(i)

        features = np.stack(fps)
        df = df.iloc[valid_idx].reset_index(drop=True)
        smiles_list = [smiles_list[i] for i in valid_idx]

        # Target: binary for classification, median-binarized for regression
        if info["task"] == "classification":
            labels = df[target_col].astype(int).values
        else:
            vals = df[target_col].astype(float).values
            labels = (vals >= np.median(vals)).astype(int)

        # Scaffold split
        scaffold_train_idx, scaffold_test_idx = _scaffold_split(smiles_list, seed=seed)

        # Within scaffold-train, do a random 80/20 for train/ID-test
        rng = np.random.RandomState(seed)
        rng.shuffle(scaffold_train_idx)
        n_id_test = max(1, int(len(scaffold_train_idx) * 0.2))
        id_test_idx = scaffold_train_idx[:n_id_test]
        train_idx = scaffold_train_idx[n_id_test:]

        if split == "train":
            idx = train_idx
        elif split == "test":
            idx = id_test_idx  # ID test (same scaffolds as train)
        elif split == "test_scaffold":
            idx = scaffold_test_idx  # OOD test (novel scaffolds)
        else:
            raise ValueError(f"Unknown split: {split}")

        self.images = torch.tensor(features[idx], dtype=torch.float32)
        self.labels = torch.tensor(labels[idx], dtype=torch.long)

        # Spurious attribute: scaffold membership (in-distribution vs novel)
        # For train: all examples are "in-distribution" (spurious=0)
        # For scaffold test: all are "novel" (spurious=1)
        # This gives WGA = min(acc_in_distribution, acc_novel_scaffold)
        if split == "test_scaffold":
            self._spurious = torch.ones(len(idx), dtype=torch.long)
        else:
            self._spurious = torch.zeros(len(idx), dtype=torch.long)

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


def _load_molnet_smiles_labels(name: str, data_dir: str) -> tuple[list[str], np.ndarray]:
    """Download a MolNet CSV and return (smiles_list, labels) for valid molecules.

    Validates SMILES with RDKit; binarizes regression targets at the median.
    """
    import pandas as pd

    assert name in _MOLNET_URLS, f"Unknown dataset: {name}"
    info = _MOLNET_COLS[name]
    url = _MOLNET_URLS[name]
    ext = ".csv.gz" if url.endswith(".gz") else ".csv"
    data_path = Path(data_dir) / f"{name}{ext}"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        print(f"Downloading {name} from {url}...")
        urlretrieve(url, data_path)

    df = pd.read_csv(data_path).dropna(
        subset=[info["smiles"], info["target"]]
    ).reset_index(drop=True)
    smiles_list = df[info["smiles"]].tolist()

    valid_idx = _validate_smiles(smiles_list)
    smiles_list = [smiles_list[i] for i in valid_idx]
    df = df.iloc[valid_idx].reset_index(drop=True)

    if info["task"] == "classification":
        labels = df[info["target"]].astype(int).values
    else:
        vals = df[info["target"]].astype(float).values
        labels = (vals >= np.median(vals)).astype(int)
    return smiles_list, labels


def _scaffold_split_with_id_holdout(
    smiles_list: list[str], seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Scaffold split + 20% random holdout from scaffold-train for ID test.

    Returns (train_idx, id_test_idx, scaffold_test_idx).
    """
    scaffold_train_idx, scaffold_test_idx = _scaffold_split(smiles_list, seed=seed)
    rng = np.random.RandomState(seed)
    rng.shuffle(scaffold_train_idx)
    n_id_test = max(1, int(len(scaffold_train_idx) * 0.2))
    id_test_idx = scaffold_train_idx[:n_id_test]
    train_idx = scaffold_train_idx[n_id_test:]
    return train_idx, id_test_idx, scaffold_test_idx


def _smiles_to_graph(smi: str):
    """Return (atom_features [N, A_dim], edge_index [2, E], edge_attr [E, B_dim]) or None.

    Atom features: one-hot atomic number bucket + degree + formal charge + aromatic + Hs.
    Bond features: bond-type one-hot (single / double / triple / aromatic).
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    # Atom features (lightweight, GIN-style).
    atom_buckets = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]   # H,C,N,O,F,P,S,Cl,Br,I
    A_dim = len(atom_buckets) + 1 + 4 + 4 + 4 + 1
    feats = []
    for atom in mol.GetAtoms():
        an = atom.GetAtomicNum()
        atom_one = [1.0 if an == b else 0.0 for b in atom_buckets] + [
            1.0 if an not in atom_buckets else 0.0
        ]
        deg = atom.GetDegree()
        deg_one = [1.0 if deg == k else 0.0 for k in range(4)]
        charge = atom.GetFormalCharge()
        ch_one = [1.0 if charge == c else 0.0 for c in (-1, 0, 1, 2)]
        nH = atom.GetTotalNumHs()
        nH_one = [1.0 if nH == k else 0.0 for k in range(4)]
        arom = [1.0 if atom.GetIsAromatic() else 0.0]
        feats.append(atom_one + deg_one + ch_one + nH_one + arom)
    x = torch.tensor(feats, dtype=torch.float32)

    # Bonds (bidirectional).
    src, dst, btypes = [], [], []
    bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                  Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        bt_one = [1.0 if bt == t else 0.0 for t in bond_types]
        src.extend([i, j]); dst.extend([j, i])
        btypes.extend([bt_one, bt_one])
    if not src:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, len(bond_types)), dtype=torch.float32)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(btypes, dtype=torch.float32)
    return x, edge_index, edge_attr


class MolNetGraphDataset(Dataset):
    """MoleculeNet dataset with on-the-fly RDKit graphs and scaffold split.

    Mirrors ``MolNetDataset`` but yields ``torch_geometric.data.Data``
    objects so a GIN backbone can consume the molecular graph directly.
    """

    def __init__(
        self,
        name: str = "bace",
        split: str = "train",
        seed: int = 42,
        data_dir: str = "./data/molnet",
    ) -> None:
        from torch_geometric.data import Data
        smiles_list, labels = _load_molnet_smiles_labels(name, data_dir)
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
            raise ValueError(f"Unknown split: {split}")

        self._graphs = []
        self.labels = []
        for i in idx:
            g = _smiles_to_graph(smiles_list[i])
            if g is None:
                continue
            x, edge_index, edge_attr = g
            self._graphs.append(
                Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                     y=torch.tensor(int(labels[i]), dtype=torch.long))
            )
            self.labels.append(int(labels[i]))
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self._spurious = torch.full(
            (len(self.labels),), int(split == "test_scaffold"), dtype=torch.long
        )

    @property
    def spurious(self) -> torch.Tensor:
        return self._spurious

    @property
    def input_dim(self) -> int:
        raise NotImplementedError("Graph backbone uses Data objects, not input_dim.")

    @property
    def atom_feature_dim(self) -> int:
        return self._graphs[0].x.shape[1] if self._graphs else 0

    def __getitem__(self, idx: int) -> dict:
        g = self._graphs[idx]
        return {
            "image": g,
            "label": int(self.labels[idx]),
            "spurious": int(self._spurious[idx]),
            "index": idx,
        }

    def __len__(self) -> int:
        return len(self.labels)


class MolNetTokenDataset(Dataset):
    """MoleculeNet dataset with ChemBERTa-tokenized SMILES and scaffold split.

    Mirrors ``MolNetDataset`` but emits ``__getitem__["image"]`` as a dict
    ``{"input_ids", "attention_mask"}`` for transformer backbones. The split
    logic is identical so cross-sample protocols match across featurizers.
    """

    def __init__(
        self,
        name: str = "bace",
        split: str = "train",
        seed: int = 42,
        data_dir: str = "./data/molnet",
        max_length: int = 128,
    ) -> None:
        smiles_list, labels = _load_molnet_smiles_labels(name, data_dir)
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
            raise ValueError(f"Unknown split: {split}")

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
