"""Dataset/model construction shared with `scripts/cross_sample_train.py`.

Two entry points:
- `_build_model(cfg, loaders, device)` instantiates the architecture
  (MLP / pretrained ResNet-50 / ChemBERTa / GIN) for a given run.
- `make_dataloaders(cfg)` builds canonical train/test loaders from
  the raw dataset, using the canonical-data seed for the split.
The cross-sample training loop (bagging, twin-bootstrap, MC dropout,
deep ensembles, SWA) lives in `scripts/cross_sample_train.py`.
"""
from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data import (
    ChemistryDataset,
    TADFDataset,
    WaterbirdsDataset,
)
from models import MLP, make_resnet_backbone


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _build_model(cfg, loaders, device):
    """Build a fresh model for the given architecture.

    Regression heads (``num_classes=1``) are used when ``cfg.dataset.task``
    is ``"regression"``; otherwise the default 2-class classifier head.
    """
    n_out = 1 if getattr(cfg.dataset, "task", "classification") == "regression" else 2
    arch = cfg.dataset.arch
    if arch == "resnet":
        backbone, out_dim = make_resnet_backbone()
        return MLP(backbone=backbone, backbone_out_dim=out_dim, num_classes=n_out).to(device)
    if arch == "chemberta":
        from models import make_chemberta_backbone
        backbone, out_dim = make_chemberta_backbone()
        return MLP(backbone=backbone, backbone_out_dim=out_dim, num_classes=n_out).to(device)
    if arch == "gin":
        from models import make_gin_backbone
        in_dim = loaders["train"].dataset.atom_feature_dim
        backbone, out_dim = make_gin_backbone(in_dim=in_dim,
                                              hidden_dim=cfg.model.hidden_dim)
        return MLP(backbone=backbone, backbone_out_dim=out_dim, num_classes=n_out).to(device)
    input_dim = loaders["train"].dataset.input_dim
    return MLP(input_dim=input_dim, hidden_dim=cfg.model.hidden_dim, num_classes=n_out).to(device)


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------

def make_dataloaders(cfg: DictConfig) -> dict[str, DataLoader]:
    """Build train, ID-test, and OOD-test dataloaders.

    Dispatches on `cfg.dataset.name`.  Returns ``{"train", "id_test", "ood_test"}``.
    """
    name = cfg.dataset.name
    seed = cfg.training.seed

    if name == "tadf":
        ppath = cfg.dataset.parquet_path
        spur_prop = getattr(cfg.dataset, "spurious_property", None)
        spur_corr = getattr(cfg.dataset, "spurious_correlation", 0.9)
        train_ds = TADFDataset(parquet_path=ppath, split="train", seed=seed,
                               spurious_property=spur_prop, spurious_correlation=spur_corr)
        id_test_ds = TADFDataset(parquet_path=ppath, split="test", seed=seed)
        ood_test_ds = TADFDataset(parquet_path=ppath, split="test_misaligned", seed=seed,
                                  spurious_property=spur_prop)

    elif name in ("mof_thermal", "mof_solvent"):
        ppath = cfg.dataset.parquet_path
        target_col = cfg.dataset.target_column
        spur_prop = getattr(cfg.dataset, "spurious_property", None)
        spur_corr = getattr(cfg.dataset, "spurious_correlation", 0.9)
        train_ds = ChemistryDataset(parquet_path=ppath, target_column=target_col,
                                    split="train", seed=seed,
                                    spurious_property=spur_prop, spurious_correlation=spur_corr)
        id_test_ds = ChemistryDataset(parquet_path=ppath, target_column=target_col,
                                      split="test", seed=seed)
        ood_test_ds = ChemistryDataset(parquet_path=ppath, target_column=target_col,
                                       split="test_misaligned", seed=seed,
                                       spurious_property=spur_prop)

    elif name in ("bace", "bbbp", "clintox"):
        from data_molnet import MolNetDataset
        data_dir = getattr(cfg.dataset, "data_dir", "./data/molnet")
        train_ds = MolNetDataset(name=name, split="train", seed=seed, data_dir=data_dir)
        id_test_ds = MolNetDataset(name=name, split="test", seed=seed, data_dir=data_dir)
        ood_test_ds = MolNetDataset(name=name, split="test_scaffold", seed=seed, data_dir=data_dir)

    elif name in ("esol_reg", "freesolv_reg", "lipo_reg"):
        # Regression variant: keeps continuous target instead of median-binarising.
        from data_molnet import MolNetDataset
        data_dir = getattr(cfg.dataset, "data_dir", "./data/molnet")
        molnet_name = name.replace("_reg", "")
        train_ds = MolNetDataset(name=molnet_name, split="train", seed=seed,
                                 data_dir=data_dir, regression=True)
        id_test_ds = MolNetDataset(name=molnet_name, split="test", seed=seed,
                                   data_dir=data_dir, regression=True)
        ood_test_ds = MolNetDataset(name=molnet_name, split="test_scaffold",
                                    seed=seed, data_dir=data_dir, regression=True)

    elif name in ("bace_chemberta", "bbbp_chemberta"):
        from data_molnet import MolNetTokenDataset
        data_dir = getattr(cfg.dataset, "data_dir", "./data/molnet")
        molnet_name = name.replace("_chemberta", "")
        train_ds = MolNetTokenDataset(name=molnet_name, split="train",
                                      seed=seed, data_dir=data_dir)
        id_test_ds = MolNetTokenDataset(name=molnet_name, split="test",
                                        seed=seed, data_dir=data_dir)
        ood_test_ds = MolNetTokenDataset(name=molnet_name, split="test_scaffold",
                                         seed=seed, data_dir=data_dir)

    elif name.endswith("_gin") and name.replace("_gin", "") in ("bace", "bbbp"):
        from data_molnet import MolNetGraphDataset
        data_dir = getattr(cfg.dataset, "data_dir", "./data/molnet")
        molnet_name = name.replace("_gin", "")
        train_ds = MolNetGraphDataset(name=molnet_name, split="train",
                                      seed=seed, data_dir=data_dir)
        id_test_ds = MolNetGraphDataset(name=molnet_name, split="test",
                                        seed=seed, data_dir=data_dir)
        ood_test_ds = MolNetGraphDataset(name=molnet_name, split="test_scaffold",
                                         seed=seed, data_dir=data_dir)

    elif name in ("hia_hou", "bioavailability_ma", "pgp_broccatelli",
                  "bbb_martins", "herg", "dili", "ames", "skin_reaction",
                  "cyp2c9_substrate", "cyp2d6_substrate", "cyp3a4_substrate"):
        from data_tdc import TDCDataset
        data_dir = getattr(cfg.dataset, "data_dir", "./data/tdc")
        train_ds   = TDCDataset(name=name, split="train",         seed=seed, data_dir=data_dir)
        id_test_ds = TDCDataset(name=name, split="test",          seed=seed, data_dir=data_dir)
        ood_test_ds = TDCDataset(name=name, split="test_scaffold", seed=seed, data_dir=data_dir)

    elif name.endswith("_chemberta") and name.replace("_chemberta", "") in (
            "hia_hou", "bioavailability_ma", "pgp_broccatelli",
            "bbb_martins", "herg", "dili", "ames", "skin_reaction"):
        from data_tdc import TDCTokenDataset
        data_dir = getattr(cfg.dataset, "data_dir", "./data/tdc")
        tdc_name = name.replace("_chemberta", "")
        train_ds = TDCTokenDataset(name=tdc_name, split="train",
                                   seed=seed, data_dir=data_dir)
        id_test_ds = TDCTokenDataset(name=tdc_name, split="test",
                                     seed=seed, data_dir=data_dir)
        ood_test_ds = TDCTokenDataset(name=tdc_name, split="test_scaffold",
                                      seed=seed, data_dir=data_dir)

    elif name == "waterbirds":
        train_ds = WaterbirdsDataset(split="train", data_dir=cfg.dataset.data_dir)
        id_test_ds = WaterbirdsDataset(split="val", data_dir=cfg.dataset.data_dir)
        ood_test_ds = WaterbirdsDataset(split="test", data_dir=cfg.dataset.data_dir)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    use_cuda = torch.cuda.is_available()
    kwargs = dict(
        batch_size=cfg.training.batch_size,
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
        persistent_workers=True if use_cuda else False,
    )
    # GIN datasets yield torch_geometric Data objects in the "image" field;
    # the default collate fails on them.
    if cfg.dataset.arch == "gin":
        def graph_collate(items):
            from torch_geometric.data import Batch
            return {
                "image":    Batch.from_data_list([it["image"] for it in items]),
                "label":    torch.tensor([it["label"]    for it in items], dtype=torch.long),
                "spurious": torch.tensor([it["spurious"] for it in items], dtype=torch.long),
                "index":    torch.tensor([it["index"]    for it in items], dtype=torch.long),
            }
        kwargs["collate_fn"] = graph_collate

    return {
        "train":    DataLoader(train_ds, shuffle=True, **kwargs),
        "id_test":  DataLoader(id_test_ds, shuffle=False, **kwargs),
        "ood_test": DataLoader(ood_test_ds, shuffle=False, **kwargs),
    }
