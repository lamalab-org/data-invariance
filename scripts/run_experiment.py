"""Per-dataset hyperparameters + config builder used by `cross_sample_train.py`.

The original ``run_experiment.py`` was a Hydra entry point for a prior
research direction (JTT / LfF / V-REx on group-robustness benchmarks).
Only the per-dataset hyperparameter table and ``build_cfg`` are needed
by the current paper pipeline; everything else has been removed.

Active consumers
----------------
``scripts/cross_sample_train.py`` imports ``HPARAMS``, ``build_cfg``,
and ``_build_model`` (re-exported from ``train``).  Nothing else in
the active codebase imports from this module.
"""
from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import _build_model  # noqa: F401, E402  (re-exported for cross_sample_train)


# ---------------------------------------------------------------------------
# Per-dataset training hyperparameters.  Standard training hparams only;
# no per-dataset method-specific tuning (twin-bootstrap λ is fixed by the
# selection rule on BACE; bagging K is fixed at the call-site).
# ---------------------------------------------------------------------------
HPARAMS = {
    # Default-MLP datasets (Morgan fingerprints + descriptors)
    "tadf":              {"lr": 1e-3, "batch_size": 128, "epochs": 20},
    "mof_thermal":       {"lr": 1e-3, "batch_size": 128, "epochs": 20},
    "mof_solvent":       {"lr": 1e-3, "batch_size": 128, "epochs": 20},
    "bace":              {"lr": 1e-3, "batch_size": 64,  "epochs": 30},
    "bbbp":              {"lr": 1e-3, "batch_size": 64,  "epochs": 30},
    "clintox":           {"lr": 1e-3, "batch_size": 64,  "epochs": 30},

    # ChemBERTa fine-tune (transformer fine-tune defaults)
    "bace_chemberta":             {"lr": 2e-5, "batch_size": 32, "epochs": 10},
    "bbbp_chemberta":             {"lr": 2e-5, "batch_size": 32, "epochs": 10},
    "pgp_broccatelli_chemberta":  {"lr": 2e-5, "batch_size": 32, "epochs": 10},
    "bbb_martins_chemberta":      {"lr": 2e-5, "batch_size": 32, "epochs": 10},
    "ames_chemberta":             {"lr": 2e-5, "batch_size": 32, "epochs": 10},
    "dili_chemberta":             {"lr": 2e-5, "batch_size": 32, "epochs": 10},

    # GIN (graph network)
    "bace_gin":          {"lr": 1e-3, "batch_size": 32, "epochs": 50},

    # TDC ADME / Tox single-task classification (MLP + Morgan FP)
    "hia_hou":            {"lr": 1e-3, "batch_size": 64, "epochs": 30},
    "bioavailability_ma": {"lr": 1e-3, "batch_size": 64, "epochs": 30},
    "pgp_broccatelli":    {"lr": 1e-3, "batch_size": 64, "epochs": 30},
    "bbb_martins":        {"lr": 1e-3, "batch_size": 64, "epochs": 30},
    "herg":               {"lr": 1e-3, "batch_size": 64, "epochs": 30},
    "dili":               {"lr": 1e-3, "batch_size": 64, "epochs": 30},
    "ames":               {"lr": 1e-3, "batch_size": 64, "epochs": 30},
    "skin_reaction":      {"lr": 1e-3, "batch_size": 64, "epochs": 30},
    "cyp2c9_substrate":   {"lr": 1e-3, "batch_size": 64, "epochs": 30},
    "cyp2d6_substrate":   {"lr": 1e-3, "batch_size": 64, "epochs": 30},
    "cyp3a4_substrate":   {"lr": 1e-3, "batch_size": 64, "epochs": 30},

    # Regression (MoleculeNet)
    "esol_reg":          {"lr": 1e-3, "batch_size": 64, "epochs": 50},
    "freesolv_reg":      {"lr": 1e-3, "batch_size": 32, "epochs": 50},
    "lipo_reg":          {"lr": 1e-3, "batch_size": 64, "epochs": 50},

    # Vision (ImageNet-pretrained ResNet-50)
    "waterbirds":        {"lr": 1e-4, "batch_size": 64, "epochs": 15},
}


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_cfg(dataset_name: str):
    """Build the OmegaConf cfg consumed by `make_dataloaders` and `_build_model`.

    Loads the dataset YAML from ``configs/dataset/<name>.yaml`` and stamps in
    the per-dataset hyperparameters from `HPARAMS`.  Other ``training.*``
    fields default to the values used across the paper sweep.
    """
    repo_root = Path(__file__).resolve().parent.parent
    dataset_cfg = OmegaConf.load(repo_root / f"configs/dataset/{dataset_name}.yaml")
    hp = HPARAMS[dataset_name]
    return OmegaConf.create({
        "dataset": OmegaConf.to_container(dataset_cfg),
        "model":    {"hidden_dim": 256},
        "training": {
            "lr": hp["lr"],
            "weight_decay": 1e-4,
            "batch_size": hp["batch_size"],
            "epochs": hp["epochs"],
            "seed": 42,
        },
        "wandb":    {"enabled": False},
    })
