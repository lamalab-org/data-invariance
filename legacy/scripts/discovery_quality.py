"""Compute discovery quality metrics across all datasets.

Reports signal_ratio, reliability, assignment-spurious correlation,
and per-environment spurious-label correlation for each dataset/seed.

Usage:
    uv run python scripts/discovery_quality.py --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_experiment import build_cfg, HPARAMS  # noqa: E402
from train import discover_environments, make_dataloaders  # noqa: E402
from utils import set_seed  # noqa: E402

# Skip vision/NLP datasets that need GPU or special downloads
CPU_DATASETS = [
    "cmnist", "continuous_cmnist", "multi_cmnist",
    "tadf", "mof_thermal", "mof_solvent", "perovskite", "battery",
    "bace", "bbbp", "hiv",
]

SEEDS = [42, 123, 789]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--datasets", default=None,
                    help="Comma-separated subset of datasets (default: all CPU)")
    args = ap.parse_args()

    device = torch.device(args.device)
    datasets = args.datasets.split(",") if args.datasets else CPU_DATASETS

    print(f"{'Dataset':20s} {'N':>7s} {'signal_ratio':>13s} {'reliability':>12s} "
          f"{'assign_corr':>12s} {'corr_A':>8s} {'corr_B':>8s}")
    print("-" * 85)

    for ds_name in datasets:
        if ds_name not in HPARAMS:
            print(f"{ds_name:20s}  SKIPPED (not in HPARAMS)")
            continue

        ratios, reliabs, corrs, corr_as, corr_bs = [], [], [], [], []

        for seed in SEEDS:
            cfg = build_cfg(ds_name)
            cfg.training.seed = seed
            set_seed(seed)
            loaders = make_dataloaders(cfg)
            set_seed(seed)
            _, _, dm = discover_environments(cfg, loaders, device)

            ratios.append(dm["adaptive/signal_ratio"])
            reliabs.append(dm["adaptive/reliability"])
            corrs.append(dm.get("discovery/assignment_color_abs_corr", float("nan")))
            corr_as.append(dm.get("discovery/color_label_corr_A", float("nan")))
            corr_bs.append(dm.get("discovery/color_label_corr_B", float("nan")))

        N = len(loaders["train"].dataset)
        print(f"{ds_name:20s} {N:7d} {np.mean(ratios):13.1f} {np.mean(reliabs):12.2f} "
              f"{np.nanmean(corrs):12.3f} {np.nanmean(corr_as):8.3f} {np.nanmean(corr_bs):8.3f}")


if __name__ == "__main__":
    main()
