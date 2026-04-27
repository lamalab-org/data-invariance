"""Ingredient ablation: which components of our method contribute what?

Tests on a given dataset:
  A) ERM (no discovery, no upweight, no V-REx)
  B) Discovery + upweight only (λ=0, equivalent to loss-based JTT)
  C) Discovery + V-REx only (no upweight, α=0)
  D) Full method (upweight + V-REx + auto-λ)
  E) Full + SWA model selection

Usage:
    uv run python scripts/ingredient_ablation.py --dataset cmnist --device cpu
    uv run python scripts/ingredient_ablation.py --dataset waterbirds --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_experiment import build_cfg, HPARAMS, run_method, log  # noqa: E402
from train import auto_lambda, discover_environments, make_dataloaders  # noqa: E402
from utils import set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HPARAMS.keys()))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", default="42,123,789,2024,7")
    args = ap.parse_args()

    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")]
    epochs = HPARAMS[args.dataset]["epochs"]

    conditions = {
        "ERM": {},
        "+upweight": {"lam_override": 0.0},
        "+VREx": {"no_upweight": True},
        "Full": {},
        "Full+SWA": {},  # SWA is always computed; this just highlights it
    }

    results = {name: [] for name in conditions}

    for seed in seeds:
        log(f"\n--- SEED {seed} ---")
        cfg = build_cfg(args.dataset)
        cfg.training.seed = seed
        set_seed(seed)
        loaders = make_dataloaders(cfg)

        # Discovery
        set_seed(seed)
        assignment, weights, disc_m = discover_environments(cfg, loaders, device)
        lam = auto_lambda(disc_m, cfg)

        # A) ERM
        set_seed(seed)
        r = run_method("erm", cfg, loaders, device, seed, epochs)
        results["ERM"].append(r["swa_free"])
        log(f"  ERM: {r['swa_free']:.4f}")

        # B) Discovery + upweight only (λ=0)
        set_seed(seed)
        r = run_method("ours", cfg, loaders, device, seed, epochs,
                        assignment=assignment, weights=weights, disc_m=disc_m, lam=0.0)
        results["+upweight"].append(r["swa_free"])
        log(f"  +upweight (λ=0): {r['swa_free']:.4f}")

        # C) Discovery + V-REx only (no upweight)
        no_upweight_weights = torch.ones_like(weights)
        set_seed(seed)
        r = run_method("ours", cfg, loaders, device, seed, epochs,
                        assignment=assignment, weights=no_upweight_weights, disc_m=disc_m, lam=lam)
        results["+VREx"].append(r["swa_free"])
        log(f"  +VREx (no upweight): {r['swa_free']:.4f}")

        # D) Full method
        set_seed(seed)
        r = run_method("ours", cfg, loaders, device, seed, epochs,
                        assignment=assignment, weights=weights, disc_m=disc_m, lam=lam)
        results["Full"].append(r["swa_free"])
        results["Full+SWA"].append(r["swa_free"])  # SWA is always used
        log(f"  Full: {r['swa_free']:.4f}")

    log(f"\n{'='*50}")
    log(f"Ingredient ablation on {args.dataset} ({len(seeds)} seeds)")
    log(f"{'='*50}")
    for name in ["ERM", "+upweight", "+VREx", "Full"]:
        vals = results[name]
        log(f"  {name:15s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
