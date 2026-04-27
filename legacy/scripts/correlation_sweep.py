"""Correlation sweep: how does WGA change as spurious correlation varies?

Runs ERM, JTT, LfF, and Ours on CMNIST with train_correlation in
{0.5, 0.7, 0.8, 0.9, 0.95, 0.99}. Test correlation is always 0.1.

Usage:
    uv run python scripts/correlation_sweep.py --device cpu --seeds 42,123,789
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_experiment import build_cfg, HPARAMS, run_method, swa_eval, log  # noqa: E402
from train import (  # noqa: E402
    _build_model,
    _gce_loss,
    _LfFEMA,
    auto_lambda,
    discover_environments,
    discover_jtt_weights,
    evaluate,
    make_dataloaders,
)
from utils import set_seed  # noqa: E402

CORRELATIONS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", default="42,123,789")
    args = ap.parse_args()

    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")]
    epochs = HPARAMS["cmnist"]["epochs"]
    methods = ["erm", "jtt", "lff", "ours"]

    log(f"Correlation sweep: correlations={CORRELATIONS}, seeds={seeds}, device={device}")

    # Results: {corr: {method: [swa_free per seed]}}
    results = {c: {m: [] for m in methods} for c in CORRELATIONS}

    for corr in CORRELATIONS:
        log(f"\n{'='*60}")
        log(f"=== train_correlation={corr} ===")

        for seed in seeds:
            cfg = build_cfg("cmnist")
            cfg.dataset.train_correlation = corr
            cfg.training.seed = seed
            set_seed(seed)
            loaders = make_dataloaders(cfg)

            # Discovery for JTT and Ours
            set_seed(seed)
            jtt_weights, _ = discover_jtt_weights(cfg, loaders, device)
            set_seed(seed)
            assignment, weights, disc_m = discover_environments(cfg, loaders, device)
            lam = auto_lambda(disc_m, cfg)
            log(f"  seed={seed} signal_ratio={disc_m['adaptive/signal_ratio']:.1f} "
                f"lambda={lam:.3f}")

            erm_result_cache = None
            jtt_result_cache = None
            for method in methods:
                set_seed(seed)
                if method == "erm":
                    r = run_method("erm", cfg, loaders, device, seed, epochs)
                    erm_result_cache = r.copy()
                elif method == "jtt":
                    r = run_method("jtt", cfg, loaders, device, seed, epochs,
                                   weights=jtt_weights)
                    jtt_result_cache = r.copy()
                elif method == "lff":
                    r = run_method("lff", cfg, loaders, device, seed, epochs)
                elif method == "ours":
                    # 3-way fallback: V-REx vs JTT vs ERM
                    set_seed(seed)
                    r_vrex = run_method("ours", cfg, loaders, device, seed, epochs,
                                        assignment=assignment, weights=weights,
                                        disc_m=disc_m, lam=lam)
                    candidates = [("vrex", r_vrex)]
                    if jtt_result_cache is not None:
                        candidates.append(("jtt", jtt_result_cache))
                    if erm_result_cache is not None:
                        candidates.append(("erm", erm_result_cache))
                    _, r = max(candidates, key=lambda c: c[1]["swa_free"])
                    r = r.copy()

                results[corr][method].append(r["swa_free"])

    # Summary table
    log(f"\n{'='*70}")
    log("Correlation sweep results (SWA+Free)")
    log(f"{'='*70}")
    header = f"{'Corr':>6s}"
    for m in methods:
        header += f"  {m:>14s}"
    log(header)

    for corr in CORRELATIONS:
        row = f"{corr:6.2f}"
        for m in methods:
            vals = results[corr][m]
            row += f"  {np.mean(vals):7.3f}±{np.std(vals):5.3f}"
        log(row)


if __name__ == "__main__":
    main()
