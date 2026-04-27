"""Resampling-OOD experiment on Waterbirds — the key validation.

Tests whether our method provides **reliably good OOD performance across
training data compositions** on real images (ResNet-50 + Waterbirds).

Protocol:
    1. For each of N_SUBSAMPLES random 90%-subsets of training data:
       a. Train an ERM model on that subset
       b. Train our method (discovery + upweight + V-REx with auto-λ + SWA)
       c. Evaluate both on the FULL test set
    2. Report per-subsample WGA and the distribution across subsamples

The critical test: does our worst subsample beat ERM's best?

Usage:
    uv run python scripts/resampling_ood_waterbirds.py --device mps
    uv run python scripts/resampling_ood_waterbirds.py --device mps --n_subsamples 5

Expected runtime: ~25 min per subsample × 2 methods × N_SUBSAMPLES.
Default 10 subsamples ≈ 8 hours.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import MLP, make_resnet_backbone  # noqa: E402
from train import (  # noqa: E402
    _ModelSelector,
    _val_score,
    discover_environments,
    evaluate,
    make_dataloaders,
)
from utils import set_seed  # noqa: E402
from dro_discovered import build_cfg  # noqa: E402
from resampling_stability_test import _ReindexedSubset, make_subsampled_loader  # noqa: E402

N_SUBSAMPLES = 10
SUBSAMPLE_FRAC = 0.9
SWA_WINDOW = 5
EPOCHS = 15
LR = 1e-4
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-4
UPWEIGHT = 50.0


def log(msg: str) -> None:
    print(msg, flush=True)


def train_erm_waterbirds(full_loaders, subset_indices, device, seed):
    """Train plain ERM on a subset of Waterbirds."""
    train_ds = full_loaders["train"].dataset
    sub_loader = make_subsampled_loader(train_ds, subset_indices, BATCH_SIZE)

    set_seed(seed)
    backbone, out_dim = make_resnet_backbone()
    model = MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    selector = _ModelSelector()
    for epoch in range(EPOCHS):
        model.train()
        for batch in sub_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        id_m = evaluate(model, full_loaders["id_test"], device)
        ood_m = evaluate(model, full_loaders["ood_test"], device)
        selector.update(
            _val_score(id_m),
            model,
            {
                "wga": ood_m.get("worst_group_acc", 0.0),
                "acc": ood_m.get("acc", 0.0),
            },
        )

    best = selector.restore(model)
    return best


def train_ours_waterbirds(cfg, full_loaders, subset_indices, device, seed):
    """Train our method (discovery + upweight + V-REx + SWA) on a subset."""
    train_ds = full_loaders["train"].dataset
    N_sub = len(subset_indices)
    sub_loader = make_subsampled_loader(train_ds, subset_indices, BATCH_SIZE)
    sub_loaders = {
        "train": sub_loader,
        "id_test": full_loaders["id_test"],
        "ood_test": full_loaders["ood_test"],
    }

    # Discovery on the subset
    set_seed(seed + 1000)
    assignment, weights, disc_m = discover_environments(cfg, sub_loaders, device)
    reliability = disc_m.get("adaptive/reliability", 1.0)
    actual_rv = disc_m["adaptive/actual_risk_var"]
    signal_ratio = disc_m.get("adaptive/signal_ratio", 0.0)

    # Auto-lambda from discovery quantities
    # Estimate L_mean from the per-env losses (need a quick forward pass)
    # Since we can't easily extract L_mean from discover_environments,
    # use the N-scaling rule as a reliable fallback:
    # λ = 10 * (5000 / N_sub) * reliability ≈ 10.4 * reliability for 90% Waterbirds
    # This is the rule validated on both Waterbirds (λ=10) and CMNIST (λ=0.83).
    lam = 10.0 * (5000 / N_sub) * reliability

    log(
        f"    discovery: signal_ratio={signal_ratio:.1f}  "
        f"reliability={reliability:.2f}  lambda={lam:.2f}"
    )

    # Fresh model for V-REx training
    set_seed(seed)
    backbone, out_dim = make_resnet_backbone()
    model = MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    assignment_dev = assignment.to(device)
    weights_dev = weights.to(device)
    K = 2

    selector = _ModelSelector()
    all_states: list = []

    for epoch in range(EPOCHS):
        model.train()
        for batch in sub_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            idx = batch["index"].to(device)

            ce = F.cross_entropy(model(x), y, reduction="none")
            a = assignment_dev[idx]
            w = weights_dev[idx]

            env_losses = []
            for k in range(K):
                mask = a == k
                if mask.any():
                    wk = w[mask]
                    env_losses.append((wk * ce[mask]).sum() / wk.sum())

            if len(env_losses) >= 2:
                env_t = torch.stack(env_losses)
                mean_loss = env_t.mean()
                risk_var = ((env_t - mean_loss) ** 2).sum()
                loss = mean_loss + lam * risk_var
            else:
                loss = ce.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        id_m = evaluate(model, full_loaders["id_test"], device)
        ood_m = evaluate(model, full_loaders["ood_test"], device)
        val_wga = id_m.get("worst_group_acc", 0.0)
        ood_wga = ood_m.get("worst_group_acc", 0.0)
        ood_acc = ood_m.get("acc", 0.0)

        selector.update(
            _val_score(id_m),
            model,
            {"epoch": epoch, "wga": ood_wga, "acc": ood_acc},
        )
        all_states.append(
            {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        )

    # SWA anchored at best-by-val epoch
    best_epoch = selector.best_metrics.get("epoch", len(all_states) - 1)
    swa_end = best_epoch + 1
    swa_start = max(0, swa_end - SWA_WINDOW)
    swa_window = all_states[swa_start:swa_end]

    swa_state = {}
    for key in swa_window[-1]:
        tensors = [s[key] for s in swa_window]
        if tensors[0].is_floating_point():
            swa_state[key] = torch.stack(tensors, dim=0).mean(dim=0)
        else:
            swa_state[key] = tensors[-1]

    model.load_state_dict({k: v.to(device) for k, v in swa_state.items()})

    # Recompute BN stats
    model.train()
    with torch.no_grad():
        for batch in sub_loader:
            model(batch["image"].to(device))

    swa_ood_m = evaluate(model, full_loaders["ood_test"], device)
    swa_wga = swa_ood_m.get("worst_group_acc", 0.0)
    swa_acc = swa_ood_m.get("acc", 0.0)

    best = selector.restore(model)
    best["swa_wga"] = swa_wga
    best["swa_acc"] = swa_acc
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n_subsamples", type=int, default=N_SUBSAMPLES)
    ap.add_argument("--subsample_frac", type=float, default=SUBSAMPLE_FRAC)
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = build_cfg("waterbirds")
    cfg.training.seed = 42
    set_seed(42)

    log(f"=== Waterbirds resampling-OOD experiment ===")
    log(f"device={device}  n_subsamples={args.n_subsamples}  frac={args.subsample_frac}")

    loaders = make_dataloaders(cfg)
    N = len(loaders["train"].dataset)
    N_sub = int(N * args.subsample_frac)
    log(f"N_train={N}  N_sub={N_sub}")

    g = torch.Generator().manual_seed(42)

    erm_results = []
    ours_results = []

    for s in range(args.n_subsamples):
        log(f"\n--- Subsample {s+1}/{args.n_subsamples} ---")

        perm = torch.randperm(N, generator=g)
        subset_indices = perm[:N_sub].tolist()

        # ERM
        log(f"  Training ERM ...")
        erm_r = train_erm_waterbirds(loaders, subset_indices, device, seed=100 + s)
        erm_results.append(erm_r)
        log(f"  ERM:  wga={erm_r['wga']:.4f}  acc={erm_r['acc']:.4f}")

        # Ours
        log(f"  Training ours ...")
        ours_r = train_ours_waterbirds(cfg, loaders, subset_indices, device, seed=200 + s)
        ours_results.append(ours_r)
        log(
            f"  Ours: wga={ours_r['wga']:.4f} (best)  "
            f"swa_wga={ours_r['swa_wga']:.4f}  acc={ours_r['acc']:.4f}"
        )

    # Summary
    erm_wgas = np.array([r["wga"] for r in erm_results])
    ours_wgas = np.array([r["wga"] for r in ours_results])
    ours_swa_wgas = np.array([r["swa_wga"] for r in ours_results])
    erm_accs = np.array([r["acc"] for r in erm_results])
    ours_accs = np.array([r["acc"] for r in ours_results])

    log(f"\n{'='*60}")
    log(f"=== RESULTS: Waterbirds resampling-OOD ({args.n_subsamples} subsamples) ===")
    log(f"{'='*60}")
    log(f"")
    log(f"WGA across subsamples:")
    log(f"  ERM:       mean={erm_wgas.mean():.4f} ± {erm_wgas.std():.4f}  "
        f"min={erm_wgas.min():.4f}  max={erm_wgas.max():.4f}")
    log(f"  Ours BEST: mean={ours_wgas.mean():.4f} ± {ours_wgas.std():.4f}  "
        f"min={ours_wgas.min():.4f}  max={ours_wgas.max():.4f}")
    log(f"  Ours SWA:  mean={ours_swa_wgas.mean():.4f} ± {ours_swa_wgas.std():.4f}  "
        f"min={ours_swa_wgas.min():.4f}  max={ours_swa_wgas.max():.4f}")
    log(f"")
    log(f"Ours-SWA worst > ERM best? {ours_swa_wgas.min() > erm_wgas.max()}  "
        f"(gap = {ours_swa_wgas.min() - erm_wgas.max():.4f})")
    log(f"")
    log(f"OOD accuracy:")
    log(f"  ERM:  mean={erm_accs.mean():.4f} ± {erm_accs.std():.4f}")
    log(f"  Ours: mean={ours_accs.mean():.4f} ± {ours_accs.std():.4f}")


if __name__ == "__main__":
    main()
