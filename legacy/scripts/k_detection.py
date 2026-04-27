"""K detection from the discovery ERM's per-example loss histogram.

Tests whether we can automatically pick K (the number of environments) from
the shape of the loss distribution, rather than hard-coding K=2.

Approach:
    1. Train a discovery ERM (5 epochs, frozen backbone for ResNet, full
       fine-tune for MLP) — same protocol as discover_environments.
    2. Dump per-example training losses (averaged over the last few epochs).
    3. Fit Gaussian mixture models with K = 1, 2, 3, 4, 5, 6.
    4. Pick the K that minimizes BIC (Bayesian Information Criterion).
    5. Compare to ground truth: how many distinct minority groups exist?

Usage:
    uv run python scripts/k_detection.py --dataset cmnist
    uv run python scripts/k_detection.py --dataset multi_cmnist
    uv run python scripts/k_detection.py --dataset tadf

Validation targets (ground-truth K):
    cmnist:        2 (single spurious feature → 2 groups: aligned, misaligned)
    multi_cmnist:  4 (two spurious features → 4 combinations)
    tadf:          ~2 (weak signal → bimodal but possibly 1)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import MLP, make_resnet_backbone  # noqa: E402
from train import make_dataloaders  # noqa: E402
from utils import set_seed  # noqa: E402

from run_experiment import build_cfg, HPARAMS  # noqa: E402

K_RANGE = list(range(1, 7))  # try K=1..6
DISCOVERY_EPOCHS = 5
SEED = 42


def log(msg: str) -> None:
    print(msg, flush=True)


def collect_discovery_losses(cfg, device: torch.device) -> tuple[np.ndarray, dict]:
    """Train a throw-away ERM and return per-example losses on the train set.

    Mirrors the loss-averaging strategy used by discover_environments
    (average over the last half of training epochs for stability).
    Also returns ground-truth labels and spurious attributes if available.
    """
    set_seed(SEED)
    cfg.training.seed = SEED
    loaders = make_dataloaders(cfg)
    train_ds = loaders["train"].dataset
    N = len(train_ds)

    if cfg.dataset.arch == "resnet":
        backbone, out_dim = make_resnet_backbone(freeze=True)
        model = MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
    else:
        model = MLP(input_dim=train_ds.input_dim, hidden_dim=cfg.model.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    avg_window = max(1, DISCOVERY_EPOCHS // 2)
    accumulated = torch.zeros(N)
    n_acc = 0

    for ep in range(DISCOVERY_EPOCHS):
        model.train()
        for batch in loaders["train"]:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if ep >= DISCOVERY_EPOCHS - avg_window:
            model.eval()
            with torch.no_grad():
                for batch in loaders["train"]:
                    x = batch["image"].to(device)
                    y = batch["label"].to(device)
                    idx = batch["index"]
                    ce = F.cross_entropy(model(x), y, reduction="none").cpu()
                    accumulated[idx] += ce
            n_acc += 1

    losses = (accumulated / max(n_acc, 1)).numpy()

    # Collect ground-truth attributes if available, for validation.
    extras = {}
    if hasattr(train_ds, "labels"):
        extras["labels"] = torch.as_tensor(train_ds.labels).numpy()
    if hasattr(train_ds, "spurious"):
        extras["spurious"] = torch.as_tensor(train_ds.spurious).numpy()
    # Multi-spurious dataset exposes a second .brightness attribute
    if hasattr(train_ds, "brightness"):
        extras["brightness"] = torch.as_tensor(train_ds.brightness).numpy()

    return losses, extras


def fit_gmm_bic(losses: np.ndarray, k_range=K_RANGE, min_frac: float = 0.02) -> dict:
    """Fit GMMs with various K and return BIC, AIC, and chosen K.

    A K is marked INVALID if its smallest component has fewer than
    ``min_frac * N`` examples — this filters out GMM solutions that overfit
    the loss histogram with tiny noise clusters.
    """
    losses_2d = losses.reshape(-1, 1)
    N = len(losses)
    min_count = max(20, int(min_frac * N))

    results = {}
    bics = {}
    aics = {}
    valid = {}  # K → True/False
    min_counts = {}
    for K in k_range:
        try:
            gmm = GaussianMixture(
                n_components=K,
                covariance_type="full",
                random_state=SEED,
                n_init=3,
                max_iter=200,
            )
            gmm.fit(losses_2d)
            assignments = gmm.predict(losses_2d)
            counts = np.bincount(assignments, minlength=K)
            min_counts[K] = int(counts.min())
            valid[K] = counts.min() >= min_count
            bics[K] = float(gmm.bic(losses_2d))
            aics[K] = float(gmm.aic(losses_2d))
            results[K] = gmm
        except Exception as e:
            log(f"  K={K}: failed ({e})")
            bics[K] = np.inf
            aics[K] = np.inf
            valid[K] = False
            min_counts[K] = 0

    best_k_bic = min(bics, key=bics.get)
    best_k_aic = min(aics, key=aics.get)
    elbow_k = pick_elbow_k(bics, valid)

    return {
        "bics": bics,
        "aics": aics,
        "valid": valid,
        "min_counts": min_counts,
        "min_required": min_count,
        "best_k_bic": best_k_bic,
        "best_k_aic": best_k_aic,
        "elbow_k": elbow_k,
        "models": results,
    }


def pick_elbow_k(bics: dict, valid: dict, drop_ratio: float = 0.05) -> int:
    """Find the elbow in BIC vs K, restricted to K values with no tiny components.

    Pure min-BIC overfits because the loss distribution has fine sub-structure
    (label noise, instance difficulty) that GMM can chase but is irrelevant
    for DRO. The elbow captures the largest K that produces a *meaningful*
    improvement — defined as a drop ≥ ``drop_ratio`` × max(drop) — and where
    every component has at least ``min_frac * N`` examples.

    Args:
        bics: dict mapping K → BIC value (smaller is better)
        valid: dict mapping K → bool (True if min component size satisfied)
        drop_ratio: threshold for "meaningful drop" as a fraction of the
            largest single-step BIC improvement among VALID Ks.

    Returns:
        The elbow K, with a floor of 2 (DRO needs at least 2 environments).
    """
    valid_ks = sorted([K for K in bics if valid.get(K, True)])
    if len(valid_ks) < 2:
        return 2
    drops = {
        valid_ks[i]: bics[valid_ks[i - 1]] - bics[valid_ks[i]]
        for i in range(1, len(valid_ks))
    }
    max_drop = max(drops.values()) if drops else 0
    if max_drop <= 0:
        return 2
    threshold = max_drop * drop_ratio
    chosen = valid_ks[0]
    for K in valid_ks[1:]:
        if drops[K] >= threshold:
            chosen = K
    return max(2, chosen)


def describe_loss_distribution(losses: np.ndarray) -> str:
    """One-line summary of the loss distribution shape."""
    return (
        f"min={losses.min():.4f}  max={losses.max():.4f}  "
        f"mean={losses.mean():.4f}  std={losses.std():.4f}  "
        f"median={np.median(losses):.4f}  N={len(losses)}"
    )


def compute_oracle_groups(extras: dict) -> tuple[int, dict]:
    """Compute the number of distinct ground-truth (label, spurious) groups
    and report each group's loss statistics.
    """
    if "brightness" in extras and "spurious" in extras and "labels" in extras:
        # Multi-spurious CMNIST: relevant grouping is by (color match, brightness
        # match) combinations: 4 groups.
        labels = extras["labels"]
        c = extras["spurious"]  # spurious = colors for MultiSpuriousCMNIST
        b = extras["brightness"]
        key = ((c == labels).astype(int) << 1) | (b == labels).astype(int)
        return 4, {"groups": key}
    if "labels" in extras and "spurious" in extras:
        labels = extras["labels"]
        s = extras["spurious"]
        # Group key: 2 * label + spurious → up to 4 groups
        key = 2 * labels + s
        return int(key.max() + 1), {"groups": key}
    return 0, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HPARAMS.keys()))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save", default=None, help="Optional path to save losses + extras as .npz")
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = build_cfg(args.dataset)

    log(f"=== K detection for {args.dataset} (device={device}, seed={SEED}) ===")
    log("Training discovery ERM ...")
    losses, extras = collect_discovery_losses(cfg, device)
    log(f"loss distribution: {describe_loss_distribution(losses)}")

    log("\nFitting GMMs over K ∈ {1..6} ...")
    res = fit_gmm_bic(losses)
    log(f"\nBIC and AIC vs K  (min component size required = {res['min_required']}):")
    log(f"  K  {'BIC':>14s}  {'AIC':>14s}  {'min_count':>10s}  valid?")
    for K in K_RANGE:
        marker = ""
        if K == res["elbow_k"]:
            marker = " ← elbow"
        elif K == res["best_k_bic"]:
            marker = " (min BIC)"
        log(
            f"  {K}  {res['bics'][K]:14.1f}  {res['aics'][K]:14.1f}  "
            f"{res['min_counts'][K]:10d}  "
            f"{'✓' if res['valid'][K] else '✗'}{marker}"
        )
    log(f"\n→ Best K (min BIC)  = {res['best_k_bic']}  (overfits — keeps fine structure)")
    log(f"→ Best K (min AIC)  = {res['best_k_aic']}")
    log(f"→ Elbow K           = {res['elbow_k']}  ← used for DRO")

    # Validate against oracle groups
    oracle_k, oracle_info = compute_oracle_groups(extras)
    if oracle_k > 0:
        log(f"\nOracle group count: {oracle_k}")
        oracle_groups = oracle_info["groups"]
        log("Per-oracle-group loss statistics:")
        for g in sorted(np.unique(oracle_groups)):
            mask = oracle_groups == g
            ng = mask.sum()
            log(
                f"  group {g}: n={ng:6d}  "
                f"loss mean={losses[mask].mean():.4f}  "
                f"std={losses[mask].std():.4f}  "
                f"median={np.median(losses[mask]):.4f}"
            )

        log(f"\nVerdict: detected K={res['best_k_bic']}  vs  oracle K={oracle_k}")

    if args.save:
        np.savez(
            args.save,
            losses=losses,
            **{k: v for k, v in extras.items() if isinstance(v, np.ndarray)},
        )
        log(f"\nSaved losses + extras to {args.save}")


if __name__ == "__main__":
    main()
