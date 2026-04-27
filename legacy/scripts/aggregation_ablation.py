"""Aggregation ablation — same discovery, different training objectives.

Produces the core ablation table showing that the choice of aggregator matters
more than the choice of discovery. All four methods use the same K=2 loss-based
environments and the same upweight=50 factor; only the loss aggregation differs.

Aggregators:
    erm      :  uniform mean (standard ERM on reweighted data)
    vrex     :  (L_A - L_B)^2 penalty added to mean
    dro      :  pure Group DRO on discovered envs (our method)
    dro_oracle : pure Group DRO on ground-truth groups (upper bound for discovery)

Usage:
    uv run python scripts/aggregation_ablation.py --dataset waterbirds --seeds 42,123,789

Outputs per-seed metrics and a final summary table.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

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

# Reuse the per-dataset config from dro_discovered.py to guarantee identical
# hyper-parameters across runs.
from dro_discovered import DATASET_TRAINING, build_cfg  # noqa: E402

AGGREGATORS = ["erm", "vrex", "dro", "dro_oracle"]

def _to_device(x, device):
    if isinstance(x, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
    return x.to(device)

DRO_STEP_SIZE = 0.01
VREX_LAMBDA = 10.0
SWA_WINDOW = 5


def log(msg: str) -> None:
    print(msg, flush=True)


def oracle_assignment(train_ds) -> torch.Tensor:
    """Return ground-truth group assignments, collapsed to K=2 (majority vs minority).

    Minority = examples where the spurious feature disagrees with the label
    (waterbird on land, landbird on water; colour ≠ label on CMNIST). This is
    the standard 'majority vs minority' split that matches what loss-based
    discovery is trying to recover.
    """
    if not hasattr(train_ds, "spurious"):
        raise ValueError("Dataset has no 'spurious' attribute for oracle grouping")
    y = torch.as_tensor(train_ds.labels)
    s = torch.as_tensor(train_ds.spurious)
    return (y != s).long()


def build_model(cfg, loaders, device):
    if cfg.dataset.arch == "resnet":
        backbone, out_dim = make_resnet_backbone()
        return MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
    if cfg.dataset.arch == "distilbert":
        from models import make_distilbert_backbone
        backbone, out_dim = make_distilbert_backbone()
        return MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
    input_dim = loaders["train"].dataset.input_dim
    return MLP(input_dim=input_dim, hidden_dim=cfg.model.hidden_dim).to(device)


def train_one(
    aggregator: str,
    cfg,
    seed: int,
    device: torch.device,
    epochs: int,
    assignment: torch.Tensor,
    weights: torch.Tensor,
) -> dict:
    """Run one training with the specified aggregator. Returns best-by-val + SWA metrics."""
    set_seed(seed)
    loaders = make_dataloaders(cfg)
    model = build_model(cfg, loaders, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )

    assignment_dev = assignment.to(device)
    weights_dev = weights.to(device)
    K = int(assignment.max().item()) + 1
    group_weights = torch.ones(K, device=device) / K

    selector = _ModelSelector()
    all_states: list = []

    for epoch in range(epochs):
        model.train()
        for batch in loaders["train"]:
            x = _to_device(batch["image"], device)
            y = batch["label"].to(device)
            idx = batch["index"].to(device)

            ce = F.cross_entropy(model(x), y, reduction="none")
            a = assignment_dev[idx]
            w = weights_dev[idx]

            env_losses = []
            env_present = []
            for k in range(K):
                mask = a == k
                if mask.any():
                    wk = w[mask]
                    env_losses.append((wk * ce[mask]).sum() / wk.sum())
                    env_present.append(k)

            if len(env_losses) >= 2:
                env_t = torch.stack(env_losses)
                if aggregator == "erm":
                    # Weighted mean only — the reweighting is via `weights_dev`.
                    loss = (w * ce).sum() / w.sum()
                elif aggregator == "vrex":
                    # Mean + variance penalty. Matches our prior V-REx code.
                    mean_loss = env_t.mean()
                    variance = ((env_t - mean_loss) ** 2).sum()
                    loss = mean_loss + VREX_LAMBDA * variance
                elif aggregator in ("dro", "dro_oracle"):
                    # Pure Group DRO — weighted sum over env losses.
                    gw = group_weights[env_present].detach()
                    loss = (gw * env_t).sum()
                else:
                    raise ValueError(f"Unknown aggregator: {aggregator}")
            else:
                loss = ce.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # DRO: update group weights AFTER the gradient step.
            if aggregator in ("dro", "dro_oracle") and len(env_losses) >= 2:
                with torch.no_grad():
                    group_weights[env_present] *= torch.exp(
                        DRO_STEP_SIZE * env_t.detach()
                    )
                    group_weights /= group_weights.sum()

        id_m = evaluate(model, loaders["id_test"], device)
        ood_m = evaluate(model, loaders["ood_test"], device)
        val_wga = id_m.get("worst_group_acc", id_m.get("acc", 0.0))
        ood_wga = ood_m.get("worst_group_acc", ood_m.get("acc", 0.0))
        ood_acc = ood_m.get("acc", 0.0)

        selector.update(
            _val_score(id_m),
            model,
            {"epoch": epoch, "val_wga": val_wga, "ood_wga": ood_wga, "ood_acc": ood_acc},
        )
        all_states.append(
            {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        )

        log(
            f"  [{aggregator:11s} seed={seed}] ep{epoch:02d}  "
            f"val={val_wga:.4f}  ood_wga={ood_wga:.4f}  ood_acc={ood_acc:.4f}"
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
    if cfg.dataset.arch == "resnet":
        model.train()
        with torch.no_grad():
            for batch in loaders["train"]:
                model(_to_device(batch["image"], device))

    swa_id_m = evaluate(model, loaders["id_test"], device)
    swa_ood_m = evaluate(model, loaders["ood_test"], device)
    swa_ood_wga = swa_ood_m.get("worst_group_acc", swa_ood_m.get("acc", 0.0))
    swa_ood_acc = swa_ood_m.get("acc", 0.0)

    best = selector.restore(model)
    best["swa_ood_wga"] = swa_ood_wga
    best["swa_ood_acc"] = swa_ood_acc
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_TRAINING.keys()))
    ap.add_argument("--seeds", default="42,123,789")
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--aggregators",
        default=",".join(AGGREGATORS),
        help="Comma-separated subset of {erm,vrex,dro,dro_oracle}",
    )
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )
    else:
        device = torch.device(args.device)

    seeds = [int(s) for s in args.seeds.split(",")]
    aggregators = args.aggregators.split(",")
    cfg = build_cfg(args.dataset)
    epochs = cfg.training.epochs

    log(f"dataset={args.dataset}  device={device}  seeds={seeds}  aggregators={aggregators}")

    # results[aggregator][seed] = best_metrics
    results: dict[str, dict[int, dict]] = {agg: {} for agg in aggregators}

    for seed in seeds:
        log(f"\n========== SEED {seed} ==========")
        t0 = time.time()
        cfg.training.seed = seed
        set_seed(seed)
        loaders = make_dataloaders(cfg)

        # Shared discovery (used by erm, vrex, dro)
        log(f"[seed={seed}] discovering environments ...")
        assignment, weights, disc_m = discover_environments(cfg, loaders, device)
        log(
            f"[seed={seed}] discovery signal_ratio="
            f"{disc_m.get('adaptive/signal_ratio', float('nan')):.1f}"
        )

        # Oracle discovery (used by dro_oracle)
        oracle_a = None
        if "dro_oracle" in aggregators:
            try:
                oracle_a = oracle_assignment(loaders["train"].dataset)
                oracle_w = torch.ones(len(oracle_a))  # uniform weights for oracle
                log(
                    f"[seed={seed}] oracle groups: n_majority="
                    f"{(oracle_a == 0).sum().item()}  "
                    f"n_minority={(oracle_a == 1).sum().item()}"
                )
            except ValueError as e:
                log(f"[seed={seed}] no oracle available: {e}")
                if "dro_oracle" in aggregators:
                    aggregators = [a for a in aggregators if a != "dro_oracle"]

        for agg in aggregators:
            if agg == "dro_oracle" and oracle_a is None:
                continue
            log(f"\n  -- {agg} --")
            if agg == "dro_oracle":
                r = train_one(agg, cfg, seed, device, epochs, oracle_a, oracle_w)
            else:
                r = train_one(agg, cfg, seed, device, epochs, assignment, weights)
            results[agg][seed] = r

        log(f"[seed={seed}] done in {(time.time() - t0) / 60:.1f} min")

    log("\n" + "=" * 70)
    log(f"=== FINAL: {args.dataset} aggregation ablation ===")
    log("=" * 70)
    header = f"{'aggregator':12s}  {'BEST ood_wga':>18s}  {'SWA ood_wga':>18s}  {'BEST ood_acc':>18s}"
    log(header)
    for agg in aggregators:
        if not results[agg]:
            continue
        best_wgas = [r["ood_wga"] for r in results[agg].values()]
        swa_wgas = [r["swa_ood_wga"] for r in results[agg].values()]
        best_accs = [r["ood_acc"] for r in results[agg].values()]
        log(
            f"{agg:12s}  "
            f"{np.mean(best_wgas):.4f} ± {np.std(best_wgas):.4f}  "
            f"{np.mean(swa_wgas):.4f} ± {np.std(swa_wgas):.4f}  "
            f"{np.mean(best_accs):.4f} ± {np.std(best_accs):.4f}"
        )


if __name__ == "__main__":
    main()
