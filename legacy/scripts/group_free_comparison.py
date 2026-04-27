"""Group-free model selection comparison: Ours vs JTT vs ERM.

The critical experiment: does JTT collapse without group-labeled validation
while our method holds up?

For each method, trains with TWO model selection strategies:
1. WGA-selected (standard: uses worst-group accuracy on group-labeled val set)
2. AvgAcc-selected (group-free: uses average val accuracy, no group labels)
3. SWA-anchored at the group-free best epoch (group-free + SWA)

If JTT drops 10+ pp under group-free selection while ours drops <3 pp, the
story is: "our auto-calibrated recipe makes group-free model selection viable
for the first time."

Usage:
    uv run python scripts/group_free_comparison.py --dataset waterbirds --device cpu
    uv run python scripts/group_free_comparison.py --dataset waterbirds --device cuda --seeds 42,123,789,2024,7
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import MLP, make_resnet_backbone  # noqa: E402
from train import (  # noqa: E402
    _ModelSelector,
    _to_device,
    discover_environments,
    evaluate,
    make_dataloaders,
)
from utils import set_seed  # noqa: E402
from dro_discovered import build_cfg, DATASET_TRAINING  # noqa: E402

SWA_WINDOW = 5


def log(msg: str) -> None:
    print(msg, flush=True)


class _GroupFreeSelector:
    """Model selection by average validation accuracy — no group labels needed."""
    def __init__(self):
        import copy
        self._copy = copy
        self.best_score = float("-inf")
        self.best_state = None
        self.best_metrics: dict = {}

    def update(self, val_metrics: dict, model, extra_metrics: dict):
        score = val_metrics["acc"]  # average accuracy, group-free
        if score > self.best_score:
            self.best_score = score
            self.best_state = self._copy.deepcopy(model.state_dict())
            self.best_metrics = dict(extra_metrics)

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)
        return self.best_metrics


def _build_model(cfg, loaders, device):
    if cfg.dataset.arch == "resnet":
        backbone, out_dim = make_resnet_backbone()
        return MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
    elif cfg.dataset.arch == "distilbert":
        from models import make_distilbert_backbone
        backbone, out_dim = make_distilbert_backbone()
        return MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
    return MLP(input_dim=loaders["train"].dataset.input_dim,
               hidden_dim=cfg.model.hidden_dim).to(device)


def _swa_eval(model, all_states, anchor_epoch, loaders, device, cfg):
    """SWA evaluation anchored at a given epoch."""
    swa_end = anchor_epoch + 1
    swa_start = max(0, swa_end - SWA_WINDOW)
    window = all_states[swa_start:swa_end]
    if not window:
        return {}

    swa_state = {}
    for key in window[-1]:
        tensors = [s[key] for s in window]
        if tensors[0].is_floating_point():
            swa_state[key] = torch.stack(tensors).mean(0)
        else:
            swa_state[key] = tensors[-1]

    model.load_state_dict({k: v.to(device) for k, v in swa_state.items()})

    # BN stats update for ResNet
    if cfg.dataset.arch == "resnet":
        model.train()
        with torch.no_grad():
            for batch in loaders["train"]:
                model(_to_device(batch["image"], device))

    ood_m = evaluate(model, loaders["ood_test"], device)
    return {
        "swa_wga": ood_m.get("worst_group_acc", ood_m.get("acc", 0)),
        "swa_acc": ood_m.get("acc", 0),
    }


def run_erm(cfg, loaders, device, seed, epochs):
    """Plain ERM baseline."""
    set_seed(seed)
    model = _build_model(cfg, loaders, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr,
                            weight_decay=cfg.training.weight_decay)

    sel_wga = _ModelSelector()
    sel_free = _GroupFreeSelector()
    all_states = []

    for epoch in range(epochs):
        model.train()
        for batch in loaders["train"]:
            x = _to_device(batch["image"], device)
            y = batch["label"].to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()

        id_m = evaluate(model, loaders["id_test"], device)
        ood_m = evaluate(model, loaders["ood_test"], device)
        wga = ood_m.get("worst_group_acc", ood_m.get("acc", 0))
        metrics = {"wga": wga, "acc": ood_m["acc"], "epoch": epoch}

        sel_wga.update(id_m.get("worst_group_acc", id_m.get("acc", 0)),
                       model, metrics)
        sel_free.update(id_m, model, metrics)
        all_states.append({k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()})

    best_wga = sel_wga.restore(model)
    best_free = sel_free.restore(model)
    swa_free = _swa_eval(model, all_states, best_free.get("epoch", len(all_states)-1),
                         loaders, device, cfg)

    return {
        "wga_sel": best_wga.get("wga", 0),
        "free_sel": best_free.get("wga", 0),
        "swa_free": swa_free.get("swa_wga", 0),
        "wga_epoch": best_wga.get("epoch", -1),
        "free_epoch": best_free.get("epoch", -1),
    }


def run_jtt(cfg, loaders, device, seed, epochs):
    """JTT: train ERM, identify misclassified, upweight, retrain ERM."""
    train_ds = loaders["train"].dataset
    N = len(train_ds)

    # Phase 1: discovery ERM (same as ours)
    set_seed(seed + 1000)
    if cfg.dataset.arch == "resnet":
        backbone, out_dim = make_resnet_backbone(freeze=True)
        disc = MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
    else:
        disc = MLP(input_dim=train_ds.input_dim, hidden_dim=cfg.model.hidden_dim).to(device)
    opt = torch.optim.AdamW(disc.parameters(), lr=cfg.training.lr,
                            weight_decay=cfg.training.weight_decay)

    for _ in range(cfg.training.discovery_epochs):
        disc.train()
        for batch in loaders["train"]:
            x = _to_device(batch["image"], device)
            y = batch["label"].to(device)
            loss = F.cross_entropy(disc(x), y)
            opt.zero_grad(); loss.backward(); opt.step()

    # Identify misclassified examples (JTT's binary identification)
    disc.eval()
    misclassified = torch.zeros(N, dtype=torch.bool)
    with torch.no_grad():
        for batch in loaders["train"]:
            x = _to_device(batch["image"], device)
            y = batch["label"].to(device)
            idx = batch["index"]
            preds = disc(x).argmax(1)
            misclassified[idx] = (preds != y).cpu()

    # Binary weights: misclassified -> upweight, correct -> 1
    upweight = 50.0
    weights = torch.ones(N)
    weights[misclassified] = upweight
    weights = weights.to(device)
    del disc

    # Phase 2: retrain with upweighting (standard JTT)
    set_seed(seed)
    model = _build_model(cfg, loaders, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr,
                            weight_decay=cfg.training.weight_decay)

    sel_wga = _ModelSelector()
    sel_free = _GroupFreeSelector()
    all_states = []

    for epoch in range(epochs):
        model.train()
        for batch in loaders["train"]:
            x = _to_device(batch["image"], device)
            y = batch["label"].to(device)
            idx = batch["index"].to(device)
            ce = F.cross_entropy(model(x), y, reduction="none")
            w = weights[idx]
            loss = (w * ce).sum() / w.sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        id_m = evaluate(model, loaders["id_test"], device)
        ood_m = evaluate(model, loaders["ood_test"], device)
        wga = ood_m.get("worst_group_acc", ood_m.get("acc", 0))
        metrics = {"wga": wga, "acc": ood_m["acc"], "epoch": epoch}

        sel_wga.update(id_m.get("worst_group_acc", id_m.get("acc", 0)),
                       model, metrics)
        sel_free.update(id_m, model, metrics)
        all_states.append({k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()})

    best_wga = sel_wga.restore(model)
    best_free = sel_free.restore(model)
    swa_free = _swa_eval(model, all_states, best_free.get("epoch", len(all_states)-1),
                         loaders, device, cfg)

    return {
        "wga_sel": best_wga.get("wga", 0),
        "free_sel": best_free.get("wga", 0),
        "swa_free": swa_free.get("swa_wga", 0),
        "wga_epoch": best_wga.get("epoch", -1),
        "free_epoch": best_free.get("epoch", -1),
        "n_misclassified": int(misclassified.sum().item()),
    }


def run_ours(cfg, loaders, device, seed, epochs):
    """Our method: discovery + upweight + V-REx + auto-λ + SWA."""
    set_seed(seed)
    assignment, weights, disc_m = discover_environments(cfg, loaders, device)
    reliability = disc_m.get("adaptive/reliability", 1.0)

    # Auto-lambda from discovery quantities
    N = len(loaders["train"].dataset)
    B = cfg.training.batch_size
    lr = cfg.training.lr
    # Use the N-scaling rule (validated on Waterbirds + CMNIST)
    lam = 10.0 * (5000 / N) * reliability

    set_seed(seed)
    model = _build_model(cfg, loaders, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=cfg.training.weight_decay)
    a = assignment.to(device)
    w = weights.to(device)
    K = 2

    sel_wga = _ModelSelector()
    sel_free = _GroupFreeSelector()
    all_states = []

    for epoch in range(epochs):
        model.train()
        for batch in loaders["train"]:
            x = _to_device(batch["image"], device)
            y = batch["label"].to(device)
            idx = batch["index"].to(device)
            ce = F.cross_entropy(model(x), y, reduction="none")
            batch_a = a[idx]
            batch_w = w[idx]

            env_losses = []
            for k in range(K):
                mask = batch_a == k
                if mask.any():
                    wk = batch_w[mask]
                    env_losses.append((wk * ce[mask]).sum() / wk.sum())

            if len(env_losses) >= 2:
                env_t = torch.stack(env_losses)
                mean_loss = env_t.mean()
                risk_var = ((env_t - mean_loss) ** 2).sum()
                loss = mean_loss + lam * risk_var
            else:
                loss = ce.mean()

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        id_m = evaluate(model, loaders["id_test"], device)
        ood_m = evaluate(model, loaders["ood_test"], device)
        wga = ood_m.get("worst_group_acc", ood_m.get("acc", 0))
        metrics = {"wga": wga, "acc": ood_m["acc"], "epoch": epoch}

        sel_wga.update(id_m.get("worst_group_acc", id_m.get("acc", 0)),
                       model, metrics)
        sel_free.update(id_m, model, metrics)
        all_states.append({k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()})

    best_wga = sel_wga.restore(model)
    best_free = sel_free.restore(model)
    swa_free = _swa_eval(model, all_states, best_free.get("epoch", len(all_states)-1),
                         loaders, device, cfg)

    return {
        "wga_sel": best_wga.get("wga", 0),
        "free_sel": best_free.get("wga", 0),
        "swa_free": swa_free.get("swa_wga", 0),
        "wga_epoch": best_wga.get("epoch", -1),
        "free_epoch": best_free.get("epoch", -1),
        "lambda": lam,
        "signal_ratio": disc_m.get("adaptive/signal_ratio", 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="waterbirds",
                    choices=list(DATASET_TRAINING.keys()))
    ap.add_argument("--seeds", default="42,123,789")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")]
    cfg = build_cfg(args.dataset)
    epochs = DATASET_TRAINING[args.dataset]["epochs"]

    log(f"=== Group-Free Model Selection Comparison ===")
    log(f"dataset={args.dataset}  device={device}  seeds={seeds}  epochs={epochs}")
    log(f"Methods: ERM, JTT (upweight=50), Ours (V-REx + auto-λ + SWA)")
    log(f"Selection: WGA (group-aware) vs AvgAcc (group-free) vs SWA+AvgAcc (group-free)")
    log("")

    all_results = {"erm": [], "jtt": [], "ours": []}

    for seed in seeds:
        log(f"--- SEED {seed} ---")
        cfg.training.seed = seed
        set_seed(seed)
        loaders = make_dataloaders(cfg)

        t0 = time.time()
        erm_r = run_erm(cfg, loaders, device, seed, epochs)
        log(f"  ERM:  WGA-sel={erm_r['wga_sel']:.4f}(ep{erm_r['wga_epoch']})  "
            f"Free-sel={erm_r['free_sel']:.4f}(ep{erm_r['free_epoch']})  "
            f"SWA-free={erm_r['swa_free']:.4f}  [{time.time()-t0:.0f}s]")
        all_results["erm"].append(erm_r)

        t0 = time.time()
        jtt_r = run_jtt(cfg, loaders, device, seed, epochs)
        log(f"  JTT:  WGA-sel={jtt_r['wga_sel']:.4f}(ep{jtt_r['wga_epoch']})  "
            f"Free-sel={jtt_r['free_sel']:.4f}(ep{jtt_r['free_epoch']})  "
            f"SWA-free={jtt_r['swa_free']:.4f}  "
            f"[misclass={jtt_r['n_misclassified']}, {time.time()-t0:.0f}s]")
        all_results["jtt"].append(jtt_r)

        t0 = time.time()
        ours_r = run_ours(cfg, loaders, device, seed, epochs)
        log(f"  Ours: WGA-sel={ours_r['wga_sel']:.4f}(ep{ours_r['wga_epoch']})  "
            f"Free-sel={ours_r['free_sel']:.4f}(ep{ours_r['free_epoch']})  "
            f"SWA-free={ours_r['swa_free']:.4f}  "
            f"[λ={ours_r['lambda']:.2f}, sig={ours_r['signal_ratio']:.0f}, {time.time()-t0:.0f}s]")
        all_results["ours"].append(ours_r)
        log("")

    log("=" * 70)
    log(f"=== FINAL: {args.dataset} group-free model selection comparison ===")
    log("=" * 70)
    log("")

    header = f"{'Method':8s}  {'WGA-selected':>14s}  {'Free-selected':>14s}  {'SWA+Free':>14s}  {'WGA→Free drop':>14s}"
    log(header)
    for method in ["erm", "jtt", "ours"]:
        results = all_results[method]
        wga_vals = [r["wga_sel"] for r in results]
        free_vals = [r["free_sel"] for r in results]
        swa_vals = [r["swa_free"] for r in results]
        drop = np.mean(wga_vals) - np.mean(free_vals)
        log(f"{method:8s}  "
            f"{np.mean(wga_vals):.4f}±{np.std(wga_vals):.4f}  "
            f"{np.mean(free_vals):.4f}±{np.std(free_vals):.4f}  "
            f"{np.mean(swa_vals):.4f}±{np.std(swa_vals):.4f}  "
            f"{drop:+.4f}")

    log("")
    log("Key question: does JTT collapse under group-free selection while ours holds?")
    jtt_drop = np.mean([r["wga_sel"] for r in all_results["jtt"]]) - \
               np.mean([r["swa_free"] for r in all_results["jtt"]])
    ours_drop = np.mean([r["wga_sel"] for r in all_results["ours"]]) - \
                np.mean([r["swa_free"] for r in all_results["ours"]])
    log(f"  JTT  drop (WGA-sel → SWA-free): {jtt_drop:+.4f}")
    log(f"  Ours drop (WGA-sel → SWA-free): {ours_drop:+.4f}")
    if abs(jtt_drop) > 2 * abs(ours_drop) and ours_drop < 0.05:
        log("  → YES: JTT drops significantly more. Our method is robust to group-free selection.")
    else:
        log("  → Results are closer than expected. Need more analysis.")


if __name__ == "__main__":
    main()
