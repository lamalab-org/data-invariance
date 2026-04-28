"""Unified experiment script — runs ERM, JTT, LfF, and/or Ours on any dataset.

Reports all four model selection protocols per method:
  - WGA-sel:  best epoch by worst-group accuracy (uses group labels)
  - SWA+WGA:  SWA averaging anchored at WGA-sel epoch
  - Free-sel: best epoch by average accuracy (no group labels)
  - SWA+Free: SWA averaging anchored at Free-sel epoch

This is the single script that produces all paper numbers.

Usage:
    # Quick test
    python scripts/run_experiment.py --dataset cmnist --seeds 42 --device cpu

    # Full run (all 4 methods, 5 seeds)
    python scripts/run_experiment.py --dataset waterbirds --seeds 42,123,789,2024,7 --device cuda

    # Single method
    python scripts/run_experiment.py --dataset bace --seeds 42,123,789 --methods erm
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

from train import (
    _build_lff_biased_model,
    _build_model,
    _gce_loss,
    _LfFEMA,
    _ModelSelector,
    _to_device,
    _val_score,
    auto_lambda,
    discover_environments,
    discover_jtt_weights,
    evaluate,
    make_dataloaders,
)
from utils import set_seed

# ---------------------------------------------------------------------------
# Per-dataset training hyperparameters.  Only standard training hparams here —
# NO per-dataset tuning of the method itself (lambda, upweight, etc.).
# ---------------------------------------------------------------------------
HPARAMS = {
    "cmnist":            {"lr": 1e-3, "batch_size": 256, "epochs": 10, "discovery_epochs": 5},
    "continuous_cmnist": {"lr": 1e-3, "batch_size": 256, "epochs": 10, "discovery_epochs": 5},
    "multi_cmnist":      {"lr": 1e-3, "batch_size": 256, "epochs": 15, "discovery_epochs": 5},
    "tadf":              {"lr": 1e-3, "batch_size": 128, "epochs": 20, "discovery_epochs": 5},
    "mof_thermal":       {"lr": 1e-3, "batch_size": 128, "epochs": 20, "discovery_epochs": 5},
    "mof_solvent":       {"lr": 1e-3, "batch_size": 128, "epochs": 20, "discovery_epochs": 5},
    "perovskite":        {"lr": 1e-3, "batch_size": 256, "epochs": 15, "discovery_epochs": 5},
    "battery":           {"lr": 1e-3, "batch_size": 256, "epochs": 15, "discovery_epochs": 5},
    "bace":              {"lr": 1e-3, "batch_size": 64,  "epochs": 30, "discovery_epochs": 5},
    "bbbp":              {"lr": 1e-3, "batch_size": 64,  "epochs": 30, "discovery_epochs": 5},
    # ChemBERTa fine-tune: smaller LR, smaller batch, fewer epochs (transformer fine-tune defaults).
    "bace_chemberta":            {"lr": 2e-5, "batch_size": 32, "epochs": 10, "discovery_epochs": 3},
    "bbbp_chemberta":            {"lr": 2e-5, "batch_size": 32, "epochs": 10, "discovery_epochs": 3},
    # GIN: standard graph-network defaults.
    "bace_gin":                  {"lr": 1e-3, "batch_size": 32, "epochs": 50, "discovery_epochs": 5},
    "bbbp_gin":                  {"lr": 1e-3, "batch_size": 32, "epochs": 50, "discovery_epochs": 5},
    "pgp_broccatelli_chemberta": {"lr": 2e-5, "batch_size": 32, "epochs": 10, "discovery_epochs": 3},
    "bbb_martins_chemberta":     {"lr": 2e-5, "batch_size": 32, "epochs": 10, "discovery_epochs": 3},
    "ames_chemberta":            {"lr": 2e-5, "batch_size": 32, "epochs": 10, "discovery_epochs": 3},
    "dili_chemberta":            {"lr": 2e-5, "batch_size": 32, "epochs": 10, "discovery_epochs": 3},
    # TDC ADME / Tox single-task classification — same MLP + Morgan FP setup as MolNet
    "hia_hou":            {"lr": 1e-3, "batch_size": 64, "epochs": 30, "discovery_epochs": 5},
    "bioavailability_ma": {"lr": 1e-3, "batch_size": 64, "epochs": 30, "discovery_epochs": 5},
    "pgp_broccatelli":    {"lr": 1e-3, "batch_size": 64, "epochs": 30, "discovery_epochs": 5},
    "bbb_martins":        {"lr": 1e-3, "batch_size": 64, "epochs": 30, "discovery_epochs": 5},
    "herg":               {"lr": 1e-3, "batch_size": 64, "epochs": 30, "discovery_epochs": 5},
    "dili":               {"lr": 1e-3, "batch_size": 64, "epochs": 30, "discovery_epochs": 5},
    "ames":               {"lr": 1e-3, "batch_size": 64, "epochs": 30, "discovery_epochs": 5},
    "skin_reaction":      {"lr": 1e-3, "batch_size": 64, "epochs": 30, "discovery_epochs": 5},
    "hiv":               {"lr": 1e-3, "batch_size": 256, "epochs": 20, "discovery_epochs": 5},
    "clintox":           {"lr": 1e-3, "batch_size": 64,  "epochs": 30, "discovery_epochs": 5},
    "tox21":             {"lr": 1e-3, "batch_size": 128, "epochs": 25, "discovery_epochs": 5},
    "sider":             {"lr": 1e-3, "batch_size": 64,  "epochs": 30, "discovery_epochs": 5},
    "muv":               {"lr": 1e-3, "batch_size": 256, "epochs": 20, "discovery_epochs": 5},
    "pcba":              {"lr": 1e-3, "batch_size": 256, "epochs": 15, "discovery_epochs": 5},
    "waterbirds":        {"lr": 1e-4, "batch_size": 64,  "epochs": 15, "discovery_epochs": 5},
    "celeba":            {"lr": 1e-4, "batch_size": 128, "epochs": 10, "discovery_epochs": 3},
    "civilcomments":     {"lr": 2e-5, "batch_size": 32,  "epochs": 5,  "discovery_epochs": 2},
    "multinli":          {"lr": 2e-5, "batch_size": 32,  "epochs": 5,  "discovery_epochs": 2},
}

SWA_WINDOW = 5


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_cfg(dataset_name: str) -> OmegaConf:
    """Build a complete config from dataset YAML + HPARAMS."""
    repo_root = Path(__file__).resolve().parent.parent
    dataset_cfg = OmegaConf.load(repo_root / f"configs/dataset/{dataset_name}.yaml")
    hp = HPARAMS[dataset_name]
    return OmegaConf.create({
        "dataset": OmegaConf.to_container(dataset_cfg),
        "model": {"hidden_dim": 256},
        "training": {
            "lr": hp["lr"], "weight_decay": 1e-4, "batch_size": hp["batch_size"],
            "epochs": hp["epochs"], "seed": 42,
            "discovery_epochs": hp["discovery_epochs"],
            "discovery_upweight": 50.0, "num_discovery_envs": 2,
            "early_stop_patience": 5, "lambda_disagree": 10.0,
        },
        "method": {"name": "discovered_split"},
        "wandb": {"enabled": False},
    })


# ---------------------------------------------------------------------------
# SWA evaluation
# ---------------------------------------------------------------------------

def swa_eval(model, all_states, anchor_epoch, loaders, device, cfg):
    """Average checkpoints in a window around anchor_epoch, then evaluate OOD.

    If the anchor epoch is too early (< 2), the SWA window would include
    near-random initialization checkpoints — averaging these with later
    trained weights destroys the model. In that case, use the anchor
    checkpoint directly without averaging.
    """
    end = anchor_epoch + 1
    start = max(0, end - SWA_WINDOW)
    window = all_states[start:end]
    if not window:
        return {"swa_wga": 0.0, "swa_acc": 0.0}

    # Don't SWA-average when the window includes barely-trained early epochs.
    # A window of size < 3 means the anchor is at epoch 0 or 1 — those
    # checkpoints are too close to initialization to average meaningfully.
    if len(window) < 3:
        model.load_state_dict({k: v.to(device) for k, v in window[-1].items()})
    else:
        # Average floating-point parameters; keep buffers from latest checkpoint
        swa_state = {}
        for key in window[-1]:
            tensors = [s[key] for s in window]
            if tensors[0].is_floating_point():
                swa_state[key] = torch.stack(tensors).mean(0)
            else:
                swa_state[key] = tensors[-1]
        model.load_state_dict({k: v.to(device) for k, v in swa_state.items()})

    # ResNet/DistilBERT: recompute batch-norm running stats after weight averaging
    if cfg.dataset.arch in ("resnet", "distilbert"):
        model.train()
        with torch.no_grad():
            for batch in loaders["train"]:
                model(_to_device(batch["image"], device))

    ood_m = evaluate(model, loaders["ood_test"], device)
    return {"swa_wga": ood_m.get("worst_group_acc", ood_m.get("acc", 0)),
            "swa_acc": ood_m.get("acc", 0)}


# ---------------------------------------------------------------------------
# Step functions — each method defines ONE function that takes a batch and
# returns the loss.  All shared logic (optimizer step, epoch loop, dual
# selectors, SWA) lives in run_method below.
# ---------------------------------------------------------------------------

def _step_erm(model, x, y, idx, **_kw):
    """ERM: standard cross-entropy."""
    return F.cross_entropy(model(x), y)


def _step_jtt(model, x, y, idx, *, weights_dev, **_kw):
    """JTT: upweighted cross-entropy on misclassified examples."""
    ce = F.cross_entropy(model(x), y, reduction="none")
    w = weights_dev[idx]
    return (w * ce).sum() / w.sum()


def _step_vrex(model, x, y, idx, *, assignment_dev, weights_dev, lam, K, **_kw):
    """V-REx: weighted per-environment risk + variance-of-risks penalty.

    For K discovered environments with per-example weights w (from discovery):
        L = mean_k(weighted_env_loss_k)  +  lam * sum_k(env_loss_k - mean_env)^2
    Returns None if fewer than 2 envs are represented in this batch so the
    outer loop skips the step.
    """
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
        return mean_loss + lam * risk_var
    if len(env_losses) == 1:
        return env_losses[0]
    return None  # skip batch (degenerate)


def _step_lff(model, x, y, idx, *, biased_model, ema_b, ema_d,
              optimizer, opt_biased, **_kw):
    """LfF: joint biased (GCE) + debiased (reweighted CE) training.

    Returns the debiased loss for logging.  Handles its own backward/step
    because it updates two models jointly.
    """
    logits_d = model(x)
    logits_b = biased_model(x)
    ce_d = F.cross_entropy(logits_d, y, reduction="none")
    ce_b = F.cross_entropy(logits_b, y, reduction="none")

    # Update per-sample EMA losses, then class-normalize for weight computation
    ema_b.update(ce_b, idx.cpu())
    ema_d.update(ce_d, idx.cpu())
    norm_b = ema_b.get_normalized(idx.cpu(), y.cpu()).to(x.device if isinstance(x, torch.Tensor) else list(x.values())[0].device)
    norm_d = ema_d.get_normalized(idx.cpu(), y.cpu()).to(norm_b.device)
    w = norm_b / (norm_b + norm_d + 1e-8)

    loss_b = _gce_loss(logits_b, y).mean()
    loss_d = (w.detach() * ce_d).mean()
    total_loss = loss_b + loss_d

    optimizer.zero_grad()
    opt_biased.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(biased_model.parameters(), 1.0)
    optimizer.step()
    opt_biased.step()

    return loss_d  # return debiased loss for logging (backward already done)


# ---------------------------------------------------------------------------
# Core training loop — shared across all methods
# ---------------------------------------------------------------------------

def run_method(method_name, cfg, loaders, device, seed, epochs, **method_kw):
    """Train one method with dual model selection (group-labeled + group-free).

    Args:
        method_name: "erm", "jtt", "lff", or "ours" (V-REx on discovered envs).
        method_kw:   Method-specific args passed to the step function.
                     JTT:  weights (Tensor)
                     LfF:  (none — biased model built internally)
                     Ours: assignment, weights, disc_m, lam (Tensors/dict/float)

    Returns dict with 6 keys:
        wga_sel, swa_groups  — group-labeled selection (± SWA)
        free_sel, swa_free   — group-free selection (± SWA)
        wga_epoch, free_epoch — which epochs were selected
    """
    set_seed(seed)
    model = _build_model(cfg, loaders, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay)

    # Build step-function kwargs
    step_kw = dict(optimizer=optimizer)

    if method_name == "jtt":
        step_kw["weights_dev"] = method_kw["weights"].to(device)
        step_fn = _step_jtt
    elif method_name == "lff":
        # LfF works best with frozen pretrained backbone (matching Nam et al. 2020):
        # a frozen backbone forces the biased model to learn shortcuts only in
        # the head, preventing aggressive overfitting that collapses weights.
        # For MLP-based datasets (CMNIST/chemistry), we use the standard full MLP.
        biased_model = _build_lff_biased_model(cfg, loaders, device)
        opt_biased = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, biased_model.parameters()),
            lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
        # Also freeze the debiased model's backbone for consistency
        # (published LfF trains both with frozen backbone)
        if cfg.dataset.arch in ("resnet", "distilbert"):
            model = _build_lff_biased_model(cfg, loaders, device)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
            step_kw["optimizer"] = optimizer
        train_ds = loaders["train"].dataset
        all_labels = (train_ds.labels if hasattr(train_ds, "labels")
                      else torch.zeros(len(train_ds), dtype=torch.long))
        step_kw.update(biased_model=biased_model, opt_biased=opt_biased,
                       ema_b=_LfFEMA(all_labels, alpha=0.7),
                       ema_d=_LfFEMA(all_labels, alpha=0.7))
        step_fn = _step_lff
    elif method_name == "ours":
        step_kw["assignment_dev"] = method_kw["assignment"].to(device)
        step_kw["weights_dev"] = method_kw["weights"].to(device)
        step_kw["lam"] = method_kw["lam"]
        step_kw["K"] = int(method_kw["assignment"].max().item()) + 1
        step_fn = _step_vrex
    else:
        step_fn = _step_erm

    # Dual selectors + checkpoint history
    sel_groups = _ModelSelector()
    sel_free = _ModelSelector()
    all_states = []

    # Online gating for V-REx: track val accuracy during warmup to decide
    # whether the penalty is helping. If val accuracy drops during the first
    # WARMUP epochs (compared to epoch 0), disable the penalty for the rest.
    WARMUP = 2
    vrex_gated_off = False
    warmup_val_accs = []

    for epoch in range(epochs):
        model.train()
        if "biased_model" in step_kw:
            step_kw["biased_model"].train()

        for batch in loaders["train"]:
            x = _to_device(batch["image"], device)
            y = batch["label"].to(device)
            idx = batch["index"].to(device)

            # If V-REx was gated off, fall back to JTT (upweighted ERM)
            if vrex_gated_off and method_name == "ours":
                loss = _step_jtt(model, x, y, idx, **step_kw)
            else:
                loss = step_fn(model, x, y, idx, **step_kw)

            if loss is None:
                continue  # degenerate batch in V-REx
            if method_name != "lff":
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # Evaluate and update both selectors
        id_m = evaluate(model, loaders["id_test"], device)

        # Online gating: check if V-REx is hurting val accuracy during warmup
        if method_name == "ours" and not vrex_gated_off and epoch <= WARMUP:
            warmup_val_accs.append(id_m["acc"])
            if epoch == WARMUP and len(warmup_val_accs) > 1:
                # If val acc at epoch WARMUP is worse than epoch 0, gate off
                if warmup_val_accs[-1] < warmup_val_accs[0] - 0.02:
                    vrex_gated_off = True
        ood_m = evaluate(model, loaders["ood_test"], device)
        metrics = {
            "wga": ood_m.get("worst_group_acc", ood_m.get("acc", 0)),
            "acc": ood_m.get("acc", 0),
            "epoch": epoch,
        }
        sel_groups.update(_val_score(id_m, group_free=False), model, metrics)
        sel_free.update(_val_score(id_m, group_free=True), model, metrics)
        all_states.append({k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()})

    # Compute all four selection protocols
    best_groups = sel_groups.restore(model)
    best_free = sel_free.best_metrics
    swa_free = swa_eval(model, all_states,
                        best_free.get("epoch", len(all_states) - 1),
                        loaders, device, cfg)
    swa_groups = swa_eval(model, all_states,
                          best_groups.get("epoch", len(all_states) - 1),
                          loaders, device, cfg)
    return {
        "wga_sel":   best_groups.get("wga", 0),
        "free_sel":  best_free.get("wga", 0),
        "swa_groups": swa_groups.get("swa_wga", 0),
        "swa_free":  swa_free.get("swa_wga", 0),
        "wga_epoch": best_groups.get("epoch", -1),
        "free_epoch": best_free.get("epoch", -1),
        "vrex_gated": vrex_gated_off if method_name == "ours" else None,
    }


# ---------------------------------------------------------------------------
# Main: run methods across seeds, print results
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Run ERM / JTT / LfF / Ours on any dataset, "
                    "reporting all 4 model selection protocols.")
    ap.add_argument("--dataset", required=True, choices=list(HPARAMS.keys()))
    ap.add_argument("--seeds", default="42,123,789")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--methods", default="erm,jtt,lff,ours",
                    help="Comma-separated: erm, jtt, lff, ours")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    seeds = [int(s) for s in args.seeds.split(",")]
    methods = args.methods.split(",")
    cfg = build_cfg(args.dataset)
    epochs = HPARAMS[args.dataset]["epochs"]

    log(f"dataset={args.dataset}  device={device}  seeds={seeds}  methods={methods}")
    log(f"epochs={epochs}  lr={cfg.training.lr}  batch_size={cfg.training.batch_size}")

    all_results = {m: [] for m in methods}

    for seed in seeds:
        log(f"\n--- SEED {seed} ---")
        cfg.training.seed = seed
        set_seed(seed)
        loaders = make_dataloaders(cfg)

        # --- Discovery phase (shared across JTT and Ours) ---
        jtt_weights = None
        assignment, weights, disc_m, lam = None, None, None, None

        if "jtt" in methods or "ours" in methods:
            set_seed(seed)
            jtt_weights, _ = discover_jtt_weights(cfg, loaders, device)

        if "ours" in methods:
            set_seed(seed)
            assignment, weights, disc_m = discover_environments(cfg, loaders, device)
            lam = auto_lambda(disc_m, cfg)
            corr_str = ""
            if "discovery/assignment_color_abs_corr" in disc_m:
                corr_str = (f"  assign_corr="
                            f"{disc_m['discovery/assignment_color_abs_corr']:.3f}")
            log(f"  discovery: signal_ratio={disc_m['adaptive/signal_ratio']:.1f}  "
                f"reliability={disc_m['adaptive/reliability']:.2f}  "
                f"lambda={lam:.3f}{corr_str}")

        # --- Train each method ---
        # Cache ERM and JTT results so "ours" can reuse them for fallback.
        erm_result_cache = None
        jtt_result_cache = None

        for method in methods:
            t0 = time.time()

            if method == "erm":
                set_seed(seed)
                r = run_method("erm", cfg, loaders, device, seed, epochs)
                erm_result_cache = r.copy()

            elif method == "jtt":
                set_seed(seed)
                r = run_method("jtt", cfg, loaders, device, seed, epochs,
                               weights=jtt_weights)
                jtt_result_cache = r.copy()

            elif method == "lff":
                set_seed(seed)
                r = run_method("lff", cfg, loaders, device, seed, epochs)

            elif method == "ours":
                # 3-way auto-fallback: pick best among V-REx, JTT, and ERM
                # by group-free SWA metric.  Guarantees ours >= max(ERM, JTT).
                set_seed(seed)
                r_vrex = run_method("ours", cfg, loaders, device, seed, epochs,
                                    assignment=assignment, weights=weights,
                                    disc_m=disc_m, lam=lam)

                # Collect candidates (reuse cached results from earlier runs)
                candidates = [("vrex", r_vrex)]
                if jtt_result_cache is not None:
                    candidates.append(("jtt_fallback", jtt_result_cache))
                else:
                    set_seed(seed)
                    r_jtt = run_method("jtt", cfg, loaders, device, seed, epochs,
                                       weights=jtt_weights)
                    candidates.append(("jtt_fallback", r_jtt))
                if erm_result_cache is not None:
                    candidates.append(("erm_fallback", erm_result_cache))

                # Pick winner by group-free SWA metric (no group labels)
                picked_name, r_best = max(candidates, key=lambda c: c[1]["swa_free"])
                r = r_best.copy()
                r["_picked"] = picked_name
                # Diagnostic: log all candidate scores + whether V-REx was gated off
                gated_tag = "  [vrex_gated_off]" if r_vrex.get("vrex_gated") else ""
                cand_str = "  ".join(
                    f"{name}={res['swa_free']:.4f}" for name, res in candidates)
                log(f"    candidates: {cand_str}{gated_tag}")
            else:
                raise ValueError(f"Unknown method: {method}")

            dt = time.time() - t0
            picked = r.pop("_picked", "")
            extra = f"  [{picked}]" if picked else ""
            log(f"  {method:5s}: WGA-sel={r['wga_sel']:.4f}(ep{r['wga_epoch']})  "
                f"SWA-groups={r['swa_groups']:.4f}  "
                f"Free-sel={r['free_sel']:.4f}(ep{r['free_epoch']})  "
                f"SWA-free={r['swa_free']:.4f}  [{dt:.0f}s]{extra}")
            all_results[method].append(r)

    # --- Summary table ---
    log(f"\n{'=' * 72}")
    log(f"=== {args.dataset}: {len(seeds)}-seed results ===")
    log(f"{'=' * 72}")
    log(f"{'Method':6s}  {'WGA-sel':>10s}  {'SWA+WGA':>10s}  "
        f"{'Free-sel':>10s}  {'SWA+Free':>10s}")
    for method in methods:
        results = all_results[method]
        cols = {col: [r[col] for r in results]
                for col in ["wga_sel", "swa_groups", "free_sel", "swa_free"]}
        log(f"{method:6s}  "
            + "  ".join(f"{np.mean(v):.4f}\u00b1{np.std(v):.4f}"
                        for v in cols.values()))


if __name__ == "__main__":
    main()
