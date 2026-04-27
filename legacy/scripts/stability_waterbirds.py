"""Validate stability scores on Waterbirds using multi-seed ERM predictions.

We have 5 ERM seeds → 5 different models → 5 predictions per test example.
Some examples get consistent predictions (stable), others flip (fragile).

The ground truth: per-example prediction variance across the 5 ERMs.
The test: does our single-model stability score predict this variance?

Usage:
    uv run python scripts/stability_waterbirds.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
import wandb
import numpy as np
from omegaconf import OmegaConf

from evaluate import compute_stability_scores
from models import MLP, make_resnet_backbone
from train import discover_environments, make_dataloaders, train_discovered_split, train_erm
from utils import set_seed


def train_erm_waterbirds(cfg, loaders, device, seed):
    """Train ERM on Waterbirds with a specific seed, return test predictions + confidences."""
    set_seed(seed)
    backbone, out_dim = make_resnet_backbone()
    model = MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    # Train for 15 epochs, val-based selection
    from train import _ModelSelector, _val_score, evaluate
    selector = _ModelSelector()
    for epoch in range(cfg.training.epochs):
        model.train()
        for batch in loaders["train"]:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        id_metrics = evaluate(model, loaders["id_test"], device)
        selector.update(_val_score(id_metrics), model, {})

    selector.restore(model)

    # Get test predictions
    model.eval()
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch in loaders["ood_test"]:
            x = batch["image"].to(device)
            logits = model(x)
            probs = logits.softmax(1)
            all_preds.append(probs.argmax(1).cpu())
            all_probs.append(probs[:, 1].cpu())  # P(class=1)

    return torch.cat(all_preds), torch.cat(all_probs), model


def main():
    cfg = OmegaConf.create({
        "dataset": {"name": "waterbirds", "arch": "resnet", "data_dir": "./data/waterbirds"},
        "model": {"hidden_dim": 256, "separate_backbones": False, "num_heads": 2},
        "training": {
            "lr": 1e-4, "weight_decay": 1e-4, "batch_size": 64,
            "epochs": 15, "seed": 42,
            "lambda_disagree": 10.0, "adv_lr": 1e-2,
            "discovery_epochs": 5, "discovery_criterion": "loss",
            "discovery_quantile": 0.5, "discovery_upweight": 50.0,
            "discovery_reweight": 0.0, "discovery_rounds": 1,
            "lambda_anneal_factor": 1.0, "early_stop_patience": 5,
            "num_discovery_envs": 2, "freeze_backbone": False,
            "balanced_sampling": False, "env_mixup": 0.0,
            "training_noise": 0.0,
            "adv_init": "zeros", "adv_init_scale": 1.0,
            "head_noise": 0.0, "adv_warmup_epochs": 0,
            "adv_steps_per_model_step": 1, "lambda_warmup_epochs": 0,
            "adv_entropy_bonus": 0.0, "lambda_threshold": 0.0,
            "lambda_ramp_range": 0.0, "adv_mode": "task_loss",
        },
        "method": {"name": "discovered_split"},
        "wandb": {"enabled": False},
    })

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(42)
    loaders = make_dataloaders(cfg)

    # Get test labels
    test_labels = []
    test_spurious = []
    for batch in loaders["ood_test"]:
        test_labels.append(torch.tensor(batch["label"]) if not isinstance(batch["label"], torch.Tensor) else batch["label"])
        test_spurious.append(torch.tensor(batch["spurious"]) if not isinstance(batch["spurious"], torch.Tensor) else batch["spurious"])
    test_labels = torch.cat(test_labels)
    test_spurious = torch.cat(test_spurious)
    n_test = len(test_labels)

    # =====================================================================
    # Step 1: Train 5 ERMs, collect per-example predictions
    # =====================================================================
    seeds = [42, 123, 456, 789, 1337]
    all_preds = []
    all_probs = []

    print(f"Training {len(seeds)} ERMs on Waterbirds...")
    for seed in seeds:
        print(f"  Seed {seed}...", end=" ", flush=True)
        preds, probs, _ = train_erm_waterbirds(cfg, loaders, device, seed)
        acc = (preds == test_labels).float().mean()
        print(f"test acc = {acc:.3f}")
        all_preds.append(preds)
        all_probs.append(probs)

    pred_matrix = torch.stack(all_preds)   # (5, N_test)
    prob_matrix = torch.stack(all_probs)   # (5, N_test)

    # Ground truth: per-example prediction variance
    pred_variance = prob_matrix.var(dim=0)   # (N_test,) — variance of P(class=1)
    majority = pred_matrix.mode(dim=0).values
    flip_rate = (pred_matrix != majority.unsqueeze(0)).float().mean(dim=0)

    print(f"\nGround truth composition sensitivity:")
    print(f"  Mean prediction variance: {pred_variance.mean():.4f}")
    print(f"  Examples with any flip: {(flip_rate > 0).float().mean():.1%}")
    print(f"  Examples with flip_rate > 0.3: {(flip_rate > 0.3).float().mean():.1%}")

    # =====================================================================
    # Step 2: Train our method ONCE, compute stability scores
    # =====================================================================
    print(f"\nTraining our method (discovered split + V-REx)...")
    set_seed(42)
    assignment, weights, disc_metrics = discover_environments(cfg, loaders, device)
    set_seed(42)
    backbone, out_dim = make_resnet_backbone()
    our_model = MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
    run = wandb.init(mode="disabled")
    train_discovered_split(cfg, our_model, loaders, device, run, assignment, weights, disc_metrics)
    run.finish()

    # Also train single ERM for comparison
    print("Training single ERM for comparison...")
    _, _, erm_model = train_erm_waterbirds(cfg, loaders, device, seed=42)

    print("\nComputing stability scores...")
    scores_ours = compute_stability_scores(our_model, loaders["ood_test"], device)
    scores_erm = compute_stability_scores(erm_model, loaders["ood_test"], device)

    # =====================================================================
    # Step 3: Correlation between stability scores and ground truth
    # =====================================================================
    import torchmetrics
    from scipy.stats import spearmanr

    print(f"\n{'='*60}")
    print("STABILITY SCORE VALIDATION — Waterbirds")
    print(f"{'='*60}")

    # Spearman correlation with continuous prediction variance
    print(f"\n--- Spearman ρ with prediction variance (continuous) ---")
    print(f"{'Score':<25} {'Our model':>10} {'ERM':>10}")
    print("-" * 47)
    for name in ["entropy", "loss"]:
        r_ours, _ = spearmanr(scores_ours[name].numpy(), pred_variance.numpy())
        r_erm, _ = spearmanr(scores_erm[name].numpy(), pred_variance.numpy())
        print(f"{name:<25} {r_ours:>10.3f} {r_erm:>10.3f}")
    r_ours, _ = spearmanr((1 - scores_ours["confidence"]).numpy(), pred_variance.numpy())
    r_erm, _ = spearmanr((1 - scores_erm["confidence"]).numpy(), pred_variance.numpy())
    print(f"{'1 - confidence':<25} {r_ours:>10.3f} {r_erm:>10.3f}")

    # AUROC for predicting "fragile" examples (flip_rate > 0)
    fragile = (flip_rate > 0).long()
    n_fragile = fragile.sum().item()
    print(f"\nFragile examples (any prediction flip across 5 seeds): {n_fragile}/{n_test} ({n_fragile/n_test:.1%})")

    if n_fragile > 0 and n_fragile < n_test:
        auroc = torchmetrics.AUROC(task="binary")
        print(f"\n--- AUROC for predicting fragile examples ---")
        print(f"{'Score':<25} {'Our model':>10} {'ERM':>10} {'Δ':>10}")
        print("-" * 57)
        for name in ["entropy", "loss"]:
            auroc.reset()
            a_ours = auroc(scores_ours[name], fragile).item()
            auroc.reset()
            a_erm = auroc(scores_erm[name], fragile).item()
            print(f"{name:<25} {a_ours:>10.3f} {a_erm:>10.3f} {a_ours - a_erm:>+10.3f}")

    # Per-group analysis
    print(f"\n--- Per-group flip rates ---")
    groups = test_labels * 2 + test_spurious
    for g in range(4):
        mask = groups == g
        if mask.any():
            bird = "landbird" if g < 2 else "waterbird"
            bg = "land" if g % 2 == 0 else "water"
            mean_flip = flip_rate[mask].mean()
            mean_var = pred_variance[mask].mean()
            print(f"  {bird:>10}+{bg:<5}: n={mask.sum():>5}, mean_flip={mean_flip:.3f}, mean_var={mean_var:.4f}")

    # Calibration: bin by our score, measure actual variance per bin
    print(f"\n--- Calibration: our entropy quintile → mean prediction variance ---")
    ours_entropy = scores_ours["entropy"]
    n_bins = 5
    quantiles = torch.linspace(0, 1, n_bins + 1)
    bin_edges = torch.quantile(ours_entropy, quantiles)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (ours_entropy >= lo) & (ours_entropy < hi) if i < n_bins - 1 else (ours_entropy >= lo)
        if mask.sum() > 0:
            mv = pred_variance[mask].mean()
            mf = flip_rate[mask].mean()
            print(f"  Bin {i} (n={mask.sum():>5}): mean_variance={mv:.4f}, mean_flip_rate={mf:.3f}")


if __name__ == "__main__":
    main()
