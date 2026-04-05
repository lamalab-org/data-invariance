"""Resampling stability on the ID test set, where ERM works well.

On OOD, ERM is consistently bad (0.6% fragile). On ID, ERM works well
but SOME examples are at the decision boundary and may flip under
resampling — these are the genuinely composition-sensitive ones.

Also runs resampling for our method to compare flip rates.

Usage:
    uv run python scripts/resampling_stability_id.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
import wandb
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from evaluate import compute_stability_scores
from models import MLP
from train import discover_environments, make_dataloaders, train_discovered_split, train_erm
from utils import set_seed


def train_on_subsample(cfg, train_ds, device, seed, subsample_frac=0.9):
    """Train ERM on a random subsample, return the model."""
    rng = np.random.RandomState(seed)
    n = len(train_ds)
    idx = rng.choice(n, int(n * subsample_frac), replace=False)
    subset = Subset(train_ds, idx)
    loader = DataLoader(subset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0)

    set_seed(seed)
    model = MLP(input_dim=train_ds.input_dim, hidden_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    for epoch in range(cfg.training.epochs):
        model.train()
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def get_predictions(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            preds.append(model(x).argmax(1).cpu())
    return torch.cat(preds)


def get_labels(loader):
    labs = []
    for batch in loader:
        y = batch["label"]
        labs.append(torch.tensor(y) if not isinstance(y, torch.Tensor) else y)
    return torch.cat(labs)


def main():
    cfg = OmegaConf.create({
        "dataset": {
            "name": "cmnist", "arch": "mlp",
            "train_correlation": 0.9, "test_correlation": 0.1,
            "label_noise": 0.25, "data_dir": "./data",
        },
        "model": {"hidden_dim": 256, "separate_backbones": False, "num_heads": 2},
        "training": {
            "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 256,
            "epochs": 10, "seed": 42,
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

    device = torch.device("cpu")
    set_seed(42)
    loaders = make_dataloaders(cfg)
    id_test = loaders["id_test"]
    train_ds = loaders["train"].dataset

    labels = get_labels(id_test)
    n_test = len(labels)
    n_resamples = 10

    # =====================================================================
    # Step 1: Ground truth — resample 10 ERMs, measure ID flip rates
    # =====================================================================
    print(f"Training {n_resamples} ERMs on 90% subsamples (ID test)...")
    erm_preds = []
    for i in range(n_resamples):
        model = train_on_subsample(cfg, train_ds, device, seed=42 + i * 100)
        preds = get_predictions(model, id_test, device)
        acc = (preds == labels).float().mean()
        print(f"  Resample {i+1}/{n_resamples}: ID acc = {acc:.3f}")
        erm_preds.append(preds)

    erm_pred_matrix = torch.stack(erm_preds)  # (10, N_test)
    erm_majority = erm_pred_matrix.mode(dim=0).values
    erm_flip_rate = (erm_pred_matrix != erm_majority.unsqueeze(0)).float().mean(dim=0)

    print(f"\nERM ID flip rates:")
    print(f"  Mean: {erm_flip_rate.mean():.4f}")
    print(f"  >0 (any flip): {(erm_flip_rate > 0).float().mean():.1%}")
    print(f"  >0.1: {(erm_flip_rate > 0.1).float().mean():.1%}")
    print(f"  >0.3: {(erm_flip_rate > 0.3).float().mean():.1%}")

    # =====================================================================
    # Step 2: Same for our method — resample 10 times
    # =====================================================================
    print(f"\nTraining {n_resamples} discovered_split models on 90% subsamples...")
    ours_preds = []
    for i in range(n_resamples):
        rng = np.random.RandomState(42 + i * 100)
        n = len(train_ds)
        idx = rng.choice(n, int(n * 0.9), replace=False)
        subset = Subset(train_ds, idx)
        sub_loader = DataLoader(subset, batch_size=256, shuffle=True, num_workers=0)
        sub_loaders = {"train": sub_loader, "id_test": id_test, "ood_test": loaders["ood_test"]}

        set_seed(42 + i * 100)
        assignment, weights, disc_m = discover_environments(cfg, sub_loaders, device)
        set_seed(42 + i * 100)
        model = MLP(input_dim=train_ds.input_dim, hidden_dim=256).to(device)
        run = wandb.init(mode="disabled")
        train_discovered_split(cfg, model, sub_loaders, device, run, assignment, weights, disc_m)
        run.finish()

        preds = get_predictions(model, id_test, device)
        acc = (preds == labels).float().mean()
        print(f"  Resample {i+1}/{n_resamples}: ID acc = {acc:.3f}")
        ours_preds.append(preds)

    ours_pred_matrix = torch.stack(ours_preds)
    ours_majority = ours_pred_matrix.mode(dim=0).values
    ours_flip_rate = (ours_pred_matrix != ours_majority.unsqueeze(0)).float().mean(dim=0)

    print(f"\nOurs ID flip rates:")
    print(f"  Mean: {ours_flip_rate.mean():.4f}")
    print(f"  >0 (any flip): {(ours_flip_rate > 0).float().mean():.1%}")
    print(f"  >0.1: {(ours_flip_rate > 0.1).float().mean():.1%}")
    print(f"  >0.3: {(ours_flip_rate > 0.3).float().mean():.1%}")

    # =====================================================================
    # Step 3: Compare flip rates
    # =====================================================================
    print(f"\n{'='*60}")
    print("COMPARISON: ERM vs Ours (ID test)")
    print(f"ERM  mean flip rate: {erm_flip_rate.mean():.4f}")
    print(f"Ours mean flip rate: {ours_flip_rate.mean():.4f}")
    print(f"Ratio: {erm_flip_rate.mean() / max(ours_flip_rate.mean(), 1e-6):.1f}x more stable")

    # =====================================================================
    # Step 4: Do single-model stability scores predict resampling fragility?
    # =====================================================================
    print(f"\n{'='*60}")
    print("Computing stability scores from single models...")

    # Train single ERM and single Ours
    set_seed(42)
    erm_single = MLP(input_dim=train_ds.input_dim, hidden_dim=256).to(device)
    run = wandb.init(mode="disabled")
    train_erm(cfg, erm_single, loaders, device, run)
    run.finish()

    set_seed(42)
    assignment, weights, disc_m = discover_environments(cfg, loaders, device)
    set_seed(42)
    ours_single = MLP(input_dim=train_ds.input_dim, hidden_dim=256).to(device)
    run = wandb.init(mode="disabled")
    train_discovered_split(cfg, ours_single, loaders, device, run, assignment, weights, disc_m)
    run.finish()

    scores_erm = compute_stability_scores(erm_single, id_test, device)
    scores_ours = compute_stability_scores(ours_single, id_test, device)

    # Predict ERM's resampling fragility from single-model scores
    import torchmetrics

    fragile_threshold = 0.1  # examples that flip in >10% of resamples
    erm_fragile = (erm_flip_rate > fragile_threshold).long()
    n_fragile = erm_fragile.sum().item()

    if n_fragile > 0 and n_fragile < n_test:
        print(f"\nFragile examples (ERM flip rate > {fragile_threshold}): {n_fragile}/{n_test} ({n_fragile/n_test:.1%})")

        auroc = torchmetrics.AUROC(task="binary")
        print(f"\n--- AUROC: predicting ERM resampling fragility ---")
        print(f"{'Score':<25} {'Our model':>10} {'ERM':>10} {'Δ':>10}")
        print("-" * 57)

        for name in ["entropy", "loss"]:
            auroc.reset()
            a_ours = auroc(scores_ours[name], erm_fragile).item()
            auroc.reset()
            a_erm = auroc(scores_erm[name], erm_fragile).item()
            print(f"{name:<25} {a_ours:>10.3f} {a_erm:>10.3f} {a_ours - a_erm:>+10.3f}")

        auroc.reset()
        a_ours = auroc(1 - scores_ours["confidence"], erm_fragile).item()
        auroc.reset()
        a_erm = auroc(1 - scores_erm["confidence"], erm_fragile).item()
        print(f"{'1 - confidence':<25} {a_ours:>10.3f} {a_erm:>10.3f} {a_ours - a_erm:>+10.3f}")
    else:
        print(f"\nNot enough fragile examples for AUROC (n_fragile={n_fragile})")

    # Predict OUR model's resampling fragility
    ours_fragile = (ours_flip_rate > fragile_threshold).long()
    n_fragile_ours = ours_fragile.sum().item()

    if n_fragile_ours > 0 and n_fragile_ours < n_test:
        print(f"\nFragile examples (Ours flip rate > {fragile_threshold}): {n_fragile_ours}/{n_test}")

        print(f"\n--- AUROC: predicting OUR model's resampling fragility ---")
        print(f"{'Score':<25} {'Our model':>10} {'ERM':>10}")
        print("-" * 47)

        for name in ["entropy", "loss"]:
            auroc.reset()
            a_ours = auroc(scores_ours[name], ours_fragile).item()
            auroc.reset()
            a_erm = auroc(scores_erm[name], ours_fragile).item()
            print(f"{name:<25} {a_ours:>10.3f} {a_erm:>10.3f}")


if __name__ == "__main__":
    main()
