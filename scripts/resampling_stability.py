"""The make-or-break experiment: do stability scores predict retraining fragility?

Train 10 ERMs on different 90% subsamples of CMNIST. For each test example,
measure how often the prediction flips across the 10 models. This is the
GROUND TRUTH for data-composition sensitivity.

Then train our method ONCE and compute stability scores. Measure whether
our scores predict the ground-truth flip rates.

If yes → the paper works. We've shown that training for stability produces
a model that *knows which predictions are fragile to dataset composition*.

If no → back to the drawing board.

Usage:
    uv run python scripts/resampling_stability.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from data import ColoredMNIST
from evaluate import compute_stability_scores
from models import MLP
from train import discover_environments, make_dataloaders, train_discovered_split, train_erm
from utils import set_seed

import numpy as np


def train_on_subsample(cfg, train_ds, test_loader, device, seed, subsample_frac=0.9):
    """Train ERM on a random subsample, return test predictions."""
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

    # Get test predictions
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(device)
            preds = model(x).argmax(dim=1)
            all_preds.append(preds.cpu())

    return torch.cat(all_preds)


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
    test_loader = loaders["ood_test"]  # OOD test to see composition-sensitivity
    train_ds = loaders["train"].dataset
    input_dim = train_ds.input_dim

    n_resamples = 10

    # =====================================================================
    # Step 1: Ground truth — train 10 ERMs on different 90% subsamples
    # =====================================================================
    print(f"Training {n_resamples} ERM models on 90% subsamples...")
    all_predictions = []
    for i in range(n_resamples):
        print(f"  Resample {i+1}/{n_resamples}...", end=" ", flush=True)
        preds = train_on_subsample(cfg, train_ds, test_loader, device, seed=42 + i * 100)
        all_predictions.append(preds)
        acc = (preds == torch.tensor([b["label"] for b in test_loader.dataset])).float().mean()
        print(f"OOD acc = {acc:.3f}")

    # Stack predictions: (n_resamples, n_test)
    pred_matrix = torch.stack(all_predictions)
    n_test = pred_matrix.shape[1]

    # Get labels
    labels = torch.tensor([test_loader.dataset[i]["label"] for i in range(n_test)])

    # Per-example flip rate: fraction of resamples where prediction differs from majority
    majority_vote = pred_matrix.mode(dim=0).values
    flip_rate = (pred_matrix != majority_vote.unsqueeze(0)).float().mean(dim=0)

    print(f"\nGround truth flip rates:")
    print(f"  Mean: {flip_rate.mean():.3f}")
    print(f"  >0 (any flip): {(flip_rate > 0).float().mean():.1%}")
    print(f"  >0.3 (fragile): {(flip_rate > 0.3).float().mean():.1%}")
    print(f"  >0.5 (very fragile): {(flip_rate > 0.5).float().mean():.1%}")

    # =====================================================================
    # Step 2: Train our method ONCE and compute stability scores
    # =====================================================================
    print("\n" + "=" * 60)
    print("Training our method (discovered split + V-REx)...")
    set_seed(42)
    assignment, weights, disc_metrics = discover_environments(cfg, loaders, device)
    set_seed(42)
    our_model = MLP(input_dim=input_dim, hidden_dim=256).to(device)
    run = wandb.init(mode="disabled")
    train_discovered_split(cfg, our_model, loaders, device, run, assignment, weights, disc_metrics)
    run.finish()

    # Also train a single ERM for comparison
    print("Training ERM for comparison...")
    set_seed(42)
    erm_model = MLP(input_dim=input_dim, hidden_dim=256).to(device)
    run = wandb.init(mode="disabled")
    train_erm(cfg, erm_model, loaders, device, run)
    run.finish()

    # =====================================================================
    # Step 3: Compute stability scores and compare to ground truth
    # =====================================================================
    print("\nComputing stability scores...")
    scores_ours = compute_stability_scores(our_model, test_loader, device)
    scores_erm = compute_stability_scores(erm_model, test_loader, device)

    # =====================================================================
    # Step 4: Measure correlation between scores and ground-truth flip rates
    # =====================================================================
    import torchmetrics

    # Binary target: is this example fragile (flip_rate > 0.3)?
    fragile = (flip_rate > 0.3).long()
    n_fragile = fragile.sum().item()
    print(f"\nFragile examples (flip_rate > 0.3): {n_fragile}/{n_test} ({n_fragile/n_test:.1%})")

    if n_fragile > 0 and n_fragile < n_test:
        auroc = torchmetrics.AUROC(task="binary")

        print("\n--- AUROC for predicting retraining fragility ---")
        print(f"{'Score type':<25} {'Our model':>10} {'ERM':>10} {'Δ':>10}")
        print("-" * 57)

        for score_name in ["entropy", "loss", "mc_dropout_var"]:
            auroc.reset()
            a_ours = auroc(scores_ours[score_name], fragile).item()
            auroc.reset()
            a_erm = auroc(scores_erm[score_name], fragile).item()
            delta = a_ours - a_erm
            print(f"{score_name:<25} {a_ours:>10.3f} {a_erm:>10.3f} {delta:>+10.3f}")

        auroc.reset()
        a_ours = auroc(1.0 - scores_ours["confidence"], fragile).item()
        auroc.reset()
        a_erm = auroc(1.0 - scores_erm["confidence"], fragile).item()
        print(f"{'1 - confidence':<25} {a_ours:>10.3f} {a_erm:>10.3f} {a_ours - a_erm:>+10.3f}")

    # Rank correlation between continuous flip rate and scores
    from scipy.stats import spearmanr
    print("\n--- Spearman correlation with continuous flip rate ---")
    print(f"{'Score type':<25} {'Our model':>10} {'ERM':>10}")
    print("-" * 47)
    for score_name in ["entropy", "loss"]:
        r_ours, _ = spearmanr(scores_ours[score_name].numpy(), flip_rate.numpy())
        r_erm, _ = spearmanr(scores_erm[score_name].numpy(), flip_rate.numpy())
        print(f"{score_name:<25} {r_ours:>10.3f} {r_erm:>10.3f}")

    # Calibration: bin by our score, measure actual mean flip rate per bin
    print("\n--- Calibration: our entropy quintile → mean flip rate ---")
    n_bins = 5
    ours_entropy = scores_ours["entropy"]
    quantiles = torch.linspace(0, 1, n_bins + 1)
    bin_edges = torch.quantile(ours_entropy, quantiles)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (ours_entropy >= lo) & (ours_entropy < hi) if i < n_bins - 1 else (ours_entropy >= lo)
        if mask.sum() > 0:
            mean_flip = flip_rate[mask].mean().item()
            print(f"  Bin {i} (n={mask.sum().item():>5}): mean_flip_rate = {mean_flip:.3f}")

    # Our model's flip rate vs ERM's
    ours_preds = scores_ours["predictions"]
    erm_preds = scores_erm["predictions"]
    ours_correct = (ours_preds == labels).float().mean()
    erm_correct = (erm_preds == labels).float().mean()
    print(f"\n--- Summary ---")
    print(f"ERM OOD accuracy:  {erm_correct:.3f}")
    print(f"Ours OOD accuracy: {ours_correct:.3f}")
    print(f"Ground-truth mean flip rate: {flip_rate.mean():.3f}")


if __name__ == "__main__":
    main()
