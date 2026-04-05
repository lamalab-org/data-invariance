"""Evaluate stability scores: does our model's confidence predict OOD flips
better than ERM's confidence?

Usage:
    uv run python scripts/evaluate_stability.py

Trains ERM and our method on CMNIST (corr=0.9), then measures how well
each model's confidence predicts which examples flip between ID (corr=0.9)
and OOD (corr=0.1) test sets.

AUROC > 0.5 means the score has discriminative power.
Our AUROC > ERM AUROC means our model provides better uncertainty estimates.
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import torch
from omegaconf import OmegaConf

from data import ColoredMNIST
from evaluate import compute_stability_scores, evaluate_stability_discrimination
from models import MLP
from train import discover_environments, make_dataloaders, train_discovered_split, train_erm
from utils import set_seed


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
            "epochs": 15, "seed": 42,
            "lambda_disagree": 10.0, "adv_lr": 1e-2,
            "discovery_epochs": 5, "discovery_criterion": "loss",
            "discovery_quantile": 0.5, "discovery_upweight": 50.0,
            "discovery_reweight": 0.0, "discovery_rounds": 1,
            "lambda_anneal_factor": 1.0, "early_stop_patience": 5,
            "num_discovery_envs": 2, "freeze_backbone": False,
            "balanced_sampling": False, "env_mixup": 0.0,
            "training_noise": 0.0,
            # Adversarial split params (not used but needed for config)
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

    input_dim = loaders["train"].dataset.input_dim

    # ---- Train ERM ----
    print("Training ERM...")
    set_seed(42)
    erm_model = MLP(input_dim=input_dim, hidden_dim=256).to(device)

    import wandb
    run = wandb.init(mode="disabled")
    train_erm(cfg, erm_model, loaders, device, run)
    run.finish()

    # ---- Train our method ----
    print("Training discovered split...")
    set_seed(42)
    assignment, weights, disc_metrics = discover_environments(cfg, loaders, device)
    set_seed(42)
    our_model = MLP(input_dim=input_dim, hidden_dim=256).to(device)
    run = wandb.init(mode="disabled")
    train_discovered_split(cfg, our_model, loaders, device, run, assignment, weights, disc_metrics)
    run.finish()

    # ---- Score both models on ID and OOD test sets ----
    print("\nComputing stability scores...")
    scores_erm_id = compute_stability_scores(erm_model, loaders["id_test"], assignment, weights, device)
    scores_erm_ood = compute_stability_scores(erm_model, loaders["ood_test"], assignment, weights, device)
    scores_ours_id = compute_stability_scores(our_model, loaders["id_test"], assignment, weights, device)
    scores_ours_ood = compute_stability_scores(our_model, loaders["ood_test"], assignment, weights, device)

    # ---- Evaluate: does confidence predict flips? ----
    # For ERM: predictions on ID vs OOD
    erm_id_preds = scores_erm_id["predictions"]
    erm_ood_preds = scores_erm_ood["predictions"]

    # For our model: same
    ours_id_preds = scores_ours_id["predictions"]
    ours_ood_preds = scores_ours_ood["predictions"]

    labels = scores_erm_id["labels"]

    print("\n=== ERM flip analysis ===")
    erm_flips = (erm_id_preds != erm_ood_preds).float()
    print(f"ERM flip rate: {erm_flips.mean():.3f} ({erm_flips.sum().long()}/{len(erm_flips)} examples flip)")
    print(f"ERM ID accuracy: {(erm_id_preds == labels).float().mean():.3f}")
    print(f"ERM OOD accuracy: {(erm_ood_preds == labels).float().mean():.3f}")

    print("\n=== Our model flip analysis ===")
    ours_flips = (ours_id_preds != ours_ood_preds).float()
    print(f"Ours flip rate: {ours_flips.mean():.3f} ({ours_flips.sum().long()}/{len(ours_flips)} examples flip)")
    print(f"Ours ID accuracy: {(ours_id_preds == labels).float().mean():.3f}")
    print(f"Ours OOD accuracy: {(ours_ood_preds == labels).float().mean():.3f}")

    # Cross-evaluate: use OUR model's confidence to predict ERM's flips.
    # This is the key test: can our model flag examples that ANY model would
    # find fragile under distribution shift?
    print("\n=== Stability discrimination (AUROC for predicting flips) ===")
    print("Higher AUROC = better at predicting which examples flip under OOD shift")

    # ERM flips predicted by each model's confidence
    result = evaluate_stability_discrimination(
        scores_ours=scores_ours_id,
        scores_erm=scores_erm_id,
        id_predictions=erm_id_preds,
        ood_predictions=erm_ood_preds,
        labels=labels,
    )

    print(f"\nPredicting ERM flips:")
    print(f"  Our stability score AUROC:  {result['auroc_ours_stability']:.3f}")
    print(f"  ERM stability score AUROC:  {result['auroc_erm_stability']:.3f}")
    print(f"  Δ (ours - ERM):             {result['auroc_ours_stability'] - result['auroc_erm_stability']:+.3f}")

    # Also: predict our own flips
    result2 = evaluate_stability_discrimination(
        scores_ours=scores_ours_id,
        scores_erm=scores_erm_id,
        id_predictions=ours_id_preds,
        ood_predictions=ours_ood_preds,
        labels=labels,
    )

    print(f"\nPredicting our model's flips:")
    print(f"  Our stability score AUROC:  {result2['auroc_ours_stability']:.3f}")
    print(f"  ERM stability score AUROC:  {result2['auroc_erm_stability']:.3f}")

    # Summary
    print("\n=== Summary ===")
    print(f"ERM: ID={( erm_id_preds == labels).float().mean():.1%}, OOD={(erm_ood_preds == labels).float().mean():.1%}, flip rate={erm_flips.mean():.1%}")
    print(f"Ours: ID={(ours_id_preds == labels).float().mean():.1%}, OOD={(ours_ood_preds == labels).float().mean():.1%}, flip rate={ours_flips.mean():.1%}")
    print(f"Stability AUROC for predicting ERM flips: Ours={result['auroc_ours_stability']:.3f}, ERM={result['auroc_erm_stability']:.3f}")


if __name__ == "__main__":
    main()
