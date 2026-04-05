"""Evaluate stability scores: does our model's uncertainty predict OOD flips
better than ERM's uncertainty?

Trains ERM and our method on CMNIST (corr=0.9), then compares multiple
uncertainty scores (confidence, entropy, loss, MC dropout) for predicting
which examples flip between ID and OOD test sets.

Usage:
    uv run python scripts/evaluate_stability.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import torch
import wandb
from omegaconf import OmegaConf

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
    print("=" * 60)
    print("Training ERM...")
    set_seed(42)
    erm_model = MLP(input_dim=input_dim, hidden_dim=256).to(device)
    run = wandb.init(mode="disabled")
    train_erm(cfg, erm_model, loaders, device, run)
    run.finish()

    # ---- Train our method ----
    print("=" * 60)
    print("Training discovered split + V-REx...")
    set_seed(42)
    assignment, weights, disc_metrics = discover_environments(cfg, loaders, device)
    set_seed(42)
    our_model = MLP(input_dim=input_dim, hidden_dim=256).to(device)
    run = wandb.init(mode="disabled")
    train_discovered_split(cfg, our_model, loaders, device, run, assignment, weights, disc_metrics)
    run.finish()

    # ---- Compute scores on ID and OOD test sets ----
    print("=" * 60)
    print("Computing stability scores (including MC dropout)...")
    scores_erm_id = compute_stability_scores(erm_model, loaders["id_test"], device)
    scores_erm_ood = compute_stability_scores(erm_model, loaders["ood_test"], device)
    scores_ours_id = compute_stability_scores(our_model, loaders["id_test"], device)
    scores_ours_ood = compute_stability_scores(our_model, loaders["ood_test"], device)

    # ---- Evaluate ----
    print("=" * 60)
    results = evaluate_stability_discrimination(
        scores_ours=scores_ours_id,
        scores_erm=scores_erm_id,
        id_preds_erm=scores_erm_id["predictions"],
        ood_preds_erm=scores_erm_ood["predictions"],
        id_preds_ours=scores_ours_id["predictions"],
        ood_preds_ours=scores_ours_ood["predictions"],
        labels=scores_erm_id["labels"],
    )

    # ---- Print results ----
    print("\n" + "=" * 60)
    print("STABILITY SCORE EVALUATION — CMNIST (corr=0.9 → 0.1)")
    print("=" * 60)

    print(f"\n--- Accuracy ---")
    print(f"ERM:  ID={results['erm_id_acc']:.1%}  OOD={results['erm_ood_acc']:.1%}  flip_rate={results['erm_flip_rate']:.1%}")
    print(f"Ours: ID={results['ours_id_acc']:.1%}  OOD={results['ours_ood_acc']:.1%}  flip_rate={results['ours_flip_rate']:.1%}")

    print(f"\n--- AUROC: predicting ERM's flips (higher = better) ---")
    print(f"{'Score type':<25} {'Our model':>10} {'ERM':>10} {'Δ':>10}")
    print("-" * 57)
    for score_name in ["confidence_inv", "entropy", "loss", "mc_dropout_var"]:
        key_ours = f"auroc_ours_{score_name}_vs_erm_flips"
        key_erm = f"auroc_erm_{score_name}_vs_erm_flips"
        if key_ours in results and key_erm in results:
            v_ours = results[key_ours]
            v_erm = results[key_erm]
            delta = v_ours - v_erm
            print(f"{score_name:<25} {v_ours:>10.3f} {v_erm:>10.3f} {delta:>+10.3f}")

    print(f"\n--- AUROC: predicting our model's flips ---")
    print(f"{'Score type':<25} {'Our model':>10} {'ERM':>10} {'Δ':>10}")
    print("-" * 57)
    for score_name in ["confidence_inv", "entropy", "loss", "mc_dropout_var"]:
        key_ours = f"auroc_ours_{score_name}_vs_ours_flips"
        key_erm = f"auroc_erm_{score_name}_vs_ours_flips"
        if key_ours in results and key_erm in results:
            v_ours = results[key_ours]
            v_erm = results[key_erm]
            delta = v_ours - v_erm
            print(f"{score_name:<25} {v_ours:>10.3f} {v_erm:>10.3f} {delta:>+10.3f}")

    print(f"\n--- Calibration: ERM flip rate by our model's entropy quintile ---")
    print(f"(Higher entropy bins should have higher flip rates if well-calibrated)")
    for i in range(5):
        key_rate = f"calibration_bin{i}_erm_flip_rate"
        key_n = f"calibration_bin{i}_n"
        if key_rate in results:
            print(f"  Bin {i} (n={results[key_n]:.0f}): flip_rate={results[key_rate]:.3f}")


if __name__ == "__main__":
    main()
