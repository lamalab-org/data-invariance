"""Validate adaptive stability scores across all three datasets.

For each dataset:
1. Train ERM and our method
2. Compute stability scores from both
3. Compute ground truth composition sensitivity (resampling or LOO)
4. Compare: ERM scores, our scores, and ADAPTIVE (blended) scores

The adaptive score uses the permutation test reliability to blend:
  adaptive = reliability * ours + (1 - reliability) * erm

If adaptive ≥ max(ours, erm) across all datasets, the framework works.

Usage:
    uv run python scripts/validate_adaptive_scores.py
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
from scipy.stats import spearmanr
import torchmetrics

from evaluate import compute_stability_scores, adaptive_stability_scores
from models import MLP
from train import discover_environments, make_dataloaders, train_discovered_split, train_erm, _ModelSelector, _val_score, evaluate
from utils import set_seed


def train_erm_simple(cfg, loaders, device, seed):
    """Train ERM with val-based model selection."""
    set_seed(seed)
    train_ds = loaders["train"].dataset
    if cfg.dataset.arch == "resnet":
        from models import make_resnet_backbone
        backbone, out_dim = make_resnet_backbone()
        model = MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
    else:
        model = MLP(input_dim=train_ds.input_dim, hidden_dim=256).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
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
        id_m = evaluate(model, loaders["id_test"], device)
        selector.update(_val_score(id_m), model, {})

    selector.restore(model)
    return model


def get_resampling_variance(cfg, loaders, device, n_seeds=5, test_key="ood_test"):
    """Train multiple ERMs, compute per-example prediction variance."""
    seeds = [42, 123, 456, 789, 1337][:n_seeds]
    all_probs = []

    for seed in seeds:
        model = train_erm_simple(cfg, loaders, device, seed)
        model.eval()
        probs = []
        with torch.no_grad():
            for batch in loaders[test_key]:
                x = batch["image"].to(device)
                p = model(x).softmax(1)[:, 1]
                probs.append(p.cpu())
        all_probs.append(torch.cat(probs))

    prob_matrix = torch.stack(all_probs)
    variance = prob_matrix.var(dim=0)
    flip_rate = ((prob_matrix > 0.5) != (prob_matrix > 0.5)[0:1]).float().mean(dim=0)
    return variance, flip_rate


def evaluate_scores(scores_dict, ground_truth_var, label=""):
    """Compute Spearman ρ and AUROC for a set of stability scores."""
    results = {}
    fragile = (ground_truth_var > ground_truth_var.median()).long()

    for name in ["entropy", "loss"]:
        if name in scores_dict:
            rho, _ = spearmanr(scores_dict[name].numpy(), ground_truth_var.numpy())
            results[f"spearman_{name}"] = rho

            auroc = torchmetrics.AUROC(task="binary")
            results[f"auroc_{name}"] = auroc(scores_dict[name], fragile).item()

    return results


def run_dataset(dataset_name, cfg, device):
    """Run the full adaptive score validation on one dataset."""
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*60}")

    set_seed(42)
    loaders = make_dataloaders(cfg)

    # Step 1: Ground truth composition sensitivity
    print("Computing ground truth (5-seed resampling)...")
    gt_var, gt_flip = get_resampling_variance(cfg, loaders, device, n_seeds=5)
    print(f"  Mean variance: {gt_var.mean():.4f}")
    print(f"  Examples with variance > median: {(gt_var > gt_var.median()).float().mean():.1%}")

    # Step 2: Train our method
    print("Training our method...")
    set_seed(42)
    assignment, weights, disc_metrics = discover_environments(cfg, loaders, device)
    reliability = disc_metrics.get("adaptive/reliability", 1.0)
    signal_ratio = disc_metrics.get("adaptive/signal_ratio", 0.0)
    print(f"  Signal ratio: {signal_ratio:.1f}, Reliability: {reliability:.2f}")

    set_seed(42)
    train_ds = loaders["train"].dataset
    if cfg.dataset.arch == "resnet":
        from models import make_resnet_backbone
        backbone, out_dim = make_resnet_backbone()
        our_model = MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
    else:
        our_model = MLP(input_dim=train_ds.input_dim, hidden_dim=256).to(device)
    run = wandb.init(mode="disabled")
    train_discovered_split(cfg, our_model, loaders, device, run, assignment, weights, disc_metrics)
    run.finish()

    # Step 3: Train ERM
    print("Training ERM...")
    erm_model = train_erm_simple(cfg, loaders, device, seed=42)

    # Step 4: Compute scores
    print("Computing stability scores...")
    test_key = "ood_test"
    scores_ours = compute_stability_scores(our_model, loaders[test_key], device)
    scores_erm = compute_stability_scores(erm_model, loaders[test_key], device)
    scores_adaptive = adaptive_stability_scores(scores_ours, scores_erm, reliability)

    # Step 5: Evaluate all three
    res_ours = evaluate_scores(scores_ours, gt_var, "Ours")
    res_erm = evaluate_scores(scores_erm, gt_var, "ERM")
    res_adaptive = evaluate_scores(scores_adaptive, gt_var, "Adaptive")

    print(f"\n--- Results (reliability={reliability:.2f}) ---")
    print(f"{'Metric':<25} {'ERM':>8} {'Ours':>8} {'Adaptive':>8} {'Best':>8}")
    print("-" * 60)
    for metric_name in ["spearman_entropy", "spearman_loss", "auroc_entropy", "auroc_loss"]:
        v_erm = res_erm.get(metric_name, 0)
        v_ours = res_ours.get(metric_name, 0)
        v_adapt = res_adaptive.get(metric_name, 0)
        best = "Adaptive" if v_adapt >= max(v_erm, v_ours) - 0.01 else ("Ours" if v_ours > v_erm else "ERM")
        print(f"{metric_name:<25} {v_erm:>8.3f} {v_ours:>8.3f} {v_adapt:>8.3f} {best:>8}")

    return {
        "dataset": dataset_name,
        "reliability": reliability,
        "signal_ratio": signal_ratio,
        **{f"erm_{k}": v for k, v in res_erm.items()},
        **{f"ours_{k}": v for k, v in res_ours.items()},
        **{f"adaptive_{k}": v for k, v in res_adaptive.items()},
    }


def main():
    device = torch.device("cpu")

    # Dataset configs
    configs = {
        "CMNIST": OmegaConf.create({
            "dataset": {"name": "cmnist", "arch": "mlp", "train_correlation": 0.9, "test_correlation": 0.1, "label_noise": 0.25, "data_dir": "./data"},
            "model": {"hidden_dim": 256, "separate_backbones": False, "num_heads": 2},
            "training": {
                "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 256, "epochs": 10, "seed": 42,
                "lambda_disagree": 10.0, "adv_lr": 1e-2, "discovery_epochs": 5,
                "discovery_criterion": "loss", "discovery_quantile": 0.5,
                "discovery_upweight": 50.0, "discovery_reweight": 0.0,
                "discovery_rounds": 1, "lambda_anneal_factor": 1.0,
                "early_stop_patience": 5, "num_discovery_envs": 2,
                "freeze_backbone": False, "balanced_sampling": False,
                "env_mixup": 0.0, "training_noise": 0.0,
                "adv_init": "zeros", "adv_init_scale": 1.0, "head_noise": 0.0,
                "adv_warmup_epochs": 0, "adv_steps_per_model_step": 1,
                "lambda_warmup_epochs": 0, "adv_entropy_bonus": 0.0,
                "lambda_threshold": 0.0, "lambda_ramp_range": 0.0, "adv_mode": "task_loss",
            },
            "method": {"name": "discovered_split"}, "wandb": {"enabled": False},
        }),
        "TADF": OmegaConf.create({
            "dataset": {"name": "tadf", "arch": "mlp",
                "parquet_path": "/Users/kevinmaikjablonka/git/lamalab/clever-materials-hans/src/tex/output/tadf_preprocess.parquet",
                "spurious_property": None, "spurious_correlation": 0.9, "data_dir": "./data"},
            "model": {"hidden_dim": 256, "separate_backbones": False, "num_heads": 2},
            "training": {
                "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 64, "epochs": 20, "seed": 42,
                "lambda_disagree": 10.0, "adv_lr": 1e-2, "discovery_epochs": 5,
                "discovery_criterion": "loss", "discovery_quantile": 0.5,
                "discovery_upweight": 50.0, "discovery_reweight": 0.0,
                "discovery_rounds": 1, "lambda_anneal_factor": 1.0,
                "early_stop_patience": 5, "num_discovery_envs": 2,
                "freeze_backbone": False, "balanced_sampling": False,
                "env_mixup": 0.0, "training_noise": 0.0,
                "adv_init": "zeros", "adv_init_scale": 1.0, "head_noise": 0.0,
                "adv_warmup_epochs": 0, "adv_steps_per_model_step": 1,
                "lambda_warmup_epochs": 0, "adv_entropy_bonus": 0.0,
                "lambda_threshold": 0.0, "lambda_ramp_range": 0.0, "adv_mode": "task_loss",
            },
            "method": {"name": "discovered_split"}, "wandb": {"enabled": False},
        }),
    }

    all_results = []
    for name, cfg in configs.items():
        result = run_dataset(name, cfg, device)
        all_results.append(result)

    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY: Adaptive stability scores across datasets")
    print(f"{'='*60}")
    print(f"{'Dataset':<10} {'Reliability':>12} {'ERM ρ':>8} {'Ours ρ':>8} {'Adaptive ρ':>12} {'Winner':>8}")
    print("-" * 62)
    for r in all_results:
        erm_rho = r.get("erm_spearman_loss", 0)
        ours_rho = r.get("ours_spearman_loss", 0)
        adapt_rho = r.get("adaptive_spearman_loss", 0)
        best = "Adaptive" if adapt_rho >= max(erm_rho, ours_rho) - 0.01 else ("Ours" if ours_rho > erm_rho else "ERM")
        print(f"{r['dataset']:<10} {r['reliability']:>12.2f} {erm_rho:>8.3f} {ours_rho:>8.3f} {adapt_rho:>12.3f} {best:>8}")


if __name__ == "__main__":
    main()
