"""Stability scores for chemistry: which molecular property predictions
depend on the composition of the training set?

Ground truth: train models leaving out different author groups from the
TADF dataset. For each test molecule, measure how much the prediction
changes when different groups are excluded. Molecules whose predictions
depend strongly on which groups are included = composition-sensitive.

Then: does our single-model stability score predict this?

This directly operationalises the clever-hans concern: "does this
prediction depend on which lab's data was in the training set?"

Usage:
    uv run python scripts/stability_chemistry.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
import wandb
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from data import TADFDataset
from evaluate import compute_stability_scores
from models import MLP
from train import discover_environments, train_discovered_split, train_erm, _ModelSelector, _val_score, evaluate
from utils import set_seed


def main():
    parquet_path = "/Users/kevinmaikjablonka/git/lamalab/clever-materials-hans/src/tex/output/tadf_preprocess.parquet"

    # Load the full dataset to identify author groups
    df = pd.read_parquet(parquet_path)
    df["_target"] = df["standard_value"].apply(
        lambda x: float(str(x).strip("[]").split(",")[0]) if isinstance(x, str) else np.nan
    )
    df = df.dropna(subset=["_target"]).reset_index(drop=True)
    median_nm = df["_target"].median()
    df["_label"] = (df["_target"] >= median_nm).astype(int)

    # Find author groups with enough examples
    author_col = df["author_last_name"].fillna("unknown")
    author_counts = author_col.value_counts()
    top_authors = author_counts[author_counts >= 30].index.tolist()
    print(f"Authors with ≥30 papers: {len(top_authors)}")
    for a in top_authors[:10]:
        n = author_counts[a]
        frac1 = df.loc[author_col == a, "_label"].mean()
        print(f"  {a:20s}: n={n:4d}, frac_class1={frac1:.2f}")

    # Common config
    cfg = OmegaConf.create({
        "dataset": {
            "name": "tadf", "arch": "mlp",
            "parquet_path": parquet_path,
            "spurious_property": None,  # no artificial spurious correlation
            "spurious_correlation": 0.9,
            "data_dir": "./data",
        },
        "model": {"hidden_dim": 256, "separate_backbones": False, "num_heads": 2},
        "training": {
            "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 64,
            "epochs": 20, "seed": 42,
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
        "method": {"name": "erm"},
        "wandb": {"enabled": False},
    })

    device = torch.device("cpu")

    # =====================================================================
    # Step 1: Leave-one-author-group-out — ground truth composition sensitivity
    # =====================================================================
    # Use all data for a shared test set (20% holdout)
    full_ds = TADFDataset(parquet_path, split="test", seed=42)
    test_loader = DataLoader(full_ds, batch_size=64, shuffle=False)
    n_test = len(full_ds)
    input_dim = full_ds.input_dim

    print(f"\nTest set: {n_test} molecules")

    # Train models leaving out each top author group
    all_predictions = {}
    for author in top_authors[:8]:  # top 8 groups for tractability
        print(f"\nTraining ERM without '{author}'...")
        # Create training set excluding this author
        train_ds_full = TADFDataset(parquet_path, split="train", seed=42)
        # Filter out examples from this author
        # We need to access the author names — add them to the dataset
        train_df = pd.read_parquet(parquet_path)
        train_df["_target"] = train_df["standard_value"].apply(
            lambda x: float(str(x).strip("[]").split(",")[0]) if isinstance(x, str) else np.nan
        )
        train_df = train_df.dropna(subset=["_target"]).reset_index(drop=True)
        rng = np.random.RandomState(42)
        n = len(train_df)
        perm = rng.permutation(n)
        n_train = int(0.8 * n)
        train_indices = perm[:n_train]
        # Which training indices are NOT this author?
        train_authors = train_df["author_last_name"].fillna("unknown").iloc[train_indices]
        keep_mask = (train_authors != author).values
        keep_indices = np.where(keep_mask)[0]

        if len(keep_indices) < 100:
            print(f"  Skipping — too few examples ({len(keep_indices)})")
            continue

        subset = Subset(train_ds_full, keep_indices)
        sub_loader = DataLoader(subset, batch_size=64, shuffle=True, num_workers=0)

        set_seed(42)
        model = MLP(input_dim=input_dim, hidden_dim=256).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        selector = _ModelSelector()
        for epoch in range(20):
            model.train()
            for batch in sub_loader:
                x = batch["image"].to(device)
                y = batch["label"].to(device)
                loss = F.cross_entropy(model(x), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            val_metrics = evaluate(model, test_loader, device)
            selector.update(_val_score(val_metrics), model, {})
        selector.restore(model)

        # Get test predictions
        model.eval()
        preds = []
        probs = []
        with torch.no_grad():
            for batch in test_loader:
                x = batch["image"].to(device)
                logits = model(x)
                p = logits.softmax(1)
                preds.append(p.argmax(1).cpu())
                probs.append(p[:, 1].cpu())

        preds = torch.cat(preds)
        probs = torch.cat(probs)
        acc = (preds == full_ds.labels).float().mean()
        print(f"  Test acc (without {author}): {acc:.3f}")
        all_predictions[author] = {"preds": preds, "probs": probs}

    if len(all_predictions) < 3:
        print("Not enough author groups — exiting")
        return

    # Per-example composition sensitivity: variance of predictions across LOO models
    prob_matrix = torch.stack([v["probs"] for v in all_predictions.values()])
    pred_matrix = torch.stack([v["preds"] for v in all_predictions.values()])

    comp_sensitivity = prob_matrix.var(dim=0)  # (N_test,)
    majority = pred_matrix.mode(dim=0).values
    flip_rate = (pred_matrix != majority.unsqueeze(0)).float().mean(dim=0)

    print(f"\n{'='*60}")
    print("COMPOSITION SENSITIVITY (leave-one-author-group-out)")
    print(f"{'='*60}")
    print(f"Mean prediction variance: {comp_sensitivity.mean():.4f}")
    print(f"Examples with any flip: {(flip_rate > 0).float().mean():.1%}")
    print(f"Examples with flip_rate > 0.3: {(flip_rate > 0.3).float().mean():.1%}")

    # =====================================================================
    # Step 2: Train our method and ERM on full data, compute stability scores
    # =====================================================================
    print(f"\nTraining full ERM...")
    full_train_ds = TADFDataset(parquet_path, split="train", seed=42)
    full_train_loader = DataLoader(full_train_ds, batch_size=64, shuffle=True)
    full_loaders = {"train": full_train_loader, "id_test": test_loader, "ood_test": test_loader}

    set_seed(42)
    erm_model = MLP(input_dim=input_dim, hidden_dim=256).to(device)
    run = wandb.init(mode="disabled")
    train_erm(cfg, erm_model, full_loaders, device, run)
    run.finish()

    print("Training our method...")
    set_seed(42)
    assignment, weights, disc_m = discover_environments(cfg, full_loaders, device)
    set_seed(42)
    our_model = MLP(input_dim=input_dim, hidden_dim=256).to(device)
    run = wandb.init(mode="disabled")
    train_discovered_split(cfg, our_model, full_loaders, device, run, assignment, weights, disc_m)
    run.finish()

    print("Computing stability scores...")
    scores_ours = compute_stability_scores(our_model, test_loader, device)
    scores_erm = compute_stability_scores(erm_model, test_loader, device)

    # =====================================================================
    # Step 3: Does our stability score predict composition sensitivity?
    # =====================================================================
    from scipy.stats import spearmanr
    import torchmetrics

    print(f"\n{'='*60}")
    print("STABILITY SCORE VALIDATION — Chemistry (TADF)")
    print(f"{'='*60}")

    print(f"\n--- Spearman ρ with composition sensitivity ---")
    print(f"{'Score':<25} {'Our model':>10} {'ERM':>10}")
    print("-" * 47)
    for name in ["entropy", "loss"]:
        r_ours, p_ours = spearmanr(scores_ours[name].numpy(), comp_sensitivity.numpy())
        r_erm, p_erm = spearmanr(scores_erm[name].numpy(), comp_sensitivity.numpy())
        print(f"{name:<25} {r_ours:>10.3f} {r_erm:>10.3f}")

    r_ours, _ = spearmanr((1 - scores_ours["confidence"]).numpy(), comp_sensitivity.numpy())
    r_erm, _ = spearmanr((1 - scores_erm["confidence"]).numpy(), comp_sensitivity.numpy())
    print(f"{'1 - confidence':<25} {r_ours:>10.3f} {r_erm:>10.3f}")

    # AUROC for fragile examples
    fragile = (flip_rate > 0).long()
    n_fragile = fragile.sum().item()
    if n_fragile > 0 and n_fragile < n_test:
        auroc = torchmetrics.AUROC(task="binary")
        print(f"\nFragile molecules (prediction flips with author removal): {n_fragile}/{n_test} ({n_fragile/n_test:.1%})")
        print(f"\n--- AUROC for predicting composition-sensitive molecules ---")
        print(f"{'Score':<25} {'Our model':>10} {'ERM':>10} {'Δ':>10}")
        print("-" * 57)
        for name in ["entropy", "loss"]:
            auroc.reset()
            a_ours = auroc(scores_ours[name], fragile).item()
            auroc.reset()
            a_erm = auroc(scores_erm[name], fragile).item()
            print(f"{name:<25} {a_ours:>10.3f} {a_erm:>10.3f} {a_ours - a_erm:>+10.3f}")

    # Calibration
    print(f"\n--- Calibration: our entropy quintile → mean composition sensitivity ---")
    ours_entropy = scores_ours["entropy"]
    n_bins = 5
    quantiles = torch.linspace(0, 1, n_bins + 1)
    bin_edges = torch.quantile(ours_entropy, quantiles)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (ours_entropy >= lo) & (ours_entropy < hi) if i < n_bins - 1 else (ours_entropy >= lo)
        if mask.sum() > 0:
            mv = comp_sensitivity[mask].mean()
            mf = flip_rate[mask].mean()
            print(f"  Bin {i} (n={mask.sum():>4}): mean_sensitivity={mv:.4f}, mean_flip={mf:.3f}")


if __name__ == "__main__":
    main()
