"""Resampling stability experiment — the make-or-break test.

Tests the core claim: "our method's predictions are more stable under
training data composition changes than ERM's."

Protocol:
    1. For each of N_subsamples random 90%-subsets of training data:
       a. Train an ERM model on that subset
       b. Train our method (discovery + upweight + V-REx) on that subset
       c. Evaluate both on the FULL test set, record per-example predictions
    2. Compute per-example flip rate: fraction of subsamples where the
       prediction differs from the majority prediction for that example
    3. Compare: if our method's flip rate < ERM's flip rate, the model
       IS more stable to data composition — validating the core story

This directly measures algorithmic stability (Bousquet & Elisseeff 2002)
as a training objective rather than an analysis tool.

Usage:
    uv run python scripts/resampling_stability_test.py --dataset cmnist --device cpu
    uv run python scripts/resampling_stability_test.py --dataset waterbirds --device mps
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

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
from run_experiment import build_cfg, HPARAMS  # noqa: E402

N_SUBSAMPLES = 10
SUBSAMPLE_FRAC = 0.9
SWA_WINDOW = 5


def log(msg: str) -> None:
    print(msg, flush=True)


def build_model(cfg, loaders, device):
    if cfg.dataset.arch == "resnet":
        backbone, out_dim = make_resnet_backbone()
        return MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
    input_dim = loaders["train"].dataset.input_dim
    return MLP(input_dim=input_dim, hidden_dim=cfg.model.hidden_dim).to(device)


class _ReindexedSubset(torch.utils.data.Dataset):
    """Subset that remaps indices to 0..len-1 and proxies dataset attributes.

    discover_environments expects batch["index"] ∈ [0, N) where N = len(dataset).
    A plain Subset returns the ORIGINAL dataset indices, which are out-of-range
    for the N-sized tensors inside discovery. This wrapper remaps indices so
    discovery sees a clean 0..N_sub-1 range, and stores the original→local
    mapping for later reconstruction.
    """

    def __init__(self, dataset, indices):
        self._dataset = dataset
        self._indices = list(indices)
        # Map: original_idx → local_idx (0..N_sub-1)
        self._orig_to_local = {orig: local for local, orig in enumerate(self._indices)}

    def __getitem__(self, local_idx):
        orig_idx = self._indices[local_idx]
        item = self._dataset[orig_idx]
        item["index"] = local_idx  # REMAP to contiguous index
        item["_orig_index"] = orig_idx  # keep original for later
        return item

    def __len__(self):
        return len(self._indices)

    @property
    def spurious(self):
        full = self._dataset.spurious
        return full[self._indices]

    @property
    def labels(self):
        full = self._dataset.labels
        return full[self._indices]

    @property
    def input_dim(self):
        return self._dataset.input_dim

    def __getattr__(self, name):
        # Proxy remaining attribute access to the underlying dataset
        return getattr(self._dataset, name)


def make_subsampled_loader(train_ds, indices, batch_size):
    """Create a dataloader from a subset of training data."""
    subset = _ReindexedSubset(train_ds, indices)
    use_cuda = torch.cuda.is_available()
    return DataLoader(
        subset, batch_size=batch_size, shuffle=True,
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
        persistent_workers=True if use_cuda else False,
    )


def get_test_predictions(model, loader, device):
    """Return per-example predicted classes on the test set."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
    return torch.cat(all_preds)


def auto_lambda(L_mean, risk_var, lr, steps_per_epoch, reliability):
    """Fully principled λ from discovery quantities."""
    T_eff = lr * steps_per_epoch
    lam = L_mean / (2.0 * max(risk_var, 1e-8) * math.sqrt(T_eff))
    return lam * reliability


def train_erm_on_subset(cfg, full_loaders, subset_indices, device, epochs):
    """Train ERM on a subset, evaluate on the full test sets."""
    train_ds = full_loaders["train"].dataset
    B = cfg.training.batch_size
    sub_loader = make_subsampled_loader(train_ds, subset_indices, B)

    model = build_model(cfg, full_loaders, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )

    for epoch in range(epochs):
        model.train()
        for batch in sub_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def train_ours_on_subset(cfg, full_loaders, subset_indices, device, epochs):
    """Train our method (discovery + upweight + V-REx) on a subset."""
    train_ds = full_loaders["train"].dataset
    B = cfg.training.batch_size
    N_sub = len(subset_indices)

    # Build a temporary cfg for the subset
    sub_loader = make_subsampled_loader(train_ds, subset_indices, B)

    # Create temporary loaders dict that discovery expects
    sub_loaders = {
        "train": sub_loader,
        "id_test": full_loaders["id_test"],
        "ood_test": full_loaders["ood_test"],
    }

    # Discovery on the subset
    assignment_sub, weights_sub, disc_m = discover_environments(cfg, sub_loaders, device)
    reliability = disc_m.get("adaptive/reliability", 1.0)
    actual_rv = disc_m["adaptive/actual_risk_var"]

    # Compute L_mean for auto-lambda
    # Use a quick forward pass with the discovery model
    # (discover_environments doesn't return L_mean, so estimate from risk_var)
    # Approximation: L_mean ≈ sqrt(risk_var) for typical loss distributions
    # Better: just use a fixed reasonable lambda since auto-lambda is a secondary contribution
    steps_per_epoch = N_sub / B
    lr = cfg.training.lr

    # For the auto-lambda, we need L_mean. Estimate from the mean of weights
    # (weights = 1 + 50 * loss/max_loss, so mean_weight ≈ 1 + 50 * mean_loss/max_loss)
    # This is crude. Instead, use the simpler N-scaling rule here:
    lam = 10.0 * (5000 / N_sub) * reliability

    model = build_model(cfg, full_loaders, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=cfg.training.weight_decay
    )

    # Map subset indices to local assignment/weights
    assignment_dev = assignment_sub.to(device)
    weights_dev = weights_sub.to(device)
    K = 2

    for epoch in range(epochs):
        model.train()
        for batch in sub_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            idx = batch["index"].to(device)

            ce = F.cross_entropy(model(x), y, reduction="none")

            # Map global indices to local subset indices
            # The Subset wraps the original dataset, so batch["index"] gives
            # the ORIGINAL dataset index. Assignment is indexed by position
            # in the subset, not the original index. We need to map.
            # Actually, discover_environments indexes by the loader's batch["index"],
            # which for a Subset gives the original dataset indices.
            # So assignment_sub[original_idx] should work... but assignment_sub
            # has length N_sub, indexed 0..N_sub-1. The indices in the batch
            # are the ORIGINAL indices (0..N-1).
            #
            # FIX: we need to create a mapping from original indices to
            # subset positions, or re-index assignment by original index.
            # Since discover_environments creates assignment of length N_sub
            # indexed by the batch order... this is tricky.
            #
            # Simplest fix: create assignment and weights tensors indexed by
            # ORIGINAL dataset index (sparse, only subset positions filled).
            # This wastes memory but is correct.
            pass  # handled below

            # Actually, the Subset's __getitem__ returns the original dataset's
            # item with the original index. So batch["index"] IS the original
            # index. But assignment_sub is of length N_sub, indexed 0..N_sub-1.
            # We need to know which position in the subset each batch item is.
            #
            # The correct approach: build a global-indexed assignment tensor.

            # For now, use a simpler approach: just do weighted ERM (no V-REx
            # per-env splitting) since the per-example weights already encode
            # the discovery signal. This is the "ERM + upweight" version.
            #
            # Note: uses weighted ERM (not V-REx) on subsets for simplicity.

            w = weights_dev[idx] if idx.max() < len(weights_dev) else torch.ones_like(ce)
            loss = (w * ce).sum() / w.sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    return model


def train_ours_on_subset_v2(cfg, full_loaders, subset_indices, device, epochs):
    """Train our method on a subset — V2 with proper index mapping."""
    train_ds = full_loaders["train"].dataset
    B = cfg.training.batch_size
    N_full = len(train_ds)
    N_sub = len(subset_indices)

    sub_loader = make_subsampled_loader(train_ds, subset_indices, B)
    sub_loaders = {
        "train": sub_loader,
        "id_test": full_loaders["id_test"],
        "ood_test": full_loaders["ood_test"],
    }

    # Discovery on the subset
    assignment_sub, weights_sub, disc_m = discover_environments(cfg, sub_loaders, device)
    reliability = disc_m.get("adaptive/reliability", 1.0)

    # With _ReindexedSubset, batch["index"] returns local indices 0..N_sub-1,
    # matching assignment_sub and weights_sub directly. No remapping needed.

    # Lambda: N-scaling rule
    lam = 10.0 * (5000 / N_sub) * reliability

    model = build_model(cfg, full_loaders, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )

    assignment_dev = assignment_sub.to(device)
    weights_dev = weights_sub.to(device)
    K = 2

    for epoch in range(epochs):
        model.train()
        for batch in sub_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            idx = batch["index"].to(device)

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
                loss = mean_loss + lam * risk_var
            else:
                loss = ce.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HPARAMS.keys()))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_subsamples", type=int, default=N_SUBSAMPLES)
    ap.add_argument("--subsample_frac", type=float, default=SUBSAMPLE_FRAC)
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = build_cfg(args.dataset)
    cfg.training.seed = 42
    set_seed(42)

    tr = HPARAMS[args.dataset]
    epochs = tr["epochs"]

    log(f"=== Resampling stability test: {args.dataset} ===")
    log(f"device={device}  n_subsamples={args.n_subsamples}  frac={args.subsample_frac}")
    log(f"epochs={epochs}")

    # Build full dataloaders (for test sets and to get the full dataset)
    loaders = make_dataloaders(cfg)
    train_ds = loaders["train"].dataset
    N = len(train_ds)
    N_sub = int(N * args.subsample_frac)
    log(f"N_train={N}  N_sub={N_sub}")

    # Use OOD test for stability evaluation (that's where composition matters)
    test_loader = loaders["ood_test"]
    N_test = len(test_loader.dataset)

    # Collect predictions from each subsample
    erm_preds = []  # list of (N_test,) tensors
    ours_preds = []

    g = torch.Generator().manual_seed(42)

    for s in range(args.n_subsamples):
        log(f"\n--- Subsample {s+1}/{args.n_subsamples} ---")

        # Random 90% subset
        perm = torch.randperm(N, generator=g)
        subset_indices = perm[:N_sub].tolist()

        # Train ERM
        set_seed(100 + s)
        log(f"  Training ERM on {N_sub} examples ...")
        erm_model = train_erm_on_subset(cfg, loaders, subset_indices, device, epochs)
        erm_pred = get_test_predictions(erm_model, test_loader, device)
        erm_preds.append(erm_pred)
        del erm_model

        # Train ours
        set_seed(200 + s)
        log(f"  Training ours on {N_sub} examples ...")
        ours_model = train_ours_on_subset_v2(cfg, loaders, subset_indices, device, epochs)
        ours_pred = get_test_predictions(ours_model, test_loader, device)
        ours_preds.append(ours_pred)
        del ours_model

        # Quick comparison
        if s > 0:
            erm_agree = (erm_preds[-1] == erm_preds[0]).float().mean().item()
            ours_agree = (ours_preds[-1] == ours_preds[0]).float().mean().item()
            log(f"  Agreement with subsample 0: ERM={erm_agree:.4f}  ours={ours_agree:.4f}")

    # Aggregate: per-example flip rate
    erm_stack = torch.stack(erm_preds)   # (n_subsamples, N_test)
    ours_stack = torch.stack(ours_preds)

    # Majority vote
    erm_majority = erm_stack.mode(dim=0).values   # (N_test,)
    ours_majority = ours_stack.mode(dim=0).values

    # Flip rate: fraction of subsamples disagreeing with majority
    erm_flips = (erm_stack != erm_majority.unsqueeze(0)).float().mean(dim=0)  # (N_test,)
    ours_flips = (ours_stack != ours_majority.unsqueeze(0)).float().mean(dim=0)

    erm_mean_flip = erm_flips.mean().item()
    ours_mean_flip = ours_flips.mean().item()

    log(f"\n{'='*60}")
    log(f"=== RESULTS: {args.dataset} resampling stability ===")
    log(f"{'='*60}")
    log(f"Mean flip rate:  ERM = {erm_mean_flip:.4f}  ours = {ours_mean_flip:.4f}")
    log(f"Flip reduction:  {(1 - ours_mean_flip / max(erm_mean_flip, 1e-8)) * 100:.1f}%")

    # Per-example: are the same examples unstable in both?
    both_stable = ((erm_flips == 0) & (ours_flips == 0)).float().mean().item()
    erm_only_unstable = ((erm_flips > 0) & (ours_flips == 0)).float().mean().item()
    ours_only_unstable = ((erm_flips == 0) & (ours_flips > 0)).float().mean().item()
    both_unstable = ((erm_flips > 0) & (ours_flips > 0)).float().mean().item()

    log(f"\nPer-example stability breakdown:")
    log(f"  Both stable:       {both_stable:.4f}")
    log(f"  ERM-only unstable: {erm_only_unstable:.4f}  (ours stabilised these)")
    log(f"  Ours-only unstable:{ours_only_unstable:.4f}  (ours destabilised these)")
    log(f"  Both unstable:     {both_unstable:.4f}")

    # Accuracy: majority-vote accuracy for both
    test_labels = []
    for batch in test_loader:
        test_labels.append(batch["label"] if isinstance(batch["label"], torch.Tensor)
                          else torch.tensor(batch["label"]))
    test_labels = torch.cat(test_labels)

    erm_acc = (erm_majority == test_labels).float().mean().item()
    ours_acc = (ours_majority == test_labels).float().mean().item()
    log(f"\nMajority-vote accuracy:  ERM = {erm_acc:.4f}  ours = {ours_acc:.4f}")

    # Do the unstable examples correlate with spurious attribute?
    if hasattr(test_loader.dataset, "spurious"):
        spurious = torch.tensor([test_loader.dataset[i]["spurious"] for i in range(N_test)])
        labels = test_labels
        minority = (spurious != labels)  # spurious ≠ label
        log(f"\nFlip rate on minority (spurious≠label) vs majority:")
        log(f"  ERM:  minority={erm_flips[minority].mean():.4f}  majority={erm_flips[~minority].mean():.4f}")
        log(f"  Ours: minority={ours_flips[minority].mean():.4f}  majority={ours_flips[~minority].mean():.4f}")


if __name__ == "__main__":
    main()
