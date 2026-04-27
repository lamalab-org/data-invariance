"""Train K models for partition-sensitivity uncertainty evaluation.

Three modes:
  - erm:       one full-data model (provides the 'prediction' used by other
               methods; also provides the softmax-entropy baseline).
  - ensemble:  K full-data models differing only in init seed (deep-ensemble
               disagreement baseline).
  - partition: K models trained on *disjoint* random partitions of the
               training data (partition-pair disagreement = our new signal).

For each trained model we save softmax probabilities on id_test and ood_test
(plus labels, spurious attribute, example indices) to an NPZ file. The
evaluation script in stability_eval.py consumes these NPZs and computes
uncertainty scores + downstream metrics.

Design choices:
  - Data construction is keyed on data_seed (fixed per protocol run). Model
    init uses a distinct model_seed per k, so "ensemble" and "partition" share
    the same underlying data — they only differ in how much of it each model
    sees.
  - Partitions are disjoint and cover the whole training set (np.array_split).
    K=2 gives the canonical partition-pair; larger K trades variance for sample
    efficiency per model.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_experiment import HPARAMS, _build_model, build_cfg  # noqa: E402
from train import _ModelSelector, evaluate, make_dataloaders  # noqa: E402
from utils import set_seed  # noqa: E402


def _move_x(batch_image, device):
    if isinstance(batch_image, dict):
        return {k: v.to(device) for k, v in batch_image.items()}
    return batch_image.to(device)


def _predict(model, loader, device):
    """Return (probs, labels, spurious, indices) over a test loader."""
    model.eval()
    probs, labels, spur, idx = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = _move_x(batch["image"], device)
            logits = model(x)
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
            labels.append(batch["label"].numpy())
            idx.append(batch["index"].numpy())
            spur.append(batch.get("spurious",
                                  torch.zeros_like(batch["label"])).numpy())
    return (np.concatenate(probs), np.concatenate(labels),
            np.concatenate(spur), np.concatenate(idx))


def _train_one(cfg, loaders, device, model_seed, epochs, sub_indices=None):
    """Train one ERM model. sub_indices (1D np.ndarray) restricts training."""
    set_seed(model_seed)
    model = _build_model(cfg, loaders, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay)

    if sub_indices is None:
        train_loader = loaders["train"]
    else:
        base = loaders["train"].dataset
        sub = Subset(base, sub_indices.tolist())
        train_loader = DataLoader(
            sub, batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=loaders["train"].num_workers,
            pin_memory=loaders["train"].pin_memory)

    sel = _ModelSelector()
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x = _move_x(batch["image"], device)
            y = batch["label"].to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        id_m = evaluate(model, loaders["id_test"], device)
        sel.update(id_m.get("acc", 0), model,
                   {"acc": id_m.get("acc", 0), "epoch": epoch})
    sel.restore(model)
    return model


def run(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu")

    out_root = Path(args.output_dir) / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args.dataset)
    if args.train_correlation is not None:
        cfg.dataset.train_correlation = args.train_correlation
    if args.label_noise is not None:
        cfg.dataset.label_noise = args.label_noise
    if args.weight_decay is not None:
        cfg.training.weight_decay = args.weight_decay
    epochs = HPARAMS[args.dataset]["epochs"]
    data_seeds = [int(s) for s in args.data_seeds.split(",")]

    for data_seed in data_seeds:
        print(f"\n=== dataset={args.dataset}  data_seed={data_seed}  "
              f"mode={args.mode}  K={args.K} ===")

        cfg.training.seed = data_seed
        set_seed(data_seed)
        loaders = make_dataloaders(cfg)
        n_train = len(loaders["train"].dataset)

        rng = np.random.default_rng(data_seed)
        shuffled = rng.permutation(n_train)
        # Optional: cap the training-set size for N-scaling experiments.
        # Applied before splitting into partitions so a K=2 partition at
        # subsample=M uses M/2 examples per model.
        if args.subsample_size is not None and args.subsample_size < n_train:
            shuffled = shuffled[: args.subsample_size]

        if args.mode == "partition":
            partitions = [np.array(p) for p in np.array_split(shuffled, args.K)]
        elif args.mode == "ensemble":
            # For subsampled ensemble, reuse the same subsample for every k
            # so ensemble captures only seed variance at the same N.
            partitions = ([shuffled] * args.K if args.subsample_size is not None
                          else [None] * args.K)
        elif args.mode == "erm":
            partitions = [shuffled if args.subsample_size is not None else None]
        else:
            raise ValueError(f"unknown mode {args.mode}")

        for k, part_idx in enumerate(partitions):
            model_seed = data_seed * 1000 + k
            n_seen = n_train if part_idx is None else len(part_idx)
            print(f"  k={k}  model_seed={model_seed}  n_train_seen={n_seen}")

            model = _train_one(cfg, loaders, device, model_seed, epochs,
                               sub_indices=part_idx)
            id_p, id_y, id_s, id_i = _predict(model, loaders["id_test"], device)
            ood_p, ood_y, ood_s, ood_i = _predict(model, loaders["ood_test"], device)

            save_kwargs = dict(
                id_probs=id_p, id_labels=id_y, id_spurious=id_s, id_indices=id_i,
                ood_probs=ood_p, ood_labels=ood_y, ood_spurious=ood_s,
                ood_indices=ood_i,
                partition_indices=(np.arange(n_train) if part_idx is None
                                   else part_idx),
                data_seed=data_seed, model_seed=model_seed, k=k, mode=args.mode)

            if args.save_train_preds:
                # Deterministic-order full training-set loader for attribution.
                train_eval_loader = DataLoader(
                    loaders["train"].dataset,
                    batch_size=cfg.training.batch_size, shuffle=False,
                    num_workers=loaders["train"].num_workers,
                    pin_memory=loaders["train"].pin_memory)
                tr_p, tr_y, tr_s, tr_i = _predict(
                    model, train_eval_loader, device)
                save_kwargs.update(
                    train_probs=tr_p, train_labels=tr_y,
                    train_spurious=tr_s, train_indices=tr_i)
                # If dataset tracks label-flip ground truth, save it too.
                base_ds = loaders["train"].dataset
                if hasattr(base_ds, "flipped"):
                    save_kwargs["train_flipped"] = base_ds.flipped.numpy()

            out_path = out_root / f"{args.mode}_seed{data_seed}_k{k}.npz"
            np.savez_compressed(out_path, **save_kwargs)

            id_acc = float((id_p.argmax(1) == id_y).mean())
            ood_acc = float((ood_p.argmax(1) == ood_y).mean())
            print(f"    id_acc={id_acc:.4f}  ood_acc={ood_acc:.4f}  "
                  f"-> {out_path.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HPARAMS.keys()))
    ap.add_argument("--data_seeds", default="42",
                    help="Comma-separated data seeds (dataset construction).")
    ap.add_argument("--K", type=int, default=2,
                    help="Number of models (partitions for 'partition' mode, "
                         "ensemble members for 'ensemble', ignored for 'erm').")
    ap.add_argument("--mode",
                    choices=["erm", "ensemble", "partition"], required=True)
    ap.add_argument("--output_dir", default="outputs/stability")
    ap.add_argument("--save_train_preds", action="store_true",
                    help="Also save predictions on the full training set "
                         "(needed for label-noise attribution evaluation).")
    ap.add_argument("--train_correlation", type=float, default=None,
                    help="Override cfg.dataset.train_correlation (CMNIST).")
    ap.add_argument("--label_noise", type=float, default=None,
                    help="Override cfg.dataset.label_noise (CMNIST).")
    ap.add_argument("--subsample_size", type=int, default=None,
                    help="Cap training set to this size before partitioning "
                         "(for within-dataset N-scaling experiments).")
    ap.add_argument("--weight_decay", type=float, default=None,
                    help="Override cfg.training.weight_decay (for "
                         "regularization / algorithmic-stability sweeps).")
    args = ap.parse_args()
    run(args)
