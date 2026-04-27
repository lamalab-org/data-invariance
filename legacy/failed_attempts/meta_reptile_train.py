"""Hard-subset Reptile meta-learning for sample-fragility reduction.

Outer loop maintains a single global θ. At each meta-step:
  1. Compute per-example losses from θ on all training examples.
  2. Build *adversarial* subsets:
       hard = top-50% loss   easy = bottom-50% loss
     These are the maximally divergent partition conditional on θ.
  3. Inner training from θ on each subset for `inner_steps` SGD steps:
     θ_hard = SGD(θ, hard, K)   θ_easy = SGD(θ, easy, K)
  4. Reptile-mean meta update:
       θ ← θ + alpha · ½((θ_hard − θ) + (θ_easy − θ))
  5. Optional consistency penalty (on a held-out anchor split):
       L_cons = sym_KL(p_θ_hard(anchor), p_θ_easy(anchor))
     We backprop L_cons w.r.t. θ via the *first-order* approximation
     (treat θ_hard, θ_easy as constants given their outer trajectories);
     concretely, we add a gradient pulling θ toward the symmetric
     midpoint of (θ_hard, θ_easy) weighted by the magnitude of
     L_cons. Approximation: extra step toward Reptile-mean scaled by L_cons.

We save the final θ's predictions on id_test and ood_test, and compute
cross-sample churn against θ's trained at different data_seeds.

This is the Reptile + adversarial-difficulty first-solution attempt.
"""
import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_experiment import HPARAMS, _build_model, build_cfg  # noqa: E402
from train import make_dataloaders  # noqa: E402
from utils import set_seed  # noqa: E402


def _move_x(batch_image, device):
    if isinstance(batch_image, dict):
        return {k: v.to(device) for k, v in batch_image.items()}
    return batch_image.to(device)


def _predict(model, loader, device):
    model.eval()
    probs, labels, idx = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = _move_x(batch["image"], device)
            logits = model(x)
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
            labels.append(batch["label"].numpy())
            idx.append(batch["index"].numpy())
    return (np.concatenate(probs), np.concatenate(labels),
            np.concatenate(idx))


def _per_example_loss(model, loader, device):
    """Return ce loss for every training example, in dataset-index order."""
    model.eval()
    losses = {}
    with torch.no_grad():
        for batch in loader:
            x = _move_x(batch["image"], device)
            y = batch["label"].to(device)
            logits = model(x)
            ce = F.cross_entropy(logits, y, reduction="none").cpu().numpy()
            for i, idx in enumerate(batch["index"].numpy()):
                losses[int(idx)] = float(ce[i])
    n = max(losses) + 1
    out = np.zeros(n, dtype=np.float32)
    for k, v in losses.items():
        out[k] = v
    return out


def _make_loader(base_dataset, idx, batch_size, num_workers, pin_memory):
    """DataLoader over a subset of base_dataset specified by absolute indices."""
    sub = Subset(base_dataset, idx.tolist())
    return DataLoader(sub, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=pin_memory)


def _inner_train(model, loader, device, lr, weight_decay, inner_steps):
    """K SGD steps from `model` on `loader` (cycled if shorter than K).

    Returns the modified model in place plus the optimizer used.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    model.train()
    it = iter(loader)
    for _ in range(inner_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        x = _move_x(batch["image"], device)
        y = batch["label"].to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


def _reptile_mean_update(theta_state_dict, *inner_state_dicts, alpha):
    """Reptile-mean: θ ← θ + α · mean(θ_k − θ)."""
    new = {}
    K = len(inner_state_dicts)
    for k in theta_state_dict:
        delta = sum(sd[k] - theta_state_dict[k] for sd in inner_state_dicts) / K
        new[k] = theta_state_dict[k] + alpha * delta
    return new


def _anchor_consistency(model_hard, model_easy, anchor_loader, device):
    """sym-KL between two models' predictions on the anchor loader (in nats)."""
    model_hard.eval(); model_easy.eval()
    accum = 0.0
    n = 0
    with torch.no_grad():
        for batch in anchor_loader:
            x = _move_x(batch["image"], device)
            pH = F.softmax(model_hard(x), dim=1)
            pE = F.softmax(model_easy(x), dim=1)
            eps = 1e-12
            kl = 0.5 * (
                (pH * (torch.log(pH + eps) - torch.log(pE + eps))).sum(-1) +
                (pE * (torch.log(pE + eps) - torch.log(pH + eps))).sum(-1))
            accum += float(kl.sum())
            n += kl.numel()
    return accum / max(n, 1)


def _warmup_erm(cfg, loaders, device, model_seed, epochs):
    """Standard ERM training to a competent θ before meta-tuning."""
    set_seed(model_seed)
    model = _build_model(cfg, loaders, device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.training.lr,
                                  weight_decay=cfg.training.weight_decay)
    for epoch in range(epochs):
        model.train()
        for batch in loaders["train"]:
            x = _move_x(batch["image"], device)
            y = batch["label"].to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def train_meta(cfg, loaders, device, data_seed, model_seed,
               meta_steps, inner_steps, alpha, anchor_frac, hard_frac,
               consistency_weight, warmup_epochs=0, log_every=10):
    """Hard-subset Reptile training. Returns final θ (a model).

    If warmup_epochs > 0, first runs standard ERM for warmup_epochs to
    obtain a competent starting point, then meta-tunes from there.
    """
    if warmup_epochs > 0:
        theta = _warmup_erm(cfg, loaders, device, model_seed, warmup_epochs)
    else:
        set_seed(model_seed)
        theta = _build_model(cfg, loaders, device)

    base_ds = loaders["train"].dataset
    n_train = len(base_ds)
    bs = cfg.training.batch_size
    nw = loaders["train"].num_workers
    pm = loaders["train"].pin_memory

    # Held-out anchor set (used for consistency loss + diagnostic).
    rng = np.random.default_rng(data_seed)
    perm = rng.permutation(n_train)
    n_anchor = max(1, int(anchor_frac * n_train))
    anchor_idx = perm[:n_anchor]
    pool_idx = perm[n_anchor:]
    anchor_loader = _make_loader(base_ds, anchor_idx, bs, nw, pm)

    # Loader spanning the *pool* (used for per-example loss diagnostic).
    pool_loader = DataLoader(Subset(base_ds, pool_idx.tolist()),
                             batch_size=bs, shuffle=False,
                             num_workers=nw, pin_memory=pm)

    log_lines = []
    for step in range(meta_steps):
        # 1. Per-example losses on the pool from θ.
        ce = _per_example_loss(theta, pool_loader, device)  # length n_train
        # Restrict to pool indices and rank.
        pool_ce = ce[pool_idx]
        order = np.argsort(-pool_ce)   # descending
        n_hard = max(1, int(hard_frac * len(pool_idx)))
        hard_idx = pool_idx[order[:n_hard]]
        easy_idx = pool_idx[order[n_hard:]]

        # 2. Inner train two snapshots.
        model_hard = deepcopy(theta)
        loader_hard = _make_loader(base_ds, hard_idx, bs, nw, pm)
        _inner_train(model_hard, loader_hard, device,
                     cfg.training.lr, cfg.training.weight_decay, inner_steps)

        model_easy = deepcopy(theta)
        loader_easy = _make_loader(base_ds, easy_idx, bs, nw, pm)
        _inner_train(model_easy, loader_easy, device,
                     cfg.training.lr, cfg.training.weight_decay, inner_steps)

        # 3. Reptile-mean update.
        theta_sd = theta.state_dict()
        new_sd = _reptile_mean_update(
            theta_sd, model_hard.state_dict(), model_easy.state_dict(),
            alpha=alpha)

        # 4. Optional consistency-weighted extra step toward midpoint.
        if consistency_weight > 0:
            cons = _anchor_consistency(model_hard, model_easy, anchor_loader,
                                       device)
            for k in new_sd:
                mid = 0.5 * (model_hard.state_dict()[k]
                             + model_easy.state_dict()[k])
                new_sd[k] = new_sd[k] + consistency_weight * cons * (mid - new_sd[k])
        else:
            cons = float("nan")

        theta.load_state_dict(new_sd)

        if step % log_every == 0 or step == meta_steps - 1:
            log_lines.append(
                f"  meta-step {step:4d}  anchor_cons={cons:.4f}  "
                f"|hard|={len(hard_idx)} |easy|={len(easy_idx)}")
    return theta, "\n".join(log_lines)


def run(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu")
    out_root = Path(args.output_dir) / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args.dataset)
    if args.weight_decay is not None:
        cfg.training.weight_decay = args.weight_decay

    data_seeds = [int(s) for s in args.data_seeds.split(",")]
    for data_seed in data_seeds:
        cfg.training.seed = data_seed
        set_seed(data_seed)
        loaders = make_dataloaders(cfg)
        print(f"\n=== meta {args.dataset}  data_seed={data_seed}  "
              f"meta_steps={args.meta_steps}  inner={args.inner_steps}  "
              f"alpha={args.alpha}  cw={args.consistency_weight} ===")

        model, log = train_meta(
            cfg, loaders, device, data_seed,
            model_seed=data_seed * 1000,
            meta_steps=args.meta_steps,
            inner_steps=args.inner_steps,
            alpha=args.alpha,
            anchor_frac=args.anchor_frac,
            hard_frac=args.hard_frac,
            consistency_weight=args.consistency_weight,
            warmup_epochs=args.warmup_epochs)
        print(log)

        id_p, id_y, id_i = _predict(model, loaders["id_test"], device)
        ood_p, ood_y, ood_i = _predict(model, loaders["ood_test"], device)
        id_acc = float((id_p.argmax(1) == id_y).mean())
        ood_acc = float((ood_p.argmax(1) == ood_y).mean())
        print(f"  id_acc={id_acc:.4f}  ood_acc={ood_acc:.4f}")

        np.savez_compressed(
            out_root / f"meta_seed{data_seed}_a{args.alpha}_cw{args.consistency_weight}.npz",
            id_probs=id_p, id_labels=id_y, id_indices=id_i,
            ood_probs=ood_p, ood_labels=ood_y, ood_indices=ood_i,
            data_seed=data_seed, alpha=args.alpha,
            consistency_weight=args.consistency_weight,
            meta_steps=args.meta_steps, inner_steps=args.inner_steps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HPARAMS.keys()))
    ap.add_argument("--data_seeds", default="42,123,789")
    ap.add_argument("--meta_steps", type=int, default=200)
    ap.add_argument("--inner_steps", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="Reptile step size in [0,1].")
    ap.add_argument("--anchor_frac", type=float, default=0.2,
                    help="Fraction of training set held out as anchor.")
    ap.add_argument("--hard_frac", type=float, default=0.5,
                    help="Fraction of pool labelled `hard` (top-loss).")
    ap.add_argument("--consistency_weight", type=float, default=0.0,
                    help="Extra weight on midpoint pull, scaled by anchor cons.")
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--warmup_epochs", type=int, default=0,
                    help="ERM epochs before meta-tuning starts.")
    ap.add_argument("--output_dir", default="outputs/fragility_meta")
    args = ap.parse_args()
    run(args)
