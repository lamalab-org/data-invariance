"""Train a twin-network model with a fragility-regularization objective.

Two networks (θ_A, θ_B) are trained jointly on a fixed 50/50 random
partition of the training data:
  - θ_A only sees gradients from CE on partition_A examples
  - θ_B only sees gradients from CE on partition_B examples
  - Both networks see a symmetric-KL consistency loss on EVERY batch example
    (both partitions contribute consistency, but only their own partition
     contributes CE).

L = L_CE_A + L_CE_B + λ · L_consistency

At λ=0 this is two independent partition-pair models (the fragility we
measure). At λ→∞ this collapses to a single solution that disregards the
partition (ERM-like). The interesting regime is in between: the network
finds a decision boundary insensitive to the partition.

We save the two networks' predictions on id_test and ood_test plus the
averaged prediction (the deployment-time output). This lets us compute
both within-twin fragility (should ↓ with λ) and cross-sample churn
(deployment-time fragility — the quantity that matters).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_experiment import HPARAMS, _build_model, build_cfg  # noqa: E402
from train import evaluate, make_dataloaders  # noqa: E402
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


def _step_twin(model_A, model_B, x, y, mask_A, mask_B, lam):
    """One step. mask_A/B are 0/1 floats marking partition membership."""
    logits_A = model_A(x)
    logits_B = model_B(x)
    p_A = F.softmax(logits_A, dim=1)
    p_B = F.softmax(logits_B, dim=1)

    # Per-example CE; only count partition-owned examples.
    ce_A = F.cross_entropy(logits_A, y, reduction="none")
    ce_B = F.cross_entropy(logits_B, y, reduction="none")
    n_a, n_b = mask_A.sum().clamp_min(1), mask_B.sum().clamp_min(1)
    loss_ce = (ce_A * mask_A).sum() / n_a + (ce_B * mask_B).sum() / n_b

    # Symmetric KL on EVERY example (consistency).
    eps = 1e-12
    kl_AB = (p_A * (torch.log(p_A + eps) - torch.log(p_B + eps))).sum(-1)
    kl_BA = (p_B * (torch.log(p_B + eps) - torch.log(p_A + eps))).sum(-1)
    loss_consistency = 0.5 * (kl_AB + kl_BA).mean()

    return loss_ce + lam * loss_consistency, loss_consistency.detach()


def _make_partition_masks(n_train, partition_mode, epoch, data_seed):
    """Return a length-n_train float array `in_A` (1=A, 0=B) for this epoch.

    Modes:
      fixed:     same partition every epoch (uses data_seed only).
      per_epoch: re-randomize the 50/50 partition each epoch.
      bootstrap: each "side" is a bootstrap (sample N with replacement);
                 represented here by per-example weights; we still use a
                 binary mask of which side each example contributes to —
                 the bootstrap weighting is applied via duplicate indices
                 in the loader, which is hard, so we approximate with a
                 random per-example weight in [0,1] each epoch (equivalent
                 to expectation of bootstrap mass on this example).
    Returns (in_A_float, in_B_float) tensors-on-host.
    """
    rng = np.random.default_rng(hash((data_seed, partition_mode, epoch))
                                & 0xFFFFFFFF)
    if partition_mode == "fixed":
        rng_fixed = np.random.default_rng(data_seed)
        order = rng_fixed.permutation(n_train)
        in_A = np.zeros(n_train, dtype=np.float32)
        in_A[order[: n_train // 2]] = 1.0
    elif partition_mode == "per_epoch":
        order = rng.permutation(n_train)
        in_A = np.zeros(n_train, dtype=np.float32)
        in_A[order[: n_train // 2]] = 1.0
    elif partition_mode == "bootstrap":
        # Independent bootstrap weights per side.
        wA = np.bincount(rng.integers(0, n_train, size=n_train),
                         minlength=n_train).astype(np.float32)
        wB = np.bincount(rng.integers(0, n_train, size=n_train),
                         minlength=n_train).astype(np.float32)
        # Normalize to mean 1 within each side (so total CE is comparable).
        wA = wA / wA.mean(); wB = wB / wB.mean()
        return wA, wB
    else:
        raise ValueError(partition_mode)
    in_B = 1.0 - in_A
    return in_A, in_B


def train_twin(cfg, loaders, device, data_seed, model_seed, epochs,
               partition_mode, lam, adv_lr=1e-2):
    """Train (θ_A, θ_B) jointly. Returns the two trained models.

    Static modes (fixed/per_epoch/bootstrap) use _make_partition_masks.
    'adversarial' mode learns per-example assignment logits via alternating
    SGD; the adversary maximizes per-example task loss (which drives head
    specialization → larger consistency loss for the model to fight).
    """
    set_seed(model_seed)
    model_A = _build_model(cfg, loaders, device)
    set_seed(model_seed + 1)
    model_B = _build_model(cfg, loaders, device)

    opt_A = torch.optim.AdamW(model_A.parameters(),
                              lr=cfg.training.lr,
                              weight_decay=cfg.training.weight_decay)
    opt_B = torch.optim.AdamW(model_B.parameters(),
                              lr=cfg.training.lr,
                              weight_decay=cfg.training.weight_decay)

    n_train = len(loaders["train"].dataset)
    if partition_mode == "adversarial":
        # Per-example logits, sigmoid → soft assignment ∈ (0,1).
        assignment_logits = torch.zeros(n_train, device=device,
                                        requires_grad=True)
        opt_adv = torch.optim.SGD([assignment_logits], lr=adv_lr)
    else:
        assignment_logits = None
        opt_adv = None

    for epoch in range(epochs):
        model_A.train(); model_B.train()
        if partition_mode != "adversarial":
            wA_np, wB_np = _make_partition_masks(
                n_train, partition_mode, epoch, data_seed)

        for batch in loaders["train"]:
            x = _move_x(batch["image"], device)
            y = batch["label"].to(device)
            idx_t = batch["index"].to(device)
            idx_np = batch["index"].cpu().numpy()

            if partition_mode == "adversarial":
                s = torch.sigmoid(assignment_logits[idx_t])
                mask_A = s
                mask_B = 1.0 - s
            else:
                mask_A = torch.tensor(wA_np[idx_np], dtype=torch.float32,
                                      device=device)
                mask_B = torch.tensor(wB_np[idx_np], dtype=torch.float32,
                                      device=device)

            # ---- Model step ----
            loss, _ = _step_twin(model_A, model_B, x, y,
                                 mask_A, mask_B, lam)
            opt_A.zero_grad(); opt_B.zero_grad()
            if opt_adv is not None:
                opt_adv.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_A.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_B.parameters(), 1.0)
            opt_A.step(); opt_B.step()

            # ---- Adversary step ----
            if partition_mode == "adversarial":
                # Re-forward with no_grad on models, grad on assignment.
                with torch.no_grad():
                    logits_A = model_A(x)
                    logits_B = model_B(x)
                ce_A = F.cross_entropy(logits_A, y, reduction="none")
                ce_B = F.cross_entropy(logits_B, y, reduction="none")
                s = torch.sigmoid(assignment_logits[idx_t])
                # Adversary MAXIMIZES task loss (= MINIMIZES negation).
                adv_loss = -((s * ce_A + (1.0 - s) * ce_B).mean())
                opt_adv.zero_grad()
                adv_loss.backward()
                opt_adv.step()
                with torch.no_grad():
                    assignment_logits.clamp_(-5.0, 5.0)
    return model_A, model_B


def run(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu")

    out_root = Path(args.output_dir) / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args.dataset)
    if args.weight_decay is not None:
        cfg.training.weight_decay = args.weight_decay
    epochs = HPARAMS[args.dataset]["epochs"]

    data_seeds = [int(s) for s in args.data_seeds.split(",")]
    lambdas = [float(L) for L in args.lambdas.split(",")]

    for data_seed in data_seeds:
        cfg.training.seed = data_seed
        set_seed(data_seed)
        loaders = make_dataloaders(cfg)
        n_train = len(loaders["train"].dataset)

        for lam in lambdas:
            model_seed = data_seed * 1000
            print(f"\n=== {args.dataset}  data_seed={data_seed}  λ={lam}  "
                  f"mode={args.partition_mode} ===")
            model_A, model_B = train_twin(
                cfg, loaders, device, data_seed, model_seed, epochs,
                args.partition_mode, lam, adv_lr=args.adv_lr)

            id_pA, id_y, id_i = _predict(model_A, loaders["id_test"], device)
            id_pB, _,    _    = _predict(model_B, loaders["id_test"], device)
            ood_pA, ood_y, ood_i = _predict(model_A, loaders["ood_test"], device)
            ood_pB, _, _ = _predict(model_B, loaders["ood_test"], device)

            id_avg = 0.5 * (id_pA + id_pB)
            ood_avg = 0.5 * (ood_pA + ood_pB)
            within_frag = float(0.5 * (
                (id_pA * (np.log(id_pA + 1e-12) - np.log(id_pB + 1e-12))).sum(-1).mean() +
                (id_pB * (np.log(id_pB + 1e-12) - np.log(id_pA + 1e-12))).sum(-1).mean()))
            id_acc_avg = float((id_avg.argmax(1) == id_y).mean())
            ood_acc_avg = float((ood_avg.argmax(1) == ood_y).mean())
            print(f"  within-twin fragility = {within_frag:.4f}  "
                  f"id_acc_avg = {id_acc_avg:.4f}  ood_acc_avg = {ood_acc_avg:.4f}")

            np.savez_compressed(
                out_root / f"twin_{args.partition_mode}_seed{data_seed}_lam{lam}.npz",
                id_probs_A=id_pA, id_probs_B=id_pB,
                id_probs_avg=id_avg, id_labels=id_y, id_indices=id_i,
                ood_probs_A=ood_pA, ood_probs_B=ood_pB,
                ood_probs_avg=ood_avg, ood_labels=ood_y, ood_indices=ood_i,
                data_seed=data_seed, lam=lam,
                partition_mode=args.partition_mode, mode="twin")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HPARAMS.keys()))
    ap.add_argument("--data_seeds", default="42,123,789")
    ap.add_argument("--lambdas", default="0.0,0.1,1.0,10.0,100.0",
                    help="Comma-separated λ values for the consistency penalty.")
    ap.add_argument("--output_dir", default="outputs/fragility_twin")
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--partition_mode", default="fixed",
                    choices=["fixed", "per_epoch", "bootstrap", "adversarial"],
                    help="fixed: same 50/50 every epoch. per_epoch: refresh "
                         "each epoch. bootstrap: independent N-with-replacement "
                         "samples per side per epoch. adversarial: learnable "
                         "per-example assignment logits; adversary maximizes "
                         "task loss to specialise the heads.")
    ap.add_argument("--adv_lr", type=float, default=1e-2,
                    help="SGD lr for the adversary (assignment_logits).")
    args = ap.parse_args()
    run(args)
