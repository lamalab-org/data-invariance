"""Label-noise attribution evaluation via partition-pair sensitivity.

Question: can we identify mislabeled training examples by measuring how much
each example's prediction changes when it is excluded from training?

Setup: K=2 partition pair on CMNIST. For each training example i, record two
predictions on x_i:
  - p_in  = prediction from the model whose training partition CONTAINED i.
  - p_out = prediction from the model whose training partition did NOT
            contain i (so i was held out from this model's training).

Hypothesis: if i is mislabeled (label_noise flipped it), the in-model
overfits to the noisy label while the out-model predicts the true label.
sym_KL(p_in, p_out) should therefore be larger on flipped examples.

Baselines:
  - loss_in : CE loss of p_in against the (possibly flipped) label.
              JTT/LfF-style — high loss is a classical noise signal.
  - entropy_in : softmax entropy of p_in.
  - conf_gap : p_in[true_label] - p_out[true_label] — how much extra the
              in-model confidences the (noisy) label compared to the
              out-model.  This is a direct signed "memorization" signal.

All signals are evaluated by AUROC for 'is this training example mislabeled'.
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from stability_eval import _entropy, _sym_kl  # noqa: E402 — same dir


def _load(root, dataset, seed):
    d = Path(root) / dataset
    files = sorted(d.glob(f"partition_seed{seed}_k*.npz"),
                   key=lambda p: int(p.stem.split("_k")[-1]))
    return [dict(np.load(f, allow_pickle=True)) for f in files]


def evaluate(root, dataset, seed):
    par = _load(root, dataset, seed)
    assert len(par) == 2, f"need exactly 2 partition models, got {len(par)}"
    assert "train_probs" in par[0], "retrain with --save_train_preds"
    assert "train_flipped" in par[0], "dataset did not record flip ground truth"

    N = len(par[0]["train_probs"])
    labels = par[0]["train_labels"].astype(int)   # possibly-flipped labels seen at training
    flipped = par[0]["train_flipped"].astype(bool)  # ground truth: was this flipped?

    part0 = set(par[0]["partition_indices"].tolist())
    part1 = set(par[1]["partition_indices"].tolist())
    assert part0.isdisjoint(part1), "partitions must be disjoint"
    assert len(part0 | part1) == N, "partitions must cover the training set"

    # Build in/out predictions per example.
    # par[k]["train_probs"] is indexed by the training-set index (deterministic
    # order loader), so par[k]["train_probs"][i] is k's prediction on example i
    # regardless of whether i was in k's training partition.
    probs_0 = par[0]["train_probs"]
    probs_1 = par[1]["train_probs"]

    in_from_0 = np.isin(np.arange(N), list(part0))   # True if i was in partition 0's train set
    p_in = np.where(in_from_0[:, None], probs_0, probs_1)
    p_out = np.where(in_from_0[:, None], probs_1, probs_0)

    # --- signals ---
    partition_shift = _sym_kl(p_in, p_out)          # our signal
    loss_in = -np.log(p_in[np.arange(N), labels] + 1e-12)  # CE w.r.t. seen label
    entropy_in = _entropy(p_in)

    # conf_gap: how much more confident is the IN model about the SEEN label
    # than the OUT model?  Positive => in-model overfit the label (suspicious).
    conf_gap = p_in[np.arange(N), labels] - p_out[np.arange(N), labels]

    signals = {
        "partition_shift": partition_shift,
        "loss_in": loss_in,
        "entropy_in": entropy_in,
        "conf_gap": conf_gap,
    }

    # AUROC for detecting flipped examples.
    auc = {name: float(roc_auc_score(flipped.astype(int), s))
           for name, s in signals.items()}
    # For conf_gap, mislabeled examples expected to have HIGHER gap, so direction matches.
    # For loss_in, mislabeled should have HIGHER loss — direction matches.
    # All signals assume "higher = more suspicious."

    # Precision@k: among top-k highest-scoring examples, what fraction are actually flipped?
    k_frac = 0.2
    k = int(N * k_frac)
    precision_at_k = {}
    for name, s in signals.items():
        top_idx = np.argsort(-s)[:k]
        precision_at_k[name] = float(flipped[top_idx].mean())

    base_rate = float(flipped.mean())
    return {
        "N": N,
        "base_flip_rate": base_rate,
        "auroc": auc,
        "precision_at_top20pct": precision_at_k,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cmnist")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default="outputs/stability")
    args = ap.parse_args()

    r = evaluate(args.root, args.dataset, args.seed)
    print(f"\n=== {args.dataset} | seed {args.seed} | "
          f"N={r['N']}  flip_rate={r['base_flip_rate']:.3f} ===\n")
    print(f"{'signal':20s} {'AUROC':>8s}  {'prec@top20%':>12s}")
    for name in r["auroc"]:
        print(f"{name:20s} {r['auroc'][name]:>8.4f}  "
              f"{r['precision_at_top20pct'][name]:>12.4f}")
    print(f"\nRandom baseline:  AUROC=0.5000  prec@top20%={r['base_flip_rate']:.4f}")
