"""Evaluate uncertainty signals on saved test predictions.

Consumes NPZ files produced by partition_pair_train.py and computes three
uncertainty signals + their downstream metrics:

Uncertainty signals (per test example):
  - softmax_entropy: H(p) from a single full-data ERM model.
  - ensemble_var:    mean pairwise JS divergence across K full-data models
                     that differ only in init seed.
  - partition_disagree: mean pairwise symmetric KL across K models trained on
                     *disjoint* partitions of the training data (our signal).

Downstream metrics:
  - selective_auc(id | ood): area under risk-coverage curve. Lower is better.
  - ood_detection_auroc:    AUROC treating OOD examples as positives and
                            id examples as negatives, uncertainty score as
                            decision function.
  - flip_auc:               on examples that appear in both id_test and
                            ood_test (matched by `index`), AUC for the binary
                            classifier 'uncertainty score predicts which
                            examples flip prediction under shift.' CMNIST has
                            no such overlap -> skipped when empty.

The go/no-go question for the pivot: does `partition_disagree` decorrelate
meaningfully from `ensemble_var`? If Pearson-r > 0.95 we are not capturing
anything new.
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def _entropy(p, eps=1e-12):
    return -(p * np.log(p + eps)).sum(axis=-1)


def _kl(p, q, eps=1e-12):
    return (p * (np.log(p + eps) - np.log(q + eps))).sum(axis=-1)


def _sym_kl(p, q):
    return 0.5 * (_kl(p, q) + _kl(q, p))


def _js(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (_kl(p, m) + _kl(q, m))


def _mean_pairwise(probs_list, fn):
    """probs_list: list of (N, C). fn(p_a, p_b) -> (N,). Returns (N,)."""
    n = len(probs_list)
    if n < 2:
        return np.zeros(probs_list[0].shape[0])
    acc, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            acc = acc + fn(probs_list[i], probs_list[j])
            count += 1
    return acc / count


def _selective_auc(confidence, correct):
    """AUC of the risk-coverage curve (lower is better).

    Sorts by confidence descending; at coverage c (top c%), computes
    empirical error; integrates error vs coverage.
    """
    order = np.argsort(-confidence)
    correct_sorted = correct[order]
    n = len(correct_sorted)
    coverage = np.arange(1, n + 1) / n
    risk = 1 - np.cumsum(correct_sorted) / np.arange(1, n + 1)
    return float(np.trapezoid(risk, coverage))


def _load_predictions(root, dataset, mode, seed):
    """Load all NPZ files matching mode/seed; returns list of dicts in k order."""
    d = Path(root) / dataset
    files = sorted(d.glob(f"{mode}_seed{seed}_k*.npz"),
                   key=lambda p: int(p.stem.split("_k")[-1]))
    return [dict(np.load(f, allow_pickle=True)) for f in files]


def evaluate(root, dataset, seed):
    erm = _load_predictions(root, dataset, "erm", seed)
    ens = _load_predictions(root, dataset, "ensemble", seed)
    par = _load_predictions(root, dataset, "partition", seed)

    assert len(erm) >= 1, "need at least one ERM model"
    assert len(ens) >= 2, "need >=2 ensemble members"
    assert len(par) >= 2, "need >=2 partition models"

    # Base predictor for accuracy metrics: the single full-data ERM model.
    id_p_erm = erm[0]["id_probs"]
    ood_p_erm = erm[0]["ood_probs"]
    id_y = erm[0]["id_labels"]
    ood_y = erm[0]["ood_labels"]

    signals = {}

    # 1. softmax entropy from ERM
    signals["softmax_entropy"] = {
        "id": _entropy(id_p_erm),
        "ood": _entropy(ood_p_erm),
    }

    # 2. deep ensemble disagreement (JS between members)
    signals["ensemble_var"] = {
        "id":  _mean_pairwise([m["id_probs"] for m in ens], _js),
        "ood": _mean_pairwise([m["ood_probs"] for m in ens], _js),
    }

    # 3. partition-pair disagreement (symmetric KL between members)
    signals["partition_disagree"] = {
        "id":  _mean_pairwise([m["id_probs"] for m in par], _sym_kl),
        "ood": _mean_pairwise([m["ood_probs"] for m in par], _sym_kl),
    }

    # Correctness for selective prediction uses the ERM prediction.
    id_correct = (id_p_erm.argmax(1) == id_y).astype(float)
    ood_correct = (ood_p_erm.argmax(1) == ood_y).astype(float)

    # OOD detection: labels 0/1 for id/ood; uncertainty score = signal.
    # Higher uncertainty on OOD -> score is positive on OOD examples.
    ood_labels = np.concatenate([np.zeros(len(id_y)), np.ones(len(ood_y))])

    rows = []
    for name, sig in signals.items():
        sel_auc_id = _selective_auc(-sig["id"], id_correct)   # lower unc = higher confidence
        sel_auc_ood = _selective_auc(-sig["ood"], ood_correct)
        ood_scores = np.concatenate([sig["id"], sig["ood"]])
        try:
            ood_auroc = float(roc_auc_score(ood_labels, ood_scores))
        except ValueError:
            ood_auroc = float("nan")
        rows.append({
            "signal": name,
            "selective_auc_id": sel_auc_id,
            "selective_auc_ood": sel_auc_ood,
            "ood_detection_auroc": ood_auroc,
        })

    # Cross-signal decorrelation: r(partition_disagree, ensemble_var) on OOD.
    pd, ev = signals["partition_disagree"]["ood"], signals["ensemble_var"]["ood"]
    pd_ev_r = float(np.corrcoef(pd, ev)[0, 1])
    pd_ent = float(np.corrcoef(signals["partition_disagree"]["ood"],
                               signals["softmax_entropy"]["ood"])[0, 1])

    return rows, {"r_partition_ensemble_ood": pd_ev_r,
                  "r_partition_entropy_ood": pd_ent,
                  "id_acc": float(id_correct.mean()),
                  "ood_acc": float(ood_correct.mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--root", default="outputs/stability")
    args = ap.parse_args()

    rows, meta = evaluate(args.root, args.dataset, args.seed)
    print(f"\n=== {args.dataset} | seed {args.seed} ===")
    print(f"id_acc={meta['id_acc']:.4f}  ood_acc={meta['ood_acc']:.4f}")
    print(f"\n{'signal':20s}  {'sel_auc_id':>11s}  {'sel_auc_ood':>12s}  "
          f"{'ood_auroc':>10s}")
    for r in rows:
        print(f"{r['signal']:20s}  {r['selective_auc_id']:>11.4f}  "
              f"{r['selective_auc_ood']:>12.4f}  {r['ood_detection_auroc']:>10.4f}")
    print(f"\nPearson r:")
    print(f"  partition vs ensemble (ood): {meta['r_partition_ensemble_ood']:.3f}")
    print(f"  partition vs entropy  (ood): {meta['r_partition_entropy_ood']:.3f}")
    print(f"\nGo/no-go: if r(partition, ensemble) > 0.95 on OOD, "
          f"partition signal may be redundant with deep ensembles.")
