"""Convergence of the top-30% triage recall as a function of K.

The body claims: a single extra retraining (K=2 total ERM models)
suffices for triage.  This script verifies the claim by computing
top-30% cumulative flip-mass recall when fragility is estimated from
K randomly-sampled bootstraps out of the 10 we trained, for
K \\in {2, 3, 5, 10}.  For each K < 10 we draw 30 random K-subsets
and report mean and bootstrap-CI recall.

Output: paper/figures/fig_convergence.pdf and a CSV with the
per-dataset numbers.
"""
from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np

from _analysis_lib import load_runs, sym_kl
from paper_constants import DEV_DATASET, HEADLINE_DATASETS, display


N_TRIALS = 30
K_VALUES = [2, 3, 5, 10]


def _gold_flip_mass(probs):
    """Per-example expected flip rate over all C(K,2) pairs of probs."""
    n = probs[0].shape[0]
    flip_acc = np.zeros(n)
    npairs = 0
    for i, j in combinations(range(len(probs)), 2):
        flip_acc += (probs[i].argmax(1) != probs[j].argmax(1)).astype(float)
        npairs += 1
    return flip_acc / npairs


def _fragility_from_subset(probs, idx):
    """sym-KL based fragility from the subset of probs at indices idx."""
    sub = [probs[i] for i in idx]
    n = sub[0].shape[0]
    if len(sub) < 2:
        # K=1 has no pairs --- use predicted probability of argmax as
        # a degenerate score (max prob - second max), so K=1 is a
        # well-defined baseline (single-model entropy-ish).
        p = sub[0]
        sorted_p = np.sort(p, axis=1)
        return -(sorted_p[:, -1] - sorted_p[:, -2])
    frag = np.zeros(n)
    npairs = 0
    for i, j in combinations(range(len(sub)), 2):
        frag += sym_kl(sub[i], sub[j])
        npairs += 1
    return frag / npairs


def _topK_recall(score, gold, top_frac=0.3):
    n = len(gold)
    ntop = max(1, int(n * top_frac))
    order = np.argsort(-score)
    total = gold.sum()
    return float(gold[order[:ntop]].sum() / total) if total > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/cross_sample")
    ap.add_argument("--out_pdf", default="paper/figures/fig_convergence.pdf")
    ap.add_argument("--out_csv", default="outputs/convergence_recall.csv")
    args = ap.parse_args()
    root = Path(args.root)

    rng = np.random.default_rng(seed=99)
    datasets = [DEV_DATASET] + HEADLINE_DATASETS

    results = {}  # ds -> { K -> [trial recalls] }
    for ds in datasets:
        runs = load_runs(root / ds, "erm_train*.npz")
        if len(runs) < 10:
            continue
        probs = [d.get("id_probs_avg", d.get("id_probs")) for _, d in runs]
        gold = _gold_flip_mass(probs)

        per_K = {}
        for K in K_VALUES:
            recalls = []
            if K >= 10:
                # Only one subset (the full set) for K=10
                idx = list(range(10))
                score = _fragility_from_subset(probs, idx)
                recalls.append(_topK_recall(score, gold))
            else:
                for _ in range(N_TRIALS):
                    idx = list(rng.choice(10, size=K, replace=False))
                    score = _fragility_from_subset(probs, idx)
                    recalls.append(_topK_recall(score, gold))
            per_K[K] = np.array(recalls)
        results[ds] = per_K

    # CSV: per-dataset, per-K mean + CI
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "K", "mean_recall", "lo", "hi", "n_trials"])
        for ds, per_K in results.items():
            for K, rs in per_K.items():
                if len(rs) == 1:
                    w.writerow([ds, K, float(rs[0]), float(rs[0]),
                                float(rs[0]), 1])
                else:
                    bs_means = [rs[rng.integers(len(rs), size=len(rs))].mean()
                                for _ in range(2000)]
                    lo, hi = np.percentile(bs_means, [2.5, 97.5])
                    w.writerow([ds, K, float(rs.mean()), float(lo),
                                float(hi), len(rs)])
    print(f"Wrote {args.out_csv}")

    # Plot
    lama_aesthetics.get_style("main")
    fig_w = lama_aesthetics.TWO_COL_WIDTH * 0.55
    fig_h = fig_w * 0.7
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    held_out = [ds for ds in results if ds != DEV_DATASET]
    grey = "0.55"
    for ds in held_out:
        per_K = results[ds]
        Ks = sorted(per_K.keys())
        means = [per_K[K].mean() * 100 for K in Ks]
        ax.plot(Ks, means, color=grey, linewidth=1.0,
                marker="o", markersize=3.0, alpha=0.85, zorder=2)
    if DEV_DATASET in results:
        per_K = results[DEV_DATASET]
        Ks = sorted(per_K.keys())
        means = [per_K[K].mean() * 100 for K in Ks]
        ax.plot(Ks, means, color="#d62728", linewidth=1.8,
                marker="o", markersize=5.0, zorder=3,
                label=f"{display(DEV_DATASET)} (dev)")

    ax.set_xlabel("ERM bootstraps used to score fragility (K)")
    ax.set_ylabel("Top-30% cumulative recall (%)")
    ax.set_xticks(K_VALUES)
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(linestyle="-", linewidth=0.4, alpha=0.35, zorder=0)

    handles = [
        plt.Line2D([0], [0], color="#d62728", linewidth=1.8, marker="o",
                   markersize=5.0, label=f"{display(DEV_DATASET)} (dev)"),
        plt.Line2D([0], [0], color=grey, linewidth=1.0, marker="o",
                   markersize=3.0,
                   label="held-out: "
                         + ", ".join(display(d) for d in held_out)),
    ]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=1,
              frameon=False, fontsize=7.5, handletextpad=0.5)

    Path(args.out_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_pdf, bbox_inches="tight")
    print(f"Wrote {args.out_pdf}")

    print()
    print(f"Top-30% recall by K (averaged over {N_TRIALS} subsets per K<10):")
    print(f"{'dataset':16s}  " + "  ".join([f'K={K:>2}' for K in K_VALUES]))
    for ds, per_K in results.items():
        cells = [f"{per_K[K].mean()*100:>4.1f}%" for K in sorted(per_K)]
        print(f"  {ds:14s}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
