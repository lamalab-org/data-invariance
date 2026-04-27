"""Paper Figure (rendered as Fig 2): top-decile fragility predicts churn.

For every test example, compute (i) average cross-bootstrap fragility
(mean sym-KL across all pairs of ERM bootstraps that disagree on the
example) and (ii) average flip rate (fraction of bootstrap pairs where
the argmax predictions disagree).  Rank test examples by their fragility
and split into deciles; report the mean flip rate per decile.

Single claim: fragility identifies which predictions will flip on
retraining.  In every dataset the top-10% most fragile examples flip
at a high rate; the bottom 10% rarely flip.

Practitioner take: a single additional bootstrap suffices to flag the
fragile decile for re-review, without a full retraining cascade.

Visual design (line plot)
-------------------------
Eight datasets as eight lines over the 10 deciles.  BACE (the dev
dataset) is highlighted in red; the seven held-out datasets are muted
grey lines.  The "top decile shoots up" pattern is then a single
visual gesture across the curves.  Legend below the panel so it never
overlaps the data; figure is sized for a NeurIPS column.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np

from _analysis_lib import load_runs, sym_kl
from paper_constants import DEV_DATASET, HEADLINE_DATASETS, display


def _per_example_fragility_and_flips(ds_dir: Path):
    runs = load_runs(ds_dir, "erm_train*.npz")
    if len(runs) < 2:
        return None
    probs = [d.get("id_probs_avg", d.get("id_probs")) for _, d in runs]
    n_test = probs[0].shape[0]
    frag_acc = np.zeros(n_test)
    flip_acc = np.zeros(n_test)
    n_pairs = 0
    for i, j in combinations(range(len(probs)), 2):
        frag_acc += sym_kl(probs[i], probs[j])
        flip_acc += (probs[i].argmax(1) != probs[j].argmax(1)).astype(float)
        n_pairs += 1
    return frag_acc / n_pairs, flip_acc / n_pairs


def _decile_flip_rates(fragility, flips, n_bins=10):
    order = np.argsort(fragility)
    n = len(fragility)
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    return np.array([flips[order[edges[k]:edges[k + 1]]].mean()
                     for k in range(n_bins)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/cross_sample")
    ap.add_argument("--out", default="paper/figures/fig3_decile.pdf")
    args = ap.parse_args()
    root = Path(args.root)

    datasets = [DEV_DATASET] + HEADLINE_DATASETS
    rates = {}
    for ds in datasets:
        out = _per_example_fragility_and_flips(root / ds)
        if out is None:
            continue
        rates[ds] = _decile_flip_rates(*out)

    lama_aesthetics.get_style("main")
    plt.rcParams["axes.unicode_minus"] = False

    # Two-column-wide × short banner layout: legend below the panel never
    # overlaps the data and never collides with surrounding LaTeX text.
    fig_w = lama_aesthetics.TWO_COL_WIDTH * 0.55
    fig_h = fig_w * 0.62
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    n_bins = 10
    x = np.arange(1, n_bins + 1)
    held_out = [ds for ds in rates if ds != DEV_DATASET]

    grey = "0.55"
    for ds in held_out:
        ax.plot(x, rates[ds] * 100, color=grey, linewidth=1.0,
                marker="o", markersize=3.0, alpha=0.85, zorder=2)
    if DEV_DATASET in rates:
        ax.plot(x, rates[DEV_DATASET] * 100, color="#d62728", linewidth=1.8,
                marker="o", markersize=5.0, zorder=3,
                label=f"{display(DEV_DATASET)} (dev)")

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in x], fontsize=8)
    ax.set_xlabel("Fragility decile (1 = least fragile, 10 = most)")
    ax.set_ylabel("Argmax-flip rate per pair (%)")
    ax.set_xlim(0.5, n_bins + 0.5)
    ax.set_ylim(0, 50)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.4, zorder=0)

    handles = [
        plt.Line2D([0], [0], color="#d62728", linewidth=1.8, marker="o",
                   markersize=5.0, label=f"{display(DEV_DATASET)} (dev)"),
        plt.Line2D([0], [0], color=grey, linewidth=1.0, marker="o",
                   markersize=3.0,
                   label="held-out: " + ", ".join(display(d) for d in held_out)),
    ]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=1,
              frameon=False, fontsize=7.5, handletextpad=0.5)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}")
    print()
    print(f"Top-decile / bottom-decile flip rates (%):")
    print(f"{'dataset':16s}  {'bottom (1)':>12s}  {'top (10)':>10s}  {'ratio':>8s}")
    for ds, dr in rates.items():
        ratio = dr[-1] / dr[0] if dr[0] > 0 else float("inf")
        print(f"{ds:16s}  {dr[0]*100:>12.2f}  {dr[-1]*100:>10.2f}  "
              f"{ratio:>8.1f}x")


if __name__ == "__main__":
    main()
