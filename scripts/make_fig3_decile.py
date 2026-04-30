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


def _cumulative_recall(fragility, flips, n_points=101):
    """Cumulative flip-mass recall as a function of review fraction.

    Returns x (review fraction in [0, 1]) and y (cumulative recall in
    [0, 1]) — what fraction of all flip-mass is captured if you sort
    examples by fragility and review the top fraction.
    """
    order = np.argsort(-fragility)
    sorted_flips = flips[order]
    cum = np.cumsum(sorted_flips)
    total = cum[-1] if cum[-1] > 0 else 1.0
    n = len(flips)
    xs = np.linspace(0, 1, n_points)
    ys = np.array([cum[max(0, int(x * n) - 1)] / total if x > 0 else 0.0
                   for x in xs])
    return xs, ys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/cross_sample")
    ap.add_argument("--out", default="paper/figures/fig3_decile.pdf")
    args = ap.parse_args()
    root = Path(args.root)

    datasets = [DEV_DATASET] + HEADLINE_DATASETS
    raw = {}
    for ds in datasets:
        out = _per_example_fragility_and_flips(root / ds)
        if out is None:
            continue
        raw[ds] = out  # (fragility, flips)

    lama_aesthetics.get_style("main")
    plt.rcParams["axes.unicode_minus"] = False

    fig_w = lama_aesthetics.TWO_COL_WIDTH * 0.55
    fig_h = fig_w * 0.7
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    held_out = [ds for ds in raw if ds != DEV_DATASET]
    grey = "0.55"

    # Diagonal: random-ranking baseline.
    ax.plot([0, 100], [0, 100], color="0.7", linewidth=0.7,
            linestyle=":", zorder=1, label="random ranking")

    # Triage band: shade x in [0, 30] to highlight the practical
    # review fraction.
    ax.axvspan(0, 30, color="#d62728", alpha=0.10, zorder=1)

    # Held-out curves in grey.
    for ds in held_out:
        xs, ys = _cumulative_recall(*raw[ds])
        ax.plot(xs * 100, ys * 100, color=grey, linewidth=1.0,
                alpha=0.85, zorder=2)

    # Dev curve in red.
    if DEV_DATASET in raw:
        xs, ys = _cumulative_recall(*raw[DEV_DATASET])
        ax.plot(xs * 100, ys * 100, color="#d62728", linewidth=1.8,
                zorder=3)

    # Per-dataset recall at top-30%.
    captures_30 = []
    for ds, (frag, flips) in raw.items():
        n = len(flips)
        order = np.argsort(-frag)
        ntop = max(1, int(n * 0.3))
        if flips.sum() > 0:
            captures_30.append(flips[order[:ntop]].sum() / flips.sum())
    cap_lo, cap_hi = min(captures_30) * 100, max(captures_30) * 100

    # Reference lines at 30% review.
    ax.axvline(30, color="#7a1313", linewidth=0.7, linestyle="--",
               alpha=0.5, zorder=1)

    # Annotation for the top-30% claim.
    ax.text(
        32, 35,
        f"Review top 30%:\ncatch {cap_lo:.0f}\u2013{cap_hi:.0f}% of flips",
        fontsize=8.0, color="0.15", ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="white",
                  ec="#7a1313", lw=0.6),
        zorder=4,
    )

    ax.set_xlabel("Test predictions reviewed, ranked by fragility (%)")
    ax.set_ylabel("Cumulative % of flip-mass captured")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(linestyle="-", linewidth=0.4, alpha=0.35, zorder=0)

    handles = [
        plt.Line2D([0], [0], color="#d62728", linewidth=1.8,
                   label=f"{display(DEV_DATASET)} (dev)"),
        plt.Line2D([0], [0], color=grey, linewidth=1.0,
                   label="held-out: "
                         + ", ".join(display(d) for d in held_out)),
        plt.Line2D([0], [0], color="0.7", linewidth=0.7, linestyle=":",
                   label="random ranking"),
    ]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=1,
              frameon=False, fontsize=7.5, handletextpad=0.5)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}")
    print()
    print(f"Top-30% cumulative recall (%):")
    for ds, (frag, flips) in raw.items():
        n = len(flips)
        order = np.argsort(-frag)
        ntop = max(1, int(n * 0.3))
        if flips.sum() > 0:
            cap = flips[order[:ntop]].sum() / flips.sum() * 100
            print(f"  {ds:16s}  {cap:>5.1f}%")


if __name__ == "__main__":
    main()
