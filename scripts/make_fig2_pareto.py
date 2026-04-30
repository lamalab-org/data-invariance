"""Paper Figure 2: development Pareto curve on BACE.

Single-panel scatter on BACE (the development dataset).
  X axis: cross-sample id-churn (lower is better, the deployment metric)
  Y axis: id accuracy (higher is better)
Each point is one method.  Twin-bootstrap is shown at every $\\lambda$ in the
pre-registered grid; the pre-registered selection rule
("largest $\\lambda$ such that id-acc $\\ge$ ERM id-acc $-0.02$") picks
$\\lambda{=}300$, marked with a circle.

Single claim: the frozen-$\\lambda$ rule is not a knife edge.  A plateau
of $\\lambda \\in [10, 300]$ all preserve accuracy while reducing churn;
the rule lands inside that plateau.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import lama_aesthetics

from _analysis_lib import bootstrap_ci, load_runs, pairwise_metrics
from paper_constants import (DEV_DATASET, FROZEN_LAM, PARETO_LAMS, glob_for)


def _metrics(runs):
    if len(runs) < 2:
        return None
    accs = []
    for _, d in runs:
        idp = d.get("id_probs_avg", d.get("id_probs"))
        accs.append(float((idp.argmax(1) == d["id_labels"]).mean()))
    pair_metrics, _ = pairwise_metrics(runs)
    churns = [m["id_churn"] for m in pair_metrics.values()]
    acc_mean, acc_lo, acc_hi = bootstrap_ci(accs)
    ch_mean,  ch_lo,  ch_hi  = bootstrap_ci(churns)
    return (acc_mean, acc_lo, acc_hi, ch_mean, ch_lo, ch_hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/cross_sample")
    ap.add_argument("--out",  default="paper/figures/fig2_pareto.pdf")
    args = ap.parse_args()
    ds_dir = Path(args.root) / DEV_DATASET

    erm = _metrics(load_runs(ds_dir, "erm_train*.npz"))
    bag = _metrics(load_runs(ds_dir, "bagging_train*_K5.npz"))
    twin = {lam: _metrics(load_runs(ds_dir, glob_for("Twin-bootstrap", lam=lam)))
            for lam in PARETO_LAMS}
    twin = {lam: m for lam, m in twin.items() if m is not None}

    lama_aesthetics.get_style("main")
    plt.rcParams["axes.unicode_minus"] = False
    fig_w = lama_aesthetics.ONE_COL_WIDTH * 1.4
    fig_h = fig_w * 0.75
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Twin-bootstrap curve (sorted by λ).
    lams = sorted(twin)
    xs = [twin[l][3] * 100 for l in lams]
    ys = [twin[l][0] * 100 for l in lams]
    ax.plot(xs, ys, color="#7a1313", linewidth=1.0, alpha=0.6, zorder=2)

    # Label only the four trajectory anchors — the cluster {1, 3, 10, 30}
    # sits at near-identical coordinates so labelling each would crowd
    # the bagging-$K{=}5$ marker; λ=1 marks the cluster's rightmost edge,
    # λ=30 its leftmost, λ=100 the knee, λ=300 the rule-selected point.
    # The reader can interpolate λ=3 and λ=10 between λ=1 and λ=30.
    individually_labelled = {
        1.0:   (8, 4),
        30.0:  (-26, -10),
        100.0: (-32, 0),
        300.0: (8, 2),
    }

    for lam in lams:
        m = twin[lam]
        x, y = m[3] * 100, m[0] * 100
        is_frozen = lam == FROZEN_LAM
        ax.errorbar(x, y,
                    xerr=[[(m[3] - m[4]) * 100], [(m[5] - m[3]) * 100]],
                    yerr=[[(m[0] - m[1]) * 100], [(m[2] - m[0]) * 100]],
                    fmt="none", elinewidth=0.7, ecolor="#7a1313", alpha=0.6,
                    zorder=2)
        ax.plot(x, y,
                marker="D",
                ms=8 if is_frozen else 5,
                mfc="#d62728" if is_frozen else "white",
                mec="#7a1313",
                mew=1.5 if is_frozen else 1.0,
                zorder=4 if is_frozen else 3)
        if lam in individually_labelled:
            ax.annotate(rf"$\lambda{{=}}{lam:g}$", (x, y),
                        xytext=individually_labelled[lam],
                        textcoords="offset points",
                        fontsize=7, color="0.30",
                        fontweight="bold" if is_frozen else "normal")

    # ERM and bagging reference points (no inline labels — handled by legend).
    if erm is not None:
        ax.errorbar(erm[3] * 100, erm[0] * 100,
                    xerr=[[(erm[3] - erm[4]) * 100], [(erm[5] - erm[3]) * 100]],
                    yerr=[[(erm[0] - erm[1]) * 100], [(erm[2] - erm[0]) * 100]],
                    fmt="none", elinewidth=0.8, ecolor="0.40", zorder=2)
        ax.plot(erm[3] * 100, erm[0] * 100, marker="s", linestyle="None",
                ms=6, mfc="white", mec="0.40", zorder=3)

    if bag is not None:
        ax.errorbar(bag[3] * 100, bag[0] * 100,
                    xerr=[[(bag[3] - bag[4]) * 100], [(bag[5] - bag[3]) * 100]],
                    yerr=[[(bag[0] - bag[1]) * 100], [(bag[2] - bag[0]) * 100]],
                    fmt="none", elinewidth=0.8, ecolor="#0d3a5c", zorder=2)
        ax.plot(bag[3] * 100, bag[0] * 100, marker="o", linestyle="None",
                ms=6, mfc="#1f77b4", mec="#0d3a5c", zorder=3)

    # Legend — empty upper-right corner of the plot, no overlap with data.
    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="0.40", linestyle="None",
                   ms=6, mfc="white", mec="0.40", label="ERM"),
        plt.Line2D([0], [0], marker="o", color="#0d3a5c", linestyle="None",
                   ms=6, mfc="#1f77b4", mec="#0d3a5c", label="Bagging $K{=}5$"),
        plt.Line2D([0], [0], marker="D", color="#7a1313", linestyle="-",
                   ms=5, mfc="white", mec="#7a1313",
                   label=r"Twin-bootstrap ($\lambda \in \{1,3,10,30,100,300\}$)"),
    ]
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=3,
              frameon=False, fontsize=7.5, handletextpad=0.5,
              columnspacing=1.5)

    ax.set_xlabel("id-churn (%)")
    ax.set_ylabel("id-accuracy (%)")

    # Pad upper ylim so K=5 CI does not clip the plot border.
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo, y_hi + 0.5)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}")
    print(f"\nFrozen choice (BACE rule): λ = {FROZEN_LAM}")
    if FROZEN_LAM in twin:
        m = twin[FROZEN_LAM]
        print(f"  id_acc = {m[0]:.3f}  id_churn = {m[3]:.3f}")
    if erm:
        print(f"ERM:   id_acc = {erm[0]:.3f}  id_churn = {erm[3]:.3f}")


if __name__ == "__main__":
    main()
