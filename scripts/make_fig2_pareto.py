"""Paper Figure 2: development Pareto curve on BACE.

Single-panel scatter on BACE (the development dataset).
  X axis: cross-sample id-churn (lower is better, the deployment metric)
  Y axis: id accuracy (higher is better)
Each point is one method.  Twin-indep is shown at every $\\lambda$ in the
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
    twin = {lam: _metrics(load_runs(ds_dir, glob_for("Twin-indep", lam=lam)))
            for lam in PARETO_LAMS}
    twin = {lam: m for lam, m in twin.items() if m is not None}

    lama_aesthetics.get_style("main")
    plt.rcParams["axes.unicode_minus"] = False
    fig_w = lama_aesthetics.ONE_COL_WIDTH * 1.4
    fig_h = fig_w * 0.75
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Twin-indep curve (sorted by λ).
    lams = sorted(twin)
    xs = [twin[l][3] * 100 for l in lams]
    ys = [twin[l][0] * 100 for l in lams]
    ax.plot(xs, ys, color="#7a1313", linewidth=1.0, alpha=0.6, zorder=2)

    # Only label the load-bearing points individually: the frozen choice
    # (λ=300) and the knee (λ=100).  The {1, 3, 10, 30} plateau is so
    # tightly clustered that per-point labels overlap; we annotate it
    # once as a group.
    individually_labelled = {100.0: (-50, 4), 300.0: (8, 2)}

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

    # Group label for the {1, 3, 10, 30} plateau, placed below the cluster
    # with a leader line to the cluster centroid.
    cluster_lams = [1.0, 3.0, 10.0, 30.0]
    cx = float(np.mean([twin[l][3] * 100 for l in cluster_lams if l in twin]))
    cy = float(np.mean([twin[l][0] * 100 for l in cluster_lams if l in twin]))
    ax.annotate(r"$\lambda \in \{1, 3, 10, 30\}$",
                xy=(cx, cy), xytext=(cx + 1.6, cy - 0.9),
                fontsize=7, color="0.30",
                arrowprops=dict(arrowstyle="-", color="0.40", linewidth=0.5,
                                shrinkA=0, shrinkB=4))

    # ERM and bagging reference points.
    if erm is not None:
        ax.errorbar(erm[3] * 100, erm[0] * 100,
                    xerr=[[(erm[3] - erm[4]) * 100], [(erm[5] - erm[3]) * 100]],
                    yerr=[[(erm[0] - erm[1]) * 100], [(erm[2] - erm[0]) * 100]],
                    fmt="none", elinewidth=0.8, ecolor="0.40", zorder=2)
        ax.plot(erm[3] * 100, erm[0] * 100, marker="s", linestyle="None",
                ms=6, mfc="white", mec="0.40", zorder=3)
        ax.annotate("ERM", (erm[3] * 100, erm[0] * 100),
                    xytext=(6, -2), textcoords="offset points",
                    fontsize=8, color="0.30")

    if bag is not None:
        ax.errorbar(bag[3] * 100, bag[0] * 100,
                    xerr=[[(bag[3] - bag[4]) * 100], [(bag[5] - bag[3]) * 100]],
                    yerr=[[(bag[0] - bag[1]) * 100], [(bag[2] - bag[0]) * 100]],
                    fmt="none", elinewidth=0.8, ecolor="#0d3a5c", zorder=2)
        ax.plot(bag[3] * 100, bag[0] * 100, marker="o", linestyle="None",
                ms=6, mfc="#1f77b4", mec="#0d3a5c", zorder=3)
        ax.annotate("Bagging $K{=}5$", (bag[3] * 100, bag[0] * 100),
                    xytext=(6, -2), textcoords="offset points",
                    fontsize=8, color="0.30")

    ax.set_xlabel("id-churn (%)")
    ax.set_ylabel("id-accuracy (%)")

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
