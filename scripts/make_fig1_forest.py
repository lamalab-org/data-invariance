"""Paper Figure 1: per-dataset paired Δ vs ERM on cross-sample id-churn.

One row per held-out dataset (sorted by N, smallest at top to make the
size dimension legible).  For each dataset, four markers stacked
vertically encode {Deep Ens K=5, Bagging K=2, Bagging K=5, Twin-indep};
horizontal error bars show 95% paired-bootstrap CIs across seed pairs.
A vertical line at Δ=0 marks parity with ERM.

The figure's single claim: every method beats ERM, with twin-indep the
largest reduction; the consistency loss adds reliably to bagging at
matched compute (K=2).

Caption first sentence (the conclusion):
    Twin-indep reduces cross-sample prediction churn relative to ERM
    on every chemistry dataset; the consistency loss adds reliably
    to bagging at matched compute.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import lama_aesthetics

from _analysis_lib import bootstrap_paired, load_runs, pairwise_metrics
from paper_constants import (
    FROZEN_LAM, HEADLINE_DATASETS, METHOD_ORDER, N_SEEDS, display, glob_for,
)


def _paired_delta(root: Path, dataset: str, method: str, key: str = "id_churn"):
    """Paired-bootstrap (mean, lo, hi) for (method - ERM) churn on dataset."""
    erm = load_runs(root / dataset, "erm_train*.npz")
    method_runs = load_runs(root / dataset, glob_for(method))
    if len(erm) != N_SEEDS or len(method_runs) != N_SEEDS:
        return None
    ref_pairs, _ = pairwise_metrics(erm)
    m_pairs, _ = pairwise_metrics(method_runs)
    deltas = [m_pairs[p][key] - ref_pairs[p][key]
              for p in sorted(set(ref_pairs).intersection(m_pairs))]
    return bootstrap_paired(deltas)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/cross_sample")
    ap.add_argument("--out", default="paper/figures/fig1_forest.pdf")
    ap.add_argument("--metric", default="id_churn",
                    choices=["id_churn", "ood_churn"])
    args = ap.parse_args()
    root = Path(args.root)

    lama_aesthetics.get_style("main")
    plt.rcParams["axes.unicode_minus"] = False  # CMU Sans Serif lacks U+2212

    # Compute paired deltas: rows = datasets (HEADLINE_DATASETS), columns = methods.
    deltas = {}
    for ds in HEADLINE_DATASETS:
        for method in METHOD_ORDER:
            deltas[(ds, method)] = _paired_delta(root, ds, method, key=args.metric)

    # Layout: 1 panel, datasets on Y axis, Δ on X axis.
    fig_w = lama_aesthetics.TWO_COL_WIDTH
    fig_h = 0.45 * len(HEADLINE_DATASETS) + 1.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    n_methods = len(METHOD_ORDER)
    # Visual hierarchy: twin-indep (last) gets the strongest visual weight.
    # Two-baseline pair (deep ens, bagging) uses cool tones; twin-indep is the
    # warm contrast.  K=2 vs K=5 use different markers, not just shades.
    method_styles = {
        "Deep Ens. K=5": dict(marker="o", mfc="white",     mec="0.50", ms=4.5,  zorder=2),
        "Bagging K=2":   dict(marker="s", mfc="white",     mec="#1f77b4", ms=4.5,  zorder=3),
        "Bagging K=5":   dict(marker="s", mfc="#1f77b4",   mec="#0d3a5c", ms=5.5,  zorder=4),
        "Twin-indep":    dict(marker="D", mfc="#d62728",   mec="#7a1313", ms=6.5,  zorder=5),
    }

    yticks, yticklabels = [], []
    row_spacing = 1.0
    intra_method_offset = 0.18

    for i, ds in enumerate(reversed(HEADLINE_DATASETS)):  # smallest N at top
        y_centre = i * row_spacing
        for j, method in enumerate(METHOD_ORDER):
            res = deltas[(ds, method)]
            if res is None:
                continue
            mean, lo, hi = res
            offset = (j - (n_methods - 1) / 2) * intra_method_offset
            y = y_centre + offset
            ax.errorbar(
                mean * 100, y,
                xerr=[[(mean - lo) * 100], [(hi - mean) * 100]],
                fmt="none", elinewidth=1.0,
                ecolor=method_styles[method].get("mec", "0.55"),
                zorder=method_styles[method]["zorder"] - 0.5,
            )
            ax.plot(mean * 100, y, linestyle="None", **method_styles[method])
        yticks.append(y_centre)
        yticklabels.append(display(ds))

    ax.axvline(0, color="0.3", linewidth=0.8, zorder=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel(r"Paired $\Delta$ id-churn vs. ERM (percentage points)")
    ax.set_ylim(-0.6, len(HEADLINE_DATASETS) - 0.4)
    ax.grid(axis="x", linestyle="-", linewidth=0.4, alpha=0.4, zorder=0)

    # Legend below the plot — never overlaps the data.
    handles = []
    for method in METHOD_ORDER:
        style = dict(method_styles[method])
        style["linestyle"] = "None"
        handles.append(plt.Line2D([0], [0], label=method, **style))
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=len(METHOD_ORDER),
              frameon=False, handletextpad=0.4, columnspacing=1.2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}")

    # Also save a table-form CSV of the same numbers, for the appendix.
    csv_path = Path(args.out).with_suffix(".csv")
    with open(csv_path, "w") as fh:
        fh.write("dataset,method,delta_mean,delta_lo,delta_hi\n")
        for ds in HEADLINE_DATASETS:
            for method in METHOD_ORDER:
                r = deltas.get((ds, method))
                if r is None:
                    continue
                fh.write(f"{ds},{method},{r[0]:.5f},{r[1]:.5f},{r[2]:.5f}\n")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
