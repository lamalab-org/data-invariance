"""Paper figure: data-overlap spectrum.

Three operating points on the bootstrap-overlap axis live in our paper:
  -   0% overlap : codistillation (disjoint shards; Anil 2018-style)
  -  ~40% overlap: twin-indep λ=300 (overlapping bootstraps with replacement;
                    two networks each cover ~63% unique indices, share ~40%)
  - 100% overlap : twin-indep with shared bootstrap (ablation)

For every headline dataset where the three methods preserve ERM-level
accuracy (id-acc decline < 5pp), we plot the paired Δ id-churn vs ERM
at each of these three overlap fractions, with paired-bootstrap 95% CIs.

The pattern across datasets:
  - BACE / BBBP   : Δ-churn monotone in less overlap.
  - TADF          : U-shaped — middle overlap (twin-indep) wins.
  - MOF-thermal   : excluded (consistency-loss accuracy collapse, see §scope).

Single claim: data-overlap fraction is a meaningful hyperparameter, and
twin-indep's ~40% is a robust default but not optimal everywhere.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np

from _analysis_lib import bootstrap_paired, load_runs, pairwise_metrics, per_run_accuracies
from paper_constants import DEV_DATASET, FROZEN_LAM, HEADLINE_DATASETS, display


# (overlap_fraction, label, glob)
OVERLAP_POINTS = [
    (0,  "Codistillation",
     "codistillation_train*_lam{lam}.npz"),
    (40, "Twin-indep",
     "twin_indep_train*_lam{lam}.npz"),
    (100, "Twin shared boot",
     "twin_indep_shared_train*_lam{lam}.npz"),
]


def _paired_delta(ds_dir: Path, glob: str, lam: float):
    erm = load_runs(ds_dir, "erm_train*.npz")
    m = load_runs(ds_dir, glob.format(lam=lam))
    if not erm or not m:
        return None
    ep, _ = pairwise_metrics(erm)
    mp, _ = pairwise_metrics(m)
    common = sorted(set(ep).intersection(mp))
    if len(common) < 30:
        return None
    deltas = [mp[p]["id_churn"] - ep[p]["id_churn"] for p in common]
    return bootstrap_paired(deltas)


def _id_acc_drop(ds_dir: Path, glob: str, lam: float) -> float:
    erm = load_runs(ds_dir, "erm_train*.npz")
    m = load_runs(ds_dir, glob.format(lam=lam))
    if not erm or not m:
        return float("inf")
    ea, _ = per_run_accuracies(erm)
    ma, _ = per_run_accuracies(m)
    return float(np.mean(ea) - np.mean(ma))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/cross_sample")
    ap.add_argument("--out", default="paper/figures/fig5_overlap.pdf")
    args = ap.parse_args()
    root = Path(args.root)

    candidates = [DEV_DATASET] + HEADLINE_DATASETS
    keep, points = [], {}
    for ds in candidates:
        ds_dir = root / ds
        # require all three methods + no accuracy collapse > 5pp on any of them
        ok = True
        rows = []
        for ov, label, glob in OVERLAP_POINTS:
            res = _paired_delta(ds_dir, glob, FROZEN_LAM)
            drop = _id_acc_drop(ds_dir, glob, FROZEN_LAM)
            if res is None or drop > 0.05:
                ok = False
                break
            rows.append((ov, label, res))
        if ok:
            keep.append(ds)
            points[ds] = rows

    if not keep:
        print("No datasets with all three overlap points and no accuracy collapse.")
        return

    print(f"\nDatasets included: {keep}\n")
    print(f"{'dataset':16s}  {'overlap':>8s}  {'Δ id_churn (pp [95% CI])':>30s}")
    for ds in keep:
        for ov, label, (m, lo, hi) in points[ds]:
            print(f"{ds:16s}  {ov:>7d}%  {m*100:+6.2f} [{lo*100:+5.2f}, {hi*100:+5.2f}]")

    lama_aesthetics.get_style("main")
    plt.rcParams["axes.unicode_minus"] = False

    fig_w = lama_aesthetics.TWO_COL_WIDTH * 0.55
    fig_h = fig_w * 0.85
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Visual hierarchy carries the caption claim: TADF is the only
    # dataset that "catastrophically fails" at 0% overlap, so it must
    # dominate the figure. The other five datasets are all monotone
    # improvements toward less overlap and read as a single bundle.
    HIGHLIGHT = "tadf"
    grey = "0.65"

    # Light pink band above y=0 marks "worse than ERM"; the only line
    # that crosses into it is TADF at 0% overlap — that is the figure.
    ax.axhspan(0, 100, color="#fde6e6", alpha=0.45, zorder=0)
    ax.axhline(0, color="0.25", linewidth=0.9, zorder=1)

    bundle_handle = None
    for ds in keep:
        if ds == HIGHLIGHT:
            continue
        xs = [ov for ov, _, _ in points[ds]]
        ys = [m * 100 for _, _, (m, _, _) in points[ds]]
        los = [(m - lo) * 100 for _, _, (m, lo, _) in points[ds]]
        his = [(hi - m) * 100 for _, _, (m, _, hi) in points[ds]]
        line, _, _ = ax.errorbar(xs, ys, yerr=[los, his], color=grey,
                                 linewidth=0.9, marker="o", markersize=2.8,
                                 capsize=1.8, alpha=0.85, zorder=2)
        bundle_handle = line

    if HIGHLIGHT in points:
        xs = [ov for ov, _, _ in points[HIGHLIGHT]]
        ys = [m * 100 for _, _, (m, _, _) in points[HIGHLIGHT]]
        los = [(m - lo) * 100 for _, _, (m, lo, _) in points[HIGHLIGHT]]
        his = [(hi - m) * 100 for _, _, (m, _, hi) in points[HIGHLIGHT]]
        tadf_line, _, _ = ax.errorbar(xs, ys, yerr=[los, his],
                                      color="#d62728", linewidth=2.0,
                                      marker="o", markersize=5.5,
                                      capsize=2.5, zorder=4)
        # Annotate the codistillation failure point
        ax.annotate(
            "codistillation\nworse than ERM",
            xy=(xs[0], ys[0]), xytext=(20, ys[0] + 1.8),
            fontsize=7, color="#7a1313", ha="left",
            arrowprops=dict(arrowstyle="-", color="#7a1313",
                            linewidth=0.6),
        )

    ax.set_xticks([0, 40, 100])
    ax.set_xticklabels(["0%\n(codistillation)", "~40%\n(twin-indep)",
                        "100%\n(shared boot)"], fontsize=7.5)
    ax.set_xlabel("Bootstrap overlap between the two networks")
    ax.set_ylabel("Paired Δ id-churn vs ERM (pp)")

    # Two-entry legend: TADF and 'other five datasets' (clean).
    handles = [
        plt.Line2D([0], [0], color="#d62728", linewidth=2.0, marker="o",
                   markersize=5.5, label=display(HIGHLIGHT)),
    ]
    if bundle_handle is not None:
        other_labels = ", ".join(display(ds) for ds in keep if ds != HIGHLIGHT)
        handles.append(plt.Line2D([0], [0], color=grey, linewidth=0.9,
                                  marker="o", markersize=2.8,
                                  label=other_labels))
    ax.legend(handles=handles, loc="lower right", frameon=False,
              fontsize=7, handletextpad=0.4, labelspacing=0.4)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
