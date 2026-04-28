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

    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(keep)))
    for di, ds in enumerate(keep):
        xs = [ov for ov, _, _ in points[ds]]
        ys = [m * 100 for _, _, (m, _, _) in points[ds]]
        los = [(m - lo) * 100 for _, _, (m, lo, _) in points[ds]]
        his = [(hi - m) * 100 for _, _, (m, _, hi) in points[ds]]
        col = "#d62728" if ds == DEV_DATASET else cmap[di]
        lw = 1.6 if ds == DEV_DATASET else 1.0
        ms = 5.0 if ds == DEV_DATASET else 3.5
        ax.errorbar(xs, ys, yerr=[los, his], color=col,
                    linewidth=lw, marker="o", markersize=ms,
                    capsize=2.5, label=display(ds), zorder=3 if ds == DEV_DATASET else 2)

    ax.axhline(0, color="0.25", linewidth=0.8, zorder=1)
    ax.set_xticks([0, 40, 100])
    ax.set_xticklabels(["0%\n(codistillation)", "~40%\n(twin-indep)",
                        "100%\n(shared boot)"], fontsize=7.5)
    ax.set_xlabel("Bootstrap overlap between the two networks")
    ax.set_ylabel("Paired Δ id-churn vs ERM (pp)")
    ax.legend(loc="upper left", frameon=False, fontsize=7,
              handletextpad=0.4)
    ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.35, zorder=0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
