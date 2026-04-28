"""Appendix table: bootstrap-overlap spectrum.

Reports paired Δ id-churn vs ERM at three operating points on the
inter-network bootstrap-overlap axis:
  -   0% overlap : codistillation (disjoint shards; Anil 2018)
  -  ~40% overlap: twin-indep λ=300
  - 100% overlap : twin-indep with shared bootstrap (ablation)

across every dataset for which all three methods completed.  Cells
where the method drops id-acc by more than 5pp from ERM are flagged
as accuracy-collapse (the corresponding churn reduction is then
largely majority-class prediction).
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
    ap.add_argument("--latex", default="paper/sections/tables/overlap_spectrum.tex")
    args = ap.parse_args()
    root = Path(args.root)

    # Include every dataset for which all three methods completed; flag
    # accuracy-degradation points (any consistency method dropping id-acc
    # by more than 5pp from ERM) so they read as "failures" in the figure.
    BORDERLINE = ["mof_thermal", "skin_reaction", "herg", "hia_hou"]
    candidates = [DEV_DATASET] + HEADLINE_DATASETS + BORDERLINE
    keep, points, accuracy_collapse = [], {}, {}
    for ds in candidates:
        ds_dir = root / ds
        rows = []
        collapse_flags = []
        complete = True
        for ov, label, glob in OVERLAP_POINTS:
            res = _paired_delta(ds_dir, glob, FROZEN_LAM)
            drop = _id_acc_drop(ds_dir, glob, FROZEN_LAM)
            if res is None:
                complete = False
                break
            rows.append((ov, label, res))
            collapse_flags.append(drop > 0.05)
        if complete:
            keep.append(ds)
            points[ds] = rows
            accuracy_collapse[ds] = collapse_flags

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

    # Grouped horizontal bars: rows = datasets, three bars per row
    # (overlap fractions). The x=0 vertical line is the ERM baseline;
    # bars to its left = improvement, bars to its right = regression.
    # The figure's load-bearing observation is that exactly one bar in
    # the entire plot crosses to the right of zero (TADF, 0% overlap).
    fig_w = lama_aesthetics.TWO_COL_WIDTH * 0.55
    fig_h = 0.55 * len(keep) + 1.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    overlap_colors = {
        0:   "#7a1313",   # codistillation — dark red
        40:  "#d62728",   # twin-indep — bright red
        100: "#9bb6cf",   # shared boot — muted blue
    }
    overlap_labels = {
        0:   "codistillation (0% overlap)",
        40:  "twin-indep (~40%)",
        100: "shared boot (100%)",
    }

    n_ds = len(keep)
    bar_h = 0.22
    y_centres = np.arange(n_ds)[::-1]

    for di, ds in enumerate(keep):
        for j, (ov, _, (m, lo, hi)) in enumerate(points[ds]):
            offset = (j - 1) * bar_h * 1.05
            y = y_centres[di] + offset
            # Plot REDUCTION (positive = improvement); flip sign of Δ.
            reduction = -m * 100
            err_lo = (hi - m) * 100
            err_hi = (m - lo) * 100
            collapsed = accuracy_collapse[ds][j]
            ax.barh(
                y, reduction, height=bar_h,
                color=overlap_colors[ov] if not collapsed else "white",
                edgecolor=overlap_colors[ov],
                linewidth=0.8 if collapsed else 0.0,
                hatch="///" if collapsed else None,
                xerr=[[err_lo], [err_hi]],
                error_kw=dict(elinewidth=0.8, ecolor="0.3"),
                zorder=3,
            )

    ax.axvline(0, color="0.2", linewidth=1.0, zorder=2)
    ax.set_yticks(y_centres)
    ax.set_yticklabels([display(ds) for ds in keep])
    ax.set_xlabel("Reduction in id-churn vs ERM (pp; >0 better)")
    ax.set_ylim(-0.6, n_ds - 0.4)
    ax.grid(axis="x", linestyle="-", linewidth=0.4, alpha=0.4, zorder=0)

    # Legend below figure
    handles = [plt.Rectangle((0, 0), 1, 1, fc=overlap_colors[ov],
                             ec="none", label=overlap_labels[ov])
               for ov in [0, 40, 100]]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="white",
                                 ec="0.3", hatch="///",
                                 label="id-acc drop $>$5pp"))
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.08), ncol=2,
              frameon=False, fontsize=7.5, handletextpad=0.4,
              columnspacing=1.2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"\nWrote {args.out}")

    # Emit appendix LaTeX table (this is the primary artefact; the
    # figure above is kept only for code-internal QA).
    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \caption{Bootstrap-overlap spectrum: paired $\Delta$ id-churn"
        r" vs.\ ERM at three operating points (codistillation $0\%$,"
        r" twin-indep ${\sim}40\%$, twin-shared $100\%$ overlap),"
        r" percentage points (negative is better). Daggers ($^\dagger$)"
        r" mark cells where the method drops id-accuracy by more than"
        r" $5$pp from ERM (the corresponding churn reduction is then"
        r" largely majority-class prediction). Twin-indep at"
        r" ${\sim}40\%$ overlap is the only column with no failure case"
        r" (no daggers and no positive entries).}",
        r"  \label{tab:overlap_spectrum}",
        r"  \small",
        r"  \begin{tabular}{lrlll}",
        r"    \toprule",
        r"    Dataset & $N$ & Codistillation $0\%$"
        r" & Twin-indep ${\sim}40\%$ & Twin-shared $100\%$ \\",
        r"    \midrule",
    ]
    from paper_constants import N_TRAIN
    for ds in sorted(keep, key=lambda d: N_TRAIN.get(d, 10**9)):
        cells = []
        for j, (ov, _, (m, lo, hi)) in enumerate(points[ds]):
            txt = f"{m*100:+.1f} [{lo*100:+.1f}, {hi*100:+.1f}]"
            if accuracy_collapse[ds][j]:
                txt = txt + r"$^\dagger$"
            cells.append(txt)
        lines.append(
            f"    {display(ds)} & {N_TRAIN.get(ds, '---')} & "
            + " & ".join(cells) + r" \\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    Path(args.latex).parent.mkdir(parents=True, exist_ok=True)
    Path(args.latex).write_text("\n".join(lines))
    print(f"Wrote {args.latex}")


if __name__ == "__main__":
    main()
