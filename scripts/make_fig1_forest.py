"""Paper Figure 1 (rendered): two-panel forest plot of paired Δ id-churn.

Left panel — Δ id-churn vs ERM, all four methods (Deep Ens K=5,
Bagging K=2, Bagging K=5, Twin-bootstrap λ=300).  This shows that every
method beats ERM and that ensembles miss the data-resampling axis.

Right panel — Δ id-churn vs Bagging K=2 (matched compute), Twin-bootstrap
only.  This isolates the ``consistency loss adds reliably to bagging
at matched compute'' claim: a single forest of red diamonds with
horizontal whiskers, a vertical reference line at zero (Bagging K=2
parity), and CIs that mostly sit to its left.

Each row is one dataset (smallest N at top).  Markers are method
means; whiskers are paired-bootstrap 95% CIs across the 45 seed pairs.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np

from _analysis_lib import GLOBS, bootstrap_paired, load_runs, pairwise_metrics
from paper_constants import (
    FROZEN_LAM, HEADLINE_DATASETS, METHOD_ORDER, N_SEEDS, display, glob_for,
)


def _paired_delta(root: Path, dataset: str, method: str,
                  reference_glob: str, key: str = "id_churn"):
    """Paired-bootstrap (mean, lo, hi) for (method - reference) on dataset."""
    ref_runs = load_runs(root / dataset, reference_glob)
    method_runs = load_runs(root / dataset, glob_for(method))
    if len(ref_runs) != N_SEEDS or len(method_runs) != N_SEEDS:
        return None
    ref_pairs, _ = pairwise_metrics(ref_runs)
    m_pairs, _ = pairwise_metrics(method_runs)
    deltas = [m_pairs[p][key] - ref_pairs[p][key]
              for p in sorted(set(ref_pairs).intersection(m_pairs))]
    return bootstrap_paired(deltas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/cross_sample")
    ap.add_argument("--out", default="paper/figures/fig1_forest.pdf")
    ap.add_argument("--metric", default="id_churn",
                    choices=["id_churn", "ood_churn"])
    args = ap.parse_args()
    root = Path(args.root)

    lama_aesthetics.get_style("main")
    plt.rcParams["axes.unicode_minus"] = False

    # Left panel: Δ vs ERM, all four methods.
    deltas_vs_erm = {(ds, m): _paired_delta(root, ds, m, GLOBS["erm"], args.metric)
                     for ds in HEADLINE_DATASETS for m in METHOD_ORDER}
    # Right panel: Δ vs Bagging K=2 (matched compute), Twin-bootstrap only.
    deltas_vs_bag2 = {ds: _paired_delta(root, ds, "Twin-bootstrap",
                                        GLOBS["bagging_2"], args.metric)
                      for ds in HEADLINE_DATASETS}

    # Single marker shape (circle) across methods; fill state and edge
    # colour distinguish parameter-side (gray) from data-side
    # (blue / red), and open vs filled within each group.
    method_styles = {
        "MC dropout":    dict(marker="o", mfc="white",   mec="0.45",    ms=5.0, zorder=1),
        "SWA":           dict(marker="o", mfc="0.85",    mec="0.20",    ms=5.0, zorder=1.5),
        "Deep Ens. K=5": dict(marker="o", mfc="0.65",    mec="0.30",    ms=5.0, zorder=2),
        "Bagging K=2":   dict(marker="o", mfc="white",   mec="#1f77b4", ms=5.5, zorder=3),
        "Bagging K=5":   dict(marker="o", mfc="#1f77b4", mec="#0d3a5c", ms=6.0, zorder=4),
        "Twin-bootstrap":dict(marker="o", mfc="#d62728", mec="#7a1313", ms=7.0, zorder=5),
    }

    # Determine sensible x-ranges from the data (CI extents) plus padding.
    def _x_range(deltas: dict) -> tuple[float, float]:
        lows, highs = [], []
        for v in deltas.values():
            if v is None:
                continue
            _, lo, hi = v
            lows.append(lo * 100)
            highs.append(hi * 100)
        if not lows:
            return -10.0, 1.0
        lo, hi = min(lows), max(highs)
        span = max(hi - lo, 1.0)
        return lo - 0.06 * span, hi + 0.06 * span

    x_lo_l, x_hi_l = _x_range(deltas_vs_erm)
    x_lo_r, x_hi_r = _x_range({k: v for k, v in deltas_vs_bag2.items()})

    fig_w = lama_aesthetics.TWO_COL_WIDTH
    fig_h = 0.45 * len(HEADLINE_DATASETS) + 1.2
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(fig_w, fig_h), sharey=True,
        gridspec_kw=dict(width_ratios=[2.0, 1.0], wspace=0.10),
    )

    # ---------- Left: Δ vs ERM, all 4 methods ----------
    n_methods = len(METHOD_ORDER)
    # Tighter intra-row offset so the five method markers cluster within a
    # row instead of bleeding into adjacent rows (5 markers x 0.16 spacing
    # span ±0.32; row gap is 0.36 — adjacent rows visually separate).
    intra_offset = 0.16
    row_spacing = 1.0
    yticks, yticklabels = [], []

    n_rows = len(HEADLINE_DATASETS)
    # Row identity: alternating shading (visible but unobtrusive) plus a
    # thin separator at every row boundary so a reader can quickly tell
    # which markers belong to the same dataset.
    for i in range(n_rows):
        if i % 2 == 0:
            for ax in (ax_l, ax_r):
                ax.axhspan(
                    i * row_spacing - 0.5 * row_spacing,
                    i * row_spacing + 0.5 * row_spacing,
                    facecolor="0.88", edgecolor="none", zorder=-1,
                )
    for i in range(n_rows + 1):
        y = (i - 0.5) * row_spacing
        for ax in (ax_l, ax_r):
            ax.axhline(y, color="0.65", linewidth=0.3, zorder=0, alpha=0.7)

    for i, ds in enumerate(reversed(HEADLINE_DATASETS)):
        y_centre = i * row_spacing
        for j, method in enumerate(METHOD_ORDER):
            res = deltas_vs_erm[(ds, method)]
            if res is None:
                continue
            mean, lo, hi = res
            offset = (j - (n_methods - 1) / 2) * intra_offset
            y = y_centre + offset
            ax_l.errorbar(
                mean * 100, y,
                xerr=[[(mean - lo) * 100], [(hi - mean) * 100]],
                fmt="none", capsize=2.0,
                elinewidth=1.6 if method == "Twin-bootstrap" else 1.0,
                ecolor=method_styles[method]["mec"],
                zorder=method_styles[method]["zorder"] - 0.5,
            )
            ax_l.plot(mean * 100, y, linestyle="None", **method_styles[method])
        yticks.append(y_centre)
        yticklabels.append(display(ds))

    # Per-method across-dataset mean as vertical reference line.
    for method in METHOD_ORDER:
        mean_of_means = np.mean([deltas_vs_erm[(ds, method)][0] * 100
                                 for ds in HEADLINE_DATASETS
                                 if deltas_vs_erm[(ds, method)] is not None])
        col = method_styles[method]["mec"]
        ax_l.axvline(mean_of_means, color=col,
                     linewidth=1.2 if method == "Twin-bootstrap" else 0.7,
                     linestyle="-" if method == "Twin-bootstrap" else "--",
                     alpha=0.85 if method == "Twin-bootstrap" else 0.5,
                     zorder=1)

    ax_l.axvline(0, color="0.25", linewidth=1.0, zorder=1)
    ax_l.set_yticks(yticks)
    ax_l.set_yticklabels(yticklabels)
    ax_l.set_xlim(x_lo_l, x_hi_l)
    ax_l.set_xlabel("Paired Δ id-churn vs. ERM (pp; <0 better)")
    ax_l.set_ylim(-0.6, len(HEADLINE_DATASETS) - 0.4)
    ax_l.grid(axis="x", linestyle="-", linewidth=0.4, alpha=0.35, zorder=0)

    # ---------- Right: Δ vs Bagging K=2, Twin-bootstrap only ----------
    twin_means = []
    for i, ds in enumerate(reversed(HEADLINE_DATASETS)):
        y = i * row_spacing
        res = deltas_vs_bag2[ds]
        if res is None:
            continue
        mean, lo, hi = res
        twin_means.append(mean * 100)
        ax_r.errorbar(
            mean * 100, y,
            xerr=[[(mean - lo) * 100], [(hi - mean) * 100]],
            fmt="none", elinewidth=1.6, capsize=2.0,
            ecolor=method_styles["Twin-bootstrap"]["mec"],
            zorder=4,
        )
        ax_r.plot(mean * 100, y, linestyle="None",
                  **method_styles["Twin-bootstrap"])

    if twin_means:
        ax_r.axvline(np.mean(twin_means),
                     color=method_styles["Twin-bootstrap"]["mec"],
                     linewidth=1.2, alpha=0.85, zorder=1)
    ax_r.axvline(0, color="0.25", linewidth=1.0, zorder=1)
    ax_r.set_xlim(x_lo_r, x_hi_r)
    ax_r.set_xlabel("Δ vs. Bagging K=2 (pp)")
    ax_r.grid(axis="x", linestyle="-", linewidth=0.4, alpha=0.35, zorder=0)
    ax_r.tick_params(left=False)

    # Legend below: list all four methods (left panel) plus a note on right.
    legend_label = {"Twin-bootstrap": r"Twin-bootstrap $\lambda{=}300$"}
    handles = []
    for method in METHOD_ORDER:
        style = dict(method_styles[method])
        style["linestyle"] = "None"
        handles.append(plt.Line2D([0], [0],
                                   label=legend_label.get(method, method),
                                   **style))
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, -0.005), ncol=len(METHOD_ORDER),
               frameon=False, handletextpad=0.4, columnspacing=1.4,
               fontsize=8)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}")

    # CSV for the appendix.
    csv_path = Path(args.out).with_suffix(".csv")
    with open(csv_path, "w") as fh:
        fh.write("dataset,method,reference,delta_mean,delta_lo,delta_hi\n")
        for ds in HEADLINE_DATASETS:
            for method in METHOD_ORDER:
                r = deltas_vs_erm.get((ds, method))
                if r is None:
                    continue
                fh.write(f"{ds},{method},ERM,{r[0]:.5f},{r[1]:.5f},{r[2]:.5f}\n")
            r = deltas_vs_bag2.get(ds)
            if r is not None:
                fh.write(f"{ds},Twin-bootstrap,Bagging K=2,"
                         f"{r[0]:.5f},{r[1]:.5f},{r[2]:.5f}\n")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
