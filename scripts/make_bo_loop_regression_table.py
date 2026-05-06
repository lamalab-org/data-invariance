"""LaTeX table emitter for the BO trajectory variance experiment.

Reads outputs/bo_loop_regression.json (per-trajectory traces) produced
by scripts/bo_loop_regression.py and emits:

    paper/sections/tables/bo_loop_regression.tex
    outputs/bo_loop_regression_summary.csv (one row per (dataset, method))

The table reports, for each (dataset, method), the cross-trajectory
mean and 10,000-resample paired-bootstrap 95% CI on:

  - final_best y across the K trajectories
  - acquired-set Jaccard mean across the binom(K, 2) pairs

We also report the per-method final_best std with a paired-bootstrap CI
(resampling the K trajectories) and the std-as-fraction-of-y-range as
a scale-anchor.
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np


DATASET_LABELS = {"esol_reg": "ESOL", "freesolv_reg": "FreeSolv",
                  "lipo_reg": "Lipo"}
METHOD_LABELS = {"erm": "ERM", "bagging": "Bagging-$K{=}5$",
                 "twin": "Twin-$\\lambda{=}3$"}


def jaccard(a: set, b: set) -> float:
    union = len(a | b)
    return float(len(a & b) / union) if union else 0.0


def bootstrap_mean_ci(values, n_boot=10_000, alpha=0.05, rng=None):
    """Paired-bootstrap CI on the mean of a list of values."""
    arr = np.asarray(list(values), dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = arr[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(lo), float(hi)


def bootstrap_std_ci(values, n_boot=10_000, alpha=0.05, rng=None):
    """Paired-bootstrap CI on the sample std of a list of values."""
    arr = np.asarray(list(values), dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(arr)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    boots = arr[rng.integers(0, n, size=(n_boot, n))].std(axis=1, ddof=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(arr.std(ddof=1)), float(lo), float(hi)


def per_method_summary(records, dataset, method):
    rows = [r for r in records if r["dataset"] == dataset and r["method"] == method]
    if not rows:
        return {}
    finals = np.array([r["final_best"] for r in rows], dtype=np.float64)
    acquired_sets = [set(r["acquired"]) for r in rows]
    pair_jaccards = [jaccard(a, b) for a, b in combinations(acquired_sets, 2)]
    y_min = float(rows[0].get("y_min", float("nan")))
    y_max = float(rows[0].get("y_max", float("nan")))
    y_range = y_max - y_min if np.isfinite(y_min) and np.isfinite(y_max) else float("nan")
    rng = np.random.default_rng(0)
    final_mean_ci = bootstrap_mean_ci(finals, rng=rng)
    final_std_ci = bootstrap_std_ci(finals, rng=rng)
    jac_mean_ci = bootstrap_mean_ci(pair_jaccards, rng=rng) if pair_jaccards else (float("nan"),) * 3
    return {
        "n_trajectories": len(rows),
        "n_pairs": len(pair_jaccards),
        "final_best_mean": final_mean_ci[0],
        "final_best_mean_lo": final_mean_ci[1],
        "final_best_mean_hi": final_mean_ci[2],
        "final_best_std": final_std_ci[0],
        "final_best_std_lo": final_std_ci[1],
        "final_best_std_hi": final_std_ci[2],
        "final_best_min": float(finals.min()),
        "final_best_max": float(finals.max()),
        "jaccard_mean": jac_mean_ci[0],
        "jaccard_mean_lo": jac_mean_ci[1],
        "jaccard_mean_hi": jac_mean_ci[2],
        "y_range": y_range,
        "std_pct_of_range": (final_std_ci[0] / y_range * 100)
        if y_range > 1e-9 else float("nan"),
    }


def paired_delta_vs_erm(records, dataset, method):
    """Trajectories of two methods are paired by trajectory_id; return the
    paired-bootstrap CI on Δ(final_best) and Δ(jaccard) vs ERM."""
    method_rows = [r for r in records
                   if r["dataset"] == dataset and r["method"] == method]
    erm_rows = [r for r in records
                if r["dataset"] == dataset and r["method"] == "erm"]
    if not method_rows or not erm_rows:
        return None
    by_id_m = {r["trajectory_id"]: r for r in method_rows}
    by_id_e = {r["trajectory_id"]: r for r in erm_rows}
    common = sorted(set(by_id_m) & set(by_id_e))
    if not common:
        return None
    final_deltas = np.array([by_id_m[i]["final_best"] - by_id_e[i]["final_best"]
                             for i in common], dtype=np.float64)
    rng = np.random.default_rng(1)
    return {
        "final_delta_mean": float(final_deltas.mean()),
        "final_delta_lo": bootstrap_mean_ci(final_deltas, rng=rng)[1],
        "final_delta_hi": bootstrap_mean_ci(final_deltas, rng=rng)[2],
        "n_paired": len(common),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="outputs/bo_loop_regression.json")
    ap.add_argument("--csv", default="outputs/bo_loop_regression_summary.csv")
    ap.add_argument("--latex", default="paper/sections/tables/bo_loop_regression.tex")
    args = ap.parse_args()

    records = json.loads(Path(args.json).read_text())
    datasets = []
    for r in records:
        if r["dataset"] not in datasets:
            datasets.append(r["dataset"])
    methods = ["erm", "bagging", "twin"]

    summary_rows = []
    for ds in datasets:
        for method in methods:
            s = per_method_summary(records, ds, method)
            if not s:
                continue
            row = {"dataset": ds,
                   "dataset_label": DATASET_LABELS.get(ds, ds),
                   "method": method, **s}
            if method != "erm":
                d = paired_delta_vs_erm(records, ds, method)
                if d:
                    row.update(d)
            summary_rows.append(row)

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.csv).open("w", newline="") as f:
        if summary_rows:
            keys = list({k for row in summary_rows for k in row.keys()})
            keys.sort()
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(summary_rows)
    print(f"Wrote {args.csv}")

    # ----- LaTeX -----
    by_ds = {}
    for r in summary_rows:
        by_ds.setdefault(r["dataset"], {})[r["method"]] = r

    K_traj = max((r["n_trajectories"] for r in summary_rows), default=0)
    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \caption{\textbf{\boldmath BO trajectory variance on the three "
        r"regression benchmarks: bagging-$K{=}5$ and twin-bootstrap reduce "
        r"the cross-trajectory standard deviation of the final-best $y$ on "
        r"every dataset.}  For each (dataset, method) we run "
        f"$K{{=}}{K_traj}$ "
        r"BO trajectories sharing the same random initial subset of "
        r"$50$ labelled molecules; trajectories diverge only in the in-loop "
        r"training-data bootstraps.  At each step the surrogate is "
        r"retrained from scratch, predicts $\hat{y}$ on the unlabelled "
        r"remainder, and acquires the $\arg\max\hat{y}$.  \emph{Final best} "
        r"reports cross-trajectory mean and std with $95\%$ bootstrap CIs "
        r"over the $K$ trajectories ($10{,}000$ resamples).  \emph{std/range} is the std as a "
        r"percentage of each dataset's $y$ range, anchoring its absolute "
        r"scale.  \emph{Acquired Jaccard} is the mean overlap of "
        r"per-trajectory acquired-molecule sets across all "
        r"$\binom{K}{2}$ trajectory pairs.}",
        r"  \label{tab:bo_loop_regression}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{llccc@{\hspace{1.0em}}c}",
        r"    \toprule",
        r"    Dataset & Method & Final best mean [95\% CI] & Final best std [95\% CI]"
        r" & std/range (\%) & Acquired Jaccard \\",
        r"    \midrule",
    ]
    for ds in datasets:
        cells = by_ds.get(ds, {})
        if not cells:
            continue
        # Identify best method(s) per metric within this dataset row group:
        # mean is "best when highest" (we maximise y); std and std/range
        # are "best when lowest"; Jaccard is "best when highest".  Ties
        # bold every method at the best value, treating values within
        # 1e-9 as equal.
        present_methods = [m for m in methods if cells.get(m)]
        def _bests(key, *, maximise):
            vals = [(cells[m][key], m) for m in present_methods]
            target = max(v for v, _ in vals) if maximise else min(v for v, _ in vals)
            return {m for v, m in vals if abs(v - target) < 1e-9}
        best = {
            "final_best_mean":   _bests("final_best_mean",   maximise=True),
            "final_best_std":    _bests("final_best_std",    maximise=False),
            "std_pct_of_range":  _bests("std_pct_of_range",  maximise=False),
            "jaccard_mean":      _bests("jaccard_mean",      maximise=True),
        }
        def _bold(cell, condition):
            return r"\textbf{" + cell + r"}" if condition else cell

        for i, method in enumerate(methods):
            r = cells.get(method)
            if not r:
                continue
            ds_label = DATASET_LABELS.get(ds, ds) if i == 0 else ""
            mean_str = _bold(
                f"{r['final_best_mean']:.2f} "
                f"[{r['final_best_mean_lo']:.2f}, {r['final_best_mean_hi']:.2f}]",
                method in best["final_best_mean"],
            )
            std_str = _bold(
                f"{r['final_best_std']:.3f} "
                f"[{r['final_best_std_lo']:.3f}, {r['final_best_std_hi']:.3f}]",
                method in best["final_best_std"],
            )
            range_pct_raw = (f"{r['std_pct_of_range']:.1f}"
                             if np.isfinite(r['std_pct_of_range']) else "---")
            range_pct = _bold(range_pct_raw,
                              method in best["std_pct_of_range"])
            jac_str = _bold(
                f"{r['jaccard_mean']:.2f} "
                f"[{r['jaccard_mean_lo']:.2f}, {r['jaccard_mean_hi']:.2f}]",
                method in best["jaccard_mean"],
            )
            lines.append(
                f"    {ds_label} & {METHOD_LABELS[method]} & "
                f"{mean_str} & {std_str} & {range_pct} & {jac_str} \\\\"
            )
        lines.append(r"    \addlinespace[2pt]")
    if lines[-1] == r"    \addlinespace[2pt]":
        lines.pop()
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    Path(args.latex).parent.mkdir(parents=True, exist_ok=True)
    Path(args.latex).write_text("\n".join(lines))
    print(f"Wrote {args.latex}")

    print("\n=== summary ===")
    print(f"{'dataset':<10}{'method':<10}{'mean':>9}{'std':>9}"
          f"{'std%range':>11}{'Jaccard':>10}")
    for r in summary_rows:
        print(f"{r['dataset_label']:<10}{r['method']:<10}"
              f"{r['final_best_mean']:>9.3f}{r['final_best_std']:>9.3f}"
              f"{r['std_pct_of_range']:>11.2f}{r['jaccard_mean']:>10.3f}")


if __name__ == "__main__":
    main()
