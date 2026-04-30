"""Friedman + Nemenyi post-hoc test on cross-sample id-churn.

Aggregate test: across the 8 headline datasets, do the methods rank
significantly differently on cross-sample id-churn?  This is the
standard non-parametric multi-method comparison (no normality
assumption; no per-pair pairing assumption; treats each dataset as
a block).

Methods compared: ERM, Deep Ensemble K=5, Bagging K=2, Bagging K=5,
Twin-bootstrap λ=300.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.stats as stats

from _analysis_lib import GLOBS, load_runs, pairwise_metrics
from paper_constants import (
    DEV_DATASET,
    FROZEN_LAM,
    HEADLINE_DATASETS,
    METHOD_GLOBS,
    METHOD_ORDER,
    display,
    glob_for,
)


def _mean_id_churn(ds_dir: Path, glob: str) -> float | None:
    runs = load_runs(ds_dir, glob)
    if len(runs) < 2:
        return None
    pair_metrics, _ = pairwise_metrics(runs)
    churns = [m["id_churn"] for m in pair_metrics.values()]
    return float(np.mean(churns))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/cross_sample")
    ap.add_argument("--frozen_lam", type=float, default=FROZEN_LAM)
    args = ap.parse_args()
    root = Path(args.root)

    methods = ["ERM"] + METHOD_ORDER
    method_globs = {"ERM": GLOBS["erm"]} | METHOD_GLOBS
    datasets = [DEV_DATASET] + HEADLINE_DATASETS

    # Build (dataset, method) → mean id_churn.
    matrix = np.full((len(datasets), len(methods)), np.nan)
    for di, ds in enumerate(datasets):
        for mi, m in enumerate(methods):
            glob = method_globs[m].format(lam=args.frozen_lam) \
                if "{lam}" in method_globs[m] else method_globs[m]
            v = _mean_id_churn(root / ds, glob)
            if v is not None:
                matrix[di, mi] = v

    # Drop datasets with any missing method.
    valid = ~np.isnan(matrix).any(axis=1)
    if not valid.any():
        print("No complete (dataset × method) cells; cannot run Friedman.")
        sys.exit(1)
    valid_ds = [datasets[i] for i in range(len(datasets)) if valid[i]]
    M = matrix[valid]

    print(f"Friedman test on cross-sample id-churn over "
          f"{len(valid_ds)} datasets, {len(methods)} methods\n")
    print(f"{'dataset':16s}  " + "  ".join(f"{m:>14s}" for m in methods))
    for ds, row in zip(valid_ds, M):
        print(f"{display(ds):16s}  "
              + "  ".join(f"{v*100:>13.2f}%" for v in row))

    # Friedman.  Each row = block (dataset); columns = treatments (methods).
    chi2, p = stats.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
    print(f"\nFriedman chi-squared = {chi2:.3f},  p = {p:.3g}")

    # Per-method rank average (lower is better, since lower id_churn is better).
    ranks = np.array([stats.rankdata(M[i, :]) for i in range(M.shape[0])])
    avg_ranks = ranks.mean(axis=0)
    print(f"\nAverage ranks (1 = best, lower id_churn):")
    for m, r in sorted(zip(methods, avg_ranks), key=lambda x: x[1]):
        print(f"  {m:18s}  {r:.2f}")

    # Nemenyi post-hoc critical difference (CD) at alpha=0.05.
    # CD = q_alpha * sqrt(k * (k + 1) / (6 * N))  where k = methods, N = datasets.
    k = len(methods)
    N = len(valid_ds)
    # Studentized range critical values for alpha=0.05 (Demsar 2006, Table 5).
    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
           8: 3.031, 9: 3.102, 10: 3.164}
    CD = q05[k] * np.sqrt(k * (k + 1) / (6.0 * N))
    print(f"\nNemenyi critical difference (alpha=0.05): {CD:.3f}")
    print(f"  (rank gaps larger than this are significant)\n")

    # Pairwise rank-gap summary.
    print(f"Significant rank gaps (|rank_i - rank_j| > {CD:.2f}):")
    for (i, mi), (j, mj) in combinations(enumerate(methods), 2):
        gap = abs(avg_ranks[i] - avg_ranks[j])
        flag = "**" if gap > CD else "  "
        print(f"  {flag} {mi:18s} vs {mj:18s}  rank-gap = {gap:.2f}")

    # CSV dump for paper-macros and audit. One row per method (mean rank),
    # plus a header line with chi^2 / p-value / N / k as a summary row.
    import csv as _csv
    out_csv = Path("outputs/friedman.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["scope", "key", "value"])
        w.writerow(["test", "chi2", float(chi2)])
        w.writerow(["test", "p_value", float(p)])
        w.writerow(["test", "k_methods", int(k)])
        w.writerow(["test", "n_datasets", int(N)])
        w.writerow(["test", "nemenyi_cd", float(CD)])
        for m, r in zip(methods, avg_ranks):
            w.writerow(["mean_rank", m, float(r)])
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
