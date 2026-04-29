"""Per-dataset health filter + paired-bootstrap deltas vs ERM.

Design decisions
----------------
1. **Health filter.**  The paper's pre-registered filter is
   ``ERM_id_acc ≥ majority + 0.05`` AND ``id_test_n ≥ 60``.  The
   first guards against datasets where the classifier is essentially
   predicting the majority class (cross-sample fragility is then
   dominated by majority-class shuffling, not learned-decision
   sensitivity).  The second avoids datasets where 95% CIs span the
   entire interval.  Both rules are pre-registered in
   ``scripts/paper_constants.py`` and applied at the time a dataset
   is added to the headline sweep.

2. **Method comparison.**  Reports the four headline methods (ERM,
   bagging-K=5, MC-dropout-T=20, twin-indep λ=300) on a single
   dataset.  Each has its own NPZ glob in ``METHODS``.  The paper's
   main forest plot uses these same globs via
   ``paper_constants.METHOD_GLOBS``.

3. **Paired bootstrap.**  All Δ are paired across the
   ``binom(10, 2) = 45`` seed-pair distribution.  Compute via
   ``_analysis_lib.bootstrap_paired`` (10,000 resamples).
   Relative reductions in the printed output use the ERM mean
   churn as denominator so percentages compare like-for-like
   across datasets with different baseline magnitudes.

Usage:
  python scripts/analyze_dataset.py <dataset>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from _analysis_lib import (
    bootstrap_ci, bootstrap_paired, fmt_ci,
    load_runs, pairwise_metrics, per_run_accuracies,
)

ROOT = Path("outputs/cross_sample")
METHODS = {
    "ERM":             "erm_train*.npz",
    "Bagging-K=5":     "bagging_train*_K5.npz",
    "MC-dropout":      "mc_dropout_train*_T20.npz",
    "Twin-indep λ=300": "twin_indep_train*_lam300.0.npz",
}


def majority_baseline(runs):
    """ERM-time majority-class accuracy on canonical id_test."""
    if not runs:
        return float("nan")
    _, d = runs[0]
    labels = d["id_labels"]
    counts = np.bincount(labels)
    return float(counts.max() / counts.sum())


def main(dataset: str) -> None:
    data_dir = ROOT / dataset
    if not data_dir.exists():
        print(f"  no output dir: {data_dir}")
        sys.exit(1)
    runs = {name: load_runs(data_dir, glob) for name, glob in METHODS.items()}
    erm = runs["ERM"]
    if not erm:
        print(f"  no ERM runs found in {data_dir}")
        sys.exit(1)

    # Health filter
    _, d0 = erm[0]
    n_id = len(d0["id_labels"])
    maj = majority_baseline(erm)
    erm_accs, _ = per_run_accuracies(erm)
    erm_mean_acc = float(np.mean(erm_accs))
    pass_filter = (erm_mean_acc >= maj + 0.05) and (n_id >= 60)
    print(f"\n=== {dataset} ===")
    print(f"  test_n={n_id}  majority={maj:.3f}  ERM_id_acc={erm_mean_acc:.3f}  "
          f"  filter_pass={'✓' if pass_filter else '✗'}")

    # Per-method summaries
    summary = {}
    for name, rs in runs.items():
        if len(rs) < 2:
            print(f"  {name:18s}  insufficient runs ({len(rs)})")
            continue
        id_accs, _ = per_run_accuracies(rs)
        pm, pairs = pairwise_metrics(rs)
        churn = [pm[p]["id_churn"] for p in pairs]
        symkl = [pm[p]["id_sym_kl"] for p in pairs]
        summary[name] = {
            "id_acc": bootstrap_ci(id_accs),
            "id_churn": bootstrap_ci(churn),
            "id_sym_kl": bootstrap_ci(symkl),
            "churn_pairs": np.array(churn),
            "symkl_pairs": np.array(symkl),
            "pairs": pairs,
        }
        print(f"  {name:18s}  id_acc {fmt_ci(summary[name]['id_acc'])}  "
              f"churn {fmt_ci(summary[name]['id_churn'], pct=True)}  "
              f"sym_kl {fmt_ci(summary[name]['id_sym_kl'])}")

    # Paired Δ vs ERM
    if "ERM" not in summary:
        return
    erm_pairs = summary["ERM"]["pairs"]
    erm_churn = summary["ERM"]["churn_pairs"]
    erm_kl = summary["ERM"]["symkl_pairs"]
    print("  Paired Δ vs ERM (same 45 seed-pairs):")
    for name in METHODS:
        if name == "ERM" or name not in summary:
            continue
        m_pm, _ = pairwise_metrics(runs[name])
        if not all(p in m_pm for p in erm_pairs):
            print(f"    {name:18s}  pair mismatch — skipping")
            continue
        m_churn = np.array([m_pm[p]["id_churn"] for p in erm_pairs])
        m_kl    = np.array([m_pm[p]["id_sym_kl"] for p in erm_pairs])
        ci_c = bootstrap_paired(m_churn - erm_churn)
        ci_k = bootstrap_paired(m_kl - erm_kl)
        rel_c = 100 * ci_c[0] / summary["ERM"]["id_churn"][0]
        rel_k = 100 * ci_k[0] / summary["ERM"]["id_sym_kl"][0]
        print(f"    {name:18s}  Δ churn {fmt_ci(ci_c, pct=True)} pp ({rel_c:+.1f}%)  "
              f"Δ sym_kl {fmt_ci(ci_k)} ({rel_k:+.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: analyze_dataset.py <dataset>")
        sys.exit(1)
    main(sys.argv[1])
