"""Per-dataset health filter + paired-bootstrap deltas vs ERM.

Usage:
  python scripts/analyze_dataset.py <dataset>

Prints:
  * ERM id-accuracy + majority-class baseline + health-filter pass/fail
  * Per-method (ERM, bagging-K=5, mc_dropout, twin-indep λ=300):
    id_acc, id_churn, sym_kl with 95% bootstrap CIs
  * Paired Δ vs ERM on the 45 seed-pairs for each method
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
