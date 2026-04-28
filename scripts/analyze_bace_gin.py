"""Paired-bootstrap analysis for the BACE-GIN architecture cross-check.

Loads ERM, bagging-K=5, twin-indep (lam=300) NPZ files from
outputs/cross_sample/bace_gin/, computes paired Δ id-churn vs ERM with
95% CIs from 10,000-sample bootstrap on the 45 seed-pair distribution,
and reports per-method id-accuracy + churn + sym-KL.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from _analysis_lib import (
    bootstrap_ci,
    bootstrap_paired,
    fmt_ci,
    load_runs,
    pairwise_metrics,
    per_run_accuracies,
)

DATA_DIR = Path("outputs/cross_sample/bace_gin")
METHODS = {
    "ERM":            "erm_train*.npz",
    "Bagging-K=5":    "bagging_train*_K5.npz",
    "Twin-indep λ=300": "twin_indep_train*_lam300.0.npz",
}


def main() -> None:
    runs = {name: load_runs(DATA_DIR, glob) for name, glob in METHODS.items()}
    for name, rs in runs.items():
        print(f"  {name:20s}  loaded {len(rs)} runs")
    print()

    # ── Per-method id-accuracy + churn + sym-KL with bootstrap CIs ──
    summary = {}
    for name, rs in runs.items():
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
        print(f"{name:20s}  id_acc {fmt_ci(summary[name]['id_acc'])}  "
              f"id_churn {fmt_ci(summary[name]['id_churn'], pct=True)}  "
              f"sym_kl {fmt_ci(summary[name]['id_sym_kl'])}")
    print()

    # ── Paired Δ vs ERM (same seed-pairs) ──
    erm_churn = summary["ERM"]["churn_pairs"]
    erm_symkl = summary["ERM"]["symkl_pairs"]
    erm_pairs = summary["ERM"]["pairs"]
    print("Paired Δ vs ERM (same 45 seed-pairs):")
    for name in ("Bagging-K=5", "Twin-indep λ=300"):
        # Reorder by ERM's pair list to ensure pairing is identical.
        m_pm, _ = pairwise_metrics(runs[name])
        m_churn = np.array([m_pm[p]["id_churn"] for p in erm_pairs])
        m_symkl = np.array([m_pm[p]["id_sym_kl"] for p in erm_pairs])
        d_churn = m_churn - erm_churn
        d_symkl = m_symkl - erm_symkl
        ci_c = bootstrap_paired(d_churn)
        ci_k = bootstrap_paired(d_symkl)
        rel_churn = 100 * ci_c[0] / summary["ERM"]["id_churn"][0]
        rel_kl    = 100 * ci_k[0] / summary["ERM"]["id_sym_kl"][0]
        print(f"  {name:20s}  Δ id_churn {fmt_ci(ci_c, pct=True)} pp "
              f"({rel_churn:+.1f}%)   Δ sym_kl {fmt_ci(ci_k)} ({rel_kl:+.1f}%)")


if __name__ == "__main__":
    main()
