"""Paired-bootstrap analysis for the BACE-GIN architecture cross-check.

Design decisions
----------------
Reports ERM, bagging-K=5, and twin-indep at lam=300 (the failed
transfer from BACE-MLP) on the GIN architecture.  The companion
script ``analyze_gin_lambda.py`` runs the rule-selection sweep on
twin-indep alone; this script reports the original three-method
comparison at the unchanged BACE-MLP-frozen lam=300 so the GIN
appendix table can show both sides of the closed loop:

  - lam=300 (failed transfer): twin-indep over-regularises GIN
  - lam=10  (rule-selected, see analyze_gin_lambda.py): works

Numbers in this script feed directly into the GIN appendix table
in ``paper/sections/appendix.tex``.
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
    csv_rows = []
    csv_rows.append({
        "method": "ERM",
        "id_acc_mean": summary["ERM"]["id_acc"][0],
        "id_churn_mean": summary["ERM"]["id_churn"][0],
        "id_sym_kl_mean": summary["ERM"]["id_sym_kl"][0],
        "d_churn_mean_pp": "", "d_churn_lo_pp": "", "d_churn_hi_pp": "",
        "d_acc_pp": 0.0,
        "rel_churn_pct": "", "rel_kl_pct": "",
    })
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
        d_acc_pp = (summary[name]["id_acc"][0] - summary["ERM"]["id_acc"][0]) * 100
        print(f"  {name:20s}  Δ id_churn {fmt_ci(ci_c, pct=True)} pp "
              f"({rel_churn:+.1f}%)   Δ sym_kl {fmt_ci(ci_k)} ({rel_kl:+.1f}%)")
        csv_rows.append({
            "method": name,
            "id_acc_mean": summary[name]["id_acc"][0],
            "id_churn_mean": summary[name]["id_churn"][0],
            "id_sym_kl_mean": summary[name]["id_sym_kl"][0],
            "d_churn_mean_pp": ci_c[0] * 100,
            "d_churn_lo_pp": ci_c[1] * 100,
            "d_churn_hi_pp": ci_c[2] * 100,
            "d_acc_pp": d_acc_pp,
            "rel_churn_pct": rel_churn,
            "rel_kl_pct": rel_kl,
        })

    # CSV dump for paper-macros and audit.
    import csv as _csv
    csv_path = Path("outputs/bace_gin.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
