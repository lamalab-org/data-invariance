"""λ-sweep analysis for BACE-ChemBERTa: pick the rule-selected λ."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from _analysis_lib import (
    bootstrap_ci, bootstrap_paired, fmt_ci,
    load_runs, pairwise_metrics, per_run_accuracies,
)

ROOT = Path("outputs/cross_sample/bace_chemberta")
LAMBDAS = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def main() -> None:
    erm = load_runs(ROOT, "erm_train*.npz")
    erm_accs, _ = per_run_accuracies(erm)
    erm_mean = float(np.mean(erm_accs))
    pm_e, pairs_e = pairwise_metrics(erm)
    erm_churn = np.array([pm_e[p]["id_churn"] for p in pairs_e])

    print(f"ERM-ChemBERTa  id_acc {erm_mean:.3f}  "
          f"id_churn {fmt_ci(bootstrap_ci(erm_churn), pct=True)}")
    tol = 0.02
    print(f"\nRule: largest λ with id_acc ≥ {erm_mean - tol:.3f}\n")
    print(f"{'λ':>5}  {'id_acc':>15}  {'id_churn':>14}  "
          f"{'Δ vs ERM':>22}  {'within tol?':>12}")
    print("-" * 80)
    for lam in LAMBDAS:
        rs = load_runs(ROOT, f"twin_indep_train*_lam{lam}.npz")
        if len(rs) < 2:
            print(f"{lam:>5.1f}  no runs"); continue
        accs, _ = per_run_accuracies(rs)
        ci_acc = bootstrap_ci(accs)
        pm, _ = pairwise_metrics(rs)
        ch = np.array([pm[p]["id_churn"] for p in pairs_e if p in pm])
        if len(ch) != len(erm_churn):
            print(f"{lam:>5.1f}  pair mismatch ({len(ch)} vs {len(erm_churn)})")
            continue
        ci_d = bootstrap_paired(ch - erm_churn)
        rel = 100 * ci_d[0] / float(np.mean(erm_churn))
        within = "yes" if (ci_acc[0] >= erm_mean - tol) else "no"
        print(f"{lam:>5.1f}  {fmt_ci(ci_acc):>15}  "
              f"{fmt_ci(bootstrap_ci(ch), pct=True):>14}  "
              f"{fmt_ci(ci_d, pct=True)} ({rel:+.0f}%)   {within}")


if __name__ == "__main__":
    main()
