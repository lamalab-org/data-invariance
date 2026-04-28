"""Per-lambda summary on BACE-GIN: pick the rule-selected lambda.

Rule: largest λ ∈ {1, 3, 10, 30, 100, 300} where mean twin-indep id-acc
is within 0.02 of mean ERM-GIN id-acc (0.739).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from _analysis_lib import (
    bootstrap_ci, bootstrap_paired, fmt_ci,
    load_runs, pairwise_metrics, per_run_accuracies,
)

ROOT = Path("outputs/cross_sample/bace_gin")
LAMBDAS = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def main() -> None:
    erm_runs = load_runs(ROOT, "erm_train*.npz")
    erm_accs, _ = per_run_accuracies(erm_runs)
    erm_mean_acc = float(np.mean(erm_accs))
    pm_erm, pairs_erm = pairwise_metrics(erm_runs)
    erm_churn = np.array([pm_erm[p]["id_churn"] for p in pairs_erm])
    erm_kl    = np.array([pm_erm[p]["id_sym_kl"] for p in pairs_erm])

    print(f"ERM-GIN  id_acc {erm_mean_acc:.3f}  "
          f"id_churn {fmt_ci(bootstrap_ci(erm_churn), pct=True)}")
    print()
    print(f"{'lam':>5}  {'id_acc':>15}  {'id_churn (%)':>20}  "
          f"{'sym_kl':>22}  {'Δ churn vs ERM':>22}  {'within 0.02 acc?':>16}")
    print("-" * 110)

    for lam in LAMBDAS:
        runs = load_runs(ROOT, f"twin_indep_train*_lam{lam}.npz")
        if not runs:
            print(f"{lam:>5}  no runs")
            continue
        accs, _ = per_run_accuracies(runs)
        ci_acc = bootstrap_ci(accs)
        pm, _ = pairwise_metrics(runs)
        churn = np.array([pm[p]["id_churn"] for p in pairs_erm if p in pm])
        kl    = np.array([pm[p]["id_sym_kl"] for p in pairs_erm if p in pm])
        if len(churn) != len(erm_churn):
            print(f"{lam:>5}  pair mismatch")
            continue
        d_churn = bootstrap_paired(churn - erm_churn)
        rel = 100 * d_churn[0] / float(np.mean(erm_churn))
        within = "yes" if (ci_acc[0] >= erm_mean_acc - 0.02) else "no"
        print(f"{lam:>5}  {fmt_ci(ci_acc):>15}  "
              f"{fmt_ci(bootstrap_ci(churn), pct=True):>20}  "
              f"{fmt_ci(bootstrap_ci(kl)):>22}  "
              f"{fmt_ci(d_churn, pct=True)} ({rel:+.1f}%)   {within}")


if __name__ == "__main__":
    main()
