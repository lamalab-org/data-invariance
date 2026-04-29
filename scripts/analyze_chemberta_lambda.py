"""λ-sweep analysis for BACE-ChemBERTa.

Design decisions
----------------
The selection rule (``largest λ with id-acc ≥ ERM-id-acc - 0.02``)
was originally formulated on BACE-MLP, where it picks ``λ=300``.
This script applies the *same* rule to BACE-ChemBERTa to test whether
the rule transfers across architectures and modalities (Morgan
fingerprint MLP → SMILES-tokenised transformer fine-tune).  The
prediction is that the rule transfers but the numerical λ does not
(empirically: ``λ=10`` on ChemBERTa, vs ``λ=300`` on MLP).

The 0.02 tolerance, the canonical-data protocol (seed 99, 10 train
seeds), and the paired-bootstrap reporting are all unchanged from
the MLP sweep.  Only the ``λ`` grid here covers ``{1, 3, 10, 30, 100,
300}`` so the rule's chosen value is always within the swept range.
"""
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
