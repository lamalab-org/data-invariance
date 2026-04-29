"""Waterbirds λ-sweep: pick the rule-selected λ on the vision pretrained backbone.

Design decisions
----------------
This is the third architecture/modality the rule is applied to (after
BACE-MLP picking 300, BACE-GIN picking 10, BACE-ChemBERTa picking 10).
The Waterbirds backbone is ImageNet-pretrained ResNet-50; the dataset
is single-task binary (bird ∈ {landbird, waterbird}) with N=4795.

We sweep the same λ grid as ChemBERTa (``{1, 3, 10, 30, 60, 100,
300}``; existing runs cover ``{30, 60, 100, 300}``).  The rule
(``largest λ with id-acc ≥ ERM-id-acc - 0.02``) and the canonical-data
protocol are unchanged.  The expectation, based on ChemBERTa, is that
λ at the small end of the grid (1, 3, or 10) will satisfy the rule.

ERM-Waterbirds ``id_acc ≈ 0.875``, so the tolerance threshold is
``≥ 0.855``.  At ``λ=30``, id-acc drops to ``0.769`` (well below
tolerance), confirming that the from-scratch λ does not transfer to
this pretrained backbone either.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from _analysis_lib import (
    bootstrap_ci, bootstrap_paired, fmt_ci,
    load_runs, pairwise_metrics, per_run_accuracies,
)

ROOT = Path("outputs/cross_sample/waterbirds")
LAMBDAS = [1.0, 3.0, 10.0, 30.0, 60.0, 100.0, 300.0]


def main() -> None:
    erm = load_runs(ROOT, "erm_train*.npz")
    bag5 = load_runs(ROOT, "bagging_train*_K5.npz")
    erm_accs, _ = per_run_accuracies(erm)
    erm_mean = float(np.mean(erm_accs))
    pm_e, pairs_e = pairwise_metrics(erm)
    erm_churn = np.array([pm_e[p]["id_churn"] for p in pairs_e])

    print(f"ERM-Waterbirds  id_acc {erm_mean:.3f}  "
          f"id_churn {fmt_ci(bootstrap_ci(erm_churn), pct=True)}")
    if bag5:
        accs, _ = per_run_accuracies(bag5)
        pm, _ = pairwise_metrics(bag5)
        ch = np.array([pm[p]["id_churn"] for p in pairs_e if p in pm])
        ci_d = bootstrap_paired(ch - erm_churn) if len(ch) == len(erm_churn) else None
        rel = 100 * ci_d[0] / float(np.mean(erm_churn)) if ci_d else float("nan")
        print(f"Bagging-K=5      id_acc {np.mean(accs):.3f}  "
              f"id_churn {fmt_ci(bootstrap_ci(ch), pct=True)}  rel {rel:+.0f}%")
    tol = 0.02
    print(f"\nRule: largest λ with id_acc ≥ {erm_mean - tol:.3f} (tol={tol})\n")
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
            print(f"{lam:>5.1f}  pair mismatch"); continue
        ci_d = bootstrap_paired(ch - erm_churn)
        rel = 100 * ci_d[0] / float(np.mean(erm_churn))
        within = "yes" if (ci_acc[0] >= erm_mean - tol) else "no"
        print(f"{lam:>5.1f}  {fmt_ci(ci_acc):>15}  "
              f"{fmt_ci(bootstrap_ci(ch), pct=True):>14}  "
              f"{fmt_ci(ci_d, pct=True)} ({rel:+.0f}%)   {within}")


if __name__ == "__main__":
    main()
