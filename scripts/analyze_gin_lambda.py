"""Per-lambda summary on BACE-GIN: pick the rule-selected lambda.

Design decisions
----------------
This is the architecture-cross-check counterpart to the BACE-MLP λ
sweep used in the paper's main development analysis.  The selection
rule (``largest λ with id-acc ≥ ERM-id-acc - 0.02``) is unchanged; we
re-apply it on the GIN architecture (3 GINConv layers, hidden 128,
mean-pool readout) trained on the same canonical BACE pool.

The expected outcome — and the empirical one — is that the rule
transfers but the numerical λ does not: BACE-MLP picks ``λ=300``,
BACE-GIN picks ``λ=10``.  Same rule, different value, because the
GIN's accuracy-vs-churn curve is shaped differently from the MLP's
(less initial fragility per accuracy-cost step, so the rule's
tolerance threshold bites earlier).

ERM-GIN ``id_acc ≈ 0.739``, so the tolerance threshold is ``≥ 0.719``.
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

    csv_rows = []
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
        within = ci_acc[0] >= erm_mean_acc - 0.02
        within_str = "yes" if within else "no"
        print(f"{lam:>5}  {fmt_ci(ci_acc):>15}  "
              f"{fmt_ci(bootstrap_ci(churn), pct=True):>20}  "
              f"{fmt_ci(bootstrap_ci(kl)):>22}  "
              f"{fmt_ci(d_churn, pct=True)} ({rel:+.1f}%)   {within_str}")
        csv_rows.append({
            "lam": lam, "id_acc": ci_acc[0],
            "id_churn": float(np.mean(churn)),
            "d_churn_mean_pp": d_churn[0] * 100,
            "d_churn_lo_pp": d_churn[1] * 100,
            "d_churn_hi_pp": d_churn[2] * 100,
            "rel_churn_pct": rel,
            "within_tolerance": within,
            "d_acc_pp": (ci_acc[0] - erm_mean_acc) * 100,
        })

    import csv as _csv
    csv_path = Path("outputs/gin_lambda.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rule_lam = max((r["lam"] for r in csv_rows if r["within_tolerance"]),
                   default=None)
    with csv_path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) + ["rule_picked"])
        w.writeheader()
        for r in csv_rows:
            r2 = dict(r); r2["rule_picked"] = (r["lam"] == rule_lam)
            w.writerow(r2)
    print(f"\nWrote {csv_path}  (rule picks lam={rule_lam})")

    # Auto-generate the inline GIN λ-sweep tabular for app:gin so the
    # appendix prose stays in sync with the data.  The hand-typed
    # version it replaces drifted across the GPU re-runs.
    def _f3(t): return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
    def _fpct(t): return f"{t[0]*100:.1f} [{t[1]*100:.1f}, {t[2]*100:.1f}]"
    tex_lines = [
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{rccc}",
        r"\toprule",
        r"$\lambda$ & id-acc & id-churn (\%) & sym-KL \\",
        r"\midrule",
    ]
    for lam in LAMBDAS:
        rs = load_runs(ROOT, f"twin_indep_train*_lam{lam}.npz")
        if not rs:
            continue
        accs, _ = per_run_accuracies(rs)
        ci_acc = bootstrap_ci(accs)
        pm, _ = pairwise_metrics(rs)
        churn = np.array([pm[p]["id_churn"] for p in pairs_erm if p in pm])
        kl = np.array([pm[p]["id_sym_kl"] for p in pairs_erm if p in pm])
        if len(churn) != len(erm_churn):
            continue
        ci_ch = bootstrap_ci(churn)
        ci_kl = bootstrap_ci(kl)
        bold = (lam == rule_lam)
        cells = [_f3(ci_acc), _fpct(ci_ch), _f3(ci_kl)]
        if bold:
            cells = [r"\textbf{" + c + "}" for c in cells]
        tex_lines.append(f"{int(lam):3d} & " + " & ".join(cells) + r" \\")
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}", ""]
    tex_path = Path("paper/sections/tables/gin_lambda.tex")
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(tex_lines))
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
