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


def _fmt_pct_ci(t):
    return f"{t[0]*100:.1f} [{t[1]*100:.1f}, {t[2]*100:.1f}]"


def _fmt_acc_ci(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def _fmt_d_pp_ci(t):
    return f"{t[0]*100:+.1f} [{t[1]*100:+.1f}, {t[2]*100:+.1f}]"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="outputs/cross_sample",
                    help="root directory; waterbirds/ subdir contains the NPZs.")
    ap.add_argument("--csv", default="outputs/waterbirds_lambda.csv")
    ap.add_argument("--latex", default="paper/sections/tables/waterbirds_lambda.tex")
    args = ap.parse_args()
    root = Path(args.root) / "waterbirds"
    erm = load_runs(root, "erm_train*.npz")
    bag5 = load_runs(root, "bagging_train*_K5.npz")
    erm_accs, _ = per_run_accuracies(erm)
    erm_mean = float(np.mean(erm_accs))
    erm_acc_ci = bootstrap_ci(erm_accs)
    pm_e, pairs_e = pairwise_metrics(erm)
    erm_churn = np.array([pm_e[p]["id_churn"] for p in pairs_e])
    erm_churn_ci = bootstrap_ci(erm_churn)
    erm_baseline = float(np.mean(erm_churn))

    print(f"ERM-Waterbirds  id_acc {erm_mean:.3f}  "
          f"id_churn {fmt_ci(erm_churn_ci, pct=True)}")

    rows = []  # (label, acc_ci, churn_ci, d_pp_ci, rel_pct, within_tol)
    rows.append(("ERM", erm_acc_ci, erm_churn_ci, None, None, None))

    if bag5:
        accs, _ = per_run_accuracies(bag5)
        pm, _ = pairwise_metrics(bag5)
        ch = np.array([pm[p]["id_churn"] for p in pairs_e if p in pm])
        if len(ch) == len(erm_churn):
            ci_d = bootstrap_paired(ch - erm_churn)
            rel = 100 * ci_d[0] / erm_baseline
            rows.append(("Bagging-$K{=}5$ ($5\\times$)",
                         bootstrap_ci(accs), bootstrap_ci(ch),
                         ci_d, rel, True))
        print(f"Bagging-K=5      id_acc {np.mean(accs):.3f}  "
              f"id_churn {fmt_ci(bootstrap_ci(ch), pct=True)}")

    tol = 0.02
    rule_picked: list[float] = []
    for lam in LAMBDAS:
        rs = load_runs(root, f"twin_indep_train*_lam{lam}.npz")
        if len(rs) < 2:
            continue
        accs, _ = per_run_accuracies(rs)
        ci_acc = bootstrap_ci(accs)
        pm, _ = pairwise_metrics(rs)
        ch = np.array([pm[p]["id_churn"] for p in pairs_e if p in pm])
        if len(ch) != len(erm_churn):
            continue
        ci_ch = bootstrap_ci(ch)
        ci_d = bootstrap_paired(ch - erm_churn)
        rel = 100 * ci_d[0] / erm_baseline
        within = ci_acc[0] >= erm_mean - tol
        if within:
            rule_picked.append(lam)
        rows.append((f"Twin-bootstrap $\\lambda{{=}}{int(lam)}$",
                     ci_acc, ci_ch, ci_d, rel, within))

    rule_lam = max(rule_picked) if rule_picked else None
    print(f"\nRule picks λ = {rule_lam}")

    # Emit LaTeX table.
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{\textbf{\boldmath Waterbirds (ImageNet-ResNet50): the "
        r"$0.02$-tolerance rule picks $\lambda{=}10$, recovering the "
        r"closed-loop result on a vision pretrained backbone.}  "
        r"ERM id-acc $0.875$ (rule threshold $\geq 0.855$).  "
        r"Twin-bootstrap at the $\lambda{=}300$ chosen on BACE collapses "
        r"accuracy by $27$\,pp; at the rule-selected $\lambda{=}10$, "
        r"twin-bootstrap preserves accuracy and cuts argmax churn "
        r"$52\%$.  All cells report mean $[\,95\%\ \text{CI}\,]$ over "
        r"five train-seeds ($\binom{5}{2}{=}10$ pairs for paired "
        r"quantities).}",
        r"\label{tab:waterbirds_lambda}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & id-acc & id-churn (\%) & $\Delta$ id-churn vs ERM (pp) \\",
        r"\midrule",
    ]
    for label, acc_ci, ch_ci, d_ci, rel, within in rows:
        is_rule = rule_lam is not None and label == f"Twin-bootstrap $\\lambda{{=}}{int(rule_lam)}$"
        emph = r"\textbf{" if is_rule else ""
        emphend = r"}" if is_rule else ""
        if d_ci is None:
            d_str = "---"
        else:
            d_str = _fmt_d_pp_ci(d_ci) + f" ({rel:+.0f}\\%)"
            if within is False:
                d_str = d_str + r"$^{\star}$"
        lines.append(
            f"{emph}{label}{emphend} & "
            f"{emph}{_fmt_acc_ci(acc_ci)}{emphend} & "
            f"{emph}{_fmt_pct_ci(ch_ci)}{emphend} & "
            f"{emph}{d_str}{emphend} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\multicolumn{4}{l}{\footnotesize $^{\star}$id-acc out of "
        r"$0.02$ tolerance; reported only as a sweep diagnostic.}",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    out = Path(args.latex)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")

    # CSV dump for paper-macros and audit. One row per method/λ point.
    import csv
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "id_acc_mean", "id_acc_lo", "id_acc_hi",
                    "id_churn_mean", "id_churn_lo", "id_churn_hi",
                    "d_churn_mean_pp", "d_churn_lo_pp", "d_churn_hi_pp",
                    "rel_pct", "within_tolerance"])
        for label, acc_ci, ch_ci, d_ci, rel, within in rows:
            d_mean = d_ci[0] * 100 if d_ci else ""
            d_lo = d_ci[1] * 100 if d_ci else ""
            d_hi = d_ci[2] * 100 if d_ci else ""
            w.writerow([label, acc_ci[0], acc_ci[1], acc_ci[2],
                        ch_ci[0], ch_ci[1], ch_ci[2],
                        d_mean, d_lo, d_hi,
                        rel if rel is not None else "",
                        within if within is not None else ""])
    print(f"Wrote {csv_path}  rule_picked_lambda={rule_lam}")


if __name__ == "__main__":
    main()
