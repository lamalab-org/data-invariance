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
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="outputs/cross_sample",
                    help="root directory; bace_gin/ subdir contains the NPZs.")
    ap.add_argument("--csv",  default="outputs/bace_gin.csv")
    ap.add_argument("--latex", default="paper/sections/tables/gin.tex")
    args = ap.parse_args()
    data_dir = Path(args.root) / "bace_gin"
    runs = {name: load_runs(data_dir, glob) for name, glob in METHODS.items()}
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
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    print(f"Wrote {csv_path}")

    # Emit the GIN appendix table directly (was hand-typed before;
    # regenerable now so a future retraining auto-updates).
    def _f3(t): return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
    def _fpct(t): return f"{t[0]*100:.1f} [{t[1]*100:.1f}, {t[2]*100:.1f}]"
    def _fpp(ci): return f"{ci[0]*100:+.1f} [{ci[1]*100:+.1f}, {ci[2]*100:+.1f}]"
    erm_d = summary["ERM"]
    bag_d = summary["Bagging-K=5"]
    twin_d = summary["Twin-indep λ=300"]
    # paired Δ vs ERM, same as printed above
    bag_pm, _ = pairwise_metrics(runs["Bagging-K=5"])
    twin_pm, _ = pairwise_metrics(runs["Twin-indep λ=300"])
    bag_dchurn = bootstrap_paired(np.array([bag_pm[p]["id_churn"] for p in erm_pairs]) - erm_churn)
    bag_dkl = bootstrap_paired(np.array([bag_pm[p]["id_sym_kl"] for p in erm_pairs]) - erm_symkl)
    twin_dchurn = bootstrap_paired(np.array([twin_pm[p]["id_churn"] for p in erm_pairs]) - erm_churn)
    twin_dkl = bootstrap_paired(np.array([twin_pm[p]["id_sym_kl"] for p in erm_pairs]) - erm_symkl)
    bag_acc_pp = (bag_d["id_acc"][0] - erm_d["id_acc"][0]) * 100
    twin_acc_pp = (twin_d["id_acc"][0] - erm_d["id_acc"][0]) * 100
    bag_rel = 100 * bag_dchurn[0] / erm_d["id_churn"][0]
    twin_rel = 100 * twin_dchurn[0] / erm_d["id_churn"][0]
    bag_kl_rel = 100 * bag_dkl[0] / erm_d["id_sym_kl"][0]
    twin_kl_rel = 100 * twin_dkl[0] / erm_d["id_sym_kl"][0]

    tex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{\textbf{GIN on BACE: bagging transfers cleanly; "
        r"twin-bootstrap $\lambda$ does not.}  ERM-GIN is more fragile "
        f"than ERM-MLP ({erm_d['id_churn'][0]*100:.1f}\\% "
        r"vs.\ \baceErmChurn\% argmax churn), making the methods more "
        r"rather than less relevant on this backbone.  "
        f"Bagging-$K{{=}}5$ cuts churn $\\ginBagCutLamThreeHundred\\%$ and "
        f"improves id-accuracy by $\\ginBagAccGain$pp.  "
        r"Twin-bootstrap at the $\lambda{=}300$ chosen on the BACE MLP reduces "
        f"sym-KL by ${twin_kl_rel:+.0f}\\%$ but drops id-accuracy by "
        r"$\ginAccDropLamThreeHundred$pp --- well outside the $0.02$ "
        r"selection-rule tolerance ERM-GIN id-acc would impose.  Bold "
        r"cells mark the best mean per column among the three methods.}",
        r"\label{tab:gin}",
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & id-acc & id-churn (\%) & sym-KL \\",
        r"\midrule",
        f"ERM                       & {_f3(erm_d['id_acc'])} & "
        f"{_fpct(erm_d['id_churn'])} & {_f3(erm_d['id_sym_kl'])} \\\\",
        f"Bagging-$K{{=}}5$           & \\textbf{{{_f3(bag_d['id_acc'])}}} & "
        f"\\textbf{{{_fpct(bag_d['id_churn'])}}} & {_f3(bag_d['id_sym_kl'])} \\\\",
        f"Twin-bootstrap $\\lambda{{=}}300$ & {_f3(twin_d['id_acc'])} & "
        f"{_fpct(twin_d['id_churn'])} & "
        f"\\textbf{{{_f3(twin_d['id_sym_kl'])}}} \\\\",
        r"\midrule",
        r"\multicolumn{4}{l}{\emph{Paired $\Delta$ vs.\ ERM (same $45$ seed-pairs)}} \\",
        f"Bagging-$K{{=}}5$           & \\multicolumn{{1}}{{c}}{{${bag_acc_pp:+.2f}$ pp}} & "
        f"{_fpp(bag_dchurn)} (${bag_rel:+.0f}$\\%) & "
        f"{_f3(bag_dkl)} (${bag_kl_rel:+.0f}$\\%) \\\\",
        f"Twin-bootstrap $\\lambda{{=}}300$ & \\multicolumn{{1}}{{c}}{{${twin_acc_pp:+.2f}$ pp}} & "
        f"{_fpp(twin_dchurn)} (${twin_rel:+.0f}$\\%) & "
        f"{_f3(twin_dkl)} (${twin_kl_rel:+.0f}$\\%) \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
        "",
    ]
    tex_path = Path(args.latex)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(tex_lines))
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
