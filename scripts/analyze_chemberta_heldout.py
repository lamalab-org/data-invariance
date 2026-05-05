"""ChemBERTa held-out: compare ERM, twin-indep λ=300, twin-indep λ=10.

Design decisions
----------------
Closes the pretrained-backbone scope loop on the SMILES modality.
For each ChemBERTa dataset, we report three columns:

* **ERM** as the baseline (ChemBERTa fine-tuned, no consistency loss).
* **Twin-indep λ=300** (the BACE-MLP-frozen value).  Documents the
  failed transfer: this λ over-regularises pretrained features
  because pretraining has already collapsed most cross-sample
  variation, and a λ tuned on a from-scratch network is too strong.
* **Twin-indep λ=10** (the rule-selected value on BACE-ChemBERTa).
  Documents the closed loop: at the rule-selected λ, the method
  works on every ChemBERTa dataset (accuracy preserved within 2pp,
  churn reduced 15-82%).

The deltas are absolute pp churn changes vs ERM, paired over the
``binom(5, 2) = 10`` seed pairs that the cluster sweep generates
(ChemBERTa runs use 5 train_seeds rather than 10 for compute reasons).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from _analysis_lib import (
    bootstrap_ci, bootstrap_paired, fmt_ci,
    load_runs, pairwise_metrics, per_run_accuracies,
)

DATASETS = ["bace_chemberta", "bbbp_chemberta", "pgp_broccatelli_chemberta",
            "bbb_martins_chemberta", "ames_chemberta", "dili_chemberta"]
ROOT = Path("outputs/cross_sample")


def summarise(ds: str, root: Path = ROOT):
    erm = load_runs(root / ds, "erm_train*.npz")
    t300 = load_runs(root / ds, "twin_indep_train*_lam300.0.npz")
    t10 = load_runs(root / ds, "twin_indep_train*_lam10.0.npz")
    if not erm or not t10:
        return None

    pm_e, pairs_e = pairwise_metrics(erm)
    erm_churn = np.array([pm_e[p]["id_churn"] for p in pairs_e])
    erm_accs, _ = per_run_accuracies(erm)

    out = {"erm_acc": float(np.mean(erm_accs)),
           "erm_churn": float(np.mean(erm_churn))}

    for name, runs in [("t300", t300), ("t10", t10)]:
        if not runs:
            continue
        accs, _ = per_run_accuracies(runs)
        pm, _ = pairwise_metrics(runs)
        ch = np.array([pm[p]["id_churn"] for p in pairs_e if p in pm])
        d_ch = ch - erm_churn[:len(ch)] if len(ch) == len(erm_churn) else None
        out[f"{name}_acc"] = float(np.mean(accs))
        out[f"{name}_churn"] = float(np.mean(ch))
        if d_ch is not None:
            ci = bootstrap_paired(d_ch)
            out[f"{name}_dchurn_ci"] = ci
            out[f"{name}_rel"] = 100 * ci[0] / out["erm_churn"]
    return out


_DISPLAY = {
    "bace_chemberta":            "BACE",
    "bbbp_chemberta":             "BBBP",
    "pgp_broccatelli_chemberta": "Pgp",
    "bbb_martins_chemberta":     "BBB-Martins",
    "ames_chemberta":             "AMES",
    "dili_chemberta":             "DILI",
}


def _fmt_dchurn_pp_ci(ci: tuple[float, float, float]) -> str:
    """Format a paired-Δ churn CI in percentage points."""
    if ci is None:
        return "---"
    m, lo, hi = ci
    return f"{m*100:+.1f} [{lo*100:+.1f}, {hi*100:+.1f}]"


def write_latex_table(rows: list[dict], path: Path) -> None:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{\textbf{\boldmath The rule transfers; the value $\lambda$ takes "
        r"does not.}  At the $\lambda{=}300$ chosen on the BACE MLP, "
        r"twin-bootstrap over-regularises ChemBERTa (accuracy drops "
        r"$9\text{--}17$\,pp; churn rises on $5/6$ datasets, BBBP "
        r"collapses to majority).  Re-applying the same $0.02$-tolerance "
        r"rule on BACE-ChemBERTa picks $\lambda{=}10$, at which "
        r"twin-bootstrap preserves accuracy (within $2$\,pp of ERM) "
        r"and cuts churn $15\text{--}82\%$ on every dataset.  Paired "
        r"$\Delta$ churn columns report mean $[\,95\%\ \text{CI}\,]$ in "
        r"percentage points over $\binom{5}{2}{=}10$ seed pairs.}",
        r"\label{tab:chemberta}",
        r"\small",
        r"\begin{tabular}{lc@{\hspace{0.8em}}cc@{\hspace{0.8em}}cc}",
        r"\toprule",
        r" & \multicolumn{1}{c}{ERM}"
        r" & \multicolumn{2}{c}{Twin-bootstrap $\lambda{=}300$}"
        r" & \multicolumn{2}{c}{Twin-bootstrap $\lambda{=}10$ (rule)} \\",
        r"\cmidrule(lr){2-2} \cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"Dataset & churn (\%) & acc & $\Delta$ churn (pp) & acc & $\Delta$ churn (pp) \\",
        r"\midrule",
    ]
    for r in rows:
        ds_name = _DISPLAY.get(r["dataset"], r["dataset"])
        bbbp_collapse = r["dataset"] == "bbbp_chemberta"
        d300 = _fmt_dchurn_pp_ci(r.get("t300_dchurn_ci"))
        d10 = _fmt_dchurn_pp_ci(r.get("t10_dchurn_ci"))
        if bbbp_collapse:
            d300 = d300 + r"$^{*}$"
        lines.append(
            f"    {ds_name} & {r['erm_churn']*100:.1f} & "
            f"{r.get('t300_acc',0):.2f} & {d300} & "
            f"{r.get('t10_acc',0):.2f} & {d10} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\multicolumn{6}{l}{\footnotesize $^{*}$BBBP at $\lambda{=}300$ "
        r"collapses to the majority-class predictor (acc $0.78{=}$ majority), "
        r"so the churn drop is meaningless.}",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--csv",  default="outputs/chemberta_heldout.csv")
    ap.add_argument("--latex", default="paper/sections/tables/chemberta.tex")
    args = ap.parse_args()
    root = Path(args.root)
    print(f"{'Dataset':<25} {'ERM':>14} {'twin λ=300':>22} {'twin λ=10 (rule)':>22}")
    print(f"{'':<25} {'(acc, churn%)':>14} {'(acc, churn%, Δrel)':>22} {'(acc, churn%, Δrel)':>22}")
    print("-" * 95)
    rows = []
    for ds in DATASETS:
        r = summarise(ds, root)
        if r is None:
            print(f"{ds:<25}  no data"); continue
        r["dataset"] = ds
        rows.append(r)
        s_erm = f"{r['erm_acc']:.2f}, {r['erm_churn']*100:.1f}"
        s300 = f"{r.get('t300_acc',0):.2f}, {r.get('t300_churn',0)*100:.1f}, {r.get('t300_rel',0):+.0f}%"
        s10 = f"{r.get('t10_acc',0):.2f}, {r.get('t10_churn',0)*100:.1f}, {r.get('t10_rel',0):+.0f}%"
        print(f"{ds:<25} {s_erm:>14} {s300:>22} {s10:>22}")

    if rows:
        write_latex_table(rows, Path(args.latex))
        # CSV dump for paper-macros and audit. One row per dataset, all
        # numbers the prose quotes (per-dataset ERM acc/churn, twin acc /
        # churn / paired Δ churn / relative reduction at both λ values).
        import csv
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["dataset", "erm_acc", "erm_churn",
                "t300_acc", "t300_churn", "t300_dchurn_mean", "t300_dchurn_lo",
                "t300_dchurn_hi", "t300_rel",
                "t10_acc", "t10_churn", "t10_dchurn_mean", "t10_dchurn_lo",
                "t10_dchurn_hi", "t10_rel"]
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                ci300 = r.get("t300_dchurn_ci") or (None, None, None)
                ci10 = r.get("t10_dchurn_ci") or (None, None, None)
                w.writerow([
                    r["dataset"], r.get("erm_acc"), r.get("erm_churn"),
                    r.get("t300_acc"), r.get("t300_churn"),
                    ci300[0], ci300[1], ci300[2], r.get("t300_rel"),
                    r.get("t10_acc"), r.get("t10_churn"),
                    ci10[0], ci10[1], ci10[2], r.get("t10_rel"),
                ])
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
