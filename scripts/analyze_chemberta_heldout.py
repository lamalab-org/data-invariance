"""ChemBERTa held-out: compare ERM, twin-indep λ=300 (failed transfer),
twin-indep λ=10 (rule-selected) on six datasets."""
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


def summarise(ds: str):
    erm = load_runs(ROOT / ds, "erm_train*.npz")
    t300 = load_runs(ROOT / ds, "twin_indep_train*_lam300.0.npz")
    t10 = load_runs(ROOT / ds, "twin_indep_train*_lam10.0.npz")
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


def main():
    print(f"{'Dataset':<25} {'ERM':>14} {'twin λ=300':>22} {'twin λ=10 (rule)':>22}")
    print(f"{'':<25} {'(acc, churn%)':>14} {'(acc, churn%, Δrel)':>22} {'(acc, churn%, Δrel)':>22}")
    print("-" * 95)
    for ds in DATASETS:
        r = summarise(ds)
        if r is None:
            print(f"{ds:<25}  no data"); continue
        s_erm = f"{r['erm_acc']:.2f}, {r['erm_churn']*100:.1f}"
        s300 = f"{r.get('t300_acc',0):.2f}, {r.get('t300_churn',0)*100:.1f}, {r.get('t300_rel',0):+.0f}%"
        s10 = f"{r.get('t10_acc',0):.2f}, {r.get('t10_churn',0)*100:.1f}, {r.get('t10_rel',0):+.0f}%"
        print(f"{ds:<25} {s_erm:>14} {s300:>22} {s10:>22}")


if __name__ == "__main__":
    main()
