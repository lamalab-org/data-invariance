"""Generate the distributional-disagreement (sym-KL) appendix table.

Same structure as the headline argmax-churn table (Table 1), but with
the cell quantity replaced by paired Δ sym-KL between bootstraps:
``KL(f_A(x) || f_B(x)) + KL(f_B(x) || f_A(x))`` averaged over the
canonical id-test, then paired across the 45 seed-pairs and reported
as mean [95% CI].

The headline ``twin-bootstrap reduces sym-KL by another factor of ~8''
claim in §experiments §distributional-disagreement is computed from
this script's CSV output.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from _analysis_lib import (
    GLOBS, bootstrap_paired, load_runs, pairwise_metrics,
)
from paper_constants import (
    DEV_DATASET, FROZEN_LAM, HEADLINE_DATASETS, N_TRAIN, display,
)


METHODS = [
    ("MC dropout",         "mc_dropout_train*_T20.npz"),
    ("Deep Ens.\\ $K{=}5$", GLOBS["deep_ensemble_5"]),
    ("Bagging $K{=}2$",     GLOBS["bagging_2"]),
    ("Bagging $K{=}5$",     GLOBS["bagging_5"]),
    ("Twin-bootstrap $\\lambda{=}300$",
                            f"twin_indep_train*_lam{FROZEN_LAM}.npz"),
]


def _paired_d_sym_kl(ds_dir: Path, glob: str):
    erm = load_runs(ds_dir, GLOBS["erm"])
    m = load_runs(ds_dir, glob)
    if not erm or not m:
        return None
    ep, _ = pairwise_metrics(erm)
    mp, _ = pairwise_metrics(m)
    common = sorted(set(ep).intersection(mp))
    if len(common) < 30:
        return None
    deltas = [mp[p]["id_sym_kl"] - ep[p]["id_sym_kl"] for p in common]
    erm_baseline = float(np.mean([ep[p]["id_sym_kl"] for p in common]))
    return bootstrap_paired(deltas), erm_baseline


def _fmt(d: tuple[float, float, float], baseline: float) -> str:
    m, lo, hi = d
    rel = 100 * m / baseline if baseline > 0 else 0.0
    sig = "$^{*}$" if (lo < 0 and hi < 0) else ""
    return f"{m:+.2f} ({rel:+.0f}\\%){sig}"


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="outputs/cross_sample")
    ap.add_argument("--csv",  default="outputs/distributional.csv")
    ap.add_argument("--latex", default="paper/sections/tables/distributional.tex")
    args = ap.parse_args()
    root = Path(args.root)
    datasets = [DEV_DATASET] + HEADLINE_DATASETS
    sorted_ds = sorted(datasets, key=lambda d: N_TRAIN.get(d, 10**9))

    rows: dict[str, dict[str, str]] = {}
    erm_baselines: dict[str, float] = {}

    for ds in sorted_ds:
        ds_dir = root / ds
        rows[ds] = {}
        for name, glob in METHODS:
            res = _paired_d_sym_kl(ds_dir, glob)
            if res is None:
                rows[ds][name] = "---"
                continue
            d, base = res
            erm_baselines[ds] = base
            rows[ds][name] = _fmt(d, base)

    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \caption{\textbf{\boldmath Twin-bootstrap reduces distributional "
        r"disagreement (sym-KL) by an additional factor of $\sim$$8$ "
        r"beyond the strongest argmax-churn reducer.}  Paired $\Delta$ "
        r"sym-KL vs.\ ERM on the canonical id-test (in nats; "
        r"$\Delta < 0$ better).  Cells show mean $\Delta$ and the "
        r"relative reduction vs.\ ERM in parentheses, over all $45$ "
        r"pairs of $10$ retrainings; ``$^{*}$'' marks cells whose "
        r"$95\%$ paired-bootstrap CI excludes zero.}",
        r"  \label{tab:distributional}",
        r"  \scriptsize",
        r"  \begin{tabular}{lrll@{\hspace{1.0em}}lll}",
        r"    \toprule",
        r"    & & \multicolumn{2}{c}{Parameter-side}"
        r" & \multicolumn{3}{c}{Data-side} \\",
        r"    \cmidrule(lr){3-4} \cmidrule(lr){5-7}",
        r"    Dataset & ERM & " + " & ".join(n for n, _ in METHODS) + r" \\",
        r"    \midrule",
    ]
    for ds in sorted_ds:
        if ds not in rows or ds not in erm_baselines:
            continue
        ds_label = display(ds) + (r"\,(dev)" if ds == DEV_DATASET else "")
        cells = [rows[ds].get(name, "---") for name, _ in METHODS]
        lines.append(
            f"    {ds_label} & {erm_baselines[ds]:.2f} & "
            + " & ".join(cells) + r" \\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    out = Path(args.latex)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")

    # CSV dump for paper-macros and audit. One row per (dataset, method)
    # with the ERM baseline sym-KL and the Δ vs ERM CI.
    import csv as _csv
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["dataset", "method", "erm_sym_kl_baseline",
                    "d_sym_kl_mean", "d_sym_kl_lo", "d_sym_kl_hi",
                    "rel_pct", "fold_reduction"])
        for ds in sorted_ds:
            base = erm_baselines.get(ds)
            if base is None:
                continue
            for name, glob in METHODS:
                res = _paired_d_sym_kl(root / ds, glob)
                if res is None:
                    continue
                d, _ = res
                rel = 100 * d[0] / base if base > 0 else 0.0
                # fold reduction: ERM / final.  d[0] is negative for reducers,
                # so final = base + d[0]; fold = base / max(final, eps).
                final = base + d[0]
                fold = base / final if final > 1e-9 else float("inf")
                w.writerow([ds, name, base, d[0], d[1], d[2], rel, fold])
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
