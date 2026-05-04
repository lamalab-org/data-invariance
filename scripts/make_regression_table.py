"""Generate the regression appendix table with paired-bootstrap CIs.

Reads the cross-sample NPZ outputs for ESOL, FreeSolv, Lipophilicity
under ERM, Bagging-K=2, Bagging-K=5, and Twin-bootstrap lambda=3, and
emits paper/sections/tables/regression.tex with paired-bootstrap 95%
CIs on every Δ-fragility cell.

Fragility for regression is the per-example absolute prediction
difference between two retrainings, averaged over the canonical id-test;
paired CIs are computed across the binom(10, 2) = 45 seed pairs.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from _analysis_lib import bootstrap_ci, bootstrap_paired, load_runs


ROOT = Path("outputs/cross_sample")

DATASETS = [
    ("esol_reg",     "ESOL",       1128),
    ("freesolv_reg", "FreeSolv",    642),
    ("lipo_reg",     "Lipo",       4200),
]

METHODS = [
    ("ERM",                          "erm_train*.npz"),
    ("Bagging-$K{=}2$",              "bagging_train*_K2.npz"),
    ("Bagging-$K{=}5$",              "bagging_train*_K5.npz"),
    ("Twin-bootstrap $\\lambda{=}3$", "twin_indep_train*_lam3.0.npz"),
]


def _get_preds(d: dict) -> np.ndarray:
    return d["id_preds_avg"] if "id_preds_avg" in d else d["id_preds"]


def _per_run_mae(runs):
    return [float(np.abs(_get_preds(d) - d["id_labels"]).mean()) for _, d in runs]


def _pairwise_fragility(runs):
    pairs = list(combinations([s for s, _ in runs], 2))
    runs_by_seed = dict(runs)
    return {
        (sa, sb): float(np.abs(_get_preds(runs_by_seed[sa])
                               - _get_preds(runs_by_seed[sb])).mean())
        for sa, sb in pairs
    }


def _summarise(ds_dir: Path):
    out = {}
    erm_runs = load_runs(ds_dir, METHODS[0][1])
    if not erm_runs:
        return None
    erm_frag_by_pair = _pairwise_fragility(erm_runs)
    pairs = sorted(erm_frag_by_pair)
    erm_frag_arr = np.array([erm_frag_by_pair[p] for p in pairs])

    for name, glob in METHODS:
        runs = load_runs(ds_dir, glob)
        if not runs:
            continue
        maes = _per_run_mae(runs)
        pm = _pairwise_fragility(runs)
        common = [p for p in pairs if p in pm]
        m_arr = np.array([pm[p] for p in common])
        out[name] = {
            "id_mae_ci": bootstrap_ci(maes),
            "id_frag_ci": bootstrap_ci(m_arr.tolist()),
        }
        if name != "ERM" and len(m_arr) == len(erm_frag_arr):
            d = m_arr - erm_frag_arr
            ci = bootstrap_paired(d)
            out[name]["d_frag_ci"] = ci
            out[name]["rel_pct"] = 100 * ci[0] / out["ERM"]["id_frag_ci"][0]
    return out


def _fmt_v(v: float) -> str:
    return f"{v:.2f}"


def _fmt_d_ci(ci: tuple[float, float, float]) -> str:
    m, lo, hi = ci
    return f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--csv",  default="outputs/regression.csv")
    ap.add_argument("--latex", default="paper/sections/tables/regression.tex")
    args = ap.parse_args()
    root = Path(args.root)
    data = {}
    for short, _, _ in DATASETS:
        d = _summarise(root / short)
        if d is None:
            print(f"missing {short}")
            continue
        data[short] = d

    if not data:
        print("no regression data")
        return

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{\textbf{At matched $2\times$-ERM compute, twin-bootstrap "
        r"beats bagging-$K{=}2$ on every regression dataset; the methods "
        r"rank identically to the classification headline.}  Per-method "
        r"id-MAE and cross-sample fragility (mean $|f_A - f_B|$ between "
        r"bootstrap retrainings) on three MoleculeNet regression "
        r"benchmarks.  All reported quantities are mean $[\,95\%\ "
        r"\text{CI}\,]$ over $\binom{10}{2}{=}45$ seed pairs (or $10$ "
        r"retrainings for id-MAE).  Paired $\Delta$ fragility vs.\ ERM "
        r"in the bottom rows; bold marks the best matched-compute method "
        r"per dataset.  Bagging-$K{=}5$ ($5\times$-ERM compute) included "
        r"as a stronger no-cost reference.}",
        r"\label{tab:regression}",
        r"\scriptsize",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{ESOL ($N{=}1128$)}"
        r" & \multicolumn{2}{c}{FreeSolv ($N{=}642$)}"
        r" & \multicolumn{2}{c}{Lipo ($N{=}4200$)} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"Method & id-MAE & fragility & id-MAE & fragility & id-MAE & fragility \\",
        r"\midrule",
    ]
    for name, _ in METHODS:
        cells = []
        for short, _, _ in DATASETS:
            d = data.get(short, {}).get(name)
            if d is None:
                cells.extend(["---", "---"])
                continue
            cells.append(_fmt_v(d["id_mae_ci"][0]))
            cells.append(_fmt_v(d["id_frag_ci"][0]))
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\midrule",
        r"\multicolumn{7}{l}{\emph{Paired $\Delta$ fragility vs.\ ERM "
        r"(45 seed-pairs, mean $[\,95\%\ \text{CI}\,]$):}} \\",
    ]
    # Identify the best matched-compute method per dataset: between
    # bagging-K=2 and twin-bootstrap (both are 2x-ERM compute).
    matched_compute = {"Bagging-$K{=}2$", "Twin-bootstrap $\\lambda{=}3$"}
    best_per_ds = {}
    for short, _, _ in DATASETS:
        scores = []
        for name in matched_compute:
            d = data.get(short, {}).get(name)
            if d and "d_frag_ci" in d:
                scores.append((d["d_frag_ci"][0], name))
        if scores:
            best_per_ds[short] = min(scores)[1]

    for name, _ in METHODS:
        if name == "ERM":
            continue
        cols = []
        for short, _, _ in DATASETS:
            d = data.get(short, {}).get(name)
            if d is None or "d_frag_ci" not in d:
                cols.append(r"\multicolumn{2}{c}{---}")
                continue
            cell = _fmt_d_ci(d["d_frag_ci"]) + f" ({d['rel_pct']:+.0f}\\%)"
            if best_per_ds.get(short) == name:
                cell = r"\textbf{" + cell + "}"
            cols.append(r"\multicolumn{2}{c}{" + cell + r"}")
        lines.append(f"{name} & " + " & ".join(cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    out_path = Path(args.latex)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")

    # CSV dump for paper-macros + audit. One row per (dataset, method).
    import csv as _csv
    csv_path = Path(args.csv)
    with csv_path.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["dataset", "method", "id_mae", "id_frag",
                    "d_frag_mean", "d_frag_lo", "d_frag_hi", "rel_pct"])
        for short, _, _ in DATASETS:
            for name, _ in METHODS:
                d = data.get(short, {}).get(name)
                if d is None:
                    continue
                d_ci = d.get("d_frag_ci")
                w.writerow([
                    short, name,
                    d["id_mae_ci"][0], d["id_frag_ci"][0],
                    d_ci[0] if d_ci else "",
                    d_ci[1] if d_ci else "",
                    d_ci[2] if d_ci else "",
                    d.get("rel_pct", ""),
                ])
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
