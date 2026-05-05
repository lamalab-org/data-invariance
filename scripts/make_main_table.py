"""Paper Table 1: cross-sample fragility on held-out datasets.

Method comparison under the strict dev/test split:
  - λ for ``twin_indep`` is FROZEN at the value chosen on BACE (development).
  - All other hyperparameters are dataset-default (no tuning).
  - Held-out datasets are reported here; BACE (dev) is reported separately
    by ``make_pareto.py``.

For each (dataset, method) we report:
  - id_acc, ood_acc          mean ± bootstrap 95 % CI across train_seeds.
  - id_churn, ood_churn      mean ± bootstrap 95 % CI across seed pairs.
  - id_sym_kl, ood_sym_kl    same, distributional disagreement.
  - paired Δ vs ERM and vs Bagging K=2 with paired-bootstrap 95 % CI;
    ``**`` flags Δ whose CI excludes 0.

Usage::

    uv run python scripts/make_main_table.py \\
        --datasets bbbp tadf mof_solvent mof_thermal \\
        --frozen_lam 300.0
"""
from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np

from _analysis_lib import (
    GLOBS, bootstrap_ci, bootstrap_paired, fmt_ci, get_probs, load_runs,
    pairwise_metrics, per_run_accuracies,
)
from paper_constants import (
    DEV_DATASET, FROZEN_LAM, HEADLINE_DATASETS, N_TRAIN, display,
)

_CITE = {
    "bace":            "TODO_wu2018_moleculenet",
    "bbbp":            "TODO_wu2018_moleculenet",
    "tadf":            "TODO_tadf_dataset",
    "mof_thermal":     "TODO_mof_thermal_dataset",
    "mof_solvent":     "TODO_mof_solvent_dataset",
    "perovskite":      "TODO_perovskite_dataset",
    "battery":         "TODO_battery_dataset",
    "dili":            "TODO_huang2021_tdc",
    "pgp_broccatelli": "TODO_huang2021_tdc",
    "bbb_martins":     "TODO_huang2021_tdc",
    "ames":            "TODO_huang2021_tdc",
}


def _cited(ds: str) -> str:
    cite = _CITE.get(ds)
    return f"{display(ds)}~\\citep{{{cite}}}" if cite else display(ds)


# ---------------------------------------------------------------------------
# Per-dataset analysis
# ---------------------------------------------------------------------------

def metrics_with_cis(dataset_dir: Path, glob: str):
    """Return mean+CI for accuracy/churn/sym_kl for one method on one dataset."""
    runs = load_runs(dataset_dir, glob)
    if not runs:
        return None
    id_accs, ood_accs = per_run_accuracies(runs)
    pair_metrics, _ = pairwise_metrics(runs)
    rows = list(pair_metrics.values())
    return {
        "n_seeds": len(runs),
        "id_acc":    bootstrap_ci(id_accs),
        "ood_acc":   bootstrap_ci(ood_accs),
        "id_churn":  bootstrap_ci(r["id_churn"] for r in rows),
        "ood_churn": bootstrap_ci(r["ood_churn"] for r in rows),
        "id_sym_kl": bootstrap_ci(r["id_sym_kl"] for r in rows),
        "ood_sym_kl": bootstrap_ci(r["ood_sym_kl"] for r in rows),
    }


def paired_delta_vs(reference_glob: str, method_glob: str,
                    dataset_dir: Path, key: str):
    """Return paired-bootstrap (mean, lo, hi) for (method − reference) on `key`.

    Pairs are matched by (seed_a, seed_b); both methods must have run on the
    same train_seeds. ``key`` is one of {id_churn, ood_churn, id_sym_kl, ood_sym_kl}.
    """
    ref_runs = load_runs(dataset_dir, reference_glob)
    m_runs   = load_runs(dataset_dir, method_glob)
    if not ref_runs or not m_runs:
        return None
    ref_pairs, _ = pairwise_metrics(ref_runs)
    m_pairs,   _ = pairwise_metrics(m_runs)
    common = sorted(set(ref_pairs).intersection(m_pairs))
    if not common:
        return None
    deltas = [m_pairs[p][key] - ref_pairs[p][key] for p in common]
    return bootstrap_paired(deltas), len(deltas)


# ---------------------------------------------------------------------------
# Pretty printing + CSV
# ---------------------------------------------------------------------------

METHODS = [
    ("ERM",              GLOBS["erm"]),
    ("SWA",              "swa_train*.npz"),
    ("MC dropout",        "mc_dropout_train*_T20.npz"),
    ("Deep Ensemble K=5", GLOBS["deep_ensemble_5"]),
    ("Bagging K=2",       GLOBS["bagging_2"]),
    ("Bagging K=5",       GLOBS["bagging_5"]),
    # Twin_indep glob is templated on lam at run-time.
]


def _fmt_delta_pp(t: tuple[float, float, float]) -> str:
    """Format a paired-Δ id-churn (mean, lo, hi) tuple as `m [lo, hi]` in pp.

    All in text mode (no $...$ math) so that \\textbf{} bolds the entire
    cell when wrapped, not just the bracketed range."""
    m, lo, hi = t
    return f"{m*100:+.1f} [{lo*100:+.1f}, {hi*100:+.1f}]"


def write_latex_table(rows, path, frozen_lam):
    """Emit `paper/sections/tables/main.tex`: per-dataset paired Δ id-churn vs ERM."""
    methods = ["MC dropout", "Deep Ensemble K=5", "Bagging K=2", "Bagging K=5",
               f"Twin_indep λ={frozen_lam} (frozen)"]
    headers = ["MC dropout", "Deep Ens.\\ $K{=}5$", "Bagging $K{=}2$",
               "Bagging $K{=}5$", "Twin-bootstrap $\\lambda{=}300$"]
    by_ds: dict[str, dict[str, str]] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], {})[r["method"]] = r

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{\textbf{Bagging and twin-bootstrap reduce the "
        r"class-flip rate on every chemistry benchmark; MC dropout and "
        r"deep ensembles do not.}  Paired $\Delta$ id-churn vs.\ ERM "
        r"in percentage points (negative is better).  Each cell is the "
        r"mean across $\nCanonicalSeeds$ canonical-seed replicates of a "
        r"per-replicate paired-bootstrap estimate (mean over "
        r"$\binom{10}{2}=45$ seed pairs with paired-bootstrap $95\%$ CIs "
        r"from $10{,}000$ resamples); per-replicate values in "
        r"App.~\ref{app:seed_sensitivity}.  \textbf{Best} per dataset in "
        r"bold; entries whose CI excludes zero are significant at "
        r"$\alpha{=}0.05$.  Twin-bootstrap $\lambda{=}300$ is selected on "
        r"BACE only and applied unchanged to every held-out benchmark.}",
        r"  \label{tab:main}",
        r"  \scriptsize",
        r"  \begin{tabular}{lrll@{\hspace{1.2em}}lll}",
        r"    \toprule",
        r"    & & \multicolumn{2}{c}{Parameter-side}"
        r" & \multicolumn{3}{c}{Data-side} \\",
        r"    \cmidrule(lr){3-4} \cmidrule(lr){5-7}",
        r"    Dataset & $N$ & " + " & ".join(headers) + r" \\",
        r"    \midrule",
    ]
    ds_order = sorted(by_ds.keys(), key=lambda d: N_TRAIN.get(d, 10**9))
    for ds in ds_order:
        # Best (most negative mean) cell gets boldened.
        means = []
        for m in methods:
            r = by_ds[ds].get(m)
            mean = r.get("delta_id_churn_mean") if r else None
            means.append(mean if mean is not None else float("inf"))
        best_idx = int(np.argmin(means))
        cells = []
        for i, m in enumerate(methods):
            r = by_ds[ds].get(m)
            cell = _fmt_delta_pp_from_row(r) if r else "---"
            if i == best_idx and cell != "---":
                cell = r"\textbf{" + cell + r"}"
            cells.append(cell)
        ds_label = display(ds) + (r"\,(dev)" if ds == DEV_DATASET else "")
        lines.append(f"    {ds_label} & {N_TRAIN.get(ds, '---')} & "
                     + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))
    print(f"Wrote {path}")


def _fmt_delta_pp_from_row(r):
    if r.get("delta_id_churn_mean") is None:
        return "---"
    return _fmt_delta_pp((r["delta_id_churn_mean"],
                          r["delta_id_churn_lo"],
                          r["delta_id_churn_hi"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/cross_sample")
    ap.add_argument("--datasets", nargs="+",
                    default=[DEV_DATASET] + HEADLINE_DATASETS)
    ap.add_argument("--frozen_lam", type=float, default=FROZEN_LAM,
                    help="Twin_indep λ frozen on dev dataset (BACE).")
    ap.add_argument("--csv", default="outputs/main_table.csv")
    ap.add_argument("--latex", default="paper/sections/tables/main.tex")
    args = ap.parse_args()
    root = Path(args.root)

    methods = list(METHODS) + [
        (f"Twin_indep λ={args.frozen_lam} (frozen)",
         GLOBS["twin_indep"].format(lam=args.frozen_lam)),
    ]

    rows = []
    print("== Paper Table 1: cross-sample fragility on held-out datasets ==\n")
    print(f"{'dataset':14s} {'method':32s}  {'id_acc':>17s}  {'ood_acc':>17s}  "
          f"{'id_churn':>17s}  {'ood_churn':>17s}")
    for ds in args.datasets:
        ds_dir = root / ds
        for tag, glob in methods:
            m = metrics_with_cis(ds_dir, glob)
            if m is None:
                continue
            print(f"{ds:14s} {tag:32s}  "
                  f"{fmt_ci(m['id_acc']):>17s}  {fmt_ci(m['ood_acc']):>17s}  "
                  f"{fmt_ci(m['id_churn']):>17s}  {fmt_ci(m['ood_churn']):>17s}")
            row = {"dataset": ds, "method": tag, "n_seeds": m["n_seeds"]}
            for k in ("id_acc", "ood_acc", "id_churn", "ood_churn",
                      "id_sym_kl", "ood_sym_kl"):
                row[f"{k}_mean"], row[f"{k}_lo"], row[f"{k}_hi"] = m[k]
            # Paired Δ id-churn vs ERM (used by the LaTeX table).
            res = paired_delta_vs(GLOBS["erm"], glob, ds_dir, "id_churn")
            if res is not None:
                (dm, dlo, dhi), _ = res
                row["delta_id_churn_mean"] = dm
                row["delta_id_churn_lo"] = dlo
                row["delta_id_churn_hi"] = dhi
            rows.append(row)
        print()

    # Paired comparisons (vs ERM, vs Bagging K=2).
    print("\n== Paired Δ vs ERM (id_churn): negative = method better ==")
    print(f"{'dataset':14s} {'method':32s}  {'Δ_id_churn (95%CI)':>32s}")
    for ds in args.datasets:
        ds_dir = root / ds
        for tag, glob in methods:
            if tag == "ERM":
                continue
            res = paired_delta_vs(GLOBS["erm"], glob, ds_dir, "id_churn")
            if res is None:
                continue
            (mean, lo, hi), n = res
            sig = "  **" if (lo < 0 and hi < 0) else "    "
            print(f"{ds:14s} {tag:32s}  "
                  f"{mean:+.4f} [{lo:+.4f},{hi:+.4f}]{sig}  (n={n} pairs)")
    print()
    print("== Paired Δ vs Bagging K=2 (compute-matched, id_churn) ==")
    for ds in args.datasets:
        ds_dir = root / ds
        for tag, glob in methods:
            if tag in ("ERM", "Bagging K=2"):
                continue
            if tag == "Bagging K=5":
                continue   # not compute-matched against K=2 in this column
            res = paired_delta_vs(GLOBS["bagging_2"], glob, ds_dir, "id_churn")
            if res is None:
                continue
            (mean, lo, hi), n = res
            sig = "  **" if (lo < 0 and hi < 0) else "    "
            print(f"{ds:14s} {tag:32s}  "
                  f"{mean:+.4f} [{lo:+.4f},{hi:+.4f}]{sig}  (n={n} pairs)")

    # Save tidy CSV.
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    if rows:
        all_keys = sorted({k for r in rows for k in r.keys()})
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_keys)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in all_keys})
        print(f"\nWrote {args.csv}")

    # Emit the LaTeX table that experiments.tex \input's.
    if rows:
        write_latex_table(rows, args.latex, args.frozen_lam)


if __name__ == "__main__":
    main()
