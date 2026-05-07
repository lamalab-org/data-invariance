"""Generate the fragility-magnitudes table for paper §4.

For every dev + headline dataset, compute mean cross-bootstrap sym-KL and
mean argmax churn between ERM models trained on independent bootstraps,
with paired-bootstrap 95% CIs over the C(N_seeds, 2) seed pairs.

Reads:  outputs/cross_sample/{dataset}/erm_train*.npz  (10 seeds, sorted)
Writes: paper/sections/tables/fragility_magnitudes.tex
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from _analysis_lib import bootstrap_ci, load_runs, pairwise_metrics, per_run_accuracies
from paper_constants import (
    BORDERLINE_DATASETS,
    DEV_DATASET,
    HEADLINE_DATASETS,
    MAGNITUDES_EXTRA,
    N_TRAIN,
    display,
)  # noqa: F401


# Per-dataset citation key (TODO_ until references.bib is wired).
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
    "herg":            "TODO_huang2021_tdc",
    "hia_hou":         "TODO_huang2021_tdc",
    "skin_reaction":   "TODO_huang2021_tdc",
    "cyp2c9_substrate":"TODO_huang2021_tdc",
    "cyp2d6_substrate":"TODO_huang2021_tdc",
    "cyp3a4_substrate":"TODO_huang2021_tdc",
    "clintox":         "TODO_wu2018_moleculenet",
}


def cited(ds: str) -> str:
    """Return display name with a \\citep{} attached."""
    cite = _CITE.get(ds)
    return f"{display(ds)}~\\citep{{{cite}}}" if cite else display(ds)


def _row(dataset: str, root: Path) -> dict | None:
    runs = load_runs(root / dataset, "erm_train*.npz")
    if len(runs) < 2:
        return None
    id_accs, _ = per_run_accuracies(runs)
    pair_metrics, _ = pairwise_metrics(runs)
    churns = [m["id_churn"] for m in pair_metrics.values()]
    sym_kls = [m["id_sym_kl"] for m in pair_metrics.values()]
    # Aggregate-accuracy analogue of churn: |acc_A - acc_B| in pp,
    # averaged over the same 45 seed pairs.  Two retrainings differ in
    # aggregate accuracy by this amount; they disagree on `churn` of
    # individual predictions.  The contrast is the headline of the table.
    acc_diffs_pp = [abs(id_accs[i] - id_accs[j]) * 100
                    for i, j in combinations(range(len(id_accs)), 2)]
    # id-test set size (constant across train_seeds under the canonical
    # protocol; surface it so per-dataset CI widths are auditable).
    n_id_test = int(len(runs[0][1]["id_indices"]))
    return {
        "dataset": dataset,
        "n_train": N_TRAIN.get(dataset, len(runs[0][1]["id_indices"])),
        "n_id_test": n_id_test,
        "id_acc_mean": float(np.mean(id_accs)),
        "id_acc_min": float(np.min(id_accs)),
        "id_acc_max": float(np.max(id_accs)),
        "acc_diff_pp_ci": bootstrap_ci(acc_diffs_pp),
        "churn": bootstrap_ci(churns),
        "sym_kl": bootstrap_ci(sym_kls),
        "n_seeds": len(runs),
    }


def _fmt_pct_ci(t: tuple[float, float, float]) -> str:
    return f"{t[0]*100:.1f} [{t[1]*100:.1f}, {t[2]*100:.1f}]"


def _fmt_pp_ci(t: tuple[float, float, float]) -> str:
    """Format a CI already in percentage-point units (no ×100)."""
    return f"{t[0]:.1f} [{t[1]:.1f}, {t[2]:.1f}]"


def _fmt_ci(t: tuple[float, float, float]) -> str:
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="outputs/cross_sample")
    ap.add_argument("--csv",  default="outputs/fragility_magnitudes.csv")
    ap.add_argument("--latex", default="paper/sections/tables/fragility_magnitudes.tex")
    args = ap.parse_args()
    root = Path(args.root)
    out_path = Path(args.latex)
    headline = [DEV_DATASET] + HEADLINE_DATASETS + MAGNITUDES_EXTRA
    headline_rows = [r for r in (_row(ds, root) for ds in headline) if r is not None]
    borderline_rows = [
        r for r in (_row(ds, root) for ds in BORDERLINE_DATASETS) if r is not None
    ]
    if not headline_rows:
        print(f"No ERM runs found under {root}/.")
        return

    headline_rows.sort(key=lambda r: r["n_train"])
    borderline_rows.sort(key=lambda r: r["n_train"])
    min_acc_diff = min(r["acc_diff_pp_ci"][0] for r in headline_rows)
    max_acc_diff = max(r["acc_diff_pp_ci"][0] for r in headline_rows)
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{\textbf{\boldmath Two retrainings on independent bootstraps "
        r"differ in aggregate accuracy by "
        + f"{min_acc_diff:.1f}" + r"\,--\," + f"{max_acc_diff:.1f}"
        + r"\,pp on average, but disagree on $8\text{--}22\%$ of "
        r"individual test predictions.}  "
        r"Cross-bootstrap class-flip rate on the canonical id-test of "
        r"the nine chemistry datasets that pass a $+5$pp ERM-vs-majority "
        r"filter on test sets of at least $60$ examples (BACE is the "
        r"development dataset; the other eight are held-out).  Three "
        r"datasets that pass the filter only marginally are reported in "
        r"Appendix~\ref{app:borderline}.  ERM id-acc is the mean across "
        r"$10$ retrainings; $|\Delta\text{acc}|$ is the mean absolute "
        r"accuracy difference between two retrainings, averaged over "
        r"the same $45$ pairs of $10$ retrainings as the churn column.  Class-flip rate is the "
        r"per-example argmax-disagreement rate (cross-sample churn); "
        r"sym-KL is the corresponding distributional gap.  All paired "
        r"columns report mean with $95\%$ paired-bootstrap CIs "
        r"($10{,}000$ resamples).}",
        r"  \label{tab:fragility-magnitudes}",
        r"  \scriptsize",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{lrrrl@{\hspace{1em}}ll}",
        r"    \toprule",
        r"    & & & \multicolumn{2}{c}{Aggregate accuracy}"
        r" & \multicolumn{2}{c}{Per-prediction disagreement} \\",
        r"    \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"    Dataset & $N_{\text{train}}$ & $N_{\text{id-test}}$ &"
        r" ERM id-acc & $|\Delta\text{acc}|$ (pp)"
        r" & Argmax churn (\%) & Sym-KL (nats) \\",
        r"    \midrule",
    ]
    for r in headline_rows:
        ds_label = display(r['dataset']) + (r"\,(dev)" if r['dataset'] == DEV_DATASET else "")
        # ERM id-acc shown as mean [min, max] across the 10 retrainings
        # so the reader sees where the model is sitting (e.g.\ 0.78
        # [0.77, 0.79]) and can put |dAcc| in context.
        acc_str = f"{r['id_acc_mean']:.3f} [{r['id_acc_min']:.3f}, {r['id_acc_max']:.3f}]"
        # Bold the per-prediction disagreement cells: the central
        # observation is that argmax churn (and its sym-KL companion)
        # are the per-example complement to |dAcc|.
        lines.append(
            f"    {ds_label} & {r['n_train']} & {r['n_id_test']} & "
            f"{acc_str} & {_fmt_pp_ci(r['acc_diff_pp_ci'])} & "
            f"\\textbf{{{_fmt_pct_ci(r['churn'])}}} & "
            f"\\textbf{{{_fmt_ci(r['sym_kl'])}}} \\\\"
        )
    # Borderline rows are reported only in App.~\ref{app:borderline}; the
    # body table stays focused on the nine datasets that pass the filter
    # cleanly.
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  }",
        r"\end{table}",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")

    # Emit a separate borderline-magnitudes table for the appendix so the
    # body table is not cluttered with the marginal-filter-pass rows.
    if borderline_rows:
        bl_lines = [
            r"\begin{table}[h]",
            r"  \centering",
            r"  \caption{\textbf{\boldmath Cross-sample magnitudes for the three "
            r"borderline datasets.}  These pass the ERM-vs-majority filter "
            r"only marginally (+3 to +4\,pp on test sets of 57--104 examples) "
            r"and are reported here for transparency; method comparisons are "
            r"not run on them because the small test sets do not give enough "
            r"statistical power.  Columns and CI conventions match "
            r"Table~\ref{tab:fragility-magnitudes}.}",
            r"  \label{tab:borderline-magnitudes}",
            r"  \scriptsize",
            r"  \resizebox{\linewidth}{!}{%",
            r"  \begin{tabular}{lrrrl@{\hspace{1em}}ll}",
            r"    \toprule",
            r"    & & & \multicolumn{2}{c}{Aggregate accuracy}"
            r" & \multicolumn{2}{c}{Per-prediction disagreement} \\",
            r"    \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
            r"    Dataset & $N_{\text{train}}$ & $N_{\text{id-test}}$ &"
            r" ERM id-acc & $|\Delta\text{acc}|$ (pp)"
            r" & Argmax churn (\%) & Sym-KL (nats) \\",
            r"    \midrule",
        ]
        for r in borderline_rows:
            acc_str = (f"{r['id_acc_mean']:.3f} "
                       f"[{r['id_acc_min']:.3f}, {r['id_acc_max']:.3f}]")
            bl_lines.append(
                f"    {display(r['dataset'])} & {r['n_train']} & "
                f"{r['n_id_test']} & "
                f"{acc_str} & {_fmt_pp_ci(r['acc_diff_pp_ci'])} & "
                f"\\textbf{{{_fmt_pct_ci(r['churn'])}}} & "
                f"\\textbf{{{_fmt_ci(r['sym_kl'])}}} \\\\"
            )
        bl_lines += [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  }",
            r"\end{table}",
            "",
        ]
        bl_path = out_path.parent / "borderline_magnitudes.tex"
        bl_path.write_text("\n".join(bl_lines))
        print(f"Wrote {bl_path}")

    # CSV dump for paper-macros and audit (single row per dataset, one
    # ``group`` column to distinguish headline / borderline).
    import csv as _csv
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["dataset", "group", "n_train", "n_id_test",
                    "erm_id_acc_mean",
                    "acc_diff_pp_mean", "acc_diff_pp_lo", "acc_diff_pp_hi",
                    "churn_mean", "churn_lo", "churn_hi",
                    "sym_kl_mean", "sym_kl_lo", "sym_kl_hi", "n_seeds"])
        for group_name, group in [("headline", headline_rows),
                                  ("borderline", borderline_rows)]:
            for r in group:
                w.writerow([r["dataset"], group_name, r["n_train"], r["n_id_test"],
                            r["id_acc_mean"],
                            r["acc_diff_pp_ci"][0], r["acc_diff_pp_ci"][1], r["acc_diff_pp_ci"][2],
                            r["churn"][0], r["churn"][1], r["churn"][2],
                            r["sym_kl"][0], r["sym_kl"][1], r["sym_kl"][2],
                            r["n_seeds"]])
    print(f"Wrote {csv_path}")
    for r in headline_rows + borderline_rows:
        print(f"  {r['dataset']:16s}  N={r['n_train']:>5d}  "
              f"acc={r['id_acc_mean']:.3f}  "
              f"|Δacc|={_fmt_pp_ci(r['acc_diff_pp_ci'])}pp  "
              f"churn={_fmt_pct_ci(r['churn'])}  "
              f"sym-kl={_fmt_ci(r['sym_kl'])}")


if __name__ == "__main__":
    main()
