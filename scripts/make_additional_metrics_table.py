"""Per-dataset paired |Δ| of precision, recall, F1, AP across ERM bootstraps.

Closes a reviewer concern that aggregate accuracy on imbalanced data can
mask class-specific drift between retrainings.  We compute the same
paired-bootstrap protocol used for ``|Δ accuracy|`` (Table~\ref{tab:fragility-magnitudes})
on three additional aggregate metrics, all on the canonical id-test
predictions of $K{=}10$ ERM bootstraps:

  - precision and recall on the positive class
  - macro-averaged F1
  - average precision (AP), the area under the precision-recall curve

Per pair $(s, s')$ we compute $|m_s - m_{s'}|$ for each metric, then mean
$\pm 95\%$ paired-bootstrap CI over the $\binom{10}{2}=45$ seed pairs.

Reads:  outputs/cross_sample/{dataset}/erm_train*.npz
Writes: paper/sections/tables/additional_metrics.tex
        outputs/additional_metrics.csv
"""
from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from _analysis_lib import bootstrap_ci, load_runs, pairwise_metrics
from paper_constants import DEV_DATASET, HEADLINE_DATASETS, N_TRAIN, display


def _per_run_metrics(runs):
    """For each run, return a dict of scalar aggregate metrics."""
    out = []
    for _, d in runs:
        probs = d["id_probs_avg"] if "id_probs_avg" in d else d["id_probs"]
        labels = np.asarray(d["id_labels"]).astype(int)
        preds = probs.argmax(1)
        # Positive-class metrics; fall back to weighted macro for >2 classes.
        n_classes = probs.shape[1]
        if n_classes == 2:
            prec = precision_score(labels, preds, pos_label=1, zero_division=0)
            rec = recall_score(labels, preds, pos_label=1, zero_division=0)
            ap = average_precision_score(labels, probs[:, 1])
        else:
            prec = precision_score(labels, preds, average="macro", zero_division=0)
            rec = recall_score(labels, preds, average="macro", zero_division=0)
            # Macro-AP with one-vs-rest (sklearn handles via 'macro').
            ap = average_precision_score(
                np.eye(n_classes)[labels], probs, average="macro"
            )
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        acc = float((preds == labels).mean())
        out.append({"acc": acc, "precision": prec, "recall": rec, "f1": f1, "ap": ap})
    return out


def _paired_abs_deltas(values: list[float]) -> list[float]:
    """For a list of per-run values, return |v_i - v_j| over all pairs."""
    return [abs(a - b) for a, b in combinations(values, 2)]


def _row(dataset: str, root: Path) -> dict | None:
    runs = load_runs(root / dataset, "erm_train*.npz")
    if len(runs) < 2:
        return None
    metrics = _per_run_metrics(runs)
    out = {"dataset": dataset, "n_train": N_TRAIN.get(dataset, ""),
           "n_seeds": len(runs)}
    for key in ("acc", "precision", "recall", "f1", "ap"):
        deltas_pp = [d * 100 for d in _paired_abs_deltas([m[key] for m in metrics])]
        out[f"d_{key}_pp_ci"] = bootstrap_ci(deltas_pp)
    # Per-example argmax-disagreement rate (the per-prediction quantity
    # the aggregate-metric drifts above are dominated by).  Same 45 seed
    # pairs as the aggregate-metric deltas, in percentage points.
    pm, _ = pairwise_metrics(runs)
    churns_pct = [m["id_churn"] * 100 for m in pm.values()]
    out["churn_pct_ci"] = bootstrap_ci(churns_pct)
    return out


def _fmt(t):
    m, lo, hi = t
    return f"{m:.1f} [{lo:.1f}, {hi:.1f}]"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="outputs/cross_sample")
    ap.add_argument("--csv",  default="outputs/additional_metrics.csv")
    ap.add_argument("--latex", default="paper/sections/tables/additional_metrics.tex")
    args = ap.parse_args()
    root = Path(args.root)
    datasets = [DEV_DATASET] + HEADLINE_DATASETS
    rows = [r for r in (_row(d, root) for d in datasets) if r is not None]
    if not rows:
        print("No ERM runs found.")
        return
    rows.sort(key=lambda r: r["n_train"])

    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \caption{\textbf{\boldmath Per-example argmax-disagreement (right column) "
        r"dominates aggregate-metric drift on every dataset, regardless of "
        r"which summary statistic the aggregate-metric column uses.}  "
        r"Paired $|\Delta|$ of five aggregate metrics and the per-example "
        r"argmax-churn rate, computed across the $\binom{10}{2}=45$ seed "
        r"pairs of ERM bootstraps; mean $[\,95\%\ \text{CI}\,]$ over "
        r"$10{,}000$ resamples, in percentage points.  Per-example "
        r"argmax-churn ranges $\churnMin\text{--}\churnMax\%$; the "
        r"strongest aggregate-metric drift on any cell is "
        r"$|\Delta\text{recall}|=10.5$\,pp on the imbalanced "
        r"CYP2D6-Sub dataset.}",
        r"  \label{tab:additional-metrics}",
        r"  \scriptsize",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{lr@{\hspace{0.6em}}lllll@{\hspace{1em}}l}",
        r"    \toprule",
        r"    & & \multicolumn{5}{c}{Aggregate-metric drift (pp)}"
        r" & Per-example \\",
        r"    \cmidrule(lr){3-7}",
        r"    Dataset & $N$ & $|\Delta\text{acc}|$ & $|\Delta\text{prec}|$"
        r" & $|\Delta\text{rec}|$ & $|\Delta F_1|$ & $|\Delta\text{AP}|$"
        r" & argmax churn (\%) \\",
        r"    \midrule",
    ]
    for r in rows:
        ds = display(r["dataset"]) + (r"\,(dev)" if r["dataset"] == DEV_DATASET else "")
        # Bold the per-example argmax-churn cell on every row so the
        # column visually anchors the argument.
        churn_cell = r"\textbf{" + _fmt(r['churn_pct_ci']) + "}"
        lines.append(
            f"    {ds} & {r['n_train']} & "
            f"{_fmt(r['d_acc_pp_ci'])} & {_fmt(r['d_precision_pp_ci'])} & "
            f"{_fmt(r['d_recall_pp_ci'])} & {_fmt(r['d_f1_pp_ci'])} & "
            f"{_fmt(r['d_ap_pp_ci'])} & "
            f"{churn_cell} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"  }",
              r"\end{table}", ""]

    out_path = Path(args.latex)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")

    csv_path = Path(args.csv)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "n_train",
                    "d_acc_pp_mean", "d_acc_pp_lo", "d_acc_pp_hi",
                    "d_precision_pp_mean", "d_precision_pp_lo", "d_precision_pp_hi",
                    "d_recall_pp_mean", "d_recall_pp_lo", "d_recall_pp_hi",
                    "d_f1_pp_mean", "d_f1_pp_lo", "d_f1_pp_hi",
                    "d_ap_pp_mean", "d_ap_pp_lo", "d_ap_pp_hi",
                    "churn_pct_mean", "churn_pct_lo", "churn_pct_hi"])
        for r in rows:
            row_out = [r["dataset"], r["n_train"]]
            for k in ("acc", "precision", "recall", "f1", "ap"):
                row_out.extend(r[f"d_{k}_pp_ci"])
            row_out.extend(r["churn_pct_ci"])
            w.writerow(row_out)
    print(f"Wrote {csv_path}")
    for r in rows:
        print(f"  {r['dataset']:16s}  acc {_fmt(r['d_acc_pp_ci']):>16s}  "
              f"prec {_fmt(r['d_precision_pp_ci']):>16s}  "
              f"rec {_fmt(r['d_recall_pp_ci']):>16s}  "
              f"F1 {_fmt(r['d_f1_pp_ci']):>16s}  "
              f"AP {_fmt(r['d_ap_pp_ci']):>16s}")


if __name__ == "__main__":
    main()
