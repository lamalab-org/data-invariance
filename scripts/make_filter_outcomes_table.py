"""Generate the filter-outcomes table justifying which datasets we drop.

For every dataset in ``ALL_CHEMISTRY`` (headline + borderline + excluded),
read the saved ERM NPZs, compute mean id-accuracy across train_seeds and
the canonical majority-class baseline, and emit a table grouped by filter
outcome.  Closes the prose claim in the §Datasets paragraph: "five are
excluded because ERM does not exceed majority at the +5pp threshold".

Reads:  outputs/cross_sample/{dataset}/erm_train*.npz
Writes: paper/sections/tables/filter_outcomes.tex
        outputs/filter_outcomes.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from _analysis_lib import load_runs, per_run_accuracies
from paper_constants import (
    ALL_CHEMISTRY,
    BORDERLINE_DATASETS,
    DEV_DATASET,
    EXCLUDED_DATASETS,
    HEADLINE_DATASETS,
    N_TRAIN,
    display,
)


def _majority_baseline(runs) -> float:
    """Largest class proportion in the canonical id-test labels.

    The canonical test set is byte-identical across runs (same
    ``canonical_data_seed``), so any run gives the same baseline; we read
    the first one's labels.
    """
    if not runs:
        return float("nan")
    _, d = runs[0]
    labels = d["id_labels"]
    counts = np.bincount(np.asarray(labels, dtype=int))
    return float(counts.max()) / float(counts.sum())


def _row(dataset: str, root: Path) -> dict | None:
    runs = load_runs(root / dataset, "erm_train*.npz")
    if not runs:
        return None
    id_accs, _ = per_run_accuracies(runs)
    erm_mean = float(np.mean(id_accs))
    majority = _majority_baseline(runs)
    n_id_test = int(len(runs[0][1]["id_indices"]))
    return {
        "dataset": dataset,
        "n_train": N_TRAIN.get(dataset, ""),
        "n_id_test": n_id_test,
        "majority": majority,
        "erm_acc": erm_mean,
        "gap_pp": (erm_mean - majority) * 100.0,
        "n_seeds": len(runs),
    }


def _classify(ds: str) -> str:
    if ds == DEV_DATASET:
        return "dev"
    if ds in HEADLINE_DATASETS:
        return "headline"
    if ds in BORDERLINE_DATASETS:
        return "borderline"
    if ds in EXCLUDED_DATASETS:
        return "excluded"
    return "other"


def main() -> None:
    root = Path("outputs/cross_sample")
    rows = []
    for ds in ALL_CHEMISTRY:
        r = _row(ds, root)
        if r is None:
            print(f"  {ds}: no ERM runs")
            continue
        r["group"] = _classify(ds)
        rows.append(r)

    excluded = sorted(
        (r for r in rows if r["group"] == "excluded"),
        key=lambda r: r["gap_pp"],
    )

    if not excluded:
        print("No excluded-dataset ERM runs found.")
        return

    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \caption{\textbf{Five datasets fail the +$5$pp ERM-vs-majority "
        r"filter and are excluded from the main analysis.}  ERM id-acc "
        r"is the mean across $10$ retrainings; majority is the largest "
        r"class proportion on the canonical id-test set.  The "
        r"filter requires ERM to exceed majority by at least $5$pp; on "
        r"each of the five rows below it does not, so cross-sample churn "
        r"would conflate ``method shifts the decision boundary'' with "
        r"``majority-class shuffling under noise''.  Reported here for "
        r"transparency.}",
        r"  \label{tab:filter-outcomes}",
        r"  \small",
        r"  \begin{tabular}{lrrrrr}",
        r"    \toprule",
        r"    Dataset & $N_{\text{train}}$ & $N_{\text{id-test}}$ "
        r"& Majority & ERM id-acc & Gap (pp) \\",
        r"    \midrule",
    ]
    for r in excluded:
        lines.append(
            f"    {display(r['dataset'])} & {r['n_train']} & {r['n_id_test']} & "
            f"{r['majority']:.3f} & {r['erm_acc']:.3f} & "
            f"{r['gap_pp']:+.1f} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    out_path = Path("paper/sections/tables/filter_outcomes.tex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")

    csv_path = Path("outputs/filter_outcomes.csv")
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "group", "n_train", "n_id_test",
                    "majority_baseline", "erm_acc_mean", "gap_pp", "n_seeds"])
        for r in rows:
            w.writerow([r["dataset"], r["group"], r["n_train"], r["n_id_test"],
                        r["majority"], r["erm_acc"], r["gap_pp"], r["n_seeds"]])
    print(f"Wrote {csv_path}")
    print(f"  {len(excluded)} excluded datasets tabulated.")


if __name__ == "__main__":
    main()
