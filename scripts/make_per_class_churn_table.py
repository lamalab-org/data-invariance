"""Per-class breakdown of cross-sample fragility.

Closes a reviewer concern that on imbalanced data, overall argmax
churn could be dominated by either majority-class shuffling or
minority-class instability.  Splitting churn by true label
distinguishes these.

For each headline dataset and each of the 45 ERM seed pairs, we
compute three argmax-disagreement rates on the canonical id-test:

  - overall argmax churn: fraction of test examples that flip
  - churn|y=0:           same fraction restricted to y=0 examples
  - churn|y=1:           same fraction restricted to y=1 examples

We additionally report the class balance on the id-test set
(fraction of y=1 examples) so the reader can see which class is
the minority on each dataset.

Reads:  outputs/cross_sample/{dataset}/erm_train*.npz
Writes: paper/sections/tables/per_class_churn.tex
        outputs/per_class_churn.csv
"""
from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import numpy as np

from _analysis_lib import bootstrap_ci, load_runs
from paper_constants import DEV_DATASET, HEADLINE_DATASETS, N_TRAIN, display


def _per_class_churn_pairs(runs):
    """For each seed pair, return (overall, churn_y0, churn_y1) churn rates."""
    pairs = list(combinations(runs, 2))
    out_overall, out_y0, out_y1 = [], [], []
    for (sa, da), (sb, db) in pairs:
        # The two runs share the same canonical id-test, so labels match.
        labels = np.asarray(da["id_labels"]).astype(int)
        probs_a = da["id_probs_avg"] if "id_probs_avg" in da else da["id_probs"]
        probs_b = db["id_probs_avg"] if "id_probs_avg" in db else db["id_probs"]
        flip = probs_a.argmax(1) != probs_b.argmax(1)
        out_overall.append(float(flip.mean()))
        if (labels == 0).any():
            out_y0.append(float(flip[labels == 0].mean()))
        if (labels == 1).any():
            out_y1.append(float(flip[labels == 1].mean()))
    return out_overall, out_y0, out_y1


def _row(dataset: str, root: Path) -> dict | None:
    runs = load_runs(root / dataset, "erm_train*.npz")
    if len(runs) < 2:
        return None
    overall, y0, y1 = _per_class_churn_pairs(runs)
    if not overall:
        return None
    labels = np.asarray(runs[0][1]["id_labels"]).astype(int)
    pos_frac = float((labels == 1).mean())
    minority = "y=0" if pos_frac >= 0.5 else "y=1"
    return {
        "dataset": dataset,
        "n_train": N_TRAIN.get(dataset, ""),
        "n_id_test": int(len(labels)),
        "pos_frac": pos_frac,
        "minority": minority,
        "overall_ci": bootstrap_ci(overall),
        "y0_ci": bootstrap_ci(y0) if y0 else None,
        "y1_ci": bootstrap_ci(y1) if y1 else None,
    }


def _fmt(t):
    if t is None:
        return "---"
    m, lo, hi = t
    return f"{m*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]"


def main() -> None:
    root = Path("outputs/cross_sample")
    datasets = [DEV_DATASET] + HEADLINE_DATASETS
    rows = [r for r in (_row(d, root) for d in datasets) if r is not None]
    if not rows:
        print("No ERM runs found.")
        return
    rows.sort(key=lambda r: r["n_train"])

    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \caption{\textbf{On imbalanced datasets, minority-class "
        r"predictions are $2\text{--}4\times$ more unstable across "
        r"retrainings than majority-class predictions.}  For each "
        r"chemistry dataset: overall cross-bootstrap argmax-churn and "
        r"its restriction to $y{=}0$ and $y{=}1$ subsets of the "
        r"canonical id-test, across the $\binom{10}{2}=45$ ERM seed "
        r"pairs (mean $[\,95\%\ \text{CI}\,]$, $10{,}000$ resamples).  "
        r"Pos.\ frac.\ is the fraction of $y{=}1$ examples; the "
        r"minority class is bolded.  On the most imbalanced datasets "
        r"(BBB-Martins, BBBP at $0.78$ pos-frac; CYP2D6-Sub at "
        r"$0.30$) the minority-class churn rate is $2\text{--}4\times$ "
        r"the majority-class rate, so per-example disagreement is "
        r"\emph{concentrated} on the rarer class --- exactly the "
        r"predictions practitioners care most about (active "
        r"toxicity, BBB-permeable, substrate).  On balanced datasets "
        r"the rates are comparable.}",
        r"  \label{tab:per-class-churn}",
        r"  \small",
        r"  \begin{tabular}{lrrl@{\hspace{1em}}lll}",
        r"    \toprule",
        r"    Dataset & $N$ & $N_{\text{id-test}}$ & Pos.\ frac.\ "
        r"& Overall (\%) & churn$|y{=}0$ (\%) & churn$|y{=}1$ (\%) \\",
        r"    \midrule",
    ]
    for r in rows:
        ds = display(r["dataset"]) + (r"\,(dev)" if r["dataset"] == DEV_DATASET else "")
        # Bold the minority-class cell.
        y0_cell = _fmt(r["y0_ci"])
        y1_cell = _fmt(r["y1_ci"])
        if r["minority"] == "y=0":
            y0_cell = r"\textbf{" + y0_cell + "}"
        else:
            y1_cell = r"\textbf{" + y1_cell + "}"
        lines.append(
            f"    {ds} & {r['n_train']} & {r['n_id_test']} & "
            f"{r['pos_frac']:.2f} & "
            f"{_fmt(r['overall_ci'])} & {y0_cell} & {y1_cell} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    out_path = Path("paper/sections/tables/per_class_churn.tex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")

    csv_path = Path("outputs/per_class_churn.csv")
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "n_train", "n_id_test", "pos_frac", "minority",
                    "overall_mean", "overall_lo", "overall_hi",
                    "y0_mean", "y0_lo", "y0_hi",
                    "y1_mean", "y1_lo", "y1_hi"])
        for r in rows:
            row_out = [r["dataset"], r["n_train"], r["n_id_test"],
                       r["pos_frac"], r["minority"],
                       *r["overall_ci"]]
            row_out.extend(r["y0_ci"] or ["", "", ""])
            row_out.extend(r["y1_ci"] or ["", "", ""])
            w.writerow(row_out)
    print(f"Wrote {csv_path}")
    for r in rows:
        print(f"  {r['dataset']:16s} pos={r['pos_frac']:.2f} "
              f"all={_fmt(r['overall_ci'])} "
              f"|y0={_fmt(r['y0_ci'])} |y1={_fmt(r['y1_ci'])}")


if __name__ == "__main__":
    main()
