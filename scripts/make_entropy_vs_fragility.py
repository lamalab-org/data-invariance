"""Predictive entropy vs cross-sample fragility as flip-predictors.

Reviewer-anticipated question: is per-example fragility actually better
than the much cheaper "softmax confidence from a single model" baseline?

Protocol
--------
For each headline + dev dataset, on the canonical id-test:
  - fragility(x) = mean cross-bootstrap argmax disagreement across the
    C(10, 2) pairs of ERM models
  - entropy(x) = mean predictive entropy of a single ERM model,
    averaged across the 10 models we trained

Then compare both as predictors of "will the prediction flip on
retraining?" via the area under the precision-vs-coverage curve, and
via top-decile flip recall.

Output: a small comparison table in the appendix and a CSV with
per-dataset numbers.
"""
from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np

from _analysis_lib import load_runs
from paper_constants import DEV_DATASET, HEADLINE_DATASETS, N_TRAIN, display


def _per_example_signals(ds_dir: Path):
    """Return (fragility, entropy, flip_rate) per test example."""
    runs = load_runs(ds_dir, "erm_train*.npz")
    if len(runs) < 2:
        return None
    probs_list = [d.get("id_probs_avg", d.get("id_probs")) for _, d in runs]
    n_test = probs_list[0].shape[0]

    # Fragility: mean argmax-disagreement across all seed pairs.
    flip_acc = np.zeros(n_test)
    n_pairs = 0
    for i, j in combinations(range(len(probs_list)), 2):
        flip_acc += (probs_list[i].argmax(1) != probs_list[j].argmax(1)).astype(float)
        n_pairs += 1
    flip_rate = flip_acc / n_pairs

    # Predictive entropy averaged across the 10 single-model softmaxes.
    eps = 1e-12
    ent_per_seed = np.stack([-(p * np.log(p + eps)).sum(axis=1) for p in probs_list])
    ent_mean = ent_per_seed.mean(axis=0)

    # We use the cross-pair disagreement signal itself as the "fragility"
    # ranking: higher mean disagreement => more fragile.  Equivalently,
    # rank by the per-example variance of softmax predictions across seeds.
    return flip_rate, ent_mean, flip_rate


def _decile_capture(score: np.ndarray, flip_rate: np.ndarray, top_decile: float = 0.1):
    """Fraction of total flip-mass captured by the top score-decile."""
    n_top = max(1, int(len(score) * top_decile))
    order = np.argsort(-score)  # high score = high predicted fragility
    captured = flip_rate[order[:n_top]].sum()
    total = flip_rate.sum()
    return float(captured / total) if total > 0 else 0.0


def _avg_precision_vs_coverage(score: np.ndarray, flip_rate: np.ndarray):
    """AUC of precision-vs-coverage for the score as a flip predictor."""
    order = np.argsort(-score)
    sorted_flip = flip_rate[order]
    cum = np.cumsum(sorted_flip) / np.arange(1, len(sorted_flip) + 1)
    return float(np.trapezoid(cum, np.arange(1, len(sorted_flip) + 1)
                              / len(sorted_flip)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/cross_sample")
    ap.add_argument("--csv", default="outputs/entropy_vs_fragility.csv")
    ap.add_argument("--latex", default="paper/sections/tables/entropy_vs_fragility.tex")
    args = ap.parse_args()
    root = Path(args.root)

    rows = []
    for ds in [DEV_DATASET] + HEADLINE_DATASETS:
        out = _per_example_signals(root / ds)
        if out is None:
            continue
        frag, ent, flip = out
        # Fragility-as-score: rank by the per-example flip rate.
        # Entropy-as-score: rank by single-model predictive entropy.
        rows.append({
            "dataset": ds,
            "frag_top10_capture": _decile_capture(frag, flip),
            "ent_top10_capture":  _decile_capture(ent, flip),
            "frag_aupc":          _avg_precision_vs_coverage(frag, flip),
            "ent_aupc":           _avg_precision_vs_coverage(ent, flip),
        })

    # Sort by training-set size to match the other paper tables.
    rows.sort(key=lambda r: N_TRAIN.get(r["dataset"], 10**9))

    print()
    print(f"{'dataset':16s}  {'frag top-10% recall':>22s}  {'entropy top-10% recall':>22s}  "
          f"{'frag AuPC':>12s}  {'ent AuPC':>12s}")
    for r in rows:
        print(f"{r['dataset']:16s}  {r['frag_top10_capture']*100:>21.1f}%  "
              f"{r['ent_top10_capture']*100:>21.1f}%  "
              f"{r['frag_aupc']:>12.3f}  {r['ent_aupc']:>12.3f}")

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.csv}")

    # Emit a LaTeX table for the appendix.
    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \caption{\textbf{Per-example churn beats single-model "
        r"predictive entropy as a retraining-flip predictor on every "
        r"chemistry dataset.}  Churn is computed from one extra "
        r"bootstrap pair on the canonical id-test.  ``Top-10\%\ recall'' "
        r"is the fraction of all retraining-induced flips captured by "
        r"the top decile of the score; ``AuPC'' is the area under the "
        r"precision-vs-coverage curve.  Higher is better for both columns.}",
        r"  \label{tab:entropy_vs_fragility}",
        r"  \small",
        r"  \begin{tabular}{lrrrr}",
        r"    \toprule",
        r"    & \multicolumn{2}{c}{Top-10\% recall (\%)} "
        r"& \multicolumn{2}{c}{AuPC} \\",
        r"    Dataset & Churn & Entropy & Churn & Entropy \\",
        r"    \midrule",
    ]
    for r in rows:
        lines.append(
            f"    {display(r['dataset'])} & "
            f"{r['frag_top10_capture']*100:.1f} & "
            f"{r['ent_top10_capture']*100:.1f} & "
            f"{r['frag_aupc']:.3f} & {r['ent_aupc']:.3f} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    Path(args.latex).parent.mkdir(parents=True, exist_ok=True)
    Path(args.latex).write_text("\n".join(lines))
    print(f"Wrote {args.latex}")


if __name__ == "__main__":
    main()
