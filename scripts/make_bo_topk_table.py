"""Bayesian-optimisation analogue: top-K stability across retrainings.

A virtual-screening or Bayesian-optimisation pipeline acquires the
top-K candidates ranked by predicted P(active).  Cross-sample churn
matters here as the rate at which the top-K *set* changes between two
retrainings on independent bootstraps of the same training pool: the
fraction of candidates that would be re-routed to different
downstream evaluations depending on which bootstrap the surrogate
happened to land on.

For each chemistry dataset and each of {ERM, Bagging-K=5,
twin-bootstrap lambda=300}, we report the mean Jaccard overlap of the
top-K-by-P(class=1) sets between every pair of independent
retrainings (45 paired comparisons over the 10 train_seeds), with
paired-bootstrap 95% CIs.  The same protocol as the main argmax-churn
table, just substituting "top-K Jaccard" for "argmax disagreement" as
the per-pair metric.

Reads NPZs in outputs/cross_sample/<dataset>/.
Writes paper/sections/tables/bo_topk.tex and outputs/bo_topk.csv.
"""
from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np

from _analysis_lib import bootstrap_ci, bootstrap_paired, get_probs, load_runs
from paper_constants import DEV_DATASET, FROZEN_LAM, HEADLINE_DATASETS, display


METHODS = [
    ("ERM",                 "erm_train*.npz"),
    ("Bagging-K=5",         "bagging_train*_K5.npz"),
    (f"Twin-lam{int(FROZEN_LAM)}",
     f"twin_indep_train*_lam{FROZEN_LAM}.npz"),
]

K_DEFAULT = 10  # single-batch acquisition size; common BO budget


def topk_set(probs: np.ndarray, k: int) -> set[int]:
    """Indices of the top-k molecules ranked by P(class=1)."""
    if probs.shape[1] != 2:
        raise ValueError(f"top-K only defined for binary classification; "
                         f"got {probs.shape[1]} classes.")
    return set(np.argsort(-probs[:, 1])[:k].tolist())


def jaccard(a: set, b: set) -> float:
    union = len(a | b)
    return float(len(a & b) / union) if union else 0.0


def hit_rate(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    """Fraction of the top-k that are actually class 1.  Sanity check."""
    topk = np.argsort(-probs[:, 1])[:k]
    return float((y[topk] == 1).mean())


def per_method_jaccard_pairs(runs: list, k: int) -> tuple[list[float], list[float]]:
    """Return (jaccard_per_pair, hit_rate_per_run) over the 45 pairs."""
    if not runs:
        return [], []
    sets, hits = [], []
    for _, d in runs:
        idp, _ = get_probs(d)
        sets.append(topk_set(idp, k))
        hits.append(hit_rate(idp, d["id_labels"], k))
    pair_jaccards = [jaccard(a, b) for a, b in combinations(sets, 2)]
    return pair_jaccards, hits


def class_prior_from_runs(runs: list) -> float:
    """Positive-class fraction in the canonical id-test pool.

    The id_labels array is identical across retrainings (canonical-seed
    protocol), so we can read it off any one of them.
    """
    if not runs:
        return float("nan")
    _, d = runs[0]
    y = d["id_labels"]
    return float((y == 1).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/cross_sample")
    ap.add_argument("--csv", default="outputs/bo_topk.csv")
    ap.add_argument("--latex", default="paper/sections/tables/bo_topk.tex")
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    args = ap.parse_args()
    root = Path(args.root)
    K = args.k

    datasets = [DEV_DATASET] + list(HEADLINE_DATASETS)
    rows: list[dict] = []

    for ds in datasets:
        ds_dir = root / ds
        method_pair_jaccards: dict[str, list[float]] = {}
        method_hits: dict[str, list[float]] = {}
        prior = float("nan")
        for method_name, glob in METHODS:
            runs = load_runs(ds_dir, glob)
            if method_name == "ERM" and np.isnan(prior):
                prior = class_prior_from_runs(runs)
            pair_j, hits = per_method_jaccard_pairs(runs, K)
            method_pair_jaccards[method_name] = pair_j
            method_hits[method_name] = hits

        # Per-method mean Jaccard and 95% bootstrap CI over pairs.
        # Paired Δ vs ERM uses the same ordered pair list across methods
        # (combinations is deterministic, so positions correspond).
        erm_pairs = method_pair_jaccards.get("ERM", [])
        for method_name, _ in METHODS:
            pair_j = method_pair_jaccards.get(method_name, [])
            if not pair_j:
                continue
            ci = bootstrap_ci(pair_j)
            hits = method_hits[method_name]
            d = {
                "dataset":      ds,
                "method":       method_name,
                "K":            K,
                "n_pairs":      len(pair_j),
                "class_prior":  prior,
                "jaccard_mean": ci[0],
                "jaccard_lo":   ci[1],
                "jaccard_hi":   ci[2],
                "hit_rate_mean": float(np.mean(hits)) if hits else float("nan"),
                "delta_mean":   float("nan"),
                "delta_lo":     float("nan"),
                "delta_hi":     float("nan"),
            }
            if method_name != "ERM" and erm_pairs and len(pair_j) == len(erm_pairs):
                deltas = np.array(pair_j) - np.array(erm_pairs)
                d_ci = bootstrap_paired(deltas)
                d["delta_mean"] = d_ci[0]
                d["delta_lo"]   = d_ci[1]
                d["delta_hi"]   = d_ci[2]
            rows.append(d)
        print(
            f"{display(ds):<14}  "
            + "  ".join(
                f"{m}: J={method_pair_jaccards.get(m, [0]) and np.mean(method_pair_jaccards[m]):.2f}"
                for m, _ in METHODS if method_pair_jaccards.get(m)
            )
        )

    # ----- CSV -----
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {csv_path}")

    # ----- LaTeX -----
    def cell_jac(r):
        return (f"{r['jaccard_mean']:.2f} "
                f"[{r['jaccard_lo']:.2f}, {r['jaccard_hi']:.2f}]")

    def cell_delta(r):
        if np.isnan(r["delta_mean"]):
            return "---"
        return (f"{r['delta_mean']:+.2f} "
                f"[{r['delta_lo']:+.2f}, {r['delta_hi']:+.2f}]")

    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], {})[r["method"]] = r

    # Layout: 8 columns -- Dataset, class prior, ERM hit rate,
    # ERM Jaccard, Bagging Jaccard, Δ Bagging vs ERM,
    # Twin Jaccard, Δ Twin vs ERM.
    # The class-prior column is the chance baseline for top-K hit rate
    # (a random ranker would score the dataset's positive fraction);
    # ERM hit rate well above prior confirms the surrogate has signal.
    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \caption{\textbf{\boldmath Top-$K$ ranking stability "
        f"(${K}$ molecules) between independent retrainings: a Bayesian-"
        r"optimisation analogue of cross-sample churn.}  Jaccard "
        r"overlap of the top-$K$ predicted-active sets across the same "
        r"$45$ pairs of $10$ retrainings as the main table; $1.0$ = "
        r"identical sets, $0.0$ = disjoint.  Paired $\Delta$ vs.\ ERM "
        r"in the right two columns (positive = stabler ranking).  All "
        r"cells: mean $[\,95\%\ \text{CI}\,]$ from $10{,}000$ "
        r"paired-bootstrap resamples.  Class prior is the positive-class "
        r"fraction on the canonical id-test, the chance baseline for "
        r"top-$K$ hit rate; the surrogate has signal whenever the ERM "
        r"hit rate exceeds the prior.  The Jaccard difference is the "
        r"BO-relevant consequence of cross-sample churn.}",
        r"  \label{tab:bo_topk}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{lrr@{\hspace{0.6em}}c@{\hspace{0.6em}}cc@{\hspace{0.6em}}cc}",
        r"    \toprule",
        r"    & class & ERM & ERM & \multicolumn{2}{c}{Bagging-$K{=}5$}"
        r" & \multicolumn{2}{c}{Twin-bootstrap $\lambda{=}300$} \\",
        r"    \cmidrule(lr){5-6} \cmidrule(lr){7-8}",
        r"    Dataset & prior (\%) & hit rate (\%) & Jaccard"
        r" & Jaccard & $\Delta$ vs ERM"
        r" & Jaccard & $\Delta$ vs ERM \\",
        r"    \midrule",
    ]
    def _bold(cell: str, do: bool) -> str:
        return r"\textbf{" + cell + r"}" if do else cell

    for ds in datasets:
        cells = by_ds.get(ds, {})
        if "ERM" not in cells:
            continue
        erm = cells["ERM"]
        bag = cells.get("Bagging-K=5")
        twin = cells.get(f"Twin-lam{int(FROZEN_LAM)}")
        ds_label = display(ds) + (r"\,(dev)" if ds == DEV_DATASET else "")
        # Bold the highest Jaccard across {ERM, Bag, Twin} per row, ties bold
        # both; bold the larger paired Δ vs ERM between Bag and Twin.
        jac_vals = [(erm["jaccard_mean"], "erm")]
        if bag:  jac_vals.append((bag["jaccard_mean"],  "bag"))
        if twin: jac_vals.append((twin["jaccard_mean"], "twin"))
        jac_best = max(v for v, _ in jac_vals)
        jac_winners = {n for v, n in jac_vals if abs(v - jac_best) < 1e-9}
        delta_winners = set()
        if bag and twin and not np.isnan(bag["delta_mean"]) \
                and not np.isnan(twin["delta_mean"]):
            d_best = max(bag["delta_mean"], twin["delta_mean"])
            if abs(bag["delta_mean"] - d_best) < 1e-9:
                delta_winners.add("bag")
            if abs(twin["delta_mean"] - d_best) < 1e-9:
                delta_winners.add("twin")
        lines.append(
            f"    {ds_label} & "
            f"{erm['class_prior']*100:.0f} & "
            f"{erm['hit_rate_mean']*100:.0f} & "
            f"{_bold(cell_jac(erm), 'erm' in jac_winners)} & "
            f"{_bold(cell_jac(bag), 'bag' in jac_winners) if bag else '---'} & "
            f"{_bold(cell_delta(bag), 'bag' in delta_winners) if bag else '---'} & "
            f"{_bold(cell_jac(twin), 'twin' in jac_winners) if twin else '---'} & "
            f"{_bold(cell_delta(twin), 'twin' in delta_winners) if twin else '---'} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  }",
        r"\end{table}",
        "",
    ]
    out_path = Path(args.latex)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
