"""Pareto curve on the *development* dataset (BACE).

For each candidate λ in the grid, reports id_acc, ood_acc, id_churn, and
ood_churn (with bootstrap 95 % CIs across train_seeds and seed pairs).
This is what we used to pick the frozen λ for the held-out evaluation —
``make_main_table.py`` consumes that frozen value.

Usage::

    uv run python scripts/make_pareto.py --dataset bace
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from _analysis_lib import (
    GLOBS, bootstrap_ci, fmt_ci, load_runs, pairwise_metrics,
    per_run_accuracies,
)


DEFAULT_LAMS = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def metrics_with_cis(dataset_dir: Path, glob: str):
    runs = load_runs(dataset_dir, glob)
    if not runs:
        return None
    id_accs, ood_accs = per_run_accuracies(runs)
    pair_metrics, _ = pairwise_metrics(runs)
    rows = list(pair_metrics.values())
    return {
        "n_seeds":   len(runs),
        "id_acc":    bootstrap_ci(id_accs),
        "ood_acc":   bootstrap_ci(ood_accs),
        "id_churn":  bootstrap_ci(r["id_churn"] for r in rows),
        "ood_churn": bootstrap_ci(r["ood_churn"] for r in rows),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/cross_sample")
    ap.add_argument("--dataset", default="bace",
                    help="Development dataset; held-out is reported by make_main_table.py.")
    ap.add_argument("--lams", type=float, nargs="+", default=DEFAULT_LAMS)
    ap.add_argument("--acc_tol", type=float, default=0.02,
                    help="Max id_acc drop (vs ERM) considered acceptable.")
    ap.add_argument("--csv", default="outputs/pareto_table.csv")
    args = ap.parse_args()
    ds_dir = Path(args.root) / args.dataset

    methods = [("ERM", GLOBS["erm"])]
    methods += [(f"Twin_indep λ={lam}",
                 GLOBS["twin_indep"].format(lam=lam)) for lam in args.lams]

    rows = []
    print(f"== Pareto curve on development dataset: {args.dataset} ==\n")
    print(f"{'method':24s}  {'id_acc':>17s}  {'ood_acc':>17s}  "
          f"{'id_churn':>17s}  {'ood_churn':>17s}")

    erm = None
    for tag, glob in methods:
        m = metrics_with_cis(ds_dir, glob)
        if m is None:
            continue
        if tag == "ERM":
            erm = m
        print(f"{tag:24s}  "
              f"{fmt_ci(m['id_acc']):>17s}  {fmt_ci(m['ood_acc']):>17s}  "
              f"{fmt_ci(m['id_churn']):>17s}  {fmt_ci(m['ood_churn']):>17s}")
        rows.append({"dataset": args.dataset, "method": tag, "n_seeds": m["n_seeds"],
                     **{f"{k}_{x}": m[k][i]
                        for k in ("id_acc", "ood_acc", "id_churn", "ood_churn")
                        for i, x in enumerate(("mean", "lo", "hi"))}})

    # Pre-registered selection rule.
    if erm:
        floor = erm["id_acc"][0] - args.acc_tol
        feasible = [(tag, m) for tag, m in
                    ((tag, metrics_with_cis(ds_dir, glob)) for tag, glob in methods if tag != "ERM")
                    if m is not None and m["id_acc"][0] >= floor]
        if feasible:
            picked_tag, picked = max(feasible, key=lambda t: float(t[0].split("=")[1]))
            print(f"\n>>> Pre-registered rule (largest λ with id_acc ≥ ERM−{args.acc_tol}):")
            print(f"    {picked_tag}  →  freeze for held-out evaluation")
        else:
            print(f"\n>>> No λ satisfies id_acc ≥ ERM−{args.acc_tol}. Use ERM as method.")

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
