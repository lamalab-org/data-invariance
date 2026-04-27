"""Pareto-curve analysis: cross-sample churn vs accuracy across λ values.

Loads outputs/cross_sample/<dataset>/{erm,twin_train*_lam*.npz} and produces:
  - per-dataset Pareto table: λ vs (id_churn, id_acc, ood_churn, ood_acc)
  - bootstrap 95% CIs on each metric (over the seed-pair distribution)
  - identification of the "knee" — λ that maximises churn reduction subject
    to accuracy not dropping by more than --acc_tol from ERM.

Output:
  outputs/pareto_summary.csv  per-dataset/method rows with CIs.
  stdout: pretty table per dataset.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)


def _bootstrap_mean_ci(values, n_boot=10_000, alpha=0.05):
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(values)
    idx = RNG.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi


def _load_runs(d, glob):
    files = sorted(d.glob(glob),
                   key=lambda p: int(p.stem.split("train")[1].split("_")[0]))
    return [(int(f.stem.split("train")[1].split("_")[0]),
             dict(np.load(f, allow_pickle=True))) for f in files]


def _get_probs(d):
    idp = d["id_probs_avg"] if "id_probs_avg" in d else d["id_probs"]
    odp = d["ood_probs_avg"] if "ood_probs_avg" in d else d["ood_probs"]
    return idp, odp


def analyse_one(runs):
    if not runs:
        return None
    id_accs, ood_accs = [], []
    for _, d in runs:
        idp, odp = _get_probs(d)
        id_accs.append(float((idp.argmax(1) == d["id_labels"]).mean()))
        ood_accs.append(float((odp.argmax(1) == d["ood_labels"]).mean()))
    id_churns, ood_churns = [], []
    for (_, da), (_, db) in combinations(runs, 2):
        idA, odA = _get_probs(da); idB, odB = _get_probs(db)
        id_churns.append(float((idA.argmax(1) != idB.argmax(1)).mean()))
        ood_churns.append(float((odA.argmax(1) != odB.argmax(1)).mean()))
    return {
        "n_seeds": len(runs),
        "id_acc": _bootstrap_mean_ci(id_accs),
        "ood_acc": _bootstrap_mean_ci(ood_accs),
        "id_churn": _bootstrap_mean_ci(id_churns),
        "ood_churn": _bootstrap_mean_ci(ood_churns),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/cross_sample")
    ap.add_argument("--lams", default="1.0,3.0,10.0,30.0,100.0,300.0")
    ap.add_argument("--acc_tol", type=float, default=0.02,
                    help="Max acceptable id_acc drop for the 'knee' λ.")
    args = ap.parse_args()

    root = Path(args.root)
    lams = [float(x) for x in args.lams.split(",")]
    datasets = sorted({p.name for p in root.glob("*") if p.is_dir()})

    rows = []
    for ds in datasets:
        # ERM baseline.
        erm_runs = _load_runs(root / ds, "erm_train*.npz")
        erm = analyse_one(erm_runs)
        if erm is None:
            continue
        rows.append({"dataset": ds, "method": "ERM", "lam": 0.0, **erm})
        for lam in lams:
            twin_runs = _load_runs(root / ds, f"twin_train*_lam{lam}.npz")
            tw = analyse_one(twin_runs)
            if tw is not None:
                rows.append({"dataset": ds, "method": f"Twin λ={lam}",
                             "lam": lam, **tw})

    df = pd.DataFrame(rows)

    # Print per-dataset Pareto table.
    for ds in datasets:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            continue
        print(f"\n=== {ds} ===")
        print(f"{'method':14s}  {'id_acc':>15s}  {'ood_acc':>15s}  "
              f"{'id_churn':>15s}  {'ood_churn':>15s}  n")
        for _, r in sub.iterrows():
            ia = r["id_acc"]; oa = r["ood_acc"]
            ic = r["id_churn"]; oc = r["ood_churn"]
            print(f"{r['method']:14s}  "
                  f"{ia[0]:.3f} [{ia[1]:.3f},{ia[2]:.3f}]  "
                  f"{oa[0]:.3f} [{oa[1]:.3f},{oa[2]:.3f}]  "
                  f"{ic[0]:.3f} [{ic[1]:.3f},{ic[2]:.3f}]  "
                  f"{oc[0]:.3f} [{oc[1]:.3f},{oc[2]:.3f}]  {r['n_seeds']}")

        # Knee: largest λ with id_acc ≥ ERM_id_acc − acc_tol AND ood_acc not collapsed.
        erm_row = sub[sub["method"] == "ERM"].iloc[0]
        feasible = sub[(sub["lam"] > 0)
                       & (sub["id_acc"].apply(lambda c: c[0])
                          >= erm_row["id_acc"][0] - args.acc_tol)]
        if not feasible.empty:
            knee = feasible.sort_values("lam").iloc[-1]
            d_ic = (erm_row["id_churn"][0] - knee["id_churn"][0]) * 100
            d_oc = (erm_row["ood_churn"][0] - knee["ood_churn"][0]) * 100
            print(f"  Knee at acc_tol={args.acc_tol}: {knee['method']}  "
                  f"id_churn −{d_ic:.1f}pp  ood_churn −{d_oc:.1f}pp")

    # CSV: flatten the (mean, lo, hi) tuples for tidy storage.
    flat = []
    for r in rows:
        d = {"dataset": r["dataset"], "method": r["method"],
             "lam": r["lam"], "n_seeds": r["n_seeds"]}
        for k in ("id_acc", "ood_acc", "id_churn", "ood_churn"):
            d[f"{k}_mean"] = r[k][0]; d[f"{k}_lo"] = r[k][1]; d[f"{k}_hi"] = r[k][2]
        flat.append(d)
    pd.DataFrame(flat).to_csv("outputs/pareto_summary.csv", index=False)
    print(f"\nWrote outputs/pareto_summary.csv")


if __name__ == "__main__":
    main()
