"""Strict dev/test split: pre-register a selection rule on BACE only,
freeze the hyperparameter, evaluate on the 4 held-out datasets without
further tuning.

Pre-registered rule (decided before looking at held-out data):
  Among Twin λ ∈ {1, 3, 10, 30, 100, 300}, pick the largest λ such that
  BACE id_acc[mean] ≥ ERM id_acc[mean] − 0.02.  Rationale: this is the
  natural "no-significant-accuracy-cost" knee, with a conventional 2pp
  tolerance.  Larger λ is preferred among feasible ones because the
  Pareto curve shows churn reduction is monotone-ish in λ within the
  feasible region.

Reports two methods on held-out datasets:
  - Twin@λ_dev:           fixed-λ chosen on BACE only.
  - Twin@gradnorm tr=1.0: parameter-free, no tuning at all.

Held-out (test) datasets: BBBP, TADF, MOF_solvent, MOF_thermal.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)
DEV_DATASET = "bace"
TEST_DATASETS = ["bbbp", "tadf", "mof_solvent", "mof_thermal"]
ACC_TOL = 0.02
LAMBDA_GRID = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def _bootstrap_ci(arr, n_boot=10_000, alpha=0.05):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = RNG.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _load_runs(d, glob):
    files = sorted(d.glob(glob),
                   key=lambda p: int(p.stem.split("train")[1].split("_")[0]))
    return [(int(f.stem.split("train")[1].split("_")[0]),
             dict(np.load(f, allow_pickle=True))) for f in files]


def _get_probs(d):
    idp = d["id_probs_avg"] if "id_probs_avg" in d else d["id_probs"]
    odp = d["ood_probs_avg"] if "ood_probs_avg" in d else d["ood_probs"]
    return idp, odp


def metrics(runs):
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
        "n": len(runs),
        "id_acc": _bootstrap_ci(id_accs),
        "ood_acc": _bootstrap_ci(ood_accs),
        "id_churn": _bootstrap_ci(id_churns),
        "ood_churn": _bootstrap_ci(ood_churns),
    }


def fmt(t):
    return f"{t[0]:.3f} [{t[1]:.3f},{t[2]:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/cross_sample")
    args = ap.parse_args()
    root = Path(args.root)

    # ---- DEVELOPMENT on BACE ----
    print("=" * 78)
    print(f"DEVELOPMENT (BACE only) — pre-registered rule:")
    print(f"  pick largest λ in {LAMBDA_GRID} such that id_acc ≥ ERM − {ACC_TOL}")
    print("=" * 78)

    bace_erm = metrics(_load_runs(root / DEV_DATASET, "erm_train*.npz"))
    print(f"  BACE  ERM             id_acc={fmt(bace_erm['id_acc'])}")

    qualifying = []
    for lam in LAMBDA_GRID:
        m = metrics(_load_runs(root / DEV_DATASET,
                               f"twin_train*_lam{lam}.npz"))
        if m is None:
            continue
        gap = bace_erm["id_acc"][0] - m["id_acc"][0]
        ok = gap <= ACC_TOL
        marker = "✓" if ok else "✗"
        print(f"  BACE  Twin λ={lam:>5}  id_acc={fmt(m['id_acc'])}  "
              f"gap={gap:+.3f}  {marker}")
        if ok:
            qualifying.append((lam, m))

    if not qualifying:
        chosen_lam = None
        print("\n  No λ qualifies → falling back to ERM as frozen choice.")
    else:
        chosen_lam, _ = max(qualifying, key=lambda t: t[0])
        print(f"\n  >>> FROZEN λ = {chosen_lam}  (largest qualifying)")

    # GradNorm has no λ to tune. We treat target_ratio=1.0 as the theory
    # default, set a priori, never modified.
    print("\n  GradNorm twin: target_ratio=1.0 (a priori, no tuning).")

    # ---- TEST on the 4 held-out datasets ----
    print()
    print("=" * 78)
    print("TEST (held-out: BBBP, TADF, MOF_solvent, MOF_thermal)")
    print("Frozen Twin λ from BACE-development; GradNorm with tr=1.0 untouched.")
    print("=" * 78)
    print(f"{'dataset':14s}  {'method':22s}  "
          f"{'id_acc':>20s}  {'ood_acc':>20s}  "
          f"{'id_churn':>20s}  {'ood_churn':>20s}")

    rows = []
    for ds in TEST_DATASETS:
        erm = metrics(_load_runs(root / ds, "erm_train*.npz"))
        if erm is None:
            continue
        rows_ds = [("ERM", erm)]
        if chosen_lam is not None:
            tw = metrics(_load_runs(root / ds, f"twin_train*_lam{chosen_lam}.npz"))
            if tw is not None:
                rows_ds.append((f"Twin λ={chosen_lam} (frozen)", tw))
        gn = metrics(_load_runs(root / ds, "twin_gradnorm_train*.npz"))
        if gn is not None:
            rows_ds.append(("Twin gradnorm tr=1.0", gn))

        for tag, m in rows_ds:
            print(f"{ds:14s}  {tag:22s}  "
                  f"{fmt(m['id_acc']):>20s}  {fmt(m['ood_acc']):>20s}  "
                  f"{fmt(m['id_churn']):>20s}  {fmt(m['ood_churn']):>20s}")
            row_data = {"dataset": ds, "method": tag, "n_seeds": m["n"]}
            for k in ("id_acc", "ood_acc", "id_churn", "ood_churn"):
                for i, x in enumerate(("mean", "lo", "hi")):
                    row_data[f"{k}_{x}"] = m[k][i]
            rows.append(row_data)
        print()

    pd.DataFrame(rows).to_csv("outputs/strict_dev_test.csv", index=False)
    print(f"Wrote outputs/strict_dev_test.csv")

    # ---- Aggregate effect size across held-out datasets ----
    print()
    print("=" * 78)
    print("AGGREGATE across 4 held-out datasets (mean of per-dataset means):")
    print("=" * 78)
    df = pd.DataFrame(rows)
    for method in df["method"].unique():
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        # Average the means across datasets.
        print(f"  {method:24s}  "
              f"id_acc={sub['id_acc_mean'].mean():.3f}  "
              f"ood_acc={sub['ood_acc_mean'].mean():.3f}  "
              f"id_churn={sub['id_churn_mean'].mean():.3f}  "
              f"ood_churn={sub['ood_churn_mean'].mean():.3f}")


if __name__ == "__main__":
    main()
