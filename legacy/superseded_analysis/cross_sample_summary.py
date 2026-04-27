"""Summary table for cross-sample fragility experiments (fixed test set).

Loads outputs/cross_sample/<dataset>/{erm,twin}_train*_lam*.npz and
reports per-dataset cross-sample churn for ERM vs Twin λ=10 vs Twin λ=100,
plus id_acc and ood_acc.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def _load(d, glob):
    files = sorted(d.glob(glob))
    if not files:
        return None
    return [(int(f.stem.split("train")[1].split("_")[0]),
             dict(np.load(f, allow_pickle=True))) for f in files]


def _get(d, kind):
    """Return id/ood probs and labels regardless of erm/twin format."""
    idp = d["id_probs_avg"] if "id_probs_avg" in d else d["id_probs"]
    odp = d["ood_probs_avg"] if "ood_probs_avg" in d else d["ood_probs"]
    return idp, odp, d["id_labels"], d["ood_labels"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/cross_sample")
    args = ap.parse_args()
    root = Path(args.root)

    methods = [
        ("ERM",         "erm_train*.npz"),
        ("Twin λ=10",   "twin_train*_lam10.0.npz"),
        ("Twin λ=100",  "twin_train*_lam100.0.npz"),
    ]
    datasets = sorted({p.name for p in root.glob("*") if p.is_dir()})

    rows = []
    for ds in datasets:
        for tag, glob in methods:
            data = _load(root / ds, glob)
            if not data:
                continue
            id_accs, ood_accs = [], []
            for _, d in data:
                idp, odp, idy, ody = _get(d, "")
                id_accs.append((idp.argmax(1) == idy).mean())
                ood_accs.append((odp.argmax(1) == ody).mean())
            id_churns, ood_churns = [], []
            for (sa, da), (sb, db) in combinations(data, 2):
                idA, odA, _, _ = _get(da, "")
                idB, odB, _, _ = _get(db, "")
                id_churns.append((idA.argmax(1) != idB.argmax(1)).mean())
                ood_churns.append((odA.argmax(1) != odB.argmax(1)).mean())
            rows.append({
                "dataset": ds, "method": tag,
                "id_acc": float(np.mean(id_accs)),
                "ood_acc": float(np.mean(ood_accs)),
                "id_churn": float(np.mean(id_churns)),
                "id_churn_std": float(np.std(id_churns)),
                "ood_churn": float(np.mean(ood_churns)),
                "ood_churn_std": float(np.std(ood_churns)),
            })

    df = pd.DataFrame(rows)
    print(f"{'dataset':14s} {'method':12s}  {'id_acc':>7s} {'ood_acc':>7s}  "
          f"{'id_churn':>13s}  {'ood_churn':>13s}")
    for ds in datasets:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            continue
        for _, r in sub.iterrows():
            print(f"{r['dataset']:14s} {r['method']:12s}  "
                  f"{r['id_acc']:>7.3f} {r['ood_acc']:>7.3f}  "
                  f"{r['id_churn']:>7.3f}±{r['id_churn_std']:.3f}  "
                  f"{r['ood_churn']:>7.3f}±{r['ood_churn_std']:.3f}")
        # Reduction at λ=100 vs ERM, if both present.
        erm_row = sub[sub["method"] == "ERM"]
        twin_row = sub[sub["method"] == "Twin λ=100"]
        if not erm_row.empty and not twin_row.empty:
            d_id = (erm_row["id_churn"].iloc[0] - twin_row["id_churn"].iloc[0]) * 100
            d_ood = (erm_row["ood_churn"].iloc[0] - twin_row["ood_churn"].iloc[0]) * 100
            rel_id = d_id / (erm_row["id_churn"].iloc[0] * 100) * 100
            rel_ood = d_ood / (erm_row["ood_churn"].iloc[0] * 100 + 1e-9) * 100
            print(f"{'  Δ vs ERM':14s} {'(λ=100)':12s}  "
                  f"  id: −{d_id:.2f}pp ({rel_id:.0f}% relative)  "
                  f"ood: −{d_ood:.2f}pp ({rel_ood:.0f}% relative)")
        print()

    df.to_csv("outputs/cross_sample_summary.csv", index=False)
    print(f"Wrote outputs/cross_sample_summary.csv")


if __name__ == "__main__":
    main()
