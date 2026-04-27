"""Prediction churn simulation from existing partition-pair NPZs.

Given two models trained on disjoint halves of the training data, compute
the test-time fraction of examples whose argmax prediction differs. This is
the concrete industrial quantity — *if I retrain next week on a different
data sample, what fraction of my per-example predictions will change?* —
that sample fragility predicts in the limit.

Also reports:
  - churn on the subset where fragility is above its 90th percentile
    vs below its 10th percentile. If fragility is a meaningful ranking of
    prediction-stability, high-fragility examples should churn much more.
  - churn vs ensemble disagreement (which holds data fixed) for comparison.

Output: stdout table, plus CSV at outputs/fragility_churn.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load(root, dataset, mode, seed):
    d = root / dataset
    files = sorted(d.glob(f"{mode}_seed{seed}_k*.npz"),
                   key=lambda p: int(p.stem.split("_k")[-1]))
    return [dict(np.load(f, allow_pickle=True)) for f in files]


def _sym_kl(p, q, eps=1e-12):
    kl = lambda a, b: (a * (np.log(a + eps) - np.log(b + eps))).sum(axis=-1)
    return 0.5 * (kl(p, q) + kl(q, p))


def analyze(root, dataset, seeds):
    rows = []
    for seed in seeds:
        par = _load(root, dataset, "partition", seed)
        ens = _load(root, dataset, "ensemble", seed)
        if len(par) < 2 or len(ens) < 2:
            continue
        pA, pB = par[0]["id_probs"], par[1]["id_probs"]
        fragility = _sym_kl(pA, pB)
        churn_partition = (pA.argmax(1) != pB.argmax(1)).astype(float)
        # Ensemble-based churn: use two arbitrary ensemble members as the
        # "same-data, different-seed" comparison.
        eA, eB = ens[0]["id_probs"], ens[1]["id_probs"]
        churn_ensemble = (eA.argmax(1) != eB.argmax(1)).astype(float)

        # Fragility-stratified churn: top vs bottom decile.
        q10 = np.percentile(fragility, 10)
        q90 = np.percentile(fragility, 90)
        low  = fragility <= q10
        high = fragility >= q90

        n_test = len(fragility)
        rows.append({
            "dataset": dataset,
            "seed": seed,
            "n_test": n_test,
            "churn_partition": float(churn_partition.mean()),
            "churn_ensemble":  float(churn_ensemble.mean()),
            "churn_high_frag_decile": float(churn_partition[high].mean()) if high.any() else float("nan"),
            "churn_low_frag_decile":  float(churn_partition[low].mean())  if low.any() else float("nan"),
            "fragility_mean": float(fragility.mean()),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/fragility")
    ap.add_argument("--seeds", default="42,123,789")
    ap.add_argument("--out", default="outputs/fragility_churn.csv")
    args = ap.parse_args()

    root = Path(args.root)
    seeds = [int(s) for s in args.seeds.split(",")]
    datasets = sorted({p.name for p in root.glob("*") if p.is_dir()})

    rows = []
    for ds in datasets:
        rows.extend(analyze(root, ds, seeds))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    agg = (df.groupby("dataset")
             .agg(n_test=("n_test", "first"),
                  churn_partition=("churn_partition", "mean"),
                  churn_partition_std=("churn_partition", "std"),
                  churn_ensemble=("churn_ensemble", "mean"),
                  churn_high_frag=("churn_high_frag_decile", "mean"),
                  churn_low_frag=("churn_low_frag_decile", "mean"))
             .reset_index())

    print("\n=== Prediction churn (fraction of test examples whose argmax "
          "flips between partition A and B) ===\n")
    print(f"{'dataset':16s} {'n_test':>7s}  "
          f"{'part':>7s} {'(±)':>6s}   "
          f"{'ensemble':>8s}   "
          f"{'top10%_frag':>11s}  {'bot10%_frag':>11s}  {'discrim':>8s}")
    for _, r in agg.iterrows():
        discrim = (r["churn_high_frag"] / r["churn_low_frag"]
                   if r["churn_low_frag"] > 0 else float("inf"))
        print(f"{r['dataset']:16s} {int(r['n_test']):>7d}  "
              f"{r['churn_partition']:>7.1%} "
              f"{(r['churn_partition_std'] or 0):>6.1%}   "
              f"{r['churn_ensemble']:>8.1%}   "
              f"{r['churn_high_frag']:>11.1%}  "
              f"{r['churn_low_frag']:>11.1%}  "
              f"{discrim:>7.1f}x")

    print(f"\nWrote rows to {args.out}")


if __name__ == "__main__":
    main()
