"""Within-dataset N-scaling analysis: fragility(M) curve on BACE.

Loads partition-pair and ERM NPZs from outputs/fragility_nscaling/M{200,…}/
and reports:
  - fragility (mean sym-KL on id_test) vs M
  - ERM id/ood accuracy vs M
  - log-log fit:  log(fragility) = alpha * log(M) + c

Theory (β = O(1/M) from algorithmic stability) predicts alpha ≈ -1.
The cross-dataset fit gave alpha = -0.88 but mixed four domains. This
controlled within-dataset experiment isolates N from domain/difficulty.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _sym_kl(p, q, eps=1e-12):
    kl = lambda a, b: (a * (np.log(a + eps) - np.log(b + eps))).sum(axis=-1)
    return 0.5 * (kl(p, q) + kl(q, p))


def _load(root, dataset, mode, seed):
    d = root / dataset
    files = sorted(d.glob(f"{mode}_seed{seed}_k*.npz"),
                   key=lambda p: int(p.stem.split("_k")[-1]))
    return [dict(np.load(f, allow_pickle=True)) for f in files]


def run(root, dataset, seeds):
    root = Path(root)
    sizes = sorted([int(p.name.lstrip("M")) for p in root.glob("M*")])
    rows = []
    for M in sizes:
        for seed in seeds:
            par = _load(root / f"M{M}", dataset, "partition", seed)
            erm = _load(root / f"M{M}", dataset, "erm", seed)
            if len(par) < 2 or not erm:
                continue
            frag = _sym_kl(par[0]["id_probs"], par[1]["id_probs"])
            pred = erm[0]["id_probs"].argmax(1)
            id_acc = float((pred == erm[0]["id_labels"]).mean())
            ood_pred = erm[0]["ood_probs"].argmax(1)
            ood_acc = float((ood_pred == erm[0]["ood_labels"]).mean())
            rows.append({
                "M": M, "seed": seed,
                "fragility": float(frag.mean()),
                "fragility_median": float(np.median(frag)),
                "id_acc": id_acc, "ood_acc": ood_acc,
                "churn": float((par[0]["id_probs"].argmax(1)
                                != par[1]["id_probs"].argmax(1)).mean()),
            })
    df = pd.DataFrame(rows)

    agg = (df.groupby("M")
             .agg(fragility=("fragility", "mean"),
                  fragility_std=("fragility", "std"),
                  id_acc=("id_acc", "mean"),
                  ood_acc=("ood_acc", "mean"),
                  churn=("churn", "mean"))
             .reset_index()
             .sort_values("M"))

    print(f"\n=== {dataset} within-dataset N-scaling ===\n")
    print(f"{'M':>6s}  {'fragility':>10s} {'(±)':>8s}  "
          f"{'id_acc':>7s}  {'ood_acc':>7s}  {'churn':>6s}")
    for _, r in agg.iterrows():
        print(f"{int(r['M']):>6d}  {r['fragility']:>10.4f} "
              f"{(r['fragility_std'] or 0):>8.4f}  "
              f"{r['id_acc']:>7.3f}  {r['ood_acc']:>7.3f}  "
              f"{r['churn']:>6.1%}")

    if len(agg) >= 3:
        x = np.log(agg["M"].to_numpy().astype(float))
        y = np.log(agg["fragility"].to_numpy().astype(float))
        slope, intercept = np.polyfit(x, y, 1)
        print(f"\nlog(fragility) ~ {slope:+.2f} * log(M) + {intercept:+.2f}")
        print(f"  theory predicts slope ≈ -1 "
              f"(β = O(1/N) in Bousquet & Elisseeff 2002)")
        print(f"  cross-dataset fit (earlier, 8 datasets): -0.88")
        print(f"  within-dataset fit (BACE only):           {slope:.2f}")
    return df, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/fragility_nscaling")
    ap.add_argument("--dataset", default="bace")
    ap.add_argument("--seeds", default="42,123,789")
    ap.add_argument("--out", default="outputs/nscaling_summary.csv")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    df, agg = run(args.root, args.dataset, seeds)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote per-seed rows to {args.out}")


if __name__ == "__main__":
    main()
