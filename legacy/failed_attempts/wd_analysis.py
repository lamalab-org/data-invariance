"""Weight-decay sweep analysis: does regularization reduce fragility?

For each (dataset, weight_decay) combination, computes mean fragility on
id_test (sym-KL between partition pair) and ERM id/ood accuracy. Reports a
table and the slope of log(fragility) vs log(weight_decay).

Theoretical motivation: algorithmic stability β tightens with stronger
regularization (Bousquet & Elisseeff §3). If fragility is a good proxy
for β, fragility should decrease monotonically with weight_decay — at the
expected cost of some accuracy. Demonstrates fragility is *actionable*
via a knob practitioners already have.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _sym_kl(p, q, eps=1e-12):
    kl = lambda a, b: (a * (np.log(a + eps) - np.log(b + eps))).sum(axis=-1)
    return 0.5 * (kl(p, q) + kl(q, p))


def _load(d, mode, seed):
    files = sorted(d.glob(f"{mode}_seed{seed}_k*.npz"),
                   key=lambda p: int(p.stem.split("_k")[-1]))
    return [dict(np.load(f, allow_pickle=True)) for f in files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/fragility_wd")
    ap.add_argument("--datasets", default="bace,bbbp,tadf")
    ap.add_argument("--seeds", default="42,123,789")
    ap.add_argument("--out", default="outputs/wd_summary.csv")
    args = ap.parse_args()

    root = Path(args.root)
    seeds = [int(s) for s in args.seeds.split(",")]
    datasets = args.datasets.split(",")

    rows = []
    for wd_dir in sorted(root.glob("wd*")):
        wd_str = wd_dir.name.replace("wd", "")
        wd = float(wd_str)
        for ds in datasets:
            d = wd_dir / ds
            if not d.exists():
                continue
            for seed in seeds:
                par = _load(d, "partition", seed)
                erm = _load(d, "erm", seed)
                if len(par) < 2 or not erm:
                    continue
                frag = _sym_kl(par[0]["id_probs"], par[1]["id_probs"])
                pred = erm[0]["id_probs"].argmax(1)
                id_acc = float((pred == erm[0]["id_labels"]).mean())
                ood_pred = erm[0]["ood_probs"].argmax(1)
                ood_acc = float((ood_pred == erm[0]["ood_labels"]).mean())
                churn = float((par[0]["id_probs"].argmax(1)
                               != par[1]["id_probs"].argmax(1)).mean())
                rows.append({
                    "dataset": ds, "wd": wd, "seed": seed,
                    "fragility": float(frag.mean()),
                    "id_acc": id_acc, "ood_acc": ood_acc, "churn": churn,
                })

    if not rows:
        print("No data yet.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    for ds in datasets:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            continue
        agg = (sub.groupby("wd")
                  .agg(fragility=("fragility", "mean"),
                       fragility_std=("fragility", "std"),
                       id_acc=("id_acc", "mean"),
                       ood_acc=("ood_acc", "mean"),
                       churn=("churn", "mean"))
                  .reset_index()
                  .sort_values("wd"))
        print(f"\n=== {ds} weight-decay sweep ===")
        print(f"{'wd':>8s}  {'fragility':>10s} {'(±)':>8s}  "
              f"{'id_acc':>7s}  {'ood_acc':>7s}  {'churn':>6s}")
        for _, r in agg.iterrows():
            print(f"{r['wd']:>8.0e}  {r['fragility']:>10.4f} "
                  f"{(r['fragility_std'] or 0):>8.4f}  "
                  f"{r['id_acc']:>7.3f}  {r['ood_acc']:>7.3f}  "
                  f"{r['churn']:>6.1%}")
        if len(agg) >= 3:
            x = np.log(agg["wd"].to_numpy().astype(float))
            y = np.log(agg["fragility"].to_numpy().astype(float))
            slope, _ = np.polyfit(x, y, 1)
            print(f"  log(fragility) ~ {slope:+.2f} * log(wd) + c")
            print(f"  (negative slope → fragility ↓ as regularization ↑)")

    print(f"\nWrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
