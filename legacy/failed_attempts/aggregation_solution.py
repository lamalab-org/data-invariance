"""First-solution test: does partition-pair averaging reduce churn at
deployment time?

We have, for each dataset, 3 independent data_seeds. Each seed produces:
  - 1 ERM model trained on the full data sample
  - K=2 partition models trained on the two disjoint halves of that sample
  - K=5 ensemble models trained on the full sample with different inits

For two independent data_seeds A and B (different data samples), we
compute the per-example argmax-flip rate ("churn") between deployment
strategies:

  - single_erm:        argmax(ERM_A) vs argmax(ERM_B)
  - bagged_partition:  argmax((p_A0+p_A1)/2) vs argmax((p_B0+p_B1)/2)
  - ensembled_5:       argmax(mean p_A_k) vs argmax(mean p_B_k)

If bagged_partition has lower churn than single_erm, then averaging
partition-pair predictions reduces the fragility a practitioner sees
between retrainings — even though each individual partition model saw
*half* the data of a single ERM.

That is an honest, cheap, deployment-ready first solution to fragility.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def _load(d, mode, seed):
    files = sorted(d.glob(f"{mode}_seed{seed}_k*.npz"),
                   key=lambda p: int(p.stem.split("_k")[-1]))
    return [dict(np.load(f, allow_pickle=True)) for f in files]


def churn_between(probs_a, probs_b):
    """Per-example argmax disagreement rate."""
    return float((probs_a.argmax(1) != probs_b.argmax(1)).mean())


def run(root, dataset, seeds):
    d = root / dataset
    erms = {s: _load(d, "erm", s) for s in seeds}
    pars = {s: _load(d, "partition", s) for s in seeds}
    enss = {s: _load(d, "ensemble", s) for s in seeds}

    rows = []
    for sa, sb in combinations(seeds, 2):
        if not erms[sa] or not erms[sb]:
            continue
        # Single ERM churn.
        erm_a = erms[sa][0]["id_probs"]
        erm_b = erms[sb][0]["id_probs"]
        c_erm = churn_between(erm_a, erm_b)

        # Partition-bagged churn.
        if len(pars[sa]) == 2 and len(pars[sb]) == 2:
            avg_a = 0.5 * (pars[sa][0]["id_probs"] + pars[sa][1]["id_probs"])
            avg_b = 0.5 * (pars[sb][0]["id_probs"] + pars[sb][1]["id_probs"])
            c_part = churn_between(avg_a, avg_b)
        else:
            c_part = float("nan")

        # K=5 ensemble churn.
        if len(enss[sa]) >= 2 and len(enss[sb]) >= 2:
            avg_a5 = np.mean([m["id_probs"] for m in enss[sa]], axis=0)
            avg_b5 = np.mean([m["id_probs"] for m in enss[sb]], axis=0)
            c_ens = churn_between(avg_a5, avg_b5)
        else:
            c_ens = float("nan")

        # Accuracy of each strategy on seed A (for tradeoff transparency).
        labels = erms[sa][0]["id_labels"]
        acc_erm = float((erm_a.argmax(1) == labels).mean())
        if len(pars[sa]) == 2:
            avg_a_acc = (avg_a.argmax(1) == labels).mean()
        else:
            avg_a_acc = float("nan")
        if len(enss[sa]) >= 2:
            avg_a5_acc = (avg_a5.argmax(1) == labels).mean()
        else:
            avg_a5_acc = float("nan")

        rows.append({
            "dataset": dataset, "seed_a": sa, "seed_b": sb,
            "churn_erm": c_erm,
            "churn_partition_bag": c_part,
            "churn_ensemble5": c_ens,
            "acc_erm_on_a": acc_erm,
            "acc_partition_on_a": float(avg_a_acc),
            "acc_ensemble5_on_a": float(avg_a5_acc),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/fragility")
    ap.add_argument("--seeds", default="42,123,789")
    ap.add_argument("--out", default="outputs/aggregation_solution.csv")
    args = ap.parse_args()

    root = Path(args.root)
    seeds = [int(s) for s in args.seeds.split(",")]
    datasets = sorted({p.name for p in root.glob("*") if p.is_dir()})

    rows = []
    for ds in datasets:
        rows.extend(run(root, ds, seeds))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    agg = (df.groupby("dataset")
             .agg(churn_erm=("churn_erm", "mean"),
                  churn_part=("churn_partition_bag", "mean"),
                  churn_ens=("churn_ensemble5", "mean"),
                  acc_erm=("acc_erm_on_a", "mean"),
                  acc_part=("acc_partition_on_a", "mean"),
                  acc_ens=("acc_ensemble5_on_a", "mean"))
             .reset_index())
    agg["churn_red_part_pp"] = (agg["churn_erm"] - agg["churn_part"]) * 100
    agg["churn_red_ens_pp"]  = (agg["churn_erm"] - agg["churn_ens"]) * 100
    agg["acc_cost_part_pp"] = (agg["acc_erm"] - agg["acc_part"]) * 100
    agg["acc_cost_ens_pp"]  = (agg["acc_erm"] - agg["acc_ens"]) * 100

    print("\n=== Deployment-time churn between independent training "
          "samples ===\n")
    print(f"{'dataset':16s} {'erm':>6s} {'part_bag':>8s} {'ens5':>6s}    "
          f"{'Δ_part':>7s} {'Δ_ens':>6s}    "
          f"{'acc_cost_part':>13s} {'acc_cost_ens':>12s}")
    for _, r in agg.iterrows():
        print(f"{r['dataset']:16s} {r['churn_erm']:>5.1%} "
              f"{r['churn_part']:>8.1%} {r['churn_ens']:>5.1%}    "
              f"{r['churn_red_part_pp']:>+6.2f}pp {r['churn_red_ens_pp']:>+5.2f}pp    "
              f"{r['acc_cost_part_pp']:>+12.2f}pp "
              f"{r['acc_cost_ens_pp']:>+11.2f}pp")
    print("\nΔ_part: churn reduction from partition-bagged vs single ERM (positive = better)")
    print("acc_cost: accuracy ERM minus method (positive = method costs accuracy)")
    print(f"\nWrote rows to {args.out}")


if __name__ == "__main__":
    main()
