"""Analyze twin-network fragility-regularization sweep.

Loads outputs/fragility_twin/<dataset>/twin_seed*_lam*.npz and:
  - At each λ, computes within-twin fragility (drops monotonically, by
    construction).
  - At each λ, computes cross-sample churn — fraction of test predictions
    whose argmax differs between the *averaged* twin trained at seed_A
    and the averaged twin trained at seed_B. This is the deployment-time
    metric we care about.
  - Compares to the baseline ERM cross-sample churn from the original
    fragility sweep (outputs/fragility/<dataset>/erm_seed*_k0.npz).
  - Reports id_acc and ood_acc averaged across seeds at each λ.

If cross-sample churn decreases with λ relative to the ERM baseline at
the same accuracy, the twin objective is a valid first solution.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def _sym_kl(p, q, eps=1e-12):
    kl = lambda a, b: (a * (np.log(a + eps) - np.log(b + eps))).sum(axis=-1)
    return 0.5 * (kl(p, q) + kl(q, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/fragility_twin")
    ap.add_argument("--baseline_root", default="outputs/fragility")
    ap.add_argument("--dataset", default="bace")
    ap.add_argument("--seeds", default="42,123,789")
    ap.add_argument("--partition_mode", default=None,
                    choices=[None, "fixed", "per_epoch", "bootstrap", "adversarial"])
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    twin_dir = Path(args.root) / args.dataset
    base_dir = Path(args.baseline_root) / args.dataset

    # Load baseline ERM.
    erms = {}
    for s in seeds:
        f = base_dir / f"erm_seed{s}_k0.npz"
        if f.exists():
            erms[s] = dict(np.load(f, allow_pickle=True))

    # Load twin runs by lambda.
    runs = {}
    glob_pat = (f"twin_{args.partition_mode}_seed*_lam*.npz"
                if args.partition_mode else "twin_seed*_lam*.npz")
    for f in sorted(twin_dir.glob(glob_pat)):
        d = dict(np.load(f, allow_pickle=True))
        seed = int(d["data_seed"])
        lam = float(d["lam"])
        runs.setdefault(lam, {})[seed] = d

    # ERM baseline cross-sample churn.
    erm_churns = []
    for sa, sb in combinations(seeds, 2):
        if sa in erms and sb in erms:
            ya, yb = erms[sa]["id_probs"].argmax(1), erms[sb]["id_probs"].argmax(1)
            erm_churns.append(float((ya != yb).mean()))
    erm_baseline = float(np.mean(erm_churns)) if erm_churns else float("nan")

    print(f"\n=== {args.dataset} — twin-network fragility regularization ===\n")
    print(f"baseline ERM cross-sample churn: {erm_baseline:.1%} "
          f"(over {len(erm_churns)} seed pairs)\n")
    print(f"{'λ':>8s}  {'within_frag':>12s}  {'cross_churn':>12s}  "
          f"{'Δ_vs_erm':>9s}  {'id_acc':>7s}  {'ood_acc':>7s}")

    rows = []
    for lam in sorted(runs):
        seeds_present = sorted(runs[lam])
        if len(seeds_present) < 2:
            continue
        within = np.mean([
            _sym_kl(runs[lam][s]["id_probs_A"],
                    runs[lam][s]["id_probs_B"]).mean()
            for s in seeds_present])
        # Cross-sample churn between *averaged* twins.
        churns = []
        for sa, sb in combinations(seeds_present, 2):
            ya = runs[lam][sa]["id_probs_avg"].argmax(1)
            yb = runs[lam][sb]["id_probs_avg"].argmax(1)
            churns.append(float((ya != yb).mean()))
        cross = float(np.mean(churns))
        # Accuracy.
        accs_id, accs_ood = [], []
        for s in seeds_present:
            d = runs[lam][s]
            accs_id.append(float((d["id_probs_avg"].argmax(1) == d["id_labels"]).mean()))
            accs_ood.append(float((d["ood_probs_avg"].argmax(1) == d["ood_labels"]).mean()))
        rows.append({
            "lam": lam, "within_frag": float(within), "cross_churn": cross,
            "id_acc": float(np.mean(accs_id)), "ood_acc": float(np.mean(accs_ood)),
        })
        delta = (erm_baseline - cross) * 100 if not np.isnan(erm_baseline) else float("nan")
        print(f"{lam:>8.2f}  {float(within):>12.4f}  {cross:>11.1%}  "
              f"{delta:>+8.2f}pp  {np.mean(accs_id):>7.3f}  {np.mean(accs_ood):>7.3f}")

    pd.DataFrame(rows).to_csv("outputs/twin_summary.csv", index=False)
    print(f"\nWrote rows to outputs/twin_summary.csv")


if __name__ == "__main__":
    main()
