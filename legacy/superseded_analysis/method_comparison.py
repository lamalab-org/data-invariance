"""Honest method comparison: ERM | best-val-fixed-λ | adaptive (gradnorm).

Three columns per dataset:
  ERM:               baseline.
  Twin@val-best-λ:   pick λ purely from id (validation) accuracy budget.
                     Rule: largest λ whose id_acc is within --acc_tol of ERM
                     id_acc.  Falls back to ERM if no λ qualifies.  No
                     test-set peeking.
  Twin@gradnorm:     adaptive λ from outputs/cross_sample/<ds>/twin_gradnorm_*.

Reports per-dataset:
  - id_acc, ood_acc (mean ± bootstrap 95% CI)
  - id_churn, ood_churn (mean ± bootstrap 95% CI on the seed-pair distribution)
  - which fixed λ was selected by the val rule
  - mean adaptive λ_t observed during gradnorm training
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)


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
    return {"id_acc": _bootstrap_ci(id_accs),
            "ood_acc": _bootstrap_ci(ood_accs),
            "id_churn": _bootstrap_ci(id_churns),
            "ood_churn": _bootstrap_ci(ood_churns),
            "n": len(runs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/cross_sample")
    ap.add_argument("--lams", default="1.0,3.0,10.0,30.0,100.0,300.0")
    ap.add_argument("--acc_tol", type=float, default=0.02,
                    help="Max id_acc drop allowed for val-anchored selection.")
    args = ap.parse_args()

    root = Path(args.root)
    lams = [float(x) for x in args.lams.split(",")]
    datasets = sorted({p.name for p in root.glob("*") if p.is_dir()})

    print(f"{'dataset':14s}  {'method':22s}  "
          f"{'id_acc':>17s}  {'ood_acc':>17s}  "
          f"{'id_churn':>17s}  {'ood_churn':>17s}")
    rows = []

    for ds in datasets:
        erm_runs = _load_runs(root / ds, "erm_train*.npz")
        erm = metrics(erm_runs)
        if erm is None:
            continue

        # Val-anchored λ selection: largest λ with id_acc[mean] >= ERM id_acc - acc_tol.
        chosen_lam, chosen_metrics = None, None
        for lam in sorted(lams):
            twin_runs = _load_runs(root / ds, f"twin_train*_lam{lam}.npz")
            tw = metrics(twin_runs)
            if tw is None:
                continue
            if tw["id_acc"][0] >= erm["id_acc"][0] - args.acc_tol:
                chosen_lam, chosen_metrics = lam, tw
        # If none qualify, fall back to ERM.
        val_anchored = chosen_metrics if chosen_metrics else erm
        val_lam_str = f"λ={chosen_lam}" if chosen_lam else "fallback to ERM"

        # GradNorm.
        gn_runs = _load_runs(root / ds, "twin_gradnorm_train*.npz")
        gn = metrics(gn_runs) if gn_runs else None

        # Build display rows for this dataset.
        for tag, m in [("ERM", erm),
                       (f"Twin@val-best ({val_lam_str})", val_anchored),
                       ("Twin@gradnorm", gn)]:
            if m is None:
                continue
            ia, oa, ic, oc = m["id_acc"], m["ood_acc"], m["id_churn"], m["ood_churn"]
            print(f"{ds:14s}  {tag:22s}  "
                  f"{ia[0]:.3f} [{ia[1]:.3f},{ia[2]:.3f}]  "
                  f"{oa[0]:.3f} [{oa[1]:.3f},{oa[2]:.3f}]  "
                  f"{ic[0]:.3f} [{ic[1]:.3f},{ic[2]:.3f}]  "
                  f"{oc[0]:.3f} [{oc[1]:.3f},{oc[2]:.3f}]")
            rows.append({"dataset": ds, "method": tag,
                         "id_acc_mean": ia[0], "id_acc_lo": ia[1], "id_acc_hi": ia[2],
                         "ood_acc_mean": oa[0], "ood_acc_lo": oa[1], "ood_acc_hi": oa[2],
                         "id_churn_mean": ic[0], "id_churn_lo": ic[1], "id_churn_hi": ic[2],
                         "ood_churn_mean": oc[0], "ood_churn_lo": oc[1], "ood_churn_hi": oc[2],
                         "n_seeds": m["n"]})
        print()

    pd.DataFrame(rows).to_csv("outputs/method_comparison.csv", index=False)
    print(f"Wrote outputs/method_comparison.csv")


if __name__ == "__main__":
    main()
