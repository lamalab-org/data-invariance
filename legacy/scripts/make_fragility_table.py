"""Compute sample-fragility statistics across datasets.

For each dataset, given ERM + ensemble(K>=2) + partition(K>=2) NPZ files
produced by partition_pair_train.py, compute:

  - fragility = mean symmetric KL between the two partition models on the
    id_test set (in nats). This is the measurable quantity we call
    "sample fragility" — how much a test prediction depends on which half
    of the training data the model saw.
  - ensemble_var = mean JS divergence across ensemble members on id_test.
    This is the classical deep-ensemble disagreement; different seeds
    see the same data.
  - r_partition_ensemble = Pearson correlation of the two per-example
    disagreement signals.
  - ratio = fragility / ensemble_var — how much bigger the partition
    signal is than the same-data seed-variance signal.

Cross-dataset: we also regress fragility on N_train to check the
theoretical prediction β = O(1/N).

Reports a tidy CSV + a console summary.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _entropy(p, eps=1e-12):
    return -(p * np.log(p + eps)).sum(axis=-1)


def _kl(p, q, eps=1e-12):
    return (p * (np.log(p + eps) - np.log(q + eps))).sum(axis=-1)


def _sym_kl(p, q):
    return 0.5 * (_kl(p, q) + _kl(q, p))


def _js(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (_kl(p, m) + _kl(q, m))


def _pairwise_mean(probs_list, fn):
    n = len(probs_list)
    acc, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            acc = acc + fn(probs_list[i], probs_list[j])
            count += 1
    return acc / max(count, 1)


def _load_mode(root, dataset, mode, seed):
    d = root / dataset
    files = sorted(d.glob(f"{mode}_seed{seed}_k*.npz"),
                   key=lambda p: int(p.stem.split("_k")[-1]))
    return [dict(np.load(f, allow_pickle=True)) for f in files]


def _analyze_one(root, dataset, seed):
    erm = _load_mode(root, dataset, "erm", seed)
    ens = _load_mode(root, dataset, "ensemble", seed)
    par = _load_mode(root, dataset, "partition", seed)
    if not erm or len(ens) < 2 or len(par) < 2:
        return None

    id_probs_erm = erm[0]["id_probs"]
    id_labels = erm[0]["id_labels"]
    id_pred = id_probs_erm.argmax(1)
    id_acc = float((id_pred == id_labels).mean())

    # Size of training data actually seen by each model:
    n_full = int(ens[0]["partition_indices"].size)
    n_part = int(par[0]["partition_indices"].size)

    # Per-example signals on id_test.
    frag = _pairwise_mean([m["id_probs"] for m in par], _sym_kl)
    ensv = _pairwise_mean([m["id_probs"] for m in ens], _js)
    ent  = _entropy(id_probs_erm)

    # Summary statistics.
    out = {
        "dataset": dataset,
        "seed": seed,
        "n_train_full": n_full,
        "n_train_partition": n_part,
        "id_acc_erm": id_acc,
        "fragility_mean": float(frag.mean()),
        "fragility_median": float(np.median(frag)),
        "fragility_p95": float(np.percentile(frag, 95)),
        "ensemble_var_mean": float(ensv.mean()),
        "entropy_mean": float(ent.mean()),
        "r_frag_ens": float(np.corrcoef(frag, ensv)[0, 1]),
        "r_frag_ent": float(np.corrcoef(frag, ent)[0, 1]),
        "ratio_frag_over_ens": (
            float(frag.mean() / ensv.mean()) if ensv.mean() > 0 else float("inf")
        ),
    }

    # Downstream: selective-prediction AUC of each signal against ERM correctness.
    correct = (id_pred == id_labels).astype(float)
    out["sel_auc_entropy"] = _selective_auc(-ent, correct)
    out["sel_auc_ensemble"] = _selective_auc(-ensv, correct)
    out["sel_auc_fragility"] = _selective_auc(-frag, correct)
    return out


def _selective_auc(confidence, correct):
    order = np.argsort(-confidence)
    cs = correct[order]
    n = len(cs)
    coverage = np.arange(1, n + 1) / n
    risk = 1 - np.cumsum(cs) / np.arange(1, n + 1)
    return float(np.trapezoid(risk, coverage))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/fragility")
    ap.add_argument("--seeds", default="42,123,789")
    ap.add_argument("--out", default="outputs/fragility_summary.csv")
    args = ap.parse_args()

    root = Path(args.root)
    seeds = [int(s) for s in args.seeds.split(",")]
    datasets = sorted({p.name for p in root.glob("*") if p.is_dir()})

    rows = []
    for ds in datasets:
        for seed in seeds:
            r = _analyze_one(root, ds, seed)
            if r is not None:
                rows.append(r)

    if not rows:
        print("No datasets ready.")
        return

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # Per-dataset aggregates (mean over seeds).
    agg = (df.groupby("dataset")
             .agg(n_train_full=("n_train_full", "first"),
                  id_acc=("id_acc_erm", "mean"),
                  fragility=("fragility_mean", "mean"),
                  fragility_std=("fragility_mean", "std"),
                  ensemble_var=("ensemble_var_mean", "mean"),
                  entropy=("entropy_mean", "mean"),
                  ratio_fe=("ratio_frag_over_ens", "mean"),
                  r_frag_ens=("r_frag_ens", "mean"),
                  sel_auc_ent=("sel_auc_entropy", "mean"),
                  sel_auc_ens=("sel_auc_ensemble", "mean"),
                  sel_auc_frag=("sel_auc_fragility", "mean"))
             .reset_index()
             .sort_values("n_train_full"))

    print("\n=== Sample fragility by dataset ===")
    print(f"{'dataset':16s} {'N':>7s} {'acc':>5s}  "
          f"{'frag':>8s} {'ens_var':>8s} {'ratio':>7s}  "
          f"{'r_fe':>6s}  {'sel_ent':>8s} {'sel_ens':>8s} {'sel_frag':>8s}")
    for _, r in agg.iterrows():
        print(f"{r['dataset']:16s} {int(r['n_train_full']):>7d} "
              f"{r['id_acc']:>5.3f}  "
              f"{r['fragility']:>8.4f} {r['ensemble_var']:>8.4f} "
              f"{r['ratio_fe']:>7.2f}  "
              f"{r['r_frag_ens']:>+6.3f}  "
              f"{r['sel_auc_ent']:>8.4f} {r['sel_auc_ens']:>8.4f} "
              f"{r['sel_auc_frag']:>8.4f}")

    # Theory check: does fragility scale like 1/N?
    if len(agg) >= 3:
        x = np.log(agg["n_train_full"].to_numpy().astype(float))
        y = np.log(agg["fragility"].to_numpy().astype(float) + 1e-8)
        slope, intercept = np.polyfit(x, y, 1)
        print(f"\nlog(fragility) ~ {slope:.2f} * log(N) + {intercept:.2f}")
        print(f"(β = O(1/N) predicts slope ≈ -1; "
              f"slope observed = {slope:+.2f})")

    print(f"\nWrote per-seed rows to {args.out}")


if __name__ == "__main__":
    main()
