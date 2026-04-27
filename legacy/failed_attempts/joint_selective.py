"""Fragility-aware selective prediction targeting RELIABLE predictions.

A prediction is "reliable" if it is BOTH correct AND stable under retraining
(the partition pair agrees on its argmax). Existing uncertainty methods aim
at correctness only; we ask whether combining fragility with entropy
identifies *reliable* predictions better than either alone.

Reports two downstream metrics:
  - selective AUC for *correct* predictions (the standard task; we expect
    entropy to dominate — fragility cannot help here).
  - selective AUC for *reliable* predictions (correct AND stable; we
    expect joint-rank to win because each signal handles one component).

The joint score is min(rank_entropy, rank_fragility) reversed — i.e.,
accept examples that are LOW on BOTH axes. This is the correct combination
when both signals must be low simultaneously for trustworthy deployment.
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
    acc, cnt = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            acc = acc + fn(probs_list[i], probs_list[j]); cnt += 1
    return acc / max(cnt, 1)


def _load(root, ds, mode, seed):
    files = sorted((root / ds).glob(f"{mode}_seed{seed}_k*.npz"),
                   key=lambda p: int(p.stem.split("_k")[-1]))
    return [dict(np.load(f, allow_pickle=True)) for f in files]


def selective_curve(score, correct, coverages):
    """For each coverage c, accept top-c (lowest-score) examples and report
    accuracy on the accepted set. score is the rejection metric (high=reject)."""
    order = np.argsort(score)
    rows = []
    n = len(score)
    for c in coverages:
        k = max(1, int(round(c * n)))
        kept = order[:k]
        rows.append((c, float(correct[kept].mean())))
    return rows


def joint_or_score(frag, ent):
    """Return a per-example combined rank: minimum normalized rank across
    the two scores. Lower = more confident on BOTH dimensions; higher =
    fragile OR uncertain. Used as a rejection score (reject the high)."""
    rf = (-frag).argsort().argsort()  # ascending rank by fragility
    re = (-ent).argsort().argsort()
    # Combined: max(rank_frag, rank_ent) → "safe" = low on both
    # equivalently, reject any example that is high on either axis.
    combined = np.maximum(rf, re)
    return -combined.astype(float)  # higher combined rank = more confident


def aurc(curve):
    """Area under the risk-coverage curve (lower is better risk).
    risk(c) = 1 - acc(c)."""
    cs = np.array([c for c, _ in curve])
    risk = 1 - np.array([a for _, a in curve])
    return float(np.trapezoid(risk, cs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/fragility")
    ap.add_argument("--seeds", default="42,123,789")
    ap.add_argument("--out", default="outputs/joint_selective.csv")
    args = ap.parse_args()

    root = Path(args.root)
    seeds = [int(s) for s in args.seeds.split(",")]
    datasets = sorted({p.name for p in root.glob("*") if p.is_dir()})
    coverages = np.linspace(0.05, 1.0, 20)

    rows = []
    for ds in datasets:
        for seed in seeds:
            erm = _load(root, ds, "erm", seed)
            ens = _load(root, ds, "ensemble", seed)
            par = _load(root, ds, "partition", seed)
            if not erm or len(ens) < 2 or len(par) < 2:
                continue
            p_erm = erm[0]["id_probs"]
            y = erm[0]["id_labels"]
            correct = (p_erm.argmax(1) == y).astype(float)

            # "Stable" = the two partition models agree on argmax.
            pA, pB = par[0]["id_probs"], par[1]["id_probs"]
            stable = (pA.argmax(1) == pB.argmax(1)).astype(float)
            reliable = (correct * stable)   # both correct and stable

            ent = _entropy(p_erm)
            ensv = _pairwise_mean([m["id_probs"] for m in ens], _js)
            frag = _pairwise_mean([m["id_probs"] for m in par], _sym_kl)
            # joint score: rank-based combination — accept only if low on
            # BOTH axes. We use max(rank_ent, rank_frag) ascending → low =
            # confident on both.
            r_ent = ent.argsort().argsort().astype(float)
            r_frag = frag.argsort().argsort().astype(float)
            joint_score = np.maximum(r_ent, r_frag)

            # Selective curves against two different definitions of success.
            cur_corr_e = selective_curve(ent,  correct,  coverages)
            cur_corr_v = selective_curve(ensv, correct,  coverages)
            cur_corr_f = selective_curve(frag, correct,  coverages)
            cur_corr_j = selective_curve(joint_score, correct, coverages)

            cur_rel_e = selective_curve(ent,  reliable, coverages)
            cur_rel_v = selective_curve(ensv, reliable, coverages)
            cur_rel_f = selective_curve(frag, reliable, coverages)
            cur_rel_j = selective_curve(joint_score, reliable, coverages)

            rows.append({
                "dataset": ds, "seed": seed,
                "frac_reliable": float(reliable.mean()),
                "aurc_correct_ent":      aurc(cur_corr_e),
                "aurc_correct_ens":      aurc(cur_corr_v),
                "aurc_correct_frag":     aurc(cur_corr_f),
                "aurc_correct_joint":    aurc(cur_corr_j),
                "aurc_reliable_ent":     aurc(cur_rel_e),
                "aurc_reliable_ens":     aurc(cur_rel_v),
                "aurc_reliable_frag":    aurc(cur_rel_f),
                "aurc_reliable_joint":   aurc(cur_rel_j),
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    agg = df.groupby("dataset").mean(numeric_only=True).reset_index()

    print("\n=== Selective AUC (lower=better) — target: CORRECT predictions ===")
    print(f"{'dataset':16s} {'entropy':>8s} {'ensemble':>9s} "
          f"{'fragility':>10s} {'joint':>8s}")
    for _, r in agg.iterrows():
        print(f"{r['dataset']:16s} {r['aurc_correct_ent']:>8.4f} "
              f"{r['aurc_correct_ens']:>9.4f} "
              f"{r['aurc_correct_frag']:>10.4f} "
              f"{r['aurc_correct_joint']:>8.4f}")

    print("\n=== Selective AUC — target: RELIABLE (correct AND stable) ===")
    print(f"{'dataset':16s} {'reliable_frac':>14s}  {'entropy':>8s} "
          f"{'ensemble':>9s} {'fragility':>10s} {'joint':>8s}  "
          f"{'Δ_joint_vs_ent':>14s}")
    for _, r in agg.iterrows():
        delta = (r["aurc_reliable_ent"] - r["aurc_reliable_joint"]) * 100
        print(f"{r['dataset']:16s} {r['frac_reliable']:>13.1%}   "
              f"{r['aurc_reliable_ent']:>8.4f} "
              f"{r['aurc_reliable_ens']:>9.4f} "
              f"{r['aurc_reliable_frag']:>10.4f} "
              f"{r['aurc_reliable_joint']:>8.4f}  "
              f"{delta:>+13.2f}pp")

    print(f"\nΔ_joint_vs_ent positive = joint signal beats entropy at "
          f"identifying reliable predictions.")
    print(f"Wrote rows to {args.out}")


if __name__ == "__main__":
    main()
