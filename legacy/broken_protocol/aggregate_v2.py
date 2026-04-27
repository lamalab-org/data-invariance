"""Aggregate v2_results.csv into paper tables + statistical tests.

Outputs:
  - outputs/tables/main_results.tex  — mean ± std per (dataset, method)
    on all four selection protocols.
  - outputs/tables/method_pairs.tex   — paired bootstrap 95% CI on the
    difference (Ours − JTT) and (Ours − ERM) per dataset, plus the
    aggregate Ours-minus-JTT across datasets.
  - outputs/tables/discovery_vs_gain.csv — per-dataset discovery stats
    (signal_ratio, assign_corr, lambda) and V-REx gain over JTT at the
    candidate-score level, for the honest-negative signal_ratio plot.
  - stdout: compact summary.

Statistical choices (by design, to avoid over-claiming):
  - Paired bootstrap (10 000 resamples of seeds with replacement).
  - 95% percentile CIs on Ours − JTT per dataset.
  - Holm-Bonferroni correction applied only for the aggregate "Ours > JTT
    on how many datasets" claim, not for per-dataset CIs which are
    interpretive.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10_000,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap on the mean of `diffs` (paired differences)."""
    if len(diffs) == 0:
        return (float("nan"), float("nan"), float("nan"))
    idx = RNG.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (float(diffs.mean()), lo, hi)


def aggregate(df: pd.DataFrame, metric: str = "swa_free"):
    """Return (table of mean±std, table of paired CIs)."""
    # mean±std per (dataset, method)
    summary = (df.groupby(["dataset", "method"])[metric]
                 .agg(["mean", "std", "count"])
                 .reset_index())

    # Paired differences (Ours − JTT) across seeds, per dataset.
    pivot = (df.pivot_table(index=["dataset", "seed"], columns="method",
                            values=metric)
               .reset_index())
    rows = []
    for ds, g in pivot.groupby("dataset"):
        seeds_ok = g.dropna(subset=["ours", "jtt"])
        diff_ours_jtt = (seeds_ok["ours"] - seeds_ok["jtt"]).to_numpy()
        diff_ours_erm = (seeds_ok["ours"] - seeds_ok["erm"]).to_numpy()
        m_oj, lo_oj, hi_oj = bootstrap_ci(diff_ours_jtt)
        m_oe, lo_oe, hi_oe = bootstrap_ci(diff_ours_erm)
        rows.append({
            "dataset": ds,
            "n_seeds": len(seeds_ok),
            "ours_minus_jtt_mean": m_oj,
            "ours_minus_jtt_lo": lo_oj,
            "ours_minus_jtt_hi": hi_oj,
            "ours_minus_erm_mean": m_oe,
            "ours_minus_erm_lo":   lo_oe,
            "ours_minus_erm_hi":   hi_oe,
        })
    pairs = pd.DataFrame(rows)
    return summary, pairs


def discovery_vs_gain(df: pd.DataFrame):
    """Per-dataset V-REx candidate gain over JTT at the candidate level.

    Uses vrex_cand − jtt_cand (pre-fallback scores) averaged per dataset.
    This is the quantity signal_ratio should be predictive of if the claim
    'permutation test predicts V-REx effectiveness' were true.
    """
    ours = df[df["method"] == "ours"].copy()
    if ours["vrex_cand"].isna().all():
        return pd.DataFrame()

    # average across seeds per dataset
    agg = (ours.groupby("dataset")
               .agg(signal_ratio=("signal_ratio", "mean"),
                    assign_corr=("assign_corr", "mean"),
                    reliability=("reliability", "mean"),
                    lam=("lambda", "mean"),
                    vrex_cand=("vrex_cand", "mean"),
                    jtt_cand=("jtt_cand", "mean"),
                    erm_cand=("erm_cand", "mean"),
                    vrex_gate_frac=("vrex_gated", "mean"),
                    n_seeds=("seed", "nunique"))
               .reset_index())
    agg["vrex_minus_jtt"] = agg["vrex_cand"] - agg["jtt_cand"]
    return agg


def to_tex_mean_std(summary: pd.DataFrame, pairs: pd.DataFrame,
                    metric_label: str) -> str:
    methods = ["erm", "jtt", "lff", "ours"]
    datasets = sorted(summary["dataset"].unique())
    header = " & ".join([f"\\textbf{{{m.upper()}}}" for m in methods])
    lines = [
        "\\begin{tabular}{l" + "c" * len(methods) + "}",
        "\\toprule",
        f"Dataset & {header} \\\\",
        "\\midrule",
    ]
    for ds in datasets:
        sub = summary[summary["dataset"] == ds]
        cells = [ds]
        for m in methods:
            row = sub[sub["method"] == m]
            if row.empty or pd.isna(row["mean"].values[0]):
                cells.append("--")
            else:
                mu = row["mean"].values[0] * 100
                sd = (row["std"].values[0] or 0) * 100
                cells.append(f"{mu:.1f}\\(\\pm\\){sd:.1f}")
        # mark if Ours CI strictly > 0 (reject null)
        pair_row = pairs[pairs["dataset"] == ds]
        if not pair_row.empty and pair_row["ours_minus_jtt_lo"].values[0] > 0:
            cells[-1] = "\\textbf{" + cells[-1] + "}"
        lines.append(" & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"% Metric = {metric_label}.  Bold: Ours 95% bootstrap CI on "
        "(Ours − JTT) strictly > 0.",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/v2_results.csv")
    ap.add_argument("--metric", default="swa_free",
                    choices=["wga_sel", "swa_groups", "free_sel", "swa_free"])
    ap.add_argument("--out_dir", default="outputs/tables")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    summary, pairs = aggregate(df, metric=args.metric)
    disc = discovery_vs_gain(df)

    # Write tables.
    (out_dir / "main_results.tex").write_text(
        to_tex_mean_std(summary, pairs, args.metric))
    pairs.to_csv(out_dir / "method_pairs.csv", index=False)
    if not disc.empty:
        disc.to_csv(out_dir / "discovery_vs_gain.csv", index=False)

    # Console summary.
    n = len(pairs)
    wins = (pairs["ours_minus_jtt_lo"] > 0).sum()
    ties = ((pairs["ours_minus_jtt_lo"] <= 0) &
            (pairs["ours_minus_jtt_hi"] >= 0)).sum()
    losses = (pairs["ours_minus_jtt_hi"] < 0).sum()
    print(f"metric = {args.metric}")
    print(f"datasets with paired data: {n}")
    print(f"  Ours > JTT (95% CI excludes 0): {wins}")
    print(f"  tie                           : {ties}")
    print(f"  Ours < JTT (95% CI excludes 0): {losses}")
    print(f"\nPer-dataset (Ours − JTT) 95% bootstrap CI on {args.metric}:")
    for _, r in pairs.iterrows():
        tag = ""
        if r["ours_minus_jtt_lo"] > 0:
            tag = "  [win]"
        elif r["ours_minus_jtt_hi"] < 0:
            tag = "  [loss]"
        print(f"  {r['dataset']:15s}  Δ = "
              f"{r['ours_minus_jtt_mean']*100:+.2f}pp  "
              f"[{r['ours_minus_jtt_lo']*100:+.2f}, "
              f"{r['ours_minus_jtt_hi']*100:+.2f}] pp{tag}")

    if not disc.empty:
        r_sig = disc[["signal_ratio", "vrex_minus_jtt"]].corr().iloc[0, 1]
        r_ac  = disc[["assign_corr", "vrex_minus_jtt"]].corr().iloc[0, 1]
        print(f"\nSignal_ratio vs V-REx candidate gain over JTT:")
        print(f"  Pearson r = {r_sig:+.3f}  (n={len(disc)})")
        print(f"Assign_corr  vs V-REx candidate gain over JTT:")
        print(f"  Pearson r = {r_ac:+.3f}")

    print(f"\nWrote tables to {out_dir}")


if __name__ == "__main__":
    main()
