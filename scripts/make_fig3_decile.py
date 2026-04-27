"""Paper Figure 3: top-decile fragility predicts which test predictions churn.

For every test example, compute (i) average cross-bootstrap fragility
(mean sym-KL across all pairs of ERM bootstraps that disagree on the
example) and (ii) average flip rate (fraction of bootstrap pairs where
the argmax predictions disagree).  Rank test examples by their fragility
and split into deciles; report the mean flip rate per decile.

Single claim: fragility identifies which predictions will flip on
retraining.  In every dataset the top-10% most fragile examples flip
at a high rate; the bottom 10% rarely flip.

Practitioner take: a single additional bootstrap suffices to flag the
fragile decile for re-review, without a full retraining cascade.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import lama_aesthetics

from _analysis_lib import load_runs, sym_kl
from paper_constants import DEV_DATASET, HEADLINE_DATASETS, display


def _per_example_fragility_and_flips(ds_dir: Path):
    """Return (fragility[N_test], flip_rate[N_test]) over all ERM seed pairs."""
    runs = load_runs(ds_dir, "erm_train*.npz")
    if len(runs) < 2:
        return None
    probs = [d.get("id_probs_avg", d.get("id_probs")) for _, d in runs]
    n_test = probs[0].shape[0]
    frag_acc = np.zeros(n_test)
    flip_acc = np.zeros(n_test)
    n_pairs = 0
    for i, j in combinations(range(len(probs)), 2):
        frag_acc += sym_kl(probs[i], probs[j])
        flip_acc += (probs[i].argmax(1) != probs[j].argmax(1)).astype(float)
        n_pairs += 1
    return frag_acc / n_pairs, flip_acc / n_pairs


def _decile_flip_rates(fragility, flips, n_bins=10):
    """Bin examples by fragility decile; return mean flip rate per bin."""
    order = np.argsort(fragility)
    n = len(fragility)
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    return np.array([flips[order[edges[k]:edges[k + 1]]].mean()
                     for k in range(n_bins)])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/cross_sample")
    ap.add_argument("--out", default="paper/figures/fig3_decile.pdf")
    args = ap.parse_args()
    root = Path(args.root)

    # Order: dev dataset first, then headline by N (smallest to largest).
    datasets = [DEV_DATASET] + HEADLINE_DATASETS
    rates = {}
    for ds in datasets:
        out = _per_example_fragility_and_flips(root / ds)
        if out is None:
            continue
        rates[ds] = _decile_flip_rates(*out)

    lama_aesthetics.get_style("main")
    plt.rcParams["axes.unicode_minus"] = False
    fig_w = lama_aesthetics.TWO_COL_WIDTH
    fig_h = fig_w * 0.32
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    n_bins = 10
    n_ds = len(rates)
    bar_w = 0.8 / n_ds
    bin_centres = np.arange(n_bins)

    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, n_ds))
    for di, (ds, decile_rates) in enumerate(rates.items()):
        offset = (di - (n_ds - 1) / 2) * bar_w
        ax.bar(bin_centres + offset, decile_rates * 100, bar_w,
               color=cmap[di], label=display(ds), edgecolor="none")

    ax.set_xticks(bin_centres)
    ax.set_xticklabels([str(k + 1) for k in range(n_bins)])
    ax.set_xlabel("Fragility decile (1 = least fragile, 10 = most)")
    ax.set_ylabel("Argmax-flip rate per pair (%)")
    ax.legend(ncol=min(4, n_ds), frameon=False, fontsize=7,
              loc="upper center", bbox_to_anchor=(0.5, -0.20),
              handletextpad=0.4, columnspacing=1.0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}")
    print(f"\nTop-decile / bottom-decile flip rates (%):")
    print(f"{'dataset':16s}  {'bottom (1)':>12s}  {'top (10)':>10s}  {'ratio':>8s}")
    for ds, dr in rates.items():
        ratio = dr[-1] / dr[0] if dr[0] > 0 else float("inf")
        print(f"{ds:16s}  {dr[0]*100:>12.2f}  {dr[-1]*100:>10.2f}  {ratio:>8.1f}x")


if __name__ == "__main__":
    main()
