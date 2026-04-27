"""Paper Figure 0: cross-sample fragility, before and after twin-indep.

The single most important figure in the paper.  Visualises the phenomenon
(per-example argmax flips across retrainings on independent bootstraps)
and the intervention (twin-indep removes most of the flips), in one image.

Layout
------
Two side-by-side heatmaps (shared y-axis), on BACE (the dev dataset that
also has the largest fragility magnitude):

  rows    = a sample of test molecules (sorted by ERM flip count, top
            = most fragile; only molecules with at least one ERM flip)
  columns = 10 independent retrainings (train_seed=1..10)
  cells   = predicted class (binary), coloured

The ERM panel shows visible vertical stripes wherever the class flips
across retrainings.  The twin-indep panel shows almost-uniform rows --
predictions are stable across the same 10 retrainings.

Caption first sentence (the conclusion):
  Cross-sample prediction fragility on BACE: the ERM model flips the
  class assigned to a substantial fraction of test molecules across
  independent retrainings (left); twin-indep with bootstrap consistency
  removes most of these flips at matched compute (right).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from _analysis_lib import load_runs
from paper_constants import FROZEN_LAM


def _stack_predictions(runs):
    """Return (n_seeds, n_test) array of argmax predictions, with seeds sorted."""
    runs = sorted(runs, key=lambda x: x[0])
    preds = []
    for _, d in runs:
        probs = d.get("id_probs_avg", d.get("id_probs"))
        preds.append(probs.argmax(1))
    return np.stack(preds)  # (S, N)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/cross_sample")
    ap.add_argument("--dataset", default="bace")
    ap.add_argument("--out", default="paper/figures/fig0_overview.pdf")
    ap.add_argument("--n_examples", type=int, default=80,
                    help="Number of test molecules to display (top-N by ERM flip count).")
    args = ap.parse_args()

    root = Path(args.root)
    erm_runs = load_runs(root / args.dataset, "erm_train*.npz")
    twin_runs = load_runs(root / args.dataset,
                          f"twin_indep_train*_lam{FROZEN_LAM}.npz")
    assert erm_runs and twin_runs, "Need ERM and twin_indep runs."

    erm_preds = _stack_predictions(erm_runs)         # (S, N)
    twin_preds = _stack_predictions(twin_runs)       # (S, N)

    # Per-test-example argmax churn = average pairwise disagreement across the
    # C(S, 2) seed pairs.  For binary classification with k seeds predicting
    # class 1 and (S - k) predicting class 0, this is 2 k (S - k) / (S (S - 1)).
    # The mean across test examples equals the paper's headline argmax churn.
    def _per_example_churn(preds: np.ndarray) -> np.ndarray:
        S = preds.shape[0]
        k = preds.sum(axis=0)  # number of seeds predicting class 1
        return 2.0 * k * (S - k) / (S * (S - 1))

    erm_churn = _per_example_churn(erm_preds)
    twin_churn = _per_example_churn(twin_preds)

    # Sort molecules by ERM churn (most-fragile first), keep top-N.
    order = np.argsort(-erm_churn)
    keep = order[:args.n_examples]
    erm_panel = erm_preds[:, keep].T   # (N_examples, S)
    twin_panel = twin_preds[:, keep].T

    # Visual.  Wider-than-tall panels: 30 most-fragile rows × 10 retrainings,
    # rendered with a small horizontal stretch so each cell is rectangular and
    # the vertical-stripe pattern in the ERM panel is unmistakable.
    lama_aesthetics.get_style("main")
    plt.rcParams["axes.unicode_minus"] = False

    fig_w = lama_aesthetics.TWO_COL_WIDTH
    fig_h = fig_w * 0.32  # squat, banner-shaped
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(fig_w, fig_h),
                                     sharey=True,
                                     gridspec_kw=dict(wspace=0.06))
    cmap = ListedColormap(["#3a6ea5", "#d97757"])  # cool / warm; readable in print

    erm_overall = float(erm_churn.mean()) * 100
    twin_overall = float(twin_churn.mean()) * 100
    erm_top = float(erm_churn[keep].mean()) * 100
    twin_top = float(twin_churn[keep].mean()) * 100

    panels = [
        (ax_l, erm_panel,
         f"ERM  —  {erm_overall:.1f}% of test predictions flip across retrainings"),
        (ax_r, twin_panel,
         f"Twin-indep ($\\lambda={int(FROZEN_LAM)}$)  —  {twin_overall:.1f}% flip"),
    ]
    for ax, panel, title in panels:
        ax.imshow(panel, aspect="auto", cmap=cmap, interpolation="nearest",
                  vmin=0, vmax=1)
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlabel("retraining (train_seed 1–10)", fontsize=8)
        ax.set_xticks(np.arange(panel.shape[1]))
        ax.set_xticklabels([str(i + 1) for i in range(panel.shape[1])],
                           fontsize=7)
        ax.tick_params(left=False, right=False, length=2)

    ax_l.set_ylabel(f"Top {args.n_examples} test molecules\n(by ERM argmax churn)",
                    fontsize=8)
    ax_l.set_yticks([])

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#3a6ea5", ec="none", label="class 0"),
        plt.Rectangle((0, 0), 1, 1, fc="#d97757", ec="none", label="class 1"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False,
               handletextpad=0.4, columnspacing=1.5, fontsize=8)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}")
    print(f"  ERM   overall flip rate: {erm_overall:.2f}%")
    print(f"  Twin  overall flip rate: {twin_overall:.2f}%")
    print(f"  ERM   top-{args.n_examples} flip rate: {erm_top:.2f}%")
    print(f"  Twin  top-{args.n_examples} flip rate: {twin_top:.2f}%")


if __name__ == "__main__":
    main()
