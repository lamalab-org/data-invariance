"""Paper Figure: BACE molecule-level case study.

Picks six BACE id-test molecules where ERM is fragile (high cross-bootstrap
flip rate) and twin-indep stabilises them (low cross-bootstrap flip rate),
renders the molecular structure of each via RDKit, and plots a per-molecule
prediction trace: a row of 20 coloured cells, ten under ERM (left) and ten
under twin-indep (right), one per retraining.

Single claim: stabilisation is per-molecule and visible at the chemistry
level, not just an aggregate statistic.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from _analysis_lib import load_runs
from data_molnet import _load_molnet_smiles_labels, _scaffold_split_with_id_holdout
from paper_constants import FROZEN_LAM


def _stack_predictions(runs):
    runs = sorted(runs, key=lambda x: x[0])
    return np.stack([d.get("id_probs_avg", d.get("id_probs")).argmax(1)
                     for _, d in runs])  # (S, N)


def _per_example_churn(preds: np.ndarray) -> np.ndarray:
    """Pairwise argmax-disagreement per test example, averaged across pairs."""
    S = preds.shape[0]
    k = preds.sum(axis=0)  # number of seeds predicting class 1
    return 2.0 * k * (S - k) / (S * (S - 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/cross_sample")
    ap.add_argument("--out", default="paper/figures/fig4_case_study.pdf")
    ap.add_argument("--n_mol", type=int, default=6)
    args = ap.parse_args()
    root = Path(args.root)

    erm_runs = load_runs(root / "bace", "erm_train*.npz")
    twin_runs = load_runs(root / "bace", f"twin_indep_train*_lam{FROZEN_LAM}.npz")
    erm_preds = _stack_predictions(erm_runs)
    twin_preds = _stack_predictions(twin_runs)
    erm_churn = _per_example_churn(erm_preds)
    twin_churn = _per_example_churn(twin_preds)

    # Reload BACE id-test SMILES under the same canonical seed used for
    # training so that index k of erm_preds matches `test_smiles[k]`.
    smiles_full, labels_full = _load_molnet_smiles_labels("bace", "./data/molnet")
    _, id_test_idx, _ = _scaffold_split_with_id_holdout(smiles_full, seed=99)
    test_smiles = [smiles_full[i] for i in id_test_idx]
    test_labels = labels_full[id_test_idx]
    assert len(test_smiles) == erm_preds.shape[1], \
        f"id-test mismatch: {len(test_smiles)} smiles vs {erm_preds.shape[1]} predictions"

    # Pick examples with high ERM churn and low twin churn.  We sort by
    # (ERM churn − twin churn), which prefers cases where twin-indep
    # contributes the most stabilisation; a small tie-breaker on raw ERM
    # churn ensures the top molecules are also genuinely fragile.
    score = (erm_churn - twin_churn) + 0.05 * erm_churn
    order = np.argsort(-score)
    candidates = [i for i in order
                  if erm_churn[i] >= 0.30 and twin_churn[i] < 0.20]
    if len(candidates) < args.n_mol:
        candidates = list(order[:args.n_mol])
    # De-duplicate visually similar molecules: simple choice — pick a
    # spread by stride through the candidate list.
    stride = max(1, len(candidates) // args.n_mol)
    picked = [candidates[i * stride] for i in range(args.n_mol)
              if i * stride < len(candidates)][:args.n_mol]

    print("Picked test molecules:")
    for k, idx in enumerate(picked):
        print(f"  {k+1}. idx={idx}  ERM churn={erm_churn[idx]*100:.0f}%  "
              f"twin churn={twin_churn[idx]*100:.0f}%  label={test_labels[idx]}  "
              f"smiles={test_smiles[idx]}")

    # Render structures + prediction traces.
    from rdkit import Chem
    from rdkit.Chem import Draw

    lama_aesthetics.get_style("main")
    plt.rcParams["axes.unicode_minus"] = False
    fig_w = lama_aesthetics.TWO_COL_WIDTH
    fig_h = fig_w * 0.55
    fig = plt.figure(figsize=(fig_w, fig_h))
    n_cols = 3
    n_rows = (args.n_mol + n_cols - 1) // n_cols
    gs_outer = fig.add_gridspec(n_rows, n_cols, hspace=0.55, wspace=0.18)
    cmap = ListedColormap(["#3a6ea5", "#d97757"])

    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import AllChem

    def _render_mol(smiles: str, canvas: int = 2200, bond_px: int = 56) -> np.ndarray | None:
        """Render a molecule at fixed bond-length so atoms have the same
        visual size across structures of different complexity.

        We draw on a generously sized canvas (large enough for the biggest
        molecule in the figure) and pin the bond length in pixels.  RDKit
        then leaves the molecule centred and at the same scale regardless
        of how many atoms it has; the surrounding white space differs but
        atom labels and bonds do not.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        AllChem.Compute2DCoords(mol)
        d = rdMolDraw2D.MolDraw2DCairo(canvas, canvas)
        opts = d.drawOptions()
        opts.fixedBondLength = bond_px        # px per bond, identical across molecules
        opts.bondLineWidth = 3.5
        opts.baseFontSize = 0.55
        opts.padding = 0.0                    # no per-molecule padding; canvas absorbs it
        opts.clearBackground = True
        opts.useBWAtomPalette()
        d.DrawMolecule(mol)
        d.FinishDrawing()
        png = d.GetDrawingText()
        from io import BytesIO
        from PIL import Image
        return np.asarray(Image.open(BytesIO(png)))

    for k, idx in enumerate(picked):
        r, c = k // n_cols, k % n_cols
        # Larger struct:trace height ratio gives the molecule more room
        # now that the title bar is gone.
        gs_inner = gs_outer[r, c].subgridspec(
            2, 1, height_ratios=[5, 0.5], hspace=0.04,
        )
        ax_struct = fig.add_subplot(gs_inner[0])
        ax_trace = fig.add_subplot(gs_inner[1])

        img = _render_mol(test_smiles[idx])
        if img is not None:
            ax_struct.imshow(img)
        ax_struct.axis("off")

        # Insert a 2-cell white gap between the ERM and Twin-indep blocks
        # so the two methods read as separate groups, not a single strip.
        gap = np.full(2, np.nan)
        trace = np.concatenate([
            erm_preds[:, idx].astype(float), gap, twin_preds[:, idx].astype(float),
        ])
        cmap_with_nan = ListedColormap(["#3a6ea5", "#d97757"]).copy()
        cmap_with_nan.set_bad(color="white")
        masked = np.ma.masked_invalid(trace.reshape(1, -1))
        ax_trace.imshow(masked, aspect="auto", cmap=cmap_with_nan,
                        vmin=0, vmax=1, interpolation="nearest")
        ax_trace.set_xticks([4.5, 16.5])
        ax_trace.set_xticklabels(
            [f"ERM {erm_churn[idx]*100:.0f}%",
             f"Twin-indep {twin_churn[idx]*100:.0f}%"],
            fontsize=7.5,
        )
        ax_trace.set_yticks([])
        for spine in ax_trace.spines.values():
            spine.set_visible(False)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#3a6ea5", ec="none", label="class 0"),
        plt.Rectangle((0, 0), 1, 1, fc="#d97757", ec="none", label="class 1"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False,
               handletextpad=0.4, columnspacing=1.5, fontsize=7.5)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=600)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
