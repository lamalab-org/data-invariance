"""Generate all paper figures from hardcoded experiment results.

Uses lama-aesthetics (https://github.com/lamalab-org/lama-aesthetics) for
publication-ready styling consistent with the lab's other papers.

Produces PDFs in figures/:
  - fig_correlation_sweep.pdf  — WGA vs spurious correlation per method
  - fig_swa_effect.pdf         — Free-sel vs SWA+Free on Waterbirds
  - fig_ingredient_ablation.pdf — ingredient ablation on Waterbirds & Battery

Usage:
    uv run python scripts/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np

# Apply lama-aesthetics publication style + register bundled font
lama_aesthetics.get_style("main")
lama_aesthetics.register_fonts()

# Override lama default bbox — use tight to avoid clipping labels
import matplotlib as mpl
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.pad_inches"] = 0.1

# Use the lama-aesthetics color cycle
COLORS = {
    "ERM":  "#0C5DA5",  # blue
    "JTT":  "#FF9500",  # orange
    "LfF":  "#00B945",  # green
    "Ours": "#FF2C00",  # red
}

MARKERS = {"ERM": "o", "JTT": "s", "LfF": "^", "Ours": "D"}

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


# ===================================================================
# Figure 1: Correlation sweep
# ===================================================================
def fig_correlation_sweep() -> None:
    correlations = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]

    # 3-seed mean SWA+Free. "Ours" uses 3-way auto-fallback (V-REx/JTT/ERM).
    data = {
        "ERM":  [0.720, 0.554, 0.002, 0.000, 0.000, 0.000],
        "JTT":  [0.232, 0.170, 0.020, 0.185, 0.571, 0.007],
        "LfF":  [0.263, 0.053, 0.000, 0.000, 0.000, 0.005],
        "Ours": [0.720, 0.554, 0.161, 0.365, 0.571, 0.007],
    }

    fig, ax = plt.subplots(figsize=(lama_aesthetics.ONE_COL_WIDTH, 2.2))

    for method, values in data.items():
        ax.plot(
            range(len(correlations)), values,
            marker=MARKERS[method],
            color=COLORS[method],
            label=method,
            linewidth=1.0,
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.4,
        )

    ax.set_xlabel("Spurious correlation")
    ax.set_ylabel("WGA (SWA+Free)")
    ax.set_xticks(range(len(correlations)))
    ax.set_xticklabels(["0.5", "0.7", "0.8", "0.9", "0.95", "0.99"])
    ax.set_ylim(-0.02, 0.82)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center")

    out = FIGURES_DIR / "fig_correlation_sweep.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ===================================================================
# Figure 2: SWA effect (grouped bar chart)
# ===================================================================
def fig_swa_effect() -> None:
    methods = ["ERM", "JTT", "LfF", "Ours"]
    free_sel     = [66.4, 76.3, 77.2, 77.8]
    free_sel_std = [ 9.6,  3.1,  8.2,  2.4]
    swa_free     = [74.1, 85.6, 86.2, 87.1]
    swa_free_std = [ 3.3,  1.3,  2.7,  1.2]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(lama_aesthetics.ONE_COL_WIDTH, 2.2))

    ax.bar(
        x - width / 2, free_sel, width,
        yerr=free_sel_std, capsize=2,
        label="Free-sel",
        color=[COLORS[m] for m in methods],
        alpha=0.35,
        edgecolor="none",
        error_kw={"linewidth": 0.6, "color": "#758D99"},
    )
    ax.bar(
        x + width / 2, swa_free, width,
        yerr=swa_free_std, capsize=2,
        label="SWA+Free",
        color=[COLORS[m] for m in methods],
        alpha=1.0,
        edgecolor="none",
        error_kw={"linewidth": 0.6, "color": "#758D99"},
    )

    ax.set_ylabel("WGA (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(50, 95)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title("Waterbirds", style="italic", fontsize=9)

    out = FIGURES_DIR / "fig_swa_effect.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ===================================================================
# Figure 3: Ingredient ablation (Waterbirds + Battery side by side)
# ===================================================================
def fig_ingredient_ablation() -> None:
    conditions = ["ERM", "+upweight", "+VREx", "Full"]

    waterbirds     = [77.0, 81.6, 81.4, 84.6]
    waterbirds_std = [ 1.9,  3.5,  1.9,  1.9]
    battery        = [35.8, 54.0, 35.3, 65.3]
    battery_std    = [ 1.4, 15.5,  1.9, 10.5]

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(lama_aesthetics.ONE_COL_WIDTH, 2.2))

    ax.bar(
        x - width / 2, waterbirds, width,
        yerr=waterbirds_std, capsize=2,
        label="Waterbirds",
        color="#0C5DA5",
        edgecolor="none",
        error_kw={"linewidth": 0.6, "color": "#758D99"},
    )
    ax.bar(
        x + width / 2, battery, width,
        yerr=battery_std, capsize=2,
        label="Battery",
        color="#FF9500",
        edgecolor="none",
        error_kw={"linewidth": 0.6, "color": "#758D99"},
    )

    ax.set_ylabel("WGA (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=7)

    out = FIGURES_DIR / "fig_ingredient_ablation.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating figures...")
    fig_correlation_sweep()
    fig_swa_effect()
    fig_ingredient_ablation()
    print("Done.")


if __name__ == "__main__":
    main()
