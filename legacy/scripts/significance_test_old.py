"""Statistical significance analysis: Ours vs JTT and Ours vs LfF across all datasets."""

import re
import numpy as np
from scipy import stats
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"

# Map of dataset name -> log file path
DATASETS = {
    "cmnist": LOG_DIR / "final5_cmnist.log",
    "multi_cmnist": LOG_DIR / "final5_multi_cmnist.log",
    "cont_cmnist": LOG_DIR / "final5_continuous_cmnist.log",
    "waterbirds": LOG_DIR / "final3_waterbirds.log",
    "bace": LOG_DIR / "final5_bace.log",
    "bbbp": LOG_DIR / "final5_bbbp.log",
    "battery": LOG_DIR / "final5_battery.log",
    "hiv": LOG_DIR / "final5_hiv.log",
    "mof_solvent": LOG_DIR / "final5_mof_solvent.log",
    "mof_thermal": LOG_DIR / "final5_mof_thermal.log",
    "perovskite": LOG_DIR / "final5_perovskite.log",
    "tadf": LOG_DIR / "final5_tadf.log",
}

# Regex to parse result lines
# e.g. "  erm  : WGA-sel=0.7944(ep6)  SWA-groups=0.7336  Free-sel=0.6838(ep10)  SWA-free=0.7523  [194s]"
RESULT_RE = re.compile(
    r"^\s+(erm|jtt|lff|ours)\s*:\s+"
    r"WGA-sel=[\d.]+\(ep\d+\)\s+"
    r"SWA-groups=[\d.]+\s+"
    r"Free-sel=[\d.]+\(ep\d+\)\s+"
    r"SWA-free=([\d.]+)"
)

SEED_RE = re.compile(r"^--- SEED (\d+) ---")


def parse_log(path: Path) -> dict[str, list[float]]:
    """Parse a log file and return {method: [swa_free_per_seed]}."""
    results: dict[str, list[float]] = {}
    with open(path) as f:
        for line in f:
            m = RESULT_RE.match(line)
            if m:
                method = m.group(1)
                swa_free = float(m.group(2))
                results.setdefault(method, []).append(swa_free)
    return results


def main():
    # Collect per-dataset results
    all_data: dict[str, dict[str, list[float]]] = {}
    for name, path in DATASETS.items():
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {name}")
            continue
        parsed = parse_log(path)
        if "ours" not in parsed or "jtt" not in parsed:
            print(f"WARNING: {name} missing ours or jtt, skipping")
            continue
        all_data[name] = parsed

    # Print per-dataset table
    print("=" * 100)
    print("PER-DATASET COMPARISON (SWA+Free metric)")
    print("=" * 100)
    header = f"{'Dataset':<16} {'Ours':>16} {'JTT':>16} {'LfF':>16} {'Ours-JTT':>10} {'p(t-test)':>10} {'Ours-LfF':>10} {'p(t-test)':>10}"
    print(header)
    print("-" * 100)

    # For Wilcoxon: collect per-dataset mean differences
    ours_jtt_diffs = []
    ours_lff_diffs = []

    for name in sorted(all_data.keys()):
        d = all_data[name]
        ours = np.array(d["ours"])
        jtt = np.array(d["jtt"])
        lff = np.array(d.get("lff", [np.nan] * len(ours)))

        n_seeds = min(len(ours), len(jtt))
        ours_s = ours[:n_seeds]
        jtt_s = jtt[:n_seeds]
        lff_s = lff[:n_seeds] if len(lff) >= n_seeds else np.full(n_seeds, np.nan)

        # Paired t-test: Ours vs JTT
        diff_jtt = ours_s - jtt_s
        if np.std(diff_jtt) > 0:
            t_jtt, p_jtt = stats.ttest_rel(ours_s, jtt_s)
        else:
            p_jtt = 1.0 if np.mean(diff_jtt) == 0 else 0.0

        # Paired t-test: Ours vs LfF
        diff_lff = ours_s - lff_s
        if not np.any(np.isnan(lff_s)) and np.std(diff_lff) > 0:
            t_lff, p_lff = stats.ttest_rel(ours_s, lff_s)
        else:
            p_lff = float("nan")

        mean_diff_jtt = np.mean(diff_jtt)
        mean_diff_lff = np.mean(diff_lff) if not np.any(np.isnan(diff_lff)) else float("nan")

        ours_jtt_diffs.append(mean_diff_jtt)
        if not np.isnan(mean_diff_lff):
            ours_lff_diffs.append(mean_diff_lff)

        ours_str = f"{np.mean(ours_s):.4f}+-{np.std(ours_s):.4f}"
        jtt_str = f"{np.mean(jtt_s):.4f}+-{np.std(jtt_s):.4f}"
        lff_str = f"{np.mean(lff_s):.4f}+-{np.std(lff_s):.4f}" if not np.any(np.isnan(lff_s)) else "N/A"

        p_jtt_str = f"{p_jtt:.4f}" if not np.isnan(p_jtt) else "N/A"
        p_lff_str = f"{p_lff:.4f}" if not np.isnan(p_lff) else "N/A"

        print(f"{name:<16} {ours_str:>16} {jtt_str:>16} {lff_str:>16} {mean_diff_jtt:>+10.4f} {p_jtt_str:>10} {mean_diff_lff:>+10.4f} {p_lff_str:>10}")

    # Wilcoxon signed-rank tests across datasets
    print()
    print("=" * 100)
    print("AGGREGATE TESTS (Wilcoxon signed-rank on per-dataset mean differences)")
    print("=" * 100)

    ours_jtt_arr = np.array(ours_jtt_diffs)
    print(f"\nOurs vs JTT: {len(ours_jtt_arr)} datasets")
    print(f"  Mean difference: {np.mean(ours_jtt_arr):+.4f}")
    print(f"  Datasets where Ours > JTT: {np.sum(ours_jtt_arr > 0)}/{len(ours_jtt_arr)}")
    if len(ours_jtt_arr) >= 6:
        stat, p = stats.wilcoxon(ours_jtt_arr, alternative="two-sided")
        print(f"  Wilcoxon stat = {stat:.1f}, p = {p:.6f}")
        stat1, p1 = stats.wilcoxon(ours_jtt_arr, alternative="greater")
        print(f"  Wilcoxon (one-sided, Ours > JTT): stat = {stat1:.1f}, p = {p1:.6f}")
    else:
        print("  Too few datasets for Wilcoxon test (need >= 6)")

    ours_lff_arr = np.array(ours_lff_diffs)
    print(f"\nOurs vs LfF: {len(ours_lff_arr)} datasets")
    print(f"  Mean difference: {np.mean(ours_lff_arr):+.4f}")
    print(f"  Datasets where Ours > LfF: {np.sum(ours_lff_arr > 0)}/{len(ours_lff_arr)}")
    if len(ours_lff_arr) >= 6:
        stat, p = stats.wilcoxon(ours_lff_arr, alternative="two-sided")
        print(f"  Wilcoxon stat = {stat:.1f}, p = {p:.6f}")
        stat1, p1 = stats.wilcoxon(ours_lff_arr, alternative="greater")
        print(f"  Wilcoxon (one-sided, Ours > LfF): stat = {stat1:.1f}, p = {p1:.6f}")
    else:
        print("  Too few datasets for Wilcoxon test (need >= 6)")

    # Also print a compact sign table
    print()
    print("=" * 100)
    print("SIGN TABLE")
    print("=" * 100)
    print(f"{'Dataset':<16} {'Ours>JTT?':>10} {'Ours>LfF?':>10}")
    print("-" * 40)
    for i, name in enumerate(sorted(all_data.keys())):
        jtt_sign = "+" if ours_jtt_diffs[i] > 0 else ("-" if ours_jtt_diffs[i] < 0 else "=")
        lff_sign = "+" if i < len(ours_lff_diffs) and ours_lff_diffs[i] > 0 else ("-" if i < len(ours_lff_diffs) and ours_lff_diffs[i] < 0 else "=")
        print(f"{name:<16} {jtt_sign:>10} {lff_sign:>10}")


if __name__ == "__main__":
    main()
