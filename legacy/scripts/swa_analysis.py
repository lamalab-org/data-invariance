"""SWA contribution analysis: how much does SWA help each method under each selection protocol?

Parses experiment logs and produces a table showing all 4 selection protocols
for every method/dataset combination. This is the key table for the paper.

Usage:
    uv run python scripts/swa_analysis.py logs/final5_*.log logs/final3_*.log
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict

import numpy as np


def parse_log(path: str) -> tuple[str | None, dict]:
    """Parse a run_experiment log file."""
    results = defaultdict(lambda: {"wga_sel": [], "swa_wga": [], "free_sel": [], "swa_free": []})
    dataset = None

    with open(path) as f:
        for line in f:
            m = re.match(r"dataset=(\S+)", line)
            if m:
                dataset = m.group(1)

            m = re.match(
                r"\s+(\w+)\s+:\s+WGA-sel=([\d.]+)\(ep\d+\)\s+SWA-groups=([\d.]+)\s+"
                r"Free-sel=([\d.]+)\(ep\d+\)\s+SWA-free=([\d.]+)",
                line,
            )
            if m:
                method = m.group(1)
                results[method]["wga_sel"].append(float(m.group(2)))
                results[method]["swa_wga"].append(float(m.group(3)))
                results[method]["free_sel"].append(float(m.group(4)))
                results[method]["swa_free"].append(float(m.group(5)))

    return dataset, dict(results)


def fmt(vals: list[float]) -> str:
    if not vals:
        return "    —    "
    return f"{np.mean(vals):5.1%}±{np.std(vals):4.1%}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/swa_analysis.py logs/final5_*.log")
        sys.exit(1)

    all_data = {}
    for path in sorted(sys.argv[1:]):
        dataset, results = parse_log(path)
        if dataset and results:
            all_data[dataset] = results

    methods = ["erm", "jtt", "lff", "ours"]

    # Table 1: Full comparison (all 4 protocols)
    print("=" * 100)
    print("Table: All selection protocols (WGA = uses group labels, Free = no group labels)")
    print("=" * 100)
    print(f"{'Dataset':20s} {'Method':6s} {'WGA-sel':>12s} {'SWA+WGA':>12s} "
          f"{'Free-sel':>12s} {'SWA+Free':>12s} {'SWA boost':>10s}")
    print("-" * 100)

    for dataset in all_data:
        for method in methods:
            if method not in all_data[dataset]:
                continue
            r = all_data[dataset][method]
            free_mean = np.mean(r["free_sel"]) if r["free_sel"] else 0
            swa_free_mean = np.mean(r["swa_free"]) if r["swa_free"] else 0
            boost = swa_free_mean - free_mean

            print(f"{dataset:20s} {method:6s} {fmt(r['wga_sel']):>12s} {fmt(r['swa_wga']):>12s} "
                  f"{fmt(r['free_sel']):>12s} {fmt(r['swa_free']):>12s} {boost:+9.1%}")
        print()

    # Table 2: Average SWA boost per method
    print("=" * 70)
    print("Average SWA boost on group-free selection (Free-sel → SWA+Free)")
    print("=" * 70)
    for method in methods:
        boosts = []
        for dataset in all_data:
            if method in all_data[dataset]:
                r = all_data[dataset][method]
                if r["free_sel"] and r["swa_free"]:
                    boosts.append(np.mean(r["swa_free"]) - np.mean(r["free_sel"]))
        if boosts:
            print(f"  {method:6s}: {np.mean(boosts):+.1%} ± {np.std(boosts):.1%} "
                  f"(across {len(boosts)} datasets)")

    # Table 3: Group-free gap (WGA-sel → Free-sel, without SWA)
    print()
    print("=" * 70)
    print("Group-label gap WITHOUT SWA (WGA-sel − Free-sel)")
    print("=" * 70)
    for method in methods:
        gaps = []
        for dataset in all_data:
            if method in all_data[dataset]:
                r = all_data[dataset][method]
                if r["wga_sel"] and r["free_sel"]:
                    gaps.append(np.mean(r["wga_sel"]) - np.mean(r["free_sel"]))
        if gaps:
            print(f"  {method:6s}: {np.mean(gaps):+.1%} ± {np.std(gaps):.1%}")

    # Table 4: Group-free gap WITH SWA (SWA+WGA → SWA+Free)
    print()
    print("=" * 70)
    print("Group-label gap WITH SWA (SWA+WGA − SWA+Free)")
    print("=" * 70)
    for method in methods:
        gaps = []
        for dataset in all_data:
            if method in all_data[dataset]:
                r = all_data[dataset][method]
                if r["swa_wga"] and r["swa_free"]:
                    gaps.append(np.mean(r["swa_wga"]) - np.mean(r["swa_free"]))
        if gaps:
            print(f"  {method:6s}: {np.mean(gaps):+.1%} ± {np.std(gaps):.1%}")


if __name__ == "__main__":
    main()
