"""Emit paper/sections/macros.tex with one \\newcommand per quoted number.

Reads every CSV produced by the analysis pipeline and emits a single
LaTeX file of \\newcommand definitions.  The paper's prose imports this
file once at the top and references the macros instead of literal
numbers.  Result: prose can never drift from the source CSVs --- a
retraining run regenerates the CSVs, this script regenerates the
macros, and the prose follows.

Robustness
----------
If a source CSV is missing, the corresponding macros are emitted as
``??`` so the document still compiles.  After full retraining every
source CSV exists and every macro is real.

Naming convention
-----------------
Macros are camelCase and group-prefixed:
  \\churn{Min,Max,...}              fragility_magnitudes.csv
  \\bagFive{Low,High,...}           main_table.csv (Bagging K=5 row aggregate)
  \\twin{MedianReduction,Low,High}  main_table.csv (Twin row aggregate)
  \\paramSide{Low,High}             main_table.csv (combined param-side range)
  \\symKL{BagFold,TwinFold,...}     distributional.csv
  \\entropyTop{Low,High}            entropy_vs_fragility.csv
  \\fragTop{Low,High}               entropy_vs_fragility.csv
  \\triage{KtenLow,KtenHigh,...}    convergence_recall.csv
  \\reg{TwinLow,TwinHigh,...}       regression.csv
  \\waterbirds{...}                 waterbirds_lambda.csv
  \\chemberta{...}                  chemberta_heldout.csv
  \\gin{...}                        gin_lambda.csv, bace_gin.csv
  \\friedman{Chi,P,Rank...}         friedman.csv
  \\twinWins{...}, \\bagFiveWins... per-dataset paired CIs from raw NPZs
  \\nDatasets{Headline,Heldout}     paper_constants.py
"""
from __future__ import annotations

import csv
from pathlib import Path
from statistics import median
from typing import Any

from paper_constants import (
    DEV_DATASET,
    FROZEN_LAM,
    HEADLINE_DATASETS,
    display,
)


OUTPUTS = Path("outputs")
OUT_PATH = Path("paper/sections/macros.tex")
RUN_ROOT = OUTPUTS / "cross_sample"

# (macro_name, value-or-string).  Values run through `_format_number`
# which picks integer / pp / pct format from a per-macro hint.
MACROS: list[tuple[str, str]] = []


def add(name: str, value: str | float | int, fmt: str = "raw") -> None:
    """Append a macro to the output list with format hint.

    fmt:
      raw   - string as-is (use for ranges built by caller, headline
              numbers like \\nDatasetsHeadline=9)
      pct1  - one decimal percent: 12.3
      pct0  - integer percent: 12
      pp1   - one-decimal pp: +1.6 with sign
      ratio - one-decimal multiplier: 7.8
      sci   - scientific: 3.1e-7
    """
    if isinstance(value, str):
        rendered = value
    elif fmt == "raw":
        rendered = str(value)
    elif fmt == "pct1":
        rendered = f"{value:.1f}"
    elif fmt == "pct0":
        rendered = f"{value:.0f}"
    elif fmt == "pp1":
        rendered = f"{value:+.1f}"
    elif fmt == "ratio":
        rendered = f"{value:.1f}"
    elif fmt == "sci":
        rendered = f"{value:.1g}"
    else:
        rendered = str(value)
    MACROS.append((name, rendered))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"  [skip] {path} does not exist")
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _f(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------

def emit_dataset_counts() -> None:
    add("nDatasetsHeadline", str(1 + len(HEADLINE_DATASETS)))   # dev + held-out
    add("nDatasetsHeldout", str(len(HEADLINE_DATASETS)))
    add("overlapPctMid", "40")
    add("overlapPctLo", "0")
    add("overlapPctHi", "100")
    # Pre-registered protocol constants.  Single source of truth here so
    # the prose's quoted thresholds, fold counts, and review fractions
    # can never drift from the protocol description.
    add("preregFilterPP", "5")          # ERM > majority + 5pp filter
    add("preregMinTestSize", "60")      # min canonical id-test size
    add("preregTolerance", "0.02")      # lambda-selection accuracy tolerance
    add("nSeeds", "10")                 # train_seeds per cell
    add("nSeedPairs", "45")             # binom(10, 2)
    add("canonicalSeed", "99")
    add("reviewFraction", "30")         # top-X% review fraction in triage
    add("topDecilePct", "10")           # top-decile triage cutoff (figure 3)
    # Accuracy-degradation flag in the overlap-spectrum table (daggers).
    # Threshold is set in scripts/make_fig5_overlap.py:93 (drop > 0.05);
    # if you change one, change the other.
    add("accDegradeFlagPP", "5")
    # Canonical-seed sensitivity sweep: number of canonical seeds the
    # main-table protocol was replicated on (the original 99 plus 7
    # and 42).  See scripts/run_seed_sweep.sh.
    add("nCanonicalSeeds", "3")


def emit_magnitudes() -> None:
    rows = [r for r in _read_csv(OUTPUTS / "fragility_magnitudes.csv")
            if r.get("group") == "headline"]
    if not rows:
        for k in ["churnMin", "churnMax", "accDiffMin", "accDiffMax",
                  "churnAccRatioMin", "churnAccRatioMax",
                  "symKLDatasetSpread",
                  "churnAccCorrPearson", "churnAccCorrSpearman"]:
            add(k, "??")
        return
    churns = [_f(r["churn_mean"]) * 100 for r in rows]
    acc_diffs = [_f(r["acc_diff_pp_mean"]) for r in rows]
    sym_kls = [_f(r["sym_kl_mean"]) for r in rows]
    erm_accs = [_f(r["erm_id_acc_mean"]) for r in rows]
    add("churnMin", min(churns), "pct1")
    add("churnMax", max(churns), "pct1")
    add("accDiffMin", min(acc_diffs), "pct1")
    add("accDiffMax", max(acc_diffs), "pct1")
    # Per-prediction-disagreement vs aggregate-accuracy ratio range.
    ratios = [c / a for c, a in zip(churns, acc_diffs) if a > 0]
    add("churnAccRatioMin", round(min(ratios)), "pct0")
    add("churnAccRatioMax", round(max(ratios)), "pct0")
    add("symKLDatasetSpread", round(max(sym_kls) / min(sym_kls)), "pct0")
    # Dataset-level correlation between churn and ERM accuracy
    # (Pearson, Spearman).  Reviewer-asked diagnostic: does churn
    # just track task difficulty?  Yes at the dataset level (the
    # correlation is strong) but the methods reduce churn on every
    # dataset regardless, and per-example churn carries information
    # beyond predictive entropy (app:entropy).
    try:
        from scipy.stats import pearsonr, spearmanr
        pe = pearsonr(churns, erm_accs)[0]
        sp = spearmanr(churns, erm_accs)[0]
        add("churnAccCorrPearson", f"{pe:+.2f}")
        add("churnAccCorrSpearman", f"{sp:+.2f}")
    except Exception:
        add("churnAccCorrPearson", "??")
        add("churnAccCorrSpearman", "??")


def emit_main_table() -> None:
    rows = _read_csv(OUTPUTS / "main_table.csv")
    if not rows:
        for k in ["bagFiveLow", "bagFiveHigh", "twinMedianReduction",
                  "paramSideLow", "paramSideHigh",
                  "baceErmChurn", "baceTwinChurn",
                  "bagFiveFinalLow", "bagFiveFinalHigh",
                  "twinFinalLow", "twinFinalHigh"]:
            add(k, "??")
        return

    def _by_method(name: str) -> list[dict[str, Any]]:
        return [r for r in rows if r.get("method") == name and r.get("dataset") != "AGGREGATE"]

    def _rel_pct(rs: list[dict[str, Any]]) -> list[float]:
        out = []
        for r in rs:
            d = _f(r.get("delta_id_churn_mean"))
            base = _f(r.get("id_churn_mean"))
            if d is None or base is None or base == 0:
                continue
            erm_baseline = base - d
            if erm_baseline > 0:
                out.append(100 * d / erm_baseline)
        return out

    def _by_method_substr(substr: str) -> list[dict[str, Any]]:
        return [r for r in rows
                if substr in r.get("method", "")
                and r.get("dataset") != "AGGREGATE"]

    bag5 = _rel_pct(_by_method("Bagging K=5"))
    twin = _rel_pct(_by_method_substr("Twin"))
    mcd = _rel_pct(_by_method("MC dropout"))
    de5 = _rel_pct(_by_method("Deep Ensemble K=5"))
    swa = _rel_pct(_by_method("SWA"))

    if bag5:
        add("bagFiveLow", -max(bag5), "pct0")    # most-negative = strongest cut
        add("bagFiveHigh", -min(bag5), "pct0")
    if twin:
        add("twinMedianReduction", -median(twin), "pct0")

    # Twin-bootstrap incremental reduction relative to bagging-K=2 (the
    # matched-compute baseline).  twinMedianReduction is vs ERM; this
    # macro answers the natural follow-up "how much better than the
    # matched-compute baseline?": median across datasets of
    # (twin_churn - bag2_churn) / bag2_churn.  Used in the abstract for
    # the "further median X% beyond bagging-K=2" claim.
    bag2_by_ds = {r["dataset"]: _f(r.get("id_churn_mean"))
                  for r in _by_method("Bagging K=2")}
    twin_rows = _by_method_substr("Twin")
    further = []
    for r in twin_rows:
        ds = r.get("dataset")
        twin_c = _f(r.get("id_churn_mean"))
        bag2_c = bag2_by_ds.get(ds)
        if twin_c is None or bag2_c is None or bag2_c <= 0:
            continue
        further.append(100.0 * (twin_c - bag2_c) / bag2_c)
    if further:
        add("twinFurtherMedianReductionVsBagTwo",
            -median(further), "pct0")
    # Combined param-side: MC dropout + DE-K=5 + SWA (all weight-side
    # techniques that do not vary the training-data sample).
    param_side = mcd + de5 + swa
    if param_side:
        add("paramSideLow", min(param_side), "pp1")
        add("paramSideHigh", max(param_side), "pp1")

    # Final-churn ranges (not deltas): used in the "discussion" workflow
    # paragraph.  Bagging-K=5 cuts ERM 16-22% to FinalLow-FinalHigh%, etc.
    def _final(rs: list[dict[str, Any]]) -> list[float]:
        return [_f(r.get("id_churn_mean")) * 100 for r in rs
                if _f(r.get("id_churn_mean")) is not None]

    bag5_final = _final(_by_method("Bagging K=5"))
    twin_final = _final(_by_method_substr("Twin"))
    if bag5_final:
        add("bagFiveFinalLow", min(bag5_final), "pct1")
        add("bagFiveFinalHigh", max(bag5_final), "pct1")
    if twin_final:
        add("twinFinalLow", min(twin_final), "pct1")
        add("twinFinalHigh", max(twin_final), "pct1")

    # Per-figure-0 BACE numbers (ERM, twin) so the fig0 caption can
    # \input from the macros file instead of carrying literals.
    def _bace_churn(method_substr: str) -> float | None:
        for r in rows:
            if r.get("dataset") == "bace" and method_substr in r.get("method", ""):
                v = _f(r.get("id_churn_mean"))
                if v is not None:
                    return v * 100
        return None

    bace_erm = _bace_churn("ERM")
    bace_twin = _bace_churn("Twin")
    if bace_erm is not None:
        add("baceErmChurn", bace_erm, "pct1")
    if bace_twin is not None:
        add("baceTwinChurn", bace_twin, "pct1")

    # Mean Δaccuracy across the 9 datasets, per method.  Surfaces the
    # decoupling argument: deep ensembles raise mean accuracy without
    # reducing churn; twin-bootstrap reduces churn without raising
    # accuracy; "churn is just accuracy" is therefore false at the
    # method level.
    erm_rows = _by_method("ERM")
    erm_acc = {r["dataset"]: _f(r.get("id_acc_mean")) for r in erm_rows}

    def _mean_dacc_pp(rs: list[dict[str, Any]]) -> float | None:
        accs = []
        for r in rs:
            ds = r.get("dataset")
            cur = _f(r.get("id_acc_mean"))
            if ds in erm_acc and erm_acc[ds] is not None and cur is not None:
                accs.append((cur - erm_acc[ds]) * 100)
        return sum(accs) / len(accs) if accs else None

    de_dacc = _mean_dacc_pp(_by_method("Deep Ensemble K=5"))
    twin_dacc = _mean_dacc_pp(_by_method_substr("Twin"))
    if de_dacc is not None:
        add("deepEnsAccDelta", de_dacc, "pp1")
    if twin_dacc is not None:
        add("twinAccDelta", twin_dacc, "pp1")


def emit_distributional() -> None:
    rows = _read_csv(OUTPUTS / "distributional.csv")
    if not rows:
        for k in ["symKLBagFoldLow", "symKLBagFoldHigh",
                  "symKLTwinAdditionalFoldMedian"]:
            add(k, "??")
        return

    by = {(r["dataset"], r["method"]): _f(r.get("fold_reduction")) for r in rows}
    bag5 = [v for (d, m), v in by.items()
            if "Bagging $K{=}5$" in m and v and v > 0]
    if bag5:
        add("symKLBagFoldLow", round(min(bag5)), "pct0")
        add("symKLBagFoldHigh", round(max(bag5)), "pct0")

    # Twin-bootstrap "another factor of ~X beyond bagging-K=5": ratio of
    # twin's fold-reduction-vs-ERM to bagging-K=5's fold-reduction-vs-ERM,
    # per dataset, median across datasets.
    additional_folds = []
    for ds in {d for d, _ in by.keys()}:
        twin_fold = by.get((ds, "Twin-bootstrap $\\lambda{=}300$"))
        bag_fold = by.get((ds, "Bagging $K{=}5$"))
        if twin_fold and bag_fold and bag_fold > 0:
            additional_folds.append(twin_fold / bag_fold)
    if additional_folds:
        add("symKLTwinAdditionalFoldMedian",
            round(median(additional_folds)), "pct0")


def emit_friedman() -> None:
    rows = _read_csv(OUTPUTS / "friedman.csv")
    if not rows:
        for k in ["friedmanChi", "friedmanP", "friedmanCD",
                  "friedmanRankERM", "friedmanRankMCD",
                  "friedmanRankSWA",
                  "friedmanRankDeepEns", "friedmanRankBagTwo",
                  "friedmanRankBagFive", "friedmanRankTwin",
                  "friedmanRankGapBagFiveTwin"]:
            add(k, "??")
        return
    raw_ranks: dict[str, float] = {}
    for r in rows:
        if r["scope"] == "test" and r["key"] == "chi2":
            add("friedmanChi", float(r["value"]), "ratio")
        if r["scope"] == "test" and r["key"] == "p_value":
            # Render as math-mode "$5{\times}10^{-8}$" instead of Python's
            # "5e-08", which reads as a token in the rendered PDF.
            p = float(r["value"])
            if p > 0 and p < 1:
                from math import floor, log10
                exp = int(floor(log10(p)))
                mant = p / 10 ** exp
                add("friedmanP", f"{mant:.1f}{{\\times}}10^{{{exp}}}")
            else:
                add("friedmanP", f"{p:.1g}")
        if r["scope"] == "test" and r["key"] == "nemenyi_cd":
            add("friedmanCD", float(r["value"]), "ratio")
        if r["scope"] == "mean_rank":
            method_to_macro = {
                "ERM": "friedmanRankERM",
                "MC dropout": "friedmanRankMCD",
                "SWA": "friedmanRankSWA",
                "Deep Ens. K=5": "friedmanRankDeepEns",
                "Bagging K=2": "friedmanRankBagTwo",
                "Bagging K=5": "friedmanRankBagFive",
                "Twin-bootstrap": "friedmanRankTwin",
            }
            macro = method_to_macro.get(r["key"])
            if macro:
                add(macro, f"{float(r['value']):.2f}")
                raw_ranks[r["key"]] = float(r["value"])
    # Rank gap between bagging-K=5 and twin-bootstrap.  Computed from
    # the raw (unrounded) ranks so subtraction gives the same value the
    # prose quotes; subtracting the rounded macro values introduces a
    # 1e-2 rounding error.
    if "Bagging K=5" in raw_ranks and "Twin-bootstrap" in raw_ranks:
        gap = abs(raw_ranks["Bagging K=5"] - raw_ranks["Twin-bootstrap"])
        add("friedmanRankGapBagFiveTwin", f"{gap:.2f}")


def emit_triage() -> None:
    rows = _read_csv(OUTPUTS / "convergence_recall.csv")
    if not rows:
        for k in ["triageKtenLow", "triageKtenHigh",
                  "triageKtwoLow", "triageKtwoHigh",
                  "triageKtwoToKtenGapLow", "triageKtwoToKtenGapHigh"]:
            add(k, "??")
        return

    def _by_K(K: int) -> list[tuple[str, float]]:
        return [(r["dataset"], _f(r["mean_recall"])) for r in rows
                if int(r["K"]) == K and _f(r["mean_recall"]) is not None]

    k10 = _by_K(10)
    k2 = _by_K(2)
    if k10:
        add("triageKtenLow", min(v for _, v in k10) * 100, "pct0")
        add("triageKtenHigh", max(v for _, v in k10) * 100, "pct0")
    if k2:
        add("triageKtwoLow", min(v for _, v in k2) * 100, "pct0")
        add("triageKtwoHigh", max(v for _, v in k2) * 100, "pct0")
    # Per-dataset gap K=10 - K=2 (in pp); the prose claim "within X-Y pp
    # of the K=10 gold standard on every dataset".
    if k10 and k2:
        k10_by = dict(k10)
        gaps = [(k10_by[d] - v) * 100 for d, v in k2 if d in k10_by]
        if gaps:
            add("triageKtwoToKtenGapLow", round(min(gaps)), "pct0")
            add("triageKtwoToKtenGapHigh", round(max(gaps)), "pct0")


def emit_entropy() -> None:
    rows = _read_csv(OUTPUTS / "entropy_vs_fragility.csv")
    if not rows:
        for k in ["fragTopLow", "fragTopHigh", "entropyTopLow", "entropyTopHigh",
                  "entropyGapMin", "entropyGapMax"]:
            add(k, "??")
        return
    frag = [_f(r["frag_top10_capture"]) * 100 for r in rows]
    ent = [_f(r["ent_top10_capture"]) * 100 for r in rows]
    gaps_pp = [f - e for f, e in zip(frag, ent)]
    add("fragTopLow", min(frag), "pct0")
    add("fragTopHigh", max(frag), "pct0")
    add("entropyTopLow", min(ent), "pct0")
    add("entropyTopHigh", max(ent), "pct0")
    add("entropyGapMin", round(min(gaps_pp)), "pct0")
    add("entropyGapMax", round(max(gaps_pp)), "pct0")


def emit_bo_topk() -> None:
    """Top-K Jaccard stability summary across the nine chemistry datasets.

    Three macro families:
      \\boTopkErmJaccardLow / High        ERM Jaccard range
      \\boTopkTwinJaccardLow / High       Twin-bootstrap Jaccard range
      \\boTopkTwinDeltaMin / Max          paired Δ vs ERM range (twin)
      \\boTopkErmHitRateLow / High        ERM hit-rate range (sanity)
      \\boTopkTwinDeltaPosCount           number of datasets with positive Δ
    """
    rows = _read_csv(OUTPUTS / "bo_topk.csv")
    keys = ["boTopkK",
            "boTopkErmJaccardLow", "boTopkErmJaccardHigh",
            "boTopkTwinJaccardLow", "boTopkTwinJaccardHigh",
            "boTopkTwinDeltaMin", "boTopkTwinDeltaMax",
            "boTopkErmHitRateLow", "boTopkErmHitRateHigh",
            "boTopkHitGapMin", "boTopkHitGapMax"]
    if not rows:
        for k in keys:
            add(k, "??")
        return
    erm_jac = [_f(r["jaccard_mean"]) for r in rows if r["method"] == "ERM"]
    twin_rows = [r for r in rows if r["method"].startswith("Twin")]
    twin_jac = [_f(r["jaccard_mean"]) for r in twin_rows]
    twin_deltas = [_f(r["delta_mean"]) for r in twin_rows
                   if r["delta_mean"] not in ("", "nan")]
    erm_hits = [_f(r["hit_rate_mean"]) * 100 for r in rows if r["method"] == "ERM"]
    # Hit rate vs class prior gap, per dataset (in percentage points).
    # Range across datasets characterises "above-chance" surrogate performance.
    hit_gaps = []
    for r in rows:
        if r["method"] != "ERM":
            continue
        hr = _f(r["hit_rate_mean"]) * 100
        cp = _f(r["class_prior"]) * 100
        hit_gaps.append(hr - cp)
    K_val = rows[0].get("K", "10")
    add("boTopkK", K_val)
    # Two decimal places suit Jaccard; helpers below expect a numeric format key.
    add("boTopkErmJaccardLow",  f"{min(erm_jac):.2f}")
    add("boTopkErmJaccardHigh", f"{max(erm_jac):.2f}")
    add("boTopkTwinJaccardLow",  f"{min(twin_jac):.2f}")
    add("boTopkTwinJaccardHigh", f"{max(twin_jac):.2f}")
    add("boTopkTwinDeltaMin", f"{min(twin_deltas):+.2f}")
    add("boTopkTwinDeltaMax", f"{max(twin_deltas):+.2f}")
    add("boTopkErmHitRateLow",  f"{min(erm_hits):.0f}")
    add("boTopkErmHitRateHigh", f"{max(erm_hits):.0f}")
    if hit_gaps:
        add("boTopkHitGapMin", f"{min(hit_gaps):.0f}")
        add("boTopkHitGapMax", f"{max(hit_gaps):.0f}")


def emit_bo_loop_regression() -> None:
    """BO trajectory variance summary across the three regression datasets.

    Macros emitted (per-method ERM / Bag / Twin):
      boLoopRegK / InitN / Budget / Lam            protocol values
      boLoopReg{Erm,Bag,Twin}StdLow/High           per-method std range
      boLoopReg{Bag,Twin}StdRedLow/High            std reduction vs ERM (% range)
      boLoopReg{Erm,Bag,Twin}JaccardLow/High       per-method Jaccard range
      boLoopReg{Erm,Bag,Twin}StdRangePctLow/High   std as % of dataset y-range
    """
    rows = _read_csv(OUTPUTS / "bo_loop_regression_summary.csv")
    method_tags = {"erm": "Erm", "bagging": "Bag", "twin": "Twin"}
    keys = ["boLoopRegK", "boLoopRegInitN", "boLoopRegBudget", "boLoopRegLam"]
    for tag in method_tags.values():
        keys += [
            f"boLoopReg{tag}StdLow",          f"boLoopReg{tag}StdHigh",
            f"boLoopReg{tag}JaccardLow",      f"boLoopReg{tag}JaccardHigh",
            f"boLoopReg{tag}StdRangePctLow",  f"boLoopReg{tag}StdRangePctHigh",
        ]
    for tag in ("Bag", "Twin"):
        keys += [f"boLoopReg{tag}StdRedLow", f"boLoopReg{tag}StdRedHigh"]
    if not rows:
        for k in keys:
            add(k, "??")
        return

    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], {})[r["method"]] = r

    by_method = {m: [r for r in rows if r["method"] == m]
                 for m in method_tags}
    n_traj = max((int(_f(r["n_trajectories"])) for r in rows), default=20)
    add("boLoopRegK",      str(n_traj))
    add("boLoopRegInitN",  "50")
    add("boLoopRegBudget", "10")
    add("boLoopRegLam",    "3")
    for method, tag in method_tags.items():
        m_rows = by_method[method]
        if not m_rows:
            for k in [f"boLoopReg{tag}StdLow", f"boLoopReg{tag}StdHigh",
                      f"boLoopReg{tag}JaccardLow", f"boLoopReg{tag}JaccardHigh",
                      f"boLoopReg{tag}StdRangePctLow",
                      f"boLoopReg{tag}StdRangePctHigh"]:
                add(k, "??")
            continue
        stds = [_f(r["final_best_std"]) for r in m_rows]
        jacs = [_f(r["jaccard_mean"]) for r in m_rows]
        pcts = [_f(r["std_pct_of_range"]) for r in m_rows]
        add(f"boLoopReg{tag}StdLow",  f"{min(stds):.3f}")
        add(f"boLoopReg{tag}StdHigh", f"{max(stds):.3f}")
        add(f"boLoopReg{tag}JaccardLow",  f"{min(jacs):.2f}")
        add(f"boLoopReg{tag}JaccardHigh", f"{max(jacs):.2f}")
        add(f"boLoopReg{tag}StdRangePctLow",  f"{min(pcts):.2f}")
        add(f"boLoopReg{tag}StdRangePctHigh", f"{max(pcts):.2f}")
    # Per-dataset, per-method final-best min/max across trajectories.
    # Used in the appendix prose to anchor "ERM trajectories scatter
    # across y values from X to Y" claims without drift.  Dataset short
    # names mapped to MixedCase tags for the macro names.
    ds_tags = {"esol_reg": "Esol", "freesolv_reg": "Freesolv",
               "lipo_reg": "Lipo"}
    method_tags_local = {"erm": "Erm", "bagging": "Bag", "twin": "Twin"}
    for ds, cells in by_ds.items():
        ds_tag = ds_tags.get(ds)
        if ds_tag is None:
            continue
        for method, m_tag in method_tags_local.items():
            r = cells.get(method)
            if r is None:
                continue
            lo = _f(r.get("final_best_min"))
            hi = _f(r.get("final_best_max"))
            if lo is not None:
                add(f"boLoopReg{ds_tag}{m_tag}FinalLo", f"{lo:.2f}")
            if hi is not None:
                add(f"boLoopReg{ds_tag}{m_tag}FinalHi", f"{hi:.2f}")

    # Per-dataset std-reduction vs ERM (positive % = how much variance the
    # method removes); also count the datasets where the method's std
    # actually beats ERM, so the prose can say "k/N datasets" without
    # hard-coding the count.
    n_datasets = len(by_ds)
    add("boLoopRegNumDatasets", str(n_datasets))
    for method, tag in (("bagging", "Bag"), ("twin", "Twin")):
        reductions = []
        wins = 0
        for cells in by_ds.values():
            e = cells.get("erm"); t = cells.get(method)
            if e is None or t is None:
                continue
            es = _f(e["final_best_std"]); ts = _f(t["final_best_std"])
            if es is None or ts is None or es <= 1e-9:
                continue
            reductions.append(100 * (1 - ts / es))
            if ts < es:
                wins += 1
        if reductions:
            add(f"boLoopReg{tag}StdRedLow",  f"{min(reductions):.0f}")
            add(f"boLoopReg{tag}StdRedHigh", f"{max(reductions):.0f}")
        add(f"boLoopReg{tag}StdRedWins", str(wins))


def emit_regression() -> None:
    rows = _read_csv(OUTPUTS / "regression.csv")
    if not rows:
        for k in ["regTwinLow", "regTwinHigh", "regBagTwoLow", "regBagTwoHigh",
                  "regBagFiveLow", "regBagFiveHigh", "regFragRatioLow", "regFragRatioHigh"]:
            add(k, "??")
        return

    def _rel(method_substr: str) -> list[float]:
        out = []
        for r in rows:
            if method_substr not in r.get("method", ""):
                continue
            rel = _f(r.get("rel_pct"))
            if rel is not None:
                out.append(rel)
        return out

    twin = _rel("Twin-bootstrap")
    bag2 = _rel("Bagging-$K{=}2$")
    bag5 = _rel("Bagging-$K{=}5$")
    if twin:
        add("regTwinLow", -max(twin), "pct0")
        add("regTwinHigh", -min(twin), "pct0")
    if bag2:
        add("regBagTwoLow", -max(bag2), "pct0")
        add("regBagTwoHigh", -min(bag2), "pct0")
    if bag5:
        add("regBagFiveLow", -max(bag5), "pct0")
        add("regBagFiveHigh", -min(bag5), "pct0")

    # Per-example regression fragility / mean MAE ratio.
    erm_rows = [r for r in rows if r["method"] == "ERM"]
    ratios = []
    for r in erm_rows:
        frag = _f(r.get("id_frag"))
        mae = _f(r.get("id_mae"))
        if frag and mae:
            ratios.append(100 * frag / mae)
    if ratios:
        add("regFragRatioLow", round(min(ratios)), "pct0")
        add("regFragRatioHigh", round(max(ratios)), "pct0")


def emit_chemberta() -> None:
    rows = _read_csv(OUTPUTS / "chemberta_heldout.csv")
    if not rows:
        for k in ["chembertaAccDropLow", "chembertaAccDropHigh",
                  "chembertaCutLow", "chembertaCutHigh",
                  "chembertaLamTenAccBoundPP"]:
            add(k, "??")
        return
    drops_pp = []
    cuts_pct = []
    for r in rows:
        erm_acc = _f(r.get("erm_acc"))
        t300 = _f(r.get("t300_acc"))
        if erm_acc and t300:
            drops_pp.append((erm_acc - t300) * 100)
        rel10 = _f(r.get("t10_rel"))
        if rel10:
            cuts_pct.append(-rel10)  # negate so positive = cut
    if drops_pp:
        add("chembertaAccDropLow", round(min(drops_pp)), "pct0")
        add("chembertaAccDropHigh", round(max(drops_pp)), "pct0")
    if cuts_pct:
        add("chembertaCutLow", round(min(cuts_pct)), "pct0")
        add("chembertaCutHigh", round(max(cuts_pct)), "pct0")

    # Per-dataset count of "fails to reduce churn at lambda=300": the
    # paired delta CI is non-negative on the high side (mean >= 0 or CI
    # straddles zero).  Excludes datasets where the apparent reduction
    # is a model collapse (handled in prose).  The denominator is the
    # number of ChemBERTa datasets with a measurable t300 result.
    n_fail = 0
    n_total = 0
    for r in rows:
        d_lo = _f(r.get("t300_dchurn_lo"))
        d_hi = _f(r.get("t300_dchurn_hi"))
        if d_lo is None or d_hi is None:
            continue
        n_total += 1
        if d_hi >= 0:  # CI does not lie strictly below zero -> not a real reduction
            n_fail += 1
    add("chembertaLamThreeHundredFailCount", str(n_fail))
    add("chembertaLamThreeHundredDenom", str(n_total))


def emit_waterbirds() -> None:
    rows = _read_csv(OUTPUTS / "waterbirds_lambda.csv")
    if not rows:
        for k in ["waterbirdsAccCollapse", "waterbirdsLamTenCut"]:
            add(k, "??")
        return
    erm_acc = _f(next((r["id_acc_mean"] for r in rows if r["method"] == "ERM"), None))
    t300 = next((r for r in rows if "$\\lambda{=}300$" in r["method"]), None)
    t10 = next((r for r in rows if "$\\lambda{=}10$" in r["method"]), None)
    if erm_acc and t300:
        drop = (erm_acc - _f(t300["id_acc_mean"])) * 100
        add("waterbirdsAccCollapse", round(drop), "pct0")
    if t10:
        rel = _f(t10.get("rel_pct"))
        if rel:
            add("waterbirdsLamTenCut", round(-rel), "pct0")


def emit_gin() -> None:
    bg = _read_csv(OUTPUTS / "bace_gin.csv")
    gl = _read_csv(OUTPUTS / "gin_lambda.csv")

    if bg:
        bag5 = next((r for r in bg if r["method"] == "Bagging-K=5"), None)
        twin300 = next((r for r in bg if r["method"] == "Twin-indep λ=300"), None)
        if bag5:
            rel = _f(bag5.get("rel_churn_pct"))
            if rel:
                add("ginBagCutLamThreeHundred", round(-rel), "pct0")
            add("ginBagAccGain", _f(bag5.get("d_acc_pp")), "pp1")
        if twin300:
            # Used by the auto-generated paper/sections/tables/gin.tex
            # caption; not in body prose.
            add("ginAccDropLamThreeHundred",
                round(-_f(twin300.get("d_acc_pp"))), "pct0")

    if gl:
        rule = next((r for r in gl if r.get("rule_picked") in ("True", True)), None)
        if rule:
            rel = _f(rule.get("rel_churn_pct"))
            if rel:
                add("ginCutLamTen", round(-rel), "pct0")
                # One-decimal precision for the rule-selected-λ Δ-churn
                # claim that the appendix prose quotes.
                add("ginRuleLamCutPct", f"{-rel:.1f}")
            add("ginRuleLam", str(int(_f(rule["lam"]) or 0)))
            add("ginRuleLamDeltaLo", _f(rule.get("d_churn_lo_pp")), "pp1")
            add("ginRuleLamDeltaHi", _f(rule.get("d_churn_hi_pp")), "pp1")
            add("ginRuleLamAccGain", _f(rule.get("d_acc_pp")), "pp1")


def emit_filter_outcomes() -> None:
    rows = [r for r in _read_csv(OUTPUTS / "filter_outcomes.csv")
            if r.get("group") == "borderline"]
    if rows:
        gaps = [_f(r["gap_pp"]) for r in rows]
        add("borderlineGapMin", round(min(gaps)), "pct0")
        add("borderlineGapMax", round(max(gaps)), "pct0")
        # Borderline test sizes (smallest, largest among the 3 marginal
        # datasets) for the prose claim "57--104-example test sets".
        ns = [int(_f(r["n_id_test"]) or 0) for r in rows]
        add("borderlineNTestMin", str(min(ns)))
        add("borderlineNTestMax", str(max(ns)))


def emit_nscaling() -> None:
    """N-scaling: log-log slope and shape characterisation.

    Surfaces `\\nScalingSlope` (-0.20 on canonical-seed 99 across the
    9-M grid), the seed-averaged endpoint sym-KL values
    (`\\nScalingSymKLLow` at the smallest M, `\\nScalingSymKLMin` at
    the floor, `\\nScalingSymKLEnd` at the full pool) and the M
    where the seed-averaged minimum is attained (`\\nScalingMmin`).
    The averages are taken across canonical seeds {99, 7, 42}, three
    independent draws of the canonical train/test split; averaging
    smooths the per-seed bootstrap noise that dominates a single-seed
    measurement at this dataset size.
    """
    rows = _read_csv(OUTPUTS / "nscaling_bace.csv")
    slope_rows = [r for r in rows if r.get("scope") == "slope"]
    if slope_rows:
        add("nScalingSlope", f"{float(slope_rows[0]['sym_kl_mean']):.2f}")
    else:
        add("nScalingSlope", "??")

    # Average sym-KL across the 3 canonical seeds for each M.
    try:
        import sys
        sys.path.insert(0, str(Path("scripts").resolve()))
        from _analysis_lib import load_runs, pairwise_metrics
        import numpy as np
    except ImportError:
        for k in ("nScalingSymKLLow", "nScalingSymKLMin",
                  "nScalingSymKLEnd", "nScalingMmin"):
            add(k, "??")
        return

    Ms = [200, 300, 400, 500, 600, 700, 800, 900, 968]
    seeds = [99, 7, 42]

    def _seed_dir(seed: int, M: int) -> Path:
        if seed == 99:
            return RUN_ROOT.parent / "cross_sample_nscaling" / f"M{M}" / "bace"
        return (RUN_ROOT.parent / "cross_sample_nscaling" / "sensitivity"
                / f"seed{seed}" / f"M{M}" / "bace")

    avg_by_M = {}
    for M in Ms:
        per_seed = []
        for s in seeds:
            runs = load_runs(_seed_dir(s, M), "erm_train*.npz")
            if not runs:
                continue
            pm, _ = pairwise_metrics(runs)
            per_seed.append(float(np.mean([m["id_sym_kl"] for m in pm.values()])))
        if per_seed:
            avg_by_M[M] = float(np.mean(per_seed))

    if not avg_by_M:
        for k in ("nScalingSymKLLow", "nScalingSymKLMin",
                  "nScalingSymKLEnd", "nScalingMmin"):
            add(k, "??")
        return

    M_at_min = min(avg_by_M, key=avg_by_M.get)
    add("nScalingSymKLLow", f"{avg_by_M[min(avg_by_M)]:.2f}")
    add("nScalingSymKLMin", f"{avg_by_M[M_at_min]:.2f}")
    add("nScalingSymKLEnd", f"{avg_by_M[max(avg_by_M)]:.2f}")
    add("nScalingMmin", str(M_at_min))


def emit_per_dataset_callouts() -> None:
    """Single-dataset claims the prose names by cell.

    DILI accuracy / test size for the experiments.tex compute-matched
    discussion; cyp2d6_substrate \\Delta-recall / \\Delta-acc for the
    measurement.tex imbalance-undersells-accuracy paragraph.
    """
    fmag = _read_csv(OUTPUTS / "fragility_magnitudes.csv")
    main = _read_csv(OUTPUTS / "main_table.csv")
    addm = _read_csv(OUTPUTS / "additional_metrics.csv")

    dili_main = next((r for r in main if r.get("dataset") == "dili"
                      and r.get("method") == "ERM"), None)
    dili_mag = next((r for r in fmag if r.get("dataset") == "dili"), None)
    if dili_main:
        add("diliErmAcc", f"{float(dili_main['id_acc_mean']):.2f}")
    if dili_mag:
        add("diliNTest", str(int(_f(dili_mag["n_id_test"]) or 0)))

    cyp = next((r for r in addm if r.get("dataset") == "cyp2d6_substrate"), None)
    if cyp:
        add("cypTwoDeltaAcc", f"{float(cyp['d_acc_pp_mean']):.1f}")
        add("cypTwoDeltaRecall", f"{float(cyp['d_recall_pp_mean']):.1f}")
        # Per-example argmax-churn for cyp2d6, used in the same paragraph.
    cyp_pcc = _read_csv(OUTPUTS / "per_class_churn.csv")
    cyp_row = next((r for r in cyp_pcc if r.get("dataset") == "cyp2d6_substrate"), None)
    if cyp_row:
        add("cypTwoOverallChurn", f"{float(cyp_row['overall_mean'])*100:.1f}")
        add("cypTwoPosFrac", f"{float(cyp_row['pos_frac']):.2f}")
    dili_pcc = next((r for r in cyp_pcc if r.get("dataset") == "dili"), None)
    if dili_pcc:
        add("diliOverallChurn", f"{float(dili_pcc['overall_mean'])*100:.1f}")
    # BBB-Martins / BBBP share the most-imbalanced positive fraction
    # (round to 2 decimals).  Use BBB-Martins' value since it's the
    # one the prose names first.
    bbbm = next((r for r in cyp_pcc if r.get("dataset") == "bbb_martins"), None)
    if bbbm:
        add("imbalPosFracHigh", f"{float(bbbm['pos_frac']):.2f}")

    # ERM-GIN id-acc / tolerance threshold for the GIN appendix.
    bgin = _read_csv(OUTPUTS / "bace_gin.csv")
    erm_gin = next((r for r in bgin if r.get("method") == "ERM"), None)
    if erm_gin:
        acc = float(erm_gin["id_acc_mean"])
        add("ginErmAcc", f"{acc:.3f}")
        add("ginErmAccTolerance", f"{acc - 0.02:.3f}")
    # GIN sym-KL fold reduction at lambda=300: prose claims "99% reduction".
    twin300_gin = next((r for r in bgin if "300" in (r.get("method") or "")), None)
    if erm_gin and twin300_gin:
        erm_kl = float(erm_gin["id_sym_kl_mean"])
        twin_kl = float(twin300_gin["id_sym_kl_mean"])
        if erm_kl > 0:
            pct_reduce = 100.0 * (1.0 - twin_kl / erm_kl)
            add("ginSymKLReductionPct", f"{pct_reduce:.0f}")

    # MOF-thermal floor in the triage-convergence figure (caption only).
    # The prose says "MOF-thermal is the floor on every K (X-Y%)".
    conv = _read_csv(OUTPUTS / "convergence_recall.csv")
    if conv:
        mof_recalls = [
            float(r["mean_recall"]) * 100
            for r in conv
            if r.get("dataset") == "mof_thermal"
            and _f(r.get("mean_recall")) is not None
        ]
        if mof_recalls:
            add("mofThermalConvLow", f"{min(mof_recalls):.0f}")
            add("mofThermalConvHigh", f"{max(mof_recalls):.0f}")

    # Regression id-MAE improvement vs ERM, range across 3 datasets x 3
    # methods (prose: "all four methods improve id-MAE by 3-9% over ERM").
    reg = _read_csv(OUTPUTS / "regression.csv")
    if reg:
        erm_by_ds = {r["dataset"]: float(r["id_mae"])
                     for r in reg if r["method"] == "ERM"}
        improvements = []
        for r in reg:
            ds = r.get("dataset")
            mae = _f(r.get("id_mae"))
            if r["method"] == "ERM" or mae is None or ds not in erm_by_ds:
                continue
            erm_mae = erm_by_ds[ds]
            if erm_mae > 0:
                improvements.append(100.0 * (erm_mae - mae) / erm_mae)
        if improvements:
            add("regIdMaeImproveLow", f"{min(improvements):.0f}")
            add("regIdMaeImproveHigh", f"{max(improvements):.0f}")


# ---------------------------------------------------------------------------

def emit_wins_counts() -> None:
    """Per-dataset paired Δ(twin - bag2) and Δ(twin - bag5) wins counts.

    The prose claims "twin-bootstrap wins on $X/N$ datasets" against
    bagging-K=2 (matched compute) and against bagging-K=5 (5x compute,
    held-out only).  These integer ratios change only if a per-dataset
    paired CI flips sign, so we recompute them here from the raw NPZs
    via the analysis lib.

    Twin "wins" on a dataset iff the paired Δ id-churn (twin - other)
    has CI strictly excluding zero on the negative side
    (mean < 0 AND hi < 0).
    """
    try:
        from _analysis_lib import (
            bootstrap_paired,
            load_runs,
            pairwise_metrics,
        )
    except ImportError:
        for k in ("twinWinsBagTwoMain", "twinWinsBagTwoMainDenom",
                  "twinWinsBagTwoHeldout", "twinWinsBagTwoHeldoutDenom",
                  "twinWinsBagFiveHeldout", "twinWinsBagFiveHeldoutDenom",
                  "bagFiveWinsTwinHeldout"):
            add(k, "??")
        return

    if not RUN_ROOT.exists():
        for k in ("twinWinsBagTwoMain", "twinWinsBagTwoMainDenom",
                  "twinWinsBagTwoHeldout", "twinWinsBagTwoHeldoutDenom",
                  "twinWinsBagFiveHeldout", "twinWinsBagFiveHeldoutDenom",
                  "bagFiveWinsTwinHeldout"):
            add(k, "??")
        return

    twin_glob = f"twin_indep_train*_lam{FROZEN_LAM}.npz"
    bag2_glob = "bagging_train*_K2.npz"
    bag5_glob = "bagging_train*_K5.npz"

    def paired_delta_churn(ds_dir: Path, glob_a: str, glob_b: str):
        """Return (mean, lo, hi) of paired (a - b) id_churn across seed pairs."""
        ra = load_runs(ds_dir, glob_a)
        rb = load_runs(ds_dir, glob_b)
        if not ra or not rb:
            return None
        pma, pairs_a = pairwise_metrics(ra)
        pmb, pairs_b = pairwise_metrics(rb)
        common = [p for p in pairs_a if p in pmb]
        if not common:
            return None
        deltas = [pma[p]["id_churn"] - pmb[p]["id_churn"] for p in common]
        return bootstrap_paired(deltas)

    main_datasets = [DEV_DATASET] + list(HEADLINE_DATASETS)

    def count_wins(datasets, other_glob):
        n_wins = 0
        n_total = 0
        for ds in datasets:
            res = paired_delta_churn(RUN_ROOT / ds, twin_glob, other_glob)
            if res is None:
                continue
            n_total += 1
            mean, lo, hi = res
            if mean < 0 and hi < 0:
                n_wins += 1
        return n_wins, n_total

    w_main_b2, n_main = count_wins(main_datasets, bag2_glob)
    w_held_b2, n_held = count_wins(HEADLINE_DATASETS, bag2_glob)
    w_held_b5, n_held5 = count_wins(HEADLINE_DATASETS, bag5_glob)
    add("twinWinsBagTwoMain", str(w_main_b2))
    add("twinWinsBagTwoMainDenom", str(n_main))
    add("twinWinsBagTwoHeldout", str(w_held_b2))
    add("twinWinsBagTwoHeldoutDenom", str(n_held))
    add("twinWinsBagFiveHeldout", str(w_held_b5))
    add("twinWinsBagFiveHeldoutDenom", str(n_held5))
    add("bagFiveWinsTwinHeldout", str(n_held5 - w_held_b5
                                       - _ties_against_twin(HEADLINE_DATASETS,
                                                            twin_glob, bag5_glob,
                                                            paired_delta_churn)))


def _ties_against_twin(datasets, twin_glob, other_glob, paired):
    """Count datasets where the paired CI straddles zero (no winner)."""
    n_tie = 0
    for ds in datasets:
        res = paired(RUN_ROOT / ds, twin_glob, other_glob)
        if res is None:
            continue
        mean, lo, hi = res
        if lo <= 0 <= hi:
            n_tie += 1
    return n_tie


def emit_seed_sensitivity() -> None:
    """Largest across-seed range of (id_churn, delta_id_churn) cells.

    Reads outputs/main_table_per_seed.csv (long format produced by
    scripts/aggregate_seed_sensitivity.py) and computes:
      * maxAcrossSeedChurnSpreadPP   - largest across-seed range of
        absolute id_churn_mean across all (dataset, method) cells, in
        percentage points.  Documents the magnitude of the test-set-
        composition effect.
      * maxAcrossSeedChurnSpreadCell - text label naming the cell that
        attains it.
      * maxAcrossSeedDeltaSpreadPP   - same for paired delta_id_churn_mean.
      * maxAcrossSeedDeltaSpreadCell - cell label.
    """
    placeholders = ("maxAcrossSeedChurnSpreadPP",
                    "maxAcrossSeedChurnSpreadCell",
                    "maxAcrossSeedDeltaSpreadPP",
                    "maxAcrossSeedDeltaSpreadCell")
    rows = _read_csv(OUTPUTS / "main_table_per_seed.csv")
    if not rows:
        for k in placeholders:
            add(k, "??")
        return

    cells: dict[tuple[str, str], dict[int, dict[str, float]]] = {}
    for r in rows:
        s = int(r.get("canonical_seed", "0") or 0)
        ds = r.get("dataset", "")
        m  = r.get("method", "")
        if not ds or not m:
            continue
        cells.setdefault((ds, m), {})[s] = {
            "churn":  _f(r.get("id_churn_mean")),
            "delta":  _f(r.get("delta_id_churn_mean")),
        }

    # Map raw method names to LaTeX-safe display strings used in prose.
    method_disp = {
        "ERM":               "ERM",
        "SWA":               "SWA",
        "MC dropout":        "MC dropout",
        "Deep Ensemble K=5": r"deep ensemble $K{=}5$",
        "Bagging K=2":       r"bagging $K{=}2$",
        "Bagging K=5":       r"bagging $K{=}5$",
    }

    def _disp_method(m: str) -> str:
        if m in method_disp:
            return method_disp[m]
        if "Twin" in m:
            return r"twin-bootstrap $\lambda{=}300$"
        return m

    def _spread(metric: str):
        best = (-1.0, "")
        for (ds, m), per_seed in cells.items():
            vals = [v[metric] for v in per_seed.values() if v[metric] is not None]
            if len(vals) < 2:
                continue
            spread_pp = (max(vals) - min(vals)) * 100   # both metrics are fractions (0-1)
            if spread_pp > best[0]:
                ds_disp = display(ds)
                label = f"{ds_disp}, {_disp_method(m)}"
                best = (spread_pp, label)
        return best

    churn_spread, churn_cell = _spread("churn")
    delta_spread, delta_cell = _spread("delta")
    add("maxAcrossSeedChurnSpreadPP",   f"{churn_spread:.1f}")
    add("maxAcrossSeedChurnSpreadCell", churn_cell)
    add("maxAcrossSeedDeltaSpreadPP",   f"{delta_spread:.1f}")
    add("maxAcrossSeedDeltaSpreadCell", delta_cell)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    emit_dataset_counts()
    emit_magnitudes()
    emit_main_table()
    emit_distributional()
    emit_friedman()
    emit_triage()
    emit_entropy()
    emit_bo_topk()
    emit_bo_loop_regression()
    emit_regression()
    emit_chemberta()
    emit_waterbirds()
    emit_gin()
    emit_filter_outcomes()
    emit_nscaling()
    emit_per_dataset_callouts()
    emit_wins_counts()
    emit_seed_sensitivity()

    lines = [
        r"% Auto-generated by scripts/make_paper_macros.py.  DO NOT EDIT.",
        r"% Numerical claims in the paper prose import from this file via",
        r"% \input{sections/macros.tex} so they cannot drift from source.",
        "",
    ]
    seen = set()
    for name, value in MACROS:
        if name in seen:
            continue
        seen.add(name)
        # \providecommand allows \input{macros} to come before or after
        # the prose declares the same names; \renewcommand keeps the
        # latest value if a macro is defined twice.
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_PATH}  ({len(seen)} macros)")
    n_unknown = sum(1 for n, v in MACROS if v == "??")
    if n_unknown:
        print(f"  [warn] {n_unknown} macros emitted as ?? (source CSV missing)")


if __name__ == "__main__":
    main()
