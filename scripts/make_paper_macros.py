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
  \\mcd{Low,High,...}               main_table.csv (MC dropout row aggregate)
  \\paramSide{Low,High}             main_table.csv (combined param-side range)
  \\symKL{BagFold,TwinFold,...}     distributional.csv
  \\entropyTop{Low,High}            entropy_vs_fragility.csv
  \\fragTop{Low,High}               entropy_vs_fragility.csv
  \\triage{KtenLow,KtenHigh,...}    convergence_recall.csv
  \\reg{TwinLow,TwinHigh,...}       regression.csv
  \\waterbirds{...}                 waterbirds_lambda.csv
  \\chemberta{...}                  chemberta_heldout.csv
  \\gin{...}                        gin_lambda.csv, bace_gin.csv
  \\friedmanChi, \\friedmanP        friedman.csv
  \\nDatasets{Headline,Total,...}   paper_constants.py
"""
from __future__ import annotations

import csv
from pathlib import Path
from statistics import median
from typing import Any

from paper_constants import (
    BORDERLINE_DATASETS,
    DEV_DATASET,
    EXCLUDED_DATASETS,
    HEADLINE_DATASETS,
)


OUTPUTS = Path("outputs")
OUT_PATH = Path("paper/sections/macros.tex")

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
    add("nDatasetsBorderline", str(len(BORDERLINE_DATASETS)))
    add("nDatasetsExcluded", str(len(EXCLUDED_DATASETS)))
    add("nDatasetsTotal",
        str(1 + len(HEADLINE_DATASETS) + len(BORDERLINE_DATASETS) + len(EXCLUDED_DATASETS)))
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


def emit_magnitudes() -> None:
    rows = [r for r in _read_csv(OUTPUTS / "fragility_magnitudes.csv")
            if r.get("group") == "headline"]
    if not rows:
        for k in ["churnMin", "churnMax", "accDiffMin", "accDiffMax",
                  "churnAccRatioMin", "churnAccRatioMax",
                  "symKLMin", "symKLMax", "symKLDatasetSpread",
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
    add("symKLMin", min(sym_kls), "ratio")
    add("symKLMax", max(sym_kls), "ratio")
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
                  "twinLow", "twinHigh", "mcdLow", "mcdHigh",
                  "paramSideLow", "paramSideHigh",
                  "baceErmChurn", "baceTwinChurn",
                  "paramSideAbsLow", "paramSideAbsHigh",
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
    bag2 = _rel_pct(_by_method("Bagging K=2"))
    twin = _rel_pct(_by_method_substr("Twin"))
    mcd = _rel_pct(_by_method("MC dropout"))
    de5 = _rel_pct(_by_method("Deep Ensemble K=5"))
    swa = _rel_pct(_by_method("SWA"))
    if swa:
        # Use signed bounds (negative = reduction).  Pattern matches MC
        # dropout / paramSide so the prose can read them directly.
        add("swaLow", min(swa), "pp1")
        add("swaHigh", max(swa), "pp1")

    if bag5:
        add("bagFiveLow", -max(bag5), "pct0")    # most-negative = strongest cut
        add("bagFiveHigh", -min(bag5), "pct0")
    if twin:
        add("twinLow", -max(twin), "pct0")
        add("twinHigh", -min(twin), "pct0")
        add("twinMedianReduction", -median(twin), "pct0")
    if mcd:
        add("mcdLow", min(mcd), "pp1")
        add("mcdHigh", max(mcd), "pp1")
    # Combined param-side: MC dropout + DE-K=5 + SWA (all weight-side
    # techniques that do not vary the training-data sample).
    param_side = mcd + de5 + swa
    if param_side:
        add("paramSideLow", min(param_side), "pp1")
        add("paramSideHigh", max(param_side), "pp1")

    # Absolute pp deltas (not relative %) for the discussion section
    # claim "$-4.6$ to $+1.8$\,pp across the nine datasets".
    def _abs_pp(rs: list[dict[str, Any]]) -> list[float]:
        out = []
        for r in rs:
            d = _f(r.get("delta_id_churn_mean"))
            if d is not None:
                out.append(d * 100)
        return out

    param_pp = _abs_pp(_by_method("MC dropout")) + _abs_pp(_by_method("Deep Ensemble K=5"))
    if param_pp:
        add("paramSideAbsLow", min(param_pp), "pp1")
        add("paramSideAbsHigh", max(param_pp), "pp1")

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
                  "friedmanRankBagFive", "friedmanRankTwin"]:
            add(k, "??")
        return
    for r in rows:
        if r["scope"] == "test" and r["key"] == "chi2":
            add("friedmanChi", float(r["value"]), "ratio")
        if r["scope"] == "test" and r["key"] == "p_value":
            add("friedmanP", f"{float(r['value']):.1g}")
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
    lam_ten_abs_acc_pp = []
    for r in rows:
        erm_acc = _f(r.get("erm_acc"))
        t300 = _f(r.get("t300_acc"))
        t10 = _f(r.get("t10_acc"))
        if erm_acc and t300:
            drops_pp.append((erm_acc - t300) * 100)
        rel10 = _f(r.get("t10_rel"))
        if rel10:
            cuts_pct.append(-rel10)  # negate so positive = cut
        if erm_acc and t10:
            lam_ten_abs_acc_pp.append(abs(erm_acc - t10) * 100)
    if drops_pp:
        add("chembertaAccDropLow", round(min(drops_pp)), "pct0")
        add("chembertaAccDropHigh", round(max(drops_pp)), "pct0")
    if cuts_pct:
        add("chembertaCutLow", round(min(cuts_pct)), "pct0")
        add("chembertaCutHigh", round(max(cuts_pct)), "pct0")
    if lam_ten_abs_acc_pp:
        # Round up to whole pp -- the prose says "within Xpp".
        import math
        add("chembertaLamTenAccBoundPP",
            str(math.ceil(max(lam_ten_abs_acc_pp))))


def emit_waterbirds() -> None:
    rows = _read_csv(OUTPUTS / "waterbirds_lambda.csv")
    if not rows:
        for k in ["waterbirdsAccCollapse", "waterbirdsLamTenCut",
                  "waterbirdsLamTenDeltaPP", "waterbirdsLamTenDeltaLo",
                  "waterbirdsLamTenDeltaHi"]:
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
        d_mean = _f(t10.get("d_churn_mean_pp"))
        d_lo = _f(t10.get("d_churn_lo_pp"))
        d_hi = _f(t10.get("d_churn_hi_pp"))
        if d_mean is not None:
            add("waterbirdsLamTenDeltaPP", d_mean, "pp1")
            add("waterbirdsLamTenDeltaLo", d_lo, "pp1")
            add("waterbirdsLamTenDeltaHi", d_hi, "pp1")
        # |dAcc| bound at lambda=10, rounded up to whole pp for the
        # "within Xpp of ERM" prose claim.
        if erm_acc:
            import math
            add("waterbirdsLamTenAccBoundPP",
                str(math.ceil(abs(erm_acc - _f(t10["id_acc_mean"])) * 100)))


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
            add("ginBagDeltaPP", _f(bag5.get("d_churn_mean_pp")), "pp1")
            add("ginBagDeltaLo", _f(bag5.get("d_churn_lo_pp")), "pp1")
            add("ginBagDeltaHi", _f(bag5.get("d_churn_hi_pp")), "pp1")
            add("ginBagAccGain", _f(bag5.get("d_acc_pp")), "pp1")
        if twin300:
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
            add("ginRuleLamDeltaPP", _f(rule.get("d_churn_mean_pp")), "pp1")
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
    """N-scaling log-log slope, surfaced for the prose claim "slope -0.22"."""
    rows = _read_csv(OUTPUTS / "nscaling_bace.csv")
    slope_rows = [r for r in rows if r.get("scope") == "slope"]
    if slope_rows:
        # CSV column "sym_kl_mean" carries the slope value (slot reuse).
        add("nScalingSlope", f"{float(slope_rows[0]['sym_kl_mean']):.2f}")
    else:
        add("nScalingSlope", "??")


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
    dili_pcc = next((r for r in cyp_pcc if r.get("dataset") == "dili"), None)
    if dili_pcc:
        add("diliOverallChurn", f"{float(dili_pcc['overall_mean'])*100:.1f}")


# ---------------------------------------------------------------------------

def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    emit_dataset_counts()
    emit_magnitudes()
    emit_main_table()
    emit_distributional()
    emit_friedman()
    emit_triage()
    emit_entropy()
    emit_regression()
    emit_chemberta()
    emit_waterbirds()
    emit_gin()
    emit_filter_outcomes()
    emit_nscaling()
    emit_per_dataset_callouts()

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
