# Numerical audit of the paper

Subagent swept every numeric claim in `paper/sections/*.tex` and cross-checked
against the source artefacts in `paper/sections/tables/*.tex`,
`paper/figures/fig1_forest.csv`, and the analysis scripts in `scripts/`.

Each line below is double-checkable: file:line of the claim, and the source
that should generate it.

---

## INCONSISTENT — fix before submission

### 1. Three-point overlap labelled `~63%` instead of `100%`
- **Where:** `paper/sections/introduction.tex:82`
- **Claim:** "Among three overlap points (0%, ~40%, **~63%**)..."
- **Source:** `paper/sections/abstract.tex:16` says `(0%, ~40%, 100%)`;
  `paper/sections/appendix.tex:130–132` (overlap_spectrum) and
  `paper/sections/tables/overlap_spectrum.tex` define the third point as
  `100%` (the same bootstrap to both networks, "Twin-shared 100%").
- **Fix:** change `~63%` → `100%`.

### 2. Magnitude ratio `4–13×` is wider than the data
- **Where:** `paper/sections/magnitudes.tex:11`
- **Claim:** "individual predictions disagree at **4–13×** that rate"
- **Source:** `paper/sections/tables/fragility_magnitudes.tex` —
  ratios churn / |Δacc| span **3.2× (CYP2D6-Sub) to 14.5× (MOF-thermal)**.
- **Fix:** `4–13×` → `3–15×` (or `3–14×`).

### 3. TADF sym-KL exemplar is wrong; ratio overstates spread
- **Where:** `paper/sections/magnitudes.tex:12–13`
- **Claim:** "Sym-KL varies by ~4× across datasets at similar N
  (TADF **0.27** vs.\ BACE 0.75 at N≈1000)"
- **Source:** `paper/sections/tables/fragility_magnitudes.tex` —
  TADF sym-KL = **0.401** (not 0.27); BACE sym-KL = 0.753; ratio = **1.88×**
  (not ~4×). Across all 9 datasets the min/max is 0.38–1.12, ratio 2.93×.
- **Fix:** change `0.27` → `0.40` and `~4×` → `~2×`. Or rephrase to use the
  full-range ratio (`~3×`) without the misleading specific exemplar.

### 4. Regression fragility ratio range
- **Where:** `paper/sections/magnitudes.tex:25`
- **Claim:** per-example regression fragility is "**40–55%** of the
  deployed model's mean absolute error"
- **Source:** `paper/sections/tables/regression.tex` —
  ESOL 0.52/0.94=55%, FreeSolv 0.69/1.65=42%, Lipo 0.38/0.66=**58%**.
- **Fix:** `40–55%` → `42–58%`.

### 5. Entropy gap quoted as `1–18pp` but only goes to 13pp
- **Where:** `paper/sections/experiments.tex:66`
- **Claim:** "per-dataset gap **1–18pp**"
- **Source:** `paper/sections/tables/entropy_vs_fragility.tex` —
  differences span 0.9pp (DILI) to 13.4pp (BBB-Martins).
  `paper/sections/appendix.tex:251–252` already says "1pp to 13pp".
- **Fix:** `1–18pp` → `1–13pp`.

### 6. Minor: BACE twin churn 6.4% vs 6.3%
- **Where:** `paper/sections/introduction.tex:17` (fig0 caption)
- **Claim:** "BACE ERM 16.1% → twin **6.4%**"
- **Source:** `paper/sections/tables/main.tex` gives twin BACE = 16.1 − 9.8 = **6.3%**;
  `paper/sections/appendix.tex:118` (Pareto) writes 6.4%. 0.1pp rounding gap.
- **Fix:** harmonise. Recommend `6.3%` (matches `main.tex` the published table).

---

## UNVERIFIABLE FROM ARTEFACTS — trust source script, but worth a quick check

These claims could not be verified from the saved tables or the figure CSV
alone; they depend on the analysis script reading raw NPZs.

- **`paper/sections/scope.tex:102`, `paper/sections/appendix.tex:344`** —
  ChemBERTa "9–17pp accuracy drop" at λ=300 across six datasets.
  Source: `scripts/analyze_chemberta_heldout.py` (per-dataset ERM accuracy
  not in `tables/chemberta.tex`).
- **`paper/sections/scope.tex:107`, chemberta caption** —
  ChemBERTa "cuts churn 15–82%" at λ=10. From per-dataset means in the table
  the relative range is 15–80%; the script-emitted 82% likely uses a
  paired-bootstrap CI bound. Probably acceptable; verify in
  `scripts/analyze_chemberta_heldout.py`.
- **`paper/sections/scope.tex:106`, chemberta caption** —
  ChemBERTa "within 2pp of ERM". Per-dataset ERM accuracies not in the
  table; trust source script.
- **`paper/figures/fig_convergence.pdf`** — 48–84% / 58–100% / MOF-floor
  48–58% are derived in `scripts/make_fig_convergence.py`; no CSV is saved
  alongside the PDF, so they were verified only against the appendix prose.
  Recommend dumping a CSV from that script for traceability.

---

## VERIFIED — these match their source

(Counts only; full lists in subagent transcript.)

- **Fragility magnitudes table**: 9/9 datasets match `fragility_magnitudes.tex`
  for argmax churn 8–22%, |Δacc| 1.3–4.2pp, ERM accuracies, N values, sym-KL.
- **Main table deltas**: MC dropout −7% to +13%; param-side combined −21% to
  +12.5% relative; absolute −4.6 to +1.8pp.
- **Bagging-K=5**: 40–54% relative reduction (replicated 40.4–54.1%);
  final churn 4–11% (3.9–11.3%).
- **Twin-bootstrap**: median 60% reduction (replicated median 60.9%);
  range 16–92% (16.1–92.5%); final churn 1–14% (0.6–14.1%).
- **Friedman test**: χ² ≈ 38.4, p ≈ 3×10⁻⁷ (replicated χ²=38.75, p=2.66e-7);
  mean ranks Twin 1.44 / Bag5 1.67 / Bag2 2.89 / DE ~4.6 / ERM ~5.0 / MC 5.33.
- **Sym-KL reduction**: bagging 5–10× (5.9–9.1×); twin further median ~8×
  (median 7.5×).
- **Bootstrap-overlap math**: 0.63² ≈ 0.40 ✓.
- **Compute-matched wins**: 8/9 vs Bag-K=2; 5/8 vs Bag-K=5; trails on
  DILI/Pgp/TADF — verified in `main.tex` and `fig1_forest.csv`.
- **Architecture transfer**: GIN 52% / −11.7pp / +3.5pp / 15pp drop /
  48% at λ=10 — verified.
- **Waterbirds**: 27pp accuracy collapse at λ=300; 52% churn cut at λ=10;
  λ=10 within 1pp ERM — verified.
- **Regression**: twin 39–42%; Bag-K=2 29–32%; Bag-K=5 55–58% — verified.
- **Triage**: 48–84% (K=2) and 58–100% (K=10) at top-30% — verified.
- **Hyperparameters**: λ=300 BACE-MLP; λ=10 elsewhere; K=5; T=20 — verified.

---

## How to double-check yourself

For each INCONSISTENT item:
1. Open the file:line in the claim.
2. Open the source file (table .tex or figure CSV) and locate the row/column.
3. Compute the range or ratio by hand.

For each UNVERIFIABLE item:
1. Open the analysis script.
2. Run it; the printed numbers should match the prose (or generate a CSV).
