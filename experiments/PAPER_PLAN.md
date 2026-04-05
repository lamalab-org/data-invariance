# Plan for a Strong NeurIPS Paper

## Honest assessment: what do we actually have?

We have a method that combines known ingredients (ERM loss scoring, upweighting,
V-REx) with one genuinely novel idea (permutation test for adaptive λ) and one
genuinely novel output (stability scores from V-REx disagreement).

The method works well on vision benchmarks (beats Group DRO on Waterbirds without
labels) but modestly on chemistry (small effect on TADF). The stability scores
show an interesting signal (ERM is anti-predictive, ours isn't) but the AUROCs
are low.

**A mediocre paper would report these numbers and call it a day.**

**A strong paper needs to identify what's genuinely new and build the whole
narrative around that, with the experimental results as supporting evidence.**

## What is genuinely new?

After honest reflection, the novel contributions are:

1. **The permutation test for invariance penalty calibration.** This is applicable
   to ANY method that uses environment-based regularisation (IRM, V-REx, FISH,
   DRO). It answers "should I even apply this penalty?" — a question no prior
   work addresses. This is a standalone contribution.

2. **Stability scores as dataset-composition sensitivity.** Not "the model is
   uncertain" but "the model's confidence would change if the training set were
   composed differently." This is a new epistemic uncertainty type. The finding
   that ERM is *anti-calibrated* (confidently wrong on fragile examples) is
   striking.

What is NOT novel (and should not be claimed as such):
- Environment discovery from ERM losses (≈ JTT, GEORGE)
- V-REx training (Krueger et al. 2021)
- Upweighting minority examples (JTT, focal loss, etc.)

## The strongest possible framing

**Don't frame this as "a new method for spurious correlations."** That space is
crowded (JTT, DFR, CnC, GEORGE, SSA, AFR, ...) and we won't beat DFR's 93%.

**Frame this as: "When and how much should you regularise for invariance?"**

The core insight: invariance penalties (V-REx, IRM) are powerful but fragile.
Too much → over-constrained, hurts performance. Too little → no effect. The
right amount depends on whether the discovered environments carry real signal.
No one has a principled answer to this. We do: the permutation test.

**Title candidates:**
- "Adaptive Invariance: A Permutation Test for Environment Quality in Robust Training"
- "Test Your Environments: When Invariance Penalties Help and When They Hurt"
- "From ERM Residuals to Calibrated Invariance"

## Paper structure for maximum impact

### Introduction (1.5 pages)

Start with the practical problem, not the theory:

"You have a model that works well on your training data but fails on a new
hospital / a new chemical series / a new demographic. You suspect spurious
correlations but don't know what they are. Methods like V-REx and IRM can
help — but they require *environments* you don't have, and a *penalty
strength* you can't tune without OOD data you don't have either."

"We show that:
(a) environments can be discovered from the ERM's own mistakes,
(b) the quality of those environments can be assessed via a permutation test,
(c) the penalty strength can be automatically calibrated from (b),
(d) the resulting model provides per-example *stability scores* that predict
    which predictions are fragile to dataset composition."

### Method (2.5 pages)

**2.1 Environment discovery** (0.5 page) — brief, not the main contribution.
Train ERM, score by loss, median split, upweight. Cite JTT, GEORGE.
Acknowledge this is not new — the new parts are 2.2 and 2.3.

**2.2 The permutation test** (1 page) — the core contribution.
- Define risk variance under true assignment vs random permutations
- The signal ratio: actual_rv / permuted_rv
- Proposition: if signal_ratio < 1 + ε, the discovered environments are
  indistinguishable from random splits and V-REx reduces to ERM in expectation
- Algorithm box: discovery → permutation test → adaptive λ → V-REx training

**2.3 Stability scores** (0.5 page) — the second contribution.
- Definition: per-example loss from the V-REx model as a stability indicator
- Why this is different from standard uncertainty: not "model is unsure" but
  "model's confidence depends on which data it saw"
- Connection to algorithmic stability (Bousquet & Elisseeff 2002)

**2.4 Analysis: when does V-REx help over upweighting?** (0.5 page)
- V-REx forces equal *environment-level* risk. Upweighting (JTT) forces high
  *example-level* weight. The model can accommodate upweighted examples without
  changing its feature representation (memorise). It cannot accommodate equal
  environment risk without changing features (must generalise).
- The permutation test detects when this distinction matters.

### Experiments (3 pages)

**The experiments should answer three questions:**

**Q1: Does adaptive V-REx work across domains without tuning?** (Table 1)
Same config, three domains. Show signal_ratio correctly identifies when to
apply V-REx. Include CelebA as a fourth domain.

**Q2: What does the permutation test look like?** (Figure 1)
Three panels: actual risk variance vs permuted distribution for each dataset.
Shows at a glance why the method applies V-REx on Waterbirds but not TADF.

**Q3: Are stability scores useful?** (Figure 2)
The "killer figure": 2D scatter of (ERM confidence, our confidence) for test
examples, coloured by "flips under OOD shift." Show that ERM puts flippers
in the high-confidence region (anti-calibrated) while our model spreads them
to the low-confidence region (correctly calibrated).

**Supporting experiments (can go in appendix):**
- Ablation (upweight alone, V-REx alone, both)
- Correlation sweep on CMNIST (0.5 → 0.99)
- Multi-spurious CMNIST
- Iterative refinement on continuous features
- TADF chemistry application

### What we need to do before submission

**Essential (week 1):**
- [ ] Multi-seed Waterbirds (5 seeds) — confirm the headline number
- [ ] CelebA dataset — standard benchmark, expected from reviewers
- [ ] Correlation sweep on CMNIST: train_corr ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99}
      Plot: x=correlation, y=OOD accuracy for ERM/JTT/Ours. Show graceful degradation.
- [ ] The confidence scatter figure (Figure 2 above)
- [ ] The permutation test figure (Figure 1 above)
- [ ] DFR baseline numbers on Waterbirds

**Essential (week 2):**
- [ ] Write proposition about permutation test (even informal: "if signal_ratio < c
      then the V-REx gradient is dominated by noise and the penalty is harmful")
- [ ] Draft the paper
- [ ] One-command reproduction scripts for every table/figure
- [ ] Clean up codebase, ensure all tests pass

**Nice to have:**
- [ ] MultiNLI / CivilComments (NLP benchmark — shows generality beyond vision)
- [ ] Formal stability bound
- [ ] More chemistry datasets
- [ ] Feature-space mixup for images

## What would make a reviewer reject this?

1. **"This is just JTT + V-REx."** → Counter: the permutation test is novel and
   applicable to any invariance method. JTT without V-REx doesn't work (Table 2).
   V-REx without calibration can hurt (TADF results).

2. **"The numbers don't beat DFR."** → Counter: DFR needs a group-balanced
   validation set. We need nothing. Different assumptions, different use cases.

3. **"Single-seed results."** → Counter: multi-seed runs (must complete before
   submission).

4. **"Why not just tune λ on a validation set?"** → Counter: which validation
   metric? Worst-group accuracy requires group labels. Overall accuracy selects
   for the shortcut. The permutation test is the only principled option when
   you don't have group labels.

5. **"The stability scores are weak (AUROC 0.56)."** → Counter: the comparison
   is to ERM's 0.32 (anti-predictive). The relative improvement is what matters.
   Also: show the confidence scatter figure — the visual is more compelling than
   the number.

## The one-sentence pitch

"We show when and how much to regularise for invariance, using a permutation
test that requires no environment labels, no group annotations, and no
hyperparameter tuning — and as a byproduct, the model produces per-example
stability scores that predict which predictions are fragile to dataset
composition."
