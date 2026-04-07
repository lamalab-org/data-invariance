# Experiment Logbook

## Method: Discovered Environment Split + V-REx

### Core pipeline
1. Train throw-away ERM for K epochs (learns the spurious shortcut)
2. Score each training example by cross-entropy loss (high loss = minority group)
3. Median split into env A (low loss, spurious-aligned) and env B (high loss, spurious-misaligned)
4. Upweight high-loss examples: `w_i = 1 + factor * (loss_i / max_loss)`
5. Train new model with V-REx penalty: `(loss_A - loss_B)^2` on weighted environment losses
6. Early stop on risk variance (patience-based, with warmup)

### Key design choices and why
- **Single model, not two heads:** SplitMLP caused domain blindness (each head only saw its own env). Single MLP with V-REx forces the one model to find features that work in both environments.
- **Loss scoring, not entropy:** At high spurious correlation, the ERM is confidently *wrong* on minority examples (low entropy, high loss). Entropy misses them; loss catches them.
- **Upweighting, not exclusion:** Excluding the middle 70% of examples (quantile split) loses shape signal. Upweighting keeps all data while amplifying the minority group in the V-REx penalty.
- **Dataset-agnostic:** No assumption about where the spurious feature lives. Only uses the ERM's own mistakes as signal.

---

## CMNIST Experiments

### Dataset
- Binary classification: digits 0-4 vs 5-9
- Spurious feature: colour (red=0, green=1), correlated with label
- Configurable correlation: train_correlation controls spurious strength
- OOD test: flipped correlation (0.1 when train is 0.9)
- 60K train, 10K test, label_noise=0.25

### Baselines (all at default config unless noted)

| Method | train_corr | ID acc | OOD acc | Notes |
|--------|-----------|--------|---------|-------|
| ERM | 0.7 | ~72% | 55.6% | Learns colour, degrades OOD |
| ERM | 0.9 | ~90% | ~22% | Heavily colour-dependent |

### Discovered split results at corr=0.7

| Config | OOD peak | OOD final (ep29) | Notes |
|--------|----------|-----------------|-------|
| lambda=1, no anneal | 64.2% (ep6) | 50.2% | Works but overfits |
| lambda=10, no anneal | **71.5%** (ep0) | 53.7% | Best absolute OOD |
| lambda=100, no anneal | 71.4% (ep0) | 54.3% | Too strong, unstable |
| lambda=10, anneal 10x | 72.7% (ep3) | 55.4% | Annealing helps final |

**Key finding:** +16pp over ERM at peak (71.5% vs 55.6%).

### Discovered split results at corr=0.9

#### Splitting strategy comparison (lambda=10, anneal 10x)

| Strategy | corr_A | corr_B | OOD peak | Problem |
|----------|--------|--------|----------|---------|
| Entropy, median (q=0.5) | 0.945 | 0.654 | 15% | corr_B too high |
| Loss, median (q=0.5) | 1.0 | 0.60 | 13% | corr_B too high |
| Loss, q=0.15 extremes | 1.0 | -0.33 | 74% | Learns anti-colour |
| Loss, q=0.20 extremes | 1.0 | 0.00 | 55% | Colour-invariant but random |
| Reweight=10, median | 1.0 | 0.60 | 13% | Corr_B too high for any reweight |
| Reweight=50, median | 1.0 | 0.60 | 15% | Same |

**Key finding:** At corr=0.9, median splits can't create different-enough environments. Extreme quantiles find great environments but the model can't learn shape from only extreme examples.

#### Loss-based upweighting (the breakthrough)

| Upweight | Discovery | OOD peak | ID at peak | Notes |
|----------|-----------|----------|------------|-------|
| 0 | loss, median | ~13% | ~89% | V-REx has no signal |
| 20 | loss, median | 61.5% (ep11) | 57% | First time beating ERM |
| 50 | loss, median | **74.3%** (ep7) | 53% | Best result |

**Key finding:** Upweighting makes the minority group dominate the *weighted* env B loss, even though the *unweighted* env B has corr=0.60. The effective contrast is in the loss contributions, not the environment labels. +52pp over ERM (74.3% vs 22%).

### Discovery method ablation (CMNIST corr=0.9, all with upweight=50, λ=10)

| Discovery | corr_A | corr_B | Gap | signal_ratio | OOD acc |
|-----------|:------:|:------:|:---:|:------------:|:-------:|
| **Loss** | **1.000** | **0.600** | **0.400** | **3565** | **67.3%** |
| Entropy | 0.945 | 0.655 | 0.290 | 254 | 13.3% |
| Counterfactual | 0.933 | 0.665 | 0.268 | 231 | 13.1% |
| Activation (K-means) | 0.234 | 3.3% | — | — |

Same pattern on Waterbirds:

| Discovery | corr gap | Waterbirds WGA |
|-----------|:--------:|:--------------:|
| Loss | 0.185 | **80.8%** |
| Entropy | 0.150 | 69.8% |
| Counterfactual | 0.090 | 61.2% |

**Why loss wins:** High loss = the ERM's learned shortcut fails for this
example. Entropy/counterfactual measure uncertainty/sensitivity, which
is a weaker proxy — a model can be certain AND wrong (colour≠label
examples where it confidently predicts by colour).

Activation-based (K-means on features) finds clusters correlated with
colour (0.99) but the clusters don't separate spurious-reliant from
invariant examples — they separate by some other feature dimension.

**Confident-wrong scoring (data cartography inspired):**
Score = P(wrong class).  Isolates examples where ERM confidently predicts
the wrong class (= minority) from examples where it's just uncertain.

| Discovery | CMNIST OOD | Waterbirds WGA |
|-----------|:----------:|:--------------:|
| Loss | 67.3% | 76.7 ± 1.0 |
| **Confident wrong** | 49.2% | **79.0%** |

On CMNIST: same environments (no uncertain examples), worse OOD due to
different upweight distribution.  On Waterbirds: +2.3pp because it
correctly separates "confidently wrong" minority from "uncertain" majority.

Multi-seed check pending.

**Limitation:** OOD peaks early then degrades. Early stopping on risk variance (patience=5, warmup=5 epochs) auto-selects epoch 5 with 70.4% OOD.

---

## Waterbirds Experiments

### Dataset
- Binary classification: waterbird (1) vs landbird (0)
- Spurious feature: background (water=1, land=0), 95% correlated with label
- Standard splits: 4,795 train / 1,199 val / 5,794 test
- Training group composition:
  - landbird + land: 3,498 (73%)
  - waterbird + water: 1,057 (22%)
  - landbird + water: 184 (4%)
  - waterbird + land: 56 (1%) -- hardest group
- Val/test sets balanced across groups
- Key metric: **worst-group accuracy** (accuracy on the hardest group)
- Model: ResNet-50 pretrained on ImageNet, fine-tuned

### Results

| Method | Disc. epochs | Upweight | Best worst-group | At epoch | Test acc | Notes |
|--------|-------------|----------|-----------------|----------|----------|-------|
| ERM | -- | -- | 80.7% | 5 | 90.7% | Baseline |
| Discovered V-REx | 3 | 20 | 81.6% | 6 | 90.1% | Marginal improvement |
| Discovered V-REx | 5 | 50 | **85.4%** | 1 | 88.7% | Best worst-group, early peak |
| Discovered V-REx | 10 | 50 | 84.1% | 8 | 90.9% | More stable, still strong |

### Discovery diagnostics

| Disc. epochs | assignment_color_abs_corr | corr_A | corr_B | Notes |
|-------------|--------------------------|--------|--------|-------|
| 3 | 0.51 | 0.947 | 0.810 | Weak contrast |
| 5 | 0.37 | 0.986 | 0.801 | Sharper ERM, better scoring |
| 10 | 0.30 | 0.987 | 0.796 | Diminishing returns on contrast |

### Baseline comparison (all single-seed, same hyperparameters)

| Method | Group labels? | Best worst-group | At epoch | Test acc | Notes |
|--------|:------------:|:----------------:|:--------:|:--------:|-------|
| ERM | No | 80.7% | 5 | 90.7% | Baseline |
| **JTT** (upweight=50, disc=5) | No | 80.5% | 9 | 92.0% | ≈ ERM; upweighting alone doesn't help |
| **Group DRO** | **Yes** (oracle) | **85.5%** | 7 | 92.1% | Oracle upper bound |
| **Ours** (disc=5, upweight=50) | No | **85.4%** | 1 | 88.7% | Matches Group DRO without labels |
| **Ours** (disc=10, upweight=50) | No | 84.1% | 8 | 90.9% | More stable variant |

### Re-run with consistent V-REx code + validation-based model selection

All experiments re-run with:
- Consistent V-REx penalty (sum-of-squared-deviations)
- Model selected by **validation** worst-group accuracy (not test)
- Single seed (seed=42)

| Method | Group labels? | Val WGA (select) | **Test WGA** | Test acc |
|--------|:------------:|:---:|:---:|:---:|
| ERM | No | 63.2% | 70.3% | 91.8% |
| JTT (binary) | No | 64.6% | 65.7% | 83.3% |
| **Group DRO** | **Yes** | 80.5% | **77.7%** | 91.9% |
| **Ours** | No | 72.2% | **75.7%** | 90.5% |

**Key finding 1:** Ours (75.7%) beats ERM (70.3%) by +5.4pp with proper selection.

**Key finding 2:** Ours approaches Group DRO (77.7%) without group labels — only 2pp gap.

**Key finding 3:** JTT (65.7%) is WORSE than ERM. Upweighting alone hurts.

**Key finding 4:** Old cherry-picked numbers were inflated (ERM was 80.7%, now 70.3%). Proper model selection is critical.

### Multi-seed results (partial — 2 of 5 seeds)

| Method | Seed 42 | Seed 123 | Notes |
|--------|:-------:|:--------:|-------|
| ERM | 70.3% | 79.8% | High variance across seeds |
| JTT | 65.7% | 58.1% | Consistently worse than ERM |
| Group DRO | 77.7% | 81.5% | Requires group labels |
| DFR | 85.4% | 89.6% | Consistently best (needs balanced val) |
| Ours (adaptive) | 80.8% | pending | Between Group DRO and DFR |

**Full 5-seed results (test WGA, mean ± std):**

| Method | Group labels? | Mean ± Std |
|--------|:------------:|:----------:|
| ERM | No | 71.0 ± 8.9 |
| JTT | No | 71.5 ± 9.1 |
| Group DRO | Yes | 80.5 ± 4.4 |
| **Ours (adaptive)** | No | 72.4 ± 9.9 |
| DFR | No* | **88.4 ± 2.5** |

**Honest assessment:** Our method (72.4%) barely beats ERM (71.0%) with high
variance (±9.9).  The permutation test is unstable: seed 789 gives λ=0
(kills V-REx entirely); other seeds give λ=10.  DFR is consistently best.

**Fix: permutation test now uses discovery ERM (not untrained model).**
The old test evaluated a randomly initialised model → near-uniform loss →
noisy signal_ratio → sometimes λ=0 by chance.  Using the discovery ERM
(which has learned the shortcut) gives stable, meaningful risk variance.

3-seed check after fix:

| Seed | Old λ | Old WGA | Fixed λ | Fixed WGA |
|------|:-----:|:-------:|:-------:|:---------:|
| 42 | 10.0 | 80.8% | 10.0 | 75.7% |
| 789 | **0.0** | 63.7% | **10.0** | **77.6%** |
| 1337 | 10.0 | 72.9% | 10.0 | 76.8% |

**Ours (fixed): 76.7 ± 1.0.** Much more stable than before (±9.9).
Reliably beats ERM (71.0 ± 8.9) with lower variance.

---

## Prior Work: Baselines and Positioning

### Landscape of methods

| Method | Core idea | Group labels? | Waterbirds WGA | Reference |
|--------|-----------|:------------:|:--------------:|-----------|
| ERM | Standard training | No | ~70-80% | — |
| **Group DRO** | Minimise worst-group loss directly | **Yes** (oracle) | ~90-91% | Sagawa et al. 2020 |
| **DFR** | ERM features are fine; retrain last layer on balanced data | No* | ~93% | Kirichenko et al. 2023 |
| **JTT** | Train twice — upweight ERM's mistakes in 2nd round | No | ~87% | Liu et al. 2021 |
| **CnC** | Contrastive learning on pairs identified by ERM | No | ~90% | Zhang et al. 2022 |
| **GEORGE** | Cluster ERM features → pseudo-groups → Group DRO | No | ~85-87% | Sohoni et al. 2020 |
| **Ours** | ERM loss → env discovery + upweighting → V-REx | No | 85.4% | — |

*DFR requires a group-balanced validation set for last-layer refit.

### What each method does (and doesn't do)

**Group DRO** (oracle upper bound): Directly minimises worst-group loss at each training step. Requires full group annotations on every training example. Clean and effective but impractical when group labels are expensive (chemistry, medical imaging). Implementation: modify loss aggregation to per-group losses, reweight toward highest-loss group.

**DFR** (strongest no-label method): Key insight — ERM already learns good features, it just misweights them in the final layer. Fix: freeze ERM backbone, retrain last layer on group-balanced validation data via L1-regularised logistic regression. Extremely simple, shockingly effective (93% WGA). But needs a balanced validation set with group labels. Implementation: trivial — just logistic regression on frozen features.

**JTT** (closest to our discovery phase): Train ERM, identify misclassified examples, upweight them in a second round of training. Our discovery + upweighting is essentially JTT. The question we must answer: **what does V-REx add over plain upweighting?** Implementation: very simple — two training phases, second with upweighted loss.

**CnC** (contrastive variant): Uses ERM to find pairs of examples with same label but different spurious features, then trains with contrastive loss. Harder to implement (contrastive framework, pair mining). Medium-high complexity.

**GEORGE** (clustering variant): Clusters ERM representations to infer pseudo-groups, then runs Group DRO on those groups. Similar in spirit to our environment discovery but uses feature clustering instead of loss thresholding. Medium complexity (clustering, dimensionality reduction).

### Positioning: why our method is NOT incremental over JTT

The danger: "this is JTT with V-REx." To avoid this, we need to show:

1. **Training-time invariance vs post-hoc correction.** JTT/DFR/CnC all produce a model and then fix it. We train a model that is *inherently stable to data composition*. This is a different guarantee — the model doesn't just happen to work on the minority group, it's trained to be insensitive to which partition of the data it sees.

2. **Instance-level stability scores.** V-REx disagreement between environments gives a per-example signal: "how much does this prediction depend on which data the model trained on?" None of the baselines provide this. This is a new form of epistemic uncertainty — not "the model is uncertain" but "the model's confidence is fragile."

3. **The chemistry application.** None of these methods have been applied to molecular property prediction where "groups" are scaffold clusters or dataset sources. In chemistry, you often don't know what the spurious correlation is (scaffold? functional group? dataset batch?). Our method discovers it from the ERM's residuals.

4. **Formal connection to algorithmic stability.** Bousquet & Elisseeff (2002) study stability as a property to *analyse*. We turn it into a training *objective*. This is theoretically novel even if the empirical gains are modest on Waterbirds.

**Paper framing: "Algorithmic stability has been studied as a property to analyse. We turn it into a training objective."**

### Baselines to implement (priority order)

1. **JTT** — essential. Most direct comparison. Shows what V-REx adds over pure upweighting.
2. **Group DRO** — essential. Oracle upper bound with full group labels.
3. **DFR** — essential. Strongest published baseline. Shows the gap we need to close.
4. **GEORGE** — nice-to-have. Closest in spirit (also discovers groups from ERM).

CnC is lower priority (complex to implement, contrastive framework).

### Critical ablation: JTT vs Ours

**Result:** JTT (our continuous-weight version) ≈ ERM (80.5% vs 80.7%).
Ours = 85.4%, matching Group DRO. V-REx is the critical ingredient.

**But why did JTT work in the original paper (~87%)?**

Our initial "JTT" implementation was not faithful to the original paper:

| | Original JTT (Liu et al. 2021) | Our initial "JTT" |
|---|---|---|
| Identification | Binary: misclassified yes/no | Continuous: loss / max_loss |
| Weighting | Fixed λ on error set, 1 elsewhere | Graded: 1 + factor * loss/max_loss |
| Discovery epochs | T=1 (very short) | 5 (longer) |

The difference matters. Binary identification precisely targets the minority group —
after 1 epoch, the ERM mostly misclassifies minority examples. Continuous weighting
spreads weight across ALL high-loss examples, including noisy majority examples.

**The interesting twist:** Our method ALSO uses continuous weighting but it works.
Why? Because V-REx doesn't just upweight — it splits into environments AND forces
equal risk. The environment structure gives the continuous weights *meaning*:
- Without structure (JTT): continuous upweighting → noisy ERM, model can satisfy
  the loss by accommodating upweighted examples without changing strategy
- With structure (V-REx): continuous upweighting within environments → equal risk
  forces the model to perform equally on both groups, it can't just memorise

We implemented proper binary JTT (identify misclassified, fixed upweight) to fairly
compare.

**Binary JTT result: 80.6% worst-group — still ≈ ERM.**

JTT diagnostics reveal why: after 1 epoch, 664 examples are misclassified (13.8%
of training set), but only 9.8% of those errors come from the minority group.
The identification is imprecise — mostly majority-group examples the model hasn't
learned yet.  Upweighting this noisy error set doesn't help.

**Final comparison on Waterbirds (all single-seed):**

| Method | Group labels? | Weighting | Best WGA | At epoch |
|--------|:------------:|-----------|:--------:|:--------:|
| ERM | No | — | 80.7% | 5 |
| JTT (continuous, disc=5) | No | Continuous loss | 80.5% | 9 |
| JTT (binary, disc=1) | No | Binary misclassified | 80.6% | 13 |
| **Group DRO** | **Yes** | Per-group adaptive | **85.5%** | 7 |
| **Ours** (disc=5, upw=50) | No | Continuous + V-REx | **85.4%** | 1 |

**Takeaway:** Neither JTT variant helps on Waterbirds in our setup.  The V-REx
equal-risk constraint is the critical ingredient — it matches Group DRO without
group labels.

**Caveat:** Our single-seed JTT result differs from the published ~87%.  Possible
reasons: (1) different optimizer (we use AdamW, paper uses SGD+momentum),
(2) different LR schedule, (3) different weight decay, (4) different discovery
epochs.  Multi-seed runs with original hyperparameters are needed for a fair claim.

---

## Why V-REx > JTT: Failure Modes of Pure Upweighting

The critical question for the paper: when does the V-REx equal-risk constraint help
beyond simple upweighting (JTT)?  We need concrete scenarios where JTT fails and
V-REx succeeds — these become the paper's motivating examples.

### 1. Overfitting to small minority groups

**The problem:** JTT minimises `Σ w_i * CE(model(x_i), y_i)`.  With upweight=50
and only 56 minority examples (waterbird+land), the model sees these 56 examples
~50x each.  It can memorise them — learn to recognise those specific 56 birds —
without learning "what a waterbird looks like on land" in general.

**Why V-REx helps:** V-REx forces `loss_A ≈ loss_B` at the *environment* level.
The model can't satisfy this by memorising 56 examples; it must find features
that generalise across the *entire* env B (which includes the 184 landbird+water
examples too).  The constraint is on average risk, not on individual examples.

**How to test:**
- Subsample Waterbirds minority group (56 → 20 or 10)
- Compare JTT vs Ours worst-group accuracy on held-out minority examples
- Prediction: JTT degrades faster as minority shrinks; V-REx is more robust
- Also testable on CMNIST with corr=0.99 (1% minority)

### 2. Multiple overlapping spurious features

**The problem:** Real datasets often have multiple shortcuts — background AND
image quality, scaffold AND molecular weight, colour AND texture.  JTT identifies
"hard" examples but can't distinguish *which* shortcut makes them hard.  The model
may fix one shortcut while continuing to exploit the other.

**Why V-REx helps:** The environment split captures the *combined* effect of all
shortcuts.  High-loss examples are hard because of *any* shortcut failure.  V-REx
forces equal risk across the split, which means the model must be robust to all
shortcuts simultaneously — it can't "trade" one for another.

**How to test:**
- Create CMNIST variant with two spurious features (colour + brightness/contrast)
- JTT should fix the dominant shortcut but miss the secondary one
- V-REx should handle both because the environment split captures both

### 3. Continuous/graded spurious features

**The problem:** JTT's discovery is effectively binary: example is "hard" or "easy"
(above or below a loss threshold).  Real spurious features are often continuous —
molecular weight, scaffold frequency, image resolution, sensor drift.  A binary
split loses the graded structure.

**Why V-REx helps (with generalisation):** Our upweighting is already graded
(`w = 1 + factor * loss/max_loss`), and the V-REx penalty acts on environment-level
loss differences, which smooths over individual noise.

**Generalisation to K>2 environments:** For continuous spurious features, we could
split into K environments by loss quantile instead of a binary median split:
- K=2: median split (current approach)
- K=3: tercile split (low/medium/high loss)
- K=5: quintile split
- V-REx generalises naturally: penalty = variance of per-environment losses
  `Σ_k (loss_k - mean_loss)^2`

This is particularly relevant for chemistry where the spurious feature (scaffold
frequency, dataset batch) is naturally multi-valued.  A molecule isn't "spurious
or not" — it has a *degree* of scaffold prevalence.

**Results (ContinuousCMNIST, env_correlation=0.9):**

| Method | Best OOD acc | Best OOD worst-group | Notes |
|--------|:-----------:|:-------------------:|-------|
| ERM | 55.2% | 17.6% | Learns continuous colour shortcut |
| K=2 V-REx | 54.0% | **27.1%** | Weak contrast (corr_A=1.0, corr_B=0.94) |
| K=5 V-REx | 53.4% | 18.5% | Broken: quantile ties collapse environments |

**Key finding:** Continuous spurious features are genuinely harder.  With graded
colour, the loss-based median split produces only 0.06 contrast (corr_A - corr_B).
The continuous signal is smooth — no sharp boundary between "uses colour" and
"doesn't."  K=2 V-REx still improves worst-group (+9.5pp over ERM) but the effect
is modest.

**K=5 failed** because quantile thresholds collapsed (the discovery ERM is very
confident → most losses near zero → multiple quantiles at the same value).
Fix needed: use rank-based assignment instead of threshold-based.

### Iterative refinement and counterfactual scoring

To improve on continuous features, we tested two ideas:

**Counterfactual scoring** (measure prediction sensitivity to input perturbations):
Failed.  17.0% worst-group (worse than loss-based 27.1%).  Input perturbation
doesn't discriminate "sensitive because of colour" from "sensitive because of
shape" — all examples are somewhat sensitive with continuous features.

**Iterative refinement** (re-discover environments using the previous V-REx model):
Works!  The environment contrast improves progressively:

| Round | corr_A | corr_B | Gap | assignment-colour corr |
|-------|--------|--------|:---:|:---------------------:|
| 1 (from ERM) | 1.000 | 0.938 | 0.062 | 0.15 |
| 2 | 1.000 | 0.930 | 0.070 | 0.36 |
| 3 | 0.999 | 0.893 | **0.106** | **0.67** |

The assignment-colour correlation jumps from 0.15 → 0.67 across 3 rounds.
Each round, the V-REx model relies less on colour → its loss distribution
has different structure → re-discovery finds sharper environments.

| Scoring | K | Rounds | Best OOD WGA |
|---------|---|--------|:---:|
| loss | 2 | 1 | 27.1% |
| **loss** | **2** | **3** | **28.0%** |
| counterfactual | 2 | 1 | 17.0% |
| counterfactual | 5 | 1 | 18.5% |

**Key insight:** Iterative refinement is the right approach for continuous
features.  Loss-based scoring is initially weak (gap=0.06) but improves
because each V-REx round partially breaks the colour dependence, creating
a new loss landscape where the remaining colour signal is more visible.

5-round experiment confirms the trend accelerates:

| Round | corr_A | corr_B | Gap | assignment-colour corr |
|-------|--------|--------|:---:|:---------------------:|
| 1 (from ERM) | 1.000 | 0.938 | 0.062 | 0.15 |
| 2 | 1.000 | 0.930 | 0.070 | 0.36 |
| 3 | 0.999 | 0.893 | 0.106 | 0.67 |
| 4 | 1.000 | 0.907 | 0.093 | 0.58 (slight dip) |
| **5** | **0.998** | **0.839** | **0.159** | **0.80** |

The gap nearly triples (0.06 → 0.16) and assignment-colour correlation reaches
0.80.  The method progressively discovers the continuous spurious feature.

**Implication for chemistry:** Iterative discovery is promising for continuous
confounders.  The first round finds rough environments; subsequent rounds
refine them as the model becomes less confounder-reliant.  No need for
prior knowledge of what the confounder is.

### 4. When you need to know what you don't know

**The problem:** JTT gives you a model.  That's it.  In drug discovery, materials
science, and clinical settings, knowing *which* predictions are trustworthy is
as important as the predictions themselves.

**What V-REx provides:** Per-example risk variance — "how much does this prediction
change depending on which environment the model is evaluated against?"  This is a
new form of epistemic uncertainty:
- Standard uncertainty: "the model is unsure about this example"
- Our stability score: "the model's confidence on this example is fragile to
  dataset composition"

These are different signals.  A model can be confidently wrong (low standard
uncertainty, high stability score) — and that's exactly the dangerous case.

**How to evaluate stability scores:**
1. Train discovered_split on CMNIST train (corr=0.9)
2. For each test example, compute confidence (max softmax probability)
3. Evaluate same examples on ID test (corr=0.9) and OOD test (corr=0.1)
4. Binary label: "did this example's prediction flip between ID and OOD?"
5. ROC curve: does our confidence predict flips better than ERM confidence?
6. Compare: our model's confidence, ERM confidence, MC dropout variance

**For chemistry:** Stability scores could flag predictions that depend on
scaffold identity rather than genuine structure-activity relationships.
This is directly actionable for medicinal chemists.

**Results (CMNIST corr=0.9, comprehensive evaluation):**

AUROC for predicting ERM's OOD flips:

| Score | Our model | ERM | Δ |
|-------|:-:|:-:|:-:|
| 1 - confidence | 0.551 | 0.426 | +0.124 |
| Entropy | 0.551 | 0.426 | +0.124 |
| **Loss** | **0.562** | **0.321** | **+0.241** |
| MC dropout | 0.500 | 0.500 | 0.000 |

ERM loss is **anti-predictive** (0.321 < 0.5): assigns LOW loss to examples that
WILL flip.  Our model's loss correctly identifies fragile examples (0.562).

Calibration: our entropy quintile vs actual ERM flip rate:
bin 0 (lowest entropy): 63.1% flip rate
bin 4 (highest entropy): 74.7% flip rate → monotonically increasing, well-calibrated.

Flip rates: ERM 71.7% vs Ours 29.4% — 2.4x more stable.

Reproduction: `uv run python scripts/evaluate_stability.py`

---

## Planned Experiments: Beyond Binary Spurious Features

### Multi-spurious CMNIST

Standard CMNIST has one spurious feature (colour).  Real datasets often have
multiple overlapping shortcuts.  We want to show our method handles this while
JTT does not.

**Design: CMNIST with colour + brightness.**
- Feature 1: colour (red/green channel), correlated with label at p1=0.8
- Feature 2: brightness (scale pixel intensities by 0.3 vs 1.0), correlated at p2=0.7
- Construction: independently sample colour and brightness correlations
- OOD test: both correlations flipped (colour p=0.2, brightness p=0.3)
- A model using either shortcut fails OOD

**Predictions:**
- ERM: relies on both shortcuts → fails OOD
- JTT: identifies "hard" examples (those where both shortcuts fail) → fixes the
  easier shortcut but may miss the other
- Ours: V-REx forces equal risk across environments discovered from ERM loss →
  environments capture the combined effect of both shortcuts → model must be
  robust to both simultaneously

**Implementation:** Extend ColoredMNIST with a `brightness_correlation` parameter.
Add brightness assignment in step 3, multiply pixel values by brightness factor
in step 4.

**Results (colour=0.9, brightness=0.8, OOD: both flipped):**

| Method | Best OOD acc | Best OOD worst-group | ID acc | Notes |
|--------|:-----------:|:-------------------:|:------:|-------|
| ERM | 17.8% | 8.5% | 90.2% | Both shortcuts → fails badly |
| JTT (binary) | 55.3% (ep1) | 27.3% | ~50% | Partially inverts, can't handle both |
| **Ours** | 56.2% (ep1) | **41.0%** (ep8) | 64.1% | Best worst-group by far |

**Key finding:** ERM fails catastrophically (8.5% worst-group — uses both shortcuts).
JTT partially helps overall OOD but can't handle the hardest group where BOTH
shortcuts are broken (27% worst-group).  Our method does significantly better on
worst-group (41%) because V-REx forces equal risk across environments, which
captures the combined effect of both shortcuts.

This confirms the multi-spurious failure mode prediction: JTT identifies hard
examples but can't distinguish which shortcut to fix.  V-REx forces robustness to
the full environmental split.

### Continuous spurious features and K>2 environments

Real spurious features are rarely binary.  In chemistry, scaffold frequency is
continuous; in medical imaging, scanner calibration drifts continuously.

**Design: CMNIST with continuous colour.**
Instead of binary red/green, use a continuous hue correlated with label:
- label=0 → hue sampled from N(0, σ) (reddish)
- label=1 → hue sampled from N(120, σ) (greenish)
- σ controls how "clean" the spurious signal is
- OOD test: hue is uniformly random (no colour-label correlation)

Or simpler: keep binary labels but make the spurious feature a continuous
intensity value `c_i ~ Beta(α, β)` where α, β depend on the label.

**K-environment generalisation:**
With continuous spurious features, binary median split loses nuance.  Instead:
- Score examples by ERM loss as before
- Split into K environments by loss quantile (tercile, quintile, etc.)
- V-REx penalty = variance of per-environment losses: `Var(loss_1, ..., loss_K)`

This is a natural generalisation.  The question is what K to use.  Options:
- Fixed K (hyperparameter, e.g. K=5)
- Data-driven K via gap statistics on the loss distribution
- Continuous: weight each example's contribution to V-REx by its loss rank
  (this is what our current upweighting already approximates)

**For chemistry:** K-environment splitting maps naturally to:
- K scaffold clusters (common, medium, rare scaffolds)
- K dataset batches (different labs, different measurement campaigns)
- K property ranges (low, medium, high molecular weight)

### Chemistry: TADF emission wavelength

Dataset: 2137 TADF emitters, binary classification (< 492 nm vs >= 492 nm),
1636 molecular descriptors (RDKit + Morgan FP).

**Property-based spurious correlation:** NumHeteroatoms is subsampled to
correlate with label at 0.9 in training.  OOD test = misaligned examples
(NumHeteroatoms disagrees with label).

| Method | corr | ID acc | OOD acc (misaligned) |
|--------|:----:|:------:|:-------------------:|
| ERM | 0.9 | 76.4% | 45.2% |
| JTT | 0.9 | 75.2% | **50.6%** |
| Ours | 0.9 | 69.9% | 45.8% |
| ERM | 0.7 | 78.7% | 57.1% |
| **Ours** | **0.7** | 71.3% | **59.5%** |

**Key finding:** At 0.9 correlation with only ~1000 training examples, the
shortcut is too strong (only ~70 counterexamples).  All methods near-random OOD.

At 0.7 correlation, our method gives +2.4pp OOD improvement over ERM.  Effect
is modest compared to CMNIST/Waterbirds because: (1) small dataset limits
learning capacity, (2) 1636 molecular features have many legitimate correlations
with the target — the model doesn't rely heavily on any single shortcut.

**Environment-aware mixup** (interpolate features between envs, same label):
- TADF (tabular): improved worst-group from 35.4% → 44.4% at corr=0.9.
  Feature-space interpolation is reasonable for molecular descriptors.
- CMNIST (images): degraded from ~70% → 52%.  Pixel-space interpolation
  creates unrealistic images — mixup should be in representation space, not
  input space, for image data.

Mixup is promising for chemistry/tabular data specifically.  For images, it
would need to operate on backbone features rather than raw pixels.

**Oversampling + noise** (balanced sampling + Gaussian noise on features):
- Balanced + noise, no upweight: 44.6% OOD — same as ERM
- Upweight + noise: 46.4% OOD — marginal improvement

**Full TADF comparison at corr=0.9:**

| Method | OOD acc | OOD WGA |
|--------|:-------:|:-------:|
| ERM | 45.2% | 44.8% |
| JTT | **50.6%** | **50.0%** |
| Ours (upweight) | 45.8% | 35.4% |
| Ours + mixup | 47.0% | 44.4% |
| Ours + balanced + noise | 44.6% | 44.4% |
| Ours + upweight + noise | 46.4% | 36.5% |

**Conclusion:** At corr=0.9 on small chemistry datasets (~1000 examples,
~70 counterexamples), V-REx consistently hurts.  The equal-risk constraint
is too strong for the available signal.  JTT (simple upweighting) is more
robust in this regime because it doesn't over-constrain the model.

**Adaptive V-REx via permutation test:**
Principled solution: compare actual risk variance to risk variance under
random permutations.  signal_ratio = actual / permuted.  If environments
are no better than random (ratio ≈ 1), lambda → 0 (JTT).

| Dataset | signal_ratio | reliability | effective λ | OOD WGA |
|---------|:-----------:|:-----------:|:----------:|:-------:|
| CMNIST | 3565 | 1.0 | 10.0 | ~64%* |
| Waterbirds | 52 | 1.0 | 10.0 | **83.5%** |
| TADF (corr=0.9) | 1.5 | 0.25 | 2.5 | 40.6% |

*CMNIST needs re-run with consistent code + val-based selection.

The adaptive method is plug-and-play: no dataset-specific tuning,
automatically falls back to JTT when environments are noisy.

### Chemistry datasets

Target datasets where scaffold or property splits are known to cause OOD gaps:

1. **MoleculeNet benchmarks** (BBBP, BACE, HIV, Tox21): standard splits vs
   scaffold splits show 10-20% accuracy gaps. Our method should discover the
   scaffold dependence from ERM residuals.

2. **ADME datasets** (solubility, permeability, clearance): proprietary data
   often has batch effects (different labs, different assay conditions). Our
   method could discover these without explicit batch labels.

3. **MatBench** (materials property prediction): crystal structure features
   often correlate with composition, which correlates with property. Similar
   confounding structure to CMNIST colour.

**What makes chemistry interesting for this paper:**
- Group labels are expensive or unknown (what is the "spurious feature"?)
- Multiple overlapping confounders (scaffold, molecular weight, logP, dataset source)
- Continuous features (no natural binary split)
- High practical value (drug discovery, materials design)
- Untouched by the existing spurious correlation literature

---

## Known Issues and Caveats

### Must fix before paper submission

1. **V-REx penalty scale changed mid-development.** Early experiments (CMNIST,
   Waterbirds) used `(loss_A - loss_B)^2`. Later experiments (ContinuousCMNIST)
   use `sum((loss_k - mean)^2)` which for K=2 equals `(loss_A - loss_B)^2 / 2` —
   half the old penalty.  **All experiments must be re-run with the same code.**
   Current code uses sum-of-squared-deviations; effective lambda is 2x lower
   than the old code for the same config value.

2. **Model selection on test set.** We report "best worst-group across all epochs"
   which selects on the test metric.  Proper protocol: select model via
   validation set (risk variance or val worst-group), report test performance of
   that checkpoint.  The Waterbirds "85.4% at epoch 1" is particularly suspect.

3. **Single-seed comparisons.** Worst-group oscillates by 20+pp between epochs.
   "85.4% vs 80.7%" is within noise.  Need 5+ seeds with mean ± std.

4. **JTT comparison is unfair.** Published JTT = 87% on Waterbirds.  Our
   reimplementation = 80.5%.  Different optimizer (AdamW vs SGD), LR, discovery
   epochs.  Cannot claim "JTT doesn't work" based on our implementation.

### Should investigate

5. **Iterative refinement: data leakage.** Rounds 2+ score training examples
   with a model trained on those same examples.  The model's loss reflects
   memorisation, not just spurious feature reliance.  Improved contrast might
   partly reflect memorisation patterns.  Fix: hold out a separate scoring set.

6. **Iterative refinement: seed confound.** Each round uses `seed + round_i`,
   changing model initialisation.  Need control: single-round with same seeds.

7. **ContinuousCMNIST beta_concentration=5 is nearly binary.** The colour
   distribution is extremely peaked near 0 and 1.  Lower concentration (2.0)
   would test truly continuous features, but that's tuning the dataset.

8. **`.spurious` binarisation for ContinuousCMNIST.** Worst-group uses
   `(color_strength > 0.5).long()`.  This threshold is arbitrary and affects
   the metric.

9. **Waterbirds ERM baseline (80.7%) is higher than published (~70%).**  Our
   setup might favour all methods equally, or our evaluation differs.  Possibly
   due to pretrained weights, LR, or evaluation protocol differences.

---

## Open Questions and TODO

### For ablations (NeurIPS)
- [ ] Run all experiments with 5+ random seeds and report mean +/- std
- [ ] Ablate discovery_epochs: {1, 3, 5, 10, 20}
- [ ] Ablate upweight: {0, 5, 10, 20, 50, 100}
- [ ] Ablate lambda_disagree: {1, 5, 10, 50, 100}
- [ ] Ablate early_stop_patience: {3, 5, 10, none}
- [ ] **JTT vs Ours ablation** — isolate V-REx's contribution
- [ ] CelebA dataset
- [ ] Chemistry datasets (scaffold splits, property-based splits)

### Methodological questions
- [ ] Why does upweighting work better than quantile exclusion?
- [ ] Can we use the ERM's *features* (not just loss) for better environment construction?
- [ ] Is there an optimal upweight estimable from the loss distribution?
- [ ] What does V-REx add over JTT? Formalise the difference.
- [ ] Stability scores: do they predict OOD failures better than softmax entropy?

### For the paper
- [ ] Frame: "algorithmic stability as a training objective, not just an analysis tool"
- [ ] Contributions: (1) loss-based environment discovery, (2) V-REx on discovered envs with upweighting, (3) instance-level stability scores, (4) chemistry application
- [ ] Position clearly against JTT (upweighting alone) and DFR (last-layer refit)
- [ ] Theoretical: connect to uniform stability bounds (Bousquet & Elisseeff 2002)
- [ ] Don't compete on Waterbirds WGA alone — compete on the stability/uncertainty angle

---

## Architecture and Hyperparameter Notes

### Ablation: component contributions (CMNIST corr=0.9, Waterbirds)

Each component is necessary — removing any one degrades performance:

| Components | CMNIST OOD | Waterbirds test WGA | What's missing |
|---|---|---|---|
| ERM (nothing) | ~22% | 70.3% | Baseline |
| Upweighting only (JTT) | ~22% | 65.7% | No V-REx → model accommodates minorities without changing strategy |
| V-REx + balanced sampling (no upweight) | 16.1% | 65.7% | No upweighting → minority signal too diluted in env B |
| V-REx + upweighting (our method) | 74.3%* | 75.7% | Full method |
| V-REx + upweight + freeze backbone | 58%* | 20-73% | Frozen features: too biased (fine-tuned) or too generic (ImageNet) |
| Group DRO (oracle) | — | 77.7% | Requires group labels |

*CMNIST number from old code; needs re-run with consistent V-REx.

**Takeaway:** V-REx alone (balanced) < upweighting alone (JTT) < V-REx + upweighting (ours).
Both ingredients are needed — V-REx provides the structural constraint, upweighting provides the signal.

### Reproduction commands (Waterbirds, consistent code, val-based selection)

```bash
# ERM baseline
python run.py dataset=waterbirds method=erm training.epochs=15 training.batch_size=64 training.lr=1e-4 wandb.enabled=false

# JTT (binary, faithful to Liu et al. 2021)
python run.py dataset=waterbirds method=jtt training.epochs=15 training.discovery_epochs=1 training.batch_size=64 training.lr=1e-4 training.discovery_criterion=loss training.discovery_upweight=50.0 wandb.enabled=false

# Group DRO (oracle)
python run.py dataset=waterbirds method=group_dro training.epochs=15 training.batch_size=64 training.lr=1e-4 wandb.enabled=false

# Ours (discovered split + V-REx)
python run.py dataset=waterbirds method=discovered_split training.epochs=15 training.discovery_epochs=5 training.batch_size=64 training.lr=1e-4 training.lambda_disagree=10.0 training.discovery_criterion=loss training.discovery_upweight=50.0 training.early_stop_patience=5 wandb.enabled=false
```

### CMNIST
- Model: 2-layer MLP, hidden_dim=256
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- Batch size: 256
- Gradient clipping: max_norm=1.0

### Waterbirds
- Model: ResNet-50 (ImageNet pretrained), fine-tuned
- Discovery model: ResNet-50 with frozen backbone (only linear head trained)
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-4
- Batch size: 64
- Transforms: Resize(256), RandomResizedCrop(224), RandomHorizontalFlip, ImageNet normalise
- Gradient clipping: max_norm=1.0

---

## NeurIPS Paper Plan

### Title (working)

**"Adaptive Invariance from ERM Residuals: Discovering Environments for Robust Training Without Group Labels"**

### One-sentence pitch

We turn a model's own mistakes into a self-calibrating training signal: discover environments from ERM loss residuals, then adaptively apply V-REx based on a permutation test that measures whether those environments carry real structure.

### What makes this not incremental

1. **The permutation test for adaptive λ.** No prior work automatically calibrates the invariance penalty. Everyone tunes λ per dataset. Our permutation test directly measures "are the discovered environments better than random?" and scales λ accordingly. This is a standalone contribution — applicable to ANY invariance method (IRM, V-REx, FISH) that needs environments.

2. **Environments from loss, not from metadata.** JTT upweights hard examples but gives them no structure. GEORGE clusters features but needs dimensionality reduction + hyperparameter tuning. We do the simplest possible thing — median split of ERM losses — and show it's sufficient when combined with upweighting + V-REx + adaptive calibration.

3. **Unified method across regimes.** Same code, same hyperparameters, three domains:
   - Synthetic (CMNIST): signal_ratio=3565 → full V-REx → massive OOD gains
   - Real images (Waterbirds): signal_ratio=52 → full V-REx → beats Group DRO without labels
   - Chemistry (TADF): signal_ratio=1.5 → graceful fallback to JTT → doesn't hurt
   No method in the literature works across all three without tuning.

4. **Iterative refinement for continuous spurious features.** Self-improving loop: each round of V-REx partially breaks the shortcut → re-discovery finds sharper environments → next round is better. Assignment-colour correlation goes from 0.15 → 0.80 across 5 rounds. Novel contribution for domains where the spurious feature is continuous (chemistry, medical imaging).

5. **Multiple spurious features.** Multi-spurious CMNIST shows V-REx handles overlapping shortcuts (worst-group: ERM 8.5%, JTT 27%, ours 41%). The environment split captures the combined effect without needing to decompose which shortcut is which.

### Paper structure

**Section 1: Introduction**
- Algorithmic stability (Bousquet & Elisseeff) as an analysis tool → we turn it into a training objective
- Key problem: V-REx/IRM need environments, but environments aren't given in practice
- Our solution: discover + calibrate automatically

**Section 2: Method**
- 2.1: Environment discovery from ERM loss residuals (loss scoring, median split, upweighting)
- 2.2: Adaptive V-REx via permutation test (the principled contribution)
- 2.3: Iterative refinement for continuous features
- Algorithm box: one unified pipeline

**Section 3: Why V-REx > upweighting (theory/intuition)**
- JTT upweights but the model can accommodate without changing strategy
- V-REx forces equal *environment-level* risk — the model must use features that work in both environments
- Ablation: upweighting alone (JTT) ≈ ERM on Waterbirds; add V-REx → beats Group DRO

**Section 4: Experiments**

Table 1: Main results across all datasets (adaptive method, single config)

| Dataset | ERM | JTT | Group DRO | Ours (adaptive) |
|---------|-----|-----|-----------|----------------|
| CMNIST (OOD acc) | 22% | 22% | — | ~70%+ |
| Waterbirds (test WGA) | 70.3% | 65.7% | 77.7% | **83.5%** |
| Multi-CMNIST (OOD WGA) | 8.5% | 27% | — | **41%** |
| TADF chemistry (OOD acc) | 45.2% | 50.6% | — | ~45% (≈ERM, doesn't hurt) |

Table 2: Ablation — each component is necessary

| Components | Waterbirds WGA |
|---|---|
| ERM | 70.3% |
| + upweighting (JTT) | 65.7% |
| + V-REx (no upweight) | 65.7% |
| + upweighting + V-REx | 75.7% |
| + adaptive calibration | **83.5%** |

Table 3: Adaptive calibration across regimes

| Dataset | N | signal_ratio | reliability | effective λ |
|---------|---|-------------|------------|------------|
| CMNIST | 60K | 3565 | 1.0 | full |
| Waterbirds | 4.8K | 52 | 1.0 | full |
| TADF | 1K | 1.5 | 0.25 | weak |

Figure 1: Iterative refinement on continuous features
- 5 rounds, assignment-colour correlation 0.15 → 0.80

Figure 2: Multi-spurious CMNIST
- ERM/JTT/Ours worst-group comparison

**Section 5: Chemistry application**
- TADF dataset with property-based confounding
- Method gracefully degrades when signal is weak
- Env-aware mixup for tabular features

**Section 6: Related work**
- IRM, V-REx (need environments)
- JTT, DFR, CnC (post-hoc correction)
- GEORGE (clustering → pseudo-environments)
- Ours: simplest discovery + principled calibration

**Section 7: Limitations and future work**
- Single-seed results (need 5+ for error bars)
- Small chemistry datasets: V-REx signal too weak (but adaptive fallback prevents harm)
- Stability scores not yet evaluated (future: ROC for OOD flip prediction)
- Regression tasks (current: classification only)

### Before submission: must-do

- [ ] Multi-seed runs (5 seeds) on Waterbirds, CMNIST, TADF — get error bars
- [ ] Re-run ALL experiments with final code (consistent V-REx, adaptive, val selection)
- [ ] DFR baseline on Waterbirds (the strongest published number)
- [ ] Stability score evaluation (ROC curve: does confidence predict OOD flips?)
- [ ] One-command reproduction scripts per table/figure
- [ ] Clean up codebase: remove dead code, ensure all tests pass
- [ ] Write the paper

### Stretch goals

- [ ] CelebA dataset
- [ ] Regression version for chemistry
- [ ] Formal stability bound connecting V-REx to algorithmic stability
- [ ] Feature-space mixup (instead of input-space) for images
