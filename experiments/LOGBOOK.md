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

**Key finding 1:** JTT ≈ ERM on Waterbirds. Pure upweighting does NOT help — the model accommodates the upweighted minority while still relying on background. This shows V-REx is the critical ingredient, not the upweighting.

**Key finding 2:** Ours ≈ Group DRO without group labels. The V-REx equal-risk constraint forces the model to find features that work across both environments, matching the oracle that has explicit group supervision.

**Key finding 3:** Method generalises from synthetic (CMNIST) to real images.

**Note:** Results are noisy — worst-group oscillates heavily across epochs. Early stopping is essential. Multiple seeds needed for reliable comparisons.

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

**Implementation note:** The K-environment V-REx is a straightforward generalisation:
```
losses_per_env = [weighted_ce[env_k].mean() for k in range(K)]
risk_variance = torch.var(torch.stack(losses_per_env))
model_loss = mean(losses_per_env) + lambda * risk_variance
```

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
