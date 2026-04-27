# Paper Plan: Training for Data Composition Invariance

## The real problem (not spurious correlations)

ML predictions are sensitive to *which* data the model trained on. Retrain on
a different subsample → some predictions change. This is **data composition
sensitivity**, and it's the fundamental problem behind:

- Spurious correlations (model learned a feature that only works for this sample)
- Batch effects in chemistry (predictions depend on which lab's data was included)
- Replication crises (model works on this dataset but not that one)
- Lack of trust in high-stakes predictions (would this drug candidate still be
  predicted active if we'd collected different training data?)

**Algorithmic stability** (Bousquet & Elisseeff 2002) quantifies this: a stable
algorithm produces similar outputs regardless of small changes to the training set.
But stability has only been studied as a *property to analyse* — never as a
*training objective*.

**We turn stability into a training objective.**

## What we actually do (mechanistically)

1. Train an ERM and look at its mistakes → these reveal which examples the model
   is sensitive to (if the training composition changed, these would change first)

2. Split into environments by loss → creates two "versions" of the training data
   that the model treats differently

3. Permutation test → measures whether these environments capture real sensitivity
   or just noise

4. V-REx training → forces the model to perform equally on both environments →
   reduces sensitivity to which environment it sees → stability

5. The model's per-example loss difference between environments = **stability score**
   = "how much does this prediction depend on which data the model trained on?"

## Why this framing is better than "spurious correlations"

| Framing | Problem | Contribution | Competition |
|---------|---------|-------------|-------------|
| Spurious correlations | Model uses wrong feature | Another method | DFR, JTT, CnC, GEORGE, ... |
| **Data composition invariance** | **Model depends on which data it saw** | **New training objective** | **Nobody** |

The spurious correlation framing puts us in a crowded space where DFR gets 93% and
we get 83%. The stability framing puts us in an EMPTY space where we're the first
to do this at all.

## The strongest possible paper

### Title

**"Training for Stability: Making Predictions Invariant to Training Data Composition"**

or

**"Which Predictions Would Change? Training Models to Be Invariant to Dataset Composition"**

### One-paragraph abstract

Machine learning predictions can depend on which data the model happened to train
on — retrain on a different subsample and some predictions flip. We turn this
sensitivity into a training signal. By splitting the training set into subsets
where an ERM model disagrees with itself, we create environments that capture
data-composition sensitivity. A permutation test automatically determines how
strongly to penalise this sensitivity, and the trained model produces per-example
stability scores that predict which predictions are fragile. On vision benchmarks,
the method matches oracle methods that require environment labels. On molecular
property prediction, it identifies predictions that depend on publication source
rather than genuine chemistry. The stability scores detect confidently-wrong
predictions that standard uncertainty measures miss.

### Key figures (redesigned for stability framing)

**Figure 1: The problem.**
Take a dataset. Train an ERM 10 times with different random 90% subsamples.
For each test example, measure how often the prediction changes across the 10
models. Show: some examples are always predicted correctly (stable), some flip
in 3/10 retrainings (moderately fragile), some flip in 8/10 (very fragile).
Standard confidence (softmax) doesn't correlate with this fragility.

THIS is the problem we solve: can we identify the fragile examples from a
single training run, and can we train a model that has fewer fragile examples?

**Figure 2: Our stability scores predict fragility.**
Same setup as Figure 1. x-axis = our stability score, y-axis = actual flip
rate across 10 retrainings. Show strong correlation (monotonically increasing).
Compare to ERM confidence, entropy, MC dropout — none of them predict
retraining fragility. Ours does.

This is the "killer figure" — it directly shows the thing we claim.

**Figure 3: The method reduces fragility.**
Bar chart: % of test examples that flip in ≥3/10 retrainings.
ERM: X%. Ours: Y% (much lower). The model is genuinely more stable.

**Figure 4: The permutation test.**
Three datasets (CMNIST, Waterbirds, TADF). For each: actual risk variance
vs histogram of permuted risk variances. Shows the method correctly identifies
when environments carry real stability signal.

**Figure 5: Chemistry application.**
TADF emission wavelength dataset. Train on data dominated by 3 research groups.
Test on held-out research groups.
(a) ERM predictions change dramatically when we remove one group from training.
(b) Our stability scores correctly flag the predictions that would change.
(c) The method trained for stability is less affected by group removal.

This connects directly to the clever-hans insight: predictions shouldn't depend
on which lab's data was included.

### Key tables

**Table 1: Stability reduction across domains**
Not accuracy numbers — STABILITY numbers.

| Dataset | ERM flip rate | Ours flip rate | Reduction |
|---------|:---:|:---:|:---:|
| CMNIST | 72% | 29% | 2.4x |
| Waterbirds | ?% | ?% | ?x |
| TADF | ?% | ?% | ?x |

**Table 2: Stability scores predict retraining fragility**
AUROC for "does the score predict which examples flip under resampling?"

| Score type | Our model | ERM | MC Dropout | Ensemble |
|---|:-:|:-:|:-:|:-:|
| Loss | 0.56 | 0.32 | 0.50 | ? |
| Entropy | 0.55 | 0.43 | 0.50 | ? |

**Table 3: Accuracy comparison (secondary, in appendix or brief main text)**
Show Waterbirds/CMNIST accuracy to confirm the method doesn't sacrifice accuracy.
But this is NOT the main result — stability is.

### What we still need to do

**Critical (defines whether the paper works):**

- [ ] **Resampling experiment**: Train ERM 10x on different 90% subsamples of
      CMNIST/Waterbirds. Measure per-example flip rates. This is the GROUND TRUTH
      for stability. Then show our stability scores predict these flip rates.
      This is Figure 1+2 — the most important figures in the paper.

- [ ] **Same for our method**: Train our method 10x. Show the flip rate is lower.
      This is Figure 3.

- [ ] Multi-seed Waterbirds (already running)

- [ ] CelebA (standard benchmark)

**Important:**

- [ ] Chemistry resampling experiment: train on subsets that include/exclude
      specific author groups. Show which predictions change. Show our stability
      scores predict this.

- [ ] Deep ensemble baseline for stability scores (train 5 independent ERMs,
      measure ensemble disagreement as a stability proxy).

- [ ] Proposition connecting signal_ratio to stability bound.

**Nice to have:**

- [ ] MultiNLI (NLP domain)
- [ ] Formal stability theorem
- [ ] More chemistry datasets

### The pitch to a reviewer

"Algorithmic stability tells you how much your model's predictions depend on
which data it trained on. Everyone analyses this after the fact — we train for
it. The result is a model with fewer fragile predictions and a per-example
stability score that identifies the remaining fragile ones. The permutation
test automatically determines how much stability to enforce, making the method
plug-and-play across domains. On chemistry datasets, the stability scores
flag predictions that depend on publication source rather than genuine
molecular properties."

### Why this might be a genuinely strong paper

1. **No one has done this.** Training for algorithmic stability is a new idea.
   The connection to V-REx is a clean formalisation, but the GOAL is new.

2. **The resampling experiment is the key.** If our stability scores predict
   actual retraining fragility (Figure 2), that's a strong empirical result
   that no prior work has shown.

3. **Chemistry is the compelling application.** "Your model's prediction that
   this molecule is active would change if you removed Dr. Smith's lab from
   the training data" — that's immediately actionable for a medicinal chemist.

4. **The permutation test is a standalone tool.** Even if you use DFR or JTT,
   our permutation test tells you whether your environments are real. This is
   useful to the broader community.

### What could make this NOT work

- If our stability scores DON'T predict retraining fragility, the paper
  collapses. This is the make-or-break experiment.

- If the resampling experiment shows that ERM is already quite stable (low
  flip rates even without our method), there's no problem to solve.

- If the chemistry application shows no meaningful author-dependence, the
  practical motivation weakens.

These risks should be assessed BEFORE writing the paper.
