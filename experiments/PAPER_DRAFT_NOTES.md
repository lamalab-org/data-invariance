# Paper Draft Notes — Section-by-Section

## Title

**"Training for Stability: Reducing Sensitivity to Training Data Composition"**

Alternative: "Which Predictions Would Change? Adaptive Invariance Training
Without Environment Labels"

---

## Abstract (bullet points for drafting)

- ML models are sensitive to training data composition — retrain on a different
  subsample and some predictions change
- This sensitivity underlies spurious correlations, batch effects, and
  replication failures
- We turn data-composition sensitivity into a training signal:
  - Discover environments from the ERM's own loss residuals
  - A permutation test automatically calibrates how strongly to penalise
    cross-environment risk variance
  - The trained model produces per-example stability scores
- Results: matches Group DRO on Waterbirds without group labels (81% vs 80%
  worst-group accuracy), with principled adaptive calibration across domains
- The stability scores detect confidently-wrong predictions that standard
  uncertainty measures miss

---

## 1. Introduction

### Opening (the problem)
- ML predictions can depend on *which* data the model happened to train on
- This is not just about "bad models" — even well-performing models can have
  predictions that would change if the training set were composed differently
- Examples:
  - A model predicts a molecule is active, but that prediction depends on
    which papers were in the training set (clever-hans in chemistry)
  - A bird classifier works well overall, but misclassifies waterbirds on land
    because most training waterbirds were on water backgrounds
  - A clinical model gives different risk scores depending on which hospitals
    contributed training data

### Background (what's been tried)
- **Algorithmic stability** (Bousquet & Elisseeff 2002, Hardt et al. 2016):
  formalises how much an algorithm's output changes with training data
  perturbations. Used to derive generalisation bounds. But purely analytical —
  no one has used it as a training objective.
- **Invariant risk minimisation** (IRM, Arjovsky et al. 2019) and
  **V-REx** (Krueger et al. 2021): train models whose representations are
  invariant across environments. Powerful but require pre-defined environment
  labels — which are rarely available in practice.
- **Post-hoc correction methods** (JTT, DFR, CnC, GEORGE): identify and
  correct for spurious correlations after initial ERM training. Effective but
  don't train for stability directly.

### Our contribution
- We turn stability into a training objective by:
  1. Discovering environments from ERM loss residuals (what the model gets wrong
     reveals composition sensitivity)
  2. A permutation test that calibrates the invariance penalty automatically
     (no hyperparameter tuning for λ)
  3. V-REx training that produces both a robust model AND per-example
     stability scores
- The method is plug-and-play: same config across synthetic, vision, and
  chemistry domains. The permutation test adapts automatically.

---

## 2. Related Work

### 2.1 Algorithmic stability
- Bousquet & Elisseeff (2002): uniform stability → generalisation bounds.
  A model is uniformly stable if replacing one training example changes the
  loss by at most ε. Our work: instead of proving stability bounds post-hoc,
  we optimise for stability during training.
- Hardt, Recht & Singer (2016): stability of SGD. Shows SGD is stable for
  convex losses. We extend this to non-convex models by explicitly training
  for cross-environment invariance.
- Connection to our permutation test: the signal ratio measures how much
  the risk variance changes under random permutations of environment labels.
  This is a finite-sample stability test.

### 2.2 Invariance-based training
- IRM (Arjovsky et al. 2019): learn representations where the optimal
  classifier is the same across environments. Requires environment labels.
- V-REx (Krueger et al. 2021): penalise variance of per-environment risks.
  Simpler than IRM, same environment requirement. We use V-REx as the
  penalty but discover environments automatically.
- FISH (Shi et al. 2022): gradient matching across domains. Also needs
  domain labels.
- **Key gap in all of these:** they assume environments are GIVEN. We
  discover them.

### 2.3 Environment discovery
- GEORGE (Sohoni et al. 2020): cluster ERM features → pseudo-environments
  → Group DRO. Closest to our approach. Differences: (a) we use loss
  scoring (simpler than feature clustering), (b) we add the permutation
  test for calibration, (c) we use V-REx not Group DRO.
- EIIL (Creager et al. 2021): learn environment labels to maximise IRM
  penalty. End-to-end but expensive.
- ZIN (Lin et al. 2022): joint environment and invariant representation
  learning. Requires auxiliary information.

### 2.4 Post-hoc correction (not invariance-based)
- JTT (Liu et al. 2021): train ERM, identify misclassified examples,
  upweight them in a second round. Our ablation shows upweighting alone
  ≈ ERM on Waterbirds — the invariance penalty is what helps.
- DFR (Kirichenko et al. 2023): freeze ERM backbone, retrain last layer
  on group-balanced validation data. Very effective (88% on Waterbirds)
  but needs group labels on validation set.
- CnC (Zhang et al. 2022): contrastive learning on pairs identified by
  ERM. Medium complexity.
- Dataset Cartography (Swayamdiha et al. 2020): categorises training
  examples by correctness × confidence. Inspires our K=4 environment
  split. They use it for data cleaning; we use it for environment
  construction.

### 2.5 Spurious correlations in chemistry
- Clever Hans in chemistry (Kappelbaum et al.): models exploit publication
  metadata (authors, journals) rather than genuine molecular features.
  Author-group splits reveal this.
- Scaffold splits (Bemis & Murcko): standard OOD evaluation in
  cheminformatics. Our method could discover scaffold-based environments
  automatically without knowing the scaffold decomposition.

---

## 3. Method

### 3.1 Problem setup
- Training set D = {(x_i, y_i)}_{i=1}^N
- Standard ERM: minimise average loss
- The model may learn features that depend on the training composition
  (spurious correlations, batch effects, etc.)
- Goal: train a model whose predictions are invariant to reasonable changes
  in training data composition
- We don't assume access to environment labels, group labels, or knowledge
  of what the spurious features are

### 3.2 Environment discovery from ERM residuals
- Train a standard ERM on D for T epochs
- Score each training example by cross-entropy loss:
  score_i = L(f_θ(x_i), y_i)
- High loss → the model's learned strategy fails on this example → likely
  from a minority group where the shortcut doesn't hold
- Split into K environments by loss rank (K=4 by default):
  - Env 0: lowest 25% loss (easy, shortcut works)
  - Env 1: 25-50% loss
  - Env 2: 50-75% loss
  - Env 3: highest 25% loss (hard, shortcut fails)
- Loss-based upweighting: w_i = 1 + α · (loss_i / max_loss)
  Amplifies the contribution of high-loss examples in the V-REx penalty

**Why loss and not entropy/counterfactual/activation-based?**
- Loss = "where the model's learned shortcut fails" (direct signal)
- Entropy = "where the model is uncertain" (weaker — can be uncertain
  for other reasons)
- Counterfactual = "where the model is sensitive to perturbation" (even
  weaker — all examples are somewhat sensitive)
- Empirical ablation: loss gives 0.40 environment gap on CMNIST vs
  0.29 for entropy, 0.27 for counterfactual
- Intuition: on CMNIST, the ERM is confidently wrong (low entropy!) on
  colour≠label examples. Loss catches these; entropy misses them.

### 3.3 Permutation test for adaptive calibration
- The invariance penalty λ should be strong when environments carry real
  signal and weak when they're noise
- **Key idea:** compare risk variance under real environments vs random
  permutations of environment labels

  1. Compute risk variance RV(a) with actual assignment a
  2. Compute RV(π_1(a)), ..., RV(π_M(a)) with M random permutations
  3. signal_ratio = RV(a) / mean(RV(π_j(a)))
  4. reliability = clip((signal_ratio - 1) / 2, 0, 1)
  5. effective_λ = λ_base × reliability

- Critically: the permutation test uses the DISCOVERY ERM (which has
  learned the shortcut), not the V-REx model. The ERM's loss landscape
  has structure that the test can detect; an untrained model doesn't.

- **Why this is principled:**
  - If signal_ratio ≈ 1: environments are no better than random splits →
    V-REx would fit noise → λ → 0 (falls back to upweighted ERM ≈ JTT)
  - If signal_ratio >> 1: environments capture real structure → full V-REx
  - No dataset-size heuristics, no correlation thresholds
  - Directly tests: "will V-REx help on this data?"

### 3.4 V-REx training with adaptive penalty
- Train a fresh model with loss:
  L = mean(L_env_k) + λ_eff · Var(L_env_1, ..., L_env_K)
- Where L_env_k = weighted mean of per-example losses in environment k
- The variance penalty forces equal risk across environments
- Model selection: best validation metric (worst-group acc when available,
  negative loss otherwise)
- Early stopping: patience-based on validation metric

### 3.5 Stability scores
- After training, the model's per-example loss serves as a stability score
- Low loss → prediction is robust to data composition changes
- High loss → prediction depends on specific training examples
- **Key finding:** ERM loss is ANTI-predictive of OOD flips (AUROC 0.32) —
  the model assigns low loss to examples that will flip under distribution
  shift. Our model's loss is correctly predictive (AUROC 0.56).
- Calibration: our entropy quintiles show monotonically increasing flip
  rates (0.63 → 0.75)

---

## 4. Theoretical motivation (informal propositions)

### Proposition 1 (permutation test soundness)
Under H0: the environment assignment is independent of the loss landscape
(i.e., environments are random). Then E[signal_ratio] = 1 and
P(signal_ratio > c) → 0 as M → ∞. The permutation test has correct
coverage: it applies V-REx only when environments carry signal above
the null.

### Proposition 2 (V-REx reduces composition sensitivity)
If the model achieves Var(L_env) = 0, then for any split of the training
data into the same K groups, the model's performance is identical.
This is a necessary condition for data-composition invariance: the model
cannot be sensitive to which environment it sees.

**Note:** this is not a sufficient condition — the model could achieve
zero variance by performing equally badly on all environments.
The upweighting addresses this by ensuring high-loss examples contribute
meaningfully to the environmental losses.

### Connection to uniform stability
A uniformly stable algorithm (Bousquet & Elisseeff) satisfies:
  |L(A(S), z) - L(A(S'), z)| ≤ β  for all z, S, S'
where S and S' differ by one example.

Our V-REx penalty approximates this by requiring equal performance across
K subsets of the training data. With K → N (one environment per example),
V-REx converges to a uniform stability objective. In practice, K=4
provides a tractable approximation.

---

## 5. Experiments outline

### Main results table
- CMNIST, Waterbirds, Multi-CMNIST, TADF
- All methods: ERM, JTT, Group DRO, DFR, Ours
- Same config for Ours across all datasets

### Discovery ablation
- Loss vs entropy vs counterfactual vs activation
- K=2 vs K=4
- Justifies design choices with data

### Permutation test across regimes
- CMNIST (signal_ratio=7937), Waterbirds (signal_ratio=52), TADF (1.5)
- Shows adaptive calibration in action

### Stability scores
- AUROC for predicting OOD flips
- Calibration plot
- ERM is anti-predictive, ours is correctly calibrated

### Component ablation
- Upweighting alone (JTT) ≈ ERM
- V-REx alone (no upweight) ≈ ERM
- Both together: works
- + adaptive: stabilises across seeds

---

## Key references to cite

1. Bousquet & Elisseeff 2002 — uniform stability
2. Hardt, Recht & Singer 2016 — stability of SGD
3. Arjovsky et al. 2019 — IRM
4. Krueger et al. 2021 — V-REx
5. Sagawa et al. 2020 — Group DRO, Waterbirds dataset
6. Liu et al. 2021 — JTT
7. Kirichenko et al. 2023 — DFR
8. Sohoni et al. 2020 — GEORGE
9. Swayamdiha et al. 2020 — Dataset Cartography
10. Zhang et al. 2022 — CnC
11. Kappelbaum et al. — clever hans in chemistry
