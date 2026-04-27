# Consolidated Status and Path Forward

## What we actually built

A pipeline: train ERM → score by loss → split into environments → permutation
test for adaptive λ → V-REx training.  All automatic, no group labels.

## What works (honestly)

### Accuracy improvements
| Dataset | ERM | Ours | Δ | Notes |
|---------|-----|------|---|-------|
| CMNIST (corr=0.9) | 22% OOD | ~67% | +45pp | Strong, synthetic |
| Waterbirds | 71.0 ± 8.9 WGA | 76.7 ± 1.0 | +5.7pp | Reliable, real images |
| Multi-CMNIST | 8.5% WGA | 41% | +32pp | Multiple shortcuts |

### Method properties
- **Permutation test**: correctly identifies when to apply V-REx (CMNIST: λ=10, TADF: λ≈0)
- **Loss-based discovery**: beats entropy, counterfactual, activation-based
- **Component necessity**: upweighting alone (JTT) ≈ ERM, V-REx alone ≈ ERM, both needed
- **Low variance**: 76.7 ± 1.0 on Waterbirds (much more stable than ERM ± 8.9)

### Distribution shift detection (CMNIST only)
- Our model's loss predicts ID→OOD flips: AUROC 0.56 vs ERM 0.32
- ERM is anti-predictive (confidently wrong on fragile examples)
- Our model has 2.4x lower flip rate (29% vs 72%)

## What doesn't work

- **Stability scores on natural data**: ERM's uncertainty is better (Waterbirds ρ=0.86 vs 0.63, TADF AUROC 0.91 vs 0.60)
- **Chemistry at high confound**: V-REx hurts on TADF at corr=0.9 (~70 counterexamples too few)
- **Closing gap to DFR**: 76.7% vs 88.4% on Waterbirds (DFR uses group-balanced val set)
- **K=4 environments**: higher ceiling but unstable across seeds
- **Freeze backbone**: doesn't work (domain blindness with fine-tuned; too generic with frozen)

## The real bottleneck

The environment quality.  Our discovery creates environments with correlation
gap ≈ 0.19 on Waterbirds.  Oracle groups have gap ≈ 1.0.  That's 5x worse.
Everything else (V-REx, upweighting, permutation test) works correctly — it's
the INPUT to V-REx (the environments) that limits us.

## How to increase OOD robustness

### Short-term (could try now)

1. **Iterative refinement on Waterbirds.**  We showed 5 rounds on ContinuousCMNIST
   improves assignment-colour correlation from 0.15 → 0.80.  We haven't tried
   multiple rounds on Waterbirds.  Each round: train V-REx → re-discover
   environments from the V-REx model's losses → train new V-REx.

2. **Asymmetric split.**  Instead of 50/50 median, split 90/10 (top 10% loss = env B).
   The minority concentration in env B jumps from ~10% to potentially 50%+.
   Combined with upweighting, V-REx has a much sharper signal.  We tried this
   on CMNIST (quantile split) — it created extreme environments but the model
   learned anti-features.  With upweighting + adaptive λ, it might work better now.

3. **Ensemble discovery.**  Train 3 ERMs with different seeds.  Examples that ALL
   ERMs get wrong are more reliably minority than examples one ERM gets wrong.
   Intersection of high-loss sets = more precise identification.

4. **Use ERM's held-out loss.**  Train ERM on 80% of training data, score on
   remaining 20%.  The held-out loss isn't contaminated by memorisation.  Repeat
   5 times (5-fold) and average for stable scoring.

### Medium-term (requires more development)

5. **Combine loss with ERM features.**  Loss tells us THAT an example is hard.
   ERM features tell us WHY (which features the model relies on).  Cluster
   high-loss examples by their features to find coherent minority subgroups.
   This is GEORGE-style but targeted at the hard examples.

6. **Train multiple V-REx models with different environments.**  Instead of
   one environment split, create 5 different splits (different seeds, different
   criteria).  Ensemble the resulting models.  Each model handles a different
   aspect of composition sensitivity.

## How to generalise beyond spurious correlations

### The general problem: distribution shift
Spurious correlations are ONE cause of OOD failure.  Others:
- **Domain shift**: hospital A → hospital B, lab A → lab B
- **Temporal shift**: training on 2020 data, deploying in 2024
- **Subpopulation shift**: training on urban → deploying on rural
- **Measurement shift**: different instruments, different protocols

### What our method actually does (abstractly)
We find subsets of the training data where the model behaves DIFFERENTLY,
then force equal performance across those subsets.  This is general — it
doesn't assume the shift is caused by a spurious feature.  It assumes:

1. The ERM's loss landscape has structure that reflects composition sensitivity
2. High-loss examples are in some sense "different" from low-loss examples
3. Equalising performance across these groups reduces sensitivity

### Where this generalises naturally
- **Domain shift**: if some training domains are harder for the ERM, they
  get higher loss → end up in env B → V-REx forces equal performance →
  reduces domain dependence.  Should work WITHOUT domain labels.
- **Temporal shift**: recent data might have different patterns → if the
  ERM is trained on the full temporal range, older examples might have
  higher loss → env B captures temporal sensitivity.
- **Subpopulation shift**: underrepresented subpopulations have higher
  loss → env B → V-REx forces the model to handle them equally.

### Where it might NOT generalise
- **Covariate shift**: if P(y|x) is the same but P(x) changes, the ERM's
  loss doesn't necessarily capture this.  The loss reflects difficulty, not
  distributional difference.
- **Label shift**: class proportions change.  Our method doesn't address this.
- **Novel-class shift**: completely new categories at test time.  Out of scope.

### How to validate generalisation
1. **DomainBed** (PACS, VLCS, OfficeHome): standard domain shift benchmarks.
   Train on N-1 domains, test on held-out domain.  Our method: train ERM on
   all N-1 domains, discover environments from loss, V-REx on discovered envs.
   Compare to methods that USE domain labels (CORAL, DRO, FISH).

2. **WILDS** (Camelyon17, FMoW, iWildCam): real-world distribution shifts
   with known group structure.  Our method: same pipeline, no group labels.

3. **Chemistry temporal split**: train on papers published before 2020,
   test on 2021+.  Our method should identify examples from older papers
   that might not generalise.

## Attempted improvements that didn't help

| Variation | Result | Why it didn't help |
|-----------|--------|-------------------|
| K=4 environments | Higher ceiling but ±7 variance | Smaller envs → noisier loss estimates |
| Asymmetric q=0.1 | 76.1% (≈ median) | Excluding 80% data loses learning signal |
| Confident wrong | 74.5 ± 6.6 | Same ranking as loss on Waterbirds |
| Cartography K=4 | Unstable (56-81%) | Extreme env sizes → noisy V-REx |
| Averaged discovery | 74.8 ± 2.7 | Smooths out useful signal |
| Iterative refinement | Same as round 1 | V-REx model has similar loss landscape to ERM |
| Freeze backbone | 20-73% | Frozen features too generic or too biased |
| Balanced sampling | 65.7% | Without upweighting, minority signal too dilute |
| Env-aware mixup | Hurts on images | Pixel interpolation unrealistic |

**Conclusion: the simple approach (K=2, median, loss, upweight=50) is the robust one.**
Every added complexity increases variance without reliably improving the mean.

## Honest assessment for a paper

### What we CAN claim
1. A principled, fully automatic pipeline for OOD robustness without group labels
2. The permutation test for adaptive λ (novel, reusable)
3. Consistent improvement over ERM on vision benchmarks (small but reliable)
4. Loss-based discovery is superior to alternatives (clean ablation)
5. The method never hurts — it degrades gracefully to ERM when environments are noise

### What we CANNOT claim
1. State-of-the-art on Waterbirds (DFR: 88%, Group DRO: 80%, us: 77%)
2. Stability scores that beat ERM's uncertainty (only on artificial confounds)
3. Large improvements on small chemistry datasets
4. Theoretical guarantees beyond informal propositions

### The most honest and compelling paper
Frame as: "A fully automatic, adaptive pipeline for OOD robustness that
requires no group labels and never hurts.  The permutation test determines
when invariance training helps, and the method degrades gracefully when it
doesn't.  We demonstrate reliable improvements on vision benchmarks and
show the method extends to molecular property prediction."

This is an **engineering contribution** (a robust, practical tool) rather
than a **methodological breakthrough** (a new insight).  Position accordingly —
workshops/applied venues might be better than NeurIPS main track with the
current results.  For NeurIPS, we'd need either:
(a) a strong theoretical contribution (stability bounds), or
(b) SOTA results on standard benchmarks, or
(c) a striking application result (e.g., stability scores that work)
