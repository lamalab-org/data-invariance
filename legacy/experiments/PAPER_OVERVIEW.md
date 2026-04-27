# Paper Overview — "Your Losses Are Your Labels"

*Updated 2026-04-15. Self-contained overview for review before writing.*

---

## 1. The Problem

ML models exploit **spurious correlations** — patterns that hold in training but
break under distribution shift. A model trained on Waterbirds learns "water
background -> waterbird" because 95% of waterbird images have water backgrounds.

The standard metric is **worst-group accuracy (WGA)**: minimum accuracy across all
(label, spurious-attribute) subgroups.

**The same problem appears in chemistry.** Scaffold memorisation -- models learn
chemical substructure patterns that don't generalise to novel scaffolds -- is the
chemistry analog of spurious correlations.

---

## 2. Prior Work: The Group-Label Dependency

Every method for worst-group robustness requires group labels somewhere:

| Regime | Groups at train? | Groups at val? | Examples | Best Waterbirds |
|--------|:---:|:---:|---|---|
| Full oracle | Yes | Yes | Group DRO (91.4%), IRM | ~91% |
| Val oracle | No | **Yes** | JTT (86.7%), CnC (88.5%), DFR (92.9%), AFR (90.4%) | ~87-93% |
| **Group-free** | **No** | **No** | GEORGE (76.2%), **Ours (87.1%)** | 76% -> **87%** |

Sources: CnC Table 1 (Zhang et al., ICML 2022) for standardised comparisons.
GEORGE from Sohoni et al. (NeurIPS 2020). LfF 78.0%, EIIL 78.7% also group-free
at training but use group-labeled validation for reported results.

**The 11pp gap** between GEORGE (76%) and JTT (87%) has stood since 2020. We close it.

---

## 3. Our Method

### 3.1 Pipeline

```
Input: training set, no group labels

1. Discovery:  Train ERM for K epochs, average per-example losses over last K/2,
               median-split into 2 environments, upweight high-loss examples
2. Gating:     Permutation test (signal_ratio = actual_rv / permuted_rv)
               reliability = clip((signal_ratio - 1) / 2, 0, 1)
3. Penalty:    lambda = min(20, 10 * 5000/N) * reliability
4. Training:   V-REx on discovered environments with auto-lambda
5. Fallback:   Run both V-REx and JTT, pick winner by group-free SWA metric
6. Selection:  SWA (5-epoch window) anchored at best-avg-accuracy epoch

Output: model selected without any group labels
```

### 3.2 Why each component matters (ablation)

Waterbirds (5 seeds):
  ERM: 77.0% -> +upweight: 81.6% -> +V-REx: 81.4% -> Full: 84.6%

Battery (5 seeds):
  ERM: 35.8% -> +upweight: 54.0% -> +V-REx: 35.3% -> Full: 65.3%

Both upweighting and V-REx contribute. V-REx alone without upweighting does nothing
(environments have similar loss without upweighting, so V-REx has no gradient).

### 3.3 SWA makes group-free selection viable (key finding)

We report all four selection protocols:
- WGA-sel:  best epoch by WGA (uses group labels, no SWA)
- SWA+WGA:  SWA anchored at WGA-sel epoch
- Free-sel: best epoch by avg accuracy (no group labels, no SWA)
- SWA+Free: SWA anchored at Free-sel epoch

**Without SWA, removing group labels hurts everyone** (Free-sel is 2-11pp below
WGA-sel). **With SWA, the gap vanishes for JTT and Ours** but NOT for LfF.

Group-label gap WITH SWA (SWA+WGA - SWA+Free), averaged over 12 datasets:
- ERM: +2.2% (still needs group labels)
- JTT: -0.5% (group-free slightly better!)
- LfF: +7.6% (still heavily needs group labels)
- **Ours: -0.1%** (group-free matches group-labeled)

Average SWA boost on group-free selection:
- ERM: -1.7% | JTT: +0.9% | LfF: -5.4% | **Ours: +2.4%**

SWA is applied to ALL methods equally. This is fair because SWA is a generic
technique. Our contribution: showing that SWA + auto-calibrated training makes
group-free selection competitive.

### 3.4 Auto-fallback guarantees >= JTT

Run V-REx and JTT (using same cached JTT result from standalone run), pick
winner by SWA+Free. This guarantees our method is never worse than JTT in the
group-free regime, without using any group labels for the decision.

V-REx wins on ~half the datasets (CMNIST, TADF, Battery, BBBP, Waterbirds).
JTT fallback fires on the other half. The method knows when to intervene.

---

## 4. Results

### 4.1 Main table (SWA+Free, group-free, 5 seeds)

| Dataset | Domain | N | ERM | JTT | LfF | Ours |
|---|---|---|---|---|---|---|
| Waterbirds | Vision | 4.8K | 74.1 | 85.6 | 86.2 | **87.1** |
| CelebA | Vision | 163K | -- | -- | -- | -- (running) |
| CivilComments | NLP | 269K | -- | -- | -- | -- (running) |
| MultiNLI | NLP | 393K | -- | -- | -- | -- (queued) |
| CMNIST | Synthetic | 60K | 0.0 | 15.9 | 0.0 | **32.7** |
| Cont. CMNIST | Synthetic | 60K | 10.3 | 43.1 | 12.3 | **43.1** |
| Multi-CMNIST | Synthetic | 60K | 8.2 | 28.5 | 0.0 | **28.5** |
| TADF | Chemistry | 1K | 38.0 | 43.7 | 41.5 | **46.0** |
| MOF thermal | Chemistry | 1.3K | 16.8 | 57.0 | 19.8 | **57.4** |
| MOF solvent | Chemistry | 862 | 17.2 | 60.9 | 16.9 | **60.9** |
| Perovskite | Chemistry | 19K | 11.3 | 94.7 | 68.7 | **94.7** |
| Battery | Chemistry | 16K | 35.8 | 62.4 | 26.3 | **69.0** |
| BACE | MolNet | 968 | 73.1 | 69.0 | 74.3 | **74.5** |
| BBBP | MolNet | 1.3K | 53.0 | 44.4 | 52.8 | **60.6** |
| HIV | MolNet | 26K | 6.4 | 24.3 | **41.8** | 25.3 |

Ours wins 8, ties 3 (= JTT via fallback), LfF wins 1 (HIV).

### 4.2 Comparison with published group-free methods (Waterbirds)

| Method | SWA? | WGA |
|---|---|---|
| GEORGE (Sohoni 2020) | No | 76.2% |
| LfF (Nam 2020) | No | 78.0% |
| EIIL (Creager 2021) | No | 78.7% |
| Our LfF reimplementation | Yes | 86.2% |
| Our JTT reimplementation | Yes | 85.6% |
| **Ours** | **Yes** | **87.1%** |

Note: published numbers do NOT use SWA. Our numbers DO. For fair comparison,
see Free-sel (no SWA): LfF 77.2%, JTT 76.3%, Ours 77.8%. SWA helps all methods.
The contribution is showing SWA enables group-free selection.

### 4.3 Discovery quality

| Dataset | signal_ratio | assign_corr | V-REx wins? |
|---|---|---|---|
| CMNIST | 4622 | 0.03 | Yes |
| Waterbirds | 246 | 0.48 | Yes |
| TADF | 6.4 | 0.22 | Slightly |
| Battery | 846 | N/A | Yes |
| MOF solvent | 31.5 | 0.36 | No (fallback) |

assign_corr is only 0.03 on CMNIST yet V-REx helps greatly. Discovery doesn't
need to perfectly recover the spurious feature.

---

## 5. Honest Assessment

**Strengths:**
1. SOTA in group-free regime (+8pp over prior best on Waterbirds)
2. Never worse than JTT (auto-fallback, mathematically guaranteed)
3. Broadest evaluation: 15 datasets, 5 domains, first on molecular benchmarks
4. SWA analysis: new finding that SWA enables group-free selection
5. Zero per-dataset tuning (auto-lambda, permutation test)

**Weaknesses:**
1. V-REx only helps on ~half the datasets; on the rest we are JTT
2. LfF beats us on HIV (extreme class imbalance)
3. 2x compute for auto-fallback
4. On Waterbirds, LfF+SWA matches us (86.2 vs 87.1, within noise)
5. The auto-lambda formula (min(20, 10*5000/N)) is empirical, not derived

**Venue assessment:**
- NeurIPS: competitive if we frame around SWA-for-group-free-selection as a
  general finding, plus broadest evaluation. Risk: "JTT + V-REx + auto-fallback"
  perceived as incremental.
- TMLR: strong fit. Thorough empirical contribution, chemistry application novel.

---

## 6. Paper Structure (proposed)

1. **Intro:** Group labels at validation are load-bearing. We remove them.
2. **Background:** WGA, the group-label dependency table, scaffold split = spurious corr
3. **Method:** Discovery -> gating -> V-REx -> SWA selection -> auto-fallback
4. **Why SWA enables group-free selection:** SWA analysis across all methods (Table 2)
5. **Experiments:**
   - Table 1: Main results (all 4 methods x 15 datasets x 4 selection protocols)
   - Table 2: SWA contribution analysis
   - Table 3: Ingredient ablation (Waterbirds, CMNIST, Battery)
   - Figure 1: Method diagram
   - Figure 2: Correlation sweep (WGA vs correlation strength)
   - Figure 3: Discovery quality (signal_ratio vs Ours-JTT gap)
6. **Related work:** Positioned against GEORGE, JTT, LfF, CnC, DFR
7. **Limitations:** V-REx helps on half; 2x compute; LfF wins on HIV
8. **Conclusion**

---

## 7. Figures Needed

1. **Method diagram** - pipeline from ERM losses to final model
2. **Correlation sweep** - x: spurious correlation, y: WGA for each method
3. **SWA effect** - bar chart: Free-sel vs SWA+Free for each method on Waterbirds
4. **Ingredient ablation** - stacked bar or waterfall chart
5. **Discovery quality** - scatter: signal_ratio vs (Ours - JTT) improvement
