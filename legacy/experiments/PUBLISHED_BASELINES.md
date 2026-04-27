# Published Baseline Numbers

Primary source: **CnC Table 1** (Zhang et al., ICML 2022) provides standardised
WGA numbers for ERM, GDRO, LfF, GEORGE, PGI, EIIL, CIM, JTT, CnC under identical
conditions. Secondary sources: original papers (DFR, AFR, SSA).

All numbers are worst-group test accuracy (%) on the standard dataset variants.

## Waterbirds (ResNet-50)

| Method | WGA (%) | Groups@train | Groups@val | Source |
|---|---|:---:|:---:|---|
| ERM | 72.6 | — | — | CnC T1 |
| Group DRO | 91.4 | Yes | Yes | CnC T1 |
| LfF | 78.0 | No | Yes* | CnC T1 |
| GEORGE | 76.2 | No | No | CnC T1 |
| EIIL | 78.7 | No | Yes | CnC T1 |
| JTT | 86.7 | No | Yes | Liu 2021 T1 |
| CnC | 88.5 | No | Yes | CnC T1 |
| SSA | ~89 | Partial | Yes | Nam 2022 |
| DFR | 92.9 | No | Yes (retrain) | Kirichenko 2023 T2 |
| AFR | 90.4 | No | Yes | Qiu 2023 T1 |

*LfF's reported results use group-labeled validation for model selection.

## CelebA (ResNet-50)

| Method | WGA (%) | Groups@train | Groups@val | Source |
|---|---|:---:|:---:|---|
| ERM | 47.2 | — | — | CnC T1 |
| Group DRO | 88.9 | Yes | Yes | Sagawa 2020 T1 |
| LfF | 77.2 | No | Yes | CnC T1 |
| GEORGE | 53.4 | No | No | CnC T1 |
| EIIL | 85.1 | No | Yes | CnC T1 |
| JTT | 81.1 | No | Yes | Liu 2021 T1 |
| CnC | 88.8 | No | Yes | CnC T1 |
| DFR | 88.3 | No | Yes (retrain) | Kirichenko 2023 T2 |
| AFR | ~85 | No | Yes | Qiu 2023 |

## CivilComments (BERT/DistilBERT)

| Method | WGA (%) | Groups@train | Groups@val | Source |
|---|---|:---:|:---:|---|
| ERM | 56.0 | — | — | CnC T1 |
| Group DRO | 69.9 | Yes | Yes | Sagawa 2020 |
| JTT | 69.3 | No | Yes | Liu 2021 T1 |
| CnC | 68.9 | No | Yes | CnC T1 |
| DFR | ~70 | No | Yes | Kirichenko 2023 |
| AFR | ~69 | No | Yes | Qiu 2023 |

## MultiNLI (BERT/DistilBERT)

| Method | WGA (%) | Groups@train | Groups@val | Source |
|---|---|:---:|:---:|---|
| ERM | ~67 | — | — | Sagawa 2020 |
| Group DRO | 77.7 | Yes | Yes | Sagawa 2020 T1 |
| JTT | 72.6 | No | Yes | Liu 2021 T1 |

## Regime classification

- **Full oracle** (groups at train + val): Group DRO, IRM
- **Val oracle** (groups at val/selection only): JTT, CnC, DFR, AFR, LfF, EIIL*, SSA
- **Truly group-free** (no groups anywhere): GEORGE, **Ours**

*EIIL: validation protocol varies by implementation; reported numbers typically
use group-labeled validation.

## How to use these numbers in the paper

**Main comparison (Table 2, Waterbirds):**
Cite published group-free baselines (GEORGE 76.2, LfF 78.0, EIIL 78.7) directly
from CnC Table 1. These numbers are without SWA.

**Our reimplementations:**
We run ERM, JTT, LfF, and Ours ourselves under identical conditions (same
backbone, same epochs, SWA for all). Report both sets of numbers:
- Our ERM+SWA, JTT+SWA, LfF+SWA, Ours
- Cite published ERM, JTT, LfF without SWA

**CelebA caveat:**
Our LfF on CelebA underperforms published (our matched-budget ~39% vs
published 77.2%). The discrepancy is due to compute budget: published LfF
trains for 127K steps with a frozen ResNet-18, while our matched-protocol
setup uses 13K steps with a full ResNet-50. The frozen-backbone LfF variant
(from the fix on 2026-04-16) partially addresses this.
