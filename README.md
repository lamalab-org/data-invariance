# Cross-sample prediction churn

Code and paper source for *Reducing cross-sample prediction churn in
scientific machine learning* (NeurIPS 2026 submission).

Two classifiers trained on independent bootstraps of the same
chemistry training set assign different classes to 8–22% of test
molecules, while their aggregate accuracy differs by only 1–4
percentage points.  We call this *cross-sample prediction churn* —
the population-level analogue of prediction churn for the small-N
from-scratch regime.  Bagging cuts churn 43–52% on every dataset at
no accuracy cost; twin-bootstrap (two networks with a sym-KL
consistency loss on independent bootstraps) cuts a further 41% beyond
matched-compute bagging-K=2.

## Repo layout

```
paper/                       NeurIPS source: main.tex + sections/*.tex
scripts/                     analysis pipeline (training drivers, table
                             and figure generators, paper-macro emitter)
configs/dataset/             per-dataset YAMLs consumed by build_cfg
data.py / data_molnet.py /   dataset loaders (MolNet, TDC, Waterbirds, TADF)
  data_tdc.py
models.py                    MLP + ResNet/ChemBERTa/GIN backbone factories
train.py                     make_dataloaders + _build_model
utils.py                     seed setting, device selection
tests/                       analysis-lib tests
outputs/                     git-ignored: NPZ predictions, derived CSVs
```

## Reproducing the paper

The paper's tables, figures, and macros all regenerate from saved
NPZs.  No model retraining required to rebuild the PDF:

```bash
make analysis    # CSVs from NPZs in outputs/cross_sample{,_seed7,_seed42}/
make tables      # paper/sections/tables/*.tex from CSVs
make macros      # paper/sections/macros.tex from CSVs
make figures     # paper/figures/*.pdf
cd paper && latexmk -pdf main.tex
```

To retrain from scratch on a single canonical seed:

```bash
bash scripts/run_cpu_blocks_local.sh             # ~2-3 hours on a 12-core Mac
```

To replicate the canonical-seed sensitivity sweep
(Appendix~\ref{app:seed_sensitivity}):

```bash
bash scripts/run_seed_sweep.sh 7  &              # parallel seeds
bash scripts/run_seed_sweep.sh 42 &
wait
uv run python scripts/aggregate_seed_sensitivity.py
```

ChemBERTa, GIN, and Waterbirds runs require a GPU and are dispatched
on the cluster via `slurm/`.

## Setup

```bash
uv sync --group dev
uv run pre-commit install
```

Python 3.13 + PyTorch 2.11.  See `pyproject.toml` for the full
dependency list.

## Tests

```bash
uv run pytest tests/
```
