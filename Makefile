.PHONY: install lint format test check figures tables \
        sweep-erm sweep-bagging sweep-deep-ensemble sweep-twin-indep \
        sweep-pareto-bace help

# === Development ===

install:
	uv sync --group dev
	uv run pre-commit install

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

check: lint test


# === Paper artifacts: one command per figure / table ===

# Headline datasets (passed ERM > majority + 5pp check). Single source of truth
# is scripts/paper_constants.py; keep these lists in sync if either changes.
DEV_DATASET      := bace
DATASETS_HEADLINE := dili pgp_broccatelli bbb_martins bbbp tadf mof_thermal ames
DATASETS_ALL     := $(DEV_DATASET) $(DATASETS_HEADLINE)
SEEDS            := 1,2,3,4,5,6,7,8,9,10
FROZEN_LAM       := 300.0
PARETO_LAMS      := 1.0 3.0 10.0 30.0 100.0 300.0
CANON_DATA_SEED  := 99

figures: figures/fig1_forest.pdf figures/fig2_pareto.pdf figures/fig3_decile.pdf

figures/fig1_forest.pdf:
	uv run python scripts/make_fig1_forest.py

figures/fig2_pareto.pdf:
	uv run python scripts/make_fig2_pareto.py

figures/fig3_decile.pdf:
	uv run python scripts/make_fig3_decile.py

tables: paper/sections/tables/main.tex paper/sections/tables/fragility_magnitudes.tex

paper/sections/tables/main.tex:
	uv run python scripts/make_main_table.py

paper/sections/tables/fragility_magnitudes.tex:
	uv run python scripts/make_fragility_magnitudes_table.py


# === Training sweeps (long-running; produce NPZs in outputs/cross_sample/) ===

sweep-erm:
	@for ds in $(DATASETS_ALL); do \
	  uv run python scripts/cross_sample_train.py \
	    --dataset $$ds --canonical_data_seed $(CANON_DATA_SEED) \
	    --train_seeds $(SEEDS) --mode erm; \
	done

sweep-bagging:
	@for ds in $(DATASETS_ALL); do \
	  for K in 2 5; do \
	    uv run python scripts/cross_sample_train.py \
	      --dataset $$ds --canonical_data_seed $(CANON_DATA_SEED) \
	      --train_seeds $(SEEDS) --mode bagging --K $$K; \
	  done; \
	done

sweep-deep-ensemble:
	@for ds in $(DATASETS_ALL); do \
	  uv run python scripts/cross_sample_train.py \
	    --dataset $$ds --canonical_data_seed $(CANON_DATA_SEED) \
	    --train_seeds $(SEEDS) --mode deep_ensemble --K 5; \
	done

sweep-twin-indep:
	@for ds in $(DATASETS_HEADLINE); do \
	  uv run python scripts/cross_sample_train.py \
	    --dataset $$ds --canonical_data_seed $(CANON_DATA_SEED) \
	    --train_seeds $(SEEDS) --mode twin_indep --lam $(FROZEN_LAM); \
	done

sweep-pareto-bace:
	@for lam in $(PARETO_LAMS); do \
	  uv run python scripts/cross_sample_train.py \
	    --dataset $(DEV_DATASET) --canonical_data_seed $(CANON_DATA_SEED) \
	    --train_seeds $(SEEDS) --mode twin_indep --lam $$lam; \
	done


help:
	@echo "Development:"
	@echo "  install          install deps + pre-commit hooks"
	@echo "  lint / format    ruff check / fix"
	@echo "  test             pytest"
	@echo ""
	@echo "Paper artifacts (re-runnable from saved NPZs in outputs/cross_sample/):"
	@echo "  figures          Fig 1 forest, Fig 2 Pareto, Fig 3 decile"
	@echo "  tables           main results table, fragility-magnitudes table"
	@echo ""
	@echo "Training sweeps (long, GPU recommended; produce NPZs):"
	@echo "  sweep-erm sweep-bagging sweep-deep-ensemble sweep-twin-indep"
	@echo "  sweep-pareto-bace  development λ sweep on BACE only"
