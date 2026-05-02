.PHONY: install lint format test check figures tables analysis macros \
        sweep-erm sweep-bagging sweep-deep-ensemble sweep-twin-indep \
        sweep-pareto-bace sweep-borderline sweep-excluded help

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

tables: paper/sections/tables/main.tex paper/sections/tables/fragility_magnitudes.tex \
        paper/sections/tables/distributional.tex \
        paper/sections/tables/regression.tex \
        paper/sections/tables/chemberta.tex \
        paper/sections/tables/waterbirds_lambda.tex \
        paper/sections/tables/filter_outcomes.tex \
        paper/sections/tables/additional_metrics.tex \
        paper/sections/tables/per_class_churn.tex \
        paper/sections/tables/entropy_vs_fragility.tex \
        paper/sections/tables/overlap_spectrum.tex \
        paper/sections/tables/nscaling_bace.tex \
        macros

paper/sections/tables/main.tex:
	uv run python scripts/make_main_table.py

paper/sections/tables/fragility_magnitudes.tex:
	uv run python scripts/make_fragility_magnitudes_table.py

paper/sections/tables/distributional.tex:
	uv run python scripts/make_distributional_table.py

paper/sections/tables/regression.tex:
	uv run python scripts/make_regression_table.py

paper/sections/tables/chemberta.tex:
	uv run python scripts/analyze_chemberta_heldout.py

paper/sections/tables/waterbirds_lambda.tex:
	uv run python scripts/analyze_waterbirds_lambda.py

paper/sections/tables/filter_outcomes.tex:
	uv run python scripts/make_filter_outcomes_table.py

paper/sections/tables/additional_metrics.tex:
	uv run python scripts/make_additional_metrics_table.py

paper/sections/tables/per_class_churn.tex:
	uv run python scripts/make_per_class_churn_table.py

paper/sections/tables/entropy_vs_fragility.tex:
	uv run python scripts/make_entropy_vs_fragility.py

paper/sections/tables/overlap_spectrum.tex:
	uv run python scripts/make_fig5_overlap.py

paper/sections/tables/nscaling_bace.tex:
	uv run python scripts/make_nscaling_bace.py

# Auto-generated \newcommand-per-quoted-number macros so paper prose
# never drifts from source.  Always rebuilt after any analysis script.
.PHONY: macros
macros: paper/sections/macros.tex
paper/sections/macros.tex: outputs/main_table.csv outputs/convergence_recall.csv \
                          outputs/entropy_vs_fragility.csv outputs/distributional.csv \
                          outputs/fragility_magnitudes.csv outputs/regression.csv \
                          outputs/friedman.csv outputs/chemberta_heldout.csv \
                          outputs/waterbirds_lambda.csv outputs/bace_gin.csv \
                          outputs/gin_lambda.csv outputs/filter_outcomes.csv
	uv run python scripts/make_paper_macros.py

# Analysis-only target: regenerate every CSV (and table-fragment)
# from saved NPZs.  Cheap (no retraining).  Use after a sweep finishes.
analysis:
	uv run python scripts/make_main_table.py
	uv run python scripts/make_fragility_magnitudes_table.py
	uv run python scripts/make_distributional_table.py
	uv run python scripts/make_regression_table.py
	uv run python scripts/make_filter_outcomes_table.py
	uv run python scripts/make_additional_metrics_table.py
	uv run python scripts/make_per_class_churn_table.py
	uv run python scripts/make_friedman_test.py
	uv run python scripts/analyze_chemberta_heldout.py
	uv run python scripts/analyze_waterbirds_lambda.py
	uv run python scripts/analyze_bace_gin.py
	uv run python scripts/analyze_gin_lambda.py
	uv run python scripts/make_entropy_vs_fragility.py
	uv run python scripts/make_fig_convergence.py
	uv run python scripts/make_paper_macros.py


# === Training sweeps (long-running; produce NPZs in outputs/cross_sample/) ===

DATASETS_BORDERLINE := skin_reaction herg hia_hou
DATASETS_EXCLUDED   := bioavailability_ma mof_solvent cyp2c9_substrate cyp3a4_substrate clintox

sweep-erm:
	@for ds in $(DATASETS_ALL); do \
	  uv run python scripts/cross_sample_train.py \
	    --dataset $$ds --canonical_data_seed $(CANON_DATA_SEED) \
	    --train_seeds $(SEEDS) --mode erm; \
	done

sweep-mc-dropout:
	@for ds in $(DATASETS_ALL); do \
	  uv run python scripts/cross_sample_train.py \
	    --dataset $$ds --canonical_data_seed $(CANON_DATA_SEED) \
	    --train_seeds $(SEEDS) --mode mc_dropout --K 20; \
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

# Borderline + excluded datasets: only ERM (10 seeds), so the
# fragility-magnitudes table can include the bottom group and the
# filter-outcomes table is fully populated.
sweep-borderline:
	@for ds in $(DATASETS_BORDERLINE); do \
	  uv run python scripts/cross_sample_train.py \
	    --dataset $$ds --canonical_data_seed $(CANON_DATA_SEED) \
	    --train_seeds $(SEEDS) --mode erm; \
	done

sweep-excluded:
	@for ds in $(DATASETS_EXCLUDED); do \
	  uv run python scripts/cross_sample_train.py \
	    --dataset $$ds --canonical_data_seed $(CANON_DATA_SEED) \
	    --train_seeds $(SEEDS) --mode erm; \
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
