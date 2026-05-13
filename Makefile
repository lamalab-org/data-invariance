.PHONY: install lint format test check figures tables analysis macros \
        sweep-erm sweep-bagging sweep-deep-ensemble sweep-twin-indep \
        sweep-pareto-bace sweep-borderline sweep-excluded \
        bo-loop-regression bo-topk \
        bayes-twin bayes-twin-bo bayes-twin-aggregate \
        bayes-twin-retrain bayes-twin-table help

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

figures: paper/figures/fig0_overview.pdf paper/figures/fig1_forest.pdf \
         paper/figures/fig2_pareto.pdf paper/figures/fig3_decile.pdf \
         paper/figures/fig4_case_study.pdf paper/figures/fig5_overlap.pdf \
         paper/figures/fig_convergence.pdf

paper/figures/fig0_overview.pdf:
	uv run python scripts/make_fig0_overview.py

paper/figures/fig1_forest.pdf:
	uv run python scripts/make_fig1_forest.py

paper/figures/fig2_pareto.pdf:
	uv run python scripts/make_fig2_pareto.py

paper/figures/fig3_decile.pdf:
	uv run python scripts/make_fig3_decile.py

paper/figures/fig4_case_study.pdf:
	uv run python scripts/make_fig4_case_study.py

paper/figures/fig5_overlap.pdf:
	uv run python scripts/make_fig5_overlap.py

paper/figures/fig_convergence.pdf:
	uv run python scripts/make_fig_convergence.py

tables: paper/sections/tables/main.tex paper/sections/tables/fragility_magnitudes.tex \
        paper/sections/tables/distributional.tex \
        paper/sections/tables/regression.tex \
        paper/sections/tables/chemberta.tex \
        paper/sections/tables/waterbirds_lambda.tex \
        paper/sections/tables/filter_outcomes.tex \
        paper/sections/tables/additional_metrics.tex \
        paper/sections/tables/per_class_churn.tex \
        paper/sections/tables/gin.tex \
        paper/sections/tables/gin_lambda.tex \
        paper/sections/tables/entropy_vs_fragility.tex \
        paper/sections/tables/overlap_spectrum.tex \
        paper/sections/tables/nscaling_bace.tex \
        paper/sections/tables/seed_sensitivity.tex \
        paper/sections/tables/bo_topk.tex \
        paper/sections/tables/bo_loop_regression.tex \
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

paper/sections/tables/gin.tex:
	uv run python scripts/analyze_bace_gin.py

paper/sections/tables/gin_lambda.tex:
	uv run python scripts/analyze_gin_lambda.py

paper/sections/tables/entropy_vs_fragility.tex:
	uv run python scripts/make_entropy_vs_fragility.py

paper/sections/tables/overlap_spectrum.tex:
	uv run python scripts/make_fig5_overlap.py

paper/sections/tables/nscaling_bace.tex:
	uv run python scripts/make_nscaling_bace.py

paper/sections/tables/seed_sensitivity.tex:
	uv run python scripts/make_seed_sensitivity_table.py

# Bayesian-optimisation analogue (App. M): top-K Jaccard stability.
# Reads the existing cross-sample NPZs only (no new training).
paper/sections/tables/bo_topk.tex:
	uv run python scripts/make_bo_topk_table.py

# Bayesian-optimisation analogue (App. N): full BO-loop trajectory
# variance on regression.  Depends on outputs/bo_loop_regression.json,
# which the expensive `bo-loop-regression` target produces (~2 hours
# CPU).  If the JSON is missing, the rule will fail; run
# `make bo-loop-regression` first.
paper/sections/tables/bo_loop_regression.tex: outputs/bo_loop_regression.json
	uv run python scripts/make_bo_loop_regression_table.py

# Auto-generated \newcommand-per-quoted-number macros so paper prose
# never drifts from source.  Always rebuilt after any analysis script.
.PHONY: macros
macros: paper/sections/macros.tex
paper/sections/macros.tex: outputs/main_table.csv outputs/convergence_recall.csv \
                          outputs/entropy_vs_fragility.csv outputs/distributional.csv \
                          outputs/fragility_magnitudes.csv outputs/regression.csv \
                          outputs/friedman.csv outputs/chemberta_heldout.csv \
                          outputs/waterbirds_lambda.csv outputs/bace_gin.csv \
                          outputs/gin_lambda.csv outputs/filter_outcomes.csv \
                          outputs/bo_topk.csv \
                          outputs/bo_loop_regression_summary.csv
	uv run python scripts/make_paper_macros.py

# Plain-text and LaTeX-with-numbers renders of the abstract for
# pasting into submission forms that don't accept \input{macros}.
.PHONY: abstract
abstract:
	uv run python scripts/render_abstract.py

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
	uv run python scripts/make_seed_sensitivity_table.py
	uv run python scripts/make_bo_topk_table.py
	@if [ -f outputs/bo_loop_regression.json ]; then \
	  uv run python scripts/make_bo_loop_regression_table.py; \
	else \
	  echo "[skip] bo_loop_regression: outputs/bo_loop_regression.json missing; run 'make bo-loop-regression' first"; \
	fi
	uv run python scripts/make_paper_macros.py


# Bayesian-optimisation trajectory variance on the three regression
# datasets (App. N).  Expensive: ~2 hours on a single laptop CPU
# (180 BO trajectories × 10 retrainings each, 3 datasets × 3 methods).
# Produces outputs/bo_loop_regression.json which feeds the table-emitter.
# Pass --K, --budget, --initial_n via BO_LOOP_ARGS to override.
BO_LOOP_ARGS ?= --K 20 --budget 10 --initial_n 50
bo-loop-regression:
	uv run python scripts/bo_loop_regression.py $(BO_LOOP_ARGS)
	uv run python scripts/make_bo_loop_regression_table.py
	uv run python scripts/make_paper_macros.py


# === Per-dataset Bayesian optimisation of lambda (App. bayes_twin) ===
#
# Two-stage protocol (the apples-to-apples version):
#   1. per-seed BO with the cross-sample-churn objective; for each
#      (dataset, train_seed) the BO returns lambda*_s.
#   2. retrain twin-bootstrap at the per-dataset median(lambda*_s)
#      across all train_seeds so cross-sample churn between any two
#      retrainings is fixed-lambda variance.
#
# Stages 1 and 2 are cluster-bound; the local Makefile targets either
# print the submission command or run the cheap local post-processing.
# Run them in order:
#
#   make bayes-twin-bo          # prints stage-1 cluster command
#   make bayes-twin-aggregate   # runs stage-1.5: median per dataset (local)
#   make bayes-twin-retrain     # prints stage-2 cluster command
#   make bayes-twin-table       # rebuilds tab:bayes_twin + macros (local)
#
# or just `make bayes-twin` for an end-to-end checklist.
BAYES_DATASETS := $(DEV_DATASET) dili pgp_broccatelli bbb_martins bbbp tadf \
                  mof_thermal ames cyp2d6_substrate
BAYES_CROSS_ROOT     := outputs/cross_sample_bayes_cross
BAYES_PERDS_ROOT     := outputs/cross_sample_bayes_perdataset
BAYES_LAM300_ROOT    := outputs/cross_sample
BAYES_PERDS_LAM_CSV  := outputs/bo_perdataset_lambdas.csv

bayes-twin:
	@echo "Per-dataset BO protocol (App. bayes_twin):"
	@echo "  1. make bayes-twin-bo        # stage 1: cluster"
	@echo "  2. make bayes-twin-aggregate # local"
	@echo "  3. make bayes-twin-retrain   # stage 2: cluster"
	@echo "  4. make bayes-twin-table     # local"

bayes-twin-bo:
	@echo "Submit stage 1 on the cluster:"
	@echo "  BO_OBJECTIVE=cross_sample_constrained CV_FOLDS=3 LAM_MAX=1e4 \\"
	@echo "    sbatch slurm/full_retraining/10_bayes_twin_headline.sh"
	@echo "Then rsync $(BAYES_CROSS_ROOT)/ back and run: make bayes-twin-aggregate"

# Stage 1.5: aggregate per-seed BO selections to one lambda per dataset.
bayes-twin-aggregate: $(BAYES_PERDS_LAM_CSV)
$(BAYES_PERDS_LAM_CSV):
	uv run python scripts/select_perdataset_lambda.py \
	  --source $(BAYES_CROSS_ROOT) \
	  --csv $(BAYES_PERDS_LAM_CSV)

bayes-twin-retrain: $(BAYES_PERDS_LAM_CSV)
	@echo "Submit stage 2 on the cluster (reads $(BAYES_PERDS_LAM_CSV)):"
	@echo "  sbatch slurm/full_retraining/11_bo_perdataset_lam.sh"
	@echo "Then rsync $(BAYES_PERDS_ROOT)/ back and run: make bayes-twin-table"

# Build Table~\ref{tab:bayes_twin} from the stage-2 NPZs and refresh
# the macros so the appendix prose's head-counts (beats / ties /
# loses) flow from the data.  Requires stage 2 to be complete.
bayes-twin-table:
	uv run python scripts/make_bayes_table.py \
	  --roots bo_perds=$(BAYES_PERDS_ROOT) lam300=$(BAYES_LAM300_ROOT) \
	  --datasets $(BAYES_DATASETS) \
	  --erm-root $(BAYES_LAM300_ROOT) \
	  --baseline-run lam300 \
	  --baseline-runs lam300 \
	  --compare-run bo_perds \
	  --csv outputs/bayes_table.csv \
	  --latex paper/sections/tables/bayes_twin.tex
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
	@echo "  sweep-pareto-bace     development λ sweep on BACE only"
	@echo ""
	@echo "Standalone BO trajectory simulation (~2 h CPU; produces"
	@echo "outputs/bo_loop_regression.json):"
	@echo "  bo-loop-regression"
	@echo ""
	@echo "Per-dataset BO of lambda (App. bayes_twin; two cluster sweeps + local):"
	@echo "  bayes-twin                end-to-end checklist"
	@echo "  bayes-twin-bo             stage 1 (cluster command)"
	@echo "  bayes-twin-aggregate      aggregate per-seed selections (local)"
	@echo "  bayes-twin-retrain        stage 2 (cluster command)"
	@echo "  bayes-twin-table          rebuild tab:bayes_twin + macros (local)"
