"""Single source of truth for paper-level dataset and method choices.

Tables, figures, and the methods text in `paper/sections/` import from here
so that any change to the dataset split, the frozen hyperparameter, or the
method ordering propagates everywhere. Keeping this in one module avoids
silent disagreement between Table 1 and Figure 1.

Decisions documented (2026-04-27):

  * Bioavailability_Ma and MOF_solvent are dropped from headline tables and
    figures: ERM on these datasets does not exceed majority-class accuracy,
    so cross-sample churn measurements would conflate "method shifts the
    decision boundary" with "majority-class shuffling under noise".

  * hERG, HIA_Hou, and Skin_Reaction are reported only in an appendix
    table: ERM exceeds majority by 3–4pp on test sets of 57–104 examples
    with severe class imbalance, which we do not consider sufficient
    statistical power for headline claims.

  * Waterbirds is reported only in the §7 scope analysis (it is a
    pretrained-backbone setting where twin-indep over-regularises).

  * The development dataset is BACE only.  λ is selected once on BACE and
    frozen for held-out evaluation.
"""

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

DEV_DATASET = "bace"

# Headline held-out chemistry datasets (passed the ERM > majority + 5pp check).
# Sorted by N (training-set size).
HEADLINE_DATASETS = [
    "dili",
    "pgp_broccatelli",
    "bbb_martins",
    "bbbp",
    "tadf",
    "mof_thermal",
    "ames",
]

# Borderline datasets (ERM > majority by < 5pp, or test set < 60 examples).
# Reported in an appendix table only.
BORDERLINE_DATASETS = [
    "skin_reaction",
    "herg",
    "hia_hou",
]

# Excluded entirely: ERM does not exceed majority-class baseline.
EXCLUDED_DATASETS = [
    "bioavailability_ma",
    "mof_solvent",
]

# Pretrained-backbone scope analysis only.
SCOPE_DATASETS = [
    "waterbirds",
]

# All chemistry datasets we ran (for the cross-dataset 1/N scaling figure
# and per-dataset health table).  Sorted by N.
ALL_CHEMISTRY = (
    [DEV_DATASET]
    + HEADLINE_DATASETS
    + BORDERLINE_DATASETS
    + EXCLUDED_DATASETS
)


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

# Glob pattern templates for `_analysis_lib.load_runs`.
METHOD_GLOBS = {
    "ERM":            "erm_train*.npz",
    "Deep Ens. K=5":   "deep_ensemble_train*_K5.npz",
    "Bagging K=2":     "bagging_train*_K2.npz",
    "Bagging K=5":     "bagging_train*_K5.npz",
    "Twin-indep":      "twin_indep_train*_lam{lam}.npz",
}

# Method order in the headline forest plot, top to bottom: weakest baseline
# at top, strongest contribution at bottom (where the eye lands last).
METHOD_ORDER = [
    "Deep Ens. K=5",
    "Bagging K=2",
    "Bagging K=5",
    "Twin-indep",
]


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Frozen by the pre-registered selection rule on the dev dataset (BACE).
FROZEN_LAM = 300.0

# Pareto curve on the dev dataset.
PARETO_LAMS = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]

# Number of train_seeds per (dataset, method) cell.
N_SEEDS = 10


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

DATASET_DISPLAY_NAMES = {
    "bace":               "BACE",
    "bbbp":               "BBBP",
    "tadf":               "TADF",
    "mof_thermal":        "MOF-thermal",
    "mof_solvent":        "MOF-solvent",
    "battery":            "Battery",
    "perovskite":         "Perovskite",
    "hia_hou":            "HIA",
    "bioavailability_ma": "Bioavailability",
    "pgp_broccatelli":    "Pgp",
    "bbb_martins":        "BBB-Martins",
    "herg":               "hERG",
    "dili":               "DILI",
    "ames":               "AMES",
    "skin_reaction":      "SkinReact",
    "waterbirds":         "Waterbirds",
}


def display(ds: str) -> str:
    return DATASET_DISPLAY_NAMES.get(ds, ds)


# Training-set sizes (canonical_data_seed=99).  Sourced once from
# `make_dataloaders`; kept here so figures don't reload datasets.
N_TRAIN = {
    "dili":               304,
    "skin_reaction":      232,
    "bace":               968,
    "bbbp":               1305,
    "tadf":               1007,
    "mof_thermal":        1251,
    "mof_solvent":        849,
    "ames":               4658,
    "hia_hou":            370,
    "herg":               420,
    "bioavailability_ma": 410,
    "pgp_broccatelli":    780,
    "bbb_martins":        1300,
    "waterbirds":         4795,
}


def glob_for(method: str, lam: float = FROZEN_LAM) -> str:
    return METHOD_GLOBS[method].format(lam=lam)
