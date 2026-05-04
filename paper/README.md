# Paper directory — NeurIPS 2026

## Contribution type

Use-Inspired (NeurIPS 2026 reviewer guidelines): the work frames
methods around small-N from-scratch scientific ML and reports
empirical evidence on chemistry benchmarks (MoleculeNet, TDC ADME/Tox,
materials).  Reviewers can evaluate whether the use case is genuine
(Section 1), the methods align with domain needs (Sections 5–6), and
the work yields insight beyond the application (Sections 3, 7).

## Build

```
cd paper/
latexmk -pdf main.tex
```

Body fits the NeurIPS 9-page budget (verified via temporary
`\label{end-of-body}`).  Total PDF is ~30 pages including references,
appendix, and reproducibility checklist.

## Layout

```
main.tex                  entry point + package preamble + section ordering
neurips_2026.sty          style file
checklist.tex             NeurIPS reproducibility checklist (mandatory)
references.bib            bibliography
sections/
  abstract.tex
  introduction.tex
  related_work.tex
  measurement.tex         Section 3 (defines cross-sample churn + protocol)
  magnitudes.tex          Section 4 (the headline magnitudes table)
  methods.tex             Section 5 (six baselines + twin-bootstrap)
  experiments.tex         Section 6 (paired Δ + cross-architecture)
  scope.tex               Section 7 (Limitations)
  discussion.tex          Section 8
  appendix.tex            13 appendix sections
  macros.tex              auto-generated; every quoted prose number
  tables/                 auto-generated table fragments
figures/                  auto-generated figures (fig0_overview ... fig5_overlap)
```

## Tables, figures, and macros

Every paper table, figure, and quoted number is regenerated from the
saved NPZs in `../outputs/cross_sample{,_seed7,_seed42}/` by the
analysis scripts in `../scripts/`.  No retraining required to rebuild
the PDF:

```
make analysis     # CSVs from NPZs
make tables       # paper/sections/tables/*.tex from CSVs (also runs macros)
make figures      # paper/figures/*.pdf
```

Numerical-claim audit invariant: `paper/sections/macros.tex` defines
124 `\newcommand`s; all 124 are referenced in the prose, tables, or
figure captions.  The check lives at the top of
`scripts/make_paper_macros.py`.
