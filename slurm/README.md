# Running on a SLURM cluster

## One-time setup

```bash
git clone <repo-url> data-invariance
cd data-invariance
bash slurm/setup.sh
```

This installs `uv` and all Python dependencies. Verify with:
```bash
uv run python -c "from train import *; print('OK')"
```

## Running all experiments

```bash
# GPU experiments (Waterbirds, CelebA, CivilComments) — 3 jobs, ~12h max
sbatch slurm/run_gpu.sh

# CPU experiments (10 datasets) — 10 parallel jobs, ~4h max
sbatch slurm/run_cpu.sh
```

Each SLURM array job runs one dataset independently. All 13 jobs can
run in parallel if resources are available.

## Checking progress

```bash
squeue -u $USER                    # job status
tail -f logs/slurm_<jobid>_0.out   # live output for array index 0
```

## Collecting results

After all jobs finish:
```bash
bash slurm/collect_results.sh > results_summary.txt
```

This prints the full results table with 4 columns per method:
- **WGA-sel**: model selected by worst-group val accuracy (uses group labels)
- **SWA+WGA**: SWA averaged around WGA-selected epoch (uses group labels)
- **Free-sel**: model selected by average val accuracy (group-free)
- **SWA+Free**: SWA averaged around Free-selected epoch (group-free)

The **SWA+Free** column is the paper's headline: truly group-free results.

## Datasets

| Job | Dataset | Type | N | Time est. |
|-----|---------|------|---|-----------|
| GPU 0 | Waterbirds | Vision | 4.8K | ~2h |
| GPU 1 | CelebA | Vision | 163K | ~8h |
| GPU 2 | CivilComments | NLP | 269K | ~4h |
| CPU 0 | CMNIST | Synthetic | 60K | ~5min |
| CPU 1 | Multi-CMNIST | Synthetic | 60K | ~10min |
| CPU 2 | TADF | Chemistry | 2K | ~1min |
| CPU 3 | MOF thermal | Chemistry | 3K | ~1min |
| CPU 4 | MOF solvent | Chemistry | 2K | ~1min |
| CPU 5 | Perovskite | Chemistry | 48K | ~5min |
| CPU 6 | Battery | Chemistry | 40K | ~5min |
| CPU 7 | BACE | MoleculeNet | 1.5K | ~2min |
| CPU 8 | BBBP | MoleculeNet | 2K | ~3min |
| CPU 9 | HIV | MoleculeNet | 41K | ~15min |

## Adjusting for your cluster

You may need to change in `run_gpu.sh` / `run_cpu.sh`:
- `--partition`: your GPU/CPU partition name
- `--gres`: GPU type (e.g., `gpu:a100:1`)
- `--mem`: memory per job
- `--time`: walltime limit

For chemistry datasets, you need the parquet files from
[clever-materials-hans](https://github.com/lamalab-org/clever-materials-hans).
Update the paths in `configs/dataset/{tadf,mof_thermal,mof_solvent,perovskite,battery}.yaml`.
