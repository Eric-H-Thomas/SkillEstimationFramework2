
# Script guide

Run these commands from the repository root. Slurm launchers, their `run_*.sbatch`
workers, and the unified 1D suite runner live at the repository root; the audit and
data-preparation utilities below live under `scripts/`.

## Publication runners

| Script | Purpose |
|---|---|
| `scripts/runners/run_publication_bench.py` | Runs local 1D/2D audits; its optional MLB component only validates an existing completed Slurm paper endpoint and never launches the MLB workflow. |
| `run_hjeeds_paper_experiments.py` | Runs, submits, aggregates, plots, or archives the complete 1D-Darts experiment suite. |

## Slurm launchers

| Script | Purpose |
|---|---|
| `submit_hjeeds_2d_cluster_tests.sh` | Submits and atomically seals the canonical 2D-Darts run (seeds 1000--1499, 25 agents/seed) with source/configuration and artifact hashes. |
| `submit_hjeeds_baseball_convergence_paper_bbip.sh` | Reproduces the paper's MLB processed-data walk/IP-proxy study. |
| `submit_hjeeds_baseball_convergence_array.sh` | Configurable MLB convergence-array launcher used by the paper wrapper. |
| `submit_hjeeds_baseball_hyperprior_calibration.sh` | Optional exploratory all-eligible 2021 FF hyperprior calibration; not used by the publication workflow. |
| `regenerate_2d_paper_figure.sh` | Chains the 2D array, aggregation, and figure-render jobs to rebuild the 2D paper figure after the execution-noise correction. |

Files named `run_*.sbatch` are workers invoked by the corresponding `submit_*.sh` launcher. `hjeeds_baseball_slurm_common.sh` contains shared cluster helpers and is not a standalone experiment.

## Utilities

| Script | Purpose |
|---|---|
| `scripts/benchmark_baseball_runtime_optimizations.py` | Compares legacy and batched MLB RNN inference plus repeated and cumulative likelihood checkpoints on a small number of real pitches; never writes publication results. |
| `scripts/compare_seeded_results.py` | Compares selected seed-level candidate CSVs with a larger reference run. |
| `scripts/prepare_baseball_data.py` | Recreates the historical pickle shape for explicit non-paper exploration; it cannot reproduce the model-paired canonical MLB artifact. |
| `scripts/validate_publication_results.py` | Rejects incomplete, failed, stale, or nonconverged 1D publication result trees before plotting or packaging. Encodes the frozen 93-scenario paper design, so it is not a general-purpose result checker. |
