#!/bin/bash

# Evaluate HJEEDS jeeds_mean_sigma values on the fixed MAXG benchmark.
# Submit with `sbatch run_hjeeds_maxg.sbatch` or override BH_* variables with
# `sbatch --export=ALL,BH_PIDS_FILE=... run_hjeeds_maxg.sbatch`.

#SBATCH --job-name=hjeeds-maxg
#SBATCH --time=32:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm-hjeeds-maxg-%j.out

set -euo pipefail

cd "${BH_REPO_ROOT:-$HOME/research/SkillEstimationFramework2}"

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate "${BH_CONDA_ENV:-skill-estimation}"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MPLBACKEND=Agg

SEASON="${BH_SEASON:-20242025}"
SHOT_GROUP="${BH_SHOT_GROUP:-wristshot_snapshot}"
BENCHMARK_TAG="${BH_BENCHMARK_TAG:-wristshot_snapshot_v1}"
BENCHMARK_DIR="${BH_BENCHMARK_DIR:-Data/Hockey/benchmarks}"
DATA_DIR="${BH_DATA_DIR:-Data/Hockey}"
HJEEDS_RESULTS="${BH_HJEEDS_RESULTS:-Data/Hockey/experiments/hjeeds_20242025/agent_level_results.csv}"
PIDS_FILE="${BH_PIDS_FILE:-Data/Hockey/forwards23-25.txt}"
OUTPUT_DIR="${BH_OUTPUT_DIR:-Data/Hockey/benchmarks/results}"

for required_path in "$HJEEDS_RESULTS" "$PIDS_FILE"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Required file not found: $required_path" >&2
    exit 1
  fi
done

python -m BlackhawksSkillEstimation.maxg_evaluator \
  --estimator hjeeds \
  --hjeeds-results "$HJEEDS_RESULTS" \
  --benchmark-tag "$BENCHMARK_TAG" \
  --benchmark-dir "$BENCHMARK_DIR" \
  --season-tag "$SEASON" \
  --shot-group "$SHOT_GROUP" \
  --data-dir "$DATA_DIR" \
  --pids-file "$PIDS_FILE" \
  --output-dir "$OUTPUT_DIR"