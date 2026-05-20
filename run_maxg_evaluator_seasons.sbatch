#!/bin/bash

#SBATCH --job-name=bh-maxg
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=slurm-maxg-%A_%a.out
#SBATCH --array=0-3

set -euo pipefail

cd ~/research/SkillEstimationFramework2

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate skill-estimation

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MPLBACKEND=Agg

BENCHMARK_TAG="wristshot_snapshot_xgnew_v1"
BENCHMARK_DIR="Data/Hockey_xg_new/benchmarks"
SHOT_GROUP="wristshot_snapshot"
DATA_DIR="Data/Hockey_xg_new"
OUTPUT_DIR="Data/Hockey_xg_new/benchmarks/results"
PIDS_FILE="Data/Hockey/forwards23-25.txt"

SEASONS=(20212022 20222023 20242025 20252026)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if (( TASK_ID < 0 || TASK_ID >= ${#SEASONS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID} for ${#SEASONS[@]} seasons" >&2
  exit 1
fi

SEASON="${SEASONS[$TASK_ID]}"

python -m BlackhawksSkillEstimation.maxg_evaluator \
  --benchmark-tag "$BENCHMARK_TAG" \
  --benchmark-dir "$BENCHMARK_DIR" \
  --season-tag "$SEASON" \
  --shot-group "$SHOT_GROUP" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --pids-file "$PIDS_FILE"
