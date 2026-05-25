#!/bin/bash

#SBATCH --job-name=bh-maxg
#SBATCH --time=32:00:00
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

MODE="${BH_MODE:-old}"
COMPARISON_MODEL="${BH_COMPARISON_MODEL:-legacy}"
BENCHMARK_TAG="${BH_BENCHMARK_TAG:-wristshot_snapshot_xgnew_v1}"
BENCHMARK_DIR="${BH_BENCHMARK_DIR:-Data/Hockey_xg_new/benchmarks}"
SHOT_GROUP="${BH_SHOT_GROUP:-wristshot_snapshot}"
PIDS_FILE="${BH_PIDS_FILE:-Data/Hockey/forwards23-25.txt}"

case "$MODE" in
  old)
    DATA_DIR="${BH_DATA_DIR:-Data/Hockey_xg_new}"
    OUTPUT_DIR="${BH_OUTPUT_DIR:-Data/Hockey_xg_new/benchmarks/results}"
    ;;
  comparison)
    DATA_DIR="${BH_DATA_DIR:-Data/Hockey_model_comparison}"
    OUTPUT_BASE="${BH_OUTPUT_BASE:-Data/Hockey_xg_new/benchmarks/results_comparison}"
    ;;
  *)
    echo "Invalid BH_MODE=$MODE (expected old or comparison)" >&2
    exit 1
    ;;
esac

SEASONS=(20212022 20222023 20242025 20252026)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if (( TASK_ID < 0 || TASK_ID >= ${#SEASONS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID} for ${#SEASONS[@]} seasons" >&2
  exit 1
fi

SEASON="${SEASONS[$TASK_ID]}"

if [[ "$MODE" == "comparison" ]]; then
  for MODEL in legacy new; do
    MODEL_OUTPUT_DIR="${OUTPUT_BASE}/${MODEL}"
    echo "Running comparison pass for model: $MODEL -> output: $MODEL_OUTPUT_DIR"
    python -m BlackhawksSkillEstimation.maxg_evaluator \
      --benchmark-tag "$BENCHMARK_TAG" \
      --benchmark-dir "$BENCHMARK_DIR" \
      --season-tag "$SEASON" \
      --shot-group "$SHOT_GROUP" \
      --data-dir "$DATA_DIR" \
      --output-dir "$MODEL_OUTPUT_DIR" \
      --pids-file "$PIDS_FILE" \
      --model-filter "$MODEL"
  done
else
  python -m BlackhawksSkillEstimation.maxg_evaluator \
    --benchmark-tag "$BENCHMARK_TAG" \
    --benchmark-dir "$BENCHMARK_DIR" \
    --season-tag "$SEASON" \
    --shot-group "$SHOT_GROUP" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --pids-file "$PIDS_FILE"
fi
