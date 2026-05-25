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
SUBMIT_OLD_BENCHMARK="${BH_SUBMIT_OLD_BENCHMARK:-0}"
OLD_BENCHMARK_TAG="${BH_OLD_BENCHMARK_TAG:-wristshot-snapshot_v1}"
OLD_BENCHMARK_DIR="${BH_OLD_BENCHMARK_DIR:-Data/Hockey/benchmarks}"
OLD_OUTPUT_BASE="${BH_OLD_OUTPUT_BASE:-Data/Hockey_xg_new/benchmarks/results_old_benchmark}"
COMPARISON_DATA_DIR="${BH_DATA_DIR:-Data/Hockey_model_comparison}"
COMPARISON_OUTPUT_BASE="${BH_OUTPUT_BASE:-Data/Hockey_xg_new/benchmarks/results_comparison}"

submit_comparison_job() {
  local benchmark_tag="$1"
  local benchmark_dir="$2"
  local data_dir="$3"
  local output_base="$4"
  local job_name="$5"

  sbatch \
    --array="0-3" \
    --time="32:00:00" \
    --mem="16G" \
    --job-name="$job_name" \
    --output="slurm-${job_name}-%A_%a.out" \
    --export=ALL,BH_WORKER_MODE=1,BH_MODE=comparison,BH_BENCHMARK_TAG="$benchmark_tag",BH_BENCHMARK_DIR="$benchmark_dir",BH_SHOT_GROUP="$SHOT_GROUP",BH_DATA_DIR="$data_dir",BH_OUTPUT_BASE="$output_base",BH_PIDS_FILE="$PIDS_FILE" \
    "$0"
}

if [[ "$MODE" == "comparison" && -z "${BH_WORKER_MODE:-}" && "$SUBMIT_OLD_BENCHMARK" != 0 ]]; then
  echo "Submitting comparison benchmark job: $BENCHMARK_TAG -> $COMPARISON_OUTPUT_BASE"
  submit_comparison_job "$BENCHMARK_TAG" "$BENCHMARK_DIR" "$COMPARISON_DATA_DIR" "$COMPARISON_OUTPUT_BASE" "bh-maxg-comparison"

  echo "Submitting old benchmark job: $OLD_BENCHMARK_TAG -> $OLD_OUTPUT_BASE"
  submit_comparison_job "$OLD_BENCHMARK_TAG" "$OLD_BENCHMARK_DIR" "$COMPARISON_DATA_DIR" "$OLD_OUTPUT_BASE" "bh-maxg-old-benchmark"

  exit 0
fi

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
