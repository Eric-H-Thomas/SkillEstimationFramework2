#!/bin/bash

#SBATCH --job-name=bh-maxg-bootstrap
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --output=slurm-bh-maxg-bootstrap-%j.out

set -euo pipefail

cd ~/research/SkillEstimationFramework2

MODE="${BH_MODE:-comparison}"
if [[ "$MODE" != "comparison" ]]; then
  echo "submit_maxg_comparison.sbatch only supports BH_MODE=comparison" >&2
  exit 1
fi

BENCHMARK_TAG="${BH_BENCHMARK_TAG:-wristshot_snapshot_xgnew_v1}"
BENCHMARK_DIR="${BH_BENCHMARK_DIR:-Data/Hockey_xg_new/benchmarks}"
OLD_BENCHMARK_TAG="${BH_OLD_BENCHMARK_TAG:-wristshot-snapshot_v1}"
OLD_BENCHMARK_DIR="${BH_OLD_BENCHMARK_DIR:-Data/Hockey/benchmarks}"
SHOT_GROUP="${BH_SHOT_GROUP:-wristshot_snapshot}"
PIDS_FILE="${BH_PIDS_FILE:-Data/Hockey/forwards23-25.txt}"
DATA_DIR="${BH_DATA_DIR:-Data/Hockey/xg_legacy_new_comparison}"
OUTPUT_BASE="${BH_OUTPUT_BASE:-Data/Hockey_xg_new/benchmarks/results_comparison}"
OLD_OUTPUT_BASE="${BH_OLD_OUTPUT_BASE:-Data/Hockey_xg_new/benchmarks/results_old_benchmark}"
ARRAY_SPEC="${BH_ARRAY_SPEC:-0-3}"
WORKER_TIME="${BH_WORKER_TIME:-32:00:00}"
WORKER_MEM="${BH_WORKER_MEM:-16G}"
SUBMIT_OLD_BENCHMARK="${BH_SUBMIT_OLD_BENCHMARK:-1}"

submit_job() {
  local benchmark_tag="$1"
  local benchmark_dir="$2"
  local output_base="$3"
  local job_name="$4"

  sbatch \
    --array="$ARRAY_SPEC" \
    --time="$WORKER_TIME" \
    --mem="$WORKER_MEM" \
    --job-name="$job_name" \
    --output="slurm-${job_name}-%A_%a.out" \
    --export=ALL,BH_WORKER_MODE=1,BH_MODE=comparison,BH_BENCHMARK_TAG="$benchmark_tag",BH_BENCHMARK_DIR="$benchmark_dir",BH_SHOT_GROUP="$SHOT_GROUP",BH_DATA_DIR="$DATA_DIR",BH_OUTPUT_BASE="$output_base",BH_PIDS_FILE="$PIDS_FILE" \
    run_maxg_evaluator_seasons.sbatch
}

echo "Submitting comparison benchmark job: $BENCHMARK_TAG -> $OUTPUT_BASE"
submit_job "$BENCHMARK_TAG" "$BENCHMARK_DIR" "$OUTPUT_BASE" "bh-maxg-comparison"

if [[ "$SUBMIT_OLD_BENCHMARK" != 0 ]]; then
  echo "Submitting old benchmark job: $OLD_BENCHMARK_TAG -> $OLD_OUTPUT_BASE"
  submit_job "$OLD_BENCHMARK_TAG" "$OLD_BENCHMARK_DIR" "$OLD_OUTPUT_BASE" "bh-maxg-old-benchmark"
fi
