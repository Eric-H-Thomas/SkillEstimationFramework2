#!/bin/bash
# Run the synchronized Blackhawks HJEEDS subsample experiment.
#
# One-command workflow: preflight submits the worker array and aggregation:
#   sbatch run_hjeeds_hockey_subsample.sbatch OUT preflight
#
# The preflight job scans the cache, then submits the array and aggregate job.
#
#SBATCH --job-name=hjeeds-hockey-subsample
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=slurm-hjeeds-hockey-subsample-%j.out

set -euo pipefail

OUTPUT_DIR="${1:?Usage: sbatch $0 OUTPUT_DIR MODE (preflight, worker, or aggregate)}"
MODE="${2:?Usage: sbatch $0 OUTPUT_DIR MODE (preflight, worker, or aggregate)}"
REPO_ROOT="${HJEEDS_REPO_ROOT:-$PWD}"

cd "$REPO_ROOT"
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate skill-estimation
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

COMMON_ARGS=(
  --output-dir "$OUTPUT_DIR"
  --players-file "${HJEEDS_HOCKEY_PLAYERS_FILE:-Data/Hockey/forwards23-25.txt}"
  --data-dir "${HJEEDS_HOCKEY_DATA_DIR:-Data/Hockey}"
  --shot-group "${HJEEDS_HOCKEY_SHOT_GROUP:-wristshot_snapshot}"
  --target-player-id "${HJEEDS_HOCKEY_TARGET_PLAYER_ID:-950160}"
  --base-seed "${HJEEDS_HOCKEY_BASE_SEED:-20260904}"
  --n-shots ${HJEEDS_HOCKEY_N_SHOTS:-100 200 400}
  --num-seeds "${HJEEDS_HOCKEY_NUM_SEEDS:-50}"
  --seasons ${HJEEDS_HOCKEY_SEASONS:-20232024 20242025 20252026}
)

case "$MODE" in
  preflight)
    python -m HJEEDS.run_hockey_subsample preflight "${COMMON_ARGS[@]}"

    N="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["tasks"]))' "$OUTPUT_DIR/manifest.json")"
    if [[ "$N" -le 0 ]]; then
      echo "Preflight produced no tasks; refusing to submit an empty worker array." >&2
      exit 1
    fi

    WORKER_JOB="$(sbatch --parsable \
      --array="1-${N}%${HJEEDS_HOCKEY_ARRAY_LIMIT:-50}" \
      --time="${HJEEDS_HOCKEY_WORKER_TIME:-2-00:00:00}" \
      --mem="${HJEEDS_HOCKEY_WORKER_MEM:-16G}" \
      --job-name="hjeeds-hockey-subsample-worker" \
      --output="${OUTPUT_DIR}/slurm-worker-%A_%a.out" \
      "$0" "$OUTPUT_DIR" worker)"
    echo "Submitted HJEEDS hockey subsample worker array: ${WORKER_JOB} (${N} tasks)"

    AGGREGATE_JOB="$(sbatch --parsable \
      --dependency="afterany:${WORKER_JOB}" \
      --time="${HJEEDS_HOCKEY_AGGREGATE_TIME:-04:00:00}" \
      --mem="${HJEEDS_HOCKEY_AGGREGATE_MEM:-16G}" \
      --job-name="hjeeds-hockey-subsample-aggregate" \
      --output="${OUTPUT_DIR}/slurm-aggregate-%j.out" \
      "$0" "$OUTPUT_DIR" aggregate)"
    echo "Submitted HJEEDS hockey subsample aggregation: ${AGGREGATE_JOB}"
    ;;
  worker)
    exec python -m HJEEDS.run_hockey_subsample worker "${COMMON_ARGS[@]}"
    ;;
  aggregate)
    exec python -m HJEEDS.run_hockey_subsample aggregate "${COMMON_ARGS[@]}"
    ;;
  *)
    echo "Unknown mode: $MODE (expected preflight, worker, or aggregate)" >&2
    exit 2
    ;;
esac
