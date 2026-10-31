#!/bin/bash
# Submit the pooled five-season HJEEDS hockey population fit on the 100 x 250
# grid. Each player's likelihood pools every season they have data for, one
# population prior is fit over the roster, and agent_level_results.csv keeps
# both the uniform-prior JEEDS and the hierarchical estimates. Once the
# aggregate succeeds, two MAXG jobs score the JEEDS and hierarchical sigmas.
#
#   bash submit_hjeeds_hockey_pooled.sh
#   bash submit_hjeeds_hockey_pooled.sh --dry-run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${REPO_ROOT}/run_hjeeds_hockey.sbatch"
CONFIG_PATH="${HJEEDS_HOCKEY_CONFIG:-Data/Hockey/jobs/jeeds_forwards_all_seasons.json}"
OUTPUT_DIR="${HJEEDS_HOCKEY_POOLED_OUTPUT:-Data/Hockey/experiments/hjeeds_pooled_20212022_20252026}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: submit_hjeeds_hockey_pooled.sh [--dry-run]

Submits one HJEEDS hockey population run that pools 2021-22 through 2025-26
per player on a 100 sigma x 250 log-lambda grid, then queues MAXG on both
the JEEDS and the hierarchical sigmas. Preflight refuses to start if any
shot-map file is not on the current xG grid.

Environment:
  HJEEDS_HOCKEY_CONFIG              Roster config (default: Data/Hockey/jobs/jeeds_forwards_all_seasons.json)
  HJEEDS_HOCKEY_POOLED_OUTPUT       Output directory (default: Data/Hockey/experiments/hjeeds_pooled_20212022_20252026)
  HJEEDS_HOCKEY_SEASONS             Seasons to pool (default: 20212022 ... 20252026)
  HJEEDS_HOCKEY_NUM_SIGMA_GRID      Sigma grid points (default: 100)
  HJEEDS_HOCKEY_NUM_LOG_LAMBDA_GRID Log-lambda grid points (default: 250)
  HJEEDS_HOCKEY_WORKER_TIME         Worker walltime (default: 24:00:00)
  HJEEDS_HOCKEY_RUN_MAXG            Set to 0 to skip the MAXG jobs (default: 1)
  HJEEDS_HOCKEY_MAXG_ESTIMATES      Sigmas to score with MAXG (default: "hierarchical jeeds")
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

export HJEEDS_HOCKEY_SEASONS="${HJEEDS_HOCKEY_SEASONS:-20212022 20222023 20232024 20242025 20252026}"
export HJEEDS_HOCKEY_NUM_SIGMA_GRID="${HJEEDS_HOCKEY_NUM_SIGMA_GRID:-100}"
export HJEEDS_HOCKEY_NUM_LOG_LAMBDA_GRID="${HJEEDS_HOCKEY_NUM_LOG_LAMBDA_GRID:-250}"
# 100x250 took about 3.6 s per shot on eight local cores, and the largest pooled
# player has about 1,900 shots (about 2 hours locally).
export HJEEDS_HOCKEY_WORKER_TIME="${HJEEDS_HOCKEY_WORKER_TIME:-24:00:00}"
export HJEEDS_HOCKEY_WORKER_MEM="${HJEEDS_HOCKEY_WORKER_MEM:-16G}"
export HJEEDS_HOCKEY_AGGREGATE_TIME="${HJEEDS_HOCKEY_AGGREGATE_TIME:-04:00:00}"
export HJEEDS_HOCKEY_AGGREGATE_MEM="${HJEEDS_HOCKEY_AGGREGATE_MEM:-16G}"
# MAXG took about 77 s per player on the 1,000-shot benchmark locally.
export HJEEDS_HOCKEY_RUN_MAXG="${HJEEDS_HOCKEY_RUN_MAXG:-1}"
export HJEEDS_HOCKEY_MAXG_ESTIMATES="${HJEEDS_HOCKEY_MAXG_ESTIMATES:-hierarchical jeeds}"
export HJEEDS_HOCKEY_MAXG_TIME="${HJEEDS_HOCKEY_MAXG_TIME:-24:00:00}"
export HJEEDS_HOCKEY_MAXG_MEM="${HJEEDS_HOCKEY_MAXG_MEM:-16G}"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "Missing sbatch script: $SBATCH_SCRIPT" >&2
  exit 1
fi

cd "$REPO_ROOT"
if [[ -f "${OUTPUT_DIR}/manifest.json" ]]; then
  echo "Note: ${OUTPUT_DIR} already has a manifest; workers will reuse existing likelihood files there."
fi

echo "seasons=${HJEEDS_HOCKEY_SEASONS} grid=${HJEEDS_HOCKEY_NUM_SIGMA_GRID}x${HJEEDS_HOCKEY_NUM_LOG_LAMBDA_GRID} worker_time=${HJEEDS_HOCKEY_WORKER_TIME} maxg=${HJEEDS_HOCKEY_RUN_MAXG} (${HJEEDS_HOCKEY_MAXG_ESTIMATES}) output=${OUTPUT_DIR}"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "sbatch ${SBATCH_SCRIPT} ${CONFIG_PATH} ${OUTPUT_DIR}"
  exit 0
fi
sbatch "$SBATCH_SCRIPT" "$CONFIG_PATH" "$OUTPUT_DIR"
