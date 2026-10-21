#!/bin/bash
# Submit the 3x3 HJEEDS hockey subsample grid-resolution sweep.
#
# Every cell uses the five-season pool, the same N values, and the same draw
# seed, so the event ids match across resolutions. Each cell is an independent
# preflight that submits its own worker arrays and aggregate job.
#
#   bash submit_hjeeds_hockey_grid_resolution.sh
#   bash submit_hjeeds_hockey_grid_resolution.sh --dry-run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${HJEEDS_HOCKEY_GRID_ROOT:-Data/Hockey/experiments/hjeeds_grid_resolution}"
SBATCH_SCRIPT="${REPO_ROOT}/run_hjeeds_hockey_subsample.sbatch"
SIGMA_COUNTS=(50 100 250)
LAMBDA_COUNTS=(50 100 250)
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: submit_hjeeds_hockey_grid_resolution.sh [--dry-run]

Submits nine HJEEDS hockey subsample preflight jobs, one for each pair of
sigma and log-lambda grid counts in {50, 100, 250}. Seasons, shot group,
players, N, and base seed come from run_hjeeds_hockey_subsample.sbatch.

Environment:
  HJEEDS_HOCKEY_GRID_ROOT     Output root (default: Data/Hockey/experiments/hjeeds_grid_resolution)
  HJEEDS_HOCKEY_SEASONS       Season list passed through to each job
  HJEEDS_HOCKEY_PLAYERS_FILE  Roster file passed through to each job
  HJEEDS_HOCKEY_WORKER_TIME   Worker walltime for every cell. Unset uses
                              02:00:00, 04:00:00, or 08:00:00 from the grid size.
  HJEEDS_HOCKEY_AGGREGATE_TIME Aggregate walltime (default: 04:00:00)
  HJEEDS_HOCKEY_WORKER_MEM    Worker memory (default: 16G)
  HJEEDS_HOCKEY_AGGREGATE_MEM Aggregate memory (default: 16G)
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

# N=400 at 250x250 took about 65 minutes on eight cores. These requests cover a
# slower single-core node without asking the scheduler for a full day.
WORKER_TIME_OVERRIDE="${HJEEDS_HOCKEY_WORKER_TIME:-}"
export HJEEDS_HOCKEY_AGGREGATE_TIME="${HJEEDS_HOCKEY_AGGREGATE_TIME:-04:00:00}"
export HJEEDS_HOCKEY_WORKER_MEM="${HJEEDS_HOCKEY_WORKER_MEM:-16G}"
export HJEEDS_HOCKEY_AGGREGATE_MEM="${HJEEDS_HOCKEY_AGGREGATE_MEM:-16G}"

worker_time_for_cell() {
  local num_sigma="$1"
  local num_lambda="$2"
  if [[ -n "$WORKER_TIME_OVERRIDE" ]]; then
    printf '%s\n' "$WORKER_TIME_OVERRIDE"
  elif [[ "$num_sigma" -eq 250 && "$num_lambda" -eq 250 ]]; then
    printf '08:00:00\n'
  elif [[ "$num_sigma" -eq 250 || "$num_lambda" -eq 250 ]]; then
    printf '04:00:00\n'
  else
    printf '02:00:00\n'
  fi
}

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "Missing sbatch script: $SBATCH_SCRIPT" >&2
  exit 1
fi

for num_sigma in "${SIGMA_COUNTS[@]}"; do
  for num_lambda in "${LAMBDA_COUNTS[@]}"; do
    output_dir="${OUTPUT_ROOT}/sigma${num_sigma}_lambda${num_lambda}"
    worker_time="$(worker_time_for_cell "$num_sigma" "$num_lambda")"
    echo "cell sigma=${num_sigma} lambda=${num_lambda} worker_time=${worker_time} output=${output_dir}"
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "HJEEDS_HOCKEY_WORKER_TIME=${worker_time} HJEEDS_HOCKEY_AGGREGATE_TIME=${HJEEDS_HOCKEY_AGGREGATE_TIME} HJEEDS_HOCKEY_NUM_SIGMA_GRID=${num_sigma} HJEEDS_HOCKEY_NUM_LOG_LAMBDA_GRID=${num_lambda} sbatch ${SBATCH_SCRIPT} ${output_dir} preflight"
      continue
    fi
    HJEEDS_HOCKEY_WORKER_TIME="${worker_time}" \
      HJEEDS_HOCKEY_NUM_SIGMA_GRID="${num_sigma}" \
      HJEEDS_HOCKEY_NUM_LOG_LAMBDA_GRID="${num_lambda}" \
      sbatch "$SBATCH_SCRIPT" "$output_dir" preflight
  done
done
