#!/bin/bash

# Submit a cached Blackhawks HJEEDS population run:
#   sbatch run_hjeeds_hockey.sbatch [config.json] [output_dir]

#SBATCH --job-name=hjeeds-hockey-bootstrap
# Preflight reads every selected player's parquet and NPZ metadata before it
# submits the worker array; this needs more than the tiny bootstrap allocation.
#SBATCH --time=00:60:00
#SBATCH --mem=8G
#SBATCH --output=slurm-hjeeds-hockey-bootstrap-%j.out

set -euo pipefail

CONFIG_PATH="${1:-Data/Hockey/jobs/jeeds_forwards_20252026.json}"
OUTPUT_DIR="${2:-Data/Hockey/experiments/hjeeds_20252026}"
SEASON="${HJEEDS_HOCKEY_SEASON:-20252026}"
REPO_ROOT="${HJEEDS_REPO_ROOT:-$PWD}"

cd "$REPO_ROOT"
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate skill-estimation
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ -z "${HJEEDS_HOCKEY_MODE:-}" ]]; then
  python -m HJEEDS.run_hockey_population preflight \
    --config "$CONFIG_PATH" --season "$SEASON" --output-dir "$OUTPUT_DIR"
  READY_COUNT="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["ready_player_ids"]))' "$OUTPUT_DIR/manifest.json")"
  if [[ "$READY_COUNT" -le 0 ]]; then
    echo "No ready hockey players in manifest."
    exit 1
  fi

  WORKER_JOB="$(sbatch --parsable \
    --array="1-${READY_COUNT}%50" \
    --time="${HJEEDS_HOCKEY_WORKER_TIME:-2-00:00:00}" \
    --mem="${HJEEDS_HOCKEY_WORKER_MEM:-16G}" \
    --job-name="hjeeds-hockey-worker" \
    --output="${OUTPUT_DIR}/slurm-worker-%A_%a.out" \
    --export=ALL,HJEEDS_HOCKEY_MODE=worker,HJEEDS_HOCKEY_CONFIG="$CONFIG_PATH",HJEEDS_HOCKEY_OUTPUT="$OUTPUT_DIR",HJEEDS_HOCKEY_SEASON="$SEASON" \
    "$0")"
  echo "Submitted worker array: $WORKER_JOB"

  # Aggregate after every worker has finished so missing/failed players are
  # recorded in failed_players.json instead of blocking the whole report.
  AGGREGATE_JOB="$(sbatch --parsable \
    --dependency="afterany:${WORKER_JOB}" \
    --time="${HJEEDS_HOCKEY_AGGREGATE_TIME:-04:00:00}" \
    --mem="${HJEEDS_HOCKEY_AGGREGATE_MEM:-16G}" \
    --job-name="hjeeds-hockey-aggregate" \
    --output="${OUTPUT_DIR}/slurm-aggregate-%j.out" \
    --export=ALL,HJEEDS_HOCKEY_MODE=aggregate,HJEEDS_HOCKEY_CONFIG="$CONFIG_PATH",HJEEDS_HOCKEY_OUTPUT="$OUTPUT_DIR",HJEEDS_HOCKEY_SEASON="$SEASON" \
    "$0")"
  echo "Submitted aggregation job: $AGGREGATE_JOB"
  exit 0
fi

if [[ "$HJEEDS_HOCKEY_MODE" == "worker" ]]; then
  exec python -m HJEEDS.run_hockey_population worker \
    --config "$HJEEDS_HOCKEY_CONFIG" --season "$HJEEDS_HOCKEY_SEASON" \
    --output-dir "$HJEEDS_HOCKEY_OUTPUT"
fi

if [[ "$HJEEDS_HOCKEY_MODE" == "aggregate" ]]; then
  exec python -m HJEEDS.run_hockey_population aggregate \
    --config "$HJEEDS_HOCKEY_CONFIG" --season "$HJEEDS_HOCKEY_SEASON" \
    --output-dir "$HJEEDS_HOCKEY_OUTPUT"
fi

echo "Unknown HJEEDS_HOCKEY_MODE: $HJEEDS_HOCKEY_MODE"
exit 1