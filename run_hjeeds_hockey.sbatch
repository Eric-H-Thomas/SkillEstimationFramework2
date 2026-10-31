#!/bin/bash

# Submit a cached Blackhawks HJEEDS population run:
#   sbatch run_hjeeds_hockey.sbatch [config.json] [output_dir]
#
# Optional environment (set on the submitting shell; sbatch exports it):
#   HJEEDS_HOCKEY_SEASON              Single season (default 20252026)
#   HJEEDS_HOCKEY_SEASONS             Space-separated seasons to pool per player;
#                                     overrides HJEEDS_HOCKEY_SEASON
#   HJEEDS_HOCKEY_NUM_SIGMA_GRID      Sigma grid points (runner default when unset)
#   HJEEDS_HOCKEY_NUM_LOG_LAMBDA_GRID Log-lambda grid points (runner default when unset)
#   HJEEDS_HOCKEY_RUN_MAXG=1          Queue run_hjeeds_maxg.sbatch after a successful aggregate
#   HJEEDS_HOCKEY_MAXG_ESTIMATES      Space-separated sigmas to score, one MAXG job each
#                                     (default "hierarchical jeeds")
#   HJEEDS_HOCKEY_MAXG_SEASON_TAG     Tag in the MAXG results filename
# Workers and the aggregate read seasons, grids, and the data directory from
# the manifest that preflight writes.

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
SEASONS="${HJEEDS_HOCKEY_SEASONS:-}"
REPO_ROOT="${HJEEDS_REPO_ROOT:-$PWD}"

cd "$REPO_ROOT"
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate skill-estimation
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ -z "${HJEEDS_HOCKEY_MODE:-}" ]]; then
  PREFLIGHT_ARGS=(--config "$CONFIG_PATH" --output-dir "$OUTPUT_DIR")
  if [[ -n "$SEASONS" ]]; then
    read -r -a SEASON_LIST <<< "$SEASONS"
    PREFLIGHT_ARGS+=(--seasons "${SEASON_LIST[@]}")
  else
    PREFLIGHT_ARGS+=(--season "$SEASON")
  fi
  if [[ -n "${HJEEDS_HOCKEY_NUM_SIGMA_GRID:-}" ]]; then
    PREFLIGHT_ARGS+=(--num-sigma-grid "$HJEEDS_HOCKEY_NUM_SIGMA_GRID")
  fi
  if [[ -n "${HJEEDS_HOCKEY_NUM_LOG_LAMBDA_GRID:-}" ]]; then
    PREFLIGHT_ARGS+=(--num-log-lambda-grid "$HJEEDS_HOCKEY_NUM_LOG_LAMBDA_GRID")
  fi
  python -m HJEEDS.run_hockey_population preflight "${PREFLIGHT_ARGS[@]}"
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
    --export=ALL,HJEEDS_HOCKEY_MODE=worker,HJEEDS_HOCKEY_OUTPUT="$OUTPUT_DIR" \
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
    --export=ALL,HJEEDS_HOCKEY_MODE=aggregate,HJEEDS_HOCKEY_OUTPUT="$OUTPUT_DIR" \
    "$0")"
  echo "Submitted aggregation job: $AGGREGATE_JOB"

  if [[ "${HJEEDS_HOCKEY_RUN_MAXG:-0}" == "1" ]]; then
    if [[ -n "$SEASONS" ]]; then
      DEFAULT_MAXG_TAG="pooled_${SEASON_LIST[0]}_${SEASON_LIST[${#SEASON_LIST[@]}-1]}"
    else
      DEFAULT_MAXG_TAG="$SEASON"
    fi
    read -r -a MAXG_ESTIMATES <<< "${HJEEDS_HOCKEY_MAXG_ESTIMATES:-hierarchical jeeds}"
    for ESTIMATE in "${MAXG_ESTIMATES[@]}"; do
      MAXG_JOB="$(sbatch --parsable \
        --dependency="afterok:${AGGREGATE_JOB}" \
        --time="${HJEEDS_HOCKEY_MAXG_TIME:-24:00:00}" \
        --mem="${HJEEDS_HOCKEY_MAXG_MEM:-16G}" \
        --job-name="hjeeds-hockey-maxg-${ESTIMATE}" \
        --output="${OUTPUT_DIR}/slurm-maxg-${ESTIMATE}-%j.out" \
        --export=ALL,BH_REPO_ROOT="$REPO_ROOT",BH_HJEEDS_RESULTS="${OUTPUT_DIR}/agent_level_results.csv",BH_HJEEDS_ESTIMATE="$ESTIMATE",BH_SEASON="${HJEEDS_HOCKEY_MAXG_SEASON_TAG:-$DEFAULT_MAXG_TAG}",BH_OUTPUT_DIR="${OUTPUT_DIR}/maxg" \
        "${REPO_ROOT}/run_hjeeds_maxg.sbatch")"
      echo "Submitted MAXG job (${ESTIMATE}): $MAXG_JOB"
    done
  fi
  exit 0
fi

if [[ "$HJEEDS_HOCKEY_MODE" == "worker" ]]; then
  exec python -m HJEEDS.run_hockey_population worker --output-dir "$HJEEDS_HOCKEY_OUTPUT"
fi

if [[ "$HJEEDS_HOCKEY_MODE" == "aggregate" ]]; then
  exec python -m HJEEDS.run_hockey_population aggregate --output-dir "$HJEEDS_HOCKEY_OUTPUT"
fi

echo "Unknown HJEEDS_HOCKEY_MODE: $HJEEDS_HOCKEY_MODE"
exit 1
