#!/bin/bash

#SBATCH --job-name=bh-xg-compare
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --output=slurm-bh-xg-compare-%A_%a.out

set -euo pipefail

PLAYERS_FILE="${BH_PLAYERS_FILE:-${1:-Data/Hockey/forwards23-25.txt}}"
OUTPUT_ROOT="${BH_OUTPUT_ROOT:-${2:-Data/Hockey/xg_legacy_new_comparison}}"
LEGACY_ROOT="${BH_LEGACY_ROOT:-Data/Hockey}"
NEW_ROOT="${BH_NEW_ROOT:-Data/Hockey_xg_new}"
SEASONS="${BH_SEASONS:-20212022 20222023 20232024 20242025}"
SHOT_TYPES="${BH_SHOT_TYPES:-wristshot snapshot}"
MIN_SHOTS="${BH_MIN_SHOTS:-75}"
NUM_EXEC="${BH_NUM_EXEC:-250}"
NUM_PLAN="${BH_NUM_PLAN:-250}"
RNG_SEED="${BH_RNG_SEED:-0}"
MAX_CONCURRENT="${BH_MAX_CONCURRENT:-100}"
SUMMARY_TIME="${BH_SUMMARY_TIME:-01:00:00}"
SUMMARY_MEM="${BH_SUMMARY_MEM:-4G}"

cd ~/research/SkillEstimationFramework2

if [[ ! -f "$PLAYERS_FILE" ]]; then
  echo "Players file not found: $PLAYERS_FILE"
  exit 1
fi

if [[ -z "${BH_WORKER_MODE:-}" && -z "${BH_SUMMARY_MODE:-}" ]]; then
  read -r TOTAL_PLAYERS <<EOF
$(python3 - "$PLAYERS_FILE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
players = [
    line.strip()
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
]
player_ids = [int(p) for p in players]
print(len(player_ids))
PY
)
EOF

  if [[ "$TOTAL_PLAYERS" -le 0 ]]; then
    echo "No players found in $PLAYERS_FILE"
    exit 1
  fi

  TOTAL_JOBS=$((TOTAL_PLAYERS * 2))
  ARRAY_SPEC="1-${TOTAL_JOBS}"
  if [[ -n "$MAX_CONCURRENT" ]]; then
    ARRAY_SPEC="1-${TOTAL_JOBS}%${MAX_CONCURRENT}"
  fi

  echo "Submitting array: $ARRAY_SPEC (players=$TOTAL_PLAYERS, models=2)"
  ARRAY_JOB_ID=$(sbatch --parsable \
    --array="$ARRAY_SPEC" \
    --time="06:00:00" \
    --mem="16G" \
    --job-name="bh-xg-compare" \
    --output="slurm-bh-xg-compare-%A_%a.out" \
    --export=ALL,BH_WORKER_MODE=1,BH_PLAYERS_FILE="$PLAYERS_FILE",BH_OUTPUT_ROOT="$OUTPUT_ROOT",BH_LEGACY_ROOT="$LEGACY_ROOT",BH_NEW_ROOT="$NEW_ROOT",BH_SEASONS="$SEASONS",BH_SHOT_TYPES="$SHOT_TYPES",BH_MIN_SHOTS="$MIN_SHOTS",BH_NUM_EXEC="$NUM_EXEC",BH_NUM_PLAN="$NUM_PLAN",BH_RNG_SEED="$RNG_SEED" \
    "$0")

  echo "Submitting summary job after array ${ARRAY_JOB_ID}"
  sbatch \
    --dependency=afterok:${ARRAY_JOB_ID} \
    --time="$SUMMARY_TIME" \
    --mem="$SUMMARY_MEM" \
    --job-name="bh-xg-compare-summary" \
    --output="slurm-bh-xg-compare-summary-%j.out" \
    --export=ALL,BH_SUMMARY_MODE=1,BH_PLAYERS_FILE="$PLAYERS_FILE",BH_OUTPUT_ROOT="$OUTPUT_ROOT",BH_LEGACY_ROOT="$LEGACY_ROOT",BH_NEW_ROOT="$NEW_ROOT",BH_SEASONS="$SEASONS",BH_SHOT_TYPES="$SHOT_TYPES",BH_MIN_SHOTS="$MIN_SHOTS",BH_NUM_EXEC="$NUM_EXEC",BH_NUM_PLAN="$NUM_PLAN",BH_RNG_SEED="$RNG_SEED" \
    "$0"

  exit 0
fi

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate skill-estimation

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ -n "${BH_SUMMARY_MODE:-}" ]]; then
  python -m BlackhawksSkillEstimation.run_legacy_new_xg_comparison \
    --mode summary \
    --players-file "$PLAYERS_FILE" \
    --seasons $SEASONS \
    --legacy-root "$LEGACY_ROOT" \
    --new-root "$NEW_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --shot-types $SHOT_TYPES \
    --min-shots "$MIN_SHOTS" \
    --num-execution-skills "$NUM_EXEC" \
    --num-planning-skills "$NUM_PLAN" \
    --rng-seed "$RNG_SEED"
  exit 0
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Missing SLURM_ARRAY_TASK_ID for worker mode."
  exit 1
fi

read -r PLAYER_ID MODEL <<EOF
$(python3 - "$PLAYERS_FILE" "$SLURM_ARRAY_TASK_ID" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
array_id = int(sys.argv[2]) - 1
players = [
    line.strip()
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
]
player_ids = [int(p) for p in players]

if not player_ids:
    raise SystemExit("No player ids found.")

if array_id < 0 or array_id >= len(player_ids) * 2:
    raise SystemExit(f"Array index {array_id + 1} out of range for {len(player_ids)} players.")

player_idx = array_id // 2
model = "legacy" if array_id % 2 == 0 else "new"
print(player_ids[player_idx], model)
PY
)
EOF

python -m BlackhawksSkillEstimation.run_legacy_new_xg_comparison \
  --mode worker \
  --players-file "$PLAYERS_FILE" \
  --player-id "$PLAYER_ID" \
  --model "$MODEL" \
  --seasons $SEASONS \
  --legacy-root "$LEGACY_ROOT" \
  --new-root "$NEW_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --shot-types $SHOT_TYPES \
  --min-shots "$MIN_SHOTS" \
  --num-execution-skills "$NUM_EXEC" \
  --num-planning-skills "$NUM_PLAN" \
  --rng-seed "$RNG_SEED"
