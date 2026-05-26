#!/bin/bash

#SBATCH --job-name=bh-xg-downsample
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --output=slurm_resolution_test/slurm-bh-xg-downsample-%A_%a.out

set -euo pipefail

CONFIG_PATH="${BH_CONFIG_PATH:-${1:-Data/Hockey/jobs/forwards21-25.json}}"
DATA_ROOT="${BH_DATA_ROOT:-Data/Hockey}"
OUTPUT_ROOT="${BH_OUTPUT_ROOT:-Data/Hockey_resolution_testing}"
TARGET_NZ="${BH_TARGET_NZ:-31}"
TARGET_NY="${BH_TARGET_NY:-51}"
MAX_CONCURRENT="${BH_MAX_CONCURRENT:-100}"
TIME_LIMIT="${BH_TIME:-24:00:00}"
MEM_LIMIT="${BH_MEM:-16G}"
MAXG_TIME="${BH_MAXG_TIME:-${TIME_LIMIT}}"
MAXG_MEM="${BH_MAXG_MEM:-16G}"
BENCHMARK_TAG="${BH_BENCHMARK_TAG:-wristshot-snapshot_v1}"
BENCHMARK_DIR="${BH_BENCHMARK_DIR:-Data/Hockey/benchmarks}"
SHOT_GROUP="${BH_SHOT_GROUP:-wristshot_snapshot}"

cd ~/research/SkillEstimationFramework2

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH"
  exit 1
fi

if [[ -z "${BH_WORKER_MODE:-}" && -z "${BH_MAXG_MODE:-}" ]]; then
  IFS='|' read -r TOTAL_JOBS SEASON_COUNT SEASONS_CSV PIDS_FILE <<EOF
$(python3 - "$CONFIG_PATH" "$OUTPUT_ROOT" <<'PY'
import json
import pathlib
import sys

cfg_path = pathlib.Path(sys.argv[1])
output_root = pathlib.Path(sys.argv[2])

with cfg_path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

cluster = cfg.get("cluster_plan", {})
jobs = cluster.get("jobs")

expanded = []
if isinstance(jobs, list) and jobs:
    expanded = [dict(job) for job in jobs]
else:
    filters = cfg.get("data_filters", {})
    player_ids = [int(x) for x in filters.get("player_ids", [])]
    seasons = [int(x) for x in filters.get("seasons", [])]
    shot_groups = [str(x) for x in filters.get("shot_groups", [])]
    split_mode = str(cluster.get("split_mode", "per_season"))
    if split_mode == "all_selected_seasons_together":
        for player_id in player_ids:
            for shot_group in shot_groups:
                expanded.append({
                    "player_id": player_id,
                    "season": -1,
                    "shot_group": shot_group,
                    "eligible": True,
                })
    else:
        for player_id in player_ids:
            for season in seasons:
                for shot_group in shot_groups:
                    expanded.append({
                        "player_id": player_id,
                        "season": season,
                        "shot_group": shot_group,
                        "eligible": True,
                    })

eligible = [j for j in expanded if j.get("eligible", True)]
seasons = sorted({int(j.get("season")) for j in eligible if int(j.get("season", -1)) >= 0})
if not seasons:
    seasons = sorted({int(s) for s in cfg.get("data_filters", {}).get("seasons", [])})

output_root.mkdir(parents=True, exist_ok=True)
reports_dir = output_root / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

player_ids = sorted({int(j.get("player_id")) for j in eligible if j.get("player_id") is not None})
players_file = reports_dir / "players_from_config.txt"
players_file.write_text("\n".join(str(pid) for pid in player_ids) + "\n")

print(len(eligible), len(seasons), ",".join(str(s) for s in seasons), players_file.as_posix(), sep="|")
PY
)
EOF

  if [[ "$TOTAL_JOBS" -le 0 ]]; then
    echo "No eligible jobs found in $CONFIG_PATH"
    exit 1
  fi

  if [[ -z "$SEASONS_CSV" ]]; then
    echo "No seasons found in $CONFIG_PATH"
    exit 1
  fi

  if [[ "$MAX_CONCURRENT" -le 0 ]]; then
    MAX_CONCURRENT=1
  fi

  ARRAY_SPEC="1-${TOTAL_JOBS}%${MAX_CONCURRENT}"
  echo "Submitting JEEDS array: $ARRAY_SPEC"

  JEEDS_JOB_ID=$(sbatch --parsable \
    --array="$ARRAY_SPEC" \
    --time="$TIME_LIMIT" \
    --mem="$MEM_LIMIT" \
    --job-name="bh-xg-downsample" \
    --output="slurm_resolution_test/slurm-bh-xg-downsample-%A_%a.out" \
    --export=ALL,BH_WORKER_MODE=1,BH_CONFIG_PATH="$CONFIG_PATH",BH_DATA_ROOT="$DATA_ROOT",BH_OUTPUT_ROOT="$OUTPUT_ROOT",BH_TARGET_NZ="$TARGET_NZ",BH_TARGET_NY="$TARGET_NY" \
    "$0")

  SEASONS="$SEASONS_CSV"
  MAXG_ARRAY_SPEC="0-$((SEASON_COUNT - 1))"
  echo "Submitting MAXG array: $MAXG_ARRAY_SPEC (afterok:$JEEDS_JOB_ID)"

  sbatch \
    --dependency=afterok:${JEEDS_JOB_ID} \
    --array="$MAXG_ARRAY_SPEC" \
    --time="$MAXG_TIME" \
    --mem="$MAXG_MEM" \
    --job-name="bh-maxg-downsample" \
    --output="slurm_resolution_test/slurm-bh-maxg-downsample-%A_%a.out" \
    --export=ALL,BH_MAXG_MODE=1,BH_SEASONS="$SEASONS",BH_BENCHMARK_TAG="$BENCHMARK_TAG",BH_BENCHMARK_DIR="$BENCHMARK_DIR",BH_OUTPUT_ROOT="$OUTPUT_ROOT",BH_PIDS_FILE="$PIDS_FILE",BH_SHOT_GROUP="$SHOT_GROUP" \
    "$0"

  exit 0
fi

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate skill-estimation

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MPLBACKEND=Agg

if [[ -n "${BH_MAXG_MODE:-}" ]]; then
  if [[ -z "${BH_SEASONS:-}" ]]; then
    echo "Missing BH_SEASONS for MAXG mode."
    exit 1
  fi

  IFS=',' read -r -a SEASON_LIST <<< "$BH_SEASONS"
  TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

  if (( TASK_ID < 0 || TASK_ID >= ${#SEASON_LIST[@]} )); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID} for ${#SEASON_LIST[@]} seasons" >&2
    exit 1
  fi

  SEASON="${SEASON_LIST[$TASK_ID]}"

  python -m BlackhawksSkillEstimation.maxg_evaluator \
    --benchmark-tag "$BH_BENCHMARK_TAG" \
    --benchmark-dir "$BH_BENCHMARK_DIR" \
    --season-tag "$SEASON" \
    --shot-group "$BH_SHOT_GROUP" \
    --data-dir "$BH_OUTPUT_ROOT" \
    --output-dir "$BH_OUTPUT_ROOT/benchmarks/results" \
    --pids-file "$BH_PIDS_FILE"

  exit 0
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Missing SLURM_ARRAY_TASK_ID for worker mode."
  exit 1
fi

python -m BlackhawksSkillEstimation.run_resolution_downsample_legacy \
  --config "$CONFIG_PATH" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --target-nz "$TARGET_NZ" \
  --target-ny "$TARGET_NY" \
  --array-base 1
