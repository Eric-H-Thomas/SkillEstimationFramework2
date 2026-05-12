#!/bin/bash

# New-XG-root config workflow entrypoint on cluster (Phase 4B temporary wrapper):
#   sbatch run_blackhawks_config_new_xg_root.sbatch [Data/Hockey/jobs/<config>.json]
#
# Behavior in bootstrap mode:
# 1) Loads the input config JSON.
# 2) Writes a derived config next to it with top-level data_root overridden.
# 3) Sizes array from len(cluster_plan.jobs) (not eligible_jobs).
# 4) Submits workers against the derived config.
#
# Worker mode then calls run_blackhawks_config.py with --include-ineligible.

#SBATCH --job-name=bh-config-bootstrap-newxg
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --output=slurm-bootstrap-newxg-%j.out

set -euo pipefail

DEFAULT_CONFIG_PATH="Data/Hockey/jobs/migration_forwards.json"
CONFIG_PATH="${BH_CONFIG_PATH:-${1:-$DEFAULT_CONFIG_PATH}}"
NEW_DATA_ROOT="${BH_NEW_DATA_ROOT:-Data/Hockey_xg_new}"

if [[ -z "$CONFIG_PATH" ]]; then
  echo "Config path is empty."
  exit 1
fi

cd ~/research/SkillEstimationFramework2

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH"
  exit 1
fi

if [[ -z "${BH_WORKER_MODE:-}" ]]; then
  read -r DERIVED_CONFIG_PATH TOTAL_JOBS MAX_CONCURRENT SB_TIME SB_MEM <<EOF
$(python3 - "$CONFIG_PATH" "$NEW_DATA_ROOT" <<'PY'
import json
import pathlib
import sys

cfg_path = pathlib.Path(sys.argv[1])
new_data_root = sys.argv[2]

with cfg_path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

cluster = cfg.get("cluster_plan")
if not isinstance(cluster, dict):
    raise SystemExit("Missing cluster_plan in config")

jobs = cluster.get("jobs")
if not isinstance(jobs, list):
    raise SystemExit("Missing cluster_plan.jobs list in config")

total_jobs = len(jobs)
if total_jobs <= 0:
    raise SystemExit("cluster_plan.jobs is empty in config")

reco = cluster.get("sbatch_recommendation", {})
if not isinstance(reco, dict):
    reco = {}

max_concurrent = int(reco.get("max_concurrent", 100))
if max_concurrent <= 0:
    max_concurrent = 100

sb_time = str(reco.get("time", "24:00:00"))
sb_mem = str(reco.get("mem", "16G"))

cfg["data_root"] = new_data_root

derived_name = f"{cfg_path.stem}.newxgroot{cfg_path.suffix}"
derived_path = cfg_path.with_name(derived_name)
with derived_path.open("w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")

print(derived_path.as_posix(), total_jobs, max_concurrent, sb_time, sb_mem)
PY
)
EOF

  if [[ "$MAX_CONCURRENT" -gt "$TOTAL_JOBS" ]]; then
    MAX_CONCURRENT="$TOTAL_JOBS"
  fi

  ARRAY_SPEC="1-${TOTAL_JOBS}%${MAX_CONCURRENT}"

  echo "Submitting array for original config: $CONFIG_PATH"
  echo "  derived config path: $DERIVED_CONFIG_PATH"
  echo "  overridden data_root: $NEW_DATA_ROOT"
  echo "  array (total jobs): $ARRAY_SPEC"
  echo "  time: $SB_TIME"
  echo "  mem:  $SB_MEM"

  sbatch \
    --array="$ARRAY_SPEC" \
    --time="$SB_TIME" \
    --mem="$SB_MEM" \
    --job-name="bh-config-newxg" \
    --output="slurm-bh-config-newxg-%A_%a.out" \
    --export=ALL,BH_WORKER_MODE=1,BH_CONFIG_PATH="$DERIVED_CONFIG_PATH",BH_NEW_DATA_ROOT="$NEW_DATA_ROOT" \
    "$0"

  exit 0
fi

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate skill-estimation

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python BlackhawksSkillEstimation/run_blackhawks_config.py \
  --config "$CONFIG_PATH" \
  --array-base 1 \
  --include-ineligible
