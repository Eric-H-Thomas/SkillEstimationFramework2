#!/bin/bash
# New-xG-root MCSE workflow (mirrors run_blackhawks_config_new_xg_root.sbatch):
#   sbatch run_blackhawks_mcse_config_new_xg_root.sbatch Data/Hockey/jobs/mcse_forwards_per_season.json
#
# Bootstrap: derives <config>.xgnew.json with data_root=Data/Hockey_xg_new and
# refreshed job eligibility, then submits the worker array.

#SBATCH --job-name=bh-mcse-bootstrap-newxg
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --output=slurm-mcse-bootstrap-newxg-%j.out

set -euo pipefail

CONFIG_PATH="${BH_CONFIG_PATH:-${1:-}}"
NEW_DATA_ROOT="${BH_NEW_DATA_ROOT:-Data/Hockey_xg_new}"

if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: sbatch run_blackhawks_mcse_config_new_xg_root.sbatch <legacy-config.json>"
  exit 1
fi

cd ~/research/SkillEstimationFramework2

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH"
  exit 1
fi

if [[ -z "${BH_WORKER_MODE:-}" ]]; then
  module load miniforge3
  eval "$(conda shell.bash hook)"
  conda activate skill-estimation
  export PYTHONPATH="$PWD:${PYTHONPATH:-}"

  read -r DERIVED_CONFIG_PATH TOTAL_JOBS MAX_CONCURRENT SB_TIME SB_MEM <<EOF
$(python - "$CONFIG_PATH" "$NEW_DATA_ROOT" <<'PY'
import json
import pathlib
import sys

from BlackhawksSkillEstimation.build_mcse_cluster_config import (
    derive_mcse_config_for_data_root,
    derived_config_path,
)

cfg_path = pathlib.Path(sys.argv[1])
new_data_root = sys.argv[2]

with cfg_path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

derived = derive_mcse_config_for_data_root(cfg, new_data_root)
derived_path = derived_config_path(cfg_path)

with derived_path.open("w", encoding="utf-8") as f:
    json.dump(derived, f, indent=2)
    f.write("\n")

cluster = derived.get("cluster_plan", {})
reco = cluster.get("sbatch_recommendation", {})
jobs = cluster.get("jobs", [])
eligible = cluster.get("eligible_jobs")
if eligible is not None:
    total_jobs = int(eligible)
else:
    total_jobs = sum(1 for j in jobs if j.get("eligible", True))
if total_jobs <= 0:
    raise SystemExit("No eligible jobs in derived config")
max_concurrent = int(reco.get("max_concurrent", 10)) or 10
if max_concurrent > total_jobs:
    max_concurrent = total_jobs
print(
    derived_path.as_posix(),
    total_jobs,
    max_concurrent,
    reco.get("time", "48:00:00"),
    reco.get("mem", "32G"),
)
PY
)
EOF

  ARRAY_SPEC="1-${TOTAL_JOBS}%${MAX_CONCURRENT}"
  echo "Original config: $CONFIG_PATH"
  echo "Derived config:  $DERIVED_CONFIG_PATH"
  echo "data_root:       $NEW_DATA_ROOT"
  echo "Array:           $ARRAY_SPEC"

  sbatch --array="$ARRAY_SPEC" --time="$SB_TIME" --mem="$SB_MEM" \
    --job-name="bh-mcse-newxg" --output="slurm-bh-mcse-newxg-%A_%a.out" \
    --export=ALL,BH_WORKER_MODE=1,BH_CONFIG_PATH="$DERIVED_CONFIG_PATH",BH_NEW_DATA_ROOT="$NEW_DATA_ROOT" \
    "$0"
  exit 0
fi

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate skill-estimation
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python BlackhawksSkillEstimation/run_blackhawks_mcse_config.py \
  --config "$CONFIG_PATH" --array-base 1
