#!/bin/bash

# Config workflow entrypoint on cluster:
#   sbatch run_blackhawks_config.sbatch Data/Hockey/jobs/<config>.json
#
# This script reads job array sizing + resource recommendations from the config,
# submits the worker array, and each worker runs run_blackhawks_config.py.

#SBATCH --job-name=bh-config-bootstrap
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --output=slurm-bootstrap-%j.out

set -euo pipefail

if [[ $# -lt 1 && -z "${BH_CONFIG_PATH:-}" ]]; then
  echo "Usage: sbatch run_blackhawks_config.sbatch <config.json>"
  exit 1
fi

CONFIG_PATH="${BH_CONFIG_PATH:-${1:-}}"
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
  read -r TOTAL_JOBS MAX_CONCURRENT SB_TIME SB_MEM <<EOF
$(python3 - "$CONFIG_PATH" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

cluster = cfg.get("cluster_plan", {})
reco = cluster.get("sbatch_recommendation", {})
jobs = cluster.get("jobs", [])

eligible = cluster.get("eligible_jobs")
if eligible is None:
    eligible = sum(1 for j in jobs if j.get("eligible", True)) if jobs else 0

total_jobs = int(eligible)
if total_jobs <= 0 and jobs:
    total_jobs = len(jobs)

max_concurrent = int(reco.get("max_concurrent", 100))
if max_concurrent <= 0:
    max_concurrent = 100

sb_time = str(reco.get("time", "24:00:00"))
sb_mem = str(reco.get("mem", "16G"))

print(total_jobs, max_concurrent, sb_time, sb_mem)
PY
)
EOF

  if [[ "$TOTAL_JOBS" -le 0 ]]; then
    echo "No jobs found in config: $CONFIG_PATH"
    exit 1
  fi

  if [[ "$MAX_CONCURRENT" -gt "$TOTAL_JOBS" ]]; then
    MAX_CONCURRENT="$TOTAL_JOBS"
  fi

  ARRAY_SPEC="1-${TOTAL_JOBS}%${MAX_CONCURRENT}"

  echo "Submitting array for config: $CONFIG_PATH"
  echo "  array: $ARRAY_SPEC"
  echo "  time:  $SB_TIME"
  echo "  mem:   $SB_MEM"

  sbatch \
    --array="$ARRAY_SPEC" \
    --time="$SB_TIME" \
    --mem="$SB_MEM" \
    --job-name="bh-config" \
    --output="slurm-bh-config-%A_%a.out" \
    --export=ALL,BH_WORKER_MODE=1,BH_CONFIG_PATH="$CONFIG_PATH" \
    "$0"

  exit 0
fi

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate skill-estimation

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python BlackhawksSkillEstimation/run_blackhawks_config.py \
  --config "$CONFIG_PATH" \
  --array-base 1
