#!/bin/bash
# sbatch run_blackhawks_mcse_config.sbatch Data/Hockey/jobs/mcse_smoke.json
#
# Legacy xG only (data_root in config, typically Data/Hockey).
# For new xG: run_blackhawks_mcse_config_new_xg_root.sbatch <same-config.json>
# For both:   run_blackhawks_mcse_config_both.sbatch <legacy-config.json>

#SBATCH --job-name=bh-mcse-bootstrap
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --output=slurm-mcse-bootstrap-%j.out

set -euo pipefail

CONFIG_PATH="${BH_CONFIG_PATH:-${1:-}}"
if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: sbatch run_blackhawks_mcse_config.sbatch <config.json>"
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
import json, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
cluster = cfg.get("cluster_plan", {})
reco = cluster.get("sbatch_recommendation", {})
jobs = cluster.get("jobs", [])
eligible = cluster.get("eligible_jobs")
if eligible is not None:
    total_jobs = int(eligible)
else:
    total_jobs = sum(1 for j in jobs if j.get("eligible", True))
if total_jobs <= 0:
    raise SystemExit("No eligible jobs in config")
max_concurrent = int(reco.get("max_concurrent", 10)) or 10
if max_concurrent > total_jobs:
    max_concurrent = total_jobs
print(total_jobs, max_concurrent, reco.get("time", "48:00:00"), reco.get("mem", "32G"))
PY
)
EOF
  ARRAY_SPEC="1-${TOTAL_JOBS}%${MAX_CONCURRENT}"
  sbatch --array="$ARRAY_SPEC" --time="$SB_TIME" --mem="$SB_MEM" \
    --job-name="bh-mcse" --output="slurm-bh-mcse-%A_%a.out" \
    --export=ALL,BH_WORKER_MODE=1,BH_CONFIG_PATH="$CONFIG_PATH" "$0"
  exit 0
fi

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate skill-estimation
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python BlackhawksSkillEstimation/run_blackhawks_mcse_config.py \
  --config "$CONFIG_PATH" --array-base 1
