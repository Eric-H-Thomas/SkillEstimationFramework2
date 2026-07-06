#!/bin/bash
# Submit MCSE cluster arrays on both legacy and new-xG caches:
#   sbatch run_blackhawks_mcse_config_both.sbatch Data/Hockey/jobs/mcse_forwards_per_season.json
#
# This only submits two independent SLURM arrays (no combined worker logic):
#   1) legacy  -> run_blackhawks_mcse_config.sbatch
#   2) new xG  -> run_blackhawks_mcse_config_new_xg_root.sbatch

#SBATCH --job-name=bh-mcse-bootstrap-both
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --output=slurm-mcse-bootstrap-both-%j.out

set -euo pipefail

CONFIG_PATH="${BH_CONFIG_PATH:-${1:-Data/Hockey/jobs/mcse_forwards_per_season.json}}"
NEW_DATA_ROOT="${BH_NEW_DATA_ROOT:-Data/Hockey_xg_new}"

if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: sbatch run_blackhawks_mcse_config_both.sbatch <legacy-config.json>"
  exit 1
fi

cd ~/research/SkillEstimationFramework2

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH"
  exit 1
fi

if [[ -z "${BH_SUBMIT_MODE:-}" ]]; then
  echo "Submitting legacy MCSE array from: $CONFIG_PATH"
  LEGACY_JOB_ID=$(sbatch --parsable \
    --export=ALL,BH_CONFIG_PATH="$CONFIG_PATH" \
    run_blackhawks_mcse_config.sbatch "$CONFIG_PATH")

  echo "Submitting new-xG MCSE array (derived from same base config)"
  NEW_JOB_ID=$(sbatch --parsable \
    --export=ALL,BH_CONFIG_PATH="$CONFIG_PATH",BH_NEW_DATA_ROOT="$NEW_DATA_ROOT" \
    run_blackhawks_mcse_config_new_xg_root.sbatch "$CONFIG_PATH")

  echo "Legacy array job id:  $LEGACY_JOB_ID"
  echo "New-xG array job id:  $NEW_JOB_ID"
  exit 0
fi

echo "run_blackhawks_mcse_config_both.sbatch should only be used in bootstrap mode."
exit 1
