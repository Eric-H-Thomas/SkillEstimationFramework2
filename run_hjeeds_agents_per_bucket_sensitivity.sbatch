#!/bin/bash
# This file was written or edited by AI and still requires human review. Delete this comment when done.
#SBATCH --job-name=hjeeds-agents-bucket
#SBATCH --qos=normal
#SBATCH --time=23:00:00
#SBATCH --mem=16G
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

# Make sure Python can import the repository packages from the Slurm working directory.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Render plots in headless mode and keep Matplotlib cache files off read-only homes.
export MPLBACKEND=Agg
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  export MPLCONFIGDIR="${MPLCONFIGDIR:-${SLURM_TMPDIR}/matplotlib}"
else
  export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}-${SLURM_JOB_ID:-manual}}"
fi
mkdir -p "${MPLCONFIGDIR}"

python_bin="${PYTHON_BIN:-$PWD/venv/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="${FALLBACK_PYTHON:-python3}"
fi

python_args=(
  -m HJEEDS.darts_agents_per_bucket_sensitivity
  --num-seeds "${NUM_SEEDS:-250}"
  --seed "${BASE_SEED:-12345}"
  --count-buckets "${COUNT_BUCKETS:-5,10,25,100,1000}"
  --agents-per-bucket-values "${AGENTS_PER_BUCKET_VALUES:-1,2,5,10,25}"
  --output-dir "${OUTPUT_DIR:-HJEEDS/results/hierarchical_darts_agents_per_bucket_sensitivity}"
  --delta "${DELTA:-0.1}"
  --num-sigma-grid "${NUM_SIGMA_GRID:-21}"
  --num-lambda-grid "${NUM_LAMBDA_GRID:-21}"
  --sigma-min "${SIGMA_MIN:-0.5}"
  --sigma-max "${SIGMA_MAX:-4.5}"
  --lambda-min "${LAMBDA_MIN:-1e-3}"
  --lambda-max "${LAMBDA_MAX:-1e2}"
  --min-success-regions "${MIN_SUCCESS_REGIONS:-2}"
  --max-success-regions "${MAX_SUCCESS_REGIONS:-6}"
  --min-region-width "${MIN_REGION_WIDTH:-0.25}"
  --weak-std-multiplier "${WEAK_STD_MULTIPLIER:-3.0}"
  --strong-std-multiplier "${STRONG_STD_MULTIPLIER:-0.3333333333333333}"
  --slight-bias-sd-units "${SLIGHT_BIAS_SD_UNITS:-0.5}"
  --significant-bias-sd-units "${SIGNIFICANT_BIAS_SD_UNITS:-1.5}"
)

if [[ -n "${CONDITION_SLUGS:-}" ]]; then
  python_args+=(--condition-slugs "${CONDITION_SLUGS}")
fi

if [[ "${AGGREGATE_RESULTS:-0}" == "1" ]]; then
  python_args+=(--aggregate-results)
else
  scenario_index="${SCENARIO_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
  python_args+=(--scenario-index "${scenario_index}")
fi

"${python_bin}" "${python_args[@]}"
