#!/bin/bash
#SBATCH --job-name=darts-jeeds-sensitivity
#SBATCH --qos=normal
#SBATCH --time=23:00:00
#SBATCH --mem=8G
#SBATCH --chdir=/home/eht23/research/SkillEstimationFramework2
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Make sure Python can import your packages
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# Headless matplotlib
export MPLBACKEND=Agg

# Determine how this invocation should partition work. ``SLURM_ARRAY_TASK_ID``
# allows the script to be used as part of a Slurm job array, while manual
# invocations can override the values via ``JOB_INDEX``/``NUM_JOBS``.
if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
  if [[ -n "${NUM_JOBS+x}" && "${NUM_JOBS}" != "${SLURM_ARRAY_TASK_COUNT}" ]]; then
    echo "Warning: NUM_JOBS (${NUM_JOBS}) does not match SLURM_ARRAY_TASK_COUNT (${SLURM_ARRAY_TASK_COUNT}). Using the array size instead." >&2
  fi
  NUM_JOBS="${SLURM_ARRAY_TASK_COUNT}"
else
  NUM_JOBS="${NUM_JOBS:-1}"
fi
JOB_INDEX="${JOB_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"

python_args=(
  -m Testing.darts_aiming_jeeds_sensitivity
  --seed 7
  --num-samples 100
  --num-aim-points 50
  --num-true-skills 40
  --min-true-skill 0.1
  --max-true-skill 3.0
  --num-grid-skills 200
  --grid-min-skill 0.05
  --grid-max-skill 3.5
  --num-planning-skills 100
  --delta 5e-3
  --output-dir Testing/results/high_res
  --num-jobs "${NUM_JOBS}"
  --job-index "${JOB_INDEX}"
)

if [[ "${AGGREGATE_RESULTS:-0}" == "1" ]]; then
  python_args+=(--aggregate-results)
fi

# Run as a module so imports behave
"$PWD/venv/bin/python" "${python_args[@]}"
