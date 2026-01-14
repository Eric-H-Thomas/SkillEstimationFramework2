#!/bin/bash
#SBATCH --job-name=darts-jeeds-multi-seed
#SBATCH --qos=normal
#SBATCH --time=23:00:00
#SBATCH --mem=8G
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Make sure Python can import your packages
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# Headless matplotlib
export MPLBACKEND=Agg

read -r -a seeds <<< "${SEEDS:-7 11 13 17 23}"
num_seeds="${#seeds[@]}"

if (( num_seeds == 0 )); then
  echo "Error: SEEDS must contain at least one seed value." >&2
  exit 1
fi

jobs_per_seed="${JOBS_PER_SEED:-1}"
if ! [[ "${jobs_per_seed}" =~ ^[0-9]+$ ]] || (( jobs_per_seed < 1 )); then
  echo "Error: JOBS_PER_SEED must be a positive integer (received '${jobs_per_seed}')." >&2
  exit 1
fi

output_dir_base="${OUTPUT_DIR_BASE:-Testing/results/high_res}"

if [[ "${AGGREGATE_RESULTS:-0}" == "1" ]]; then
  seed_index="${SLURM_ARRAY_TASK_ID:-0}"
  if (( seed_index < 0 || seed_index >= num_seeds )); then
    echo "Error: SLURM_ARRAY_TASK_ID (${seed_index}) is out of range for ${num_seeds} seeds." >&2
    exit 1
  fi
  seed="${seeds[seed_index]}"
  job_index=0
else
  task_id="${SLURM_ARRAY_TASK_ID:-0}"
  total_tasks=$(( num_seeds * jobs_per_seed ))
  if (( task_id < 0 || task_id >= total_tasks )); then
    echo "Error: SLURM_ARRAY_TASK_ID (${task_id}) is out of range for ${total_tasks} tasks." >&2
    exit 1
  fi

  seed_index=$(( task_id / jobs_per_seed ))
  job_index=$(( task_id % jobs_per_seed ))
  seed="${seeds[seed_index]}"
fi

output_dir="${output_dir_base}_seed_${seed}"

python_args=(
  -m Testing.darts_aiming_jeeds_sensitivity
  --seed "${seed}"
  # --num-samples 100
  --num-samples 5
  # --num-aim-points 50
  --num-aim-points 7
  # --num-true-skills 40
  --num-true-skills 3
  # --min-true-skill 0.1
  --min-true-skill 0.1
  # --max-true-skill 3.0
  --max-true-skill 3.0
  # --num-grid-skills 200
  --num-grid-skills 20
  # --grid-min-skill 0.05
  --grid-min-skill 0.05
  # --grid-max-skill 3.5
  --grid-max-skill 3.5
  # --num-planning-skills 100
  --num-planning-skills 10
  # --delta 5e-3
  --delta 1e-2
  --output-dir "${output_dir}"
  --num-jobs "${jobs_per_seed}"
  --job-index "${job_index}"
)

if [[ "${AGGREGATE_RESULTS:-0}" == "1" ]]; then
  python_args+=(--aggregate-results)
fi

"$PWD/venv/bin/python" "${python_args[@]}"
