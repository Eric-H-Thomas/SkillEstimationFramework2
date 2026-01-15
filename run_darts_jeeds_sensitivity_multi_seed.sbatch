#!/bin/bash
#SBATCH --job-name=darts-jeeds-multi-seed
#SBATCH --qos=normal
#SBATCH --time=23:00:00
#SBATCH --mem=8G
#SBATCH --output=slurm-%j.out

# e: exit immediately if command returns non-zero
# u: error if referencing unset variable
# o pipefail: fail if ANY stage of a pipeline fails, not just the last one
set -euo pipefail

# Make sure Python can import your packages
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Headless matplotlib (just render images to files, not to an interactive GUI)
export MPLBACKEND=Agg

# Take SEEDS if provided, otherwise use 7 11 13 17 23, split into an array called seeds, then store the number of seeds
# in num_seeds
read -r -a seeds <<< "${SEEDS:-7 11 13 17 23}"
num_seeds="${#seeds[@]}"

# Validate that we ended up with at least one seed; if not, print an error to stderr and exit.
if (( num_seeds == 0 )); then
  echo "Error: SEEDS must contain at least one seed value." >&2
  exit 1
fi

# Read JOBS_PER_SEED from the environment (default 1) to control how many sub-jobs to run per seed.
jobs_per_seed="${JOBS_PER_SEED:-1}"

# Validate JOBS_PER_SEED is a positive integer; otherwise, print an error to stderr and exit.
if ! [[ "${jobs_per_seed}" =~ ^[0-9]+$ ]] || (( jobs_per_seed < 1 )); then
  echo "Error: JOBS_PER_SEED must be a positive integer (received '${jobs_per_seed}')." >&2
  exit 1
fi

# Choose the base output directory (can be overridden via OUTPUT_DIR_BASE).
output_dir_base="${OUTPUT_DIR_BASE:-Testing/results/high_res}"

# Map the Slurm array task ID to (seed, job_index). If AGGREGATE_RESULTS=1, each task handles one seed and aggregates; otherwise, tasks cover (seed, per-seed job_index).
if [[ "${AGGREGATE_RESULTS:-0}" == "1" ]]; then
  # Aggregation mode: treat SLURM_ARRAY_TASK_ID as the seed index (default 0 when not running as an array job).
  seed_index="${SLURM_ARRAY_TASK_ID:-0}"
  # Bounds-check the seed index so we don't index past the end of the seeds array.
  if (( seed_index < 0 || seed_index >= num_seeds )); then
    echo "Error: SLURM_ARRAY_TASK_ID (${seed_index}) is out of range for ${num_seeds} seeds." >&2
    exit 1
  fi
  # Pick the seed for this task and set job_index=0 (only one job per seed when aggregating).
  seed="${seeds[seed_index]}"
  job_index=0
else
  # Non-aggregation mode: treat SLURM_ARRAY_TASK_ID as a flat task index over all (seed, job_index) combinations.
  task_id="${SLURM_ARRAY_TASK_ID:-0}"
  # Compute the total number of array tasks needed: one task per (seed, job_index) pair.
  total_tasks=$(( num_seeds * jobs_per_seed ))
  # Bounds-check the task ID so we don't compute invalid indices.
  if (( task_id < 0 || task_id >= total_tasks )); then
    echo "Error: SLURM_ARRAY_TASK_ID (${task_id}) is out of range for ${total_tasks} tasks." >&2
    exit 1
  fi
  # Decode the flat task index into seed_index (quotient) and job_index (remainder).
  seed_index=$(( task_id / jobs_per_seed ))
  job_index=$(( task_id % jobs_per_seed ))
  seed="${seeds[seed_index]}"
fi

# Build a per-seed output directory path so each seed's results are kept separate.
output_dir="${output_dir_base}_seed_${seed}"

# Assemble the Python module invocation and all experiment parameters as an args array (safe quoting, easy to append flags).
python_args=(
  -m Testing.darts_aiming_jeeds_sensitivity
  --seed "${seed}"
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
  --output-dir "${output_dir}"
  --num-jobs "${jobs_per_seed}"
  --job-index "${job_index}"
)

# If aggregating, add the --aggregate-results flag so the Python code combines per-job outputs for the seed.
if [[ "${AGGREGATE_RESULTS:-0}" == "1" ]]; then
  python_args+=(--aggregate-results)
fi

# Execute the experiment using the venv's Python interpreter, passing the args array exactly as separate CLI arguments.
"$PWD/venv/bin/python" "${python_args[@]}"
