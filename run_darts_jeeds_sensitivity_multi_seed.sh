#!/bin/bash
#
# Submit multiple JEEDS sensitivity experiments across seeds to the Slurm
# scheduler, mirroring the configuration from
# ``run_darts_jeeds_sensitivity_exp.sbatch``. Work can optionally be sharded per
# seed using a Slurm job array. When sharded, an aggregation job array is
# submitted after the main jobs finish.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_darts_jeeds_sensitivity_multi_seed.sh [--jobs-per-seed N] [--output-dir-base PATH] [seed ...]

Submit the high-resolution JEEDS sensitivity experiment for one or more random
seeds to Slurm. By default the script processes a curated list of seeds using a
job array, but you can supply your own seeds as positional arguments.

Options:
  --jobs-per-seed N     Split each seed's workload across N Slurm array tasks.
                        Defaults to 1 (no sharding).
  --output-dir-base P   Base output directory (defaults to Testing/results/high_res).
                        Each seed writes to "<base>_seed_<seed>".
  -h, --help            Display this help message and exit.
USAGE
}

jobs_per_seed=1
output_dir_base="Testing/results/high_res"
seeds=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --jobs-per-seed)
      if [[ $# -lt 2 ]]; then
        echo "Error: --jobs-per-seed requires an integer argument." >&2
        usage >&2
        exit 1
      fi
      jobs_per_seed="$2"
      shift 2
      ;;
    --output-dir-base)
      if [[ $# -lt 2 ]]; then
        echo "Error: --output-dir-base requires a path argument." >&2
        usage >&2
        exit 1
      fi
      output_dir_base="$2"
      shift 2
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        seeds+=("$1")
        shift
      done
      break
      ;;
    -* )
      echo "Error: Unrecognized option '$1'." >&2
      usage >&2
      exit 1
      ;;
    *)
      seeds+=("$1")
      shift
      ;;
  esac
done

if ! [[ "$jobs_per_seed" =~ ^[0-9]+$ ]] || (( jobs_per_seed < 1 )); then
  echo "Error: --jobs-per-seed expects a positive integer (received '${jobs_per_seed}')." >&2
  exit 1
fi

if [[ ${#seeds[@]} -eq 0 ]]; then
  # Default seeds chosen to provide a modest coverage of the reward landscapes.
  seeds=(7 11 13 17 23)
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
array_script="${script_dir}/run_darts_jeeds_sensitivity_multi_seed.sbatch"

num_seeds="${#seeds[@]}"
total_tasks=$(( num_seeds * jobs_per_seed ))
seeds_export="${seeds[*]}"

submit_out=$(sbatch \
  --array=0-$((total_tasks - 1)) \
  --export=ALL,SEEDS="${seeds_export}",JOBS_PER_SEED="${jobs_per_seed}",OUTPUT_DIR_BASE="${output_dir_base}" \
  "${array_script}")

job_id=$(awk '{print $4}' <<< "${submit_out}")

echo "Submitted ${total_tasks} tasks for ${num_seeds} seeds: ${submit_out}"

if (( jobs_per_seed > 1 )); then
  aggregate_out=$(sbatch \
    --dependency=afterok:${job_id} \
    --array=0-$((num_seeds - 1)) \
    --export=ALL,SEEDS="${seeds_export}",JOBS_PER_SEED="${jobs_per_seed}",OUTPUT_DIR_BASE="${output_dir_base}",AGGREGATE_RESULTS=1 \
    "${array_script}")

  echo "Submitted aggregation array: ${aggregate_out}"
fi
