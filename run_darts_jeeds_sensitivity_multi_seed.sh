#!/bin/bash
#
# Iterate over multiple random seeds and rerun the JEEDS sensitivity experiment
# for each value, mirroring the configuration from
# ``run_darts_jeeds_sensitivity_exp.sbatch``. Results for each seed are written
# to a dedicated output directory. When the workload is partitioned across
# multiple jobs per seed the partial CSV shards are automatically aggregated
# once all shards finish.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_darts_jeeds_sensitivity_multi_seed.sh [--jobs-per-seed N] [seed ...]

Run the high-resolution JEEDS sensitivity experiment for one or more random
seeds. By default the script processes a curated list of seeds sequentially,
but you can supply your own seeds as positional arguments. When more than one
job per seed is requested the script runs each shard back-to-back and triggers
the aggregation pass automatically once all shards complete.

Options:
  --jobs-per-seed N   Split each seed's workload across N sequential jobs.
                      Defaults to 1 (no sharding).
  -h, --help          Display this help message and exit.
EOF
}

jobs_per_seed=1
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
    --)
      shift
      while [[ $# -gt 0 ]]; do
        seeds+=("$1")
        shift
      done
      break
      ;;
    -*)
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

for seed in "${seeds[@]}"; do
  output_dir="Testing/results/high_res_seed_${seed}"
  partial_dir="${output_dir}/partials"

  echo "==== Running darts JEEDS sensitivity experiment with seed ${seed} ===="
  echo "Output directory: ${output_dir}"

  if (( jobs_per_seed > 1 )) && [[ -d "${partial_dir}" ]]; then
    echo "Cleaning partial results directory: ${partial_dir}"
    rm -rf "${partial_dir}"
  fi

  for (( job_index = 0; job_index < jobs_per_seed; ++job_index )); do
    echo "---- Launching job $((job_index + 1))/${jobs_per_seed} for seed ${seed}"
    SEED="${seed}" \
    OUTPUT_DIR="${output_dir}" \
    NUM_JOBS="${jobs_per_seed}" \
    JOB_INDEX="${job_index}" \
      bash "${script_dir}/run_darts_jeeds_sensitivity_exp.sbatch"
  done

  if (( jobs_per_seed > 1 )); then
    echo "---- Aggregating partial results for seed ${seed}"
    SEED="${seed}" \
    OUTPUT_DIR="${output_dir}" \
    NUM_JOBS="${jobs_per_seed}" \
    JOB_INDEX=0 \
    AGGREGATE_RESULTS=1 \
      bash "${script_dir}/run_darts_jeeds_sensitivity_exp.sbatch"
  else
    echo "---- Seed ${seed} completed (single job produced final artifacts)"
  fi
done
