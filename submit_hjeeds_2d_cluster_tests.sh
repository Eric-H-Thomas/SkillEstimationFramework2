#!/bin/bash
# Submit the 2D H-JEEDS array and its aggregation in one step.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

parts_per_group="10"
group_count="10"
seeds_per_group="500"
base_seed_start="1000"
dry_run="0"

usage() {
  cat <<'USAGE'
Usage: submit_hjeeds_2d_cluster_tests.sh [options]

Options:
  --parts-per-group N     Number of part_* dirs per group (default: 10).
  --group-count N         Number of group dirs (default: 10).
  --seeds-per-group N     Seeds per group (default: 500).
  --base-seed-start N     First base seed for group 0 (default: 1000).
  --dry-run               Print sbatch commands without submitting.
  -h, --help              Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --parts-per-group)
      parts_per_group="$2"
      shift 2
      ;;
    --group-count)
      group_count="$2"
      shift 2
      ;;
    --seeds-per-group)
      seeds_per_group="$2"
      shift 2
      ;;
    --base-seed-start)
      base_seed_start="$2"
      shift 2
      ;;
    --dry-run)
      dry_run="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: Unrecognized option '$1'." >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! [[ "${parts_per_group}" =~ ^[0-9]+$ ]] || (( parts_per_group < 1 )); then
  echo "Error: --parts-per-group must be a positive integer." >&2
  exit 1
fi
if ! [[ "${group_count}" =~ ^[0-9]+$ ]] || (( group_count < 1 )); then
  echo "Error: --group-count must be a positive integer." >&2
  exit 1
fi
if ! [[ "${seeds_per_group}" =~ ^[0-9]+$ ]] || (( seeds_per_group < 1 )); then
  echo "Error: --seeds-per-group must be a positive integer." >&2
  exit 1
fi
if ! [[ "${base_seed_start}" =~ ^[0-9]+$ ]] || (( base_seed_start < 0 )); then
  echo "Error: --base-seed-start must be a nonnegative integer." >&2
  exit 1
fi

array_size=$(( group_count * parts_per_group ))
array_spec="0-$((array_size - 1))"

export_env="ALL,PARTS_PER_GROUP=${parts_per_group},GROUP_COUNT=${group_count},SEEDS_PER_GROUP=${seeds_per_group},BASE_SEED_START=${base_seed_start}"

array_cmd=(
  sbatch
  --array="${array_spec}"
  --export="${export_env}"
  run_hjeeds_2d_cluster_tests.sbatch
)

agg_cmd=(
  sbatch
  --array=0
  --dependency=afterok:__ARRAY_JOB_ID__
  --export="${export_env},AGGREGATE_RESULTS=1"
  run_hjeeds_2d_cluster_tests.sbatch
)

if [[ "${dry_run}" == "1" ]]; then
  echo "Array submission: ${array_cmd[*]}"
  echo "Aggregation submission: ${agg_cmd[*]/__ARRAY_JOB_ID__/JOB_ID}"
  exit 0
fi

array_output="$(${array_cmd[@]})"
echo "${array_output}"
array_job_id="$(awk '{print $4}' <<< "${array_output}")"

agg_cmd[3]="--dependency=afterok:${array_job_id}"
agg_output="$(${agg_cmd[@]})"
echo "${agg_output}"
