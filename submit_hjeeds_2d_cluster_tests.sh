#!/bin/bash
# Paper correspondence: Main `subsec:two_d_darts`.
# Submit the 2D H-JEEDS array and its aggregation in one step.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir}"
worker_script="${script_dir}/run_hjeeds_2d_cluster_tests.sbatch"
cd "${repo_root}"

parts_per_group="10"
group_count="1"
seeds_per_group="500"
base_seed_start="1000"
expected_agents_per_seed="25"
python_bin=""
conda_env="skill-estimation"
agg_time="02:00:00"
agg_mem="4G"
dry_run="0"
agg_only="0"
require_paper_config="0"

usage() {
  cat <<'USAGE'
Usage: submit_hjeeds_2d_cluster_tests.sh [options]

Options:
  --parts-per-group N     Number of part_* dirs per group (default: 10).
  --group-count N         Number of group dirs (default: 1; canonical paper run).
  --seeds-per-group N     Seeds per group (default: 500).
  --base-seed-start N     First base seed for group 0 (default: 1000).
  --expected-agents N     Exact agent count required per seed (default: 25).
  --require-paper-config  Enable publication-only 2D checks (25 agents/seed, paper defaults).
  --python-bin PATH       Explicit Python executable; bypasses Conda activation.
  --conda-env NAME        Conda environment when --python-bin is unset (default: skill-estimation).
  --agg-time HH:MM:SS     Walltime for aggregation job (default: 02:00:00).
  --agg-mem MEM           Memory for aggregation job (default: 4G).
  --agg-only              Submit only the aggregation job.
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
    --expected-agents)
      expected_agents_per_seed="$2"
      shift 2
      ;;
    --python-bin)
      python_bin="$2"
      shift 2
      ;;
    --conda-env)
      conda_env="$2"
      shift 2
      ;;
    --agg-time)
      agg_time="$2"
      shift 2
      ;;
    --agg-mem)
      agg_mem="$2"
      shift 2
      ;;
    --dry-run)
      dry_run="1"
      shift
      ;;
    --agg-only)
      agg_only="1"
      shift
      ;;
    --require-paper-config)
      require_paper_config="1"
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
if [[ "${require_paper_config}" == "1" && "${expected_agents_per_seed}" != "25" ]]; then
  echo "Error: --expected-agents must be 25 when --require-paper-config is set." >&2
  exit 1
fi
if (( seeds_per_group % parts_per_group != 0 )); then
  echo "Error: --seeds-per-group must be divisible by --parts-per-group." >&2
  exit 1
fi
if [[ ! -f "${worker_script}" ]]; then
  echo "Error: 2D Slurm worker not found: ${worker_script}" >&2
  exit 1
fi
if [[ -n "${python_bin}" && "${python_bin}" == *,* ]]; then
  echo "Error: --python-bin cannot contain a comma because Slurm uses comma-separated exports." >&2
  exit 1
fi
if [[ -z "${python_bin}" && -z "${conda_env}" ]]; then
  echo "Error: --conda-env must be non-empty when --python-bin is unset." >&2
  exit 1
fi
if [[ "${conda_env}" == *,* ]]; then
  echo "Error: --conda-env cannot contain a comma because Slurm uses comma-separated exports." >&2
  exit 1
fi
if [[ -z "${agg_time}" ]]; then
  echo "Error: --agg-time must be non-empty." >&2
  exit 1
fi
if [[ -z "${agg_mem}" ]]; then
  echo "Error: --agg-mem must be non-empty." >&2
  exit 1
fi

array_size=$(( group_count * parts_per_group ))
array_spec="0-$((array_size - 1))"

export_env="ALL,HJEEDS_REPO_ROOT=${repo_root},PARTS_PER_GROUP=${parts_per_group},GROUP_COUNT=${group_count},SEEDS_PER_GROUP=${seeds_per_group},BASE_SEED_START=${base_seed_start},EXPECTED_AGENTS_PER_SEED=${expected_agents_per_seed},CONDA_ENV=${conda_env}"
if [[ "${require_paper_config}" == "1" ]]; then
  export_env="${export_env},HJEEDS_REQUIRE_PAPER_CONFIG=1"
fi
if [[ -n "${python_bin}" ]]; then
  export_env="${export_env},PYTHON_BIN=${python_bin}"
fi

array_cmd=(
  sbatch
  --array="${array_spec}"
  --export="${export_env}"
  "${worker_script}"
)

agg_cmd=(
  sbatch
  --array=0
  --dependency=afterok:__ARRAY_JOB_ID__
  --time="${agg_time}"
  --mem="${agg_mem}"
  --export="${export_env},AGGREGATE_RESULTS=1"
  "${worker_script}"
)

if [[ "${dry_run}" == "1" ]]; then
  if [[ "${agg_only}" == "1" ]]; then
    echo "Aggregation submission: ${agg_cmd[*]/__ARRAY_JOB_ID__/JOB_ID}"
    exit 0
  fi
  echo "Array submission: ${array_cmd[*]}"
  echo "Aggregation submission: ${agg_cmd[*]/__ARRAY_JOB_ID__/JOB_ID}"
  exit 0
fi

if [[ "${agg_only}" == "1" ]]; then
  agg_only_cmd=(
    sbatch
    --array=0
    --time="${agg_time}"
    --mem="${agg_mem}"
    --export="${export_env},AGGREGATE_RESULTS=1"
    "${worker_script}"
  )
  agg_output="$("${agg_only_cmd[@]}")"
  echo "${agg_output}"
  exit 0
fi

# Mark every existing cluster aggregate incomplete before any replacement
# worker is submitted. If submission or a worker later fails, the main-paper
# plotter cannot silently reuse the previous aggregate. Aggregation repeats the
# invalidation immediately before it starts publishing combined artifacts.
metadata_python=()
if [[ -n "${python_bin}" ]]; then
  metadata_python=("${python_bin}")
elif command -v conda >/dev/null 2>&1; then
  metadata_python=(conda run --no-capture-output -n "${conda_env}" python)
elif command -v python >/dev/null 2>&1; then
  metadata_python=("$(command -v python)")
elif command -v python3 >/dev/null 2>&1; then
  metadata_python=("$(command -v python3)")
else
  echo "Error: cannot invalidate prior 2D results; activate ${conda_env} or pass --python-bin." >&2
  exit 1
fi

for (( group_index=0; group_index<group_count; group_index++ )); do
  group_seed_start=$(( base_seed_start + group_index * seeds_per_group ))
  begin_run_args=(
    -m HJEEDS.two_d_completion begin-run
    --group-dir "HJEEDS/results/2d_cluster_tests/cluster_${group_index}"
    --seed-start "${group_seed_start}"
    --num-seeds "${seeds_per_group}"
    --expected-agents "${expected_agents_per_seed}"
    --parts-per-group "${parts_per_group}"
  )
  if [[ "${require_paper_config}" == "1" ]]; then
    begin_run_args+=(--require-paper-config)
  fi
  "${metadata_python[@]}" "${begin_run_args[@]}"
done

array_output="$("${array_cmd[@]}")"
echo "${array_output}"
if [[ ! "${array_output}" =~ ([0-9]+)$ ]]; then
  echo "Error: unable to parse the Slurm array job ID from: ${array_output}" >&2
  exit 1
fi
array_job_id="${BASH_REMATCH[1]}"

agg_cmd[2]="--dependency=afterok:${array_job_id}"
agg_output="$("${agg_cmd[@]}")"
echo "${agg_output}"
