#!/bin/bash
# This file was written or edited by AI and still requires human review. Delete this comment when done.
#
# Submit the H-JEEDS population-shape sensitivity sweep as a Slurm array
# Each array task computes one population-shape x agents-per-bucket scenario, and a
# final dependent job aggregates the scenario folders and zips the output root

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: submit_hjeeds_population_shape_sensitivity.sh [options]

Submit the H-JEEDS population-shape ablation to Slurm. By default this launches
20 scenario tasks:

  population shapes: default,uniform,bimodal,outlier_heavy
  agents per bucket: 1,2,5,10,25

The helper submits one afterok aggregation task that collects the scenario
outputs and writes OUTPUT_DIR.zip for export.

Experiment options:
  --num-seeds N                  Seeds per scenario (default: 250).
  --seed N|default               Required base seed. Use default for 12345.
  --output-dir PATH              Output root for all results.
  --python-bin PATH              Python executable on the cluster.

Slurm options:
  --job-name NAME                Scenario array job name (default: hjeeds-pop-shape).
  --qos QOS                      Slurm QOS (default: normal).
  --partition PARTITION          Optional Slurm partition.
  --account ACCOUNT              Optional Slurm account.
  --time HH:MM:SS                Time limit for scenario tasks (default: 23:00:00).
  --mem MEM                      Memory for scenario tasks (default: 16G).
  --cpus-per-task N              Optional CPUs per task.
  --output PATH                  Optional Slurm stdout pattern.
  --dry-run                      Print sbatch commands without submitting.
  -h, --help                     Show this help.
USAGE
}

format_command() {
  local quoted=()
  local part
  for part in "$@"; do
    printf -v part "%q" "${part}"
    quoted+=("${part}")
  done
  printf "%s" "${quoted[*]}"
}

num_seeds="250"
base_seed=""
output_dir="HJEEDS/results/hierarchical_darts_population_shape_sensitivity"
python_bin=""

job_name="hjeeds-pop-shape"
qos="normal"
partition=""
account=""
time_limit="23:00:00"
memory="16G"
cpus_per_task=""
slurm_output=""
dry_run="0"
population_shape_count="4"
agents_per_bucket_count="5"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --num-seeds)
      num_seeds="$2"
      shift 2
      ;;
    --seed)
      base_seed="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --python-bin)
      python_bin="$2"
      shift 2
      ;;
    --job-name)
      job_name="$2"
      shift 2
      ;;
    --qos)
      qos="$2"
      shift 2
      ;;
    --partition)
      partition="$2"
      shift 2
      ;;
    --account)
      account="$2"
      shift 2
      ;;
    --time)
      time_limit="$2"
      shift 2
      ;;
    --mem)
      memory="$2"
      shift 2
      ;;
    --cpus-per-task)
      cpus_per_task="$2"
      shift 2
      ;;
    --output)
      slurm_output="$2"
      shift 2
      ;;
    --dry-run)
      dry_run="1"
      shift
      ;;
    *)
      echo "Error: Unrecognized option '$1'." >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! [[ "${num_seeds}" =~ ^[0-9]+$ ]] || (( num_seeds < 1 )); then
  echo "Error: --num-seeds must be a positive integer." >&2
  exit 1
fi
if [[ -z "${base_seed}" ]]; then
  echo "Error: --seed is required. Use an integer seed or 'default'." >&2
  exit 1
fi
if [[ "${base_seed}" =~ ^[Dd][Ee][Ff][Aa][Uu][Ll][Tt]$ ]]; then
  base_seed="default"
elif ! [[ "${base_seed}" =~ ^[0-9]+$ ]]; then
  echo "Error: --seed must be a nonnegative integer or 'default'." >&2
  exit 1
fi

total_tasks=$(( agents_per_bucket_count * population_shape_count ))
if (( total_tasks < 1 )); then
  echo "Error: no scenario tasks requested." >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
array_script="${script_dir}/run_hjeeds_population_shape_sensitivity.sbatch"

common_sbatch_args=(
  --job-name="${job_name}"
  --qos="${qos}"
  --time="${time_limit}"
  --mem="${memory}"
  --chdir="${script_dir}"
)

if [[ -n "${partition}" ]]; then
  common_sbatch_args+=(--partition="${partition}")
fi
if [[ -n "${account}" ]]; then
  common_sbatch_args+=(--account="${account}")
fi
if [[ -n "${cpus_per_task}" ]]; then
  common_sbatch_args+=(--cpus-per-task="${cpus_per_task}")
fi
if [[ -n "${slurm_output}" ]]; then
  common_sbatch_args+=(--output="${slurm_output}")
fi

experiment_env=(
  "NUM_SEEDS=${num_seeds}"
  "BASE_SEED=${base_seed}"
  "OUTPUT_DIR=${output_dir}"
)
if [[ -n "${python_bin}" ]]; then
  experiment_env+=("PYTHON_BIN=${python_bin}")
fi

array_cmd=(
  env "${experiment_env[@]}"
  sbatch
  --export=ALL
  --array=0-$((total_tasks - 1))
  "${common_sbatch_args[@]}"
  "${array_script}"
)

echo "Submitting ${total_tasks} scenario tasks."
echo "Population shapes: default, uniform, bimodal, outlier_heavy"
echo "Scenario array command:"
echo "  $(format_command "${array_cmd[@]}")"

if [[ "${dry_run}" == "1" ]]; then
  aggregate_preview_args=("${common_sbatch_args[@]}")
  aggregate_preview_args[0]="--job-name=${job_name}-agg"
  aggregate_preview=(
    env "${experiment_env[@]}" "AGGREGATE_RESULTS=1"
    sbatch
    --export=ALL
    "--dependency=afterok:<scenario_job_id>"
    "${aggregate_preview_args[@]}"
    "${array_script}"
  )
  echo "Aggregation command:"
  echo "  $(format_command "${aggregate_preview[@]}")"
  exit 0
fi

array_output="$("${array_cmd[@]}")"
echo "${array_output}"
array_job_id="$(awk '{print $4}' <<< "${array_output}")"

aggregate_sbatch_args=("${common_sbatch_args[@]}")
aggregate_sbatch_args[0]="--job-name=${job_name}-agg"

aggregate_cmd=(
  env "${experiment_env[@]}" "AGGREGATE_RESULTS=1"
  sbatch
  --export=ALL
  --dependency=afterok:${array_job_id}
  "${aggregate_sbatch_args[@]}"
  "${array_script}"
)

echo "Submitting aggregation task after scenario job ${array_job_id}."
echo "Aggregation command:"
echo "  $(format_command "${aggregate_cmd[@]}")"
aggregate_output="$("${aggregate_cmd[@]}")"
echo "${aggregate_output}"
