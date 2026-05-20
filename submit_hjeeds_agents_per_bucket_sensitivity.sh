#!/bin/bash
# This file was written or edited by AI and still requires human review. Delete this comment when done.
#
# Submit the H-JEEDS agents-per-bucket sensitivity sweep as a Slurm array
# Each array task computes one agents-per-bucket x hyperprior scenario, and a
# final dependent job aggregates the scenario folders and zips the output root

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: submit_hjeeds_agents_per_bucket_sensitivity.sh [options]

Submit the H-JEEDS agents-per-bucket ablation to Slurm. By default this
launches 15 scenario tasks:

  agents per bucket: 1,2,5,10,25
  prior conditions: 3 representative hyperprior robustness conditions

Use --condition-preset full_60 to launch 300 scenario tasks instead. Then the
helper submits one afterok aggregation task that collects the scenario outputs
and writes OUTPUT_DIR.zip for export.

Experiment options:
  --num-seeds N                  Seeds per scenario (default: 500).
  --seed N|default               Required base seed. Use default for 12345.
  --count-buckets LIST           Observation buckets (default: 5,10,25,100,1000).
  --agents-per-bucket-values L   Population-size sweep (default: 1,2,5,10,25).
  --condition-preset PRESET      representative or full_60 (default: representative).
  --output-dir PATH              Output root for all results.
  --python-bin PATH              Python executable on the cluster.

Slurm options:
  --job-name NAME                Scenario array job name (default: hjeeds-agents-bucket).
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

count_csv_values() {
  local raw_value="$1"
  local default_count="$2"

  if [[ -z "${raw_value}" ]]; then
    echo "${default_count}"
    return
  fi

  local compact_value="${raw_value// /}"
  local old_ifs="${IFS}"
  IFS=","
  read -r -a values <<< "${compact_value}"
  IFS="${old_ifs}"
  echo "${#values[@]}"
}

condition_count_for_preset() {
  case "$1" in
    representative)
      echo "3"
      ;;
    full_60)
      echo "60"
      ;;
    *)
      echo "Error: --condition-preset must be representative or full_60." >&2
      exit 1
      ;;
  esac
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

num_seeds="500"
base_seed=""
count_buckets="5,10,25,100,1000"
agents_per_bucket_values="1,2,5,10,25"
condition_preset="representative"
output_dir="HJEEDS/results/hierarchical_darts_agents_per_bucket_sensitivity"
python_bin=""

job_name="hjeeds-agents-bucket"
qos="normal"
partition=""
account=""
time_limit="23:00:00"
memory="16G"
cpus_per_task=""
slurm_output=""
dry_run="0"

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
    --count-buckets)
      count_buckets="$2"
      shift 2
      ;;
    --agents-per-bucket-values)
      agents_per_bucket_values="$2"
      shift 2
      ;;
    --condition-preset)
      condition_preset="$2"
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

agents_count="$(count_csv_values "${agents_per_bucket_values}" 0)"
condition_count="$(condition_count_for_preset "${condition_preset}")"
total_tasks=$(( agents_count * condition_count ))
if (( total_tasks < 1 )); then
  echo "Error: no scenario tasks requested." >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
array_script="${script_dir}/run_hjeeds_agents_per_bucket_sensitivity.sbatch"

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
  "COUNT_BUCKETS=${count_buckets}"
  "AGENTS_PER_BUCKET_VALUES=${agents_per_bucket_values}"
  "CONDITION_PRESET=${condition_preset}"
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
echo "Condition preset: ${condition_preset}"
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
