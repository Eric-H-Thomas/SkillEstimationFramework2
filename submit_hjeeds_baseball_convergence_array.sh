#!/bin/bash
# This file was written or edited by AI and still requires human review. Delete this comment when done.
#
# Submit a Slurm array for per-agent baseball convergence caches, then aggregate.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: submit_hjeeds_baseball_convergence_array.sh [options]

Workflow:
  1. On the login node, write convergence_roster.json and bbip_innings_cache.json (if needed).
  2. Submit one Slurm array task per agent (expensive RNN + per-N grids).
  3. Submit one aggregation task (afterok) to fit population MAP and drift.

Experiment options:
  --seed N|default               Base seed (default: default). Unused for Statcast numerics;
                                 kept for CLI compatibility. Prefer a single seed.
  --season-year YEAR             Restrict to one Statcast season (e.g. 2021).
  --pitch-types TYPES            Comma-separated pitch types (default: FF).
  --all-eligible-agents          Use every eligible (pitcher, pitchType) pair.
  --top-pitchers N               Alternative roster selector.
  --bbip-extremes N              Top-N + bottom-N by season BB/IP.
  --pitcher-ids IDS              Alternative roster selector.
  --min-pitches-per-agent N      Minimum pitches per agent (default: 100).
  --max-agents N                 Cap roster size (smoke tests).
  --convergence-ns LIST          Comma-separated N values (default: 5,10,25,50,100).
  --max-reference-pitches N      Cap reference pitches per agent (default: 100).
  --hyperprior-preset PRESET     darts|low-confidence|baseball-2021-ff|calibrated.
  --hyperprior-config PATH       JSON hyperprior file (required for calibrated).
  --output-dir PATH              Output root.
  --python-bin PATH              Python executable.
  --conda-env NAME               Conda env when PYTHON_BIN is unset.

Slurm options:
  --job-name NAME                Array job name (default: hjeeds-baseball-conv).
  --qos QOS                      Slurm QOS (default: normal).
  --partition PARTITION          Optional Slurm partition.
  --account ACCOUNT              Optional Slurm account.
  --time HH:MM:SS                Per-agent walltime (default: 12:00:00).
  --agg-time HH:MM:SS            Aggregation walltime (default: 01:00:00).
  --mem MEM                      Memory per agent task (default: 8G).
  --agg-mem MEM                  Aggregation memory (default: 4G).
  --cpus-per-task N              CPUs per agent task (default: 1).
  --output PATH                  Slurm stdout pattern.
  --dry-run                      Print commands without submitting.
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

base_seed="default"
season_year=""
pitch_types="FF"
all_eligible_agents="0"
top_pitchers=""
bbip_extremes=""
pitcher_ids=""
min_pitches_per_agent="100"
max_agents=""
convergence_ns="5,10,25,50,100"
max_reference_pitches="100"
hyperprior_preset=""
hyperprior_config=""
output_dir="HJEEDS/results/baseball_convergence_array"
python_bin=""
conda_env="skill-estimation"

job_name="hjeeds-baseball-conv"
qos="normal"
partition=""
account=""
time_limit="12:00:00"
agg_time_limit="01:00:00"
memory="8G"
agg_memory="4G"
cpus_per_task="1"
slurm_output=""
dry_run="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --seed)
      base_seed="$2"
      shift 2
      ;;
    --season-year)
      season_year="$2"
      shift 2
      ;;
    --pitch-types)
      pitch_types="$2"
      shift 2
      ;;
    --all-eligible-agents)
      all_eligible_agents="1"
      shift
      ;;
    --top-pitchers)
      top_pitchers="$2"
      all_eligible_agents="0"
      shift 2
      ;;
    --bbip-extremes)
      bbip_extremes="$2"
      all_eligible_agents="0"
      shift 2
      ;;
    --pitcher-ids)
      pitcher_ids="$2"
      all_eligible_agents="0"
      shift 2
      ;;
    --min-pitches-per-agent)
      min_pitches_per_agent="$2"
      shift 2
      ;;
    --max-agents)
      max_agents="$2"
      shift 2
      ;;
    --convergence-ns)
      convergence_ns="$2"
      shift 2
      ;;
    --max-reference-pitches)
      max_reference_pitches="$2"
      shift 2
      ;;
    --hyperprior-preset)
      hyperprior_preset="$2"
      shift 2
      ;;
    --hyperprior-config)
      hyperprior_config="$2"
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
    --conda-env)
      conda_env="$2"
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
    --agg-time)
      agg_time_limit="$2"
      shift 2
      ;;
    --mem)
      memory="$2"
      shift 2
      ;;
    --agg-mem)
      agg_memory="$2"
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

if [[ "${base_seed}" =~ ^[Dd][Ee][Ff][Aa][Uu][Ll][Tt]$ ]]; then
  base_seed="default"
elif ! [[ "${base_seed}" =~ ^[0-9]+$ ]]; then
  echo "Error: --seed must be a nonnegative integer or 'default'." >&2
  exit 1
fi

roster_flags=0
if [[ "${all_eligible_agents}" == "1" ]]; then
  roster_flags=$((roster_flags + 1))
fi
if [[ -n "${top_pitchers}" ]]; then
  roster_flags=$((roster_flags + 1))
fi
if [[ -n "${bbip_extremes}" ]]; then
  roster_flags=$((roster_flags + 1))
fi
if [[ -n "${pitcher_ids}" ]]; then
  roster_flags=$((roster_flags + 1))
fi
if [[ "${roster_flags}" -gt 1 ]]; then
  echo "Error: Use exactly one roster selector among --all-eligible-agents, --top-pitchers, --bbip-extremes, and --pitcher-ids." >&2
  exit 1
fi
if [[ "${roster_flags}" -eq 0 ]]; then
  echo "Error: Provide one roster selector for array convergence." >&2
  exit 1
fi
if [[ -n "${bbip_extremes}" && -z "${season_year}" ]]; then
  echo "Error: --bbip-extremes requires --season-year." >&2
  exit 1
fi
if [[ "${hyperprior_preset}" == "calibrated" && -z "${hyperprior_config}" ]]; then
  echo "Error: --hyperprior-config is required when --hyperprior-preset calibrated." >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
array_script="${script_dir}/run_hjeeds_baseball_convergence_array.sbatch"

resolve_python() {
  if [[ -n "${python_bin}" ]]; then
    echo "${python_bin}"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    conda run -n "${conda_env}" which python 2>/dev/null && return
  fi
  if command -v module >/dev/null 2>&1; then
    module load miniforge3
    eval "$(conda shell.bash hook)"
    conda activate "${conda_env}"
    command -v python
    return
  fi
  command -v python3
}

resolved_python="$(resolve_python)"
if [[ ! -x "${resolved_python}" ]]; then
  echo "Error: Could not resolve a Python interpreter. Pass --python-bin." >&2
  exit 1
fi

prepare_args=(
  -m HJEEDS.baseball_convergence_study
  --prepare-roster
  --output-dir "${output_dir}"
  --pitch-types "${pitch_types}"
  --min-pitches-per-agent "${min_pitches_per_agent}"
  --convergence-ns "${convergence_ns}"
  --max-reference-pitches "${max_reference_pitches}"
)
if [[ -n "${season_year}" ]]; then
  prepare_args+=(--season-year "${season_year}")
fi
if [[ "${all_eligible_agents}" == "1" ]]; then
  prepare_args+=(--all-eligible-agents)
elif [[ -n "${top_pitchers}" ]]; then
  prepare_args+=(--top-pitchers "${top_pitchers}")
elif [[ -n "${bbip_extremes}" ]]; then
  prepare_args+=(--bbip-extremes "${bbip_extremes}")
elif [[ -n "${pitcher_ids}" ]]; then
  prepare_args+=(--pitcher-ids "${pitcher_ids}")
fi
if [[ -n "${max_agents}" ]]; then
  prepare_args+=(--max-agents "${max_agents}")
fi
if [[ -n "${hyperprior_preset}" ]]; then
  prepare_args+=(--hyperprior-preset "${hyperprior_preset}")
fi
if [[ -n "${hyperprior_config}" ]]; then
  prepare_args+=(--hyperprior-config "${hyperprior_config}")
fi

echo "Preparing convergence roster locally..."
echo "  ${resolved_python} ${prepare_args[*]}"
(
  cd "${script_dir}"
  export PYTHONPATH="${script_dir}${PYTHONPATH:+:$PYTHONPATH}"
  "${resolved_python}" "${prepare_args[@]}"
)

roster_file="${script_dir}/${output_dir}/convergence_roster.json"
agent_count="$(
  PYTHONPATH="${script_dir}${PYTHONPATH:+:$PYTHONPATH}" \
  "${resolved_python}" - "${roster_file}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
print(len(payload))
PY
)"

if ! [[ "${agent_count}" =~ ^[0-9]+$ ]] || [[ "${agent_count}" -lt 1 ]]; then
  echo "Error: roster is empty or missing (${roster_file})." >&2
  exit 1
fi

common_sbatch_args=(
  --job-name="${job_name}"
  --qos="${qos}"
  --mem="${memory}"
  --cpus-per-task="${cpus_per_task}"
  --chdir="${script_dir}"
)
if [[ -n "${partition}" ]]; then
  common_sbatch_args+=(--partition="${partition}")
fi
if [[ -n "${account}" ]]; then
  common_sbatch_args+=(--account="${account}")
fi
if [[ -n "${slurm_output}" ]]; then
  common_sbatch_args+=(--output="${slurm_output}")
fi

experiment_env=(
  "BASE_SEED=${base_seed}"
  "NUM_SEEDS=1"
  "OUTPUT_DIR=${output_dir}"
  "PITCH_TYPES=${pitch_types}"
  "CONVERGENCE_NS=${convergence_ns}"
  "MAX_REFERENCE_PITCHES=${max_reference_pitches}"
  "MIN_PITCHES_PER_AGENT=${min_pitches_per_agent}"
)
if [[ -n "${season_year}" ]]; then
  experiment_env+=("SEASON_YEAR=${season_year}")
fi
if [[ -n "${hyperprior_preset}" ]]; then
  experiment_env+=("HYPERPRIOR_PRESET=${hyperprior_preset}")
fi
if [[ -n "${hyperprior_config}" ]]; then
  experiment_env+=("HYPERPRIOR_CONFIG=${hyperprior_config}")
fi
bbip_cache_path="${script_dir}/${output_dir}/bbip_innings_cache.json"
if [[ -f "${bbip_cache_path}" ]]; then
  experiment_env+=("BBIP_CACHE_PATH=${bbip_cache_path}")
fi
if [[ -n "${python_bin}" ]]; then
  experiment_env+=("PYTHON_BIN=${python_bin}")
else
  experiment_env+=("CONDA_ENV=${conda_env}")
fi

array_sbatch_args=("${common_sbatch_args[@]}")
array_sbatch_args[0]="--job-name=${job_name}"
array_sbatch_args+=(--time="${time_limit}")

array_cmd=(
  env "${experiment_env[@]}"
  sbatch
  --export=ALL
  --array=0-$((agent_count - 1))
  "${array_sbatch_args[@]}"
  "${array_script}"
)

echo "Baseball convergence array:"
echo "  agents=${agent_count}"
echo "  season_year=${season_year:-all}"
echo "  pitch_types=${pitch_types}"
echo "  convergence_ns=${convergence_ns}"
echo "  output_dir=${output_dir}"
echo "Array command:"
echo "  $(format_command "${array_cmd[@]}")"

if [[ "${dry_run}" == "1" ]]; then
  agg_preview_args=("${common_sbatch_args[@]}")
  agg_preview_args[0]="--job-name=${job_name}-agg"
  agg_preview_args+=(--time="${agg_time_limit}" --mem="${agg_memory}")
  aggregate_preview=(
    env "${experiment_env[@]}" "AGGREGATE_RESULTS=1"
    sbatch
    --export=ALL
    "--dependency=afterok:<array_job_id>"
    "${agg_preview_args[@]}"
    "${array_script}"
  )
  echo "Aggregation command:"
  echo "  $(format_command "${aggregate_preview[@]}")"
  exit 0
fi

array_output="$("${array_cmd[@]}")"
echo "${array_output}"
array_job_id="$(awk '{print $4}' <<< "${array_output}")"

agg_sbatch_args=("${common_sbatch_args[@]}")
agg_sbatch_args[0]="--job-name=${job_name}-agg"
agg_sbatch_args+=(--time="${agg_time_limit}" --mem="${agg_memory}")

aggregate_cmd=(
  env "${experiment_env[@]}" "AGGREGATE_RESULTS=1"
  sbatch
  --export=ALL
  --dependency=afterok:${array_job_id}
  "${agg_sbatch_args[@]}"
  "${array_script}"
)

echo "Submitting aggregation task after array job ${array_job_id}."
echo "Aggregation command:"
echo "  $(format_command "${aggregate_cmd[@]}")"
aggregate_output="$("${aggregate_cmd[@]}")"
echo "${aggregate_output}"
