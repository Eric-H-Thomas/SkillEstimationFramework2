#!/bin/bash
# This file was written or edited by AI and still requires human review. Delete this comment when done.
#
# Submit a single Slurm job for the Statcast baseball HJEEDS convergence study.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: submit_hjeeds_baseball_convergence.sh [options]

Submit one Slurm job that runs HJEEDS.baseball_convergence_study and zips the
output directory when finished.

Experiment options:
  --seed N|default               Required base seed. Use default for 12345.
  --num-seeds N                  Seeds to run (default: 1).
  --season-year YEAR             Restrict to one Statcast season (e.g. 2021).
  --top-pitchers N               Select top-N eligible pitchers.
  --bbip-extremes N              Select top-N and bottom-N pitchers by season BB/IP.
  --all-eligible-agents          Use every eligible (pitcher, pitchType) pair.
  --pitcher-ids IDS              Comma-separated pitcher IDs (default roster selector).
  --pitch-types TYPES            Comma-separated pitch types (default: FF).
  --convergence-ns LIST          Comma-separated N values (default: 5,10,25,50,100).
  --max-reference-pitches N      Cap reference pitches per agent (default: 100).
  --min-pitches-per-agent N      Override auto min-pitch threshold.
  --hyperprior-preset PRESET     darts|low-confidence|baseball-2021-ff|calibrated.
  --hyperprior-config PATH       JSON hyperprior file (required for calibrated preset).
  --output-dir PATH              Output directory.
  --python-bin PATH              Python executable (default: module load miniforge3 + conda activate skill-estimation).
  --conda-env NAME               Conda env to activate when PYTHON_BIN is unset (default: skill-estimation).

Slurm options:
  --job-name NAME                Job name (default: hjeeds-baseball-conv).
  --qos QOS                      Slurm QOS (default: normal).
  --partition PARTITION          Optional Slurm partition.
  --account ACCOUNT              Optional Slurm account.
  --time HH:MM:SS                Walltime (default: 04:00:00).
  --mem MEM                      Memory (default: 8G).
  --cpus-per-task N              CPUs per task (default: 1).
  --output PATH                  Slurm stdout path pattern.
  --dry-run                      Print sbatch command without submitting.
  -h, --help                     Show this help.

Use exactly one roster selector: --all-eligible-agents, --top-pitchers,
--bbip-extremes, or --pitcher-ids.
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

base_seed=""
num_seeds="1"
season_year=""
top_pitchers=""
bbip_extremes=""
all_eligible_agents="0"
pitcher_ids=""
pitch_types="FF"
convergence_ns="5,10,25,50,100"
max_reference_pitches="100"
min_pitches_per_agent=""
hyperprior_preset=""
hyperprior_config=""
output_dir="HJEEDS/results/baseball_convergence_n100"
python_bin=""
conda_env="skill-estimation"

job_name="hjeeds-baseball-conv"
qos="normal"
partition=""
account=""
time_limit="04:00:00"
memory="8G"
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
    --num-seeds)
      num_seeds="$2"
      shift 2
      ;;
    --season-year)
      season_year="$2"
      shift 2
      ;;
    --top-pitchers)
      top_pitchers="$2"
      shift 2
      ;;
    --bbip-extremes)
      bbip_extremes="$2"
      shift 2
      ;;
    --all-eligible-agents)
      all_eligible_agents="1"
      shift
      ;;
    --pitcher-ids)
      pitcher_ids="$2"
      shift 2
      ;;
    --pitch-types)
      pitch_types="$2"
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
    --min-pitches-per-agent)
      min_pitches_per_agent="$2"
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
if [[ -n "${bbip_extremes}" && -z "${season_year}" ]]; then
  echo "Error: --bbip-extremes requires --season-year." >&2
  exit 1
fi
if [[ "${hyperprior_preset}" == "calibrated" && -z "${hyperprior_config}" ]]; then
  echo "Error: --hyperprior-config is required when --hyperprior-preset calibrated." >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch_script="${script_dir}/run_hjeeds_baseball_convergence.sbatch"

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

bbip_cache_path=""
if [[ -n "${bbip_extremes}" ]]; then
  resolved_python="$(resolve_python)"
  if [[ ! -x "${resolved_python}" ]]; then
    echo "Error: Could not resolve a Python interpreter for BB/IP cache prep. Pass --python-bin." >&2
    exit 1
  fi
  mkdir -p "${script_dir}/${output_dir}"
  echo "Writing BB/IP innings cache on login node (compute nodes have no network)..."
  (
    cd "${script_dir}"
    export PYTHONPATH="${script_dir}${PYTHONPATH:+:$PYTHONPATH}"
    "${resolved_python}" -m HJEEDS.baseball_bbip \
      --write-cache \
      --season-year "${season_year}" \
      --output-dir "${output_dir}"
  )
  bbip_cache_path="${script_dir}/${output_dir}/bbip_innings_cache.json"
fi

experiment_env=(
  "BASE_SEED=${base_seed}"
  "NUM_SEEDS=${num_seeds}"
  "CONVERGENCE_NS=${convergence_ns}"
  "OUTPUT_DIR=${output_dir}"
  "PITCH_TYPES=${pitch_types}"
)
if [[ -n "${season_year}" ]]; then
  experiment_env+=("SEASON_YEAR=${season_year}")
fi
if [[ "${all_eligible_agents}" == "1" ]]; then
  experiment_env+=("ALL_ELIGIBLE_AGENTS=1")
elif [[ -n "${top_pitchers}" ]]; then
  experiment_env+=("TOP_PITCHERS=${top_pitchers}")
elif [[ -n "${bbip_extremes}" ]]; then
  experiment_env+=("BBIP_EXTREMES=${bbip_extremes}")
elif [[ -n "${pitcher_ids}" ]]; then
  experiment_env+=("PITCHER_IDS=${pitcher_ids}")
else
  experiment_env+=("TOP_PITCHERS=2")
fi
if [[ -n "${max_reference_pitches}" ]]; then
  experiment_env+=("MAX_REFERENCE_PITCHES=${max_reference_pitches}")
fi
if [[ -n "${min_pitches_per_agent}" ]]; then
  experiment_env+=("MIN_PITCHES_PER_AGENT=${min_pitches_per_agent}")
fi
if [[ -n "${hyperprior_preset}" ]]; then
  experiment_env+=("HYPERPRIOR_PRESET=${hyperprior_preset}")
fi
if [[ -n "${hyperprior_config}" ]]; then
  experiment_env+=("HYPERPRIOR_CONFIG=${hyperprior_config}")
fi
if [[ -n "${bbip_cache_path}" ]]; then
  experiment_env+=("BBIP_CACHE_PATH=${bbip_cache_path}")
fi
if [[ -n "${python_bin}" ]]; then
  experiment_env+=("PYTHON_BIN=${python_bin}")
else
  experiment_env+=("CONDA_ENV=${conda_env}")
fi

sbatch_args=(
  --job-name="${job_name}"
  --qos="${qos}"
  --time="${time_limit}"
  --mem="${memory}"
  --cpus-per-task="${cpus_per_task}"
  --chdir="${script_dir}"
)
if [[ -n "${partition}" ]]; then
  sbatch_args+=(--partition="${partition}")
fi
if [[ -n "${account}" ]]; then
  sbatch_args+=(--account="${account}")
fi
if [[ -n "${slurm_output}" ]]; then
  sbatch_args+=(--output="${slurm_output}")
fi

submit_cmd=(
  env "${experiment_env[@]}"
  sbatch
  --export=ALL
  "${sbatch_args[@]}"
  "${sbatch_script}"
)

echo "Baseball convergence study:"
echo "  seed=${base_seed} num_seeds=${num_seeds}"
echo "  season_year=${season_year:-all}"
if [[ "${all_eligible_agents}" == "1" ]]; then
  echo "  roster=all_eligible_agents"
elif [[ -n "${top_pitchers}" ]]; then
  echo "  top_pitchers=${top_pitchers}"
elif [[ -n "${bbip_extremes}" ]]; then
  echo "  bbip_extremes=${bbip_extremes}"
elif [[ -n "${pitcher_ids}" ]]; then
  echo "  pitcher_ids=${pitcher_ids}"
else
  echo "  top_pitchers=2 (default)"
fi
echo "  hyperprior_preset=${hyperprior_preset:-low-confidence (default)}"
echo "  convergence_ns=${convergence_ns}"
echo "  max_reference_pitches=${max_reference_pitches:-all}"
echo "  output_dir=${output_dir}"
echo "Submit command:"
echo "  $(format_command "${submit_cmd[@]}")"

if [[ "${dry_run}" == "1" ]]; then
  exit 0
fi

submit_output="$("${submit_cmd[@]}")"
echo "${submit_output}"
