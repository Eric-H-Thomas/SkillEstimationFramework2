#!/bin/bash
# Paper correspondence: Main `subsec:two_d_darts`.
#SBATCH --job-name=hjeeds-2d-cluster
#SBATCH --array=0-9
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

repo_root="${HJEEDS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
cd "${repo_root}"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg

python_bin="${PYTHON_BIN:-}"
conda_env="${CONDA_ENV:-skill-estimation}"

if [[ -n "${python_bin}" ]]; then
  if [[ "${python_bin}" == */* ]]; then
    if [[ ! -x "${python_bin}" ]]; then
      echo "Error: PYTHON_BIN is not an executable file: ${python_bin}" >&2
      exit 1
    fi
  elif ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "Error: PYTHON_BIN is not an executable command: ${python_bin}" >&2
    exit 1
  fi
else
  if ! command -v conda >/dev/null 2>&1; then
    if ! command -v module >/dev/null 2>&1; then
      echo "Error: neither conda nor the cluster module command is available; set PYTHON_BIN." >&2
      exit 1
    fi
    module load miniforge3
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda is unavailable after loading miniforge3; set PYTHON_BIN." >&2
    exit 1
  fi
  eval "$(conda shell.bash hook)"
  conda activate "${conda_env}"
  python_bin="$(command -v python)"
fi

"${python_bin}" -c 'import HJEEDS, numpy, scipy' >/dev/null

parts_per_group="${PARTS_PER_GROUP:-10}"
group_count="${GROUP_COUNT:-1}"
seeds_per_group="${SEEDS_PER_GROUP:-500}"
base_seed_start="${BASE_SEED_START:-1000}"
expected_agents_per_seed="${EXPECTED_AGENTS_PER_SEED:-25}"

if ! [[ "${parts_per_group}" =~ ^[0-9]+$ ]] || (( parts_per_group < 1 )); then
  echo "Error: PARTS_PER_GROUP must be a positive integer." >&2
  exit 1
fi
if ! [[ "${group_count}" =~ ^[0-9]+$ ]] || (( group_count < 1 )); then
  echo "Error: GROUP_COUNT must be a positive integer." >&2
  exit 1
fi
if ! [[ "${seeds_per_group}" =~ ^[0-9]+$ ]] || (( seeds_per_group < 1 )); then
  echo "Error: SEEDS_PER_GROUP must be a positive integer." >&2
  exit 1
fi
if ! [[ "${base_seed_start}" =~ ^[0-9]+$ ]]; then
  echo "Error: BASE_SEED_START must be a nonnegative integer." >&2
  exit 1
fi
if [[ "${HJEEDS_REQUIRE_PAPER_CONFIG:-}" =~ ^(1|true|yes)$ && "${expected_agents_per_seed}" != "25" ]]; then
  echo "Error: EXPECTED_AGENTS_PER_SEED must be 25 when HJEEDS_REQUIRE_PAPER_CONFIG is set." >&2
  exit 1
fi
if (( seeds_per_group % parts_per_group != 0 )); then
  echo "Error: SEEDS_PER_GROUP must be divisible by PARTS_PER_GROUP." >&2
  exit 1
fi

seeds_per_part=$(( seeds_per_group / parts_per_group ))

if [[ "${AGGREGATE_RESULTS:-0}" == "1" ]]; then
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "Skipping aggregation on array task ${SLURM_ARRAY_TASK_ID}." >&2
    exit 0
  fi

  for (( group_index=0; group_index<group_count; group_index++ )); do
    group_dir="HJEEDS/results/2d_cluster_tests/cluster_${group_index}"
    group_seed_start=$(( base_seed_start + group_index * seeds_per_group ))
    aggregate_args=(
      -m HJEEDS.aggregate_cluster_seeds
      --group-dir "${group_dir}"
      --parts-per-group "${parts_per_group}"
      --expected-seed-start "${group_seed_start}"
      --expected-num-seeds "${seeds_per_group}"
      --expected-agents-per-seed "${expected_agents_per_seed}"
      --cleanup
    )
    if [[ "${HJEEDS_REQUIRE_PAPER_CONFIG:-}" =~ ^(1|true|yes)$ ]]; then
      aggregate_args+=(--require-paper-config)
    fi
    "${python_bin}" "${aggregate_args[@]}"
  done
  exit 0
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Error: SLURM_ARRAY_TASK_ID is required for array tasks." >&2
  exit 1
fi

task_index="${SLURM_ARRAY_TASK_ID}"
if ! [[ "${task_index}" =~ ^[0-9]+$ ]]; then
  echo "Error: SLURM_ARRAY_TASK_ID must be a nonnegative integer." >&2
  exit 1
fi
group_index=$(( task_index / parts_per_group ))
sub_index=$(( task_index % parts_per_group ))

if (( group_index >= group_count )); then
  echo "Error: task index ${task_index} exceeds configured group count." >&2
  exit 1
fi

base_seed=$(( base_seed_start + group_index * seeds_per_group ))
seed=$(( base_seed + sub_index * seeds_per_part ))
output_dir="HJEEDS/results/2d_cluster_tests/cluster_${group_index}/part_${sub_index}"

"${python_bin}" -m HJEEDS.two_d_completion run-paper-part \
  --part-dir "${output_dir}" \
  --seed-start "${seed}" \
  --num-seeds "${seeds_per_part}" \
  --expected-agents "${expected_agents_per_seed}"
