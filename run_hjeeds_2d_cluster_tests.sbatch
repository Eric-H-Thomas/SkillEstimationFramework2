#!/bin/bash
#SBATCH --job-name=hjeeds-2d-cluster
#SBATCH --array=0-99
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

cd "$HOME/research/SkillEstimationFramework2"

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate skill-estimation

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg

python_bin="${PYTHON_BIN:-python3}"

parts_per_group="${PARTS_PER_GROUP:-10}"
group_count="${GROUP_COUNT:-10}"
seeds_per_group="${SEEDS_PER_GROUP:-500}"
base_seed_start="${BASE_SEED_START:-1000}"

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
    group_dir="HJEEDS/results/cluster_${group_index}"
    "${python_bin}" -m HJEEDS.aggregate_cluster_seeds \
      --group-dir "${group_dir}" \
      --parts-per-group "${parts_per_group}" \
      --cleanup
  done
  exit 0
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Error: SLURM_ARRAY_TASK_ID is required for array tasks." >&2
  exit 1
fi

task_index="${SLURM_ARRAY_TASK_ID}"
group_index=$(( task_index / parts_per_group ))
sub_index=$(( task_index % parts_per_group ))

if (( group_index >= group_count )); then
  echo "Error: task index ${task_index} exceeds configured group count." >&2
  exit 1
fi

base_seed=$(( base_seed_start + group_index * 1000 ))
seed=$(( base_seed + sub_index * seeds_per_part ))
output_dir="HJEEDS/results/cluster_${group_index}/part_${sub_index}"

"${python_bin}" -m HJEEDS.darts_hierarchical_vs_jeeds \
  --num-seeds "${seeds_per_part}" \
  --environment 2d \
  --output-dir "${output_dir}" \
  --seed "${seed}"