#!/bin/bash
#SBATCH --job-name=hjeeds-2d-cluster
#SBATCH --array=0-9
#SBATCH --qos=normal
#SBATCH --time=32:00:00
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

task_index="${SLURM_ARRAY_TASK_ID:-0}"
seed=$(( (task_index + 1) * 1000 ))
output_dir="HJEEDS/results/cluster_${task_index}"

"${python_bin}" -m HJEEDS.darts_hierarchical_vs_jeeds \
  --num-seeds 500 \
  --environment 2d \
  --output-dir "${output_dir}" \
  --seed "${seed}"