#!/bin/bash
#
# Iterate over multiple random seeds and rerun the JEEDS sensitivity experiment
# for each value, mirroring the configuration from
# ``run_darts_jeeds_sensitivity_exp.sbatch``. Results for each seed are written
# to a dedicated output directory.

set -euo pipefail

if [[ $# -gt 0 ]]; then
  seeds=("$@")
else
  # Default seeds chosen to provide a modest coverage of the reward landscapes.
  seeds=(7 11 13 17 23)
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for seed in "${seeds[@]}"; do
  output_dir="Testing/results/high_res_seed_${seed}"
  echo "==== Running darts JEEDS sensitivity experiment with seed ${seed} ===="
  echo "Output directory: ${output_dir}"
  SEED="${seed}" OUTPUT_DIR="${output_dir}" bash "${script_dir}/run_darts_jeeds_sensitivity_exp.sbatch"
done
