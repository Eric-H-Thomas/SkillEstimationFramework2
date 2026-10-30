#!/bin/bash
# Synthetic hockey rationality sample size.
#
# One donor's xG situations, shots drawn with replacement, ten seeds by default.
# Submit from the repository root:
#
#   sbatch run_hockey_rationality_sample_size.sbatch
#   sbatch run_hockey_rationality_sample_size.sbatch Data/Hockey/experiments/rationality_sample_size
#
# Overrides (optional):
#   RATIONALITY_PLAYER_ID, RATIONALITY_NUM_SHOTS, RATIONALITY_NUM_SEEDS,
#   RATIONALITY_SIGMAS, RATIONALITY_LAMBDAS, RATIONALITY_NORMAL_LAMBDA_CENTERS,
#   RATIONALITY_NUM_SIGMA_GRID, RATIONALITY_NUM_LOG_LAMBDA_GRID

#SBATCH --job-name=hockey-rationality-n
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=slurm-hockey-rationality-n-%j.out

set -euo pipefail

OUTPUT_DIR="${1:-Data/Hockey/experiments/rationality_sample_size}"
REPO_ROOT="${HJEEDS_REPO_ROOT:-$PWD}"

cd "$REPO_ROOT"
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate skill-estimation
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python -m HJEEDS.run_hockey_rationality_sample_size \
  --output-dir "$OUTPUT_DIR" \
  --player-id "${RATIONALITY_PLAYER_ID:-950160}" \
  --data-dir "${RATIONALITY_DATA_DIR:-Data/Hockey}" \
  --shot-group "${RATIONALITY_SHOT_GROUP:-wristshot_snapshot}" \
  --num-shots "${RATIONALITY_NUM_SHOTS:-200}" \
  --num-seeds "${RATIONALITY_NUM_SEEDS:-10}" \
  --base-seed "${RATIONALITY_BASE_SEED:-20261023}" \
  --num-sigma-grid "${RATIONALITY_NUM_SIGMA_GRID:-50}" \
  --num-log-lambda-grid "${RATIONALITY_NUM_LOG_LAMBDA_GRID:-100}" \
  --normal-sigma-center "${RATIONALITY_NORMAL_SIGMA_CENTER:-0.075}" \
  --normal-log-sigma-sd "${RATIONALITY_NORMAL_LOG_SIGMA_SD:-0.5}" \
  --normal-log-lambda-sd "${RATIONALITY_NORMAL_LOG_LAMBDA_SD:-1.0}" \
  --sigmas ${RATIONALITY_SIGMAS:-0.075} \
  --lambdas ${RATIONALITY_LAMBDAS:-1 10 100} \
  --normal-lambda-centers ${RATIONALITY_NORMAL_LAMBDA_CENTERS:-10}
