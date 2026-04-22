#!/bin/bash
# This file still requires human verification. Delete this comment when done.
#SBATCH --job-name=hier-darts
#SBATCH --qos=normal
#SBATCH --time=23:00:00
#SBATCH --mem=8G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# This wrapper is intentionally thin.  Its job is not to reimplement the
# experiment logic, but to provide a cluster-friendly way to launch the Python
# entrypoint with sensible defaults for paths and environment setup.

# Submit this script from the repository root when possible:
#
#   sbatch HJEEDS/run_hierarchical_darts.sbatch --num-seeds 30
#
# If you submit it from another directory, set REPO_DIR explicitly:
#
#   sbatch --export=ALL,REPO_DIR=/path/to/SkillEstimationFramework2 HJEEDS/run_hierarchical_darts.sbatch --num-seeds 30
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_DIR}"

# Make sure imports resolve the same way they do when running from the repo root.
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

# Render figures to files on headless compute nodes.
export MPLBACKEND="${MPLBACKEND:-Agg}"

# Prefer the repo-local virtual environment used by the existing Slurm scripts,
# but allow PYTHON_BIN to override it if the supercomputer uses modules or conda.
if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="${PYTHON_BIN}"
elif [[ -x "${PWD}/venv/bin/python" ]]; then
  python_bin="${PWD}/venv/bin/python"
else
  python_bin="python"
fi

if ! command -v "${python_bin}" >/dev/null 2>&1 && [[ ! -x "${python_bin}" ]]; then
  echo "Error: Python interpreter not found: ${python_bin}" >&2
  echo "Set PYTHON_BIN or create ${PWD}/venv before submitting." >&2
  exit 1
fi

BASE_SEED="${BASE_SEED:-12345}"
OUTPUT_DIR="${OUTPUT_DIR:-HJEEDS/results/hierarchical_darts}"

has_num_seeds_arg="false"
for arg in "$@"; do
  case "${arg}" in
    --num-seeds|--num-seeds=*)
      has_num_seeds_arg="true"
      ;;
  esac
done

# Build the fixed part of the Python command first, then let the caller append
# or override options through normal CLI arguments.
python_args=(
  HJEEDS/darts_hierarchical_vs_jeeds.py
  --seed "${BASE_SEED}"
  --num-agents 20
  --count-buckets 5,10,25,100
  --agents-per-bucket 5
  --delta 0.1
  --num-sigma-grid 21
  --num-lambda-grid 21
  --output-dir "${OUTPUT_DIR}"
)

if [[ "${has_num_seeds_arg}" != "true" ]]; then
  # The project intentionally requires an explicit seed count so long runs do
  # not silently inherit an accidental default.
  if [[ -n "${NUM_SEEDS:-}" ]]; then
    python_args+=(--num-seeds "${NUM_SEEDS}")
  else
    echo "Error: supply the number of seeds intentionally." >&2
    echo "Pass --num-seeds to the sbatch script or set NUM_SEEDS in the environment." >&2
    exit 1
  fi
fi

# Any extra arguments passed after the sbatch script name are forwarded to the
# Python experiment, so you can provide --num-seeds and override defaults
# without editing this file.
python_args+=("$@")

echo "[hier-darts-slurm] Repository: ${PWD}"
echo "[hier-darts-slurm] Python: ${python_bin}"
echo "[hier-darts-slurm] Output directory: ${OUTPUT_DIR}"
echo "[hier-darts-slurm] Command: ${python_bin} ${python_args[*]}"

"${python_bin}" "${python_args[@]}"
