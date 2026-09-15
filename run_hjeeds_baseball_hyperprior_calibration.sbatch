#!/bin/bash
# Array / aggregation worker for baseball hyperprior calibration.
# Invoked by submit_hjeeds_baseball_hyperprior_calibration.sh (CLI overrides #SBATCH).
#SBATCH --job-name=hjeeds-baseball-calib
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

# Submit passes --chdir=<repo root>. Prefer the exported launcher directory:
# Slurm may execute a spool copy, so dirname(BASH_SOURCE) can be unreliable.
_common="${HJEEDS_SLURM_DIR:-${HJEEDS_REPO_ROOT:-${PWD}}}/hjeeds_baseball_slurm_common.sh"
if [[ ! -f "${_common}" && -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/hjeeds_baseball_slurm_common.sh" ]]; then
  _common="${SLURM_SUBMIT_DIR}/hjeeds_baseball_slurm_common.sh"
fi
if [[ ! -f "${_common}" ]]; then
  _common="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hjeeds_baseball_slurm_common.sh"
fi
if [[ ! -f "${_common}" ]]; then
  echo "Error: cannot find hjeeds_baseball_slurm_common.sh (PWD=${PWD}, SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset})." >&2
  exit 1
fi
# shellcheck source=hjeeds_baseball_slurm_common.sh
source "${_common}"

hjeeds_baseball_setup_matplotlib_env
python_bin="$(hjeeds_baseball_resolve_cluster_python submit_hjeeds_baseball_hyperprior_calibration.sh)" || exit 1
echo "[baseball-calibrate] Using python: ${python_bin}" >&2
hjeeds_baseball_preflight "${python_bin}"

python_args=(
  -m HJEEDS.baseball_calibrate_hyperpriors
  --seed "${BASE_SEED:-default}"
  --output-dir "${OUTPUT_DIR:?OUTPUT_DIR is required}"
)

if [[ -n "${SEASON_YEAR:-}" ]]; then
  python_args+=(--season-year "${SEASON_YEAR}")
fi
if [[ -n "${PITCH_TYPES:-}" ]]; then
  python_args+=(--pitch-types "${PITCH_TYPES}")
fi
if [[ -n "${MAX_PITCHES_PER_AGENT:-}" ]]; then
  python_args+=(--max-pitches-per-agent "${MAX_PITCHES_PER_AGENT}")
fi
if [[ -n "${CONFIDENCE:-}" ]]; then
  python_args+=(--confidence "${CONFIDENCE}")
fi

if [[ "${AGGREGATE_RESULTS:-0}" == "1" ]]; then
  python_args+=(--aggregate-results)
else
  agent_index="${AGENT_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
  python_args+=(--agent-index "${agent_index}")
fi
if [[ "${HJEEDS_REQUIRE_PAPER_CONFIG:-}" =~ ^(1|true|yes)$ ]]; then
  python_args+=(--require-paper-config)
fi

echo "[baseball-calibrate] Starting: ${python_bin} ${python_args[*]}" >&2
"${python_bin}" "${python_args[@]}"

if [[ "${AGGREGATE_RESULTS:-0}" == "1" ]]; then
  echo "[baseball-calibrate] Validating completion sentinel and output hashes before zip." >&2
  "${python_bin}" -m HJEEDS.baseball_calibrate_hyperpriors \
    --output-dir "${OUTPUT_DIR}" \
    --validate-results
  hjeeds_baseball_zip_results "${python_bin}" "${OUTPUT_DIR}" "baseball-calibrate"
fi
