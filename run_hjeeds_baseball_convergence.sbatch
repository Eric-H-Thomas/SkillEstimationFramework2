#!/bin/bash
# Single-job worker for Statcast baseball HJEEDS convergence.
# Invoked by submit_hjeeds_baseball_convergence.sh (CLI --time/--mem override these).
#SBATCH --job-name=hjeeds-baseball-conv
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Submit passes --chdir=<repo root>. Prefer PWD over BASH_SOURCE: Slurm may
# execute a spool copy of this .sbatch, so dirname(BASH_SOURCE) would be wrong.
_common="${PWD}/hjeeds_baseball_slurm_common.sh"
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
python_bin="$(hjeeds_baseball_resolve_cluster_python submit_hjeeds_baseball_convergence.sh)" || exit 1
echo "[baseball-convergence] Using python: ${python_bin}" >&2

"${python_bin}" - <<'PY'
import sys

required = ("numpy", "pandas", "torch", "scipy")
missing = []
for module_name in required:
    try:
        __import__(module_name)
    except ImportError:
        missing.append(module_name)

if missing:
    print(
        "Error: Python environment is missing required packages: "
        + ", ".join(missing),
        file=sys.stderr,
    )
    print(f"Interpreter: {sys.executable}", file=sys.stderr)
    print(
        "Activate conda env skill-estimation on the cluster, or pass "
        "--python-bin /path/to/python via submit_hjeeds_baseball_convergence.sh.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

python_args=(
  -m HJEEDS.baseball_convergence_study
  --seed "${BASE_SEED:?BASE_SEED is required}"
  --num-seeds "${NUM_SEEDS:-1}"
  --convergence-ns "${CONVERGENCE_NS:-5,10,25,50,100}"
  --output-dir "${OUTPUT_DIR:-HJEEDS/results/baseball_convergence_n100}"
)

if [[ -n "${SEASON_YEAR:-}" ]]; then
  python_args+=(--season-year "${SEASON_YEAR}")
fi
if [[ "${ALL_ELIGIBLE_AGENTS:-0}" == "1" ]]; then
  python_args+=(--all-eligible-agents)
elif [[ -n "${TOP_PITCHERS:-}" ]]; then
  python_args+=(--top-pitchers "${TOP_PITCHERS}")
elif [[ -n "${BBIP_EXTREMES:-}" ]]; then
  python_args+=(--bbip-extremes "${BBIP_EXTREMES}")
elif [[ -n "${PITCHER_IDS:-}" ]]; then
  python_args+=(--pitcher-ids "${PITCHER_IDS}")
fi
if [[ -n "${PITCH_TYPES:-}" ]]; then
  python_args+=(--pitch-types "${PITCH_TYPES}")
fi
if [[ -n "${MAX_REFERENCE_PITCHES:-}" ]]; then
  python_args+=(--max-reference-pitches "${MAX_REFERENCE_PITCHES}")
fi
if [[ -n "${MIN_PITCHES_PER_AGENT:-}" ]]; then
  python_args+=(--min-pitches-per-agent "${MIN_PITCHES_PER_AGENT}")
fi
if [[ -n "${HYPERPRIOR_PRESET:-}" ]]; then
  python_args+=(--hyperprior-preset "${HYPERPRIOR_PRESET}")
fi
if [[ -n "${HYPERPRIOR_CONFIG:-}" ]]; then
  python_args+=(--hyperprior-config "${HYPERPRIOR_CONFIG}")
fi
if [[ -n "${BBIP_CACHE_PATH:-}" ]]; then
  export BBIP_CACHE_PATH
fi

echo "[baseball-convergence] Starting: ${python_bin} ${python_args[*]}" >&2
"${python_bin}" "${python_args[@]}"

output_dir="${OUTPUT_DIR:-HJEEDS/results/baseball_convergence_n100}"
hjeeds_baseball_zip_results "${python_bin}" "${output_dir}" "baseball-convergence"
