#!/bin/bash
# Shared helpers for baseball HJEEDS submit_*.sh / run_*.sbatch scripts.
# Source from the repo root:  source "${script_dir}/hjeeds_baseball_slurm_common.sh"
#
# Resource defaults (baseball-tuned, not the generic 16G/24h repo rule):
#   Per-agent convergence array: ~4h observed on BBIP-20 → default 08:00:00, 8G
#   Paper BBIP wrapper pins 12:00:00 / agg 01:00:00 (proven margin)
#   Calibration agents: 12:00:00, 8G; aggregation: 00:30:00, 4G
# Submit scripts pass --time/--mem on the sbatch CLI; #SBATCH headers are fallbacks.

hjeeds_baseball_format_command() {
  local quoted=()
  local part
  for part in "$@"; do
    printf -v part "%q" "${part}"
    quoted+=("${part}")
  done
  printf "%s" "${quoted[*]}"
}

# Requires caller variables: python_bin (may be empty), conda_env.
hjeeds_baseball_resolve_python() {
  if [[ -n "${python_bin}" ]]; then
    echo "${python_bin}"
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    conda run -n "${conda_env}" which python 2>/dev/null && return 0
  fi
  if command -v module >/dev/null 2>&1; then
    module load miniforge3
    eval "$(conda shell.bash hook)"
    conda activate "${conda_env}"
    command -v python
    return 0
  fi
  command -v python3
}

# Cluster worker path: honor PYTHON_BIN, else module load miniforge3 + conda activate.
# Args: $1 = submit script name for error hints.
# Note: often called inside $(...); activation is only needed so `command -v python`
# resolves to the env binary — the absolute path remains valid in the parent shell.
hjeeds_baseball_resolve_cluster_python() {
  local submit_hint="${1:-submit_hjeeds_baseball_*.sh}"
  local resolved="${PYTHON_BIN:-}"
  local env_name="${CONDA_ENV:-skill-estimation}"
  if [[ -z "${resolved}" ]]; then
    if ! command -v module >/dev/null 2>&1; then
      echo "Error: module command not found and PYTHON_BIN is unset." >&2
      echo "Pass --python-bin to ${submit_hint}." >&2
      return 1
    fi
    module load miniforge3
    eval "$(conda shell.bash hook)"
    conda activate "${env_name}"
    resolved="$(command -v python)"
  fi
  if [[ -z "${resolved}" ]]; then
    echo "Error: could not resolve a python binary (CONDA_ENV=${env_name})." >&2
    echo "Pass --python-bin to ${submit_hint}." >&2
    return 1
  fi
  printf "%s" "${resolved}"
}

hjeeds_baseball_setup_matplotlib_env() {
  export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
  export MPLBACKEND=Agg
  if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    export MPLCONFIGDIR="${MPLCONFIGDIR:-${SLURM_TMPDIR}/matplotlib}"
  else
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}-${SLURM_JOB_ID:-manual}}"
  fi
  mkdir -p "${MPLCONFIGDIR}"
}

# Zip an output directory next to itself (or RESULTS_ZIP_PATH).
# Args: $1 = python binary, $2 = output_dir, $3 = log tag.
hjeeds_baseball_zip_results() {
  local python_bin="$1"
  local output_dir="$2"
  local log_tag="${3:-baseball}"
  local zip_path="${RESULTS_ZIP_PATH:-${output_dir%/}.zip}"
  "${python_bin}" - "${output_dir}" "${zip_path}" "${log_tag}" <<'PY'
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

source_dir = Path(sys.argv[1]).resolve()
zip_path = Path(sys.argv[2]).resolve()
log_tag = sys.argv[3]

if not source_dir.exists():
    raise FileNotFoundError(f"Cannot zip missing output directory: {source_dir}")

zip_path.parent.mkdir(parents=True, exist_ok=True)
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(source_dir.rglob("*")):
        if path.resolve() == zip_path:
            continue
        if path.is_file():
            archive.write(path, Path(source_dir.name) / path.relative_to(source_dir))

print(f"[{log_tag}] Zipped results to {zip_path}", flush=True)
PY
}

# Count agents in a roster JSON list.
# Args: $1 = python, $2 = roster path. Uses caller's script_dir for PYTHONPATH.
hjeeds_baseball_roster_agent_count() {
  local python_bin="$1"
  local roster_file="$2"
  PYTHONPATH="${script_dir}${PYTHONPATH:+:$PYTHONPATH}" \
  "${python_bin}" - "${roster_file}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
print(len(payload))
PY
}
