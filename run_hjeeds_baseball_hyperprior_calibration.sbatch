#!/bin/bash
# This file was written or edited by AI and still requires human review. Delete this comment when done.
#SBATCH --job-name=hjeeds-baseball-calib
#SBATCH --qos=normal
#SBATCH --time=02:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  export MPLCONFIGDIR="${MPLCONFIGDIR:-${SLURM_TMPDIR}/matplotlib}"
else
  export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}-${SLURM_JOB_ID:-manual}}"
fi
mkdir -p "${MPLCONFIGDIR}"

python_bin="${PYTHON_BIN:-}"
if [[ -z "${python_bin}" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    echo "Error: module command not found and PYTHON_BIN is unset." >&2
    echo "Pass --python-bin to submit_hjeeds_baseball_hyperprior_calibration.sh." >&2
    exit 1
  fi
  module load miniforge3
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-skill-estimation}"
  python_bin="$(command -v python)"
fi

echo "[baseball-calibrate] Using python: ${python_bin}" >&2

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

echo "[baseball-calibrate] Starting: ${python_bin} ${python_args[*]}" >&2
"${python_bin}" "${python_args[@]}"

if [[ "${AGGREGATE_RESULTS:-0}" == "1" ]]; then
  output_dir="${OUTPUT_DIR}"
  zip_path="${RESULTS_ZIP_PATH:-${output_dir%/}.zip}"
  "${python_bin}" - "${output_dir}" "${zip_path}" <<'PY'
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

source_dir = Path(sys.argv[1]).resolve()
zip_path = Path(sys.argv[2]).resolve()

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

print(f"[baseball-calibrate] Zipped results to {zip_path}", flush=True)
PY
fi
