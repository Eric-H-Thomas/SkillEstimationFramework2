#!/bin/bash
# This file was written or edited by AI and still requires human review. Delete this comment when done.
#SBATCH --job-name=hjeeds-anchors
#SBATCH --qos=normal
#SBATCH --time=23:00:00
#SBATCH --mem=16G
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

# Make sure Python can import the repository packages from the Slurm working directory
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Render plots in headless mode and keep Matplotlib cache files off read-only homes
export MPLBACKEND=Agg
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  export MPLCONFIGDIR="${MPLCONFIGDIR:-${SLURM_TMPDIR}/matplotlib}"
else
  export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}-${SLURM_JOB_ID:-manual}}"
fi
mkdir -p "${MPLCONFIGDIR}"

python_bin="${PYTHON_BIN:-$PWD/venv/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="${FALLBACK_PYTHON:-python3}"
fi

python_args=(
  -m HJEEDS.darts_anchor_availability_sensitivity
  --seed "${BASE_SEED:-default}"
  --num-seeds "${NUM_SEEDS:-500}"
  --output-dir "${OUTPUT_DIR:-HJEEDS/results/hierarchical_darts_anchor_availability_sensitivity}"
)

if [[ "${AGGREGATE_RESULTS:-0}" == "1" ]]; then
  python_args+=(--aggregate-results)
else
  export SCENARIO_INDEX="${SCENARIO_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
fi

"${python_bin}" "${python_args[@]}"

if [[ "${AGGREGATE_RESULTS:-0}" == "1" ]]; then
  output_dir="${OUTPUT_DIR:-HJEEDS/results/hierarchical_darts_anchor_availability_sensitivity}"
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

print(f"[anchor-availability] Zipped results to {zip_path}", flush=True)
PY
fi
