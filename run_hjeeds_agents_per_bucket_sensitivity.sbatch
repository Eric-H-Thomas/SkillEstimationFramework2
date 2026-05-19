#!/bin/bash
# This file was written or edited by AI and still requires human review. Delete this comment when done.
#SBATCH --job-name=hjeeds-agents-bucket
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
  -m HJEEDS.darts_agents_per_bucket_sensitivity
  --num-seeds "${NUM_SEEDS:-250}"
  --seed "${BASE_SEED:-default}"
  --count-buckets "${COUNT_BUCKETS:-5,10,25,100,1000}"
  --agents-per-bucket-values "${AGENTS_PER_BUCKET_VALUES:-1,2,5,10,25}"
  --condition-preset "${CONDITION_PRESET:-representative}"
  --output-dir "${OUTPUT_DIR:-HJEEDS/results/hierarchical_darts_agents_per_bucket_sensitivity}"
  --delta "${DELTA:-0.1}"
  --num-sigma-grid "${NUM_SIGMA_GRID:-21}"
  --num-lambda-grid "${NUM_LAMBDA_GRID:-21}"
  --sigma-min "${SIGMA_MIN:-0.5}"
  --sigma-max "${SIGMA_MAX:-4.5}"
  --lambda-min "${LAMBDA_MIN:-1e-3}"
  --lambda-max "${LAMBDA_MAX:-1e2}"
  --min-success-regions "${MIN_SUCCESS_REGIONS:-2}"
  --max-success-regions "${MAX_SUCCESS_REGIONS:-6}"
  --min-region-width "${MIN_REGION_WIDTH:-0.25}"
)

if [[ "${AGGREGATE_RESULTS:-0}" == "1" ]]; then
  python_args+=(--aggregate-results)
else
  scenario_index="${SCENARIO_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
  python_args+=(--scenario-index "${scenario_index}")
fi

"${python_bin}" "${python_args[@]}"

if [[ "${AGGREGATE_RESULTS:-0}" == "1" ]]; then
  output_dir="${OUTPUT_DIR:-HJEEDS/results/hierarchical_darts_agents_per_bucket_sensitivity}"
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

print(f"[agents-per-bucket] Zipped results to {zip_path}", flush=True)
PY
fi
