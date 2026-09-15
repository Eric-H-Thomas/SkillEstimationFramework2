#!/bin/bash
# Paper correspondence: Main Figure `fig:two_d_baseline_results`.
#
# Regenerate the 2D-Darts main-paper figure after the execution-noise correction.
#
# Submits three chained Slurm jobs and returns immediately:
#   1. simulation/inference array  (default 100 tasks x 5 seeds = 500 seeds)
#   2. across-seed aggregation     (afterok on the array)
#   3. figure render               (afterok on the aggregation)
#
# Any existing cluster_0 result tree is renamed aside rather than overwritten, so
# the pre-correction results stay available for comparison.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir}"
worker_script="${script_dir}/run_hjeeds_2d_cluster_tests.sbatch"
cd "${repo_root}"

parts_per_group="100"
seeds_per_group="500"
base_seed_start="1000"
group_index="0"
array_time="04:00:00"
array_mem="16G"
agg_time="02:00:00"
agg_mem="8G"
figure_time="00:20:00"
figure_mem="4G"
conda_env="skill-estimation"
dry_run="0"
skip_preflight="0"

usage() {
  cat <<'USAGE'
Usage: regenerate_2d_paper_figure.sh [options]

Submits the corrected 2D-Darts regeneration as three chained Slurm jobs.

Options:
  --seeds-per-group N     Total seeds to run (default: 500, the published design).
  --parts-per-group N     Array tasks to split those seeds across (default: 100).
                          Must divide --seeds-per-group evenly.
  --base-seed-start N     First seed (default: 1000, matching published cluster_0).
  --array-time HH:MM:SS   Walltime per array task (default: 04:00:00).
  --array-mem MEM         Memory per array task (default: 16G).
  --conda-env NAME        Conda environment to activate (default: skill-estimation).
  --skip-preflight        Skip the correction checks (not recommended).
  --dry-run               Print what would be submitted and exit.
  -h, --help              Show this help.

Examples:
  # Published design, maximum parallelism (~10 min per task):
  regenerate_2d_paper_figure.sh

  # Reduced run if the queue is congested:
  regenerate_2d_paper_figure.sh --seeds-per-group 150 --parts-per-group 30
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds-per-group) seeds_per_group="$2"; shift 2 ;;
    --parts-per-group) parts_per_group="$2"; shift 2 ;;
    --base-seed-start) base_seed_start="$2"; shift 2 ;;
    --array-time) array_time="$2"; shift 2 ;;
    --array-mem) array_mem="$2"; shift 2 ;;
    --conda-env) conda_env="$2"; shift 2 ;;
    --skip-preflight) skip_preflight="1"; shift ;;
    --dry-run) dry_run="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Error: unrecognized option '$1'." >&2; usage >&2; exit 1 ;;
  esac
done

for value_name in parts_per_group seeds_per_group base_seed_start; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "Error: --${value_name//_/-} must be a positive integer. Received '${value}'." >&2
    exit 1
  fi
done
if (( seeds_per_group % parts_per_group != 0 )); then
  echo "Error: --seeds-per-group (${seeds_per_group}) must be divisible by --parts-per-group (${parts_per_group})." >&2
  exit 1
fi
if [[ ! -f "${worker_script}" ]]; then
  echo "Error: worker script not found at ${worker_script}." >&2
  exit 1
fi

seeds_per_part=$(( seeds_per_group / parts_per_group ))
array_spec="0-$(( parts_per_group - 1 ))"
group_dir="HJEEDS/results/2d_cluster_tests/cluster_${group_index}"
summary_csv="${group_dir}/summary_by_bucket.csv"

echo "=== 2D-Darts regeneration ==="
echo "  repo root:        ${repo_root}"
echo "  seeds:            ${seeds_per_group} (from ${base_seed_start})"
echo "  array tasks:      ${parts_per_group} x ${seeds_per_part} seeds  (array ${array_spec})"
echo "  results:          ${group_dir}"
echo

# --- Preflight: confirm the execution-noise correction is actually present -----
# Cheap static check first so this works on a login node without the conda env.
if [[ "${skip_preflight}" == "0" ]]; then
  echo "--- preflight ---"
  adapter="HJEEDS/environment_adapters.py"
  if ! grep -q "rng.normal(0.0, float(sigma), size=2)" "${adapter}"; then
    echo "Error: ${adapter} does not contain the corrected 2D sampler." >&2
    echo "       You are about to regenerate with the per-run fixed-noise bug. Pull the fix first." >&2
    exit 1
  fi
  # Ignore comment lines so the explanatory note about the legacy helper does not
  # register as a live call site.
  if grep -v '^[[:space:]]*#' "${adapter}" | grep -q "two_d_darts.sample_noisy_action"; then
    echo "Error: ${adapter} still delegates to two_d_darts.sample_noisy_action." >&2
    echo "       That helper reseeds from the generator's root entropy and returns identical noise." >&2
    exit 1
  fi
  echo "  [ok] corrected 2D execution-noise sampler present"

  # Then the real regression tests, if the environment can be activated here.
  if command -v module >/dev/null 2>&1 || command -v conda >/dev/null 2>&1; then
    if module load miniforge3 >/dev/null 2>&1 || true; then :; fi
    if command -v conda >/dev/null 2>&1; then
      eval "$(conda shell.bash hook)" 2>/dev/null || true
      if conda activate "${conda_env}" 2>/dev/null; then
        export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
        export MPLBACKEND=Agg
        # -p no:cacheprovider keeps the preflight from leaving a .pytest_cache
        # directory behind, which would show up as uninventoried repository files.
        if python3 -m pytest tests/test_two_d_environment_consistency.py -q -p no:cacheprovider; then
          echo "  [ok] 2D execution-noise regression tests pass"
        else
          echo "Error: 2D regression tests failed. Not submitting." >&2
          exit 1
        fi
        conda deactivate 2>/dev/null || true
      else
        echo "  [warn] could not activate '${conda_env}' here; skipping test run (static check passed)"
      fi
    fi
  else
    echo "  [warn] no module/conda on this node; skipping test run (static check passed)"
  fi
  echo
fi

# --- Preserve any pre-correction results --------------------------------------
if [[ -e "${group_dir}" ]]; then
  backup_dir="${group_dir}.pre_noise_fix_$(date +%Y%m%d_%H%M%S)"
  echo "--- preserving existing results ---"
  echo "  ${group_dir} -> ${backup_dir}"
  if [[ "${dry_run}" == "0" ]]; then
    mv "${group_dir}" "${backup_dir}"
  fi
  echo
fi

export_env="ALL,HJEEDS_REPO_ROOT=${repo_root},PARTS_PER_GROUP=${parts_per_group},GROUP_COUNT=1,SEEDS_PER_GROUP=${seeds_per_group},BASE_SEED_START=${base_seed_start}"

array_cmd=(
  sbatch
  --job-name=hjeeds-2d-regen
  --array="${array_spec}"
  --time="${array_time}"
  --mem="${array_mem}"
  --export="${export_env}"
  "${worker_script}"
)

figure_wrap="$(cat <<WRAP
set -euo pipefail
module load miniforge3
eval "\$(conda shell.bash hook)"
conda activate ${conda_env}
cd "${repo_root}"
export PYTHONPATH="\$PWD\${PYTHONPATH:+:\$PYTHONPATH}"
export MPLBACKEND=Agg
python3 -m HJEEDS.plot_main_paper_higher_dimensional \\
  --figures 2d \\
  --two-d-summary-csv "${summary_csv}" \\
  --output-dir figures
echo "Regenerated 2D figure from ${summary_csv}"
WRAP
)"

if [[ "${dry_run}" == "1" ]]; then
  echo "--- dry run: nothing submitted ---"
  echo "1) array:       ${array_cmd[*]}"
  echo "2) aggregation: sbatch --job-name=hjeeds-2d-agg --array=0 --dependency=afterok:<ARRAY_ID>"
  echo "                --time=${agg_time} --mem=${agg_mem} --export=${export_env},AGGREGATE_RESULTS=1 ${worker_script}"
  echo "3) figure:      sbatch --job-name=hjeeds-2d-figure --dependency=afterok:<AGG_ID>"
  echo "                --time=${figure_time} --mem=${figure_mem} --wrap=<render ${summary_csv}>"
  exit 0
fi

echo "--- submitting ---"
array_output="$("${array_cmd[@]}")"
echo "  ${array_output}"
array_job_id="$(awk '{print $4}' <<< "${array_output}")"
if [[ -z "${array_job_id}" ]]; then
  echo "Error: could not parse the array job id from: ${array_output}" >&2
  exit 1
fi

agg_output="$(sbatch \
  --job-name=hjeeds-2d-agg \
  --array=0 \
  --dependency="afterok:${array_job_id}" \
  --time="${agg_time}" \
  --mem="${agg_mem}" \
  --export="${export_env},AGGREGATE_RESULTS=1" \
  "${worker_script}")"
echo "  ${agg_output}"
agg_job_id="$(awk '{print $4}' <<< "${agg_output}")"
if [[ -z "${agg_job_id}" ]]; then
  echo "Error: could not parse the aggregation job id from: ${agg_output}" >&2
  exit 1
fi

figure_output="$(sbatch \
  --job-name=hjeeds-2d-figure \
  --dependency="afterok:${agg_job_id}" \
  --time="${figure_time}" \
  --mem="${figure_mem}" \
  --output="slurm-hjeeds-2d-figure-%j.out" \
  --wrap="${figure_wrap}")"
echo "  ${figure_output}"
figure_job_id="$(awk '{print $4}' <<< "${figure_output}")"

cat <<SUMMARY

=== submitted ===
  array (${parts_per_group} tasks)  job ${array_job_id}
  aggregation                       job ${agg_job_id}   (afterok:${array_job_id})
  figure render                     job ${figure_job_id}   (afterok:${agg_job_id})

Monitor:
  squeue -j ${array_job_id},${agg_job_id},${figure_job_id}
  tail -f slurm-hjeeds-2d-regen-${array_job_id}_0.out

When finished, expect:
  ${summary_csv}
  figures/10_two_d_error_by_count_bucket.{png,pdf,svg}

Sanity check the corrected figure: execution error should now fall clearly across
observation counts rather than plateauing, and H-JEEDS should sit below JEEDS at
every count. If the curves are still flat, the run picked up uncorrected code.
SUMMARY
