#!/bin/bash

# Player subsample-stability entrypoint on the cluster:
#   sbatch run_player_subsample_config.sbatch Data/Hockey/jobs/player_subsample_950160.json
#   sbatch run_player_subsample_config.sbatch Data/Hockey/jobs/player_subsample_950160.json fullpool-jeeds
#   sbatch run_player_subsample_config.sbatch Data/Hockey/jobs/player_subsample_950160.json fullpool-mcse
#   sbatch run_player_subsample_config.sbatch Data/Hockey/jobs/player_subsample_950160.json fullpool
#
# This script reads job array sizing + resource recommendations from the config,
# submits the worker array, and each worker runs run_player_subsample_config.py.
#
# Full-pool baselines are split out of the N-shot array. Optional second argument:
#   all             (default) both full-pool jobs + the subsample array
#   fullpool-jeeds  JEEDS on every pooled shot (48G, 48h)
#   fullpool-mcse   MCSE on every pooled shot (16G, 96h)
#   fullpool        both full-pool jobs, no subsample array
#   subsample       N-shot array only (16G, 24h)
#
# Offline only: every job reads cached parquet/npz, so no network is required.

#SBATCH --job-name=bh-subsample-bootstrap
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --output=slurm-subsample-bootstrap-%j.out

set -euo pipefail

if [[ $# -lt 1 && -z "${BH_CONFIG_PATH:-}" ]]; then
  echo "Usage: sbatch run_player_subsample_config.sbatch <config.json> [all|fullpool|fullpool-jeeds|fullpool-mcse|subsample]"
  exit 1
fi

CONFIG_PATH="${BH_CONFIG_PATH:-${1:-}}"
if [[ -z "$CONFIG_PATH" ]]; then
  echo "Config path is empty."
  exit 1
fi

cd ~/research/SkillEstimationFramework2

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH"
  exit 1
fi

if [[ -z "${BH_WORKER_MODE:-}" ]]; then
  SUBMIT_ONLY="${BH_SUBMIT_ONLY:-all}"
  if [[ $# -ge 2 ]]; then
    SUBMIT_ONLY="$2"
  fi

  PLAN="$(python3 - "$CONFIG_PATH" "$SUBMIT_ONLY" <<'PY'
import json
import sys

MODES = {
    "all": ("fullpool-jeeds", "fullpool-mcse", "subsample"),
    "fullpool": ("fullpool-jeeds", "fullpool-mcse"),
    "fullpool-jeeds": ("fullpool-jeeds",),
    "fullpool-mcse": ("fullpool-mcse",),
    "subsample": ("subsample",),
    "highmem": ("fullpool-jeeds",),
    "regular": ("subsample",),
    "baselines": ("fullpool-jeeds", "fullpool-mcse"),
}
JOB_NAMES = {
    "fullpool-jeeds": "bh-subsample-jeeds",
    "fullpool-mcse": "bh-subsample-mcse",
    "subsample": "bh-subsample",
}

path = sys.argv[1]
mode = str(sys.argv[2]).strip().lower() if len(sys.argv) > 2 else "all"
if mode not in MODES:
    raise SystemExit("Unknown submit mode %r; use: %s" % (mode, ", ".join(sorted(MODES))))

with open(path, "r", encoding="utf-8") as handle:
    cfg = json.load(handle)

cluster = cfg.get("cluster_plan", {})
reco = cluster.get("sbatch_recommendation", {}) or {}
jobs = [j for j in cluster.get("jobs", []) if j.get("eligible", True)]
max_concurrent = int(reco.get("max_concurrent", 100) or 100)
array_time = str(reco.get("time", "24:00:00"))
array_mem = str(reco.get("mem", "16G"))
jeeds_mem = str(reco.get("high_mem", "48G"))
jeeds_time = str(reco.get("jeeds_baseline_time", "48:00:00"))
mcse_time = str(reco.get("mcse_baseline_time", "96:00:00"))


def group_for(job):
    tagged = job.get("submit_group")
    if tagged:
        return str(tagged)
    if job.get("is_baseline") and str(job.get("estimator", "")) == "jeeds":
        return "fullpool-jeeds"
    if job.get("is_baseline") and str(job.get("estimator", "")) == "mcse":
        return "fullpool-mcse"
    return "subsample"


def resources(job, group):
    if group == "fullpool-jeeds":
        return str(job.get("time") or jeeds_time), str(job.get("mem") or jeeds_mem)
    if group == "fullpool-mcse":
        return str(job.get("time") or mcse_time), str(job.get("mem") or array_mem)
    return str(job.get("time") or array_time), str(job.get("mem") or array_mem)


def compress(ids):
    ids = sorted(ids)
    parts = []
    start = prev = ids[0]
    for value in ids[1:]:
        if value == prev + 1:
            prev = value
            continue
        parts.append("%s-%s" % (start, prev) if start != prev else str(start))
        start = prev = value
    parts.append("%s-%s" % (start, prev) if start != prev else str(start))
    return ",".join(parts)


grouped = {}
res = {}
for index, job in enumerate(jobs):
    group = group_for(job)
    grouped.setdefault(group, []).append(index + 1)
    res[group] = resources(job, group)

n_emitted = 0
for group in MODES[mode]:
    ids = grouped.get(group) or []
    if not ids:
        continue
    spec = compress(ids)
    conc = max_concurrent if max_concurrent <= len(ids) else len(ids)
    time, mem = res[group]
    print("%s%%%s %s %s %s %s" % (spec, conc, time, mem, group, JOB_NAMES[group]))
    n_emitted += 1

if n_emitted <= 0:
    raise SystemExit("No jobs to submit for mode %r in %s" % (mode, path))
PY
)"

  if [[ -z "$PLAN" ]]; then
    echo "No jobs to submit (mode=$SUBMIT_ONLY) for config: $CONFIG_PATH"
    exit 1
  fi

  echo "Submitting subsample workers for config: $CONFIG_PATH"
  echo "  mode: $SUBMIT_ONLY"

  while read -r ARRAY_SPEC SB_TIME SB_MEM LABEL JOB_NAME; do
    echo "  $LABEL  array=$ARRAY_SPEC  time=$SB_TIME  mem=$SB_MEM  name=$JOB_NAME"
    sbatch \
      --array="$ARRAY_SPEC" \
      --time="$SB_TIME" \
      --mem="$SB_MEM" \
      --job-name="$JOB_NAME" \
      --output="slurm-bh-subsample-%A_%a.out" \
      --export=ALL,BH_WORKER_MODE=1,BH_CONFIG_PATH="$CONFIG_PATH" \
      "$0"
  done <<< "$PLAN"

  exit 0
fi

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate skill-estimation

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python BlackhawksSkillEstimation/run_player_subsample_config.py \
  --config "$CONFIG_PATH" \
  --array-base 1
