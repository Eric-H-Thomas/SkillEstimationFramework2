#!/bin/bash

# MCSE parameter-sweep entrypoint on the cluster:
#   sbatch run_mcse_param_sweep_config.sbatch Data/Hockey/jobs/mcse_param_sweep_stage1.json
#   sbatch run_mcse_param_sweep_config.sbatch Data/Hockey/jobs/mcse_param_sweep_stage2.json
#   sbatch run_mcse_param_sweep_config.sbatch Data/Hockey/jobs/mcse_param_sweep_stage2.json n1000
#
# This script reads the job list from the config, submits one worker array per
# resource group, and each worker runs run_mcse_param_sweep_config.py.
#
# Optional second argument:
#   all       (default) every group that has jobs
#   standard  24h / 16G jobs (stage 1, and the stage-2 jobs that are not 1000 particles)
#   n1000     1000-particle jobs (48h / 16G)
#
# Offline only: every job reads cached parquet/npz, so no network is required.

#SBATCH --job-name=bh-mcse-sweep-bootstrap
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --output=slurm-mcse-sweep-bootstrap-%j.out

set -euo pipefail

if [[ $# -lt 1 && -z "${BH_CONFIG_PATH:-}" ]]; then
  echo "Usage: sbatch run_mcse_param_sweep_config.sbatch <config.json> [all|standard|n1000]"
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
    "all": ("standard", "n1000"),
    "standard": ("standard",),
    "n1000": ("n1000",),
}
JOB_NAMES = {
    "standard": "bh-mcse-sweep",
    "n1000": "bh-mcse-sweep-n1000",
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
n1000_time = str(reco.get("n1000_time", "48:00:00"))
n1000_mem = str(reco.get("n1000_mem", "16G"))


def group_for(job):
    tagged = job.get("submit_group")
    if tagged:
        return str(tagged)
    if int(job.get("num_particles", 0)) >= 1000:
        return "n1000"
    return "standard"


def resources(job, group):
    if group == "n1000":
        return str(job.get("time") or n1000_time), str(job.get("mem") or n1000_mem)
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

  echo "Submitting MCSE sweep workers for config: $CONFIG_PATH"
  echo "  mode: $SUBMIT_ONLY"

  while read -r ARRAY_SPEC SB_TIME SB_MEM LABEL JOB_NAME; do
    echo "  $LABEL  array=$ARRAY_SPEC  time=$SB_TIME  mem=$SB_MEM  name=$JOB_NAME"
    sbatch \
      --array="$ARRAY_SPEC" \
      --time="$SB_TIME" \
      --mem="$SB_MEM" \
      --job-name="$JOB_NAME" \
      --output="slurm-bh-mcse-sweep-%A_%a.out" \
      --export=ALL,BH_WORKER_MODE=1,BH_CONFIG_PATH="$CONFIG_PATH" \
      "$0"
  done <<< "$PLAN"

  exit 0
fi

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate skill-estimation

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python BlackhawksSkillEstimation/run_mcse_param_sweep_config.py \
  --config "$CONFIG_PATH" \
  --array-base 1
