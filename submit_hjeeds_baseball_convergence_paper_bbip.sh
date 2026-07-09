#!/bin/bash
# This file was written or edited by AI and still requires human review. Delete this comment when done.
#
# Paper run: 20-agent BB/IP convergence (top 10 + bottom 10) via Slurm array.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${script_dir}/submit_hjeeds_baseball_convergence_array.sh" \
  --seed default \
  --season-year 2021 \
  --pitch-types FF \
  --bbip-extremes 10 \
  --min-pitches-per-agent 100 \
  --convergence-ns 5,10,25,50,100 \
  --max-reference-pitches 100 \
  --hyperprior-preset calibrated \
  --hyperprior-config HJEEDS/results/baseball_hyperprior_calib_2021_ff/suggested_hyperpriors.json \
  --output-dir HJEEDS/results/baseball_convergence_paper_bbip20_calibrated \
  --time 12:00:00 \
  --agg-time 01:00:00 \
  "$@"
