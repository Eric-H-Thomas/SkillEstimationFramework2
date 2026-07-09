#!/bin/bash
# This file was written or edited by AI and still requires human review. Delete this comment when done.
#
# Tonight smoke: 10-agent Phase 2 convergence with calibrated hyperpriors (single job).

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${script_dir}/submit_hjeeds_baseball_convergence.sh" \
  --seed default \
  --season-year 2021 \
  --pitch-types FF \
  --top-pitchers 10 \
  --min-pitches-per-agent 100 \
  --convergence-ns 5,10,25,50,100 \
  --max-reference-pitches 100 \
  --hyperprior-preset calibrated \
  --hyperprior-config HJEEDS/results/baseball_hyperprior_calib_2021_ff/suggested_hyperpriors.json \
  --output-dir HJEEDS/results/baseball_convergence_smoke10_calibrated \
  --time 18:00:00 \
  "$@"
