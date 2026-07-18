#!/bin/bash
# Ablation: same BBIP paper roster/setup as submit_hjeeds_baseball_convergence_paper_bbip.sh,
# but FF21 calibrated centers with STRONG confidence widths (darts prior-sensitivity
# "strong" = default SDs x 1/3) instead of baseball low-confidence widths.
# Zero-arg: git pull, then ./submit_hjeeds_baseball_convergence_paper_bbip_high_confidence.sh

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
  --hyperprior-config HJEEDS/data/baseball_hyperpriors_2021_ff_high_confidence.json \
  --output-dir HJEEDS/results/baseball_convergence_paper_bbip20_ff21_HIGH_CONFIDENCE \
  --time 12:00:00 \
  --agg-time 01:00:00 \
  "$@"
