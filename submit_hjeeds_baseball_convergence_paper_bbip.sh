#!/bin/bash
# Paper correspondence: Main `subsec:baseball`; Supplement `app:baseball_sigma_gap`.
# Paper run: 20-agent processed-data walk/IP-proxy convergence (highest 10 + lowest 10).
# Pins the walltimes used for the paper walk/IP-proxy workflow:
#   --time 12:00:00 (per agent; ~4h observed)  --agg-time 01:00:00

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HJEEDS_REQUIRE_PAPER_CONFIG=1

exec "${script_dir}/submit_hjeeds_baseball_convergence_array.sh" \
  --seed default \
  --season-year 2021 \
  --pitch-types FF \
  --bbip-extremes 10 \
  --min-pitches-per-agent 100 \
  --convergence-ns 5,10,25,50,100 \
  --max-reference-pitches 100 \
  --hyperprior-preset baseball-literature-informed \
  --output-dir HJEEDS/results/baseball_convergence_paper_bbip20_literature_informed \
  --time 12:00:00 \
  --agg-time 01:00:00 \
  "$@"
