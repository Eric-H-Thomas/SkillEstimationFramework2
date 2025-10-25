#!/bin/bash
#SBATCH --job-name=darts-jeeds-sensitivity
#SBATCH --qos=normal
#SBATCH --time=23:00:00
#SBATCH --mem=8G
#SBATCH --chdir=/home/eht23/research/SkillEstimationFramework2
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Make sure Python can import your packages
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# Headless matplotlib
export MPLBACKEND=Agg

# Run as a module so imports behave
$PWD/venv/bin/python -m Testing.darts_aiming_jeeds_sensitivity \
  --seed 7 \
  --num-samples 100 \
  --num-aim-points 50 \
  --num-true-skills 40 \
  --min-true-skill 0.1 \
  --max-true-skill 3.0 \
  --num-grid-skills 200 \
  --grid-min-skill 0.05 \
  --grid-max-skill 3.5 \
  --num-planning-skills 100 \
  --delta 5e-3 \
  --output-dir Testing/results/high_res
