# JEEDS Aiming Sensitivity Experiments

These utilities probe how the Joint Estimator for Execution and Decision Skill (JEEDS) reacts when a darts agent intentionally aims away from the optimal location. The workflow starts with [`Testing/darts_aiming_jeeds_sensitivity.py`](darts_aiming_jeeds_sensitivity.py), which simulates noisy dart throws at every aim/skill combination in a configurable grid and then reuses the production JEEDS pipeline to recover execution skill estimates.

The surrounding scripts automate large batches of these experiments, partition work across Slurm clusters, and verify that the parallelized execution path matches a single-process run.

## Core experiment (`Testing/darts_aiming_jeeds_sensitivity.py`)

### Purpose
The script fixes a single randomly generated reward surface and evaluates how JEEDS' skill estimates change when the agent's true execution skill is held constant but the intended aiming location drifts away from the optimal choice. For each combination of aim and skill it:

1. Samples noisy dart landings using the standard 1-D darts environment.
2. Runs those observations through the production JEEDS estimator (via `JointMethodQRE`).
3. Records the estimated execution noise and compares it against ground truth.
4. Saves the full dataset and a scatter plot of absolute error versus aiming optimality.

The output directory receives both `jeeds_skill_vs_aim.csv` and `jeeds_skill_vs_aim.png`, while JEEDS timing logs are written under `Experiments/<jeeds-results-folder>` to keep the estimator happy.

### Key arguments
* `--num-samples`, `--num-aim-points`, `--num-true-skills` – control grid resolution and runtime.
* `--num-grid-skills`, `--num-planning-skills`, `--grid-min-skill`, `--grid-max-skill` – define the JEEDS hypothesis space.
* `--num-jobs`, `--job-index`, `--partial-subdir`, `--aggregate-results` – enable deterministic sharding across workers and later recombination of the CSV shards.

Run the experiment locally with something like:

```bash
python Testing/darts_aiming_jeeds_sensitivity.py --seed 7 --num-samples 150 --num-aim-points 61
```

Use `--aggregate-results` after the sharded jobs finish to rebuild the combined CSV and visualization.

## Slurm submission helper (`submit_darts_aiming_jeeds_sensitivity.py`)

Use this wrapper when launching the experiment on a Slurm cluster. It accepts standard scheduler parameters (job name, QoS, partition, memory, etc.), validates forwarded experiment flags against the core script, and automatically converts `--num-jobs` into a job array. When more than one job is requested it also submits a dependent aggregation job that runs once the array completes and combines the partial CSV outputs.

Example:

```bash
python submit_darts_aiming_jeeds_sensitivity.py --num-jobs 20 --time 04:00:00 --mem 24G -- --num-samples 200
```

Any arguments after `--` are passed through to `darts_aiming_jeeds_sensitivity.py`.

## Batch configuration (`run_darts_jeeds_sensitivity_exp.sbatch`)

This Slurm script encodes a high-resolution experiment configuration for repeated use. It is intended to be submitted either directly via `sbatch run_darts_jeeds_sensitivity_exp.sbatch` or through the submission helper. The script respects the `NUM_JOBS`, `JOB_INDEX`, and `AGGREGATE_RESULTS` environment variables so it can participate in job arrays while still supporting manual overrides.

To split the work across 20 jobs on a Slurm cluster, launch it as an array and forward the array size via `NUM_JOBS` so each task processes a distinct shard:

```bash
sbatch --array=0-19 --export=NUM_JOBS=20 run_darts_jeeds_sensitivity_exp.sbatch
```

Each array task reads `JOB_INDEX` from `SLURM_ARRAY_TASK_ID` and runs the experiment with `--num-jobs 20 --job-index <task-id>`, ensuring the 20 shards jointly cover the full workload.

## Local multi-seed automation (`run_darts_jeeds_sensitivity_multi_seed.sh`)

When running outside of Slurm you can still reproduce the high-resolution study across multiple random seeds with the convenience wrapper `run_darts_jeeds_sensitivity_multi_seed.sh`. The script mirrors the configuration baked into the `.sbatch` file, accepts an arbitrary list of seeds, and optionally shards each seed's workload across multiple sequential jobs. After all shards complete it automatically re-runs the experiment in aggregation mode so that the combined CSV and visualization are ready without any manual follow-up.

Launch the full pipeline for ten seeds, each split into twenty shards, with:

```bash
./run_darts_jeeds_sensitivity_multi_seed.sh --jobs-per-seed 20 \
  7 11 13 17 19 23 29 31 37 41
```

The script creates per-seed output directories under `Testing/results/high_res_seed_<seed>` that contain `jeeds_skill_vs_aim.csv` and `jeeds_skill_vs_aim.png`. With `--jobs-per-seed 1` (the default) each seed runs in a single process and writes the final artifacts directly without an aggregation pass.

## Parallel verification (`Testing/verify_parallelized_jeeds.py`)

This diagnostic script confirms that slicing the experiment across multiple jobs produces identical results to the single-job path. It runs a shortened configuration serially and in parallel, aggregates the shards, and diffs the resulting CSVs. Run it whenever you change the sharding logic or upgrade dependencies that might alter floating-point behaviour.
