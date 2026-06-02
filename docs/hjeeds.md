<!-- This file was written or edited by AI and still requires human review. Delete this comment when done. -->

# H-JEEDS Experiments

H-JEEDS is the hierarchical empirical-Bayes extension of JEEDS used for the 1D darts paper experiments. The maintained H-JEEDS entry points live under `HJEEDS/` and write outputs under `HJEEDS/results/` by default.

Run all commands from the repository root so Python can import the `HJEEDS` package.

All H-JEEDS experiment entry points require `--seed`. Use an integer seed or `--seed default`, which resolves to `12345`.

For the full paper suite runner, see [Unified H-JEEDS Paper Experiments](hjeeds-paper-experiments.md).

## Main Hierarchical Darts Experiment

Use `HJEEDS.darts_hierarchical_vs_jeeds` to compare independent JEEDS against the hierarchical empirical-Bayes estimator under uneven observation counts.

Dry run:

```bash
python3 -m HJEEDS.darts_hierarchical_vs_jeeds --seed default --num-seeds 1 --dry-run
```

Full-style run:

```bash
python3 -m HJEEDS.darts_hierarchical_vs_jeeds \
  --num-seeds 500 \
  --seed 12345 \
  --count-buckets 5,10,25,100,1000 \
  --agents-per-bucket 5 \
  --output-dir HJEEDS/results/hierarchical_darts
```

Primary outputs:

- `agent_level_results.csv`
- `summary_by_bucket.csv`
- `summary_overall.csv`
- `error_by_count_bucket.png`

## Hyperprior Robustness Study

Use `HJEEDS.darts_hierarchical_prior_sensitivity` for the standalone hyperprior robustness study. The default condition preset is `full_60`:

```text
4 focus areas x 5 bias levels x 3 confidence levels = 60 conditions
```

Focus areas:

- `average_skill`
- `population_spread`
- `correlation`
- `combined`

Bias levels:

- `strong_reverse_misspecification`
- `moderate_reverse_misspecification`
- `unbiased`
- `moderate_adverse_misspecification`
- `strong_adverse_misspecification`

Confidence levels:

- `weak`
- `default`
- `strong`

Dry run:

```bash
python3 -m HJEEDS.darts_hierarchical_prior_sensitivity --seed default --num-seeds 1 --dry-run
```

Run:

```bash
python3 -m HJEEDS.darts_hierarchical_prior_sensitivity \
  --num-seeds 500 \
  --seed 12345 \
  --count-buckets 5,10,25,100,1000 \
  --agents-per-bucket 5 \
  --output-dir HJEEDS/results/hierarchical_darts_prior_sensitivity
```

Primary combined outputs:

- `prior_sensitivity_conditions.csv`
- `prior_sensitivity_agent_level_results.csv`
- `prior_sensitivity_summary_by_bucket.csv`
- `prior_sensitivity_summary_overall.csv`
- `prior_sensitivity_lowest_bucket_heatmap.png`

Each condition also gets its own subfolder with ordinary H-JEEDS outputs.

## Agents-Per-Bucket Ablation

Use `HJEEDS.darts_agents_per_bucket_sensitivity` to study sensitivity to population size, expressed as the number of agents assigned to each observation-count bucket.

The default sweep crosses five agents-per-bucket values with three representative hyperprior conditions:

```text
5 agents-per-bucket values x 3 representative conditions = 15 scenarios
```

Default agents-per-bucket values:

- `1`
- `2`
- `5`
- `10`
- `25`

Representative conditions:

- `default`
- `moderate_combined_misspecification`
- `strong_combined_misspecification`

Dry run:

```bash
python3 -m HJEEDS.darts_agents_per_bucket_sensitivity --seed default --num-seeds 1 --dry-run
```

Run locally:

```bash
python3 -m HJEEDS.darts_agents_per_bucket_sensitivity \
  --num-seeds 500 \
  --seed 12345 \
  --count-buckets 5,10,25,100,1000 \
  --agents-per-bucket-values 1,2,5,10,25 \
  --output-dir HJEEDS/results/hierarchical_darts_agents_per_bucket_sensitivity
```

To explicitly cross every agents-per-bucket value with the full 60-condition robustness preset:

```bash
python3 -m HJEEDS.darts_agents_per_bucket_sensitivity \
  --num-seeds 500 \
  --seed 12345 \
  --condition-preset full_60 \
  --output-dir HJEEDS/results/hierarchical_darts_agents_per_bucket_sensitivity_full_60
```

That run has 300 scenarios with the default five agents-per-bucket values.

Primary root outputs:

- `agents_per_bucket_sensitivity_runs.csv`
- `agents_per_bucket_sensitivity_scenarios.csv`
- `agents_per_bucket_sensitivity_agent_level_results.csv`
- `agents_per_bucket_sensitivity_summary_by_bucket.csv`
- `agents_per_bucket_sensitivity_summary_overall.csv`

Each agents-per-bucket folder also contains prior-sensitivity combined CSVs and one subfolder per hyperprior condition.

## Population-Shape Ablation

Use `HJEEDS.darts_population_shape_sensitivity` to test how sensitive H-JEEDS is when the true simulator population does not match the estimator's unimodal Gaussian population model.

The default sweep crosses five agents-per-bucket values with four true population shapes:

```text
5 agents-per-bucket values x 4 population shapes = 20 scenarios
```

Population shapes:

- `default`
- `uniform`
- `bimodal`
- `outlier_heavy`

Dry run:

```bash
python3 -m HJEEDS.darts_population_shape_sensitivity --seed default --num-seeds 1 --dry-run
```

Run locally:

```bash
python3 -m HJEEDS.darts_population_shape_sensitivity \
  --seed 12345 \
  --num-seeds 500 \
  --output-dir HJEEDS/results/hierarchical_darts_population_shape_sensitivity
```

Primary root outputs:

- `population_shape_sensitivity_scenarios.csv`
- `population_shape_sensitivity_agent_level_results.csv`
- `population_shape_sensitivity_summary_by_bucket.csv`
- `population_shape_sensitivity_summary_overall.csv`
- `population_shape_lowest_bucket_abs_sigma_error.png`
- `population_shape_lowest_bucket_abs_log_lambda_error.png`
- `population_shape_lowest_bucket_abs_rationality_percent_error.png`

Use the Slurm submit helper to launch one array task per scenario and one dependent aggregation task:

```bash
./submit_hjeeds_population_shape_sensitivity.sh \
  --seed 12345 \
  --num-seeds 500 \
  --output-dir HJEEDS/results/hierarchical_darts_population_shape_sensitivity
```

After aggregation, the Slurm runner zips the top-level output folder for export. By default it writes `OUTPUT_DIR.zip`.

## Slurm 2D Cluster Wrapper (H-JEEDS)

Use the bash wrapper to submit both the array and aggregation jobs in one step (do not call `sbatch` directly).

Default grouped run (10 groups x 10 parts x 50 seeds):

```bash
bash submit_hjeeds_2d_cluster_tests.sh
```

Seed-per-task run (500 array tasks, 1 seed per task, grouped into `cluster_0`):

```bash
bash submit_hjeeds_2d_cluster_tests.sh \
  --group-count 1 \
  --parts-per-group 500 \
  --seeds-per-group 500
```

Aggregation-only rerun (no new array tasks):

```bash
bash submit_hjeeds_2d_cluster_tests.sh --agg-only
```

The aggregation job defaults to a shorter walltime and memory. Override as needed:

```bash
bash submit_hjeeds_2d_cluster_tests.sh --agg-time 02:00:00 --agg-mem 4G
```

## High-Data-Anchor Availability Ablation

Use `HJEEDS.darts_anchor_availability_sensitivity` to test whether H-JEEDS depends on having some high-data agents available. Each scenario keeps a fixed set of one-sample agents, then adds a varying number of 25-sample anchor agents.

The default sweep has six scenarios:

```text
25 one-sample agents + {0, 1, 2, 5, 10, 25} twenty-five-sample anchor agents
```

Dry run:

```bash
python3 -m HJEEDS.darts_anchor_availability_sensitivity --seed default --num-seeds 1 --dry-run
```

Run locally:

```bash
python3 -m HJEEDS.darts_anchor_availability_sensitivity \
  --seed 12345 \
  --num-seeds 500 \
  --output-dir HJEEDS/results/hierarchical_darts_anchor_availability_sensitivity
```

Primary root outputs:

- `anchor_availability_sensitivity_scenarios.csv`
- `anchor_availability_sensitivity_agent_level_results.csv`
- `anchor_availability_sensitivity_summary_by_bucket.csv`
- `anchor_availability_sensitivity_summary_overall.csv`
- `anchor_availability_low_data_abs_sigma_error.png`
- `anchor_availability_low_data_abs_log_lambda_error.png`
- `anchor_availability_low_data_abs_rationality_percent_error.png`

The root plots focus on the one-sample agents, which is the main evaluation target for this ablation. Each scenario folder also contains ordinary H-JEEDS outputs, including `error_by_count_bucket.png`.

Use the Slurm submit helper to launch one array task per anchor-count scenario and one dependent aggregation task:

```bash
./submit_hjeeds_anchor_availability_sensitivity.sh \
  --seed 12345 \
  --num-seeds 500 \
  --output-dir HJEEDS/results/hierarchical_darts_anchor_availability_sensitivity
```

After aggregation, the Slurm runner zips the top-level output folder for export. By default it writes `OUTPUT_DIR.zip`.

## Slurm Agents-Per-Bucket Workflow

Use the submit helper to launch one Slurm array task per scenario and one dependent aggregation task.

Default 15-scenario run:

```bash
./submit_hjeeds_agents_per_bucket_sensitivity.sh \
  --num-seeds 500 \
  --seed 12345 \
  --output-dir HJEEDS/results/hierarchical_darts_agents_per_bucket_sensitivity
```

Full 300-scenario run:

```bash
./submit_hjeeds_agents_per_bucket_sensitivity.sh \
  --num-seeds 500 \
  --seed 12345 \
  --condition-preset full_60 \
  --output-dir HJEEDS/results/hierarchical_darts_agents_per_bucket_sensitivity_full_60
```

Useful submit-helper options:

- `--dry-run` prints the `sbatch` commands without submitting
- `--python-bin PATH` selects the Python executable on the cluster
- `--job-name NAME`, `--qos QOS`, `--partition PARTITION`, `--account ACCOUNT`, `--time HH:MM:SS`, and `--mem MEM` tune Slurm resources

After aggregation, the Slurm runner zips the top-level output folder for export. By default it writes:

```text
OUTPUT_DIR.zip
```

Override the zip destination by setting `RESULTS_ZIP_PATH` in the environment used by the aggregation task.

## Visual Aids

The maintained visual-aid scripts live under `HJEEDS/VisualAids/`.

Generate the population and mean-prior log-skill visuals:

```bash
python3 HJEEDS/VisualAids/generate_log_skill_visuals.py
```

Generate hyperprior robustness condition overlays:

```bash
python3 HJEEDS/VisualAids/generate_bias_confidence_visuals.py
```

The current overlay script writes to:

```text
HJEEDS/VisualAids/HyperpriorRobustnessVisuals/
```

The older `HJEEDS/VisualAids/BiasConfidenceVisuals/` folder contains generated artifacts from the retired 3x3 study and is not the current source of truth.

## Quick Checks

Before launching a large run, use:

```bash
python3 -m compileall -q HJEEDS
python3 -m HJEEDS.darts_hierarchical_vs_jeeds --seed default --num-seeds 1 --dry-run
python3 -m HJEEDS.darts_hierarchical_prior_sensitivity --seed default --num-seeds 1 --dry-run
python3 -m HJEEDS.darts_agents_per_bucket_sensitivity --seed default --num-seeds 1 --dry-run
python3 -m HJEEDS.darts_agents_per_bucket_sensitivity --seed default --num-seeds 1 --condition-preset full_60 --dry-run
python3 -m HJEEDS.darts_population_shape_sensitivity --seed default --num-seeds 1 --dry-run
bash -n run_hjeeds_agents_per_bucket_sensitivity.sbatch
bash -n submit_hjeeds_agents_per_bucket_sensitivity.sh
bash -n run_hjeeds_population_shape_sensitivity.sbatch
bash -n submit_hjeeds_population_shape_sensitivity.sh
```
