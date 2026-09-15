
# Unified H-JEEDS Paper Experiments

The unified runner `run_hjeeds_paper_experiments.py` launches the full current H-JEEDS experiment suite for the 1D darts paper. It keeps each ablation in its existing runner, writes all outputs under one paper-level root, tracks a manifest and status CSV, and creates a final zip archive for export.

## Experiment Suite

| Experiment | Scenarios | Purpose |
|---|---:|---|
| `baseline` | 1 | Compare independent JEEDS against H-JEEDS under the main uneven-data design |
| `hyperprior_robustness` | 60 | Stress the empirical-Bayes hyperpriors across bias, confidence, and misspecified prior components |
| `agents_per_bucket` | 5 | Vary population size under the default hyperpriors |
| `population_shape` | 3 | Vary only the true population shape at the default population size |
| `outlier_sensitivity` | 3 | Test the effect of replacing 0, 1, or 5 default agents with explicit skill-profile outliers |
| `anchor_availability` | 6 | Upgrade 0--25 of 25 context agents to high-data anchors while holding the 50-agent population fixed |
| `decision_model` | 4 | Vary only the true decision model while H-JEEDS retains its softmax likelihood |
| `true_correlation` | 5 | Vary only the true execution/decision skill correlation |
| `grid_resolution` | 3 | Check whether conclusions depend on the estimator grid resolution |
| `compound_stress` | 3 | Combine representative hyperprior, population-shape, decision-model, and correlation stressors |

The default suite contains 10 experiment families and 93 Slurm scenario tasks before aggregation and final zipping.

## Local Run

```bash
python3 run_hjeeds_paper_experiments.py \
  --mode local \
  --seed 12345 \
  --num-seeds 500 \
  --output-root HJEEDS/results/hjeeds_paper_experiments
```

Local mode runs all paper experiments sequentially in manifest order. Each experiment uses its normal all-in-one runner behavior, then the unified runner zips the full output root.

## Plot-Only Refresh

Use `plots-only` mode to regenerate figures from an already-computed paper output root without rerunning simulation or inference:

```bash
python3 run_hjeeds_paper_experiments.py \
  --mode plots-only \
  --seed 12345 \
  --num-seeds 500 \
  --output-root HJEEDS/results/hjeeds_paper_experiments
```

By default, paper plots show execution skill error and decision-skill percentage-point error. Add `--include-raw-rationality-error` to also include the raw log-decision-skill error panel/plots from earlier three-metric figures.

## Slurm Run

First activate an environment with the H-JEEDS scientific dependencies. The repository includes `environment-hjeeds.yml` for the minimal paper-runner stack.

```bash
module load miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate hjeeds-paper

python3 run_hjeeds_paper_experiments.py \
  --mode slurm \
  --seed 12345 \
  --num-seeds 500 \
  --output-root HJEEDS/results/hjeeds_paper_experiments \
  --python-bin "$(which python)" \
  --qos normal \
  --time 23:00:00 \
  --mem 16G
```

Slurm mode submits jobs directly with `sbatch --parsable`. Multi-scenario experiments run as arrays with one scenario per array task. Each array has a dependent aggregation job, and the final zip job depends on the baseline job plus all aggregation jobs.

## Output Layout

The default output root is:

```text
HJEEDS/results/hjeeds_paper_experiments
```

Inside that root, each experiment gets its own subdirectory:

```text
baseline/
hyperprior_robustness/
agents_per_bucket/
population_shape/
outlier_sensitivity/
anchor_availability/
decision_model/
true_correlation/
grid_resolution/
compound_stress/
```

The runner also writes:

- `paper_experiment_manifest.json`
- `paper_experiment_status.csv`
- `_runner/` for runner-owned cache/config files such as Matplotlib cache files
- `_slurm/` in Slurm mode for generated job scripts and log paths
- `<output-root>.zip` after the full experiment suite completes

## Design Logic

The baseline experiment establishes the main H-JEEDS versus independent JEEDS comparison. Hyperprior robustness checks whether low-data improvements depend on overly convenient empirical-Bayes priors. The agents-per-bucket study varies population size only under the default hyperpriors. The anchor study holds 25 one-sample evaluation agents and 25 context agents fixed, varying only how many context agents receive 25 observations.

Population-shape, outlier-contamination, decision-model, true-correlation, and grid-resolution studies each vary only their named factor while retaining the default five agents per bucket and other default settings. The compound-stress study is the sole deliberate combination: it jointly changes representative hyperpriors, population shapes, decision models, and correlations at the default population size. Its correlation prior/truth pairs are $(-0.5,-0.5)$ for default, $(-0.5,0)$ for moderate stress, and $(-0.9,+0.9)$ for strong wrong-sign stress.

The anchor study's canonical summary and plots always use the fixed first 25 one-observation evaluation agents. Files containing `all_agents_summary` preserve the full 50-agent population, including context agents, for diagnostic analysis.

After all scenarios finish, the unified runner validates exact scenario, seed, and agent coverage; estimator status; finite publication metrics; rationality bounds; rational-policy truth; and empirical-Bayes convergence. Plotting and archive creation stop if any check fails.

## Resume And Dry Run

Use `--dry-run` to print the planned local commands or Slurm submissions without launching work:

```bash
python3 run_hjeeds_paper_experiments.py \
  --mode slurm \
  --seed default \
  --num-seeds 1 \
  --output-root /private/tmp/hjeeds_paper_dry \
  --dry-run
```

If `--output-root` already exists and contains any files, the runner stops before launching non-dry-run `local` or `slurm` work. Pre-created empty directory trees are allowed, but any file under the output root is treated as existing output. This avoids accidentally mixing expensive results from different paper runs. `plots-only` and `zip-only` require an existing non-empty output root.
