<!-- This file was written or edited by AI and still requires human review. Delete this comment when done. -->

# Additional Experiments

The repository includes scripts that study how the JEEDS estimator reacts to suboptimal aiming in the darts domain.

- Supports sharded local runs, Slurm submissions, and a verification harness for the parallel path.
- See [Testing/jeeds_aiming_sensitivity.md](../Testing/jeeds_aiming_sensitivity.md) for the full workflow and usage examples.

## H-JEEDS agents-per-bucket ablation

Use `HJEEDS/darts_agents_per_bucket_sensitivity.py` to repeat the full 3x3
hyperprior-sensitivity analysis for several agents-per-bucket settings. The
default sweep runs:

- agents per bucket: `1, 2, 5, 10, 25`
- observation-count buckets: `5, 10, 25, 100, 1000`
- prior conditions: weak/default/strong confidence crossed with
  unbiased/slightly biased/significantly biased hyperprior centers

Inspect the workload without launching inference:

```bash
python3 -m HJEEDS.darts_agents_per_bucket_sensitivity --num-seeds 250 --dry-run
```

Run the full May 14 ablation:

```bash
python3 -m HJEEDS.darts_agents_per_bucket_sensitivity --num-seeds 250
```

Submit the same sweep to Slurm as 45 independent scenario tasks plus one final
aggregation task:

```bash
./submit_hjeeds_agents_per_bucket_sensitivity.sh --num-seeds 250
```

Preview the Slurm submission without calling `sbatch`:

```bash
./submit_hjeeds_agents_per_bucket_sensitivity.sh --num-seeds 250 --dry-run
```

The Slurm workflow uses `run_hjeeds_agents_per_bucket_sensitivity.sbatch` under
the hood. The scenario array runs `--scenario-index 0` through
`--scenario-index 44`, then the dependent aggregation task runs
`--aggregate-results` to collect the scenario folders into combined CSVs. If the
cluster needs a specific Python executable, pass it through the submit helper:

```bash
./submit_hjeeds_agents_per_bucket_sensitivity.sh \
  --num-seeds 250 \
  --python-bin /path/to/venv/bin/python
```

The root output directory defaults to
`HJEEDS/results/hierarchical_darts_agents_per_bucket_sensitivity`. It contains
combined CSVs across the whole sweep, plus one subdirectory per population-size
condition:

```text
agents_per_bucket_001/
  weak__unbiased/
    error_by_count_bucket.png
    agent_level_results.csv
    summary_by_bucket.csv
    summary_overall.csv
  ...
agents_per_bucket_002/
...
agents_per_bucket_025/
```

The 45 scenario-level plots live at
`agents_per_bucket_*/<prior_condition>/error_by_count_bucket.png`. The root
`agents_per_bucket_sensitivity_scenarios.csv` file lists every scenario and its
plot path.
