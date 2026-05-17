# Processing Results

<!-- This file was written or edited by AI and still requires human review. Delete this comment when done -->

Use the processing scripts to summarize experiment outputs after runs complete.

## Commands
```bash
python Processing/processResultsDarts.py -domain 1d -resultsFolder Experiments/1d/Results/
python Processing/processResultsBaseball.py -resultsFolder Experiments/baseball/Testing/
```

## H-JEEDS outputs

H-JEEDS scripts write analysis-ready CSVs directly, so they usually do not require the older `Processing/` scripts.

Main hierarchical darts run:

- `HJEEDS/results/hierarchical_darts/agent_level_results.csv`
- `HJEEDS/results/hierarchical_darts/summary_by_bucket.csv`
- `HJEEDS/results/hierarchical_darts/summary_overall.csv`

Hyperprior robustness run:

- `prior_sensitivity_conditions.csv`
- `prior_sensitivity_agent_level_results.csv`
- `prior_sensitivity_summary_by_bucket.csv`
- `prior_sensitivity_summary_overall.csv`
- `prior_sensitivity_lowest_bucket_heatmap.png`

Agents-per-bucket run:

- `agents_per_bucket_sensitivity_runs.csv`
- `agents_per_bucket_sensitivity_scenarios.csv`
- `agents_per_bucket_sensitivity_agent_level_results.csv`
- `agents_per_bucket_sensitivity_summary_by_bucket.csv`
- `agents_per_bucket_sensitivity_summary_overall.csv`

The Slurm agents-per-bucket workflow also creates `OUTPUT_DIR.zip` after the aggregation task finishes, so the completed experiment folder is ready to copy off the cluster.
