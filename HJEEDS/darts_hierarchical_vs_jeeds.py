# This file still requires human verification. Delete this comment when done.
"""Initial 1D darts hierarchical-vs-JEEDS experiments.

This module implements the first end-to-end synthetic experiment for comparing
independent JEEDS against a hierarchical empirical-Bayes extension. In
particular, this file:

1. Defines the configuration objects and command-line interface for the study.
2. Holds the data structures for true skills, simulated data, per-method
   estimates, and per-seed summaries.
3. Runs simulation, independent JEEDS inference, hierarchical inference,
   aggregation, and artifact writing.
4. Keeps a working ``--dry-run`` mode so we can validate parser/config wiring
   and inspect the intended workload before launching the numerical path.

The experiment compares:

- An independent JEEDS baseline that estimates each demonstrator separately
  with a uniform prior over the JEEDS skill grid.
- A hierarchical Bayesian alternative that learns a population-level prior over
  demonstrator skill and uses that prior to improve low-data estimates.

The main experimental twist is uneven observation counts: some demonstrators
will have only a handful of throws while others will have many. That is the
setting where the hierarchical model is expected to help most.

The implementation remains intentionally compact so later ablations can be
added without reorganizing the file.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


# Ensure the repository root is importable when this file is executed directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# This file acts as the public face of the package.  This entry module
# intentionally exports the main constants and helpers so companion scripts can
# import from one recognizable location.
from HJEEDS.aggregation import aggregate_results_across_seeds, summarize_seed_results
from HJEEDS.artifacts import _agent_result_to_row, plot_error_by_bucket, write_agent_level_csv, write_summary_csvs
from HJEEDS.config import (
    AGENT_LEVEL_CSV_HEADER,
    AGENT_LEVEL_FILENAME,
    DEFAULT_AGENTS_PER_BUCKET,
    DEFAULT_COUNT_BUCKETS,
    DEFAULT_DELTA,
    DEFAULT_HYPERPRIORS,
    DEFAULT_LAMBDA_GRID,
    DEFAULT_LAMBDA_MAX,
    DEFAULT_LAMBDA_MIN,
    DEFAULT_MAX_REGIONS,
    DEFAULT_MIN_REGION_WIDTH,
    DEFAULT_MIN_REGIONS,
    DEFAULT_NUM_AGENTS,
    DEFAULT_NUM_LAMBDA_GRID,
    DEFAULT_NUM_SIGMA_GRID,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SIGMA_GRID,
    DEFAULT_SIGMA_MAX,
    DEFAULT_SIGMA_MIN,
    DEFAULT_TRUE_POPULATION,
    ERROR_PLOT_FILENAME,
    SUMMARY_BY_BUCKET_CSV_HEADER,
    SUMMARY_BY_BUCKET_FILENAME,
    SUMMARY_OVERALL_CSV_HEADER,
    SUMMARY_OVERALL_FILENAME,
    build_config_from_args,
    parse_args,
    planned_output_paths,
    print_dry_run_summary,
)
from HJEEDS.estimation import (
    build_discrete_hierarchical_prior,
    fit_population_hyperparameters_map,
    run_hierarchical_estimator,
    run_independent_jeeds_baseline,
)
from HJEEDS.likelihood import compute_agent_log_likelihood_grid
from HJEEDS.models import (
    AgentDataset,
    AgentResult,
    AgentTruth,
    ExperimentConfig,
    HyperpriorConfig,
    MethodEstimate,
    SeedResult,
    TruePopulationConfig,
)
from HJEEDS.pipeline import run_single_seed
from HJEEDS.sampling import (
    assign_observation_counts,
    build_skill_grids,
    sample_reward_surface,
    sample_true_population_params,
    simulate_agent_dataset,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the hierarchical-vs-JEEDS experiment."""

    # Parse the CLI and immediately convert it into the package's validated
    # immutable config object.  Everything downstream operates on ``config``.
    args = parse_args(argv)
    config = build_config_from_args(args)

    # Dry-run is the safest way to confirm paths and workload before paying the
    # cost of simulation, likelihood evaluation, and optimizer calls.
    if config.dry_run:
        print_dry_run_summary(config)
        return 0

    seed_results: list[SeedResult] = []
    for seed_index, seed in enumerate(config.seed_values, start=1):
        # Each seed is an independent replicate of the same experiment design.
        print(f"[hier-darts] Running seed {seed_index}/{config.num_seeds}: {seed}", flush=True)
        seed_results.append(run_single_seed(config, seed))

    # Flatten the nested seed structure only after all per-seed work is done.
    # The artifact writers expect one list of agent rows plus the across-seed
    # summary tables.
    output_paths = planned_output_paths(config.output_dir)
    all_agent_results = [result for seed_result in seed_results for result in seed_result.agent_results]
    summary_by_bucket_rows, summary_overall_rows = aggregate_results_across_seeds(seed_results)

    # Keep CSV writing and plotting here in the entry module so the seed-level
    # pipeline stays focused on modeling rather than file I/O.
    write_agent_level_csv(output_paths["agent_level_csv"], all_agent_results)
    write_summary_csvs(config.output_dir, summary_by_bucket_rows, summary_overall_rows)
    plot_error_by_bucket(output_paths["error_plot"], summary_by_bucket_rows)
    print(f"[hier-darts] Wrote results to {config.output_dir.resolve()}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
