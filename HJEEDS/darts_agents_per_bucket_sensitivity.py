# This file has been fully edited by a human researcher as of 05/25/26 at 11:47 AM MDT.
"""Run agents-per-bucket ablations crossed with hyperprior sensitivity.

This script repeats selected hyperprior-robustness conditions for several
population sizes, where population size is controlled by the number of agents
assigned to each observation-count bucket.

Default sweep:

- agents per bucket: 1, 2, 5, 10, 25
- count buckets: 5, 10, 25, 100, 1000 observations
- prior conditions: default, moderate combined misspecification, strong
  combined misspecification

The output tree intentionally mirrors the existing prior-sensitivity runner so
each scenario has an ordinary experiment folder with CSVs and an
``error_by_count_bucket.png`` plot, while the root folder also contains
combined CSVs for spreadsheet/notebook analysis.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


# Ensure the repository root is importable when this file is executed directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HJEEDS import darts_hierarchical_prior_sensitivity as prior_sensitivity
from HJEEDS import darts_hierarchical_vs_jeeds as base_experiment


DEFAULT_OUTPUT_DIR = Path("HJEEDS/results/hierarchical_darts_agents_per_bucket_sensitivity")
DEFAULT_AGENTS_PER_BUCKET_VALUES = (1, 2, 5, 10, 25)

AGENTS_PER_BUCKET_RUNS_FILENAME = "agents_per_bucket_sensitivity_runs.csv"
AGENTS_PER_BUCKET_SCENARIOS_FILENAME = "agents_per_bucket_sensitivity_scenarios.csv"
COMBINED_AGENT_LEVEL_FILENAME = "agents_per_bucket_sensitivity_agent_level_results.csv"
COMBINED_SUMMARY_BY_BUCKET_FILENAME = "agents_per_bucket_sensitivity_summary_by_bucket.csv"
COMBINED_SUMMARY_OVERALL_FILENAME = "agents_per_bucket_sensitivity_summary_overall.csv"

AGENTS_PER_BUCKET_METADATA_HEADER = [
    "agents_per_bucket_slug",
    "agents_per_bucket",
    "num_agents",
    "count_buckets",
    "agents_per_bucket_output_dir",
]

SCENARIO_METADATA_HEADER = [
    "scenario_slug",
    "scenario_output_dir",
    "scenario_error_plot",
]


@dataclass(frozen=True)
class AgentsPerBucketScenario:
    """One concrete agents-per-bucket x hyperprior condition scenario."""

    scenario_index: int
    config: base_experiment.ExperimentConfig
    condition: prior_sensitivity.PriorSensitivityCondition

    @property
    def condition_slug(self) -> str:
        """Return the hyperprior condition slug for this scenario."""

        return self.condition.condition_slug

    @property
    def scenario_slug(self) -> str:
        """Return a stable combined scenario identifier."""

        return f"{_agents_per_bucket_slug(self.config.agents_per_bucket)}__{self.condition_slug}"

    @property
    def scenario_output_dir(self) -> Path:
        """Return the directory where this scenario's normal artifacts live."""

        return self.config.output_dir / self.condition_slug


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for the agents-per-bucket ablation."""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--seed",
        type=base_experiment.parse_seed_argument,
        required=True,
        help="Base seed used to derive per-run seeds. Use 'default' for 12345.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=base_experiment.DEFAULT_NUM_SEEDS,
        help=f"Number of random seeds to run for each scenario (default: {base_experiment.DEFAULT_NUM_SEEDS}).",
    )
    parser.add_argument(
        "--agents-per-bucket-values",
        type=str,
        default=",".join(str(value) for value in DEFAULT_AGENTS_PER_BUCKET_VALUES),
        help="Comma-separated agents-per-bucket values to sweep.",
    )
    parser.add_argument(
        "--count-buckets",
        type=str,
        default=",".join(str(bucket) for bucket in prior_sensitivity.DEFAULT_COUNT_BUCKETS),
        help="Comma-separated observation-count buckets assigned across demonstrators.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=base_experiment.DEFAULT_DELTA,
        help="Target-grid resolution used by the darts environment and likelihood code.",
    )
    parser.add_argument(
        "--num-sigma-grid",
        type=int,
        default=base_experiment.DEFAULT_NUM_SIGMA_GRID,
        help="Number of execution-skill hypotheses on the JEEDS grid.",
    )
    parser.add_argument(
        "--num-lambda-grid",
        type=int,
        default=base_experiment.DEFAULT_NUM_LAMBDA_GRID,
        help="Number of decision-skill hypotheses on the JEEDS grid.",
    )
    parser.add_argument("--sigma-min", type=float, default=base_experiment.DEFAULT_SIGMA_MIN)
    parser.add_argument("--sigma-max", type=float, default=base_experiment.DEFAULT_SIGMA_MAX)
    parser.add_argument("--lambda-min", type=float, default=base_experiment.DEFAULT_LAMBDA_MIN)
    parser.add_argument("--lambda-max", type=float, default=base_experiment.DEFAULT_LAMBDA_MAX)
    parser.add_argument(
        "--min-success-regions",
        "--min-regions",
        dest="min_success_regions",
        type=int,
        default=base_experiment.DEFAULT_MIN_SUCCESS_REGIONS,
    )
    parser.add_argument(
        "--max-success-regions",
        "--max-regions",
        dest="max_success_regions",
        type=int,
        default=base_experiment.DEFAULT_MAX_SUCCESS_REGIONS,
    )
    parser.add_argument("--min-region-width", type=float, default=base_experiment.DEFAULT_MIN_REGION_WIDTH)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Root output directory for the agents-per-bucket sweep.",
    )
    parser.add_argument(
        "--condition-preset",
        choices=prior_sensitivity.CONDITION_PRESETS,
        default=prior_sensitivity.CONDITION_PRESET_REPRESENTATIVE,
        help=(
            "Hyperprior condition set to cross with agents-per-bucket values. "
            "The default representative preset gives 15 scenarios; full_60 gives 300."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the agents-per-bucket x hyperprior workload and stop before simulation/inference.",
    )
    parser.add_argument(
        "--scenario-index",
        type=int,
        default=None,
        help=(
            "Run only one 0-based scenario from the agents-per-bucket x prior-condition grid. "
            "Used by the Slurm array helper."
        ),
    )
    parser.add_argument(
        "--aggregate-results",
        action="store_true",
        help=(
            "Collect already-computed scenario folders into per-agents-per-bucket and root combined CSVs. "
            "Used by the final Slurm dependency task."
        ),
    )

    return parser.parse_args(argv)


def _parse_positive_ints(raw_value: str, *, field_name: str) -> tuple[int, ...]:
    """Parse a comma-separated list of positive integers."""

    pieces = [piece.strip() for piece in raw_value.split(",") if piece.strip()]
    if not pieces:
        raise ValueError(f"At least one {field_name} value must be provided.")

    values = tuple(int(piece) for piece in pieces)
    if any(value <= 0 for value in values):
        raise ValueError(f"{field_name} values must all be positive integers. Received: {values}")
    return values


def _agents_per_bucket_slug(agents_per_bucket: int) -> str:
    """Return a stable folder slug for one agents-per-bucket value."""

    return f"agents_per_bucket_{agents_per_bucket:03d}"


def _count_bucket_label(count_buckets: Sequence[int]) -> str:
    """Return a compact label for CSV provenance."""

    return ",".join(str(bucket) for bucket in count_buckets)


def _seed_values_label(seed_values: Sequence[int]) -> str:
    """Return a readable seed description for dry-run output."""

    if len(seed_values) <= 10:
        return str(tuple(seed_values))
    return f"{len(seed_values)} seeds ({seed_values[0]} through {seed_values[-1]})"


def scenario_index_from_environment() -> int | None:
    """Return a scenario index supplied by a Slurm-style environment variable."""

    raw_value = os.environ.get("SCENARIO_INDEX") or os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw_value is None or raw_value == "":
        return None

    try:
        scenario_index = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            "SCENARIO_INDEX or SLURM_ARRAY_TASK_ID must be an integer. "
            f"Received: {raw_value}."
        ) from exc
    if scenario_index < 0:
        raise ValueError(f"Scenario index must be nonnegative. Received: {scenario_index}.")
    return scenario_index


def build_config_for_agents_per_bucket(
    args: argparse.Namespace,
    agents_per_bucket: int,
) -> base_experiment.ExperimentConfig:
    """Build one prior-sensitivity config for a specific population size."""

    count_buckets = _parse_positive_ints(args.count_buckets, field_name="count_bucket")
    output_dir = Path(args.output_dir) / _agents_per_bucket_slug(agents_per_bucket)

    base_args = argparse.Namespace(
        seed=args.seed,
        num_seeds=args.num_seeds,
        count_buckets=args.count_buckets,
        agents_per_bucket=agents_per_bucket,
        num_agents=len(count_buckets) * agents_per_bucket,
        delta=args.delta,
        num_sigma_grid=args.num_sigma_grid,
        num_lambda_grid=args.num_lambda_grid,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        output_dir=str(output_dir),
        dry_run=args.dry_run,
        min_success_regions=args.min_success_regions,
        max_success_regions=args.max_success_regions,
        min_region_width=args.min_region_width,
    )
    return prior_sensitivity.build_base_config_from_args(base_args)


def agents_per_bucket_metadata_row(config: base_experiment.ExperimentConfig) -> dict[str, Any]:
    """Return metadata shared by all conditions for one agents-per-bucket value."""

    slug = _agents_per_bucket_slug(config.agents_per_bucket)
    return {
        "agents_per_bucket_slug": slug,
        "agents_per_bucket": config.agents_per_bucket,
        "num_agents": config.num_agents,
        "count_buckets": _count_bucket_label(config.count_buckets),
        "agents_per_bucket_output_dir": str(config.output_dir),
    }


def scenario_metadata_row(
    config: base_experiment.ExperimentConfig,
    condition_slug: str,
) -> dict[str, Any]:
    """Return path metadata for one concrete scenario."""

    agents_slug = _agents_per_bucket_slug(config.agents_per_bucket)
    scenario_slug = f"{agents_slug}__{condition_slug}"
    scenario_output_dir = config.output_dir / condition_slug
    return {
        "scenario_slug": scenario_slug,
        "scenario_output_dir": str(scenario_output_dir),
        "scenario_error_plot": str(scenario_output_dir / base_experiment.ERROR_PLOT_FILENAME),
    }


def build_scenarios(
    configs: Sequence[base_experiment.ExperimentConfig],
    conditions: Sequence[prior_sensitivity.PriorSensitivityCondition],
) -> tuple[AgentsPerBucketScenario, ...]:
    """Return the flat scenario list used by local runs and Slurm arrays."""

    scenarios: list[AgentsPerBucketScenario] = []
    for config in configs:
        for condition in conditions:
            scenarios.append(
                AgentsPerBucketScenario(
                    scenario_index=len(scenarios),
                    config=config,
                    condition=condition,
                )
            )
    return tuple(scenarios)


def _write_dict_rows(output_path: Path, header: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    """Write dictionaries with a fixed header, leaving missing fields blank."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in header})


def _read_dict_rows(input_path: Path, scenario_slug: str) -> list[dict[str, Any]]:
    """Read one scenario CSV and fail clearly if the expected artifact is missing."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing expected artifact for scenario {scenario_slug}: {input_path}"
        )

    with input_path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _prefix_result_row(
    row: dict[str, Any],
    config: base_experiment.ExperimentConfig,
) -> dict[str, Any]:
    """Add agents-per-bucket and scenario provenance to a combined result row."""

    condition_slug = str(row["condition_slug"])
    return {
        **agents_per_bucket_metadata_row(config),
        **scenario_metadata_row(config, condition_slug),
        **row,
    }


def _condition_metadata_row(
    config: base_experiment.ExperimentConfig,
    condition: prior_sensitivity.PriorSensitivityCondition,
) -> dict[str, Any]:
    """Return the concrete hyperprior metadata for one condition under one config."""

    hyperpriors = prior_sensitivity.build_condition_hyperpriors(config.hyperpriors, condition)
    return prior_sensitivity.condition_metadata_row(condition, hyperpriors)


def run_single_scenario(scenario: AgentsPerBucketScenario) -> None:
    """Run one agents-per-bucket x hyperprior scenario."""

    print(
        "[agents-per-bucket] "
        f"Running scenario {scenario.scenario_index}: {scenario.scenario_slug} "
        f"({scenario.config.num_agents} agents/seed)",
        flush=True,
    )
    prior_sensitivity.run_condition(scenario.config, scenario.condition)
    print(
        "[agents-per-bucket] "
        f"Wrote scenario results to {scenario.scenario_output_dir.resolve()}",
        flush=True,
    )


def aggregate_existing_results(
    configs: Sequence[base_experiment.ExperimentConfig],
    conditions: Sequence[prior_sensitivity.PriorSensitivityCondition],
    output_dir: Path,
) -> None:
    """Collect precomputed scenario folders into the combined sweep artifacts."""

    # Root-level rows--written to the top-level output direction for the whole sweep
    run_rows: list[dict[str, Any]] = [] # One row per agents-per-bucket setting
    scenario_rows: list[dict[str, Any]] = [] # One row per agents-per-bucket x hyperprior scenario
    all_agent_rows: list[dict[str, Any]] = [] # Agent-level results from every scenario
    all_bucket_rows: list[dict[str, Any]] = [] # Bucket summary rows from every scenario
    all_overall_rows: list[dict[str, Any]] = [] # Overall summary rows from every scenario

    # Loop over each population size (agents-per-bucket) setting, e.g. 1, 2, 5, 10, 25
    for config in configs:

        # Add agents-per-bucket-specific metadata shared by all hyperprior conditions to the root-level run rows
        run_metadata = agents_per_bucket_metadata_row(config)
        run_rows.append(run_metadata)

        # Per-population-size rows
        per_agents_condition_rows: list[dict[str, Any]] = [] # The concrete hyperprior settings used here
        per_agents_agent_rows: list[dict[str, Any]] = [] # Agent-level result rows for this agents-per-bucket value
        per_agents_bucket_rows: list[dict[str, Any]] = [] # Bucket summary rows for this agents-per-bucket value
        per_agents_overall_rows: list[dict[str, Any]] = [] # Overall summary rows for this agents-per-bucket value

        # Loop over the hyperprior sensitivity grid for the current population size
        for condition in conditions:

            # Get numeric hyperprior metadata for the condition and paths/labels/slug for the scenario
            condition_metadata = _condition_metadata_row(config, condition)
            scenario_metadata = scenario_metadata_row(config, condition.condition_slug)
            scenario_slug = str(scenario_metadata["scenario_slug"])

            # Compute the standard artifact paths for this scenario
            scenario_output_dir = Path(str(scenario_metadata["scenario_output_dir"]))
            output_paths = base_experiment.planned_output_paths(scenario_output_dir)

            # Read the agent-level, bucket-level, and overal summary CSVs produced by the scenario task
            agent_rows = _read_dict_rows(output_paths["agent_level_csv"], scenario_slug)
            bucket_rows = _read_dict_rows(output_paths["summary_by_bucket_csv"], scenario_slug)
            overall_rows = _read_dict_rows(output_paths["summary_overall_csv"], scenario_slug)

            # Save this condition's hyperprior metadata for the per-population-size conditions CSV
            per_agents_condition_rows.append(condition_metadata)
            # Save this scenario's full metadata for the root scenario index CSV
            scenario_rows.append({**run_metadata, **scenario_metadata, **condition_metadata})

            # Prefix every agent-level, bucket-level, and overall row with the hyperprior metadata for per-population-size CSVs
            condition_agent_rows = [{**condition_metadata, **row} for row in agent_rows]
            condition_bucket_rows = [{**condition_metadata, **row} for row in bucket_rows]
            condition_overall_rows = [{**condition_metadata, **row} for row in overall_rows]

            # Add this condition's agent, bucket, and overall rows to the per-population-size combined table
            per_agents_agent_rows.extend(condition_agent_rows)
            per_agents_bucket_rows.extend(condition_bucket_rows)
            per_agents_overall_rows.extend(condition_overall_rows)

            # Add globally unique provenance to each agent-level, bucket-level, and overall row, and append
            # it to the root table
            all_agent_rows.extend(
                {**run_metadata, **scenario_metadata, **condition_metadata, **row}
                for row in agent_rows
            )
            all_bucket_rows.extend(
                {**run_metadata, **scenario_metadata, **condition_metadata, **row}
                for row in bucket_rows
            )
            all_overall_rows.extend(
                {**run_metadata, **scenario_metadata, **condition_metadata, **row}
                for row in overall_rows
            )
        # End of loop over the hyperprior sensitivity grid for the current population size

        # Reuse the prior-sensitivity metadata header for per-population-size combined CSVs
        condition_columns = prior_sensitivity.CONDITION_METADATA_HEADER
        # Write the hyperprior condition metadata rows for this agents-per-bucket value
        _write_dict_rows(
            config.output_dir / prior_sensitivity.CONDITION_METADATA_FILENAME,
            condition_columns,
            per_agents_condition_rows,
        )
        # Write the per-population-size combined agent-level CSV across all hyperprior conditions
        _write_dict_rows(
            config.output_dir / prior_sensitivity.COMBINED_AGENT_LEVEL_FILENAME,
            condition_columns + base_experiment.AGENT_LEVEL_CSV_HEADER,
            per_agents_agent_rows,
        )
        # Write the per-population-size combined bucket summary CSV across all hyperprior conditions
        _write_dict_rows(
            config.output_dir / prior_sensitivity.COMBINED_SUMMARY_BY_BUCKET_FILENAME,
            condition_columns + base_experiment.SUMMARY_BY_BUCKET_CSV_HEADER,
            per_agents_bucket_rows,
        )
        # Write the per-population-size combined overall summary CSV across all hyperprior conditions
        _write_dict_rows(
            config.output_dir / prior_sensitivity.COMBINED_SUMMARY_OVERALL_FILENAME,
            condition_columns + base_experiment.SUMMARY_OVERALL_CSV_HEADER,
            per_agents_overall_rows,
        )
        # Regenerate the per-population-size heatmap from the aggregated bucket summary rows
        prior_sensitivity.plot_lowest_bucket_heatmap(
            config.output_dir / prior_sensitivity.LOWEST_BUCKET_HEATMAP_FILENAME,
            per_agents_bucket_rows,
            conditions,
        )

    # End of loop over population sizes

    # Root combined CSVs need agents-per-bucket, scenario, and hyperprior columns before result columns
    combined_prefix_header = (
        AGENTS_PER_BUCKET_METADATA_HEADER               # Columns describing the agents-per-bucket condition
        + SCENARIO_METADATA_HEADER                      # Columns describing the exact scenario folder/path
        + prior_sensitivity.CONDITION_METADATA_HEADER   # Columns describing the hyperprior condition
    )
    # Write one row per agents-per-bucket setting
    _write_dict_rows(
        output_dir / AGENTS_PER_BUCKET_RUNS_FILENAME,
        AGENTS_PER_BUCKET_METADATA_HEADER,
        run_rows,
    )
    # Write one row per agents-per-bucket x hyperprior scenario
    _write_dict_rows(
        output_dir / AGENTS_PER_BUCKET_SCENARIOS_FILENAME,
        combined_prefix_header,
        scenario_rows,
    )
    # Write the full root agent-level CSV across all scenarios
    _write_dict_rows(
        output_dir / COMBINED_AGENT_LEVEL_FILENAME,
        combined_prefix_header + base_experiment.AGENT_LEVEL_CSV_HEADER,
        all_agent_rows,
    )
    # Write the full root bucket summary CSV across all scenarios
    _write_dict_rows(
        output_dir / COMBINED_SUMMARY_BY_BUCKET_FILENAME,
        combined_prefix_header + base_experiment.SUMMARY_BY_BUCKET_CSV_HEADER,
        all_bucket_rows,
    )
    # Write the full root overall summary CSV across all scenarios
    _write_dict_rows(
        output_dir / COMBINED_SUMMARY_OVERALL_FILENAME,
        combined_prefix_header + base_experiment.SUMMARY_OVERALL_CSV_HEADER,
        all_overall_rows,
    )

    # Print the final output location so Slurm logs show where aggregation landed
    print(f"[agents-per-bucket] Aggregated results into {output_dir.resolve()}", flush=True)


def print_dry_run_summary(
    configs: Sequence[base_experiment.ExperimentConfig],
    conditions: Sequence[prior_sensitivity.PriorSensitivityCondition],
    output_dir: Path,
) -> None:
    """Report the agents-per-bucket workload without running inference."""

    scenario_count = len(configs) * len(conditions)
    print("=== DRY RUN: Agents Per Bucket x Hyperprior Sensitivity ===")
    print("No simulation or inference functions will be executed.")
    print()
    print(f"Agents-per-bucket values: {[config.agents_per_bucket for config in configs]}")
    if conditions:
        print(f"Condition preset: {conditions[0].condition_preset}")
    print(f"Prior conditions: {[condition.condition_slug for condition in conditions]}")
    print(f"Total scenarios: {scenario_count}")
    if configs:
        print(f"Seeds per scenario: {_seed_values_label(configs[0].seed_values)}")
        print(f"Count buckets: {configs[0].count_buckets}")
    print(f"Root output directory: {output_dir.resolve()}")
    print()
    print("Scenario folders:")
    scenarios = build_scenarios(configs, conditions)
    for config in configs:
        print(
            f"  - {_agents_per_bucket_slug(config.agents_per_bucket)}: "
            f"{config.num_agents} agents total -> {config.output_dir}"
        )
        for scenario in scenarios:
            if scenario.config.agents_per_bucket == config.agents_per_bucket:
                print(f"      - [{scenario.scenario_index:02d}] {scenario.condition_slug}/")
    print()
    print("Root combined artifacts:")
    print(f"  - {output_dir / AGENTS_PER_BUCKET_RUNS_FILENAME}")
    print(f"  - {output_dir / AGENTS_PER_BUCKET_SCENARIOS_FILENAME}")
    print(f"  - {output_dir / COMBINED_AGENT_LEVEL_FILENAME}")
    print(f"  - {output_dir / COMBINED_SUMMARY_BY_BUCKET_FILENAME}")
    print(f"  - {output_dir / COMBINED_SUMMARY_OVERALL_FILENAME}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the agents-per-bucket sensitivity sweep."""

    # PARSE AND VALIDATE INPUTS ------------------------------------------------------------

    # Parse CLI arguments; ``argv`` is only supplied by tests or programmatic callers
    args = parse_args(argv)

    # Prefer the explicit CLI scenario index, but allow Slurm arrays to provide it through the environment
    environment_scenario_index = scenario_index_from_environment()
    scenario_index = args.scenario_index if args.scenario_index is not None else environment_scenario_index
    if (
        args.scenario_index is not None
        and environment_scenario_index is not None
        and args.scenario_index != environment_scenario_index
    ):
        raise ValueError(
            "--scenario-index and SCENARIO_INDEX/SLURM_ARRAY_TASK_ID disagree. "
            f"Received {args.scenario_index} and {environment_scenario_index}."
        )

    # A run cannot both compute one scenario and aggregate already-computed scenarios
    if scenario_index is not None and args.aggregate_results:
        # Fail early if mutually exclusive cluster modes were requested together
        raise ValueError("--scenario-index and --aggregate-results are mutually exclusive.")

    # Convert the comma-separated agents-per-bucket CLI string into a tuple of ints
    agents_per_bucket_values = _parse_positive_ints(
        args.agents_per_bucket_values,
        field_name="agents_per_bucket",
    )

    # BUILD SCENARIOS FOR EACH POPULATION-SIZE-HYPERPRIORS COMBO ---------------------------

    # Build the selected hyperprior sensitivity condition set
    conditions = prior_sensitivity.build_sensitivity_conditions(args)

    # Build one ExperimentConfig for each requested agents-per-bucket value
    configs = [
        # This config fixes the population size while sharing the other CLI settings
        build_config_for_agents_per_bucket(args, agents_per_bucket)
        for agents_per_bucket in agents_per_bucket_values
    ]

    # Flatten the configs x conditions grid into indexed scenarios for Slurm arrays
    scenarios = build_scenarios(configs, conditions)

    # RUN THE SINGLE REQUESTED SCENARIO OR THE AGGREGATOR, IF APPROPRIATE ------------------

    # Normalize the output directory string into a pathlib Path
    output_dir = Path(args.output_dir)

    # Dry-run reports the workload and exits before simulation/inference
    if args.dry_run:
        print_dry_run_summary(configs, conditions, output_dir)
        return 0

    # Scenario-index mode is used by one Slurm array task to compute one scenario
    if scenario_index is not None:
        # Validate that the requested index exists in the flattened scenario list
        if scenario_index < 0 or scenario_index >= len(scenarios):
            raise ValueError(
                f"scenario_index must be between 0 and {len(scenarios) - 1}. "
                f"Received {scenario_index}."
            )
        # Run exactly the requested scenario and write its scenario-level artifacts
        run_single_scenario(scenarios[scenario_index])
        return 0

    # Aggregation mode is the final Slurm dependency task after scenario tasks finish
    if args.aggregate_results:
        # Read completed scenario folders and collect them into combined CSVs/plots
        aggregate_existing_results(configs, conditions, output_dir)
        return 0

    # OTHERWISE, RUN ENTIRE EXPERIMENT LOCALLY ---------------------------------------------

    # All-in-one local run rows
    run_rows: list[dict[str, Any]] = []         # One row per agents-per-bucket setting
    scenario_rows: list[dict[str, Any]] = []    # One row per agents-per-bucket x hyperprior scenario
    all_agent_rows: list[dict[str, Any]] = []   # Every agent-level result from every scenario
    all_bucket_rows: list[dict[str, Any]] = []  # Every bucket summary row from every scenario
    all_overall_rows: list[dict[str, Any]] = [] # Every overall summary row from every scenario

    # In all-in-one mode, run each agents-per-bucket value sequentially
    for config_index, config in enumerate(configs, start=1):
        # Log which population-size condition is currently running
        print(
            "[agents-per-bucket] "
            f"Running value {config_index}/{len(configs)}: "
            f"{config.agents_per_bucket} agents per bucket ({config.num_agents} agents/seed)",
            flush=True,
        )

        # Run the selected hyperprior sensitivity condition set for this population size
        grid_result = prior_sensitivity.run_sensitivity_grid(config, conditions)

        # Record metadata for the current agents-per-bucket setting in the whole-sweep runs table
        run_metadata = agents_per_bucket_metadata_row(config)
        run_rows.append(run_metadata)

        # Convert each condition metadata row into a whole-sweep scenario metadata row
        for condition_row in grid_result.condition_rows:
            condition_slug = str(condition_row["condition_slug"])
            scenario_rows.append(
                {
                    **run_metadata, # agents-per-bucket metadata
                    **scenario_metadata_row(config, condition_slug), # scenario path metadata
                    **condition_row, # hyperprior metadata
                }
            )

        # Add agents-per-bucket/scenario provenance to every agent-level, bucket summary, and overall summary row
        all_agent_rows.extend(_prefix_result_row(row, config) for row in grid_result.agent_rows)
        all_bucket_rows.extend(_prefix_result_row(row, config) for row in grid_result.bucket_rows)
        all_overall_rows.extend(_prefix_result_row(row, config) for row in grid_result.overall_rows)

    # Root combined CSVs include agents-per-bucket, scenario, and hyperprior metadata columns
    combined_prefix_header = (
        AGENTS_PER_BUCKET_METADATA_HEADER # Columns for the population-size condition
        + SCENARIO_METADATA_HEADER # Columns for scenario slug and output paths
        + prior_sensitivity.CONDITION_METADATA_HEADER # Columns for concrete hyperprior values
    )

    # Write one row per agents-per-bucket value
    _write_dict_rows(
        output_dir / AGENTS_PER_BUCKET_RUNS_FILENAME,
        AGENTS_PER_BUCKET_METADATA_HEADER,
        run_rows,
    )
    # Write one row per agents-per-bucket x hyperprior scenario
    _write_dict_rows(
        output_dir / AGENTS_PER_BUCKET_SCENARIOS_FILENAME,
        combined_prefix_header,
        scenario_rows,
    )
    # Write every agent-level result from the whole sweep
    _write_dict_rows(
        output_dir / COMBINED_AGENT_LEVEL_FILENAME,
        combined_prefix_header + base_experiment.AGENT_LEVEL_CSV_HEADER,
        all_agent_rows,
    )
    # Write every bucket summary row from the whole sweep
    _write_dict_rows(
        output_dir / COMBINED_SUMMARY_BY_BUCKET_FILENAME,
        combined_prefix_header + base_experiment.SUMMARY_BY_BUCKET_CSV_HEADER,
        all_bucket_rows,
    )
    # Write every overall summary row from the whole sweep
    _write_dict_rows(
        output_dir / COMBINED_SUMMARY_OVERALL_FILENAME,
        combined_prefix_header + base_experiment.SUMMARY_OVERALL_CSV_HEADER,
        all_overall_rows,
    )

    # Log the final output root so local/Slurm logs show where artifacts landed
    print(f"[agents-per-bucket] Wrote combined results to {output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
