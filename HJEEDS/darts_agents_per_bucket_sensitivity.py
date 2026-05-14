# This file was written or edited by AI and still requires human review. Delete this comment when done.
"""Run agents-per-bucket ablations crossed with hyperprior sensitivity.

This script is the May 14 ablation driver. It repeats the existing 3x3
hyperprior-sensitivity analysis for several population sizes, where population
size is controlled by the number of agents assigned to each observation-count
bucket.

Default sweep:

- agents per bucket: 1, 2, 5, 10, 25
- count buckets: 5, 10, 25, 100, 1000 observations
- prior conditions: weak/default/strong confidence crossed with
  unbiased/slightly biased/significantly biased centers

The output tree intentionally mirrors the existing prior-sensitivity runner so
each of the 45 scenarios has an ordinary experiment folder with CSVs and an
``error_by_count_bucket.png`` plot, while the root folder also contains combined
CSVs for spreadsheet/notebook analysis.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


# Ensure the repository root is importable when this file is executed directly.
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

    parser.add_argument("--seed", type=int, default=12345, help="Base seed used to derive per-run seeds.")
    parser.add_argument(
        "--num-seeds",
        type=int,
        required=True,
        help="Number of random seeds to run for each of the 45 scenarios.",
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
        "--weak-std-multiplier",
        type=float,
        default=prior_sensitivity.DEFAULT_WEAK_STD_MULTIPLIER,
        help="Multiplier applied to hyperprior standard deviations for the weak condition.",
    )
    parser.add_argument(
        "--strong-std-multiplier",
        type=float,
        default=prior_sensitivity.DEFAULT_STRONG_STD_MULTIPLIER,
        help="Multiplier applied to hyperprior standard deviations for the strong condition.",
    )
    parser.add_argument(
        "--slight-bias-sd-units",
        type=float,
        default=prior_sensitivity.DEFAULT_SLIGHT_BIAS_SD_UNITS,
        help="Mean shift for the slightly biased condition, measured in default prior-SD units.",
    )
    parser.add_argument(
        "--significant-bias-sd-units",
        type=float,
        default=prior_sensitivity.DEFAULT_SIGNIFICANT_BIAS_SD_UNITS,
        help="Mean shift for the significantly biased condition, measured in default prior-SD units.",
    )
    parser.add_argument(
        "--condition-slugs",
        type=str,
        default="",
        help=(
            "Optional comma-separated subset of prior-sensitivity condition slugs to run. "
            "Leave blank to run all nine."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the planned 5 x 9 workload and stop before simulation/inference.",
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
    """Return path metadata for one of the 45 concrete scenarios."""

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

    run_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    all_agent_rows: list[dict[str, Any]] = []
    all_bucket_rows: list[dict[str, Any]] = []
    all_overall_rows: list[dict[str, Any]] = []

    for config in configs:
        run_metadata = agents_per_bucket_metadata_row(config)
        run_rows.append(run_metadata)

        per_agents_condition_rows: list[dict[str, Any]] = []
        per_agents_agent_rows: list[dict[str, Any]] = []
        per_agents_bucket_rows: list[dict[str, Any]] = []
        per_agents_overall_rows: list[dict[str, Any]] = []

        for condition in conditions:
            condition_metadata = _condition_metadata_row(config, condition)
            scenario_metadata = scenario_metadata_row(config, condition.condition_slug)
            scenario_slug = str(scenario_metadata["scenario_slug"])
            scenario_output_dir = Path(str(scenario_metadata["scenario_output_dir"]))
            output_paths = base_experiment.planned_output_paths(scenario_output_dir)

            agent_rows = _read_dict_rows(output_paths["agent_level_csv"], scenario_slug)
            bucket_rows = _read_dict_rows(output_paths["summary_by_bucket_csv"], scenario_slug)
            overall_rows = _read_dict_rows(output_paths["summary_overall_csv"], scenario_slug)

            per_agents_condition_rows.append(condition_metadata)
            scenario_rows.append({**run_metadata, **scenario_metadata, **condition_metadata})

            condition_agent_rows = [{**condition_metadata, **row} for row in agent_rows]
            condition_bucket_rows = [{**condition_metadata, **row} for row in bucket_rows]
            condition_overall_rows = [{**condition_metadata, **row} for row in overall_rows]

            per_agents_agent_rows.extend(condition_agent_rows)
            per_agents_bucket_rows.extend(condition_bucket_rows)
            per_agents_overall_rows.extend(condition_overall_rows)

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

        condition_columns = prior_sensitivity.CONDITION_METADATA_HEADER
        _write_dict_rows(
            config.output_dir / prior_sensitivity.CONDITION_METADATA_FILENAME,
            condition_columns,
            per_agents_condition_rows,
        )
        _write_dict_rows(
            config.output_dir / prior_sensitivity.COMBINED_AGENT_LEVEL_FILENAME,
            condition_columns + base_experiment.AGENT_LEVEL_CSV_HEADER,
            per_agents_agent_rows,
        )
        _write_dict_rows(
            config.output_dir / prior_sensitivity.COMBINED_SUMMARY_BY_BUCKET_FILENAME,
            condition_columns + base_experiment.SUMMARY_BY_BUCKET_CSV_HEADER,
            per_agents_bucket_rows,
        )
        _write_dict_rows(
            config.output_dir / prior_sensitivity.COMBINED_SUMMARY_OVERALL_FILENAME,
            condition_columns + base_experiment.SUMMARY_OVERALL_CSV_HEADER,
            per_agents_overall_rows,
        )
        prior_sensitivity.plot_lowest_bucket_heatmap(
            config.output_dir / prior_sensitivity.LOWEST_BUCKET_HEATMAP_FILENAME,
            per_agents_bucket_rows,
            conditions,
        )

    combined_prefix_header = (
        AGENTS_PER_BUCKET_METADATA_HEADER
        + SCENARIO_METADATA_HEADER
        + prior_sensitivity.CONDITION_METADATA_HEADER
    )
    _write_dict_rows(
        output_dir / AGENTS_PER_BUCKET_RUNS_FILENAME,
        AGENTS_PER_BUCKET_METADATA_HEADER,
        run_rows,
    )
    _write_dict_rows(
        output_dir / AGENTS_PER_BUCKET_SCENARIOS_FILENAME,
        combined_prefix_header,
        scenario_rows,
    )
    _write_dict_rows(
        output_dir / COMBINED_AGENT_LEVEL_FILENAME,
        combined_prefix_header + base_experiment.AGENT_LEVEL_CSV_HEADER,
        all_agent_rows,
    )
    _write_dict_rows(
        output_dir / COMBINED_SUMMARY_BY_BUCKET_FILENAME,
        combined_prefix_header + base_experiment.SUMMARY_BY_BUCKET_CSV_HEADER,
        all_bucket_rows,
    )
    _write_dict_rows(
        output_dir / COMBINED_SUMMARY_OVERALL_FILENAME,
        combined_prefix_header + base_experiment.SUMMARY_OVERALL_CSV_HEADER,
        all_overall_rows,
    )

    print(f"[agents-per-bucket] Aggregated results into {output_dir.resolve()}", flush=True)


def print_dry_run_summary(
    configs: Sequence[base_experiment.ExperimentConfig],
    conditions: Sequence[prior_sensitivity.PriorSensitivityCondition],
    output_dir: Path,
) -> None:
    """Report the planned agents-per-bucket workload without running inference."""

    scenario_count = len(configs) * len(conditions)
    print("=== DRY RUN: Agents Per Bucket x Hyperprior Sensitivity ===")
    print("No simulation or inference functions will be executed.")
    print()
    print(f"Agents-per-bucket values: {[config.agents_per_bucket for config in configs]}")
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

    args = parse_args(argv)
    agents_per_bucket_values = _parse_positive_ints(
        args.agents_per_bucket_values,
        field_name="agents_per_bucket",
    )
    conditions = prior_sensitivity.build_sensitivity_conditions(args)
    output_dir = Path(args.output_dir)
    configs = [
        build_config_for_agents_per_bucket(args, agents_per_bucket)
        for agents_per_bucket in agents_per_bucket_values
    ]
    scenarios = build_scenarios(configs, conditions)

    if args.scenario_index is not None and args.aggregate_results:
        raise ValueError("--scenario-index and --aggregate-results are mutually exclusive.")

    if args.dry_run:
        print_dry_run_summary(configs, conditions, output_dir)
        return 0

    if args.scenario_index is not None:
        if args.scenario_index < 0 or args.scenario_index >= len(scenarios):
            raise ValueError(
                f"scenario_index must be between 0 and {len(scenarios) - 1}. "
                f"Received {args.scenario_index}."
            )
        run_single_scenario(scenarios[args.scenario_index])
        return 0

    if args.aggregate_results:
        aggregate_existing_results(configs, conditions, output_dir)
        return 0

    run_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    all_agent_rows: list[dict[str, Any]] = []
    all_bucket_rows: list[dict[str, Any]] = []
    all_overall_rows: list[dict[str, Any]] = []

    for config_index, config in enumerate(configs, start=1):
        print(
            "[agents-per-bucket] "
            f"Running value {config_index}/{len(configs)}: "
            f"{config.agents_per_bucket} agents per bucket ({config.num_agents} agents/seed)",
            flush=True,
        )
        grid_result = prior_sensitivity.run_sensitivity_grid(config, conditions)
        run_metadata = agents_per_bucket_metadata_row(config)
        run_rows.append(run_metadata)

        for condition_row in grid_result.condition_rows:
            condition_slug = str(condition_row["condition_slug"])
            scenario_rows.append(
                {
                    **run_metadata,
                    **scenario_metadata_row(config, condition_slug),
                    **condition_row,
                }
            )

        all_agent_rows.extend(_prefix_result_row(row, config) for row in grid_result.agent_rows)
        all_bucket_rows.extend(_prefix_result_row(row, config) for row in grid_result.bucket_rows)
        all_overall_rows.extend(_prefix_result_row(row, config) for row in grid_result.overall_rows)

    combined_prefix_header = (
        AGENTS_PER_BUCKET_METADATA_HEADER
        + SCENARIO_METADATA_HEADER
        + prior_sensitivity.CONDITION_METADATA_HEADER
    )
    _write_dict_rows(
        output_dir / AGENTS_PER_BUCKET_RUNS_FILENAME,
        AGENTS_PER_BUCKET_METADATA_HEADER,
        run_rows,
    )
    _write_dict_rows(
        output_dir / AGENTS_PER_BUCKET_SCENARIOS_FILENAME,
        combined_prefix_header,
        scenario_rows,
    )
    _write_dict_rows(
        output_dir / COMBINED_AGENT_LEVEL_FILENAME,
        combined_prefix_header + base_experiment.AGENT_LEVEL_CSV_HEADER,
        all_agent_rows,
    )
    _write_dict_rows(
        output_dir / COMBINED_SUMMARY_BY_BUCKET_FILENAME,
        combined_prefix_header + base_experiment.SUMMARY_BY_BUCKET_CSV_HEADER,
        all_bucket_rows,
    )
    _write_dict_rows(
        output_dir / COMBINED_SUMMARY_OVERALL_FILENAME,
        combined_prefix_header + base_experiment.SUMMARY_OVERALL_CSV_HEADER,
        all_overall_rows,
    )

    print(f"[agents-per-bucket] Wrote combined results to {output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
