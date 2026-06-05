# This file has been fully edited by a human researcher as of 05/23/26 at 2:28 PM MDT.
"""Run H-JEEDS high-data-anchor availability experiments.

This runner tests whether H-JEEDS depends on the presence of high-data agents.
Each scenario keeps a fixed group of one-sample agents, then varies how many
25-sample anchor agents are added to the same population.

Default sweep:

- low-data agents: 25 agents with 1 observation each
- anchor agents: 0, 1, 2, 5, 10, 25 agents with 25 observations each
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# Ensure the repository root is importable when this file is executed directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HJEEDS import darts_hierarchical_vs_jeeds as base_experiment
from HJEEDS.artifacts import add_plotting_cli_arguments, error_metric_panels, METHOD_ORDER


DEFAULT_OUTPUT_DIR = Path("HJEEDS/results/hierarchical_darts_anchor_availability_sensitivity")
DEFAULT_LOW_DATA_AGENT_COUNT = 25
DEFAULT_LOW_DATA_OBSERVATIONS = 1
DEFAULT_ANCHOR_OBSERVATIONS = 25
DEFAULT_ANCHOR_AGENT_COUNTS = (0, 1, 2, 5, 10, 25)

SCENARIOS_FILENAME = "anchor_availability_sensitivity_scenarios.csv"
COMBINED_AGENT_LEVEL_FILENAME = "anchor_availability_sensitivity_agent_level_results.csv"
COMBINED_SUMMARY_BY_BUCKET_FILENAME = "anchor_availability_sensitivity_summary_by_bucket.csv"
COMBINED_SUMMARY_OVERALL_FILENAME = "anchor_availability_sensitivity_summary_overall.csv"
LOW_DATA_PLOT_TEMPLATE = "anchor_availability_low_data_{metric}.png"

ANCHOR_AVAILABILITY_METADATA_HEADER = [
    "anchor_availability_slug",
    "anchor_availability_label",
    "low_data_agent_count",
    "low_data_observations",
    "anchor_agent_count",
    "anchor_observations",
    "scenario_num_agents",
    "count_buckets",
    "anchor_availability_description",
]

SCENARIO_METADATA_HEADER = [
    "scenario_index",
    "scenario_slug",
    "scenario_output_dir",
    "scenario_error_plot",
]


@dataclass(frozen=True)
class AnchorAvailabilitySpec:
    """Metadata for one high-data-anchor availability condition."""

    slug: str
    label: str
    low_data_agent_count: int
    low_data_observations: int
    anchor_agent_count: int
    anchor_observations: int
    description: str

    @property
    def total_agents(self) -> int:
        """Return the number of demonstrators in this condition."""

        return self.low_data_agent_count + self.anchor_agent_count


@dataclass(frozen=True)
class AnchorAvailabilityScenario:
    """One concrete high-data-anchor availability scenario."""

    scenario_index: int
    config: base_experiment.ExperimentConfig
    anchor_availability: AnchorAvailabilitySpec

    @property
    def scenario_slug(self) -> str:
        """Return a stable scenario identifier."""

        return anchor_availability_folder_slug(self.anchor_availability)

    @property
    def scenario_output_dir(self) -> Path:
        """Return the directory where this scenario's normal artifacts live."""

        return self.config.output_dir


def build_anchor_availability_specs() -> tuple[AnchorAvailabilitySpec, ...]:
    """Return the default high-data-anchor availability grid."""

    specs: list[AnchorAvailabilitySpec] = []
    for anchor_agent_count in DEFAULT_ANCHOR_AGENT_COUNTS:
        anchor_agent_label = "anchor agent" if anchor_agent_count == 1 else "anchor agents"
        specs.append(
            AnchorAvailabilitySpec(
                slug=f"anchor_agents_{anchor_agent_count:03d}",
                label=f"{anchor_agent_count} {anchor_agent_label}",
                low_data_agent_count=DEFAULT_LOW_DATA_AGENT_COUNT,
                low_data_observations=DEFAULT_LOW_DATA_OBSERVATIONS,
                anchor_agent_count=anchor_agent_count,
                anchor_observations=DEFAULT_ANCHOR_OBSERVATIONS,
                description=(
                    f"{DEFAULT_LOW_DATA_AGENT_COUNT} one-sample agents plus "
                    f"{anchor_agent_count} {anchor_agent_label} with "
                    f"{DEFAULT_ANCHOR_OBSERVATIONS} observations each"
                ),
            )
        )
    return tuple(specs)


ANCHOR_AVAILABILITY_SPECS = build_anchor_availability_specs()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for the high-data-anchor ablation."""

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
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Root output directory for the high-data-anchor sweep.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the high-data-anchor workload and stop before simulation/inference.",
    )
    parser.add_argument(
        "--aggregate-results",
        action="store_true",
        help="Collect already-computed scenario folders into root combined CSVs and plots.",
    )
    add_plotting_cli_arguments(parser)
    return parser.parse_args(argv)


def _count_bucket_label(count_buckets: Sequence[int]) -> str:
    """Return a compact label for CSV provenance."""

    return ",".join(str(bucket) for bucket in count_buckets)


def _seed_values_label(seed_values: Sequence[int]) -> str:
    """Return a readable seed description for dry-run output."""

    if len(seed_values) <= 10:
        return str(tuple(seed_values))
    return f"{len(seed_values)} seeds ({seed_values[0]} through {seed_values[-1]})"


def anchor_availability_folder_slug(anchor_availability: AnchorAvailabilitySpec) -> str:
    """Return a stable folder slug for one high-data-anchor condition."""

    return anchor_availability.slug


def observation_count_design(anchor_availability: AnchorAvailabilitySpec) -> tuple[int, ...]:
    """Return the per-agent observation-count design for one scenario."""

    # This ablation treats count_buckets as a per-agent observation-count vector
    # With agents_per_bucket=1, each repeated count creates one agent, and the
    # summary code later merges repeated values by grouping on numeric count_bucket
    low_data_counts = (anchor_availability.low_data_observations,) * anchor_availability.low_data_agent_count
    anchor_counts = (anchor_availability.anchor_observations,) * anchor_availability.anchor_agent_count
    return low_data_counts + anchor_counts


def scenario_index_from_environment() -> int | None:
    """Return the Slurm scenario index from environment variables if present."""

    raw_value = os.environ.get("SCENARIO_INDEX") or os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw_value is None or raw_value == "":
        return None
    try:
        scenario_index = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Scenario index environment value must be an integer. Received: {raw_value}.") from exc
    if scenario_index < 0:
        raise ValueError(f"Scenario index environment value must be nonnegative. Received: {scenario_index}.")
    return scenario_index


def build_config_for_scenario(
    args: argparse.Namespace,
    anchor_availability: AnchorAvailabilitySpec,
) -> base_experiment.ExperimentConfig:
    """Build one base H-JEEDS config for a high-data-anchor scenario."""

    count_buckets = observation_count_design(anchor_availability)
    output_dir = Path(args.output_dir) / anchor_availability_folder_slug(anchor_availability)
    base_args = argparse.Namespace(
        seed=args.seed,
        num_seeds=args.num_seeds,
        num_agents=len(count_buckets),
        count_buckets=_count_bucket_label(count_buckets),
        agents_per_bucket=1,
        delta=base_experiment.DEFAULT_DELTA,
        environment="1d",
        num_sigma_grid=base_experiment.DEFAULT_NUM_SIGMA_GRID,
        num_lambda_grid=base_experiment.DEFAULT_NUM_LAMBDA_GRID,
        sigma_min=base_experiment.DEFAULT_SIGMA_MIN,
        sigma_max=base_experiment.DEFAULT_SIGMA_MAX,
        lambda_min=base_experiment.DEFAULT_LAMBDA_MIN,
        lambda_max=base_experiment.DEFAULT_LAMBDA_MAX,
        output_dir=str(output_dir),
        dry_run=args.dry_run,
        min_success_regions=base_experiment.DEFAULT_MIN_SUCCESS_REGIONS,
        max_success_regions=base_experiment.DEFAULT_MAX_SUCCESS_REGIONS,
        min_region_width=base_experiment.DEFAULT_MIN_REGION_WIDTH,
    )
    return base_experiment.build_config_from_args(base_args)


def build_scenarios(args: argparse.Namespace) -> tuple[AnchorAvailabilityScenario, ...]:
    """Return the flat scenario list used by local runs and Slurm arrays."""

    scenarios: list[AnchorAvailabilityScenario] = []
    for anchor_availability in ANCHOR_AVAILABILITY_SPECS:
        scenarios.append(
            AnchorAvailabilityScenario(
                scenario_index=len(scenarios),
                config=build_config_for_scenario(args, anchor_availability),
                anchor_availability=anchor_availability,
            )
        )
    return tuple(scenarios)


def anchor_availability_metadata_row(
    anchor_availability: AnchorAvailabilitySpec,
    config: base_experiment.ExperimentConfig,
) -> dict[str, Any]:
    """Return CSV metadata for one high-data-anchor condition."""

    return {
        "anchor_availability_slug": anchor_availability.slug,
        "anchor_availability_label": anchor_availability.label,
        "low_data_agent_count": anchor_availability.low_data_agent_count,
        "low_data_observations": anchor_availability.low_data_observations,
        "anchor_agent_count": anchor_availability.anchor_agent_count,
        "anchor_observations": anchor_availability.anchor_observations,
        "scenario_num_agents": config.num_agents,
        "count_buckets": _count_bucket_label(config.count_buckets),
        "anchor_availability_description": anchor_availability.description,
    }


def scenario_metadata_row(scenario: AnchorAvailabilityScenario) -> dict[str, Any]:
    """Return path metadata for one concrete scenario."""

    return {
        "scenario_index": scenario.scenario_index,
        "scenario_slug": scenario.scenario_slug,
        "scenario_output_dir": str(scenario.scenario_output_dir),
        "scenario_error_plot": str(scenario.scenario_output_dir / base_experiment.ERROR_PLOT_FILENAME),
    }


def scenario_prefix_row(scenario: AnchorAvailabilityScenario) -> dict[str, Any]:
    """Return all provenance columns for one scenario."""

    return {
        **anchor_availability_metadata_row(scenario.anchor_availability, scenario.config),
        **scenario_metadata_row(scenario),
    }


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


def run_hjeeds_config(
    config: base_experiment.ExperimentConfig,
    *,
    log_prefix: str,
    include_raw_rationality_error: bool = False,
) -> None:
    """Run the ordinary H-JEEDS experiment for one scenario config."""

    seed_results: list[base_experiment.SeedResult] = []
    for seed_index, seed in enumerate(config.seed_values, start=1):
        print(f"{log_prefix} seed {seed_index}/{config.num_seeds}: {seed}", flush=True)
        seed_results.append(base_experiment.run_single_seed(config, seed))

    output_paths = base_experiment.planned_output_paths(config.output_dir)
    all_agent_results = [result for seed_result in seed_results for result in seed_result.agent_results]
    summary_by_bucket_rows, summary_overall_rows = base_experiment.aggregate_results_across_seeds(seed_results)

    base_experiment.write_agent_level_csv(output_paths["agent_level_csv"], all_agent_results)
    base_experiment.write_summary_csvs(
        config.output_dir,
        summary_by_bucket_rows,
        summary_overall_rows,
    )
    base_experiment.plot_error_by_bucket(
        output_paths["error_plot"],
        summary_by_bucket_rows,
        include_raw_rationality_error=include_raw_rationality_error,
    )


def run_single_scenario(
    scenario: AnchorAvailabilityScenario,
    *,
    include_raw_rationality_error: bool = False,
) -> None:
    """Run one high-data-anchor availability scenario."""

    print(
        "[anchor-availability] "
        f"Running scenario {scenario.scenario_index}: {scenario.scenario_slug} "
        f"({scenario.config.num_agents} agents/seed)",
        flush=True,
    )
    run_hjeeds_config(
        scenario.config,
        log_prefix=f"[anchor-availability] {scenario.scenario_slug}:",
        include_raw_rationality_error=include_raw_rationality_error,
    )
    print(
        "[anchor-availability] "
        f"Wrote scenario results to {scenario.scenario_output_dir.resolve()}",
        flush=True,
    )


def _as_float(value: Any) -> float | None:
    """Convert CSV-ish values to floats, treating blanks as missing."""

    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _plot_low_data_metric(
    output_path: Path,
    rows: Sequence[dict[str, Any]],
    metric_name: str,
    title: str,
    ylabel: str,
    missing_message: str,
) -> None:
    """Plot one low-data-agent metric across high-data-anchor counts."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    selected_bucket = DEFAULT_LOW_DATA_OBSERVATIONS
    anchor_counts = [spec.anchor_agent_count for spec in ANCHOR_AVAILABILITY_SPECS]
    x_positions = {anchor_count: index for index, anchor_count in enumerate(anchor_counts)}

    parsed_rows: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        if str(row.get("metric", "")) != metric_name:
            continue
        row_bucket = _as_float(row.get("count_bucket"))
        mean = _as_float(row.get("mean"))
        if row_bucket is None or mean is None or int(row_bucket) != selected_bucket:
            continue
        key = (
            int(row["anchor_agent_count"]),
            str(row["method"]),
        )
        parsed_rows[key] = row

    methods = sorted(
        {
            method
            for (_anchor_count, method) in parsed_rows
        },
        key=lambda method: (METHOD_ORDER.get(method, len(METHOD_ORDER)), method),
    )

    figure, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    axis.set_title(f"{title} for {selected_bucket}-sample agents")
    axis.set_xlabel(f"{DEFAULT_ANCHOR_OBSERVATIONS}-sample anchor agents")
    axis.set_ylabel(ylabel)
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axis.set_xticks(list(x_positions.values()))
    axis.set_xticklabels([str(anchor_count) for anchor_count in anchor_counts])

    drew_rows = False
    for method in methods:
        x_values: list[int] = []
        y_values: list[float] = []
        lower_errors: list[float] = []
        upper_errors: list[float] = []

        for anchor_count in anchor_counts:
            row = parsed_rows.get((anchor_count, method))
            if row is None:
                continue
            mean = _as_float(row.get("mean"))
            if mean is None:
                continue
            ci_lower = _as_float(row.get("ci_lower"))
            ci_upper = _as_float(row.get("ci_upper"))
            x_values.append(x_positions[anchor_count])
            y_values.append(mean)
            lower_errors.append(max(0.0, mean - ci_lower) if ci_lower is not None else 0.0)
            upper_errors.append(max(0.0, ci_upper - mean) if ci_upper is not None else 0.0)

        if x_values:
            drew_rows = True
            axis.errorbar(
                x_values,
                y_values,
                yerr=np.array([lower_errors, upper_errors], dtype=float),
                marker="o",
                capsize=4,
                linewidth=2,
                label=method,
            )

    if drew_rows:
        axis.legend(title="Method")
    else:
        axis.text(
            0.5,
            0.5,
            missing_message,
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_low_data_comparisons(
    output_dir: Path,
    summary_by_bucket_rows: Sequence[dict[str, Any]],
    *,
    include_raw_rationality_error: bool = False,
) -> None:
    """Write one root low-data-agent comparison plot per metric."""

    for metric_name, title, ylabel, missing_message in error_metric_panels(include_raw_rationality_error):
        _plot_low_data_metric(
            output_dir / LOW_DATA_PLOT_TEMPLATE.format(metric=metric_name),
            summary_by_bucket_rows,
            metric_name,
            title,
            ylabel,
            missing_message,
        )


def aggregate_existing_results(
    scenarios: Sequence[AnchorAvailabilityScenario],
    output_dir: Path,
    *,
    include_raw_rationality_error: bool = False,
    regenerate_scenario_plots: bool = False,
) -> None:
    """Collect precomputed scenario folders into combined sweep artifacts."""

    scenario_rows: list[dict[str, Any]] = []
    all_agent_rows: list[dict[str, Any]] = []
    all_bucket_rows: list[dict[str, Any]] = []
    all_overall_rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        prefix = scenario_prefix_row(scenario)
        output_paths = base_experiment.planned_output_paths(scenario.scenario_output_dir)
        scenario_slug = str(prefix["scenario_slug"])

        agent_rows = _read_dict_rows(output_paths["agent_level_csv"], scenario_slug)
        bucket_rows = _read_dict_rows(output_paths["summary_by_bucket_csv"], scenario_slug)
        overall_rows = _read_dict_rows(output_paths["summary_overall_csv"], scenario_slug)
        if regenerate_scenario_plots:
            base_experiment.plot_error_by_bucket(
                output_paths["error_plot"],
                bucket_rows,
                include_raw_rationality_error=include_raw_rationality_error,
            )

        scenario_rows.append(prefix)
        all_agent_rows.extend({**prefix, **row} for row in agent_rows)
        all_bucket_rows.extend({**prefix, **row} for row in bucket_rows)
        all_overall_rows.extend({**prefix, **row} for row in overall_rows)

    combined_prefix_header = (
        ANCHOR_AVAILABILITY_METADATA_HEADER
        + SCENARIO_METADATA_HEADER
    )
    _write_dict_rows(output_dir / SCENARIOS_FILENAME, combined_prefix_header, scenario_rows)
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
    plot_low_data_comparisons(
        output_dir,
        all_bucket_rows,
        include_raw_rationality_error=include_raw_rationality_error,
    )

    print(f"[anchor-availability] Aggregated results into {output_dir.resolve()}", flush=True)


def print_dry_run_summary(
    scenarios: Sequence[AnchorAvailabilityScenario],
    output_dir: Path,
    *,
    include_raw_rationality_error: bool = False,
) -> None:
    """Report the high-data-anchor workload without running inference."""

    anchor_counts = [scenario.anchor_availability.anchor_agent_count for scenario in scenarios]

    print("=== DRY RUN: High-Data Anchor Availability Sensitivity ===")
    print("No simulation or inference functions will be executed.")
    print()
    print(f"Low-data agents: {DEFAULT_LOW_DATA_AGENT_COUNT} agents with {DEFAULT_LOW_DATA_OBSERVATIONS} observation each")
    print(f"Anchor agents: {anchor_counts}")
    print(f"Anchor observations per anchor: {DEFAULT_ANCHOR_OBSERVATIONS}")
    print(f"Total scenarios: {len(scenarios)}")
    if scenarios:
        print(f"Seeds per scenario: {_seed_values_label(scenarios[0].config.seed_values)}")
    print(f"Root output directory: {output_dir.resolve()}")
    print()
    print("Scenario folders:")
    for scenario in scenarios:
        design = scenario.anchor_availability
        print(
            f"  - [{scenario.scenario_index:02d}] {scenario.scenario_slug}/ "
            f"-> {scenario.scenario_output_dir} "
            f"({design.low_data_agent_count} low-data + {design.label})"
        )
    print()
    print("Root combined artifacts:")
    print(f"  - {output_dir / SCENARIOS_FILENAME}")
    print(f"  - {output_dir / COMBINED_AGENT_LEVEL_FILENAME}")
    print(f"  - {output_dir / COMBINED_SUMMARY_BY_BUCKET_FILENAME}")
    print(f"  - {output_dir / COMBINED_SUMMARY_OVERALL_FILENAME}")
    for metric_name, _title, _ylabel, _missing_message in error_metric_panels(include_raw_rationality_error):
        print(f"  - {output_dir / LOW_DATA_PLOT_TEMPLATE.format(metric=metric_name)}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the high-data-anchor availability sensitivity sweep."""

    args = parse_args(argv)
    scenario_index = scenario_index_from_environment()
    if scenario_index is not None and (args.aggregate_results or args.plot_only):
        raise ValueError("Scenario-index environment mode cannot be combined with --aggregate-results or --plot-only.")

    scenarios = build_scenarios(args)
    output_dir = Path(args.output_dir)

    if args.dry_run:
        print_dry_run_summary(
            scenarios,
            output_dir,
            include_raw_rationality_error=args.include_raw_rationality_error,
        )
        return 0

    if scenario_index is not None:
        if scenario_index >= len(scenarios):
            raise ValueError(
                f"scenario_index must be between 0 and {len(scenarios) - 1}. "
                f"Received {scenario_index}."
            )
        run_single_scenario(
            scenarios[scenario_index],
            include_raw_rationality_error=args.include_raw_rationality_error,
        )
        return 0

    if args.aggregate_results or args.plot_only:
        aggregate_existing_results(
            scenarios,
            output_dir,
            include_raw_rationality_error=args.include_raw_rationality_error,
            regenerate_scenario_plots=args.plot_only,
        )
        return 0

    for scenario in scenarios:
        run_single_scenario(
            scenario,
            include_raw_rationality_error=args.include_raw_rationality_error,
        )
    aggregate_existing_results(
        scenarios,
        output_dir,
        include_raw_rationality_error=args.include_raw_rationality_error,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
