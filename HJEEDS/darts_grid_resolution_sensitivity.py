# This file has been fully edited by a human researcher as of 05/22/26 at 6:01 PM MDT.
"""Run H-JEEDS grid-resolution sensitivity experiments.

This runner varies the estimator's discrete JEEDS skill-grid resolution. The
study is a compact appendix sanity check rather than a main factorial ablation.
The default sweep is:

- 11 x 11 skill grid
- 21 x 21 skill grid
- 41 x 41 skill grid
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
from HJEEDS.artifacts import (
    add_plotting_cli_arguments,
    error_metric_panels,
    method_color,
    method_label,
    method_marker,
    METHOD_ORDER,
)


DEFAULT_OUTPUT_DIR = Path("HJEEDS/results/hierarchical_darts_grid_resolution_sensitivity")
DEFAULT_COUNT_BUCKETS = base_experiment.DEFAULT_COUNT_BUCKETS
DEFAULT_AGENTS_PER_BUCKET = base_experiment.DEFAULT_AGENTS_PER_BUCKET

SCENARIOS_FILENAME = "grid_resolution_sensitivity_scenarios.csv"
COMBINED_AGENT_LEVEL_FILENAME = "grid_resolution_sensitivity_agent_level_results.csv"
COMBINED_SUMMARY_BY_BUCKET_FILENAME = "grid_resolution_sensitivity_summary_by_bucket.csv"
COMBINED_SUMMARY_OVERALL_FILENAME = "grid_resolution_sensitivity_summary_overall.csv"
LOWEST_BUCKET_PLOT_TEMPLATE = "grid_resolution_lowest_bucket_{metric}.png"

GRID_RESOLUTION_METADATA_HEADER = [
    "grid_resolution_slug",
    "grid_resolution_label",
    "num_sigma_grid",
    "num_lambda_grid",
    "num_grid_cells",
    "grid_resolution_description",
]

SCENARIO_METADATA_HEADER = [
    "scenario_index",
    "scenario_slug",
    "scenario_output_dir",
    "scenario_error_plot",
]


@dataclass(frozen=True)
class GridResolutionSpec:
    """Metadata for one skill-grid resolution condition."""

    slug: str
    label: str
    num_sigma_grid: int
    num_lambda_grid: int
    description: str


GRID_RESOLUTION_SPECS = (
    GridResolutionSpec(
        slug="grid_011x011",
        label="11 x 11",
        num_sigma_grid=11,
        num_lambda_grid=11,
        description="Coarse skill grid for discretization-sensitivity checking",
    ),
    GridResolutionSpec(
        slug="grid_021x021",
        label="21 x 21",
        num_sigma_grid=21,
        num_lambda_grid=21,
        description="Default skill grid used by the main H-JEEDS experiments",
    ),
    GridResolutionSpec(
        slug="grid_041x041",
        label="41 x 41",
        num_sigma_grid=41,
        num_lambda_grid=41,
        description="Finer skill grid for discretization-sensitivity checking",
    ),
)


@dataclass(frozen=True)
class GridResolutionScenario:
    """One concrete grid-resolution scenario."""

    scenario_index: int
    config: base_experiment.ExperimentConfig
    grid_resolution: GridResolutionSpec

    @property
    def scenario_slug(self) -> str:
        """Return a stable combined scenario identifier."""

        return self.grid_resolution.slug

    @property
    def scenario_output_dir(self) -> Path:
        """Return the directory where this scenario's normal artifacts will live."""

        return self.config.output_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for the grid-resolution ablation."""

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
        help="Root output directory for the grid-resolution sweep.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the grid-resolution workload and stop before simulation/inference.",
    )
    parser.add_argument(
        "--aggregate-results",
        action="store_true",
        help="Collect already-computed scenario folders into root combined CSVs.",
    )
    add_plotting_cli_arguments(parser)
    return parser.parse_args(argv)


def _seed_values_label(seed_values: Sequence[int]) -> str:
    """Return a readable seed description for dry-run output."""

    if len(seed_values) <= 10:
        return str(tuple(seed_values))
    return f"{len(seed_values)} seeds ({seed_values[0]} through {seed_values[-1]})"


def scenario_index_from_environment() -> int | None:
    """Return a Slurm-style scenario index from the environment when present."""

    raw_value = os.environ.get("SCENARIO_INDEX") or os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw_value is None:
        return None
    try:
        scenario_index = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Scenario index must be an integer. Received {raw_value}.") from exc
    if scenario_index < 0:
        raise ValueError(f"Scenario index must be nonnegative. Received {scenario_index}.")
    return scenario_index


def build_config_for_scenario(
    args: argparse.Namespace,
    grid_resolution: GridResolutionSpec,
) -> base_experiment.ExperimentConfig:
    """Build one base H-JEEDS config for a grid-resolution scenario."""

    output_dir = Path(args.output_dir) / grid_resolution.slug
    base_args = argparse.Namespace(
        seed=args.seed,
        num_seeds=args.num_seeds,
        count_buckets=",".join(str(bucket) for bucket in DEFAULT_COUNT_BUCKETS),
        agents_per_bucket=DEFAULT_AGENTS_PER_BUCKET,
        num_agents=len(DEFAULT_COUNT_BUCKETS) * DEFAULT_AGENTS_PER_BUCKET,
        delta=base_experiment.DEFAULT_DELTA,
        num_sigma_grid=grid_resolution.num_sigma_grid,
        num_lambda_grid=grid_resolution.num_lambda_grid,
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


def build_scenarios(args: argparse.Namespace) -> tuple[GridResolutionScenario, ...]:
    """Build the grid-resolution scenarios."""

    scenarios: list[GridResolutionScenario] = []
    for grid_resolution in GRID_RESOLUTION_SPECS:
        scenarios.append(
            GridResolutionScenario(
                scenario_index=len(scenarios),
                config=build_config_for_scenario(args, grid_resolution),
                grid_resolution=grid_resolution,
            )
        )
    return tuple(scenarios)


def grid_resolution_metadata_row(grid_resolution: GridResolutionSpec) -> dict[str, Any]:
    """Return CSV metadata for one grid-resolution condition."""

    return {
        "grid_resolution_slug": grid_resolution.slug,
        "grid_resolution_label": grid_resolution.label,
        "num_sigma_grid": grid_resolution.num_sigma_grid,
        "num_lambda_grid": grid_resolution.num_lambda_grid,
        "num_grid_cells": grid_resolution.num_sigma_grid * grid_resolution.num_lambda_grid,
        "grid_resolution_description": grid_resolution.description,
    }


def scenario_metadata_row(scenario: GridResolutionScenario) -> dict[str, Any]:
    """Return path metadata for one concrete scenario."""

    return {
        "scenario_index": scenario.scenario_index,
        "scenario_slug": scenario.scenario_slug,
        "scenario_output_dir": str(scenario.scenario_output_dir),
        "scenario_error_plot": str(scenario.scenario_output_dir / base_experiment.ERROR_PLOT_FILENAME),
    }


def scenario_prefix_row(scenario: GridResolutionScenario) -> dict[str, Any]:
    """Return all provenance columns for one scenario."""

    return {
        **grid_resolution_metadata_row(scenario.grid_resolution),
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


def _as_float(value: Any) -> float | None:
    """Convert CSV-ish values to floats, treating blanks as missing."""

    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_single_scenario(
    scenario: GridResolutionScenario,
    *,
    include_raw_rationality_error: bool = False,
) -> None:
    """Run one grid-resolution scenario."""

    config = scenario.config
    print(
        "[grid-resolution] "
        f"Running scenario {scenario.scenario_index}: {scenario.scenario_slug} "
        f"({config.num_sigma_grid} x {config.num_lambda_grid} grid)",
        flush=True,
    )

    seed_results: list[base_experiment.SeedResult] = []
    for seed_index, seed in enumerate(config.seed_values, start=1):
        print(
            "[grid-resolution] "
            f"{scenario.scenario_slug}: seed {seed_index}/{config.num_seeds}: {seed}",
            flush=True,
        )
        seed_results.append(base_experiment.run_single_seed(config, seed))

    output_paths = base_experiment.planned_output_paths(config.output_dir)
    all_agent_results = [
        result
        for seed_result in seed_results
        for result in seed_result.agent_results
    ]
    summary_by_bucket_rows, summary_overall_rows = (
        base_experiment.aggregate_results_across_seeds(seed_results)
    )

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
    print(
        "[grid-resolution] "
        f"Wrote scenario results to {scenario.scenario_output_dir.resolve()}",
        flush=True,
    )


def _plot_lowest_bucket_metric(
    output_path: Path,
    rows: Sequence[dict[str, Any]],
    scenarios: Sequence[GridResolutionScenario],
    metric_name: str,
    title: str,
    ylabel: str,
    missing_message: str,
) -> None:
    """Plot one lowest-bucket metric across grid resolutions."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    bucket_values = sorted(
        {
            int(float(row["count_bucket"]))
            for row in rows
            if str(row.get("metric", "")) == metric_name and _as_float(row.get("count_bucket")) is not None
        }
    )
    if not bucket_values:
        return

    selected_bucket = bucket_values[0]
    grid_slugs = [scenario.grid_resolution.slug for scenario in scenarios]
    grid_labels = {
        scenario.grid_resolution.slug: scenario.grid_resolution.label
        for scenario in scenarios
    }
    x_positions = {grid_slug: index for index, grid_slug in enumerate(grid_slugs)}

    parsed_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if str(row.get("metric", "")) != metric_name:
            continue
        if int(float(row["count_bucket"])) != selected_bucket:
            continue
        mean = _as_float(row.get("mean"))
        if mean is None:
            continue
        key = (
            str(row["grid_resolution_slug"]),
            str(row["method"]),
        )
        parsed_rows[key] = row

    figure, axis = plt.subplots(
        1,
        1,
        figsize=(6.2, 4.2),
        constrained_layout=True,
    )
    methods = sorted(
        {
            str(row.get("method", ""))
            for row in rows
            if str(row.get("metric", "")) == metric_name and int(float(row["count_bucket"])) == selected_bucket
        },
        key=lambda method: (METHOD_ORDER.get(method, len(METHOD_ORDER)), method),
    )

    drew_rows = False
    for method in methods:
        x_values: list[int] = []
        y_values: list[float] = []
        lower_errors: list[float] = []
        upper_errors: list[float] = []

        for grid_slug in grid_slugs:
            row = parsed_rows.get((grid_slug, method))
            if row is None:
                continue
            mean = _as_float(row.get("mean"))
            if mean is None:
                continue
            ci_lower = _as_float(row.get("ci_lower"))
            ci_upper = _as_float(row.get("ci_upper"))
            x_values.append(x_positions[grid_slug])
            y_values.append(mean)
            lower_errors.append(max(0.0, mean - ci_lower) if ci_lower is not None else 0.0)
            upper_errors.append(max(0.0, ci_upper - mean) if ci_upper is not None else 0.0)

        if x_values:
            drew_rows = True
            axis.errorbar(
                x_values,
                y_values,
                yerr=np.array([lower_errors, upper_errors], dtype=float),
                color=method_color(method),
                marker=method_marker(method),
                capsize=4,
                linewidth=2,
                label=method_label(method),
            )

    axis.set_title(f"{title} at lowest observation bucket ({selected_bucket})")
    axis.set_xlabel("Grid resolution")
    axis.set_ylabel(ylabel)
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axis.set_xticks(list(x_positions.values()))
    axis.set_xticklabels([grid_labels[grid_slug] for grid_slug in grid_slugs])

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


def plot_lowest_bucket_comparisons(
    output_dir: Path,
    summary_by_bucket_rows: Sequence[dict[str, Any]],
    scenarios: Sequence[GridResolutionScenario],
    *,
    include_raw_rationality_error: bool = False,
) -> None:
    """Write one root comparison plot per metric."""

    for metric_name, title, ylabel, missing_message in error_metric_panels(include_raw_rationality_error):
        _plot_lowest_bucket_metric(
            output_dir / LOWEST_BUCKET_PLOT_TEMPLATE.format(metric=metric_name),
            summary_by_bucket_rows,
            scenarios,
            metric_name,
            title,
            ylabel,
            missing_message,
        )


def aggregate_existing_results(
    scenarios: Sequence[GridResolutionScenario],
    output_dir: Path,
    *,
    include_raw_rationality_error: bool = False,
    regenerate_scenario_plots: bool = False,
) -> None:
    """Collect already-computed grid-resolution scenario folders."""

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

    combined_prefix_header = GRID_RESOLUTION_METADATA_HEADER + SCENARIO_METADATA_HEADER
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
    plot_lowest_bucket_comparisons(
        output_dir,
        all_bucket_rows,
        scenarios,
        include_raw_rationality_error=include_raw_rationality_error,
    )

    print(f"[grid-resolution] Aggregated results into {output_dir.resolve()}", flush=True)


def print_dry_run_summary(
    scenarios: Sequence[GridResolutionScenario],
    output_dir: Path,
    *,
    include_raw_rationality_error: bool = False,
) -> None:
    """Report the grid-resolution workload without running inference."""

    grid_labels = [spec.label for spec in GRID_RESOLUTION_SPECS]

    print("=== DRY RUN: Grid Resolution Sensitivity ===")
    print("No simulation or inference functions will be executed.")
    print()
    print(f"Grid resolutions: {', '.join(grid_labels)}")
    print(f"Agents per bucket: {DEFAULT_AGENTS_PER_BUCKET}")
    print(f"Total scenarios: {len(scenarios)}")
    if scenarios:
        print(f"Seeds per scenario: {_seed_values_label(scenarios[0].config.seed_values)}")
        print(f"Count buckets: {scenarios[0].config.count_buckets}")
    print(f"Root output directory: {output_dir.resolve()}")
    print()
    print("Scenario folders:")
    for scenario in scenarios:
        spec = scenario.grid_resolution
        total_cells = spec.num_sigma_grid * spec.num_lambda_grid
        print(
            f"  - [{scenario.scenario_index:02d}] {scenario.scenario_slug}/ "
            f"({spec.num_sigma_grid} x {spec.num_lambda_grid} = {total_cells} cells) "
            f"-> {scenario.scenario_output_dir}"
        )
    print()
    print("Planned root combined artifacts:")
    print(f"  - {output_dir / SCENARIOS_FILENAME}")
    print(f"  - {output_dir / COMBINED_AGENT_LEVEL_FILENAME}")
    print(f"  - {output_dir / COMBINED_SUMMARY_BY_BUCKET_FILENAME}")
    print(f"  - {output_dir / COMBINED_SUMMARY_OVERALL_FILENAME}")
    for metric_name, _title, _ylabel, _missing_message in error_metric_panels(include_raw_rationality_error):
        print(f"  - {output_dir / LOWEST_BUCKET_PLOT_TEMPLATE.format(metric=metric_name)}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the grid-resolution sensitivity sweep."""

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

    if args.aggregate_results or args.plot_only:
        aggregate_existing_results(
            scenarios,
            output_dir,
            include_raw_rationality_error=args.include_raw_rationality_error,
            regenerate_scenario_plots=args.plot_only,
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
