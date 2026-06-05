# This file has been fully edited by a human researcher as of 05/22/26 at 6:01 PM MDT.
"""Run H-JEEDS agents-per-bucket ablations across true population shapes.

This script varies the simulator's true population shape while keeping the
H-JEEDS estimator's Gaussian population model and default hyperpriors fixed.
That isolates how sensitive the method is to population-shape misspecification.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, replace
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
from HJEEDS.population_shapes import (
    POPULATION_SHAPE_SPECS,
    PopulationShapeSpec,
    population_shape_folder_slug,
    population_shape_metadata_row,
)


DEFAULT_OUTPUT_DIR = Path("HJEEDS/results/hierarchical_darts_population_shape_sensitivity")
DEFAULT_AGENTS_PER_BUCKET_VALUES = (1, 2, 5, 10, 25)
DEFAULT_COUNT_BUCKETS = base_experiment.DEFAULT_COUNT_BUCKETS

SCENARIOS_FILENAME = "population_shape_sensitivity_scenarios.csv"
COMBINED_AGENT_LEVEL_FILENAME = "population_shape_sensitivity_agent_level_results.csv"
COMBINED_SUMMARY_BY_BUCKET_FILENAME = "population_shape_sensitivity_summary_by_bucket.csv"
COMBINED_SUMMARY_OVERALL_FILENAME = "population_shape_sensitivity_summary_overall.csv"
LOWEST_BUCKET_PLOT_TEMPLATE = "population_shape_lowest_bucket_{metric}.png"

POPULATION_SHAPE_METADATA_HEADER = [
    "population_shape_slug",
    "population_shape_label",
    "population_shape_description",
]

AGENTS_PER_BUCKET_METADATA_HEADER = [
    "agents_per_bucket_slug",
    "agents_per_bucket",
    "scenario_num_agents",
    "count_buckets",
]

SCENARIO_METADATA_HEADER = [
    "scenario_index",
    "scenario_slug",
    "scenario_output_dir",
    "scenario_error_plot",
]


@dataclass(frozen=True)
class PopulationShapeScenario:
    """One concrete population-shape x agents-per-bucket scenario."""

    scenario_index: int
    config: base_experiment.ExperimentConfig
    population_shape: PopulationShapeSpec

    @property
    def scenario_slug(self) -> str:
        """Return a stable combined scenario identifier."""

        return f"{population_shape_folder_slug(self.population_shape.slug)}__{_agents_per_bucket_slug(self.config.agents_per_bucket)}"

    @property
    def scenario_output_dir(self) -> Path:
        """Return the directory where this scenario's normal artifacts live."""

        return self.config.output_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for the population-shape ablation."""

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
        help="Root output directory for the population-shape sweep.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the population-shape workload and stop before simulation/inference.",
    )
    parser.add_argument(
        "--aggregate-results",
        action="store_true",
        help="Collect already-computed scenario folders into root combined CSVs and plots.",
    )
    add_plotting_cli_arguments(parser)

    return parser.parse_args(argv)


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


def build_config_for_scenario(
    args: argparse.Namespace,
    population_shape: PopulationShapeSpec,
    agents_per_bucket: int,
) -> base_experiment.ExperimentConfig:
    """Build one base H-JEEDS config for a true-population-shape scenario."""

    output_dir = (
        Path(args.output_dir)
        / population_shape_folder_slug(population_shape.slug)
        / _agents_per_bucket_slug(agents_per_bucket)
    )
    base_args = argparse.Namespace(
        seed=args.seed,
        num_seeds=args.num_seeds,
        count_buckets=",".join(str(bucket) for bucket in DEFAULT_COUNT_BUCKETS),
        agents_per_bucket=agents_per_bucket,
        num_agents=len(DEFAULT_COUNT_BUCKETS) * agents_per_bucket,
        delta=base_experiment.DEFAULT_DELTA,
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
    config = base_experiment.build_config_from_args(base_args)
    true_population = replace(config.true_population, population_shape_slug=population_shape.slug)
    return replace(config, true_population=true_population)


def build_scenarios(
    args: argparse.Namespace,
    agents_per_bucket_values: Sequence[int],
) -> tuple[PopulationShapeScenario, ...]:
    """Return the flat scenario list used by local runs and Slurm arrays."""

    scenarios: list[PopulationShapeScenario] = []
    for population_shape in POPULATION_SHAPE_SPECS:
        for agents_per_bucket in agents_per_bucket_values:
            scenarios.append(
                PopulationShapeScenario(
                    scenario_index=len(scenarios),
                    config=build_config_for_scenario(args, population_shape, agents_per_bucket),
                    population_shape=population_shape,
                )
            )
    return tuple(scenarios)


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


def agents_per_bucket_metadata_row(config: base_experiment.ExperimentConfig) -> dict[str, Any]:
    """Return metadata for one agents-per-bucket value."""

    return {
        "agents_per_bucket_slug": _agents_per_bucket_slug(config.agents_per_bucket),
        "agents_per_bucket": config.agents_per_bucket,
        "scenario_num_agents": config.num_agents,
        "count_buckets": _count_bucket_label(config.count_buckets),
    }


def scenario_metadata_row(scenario: PopulationShapeScenario) -> dict[str, Any]:
    """Return path metadata for one concrete scenario."""

    return {
        "scenario_index": scenario.scenario_index,
        "scenario_slug": scenario.scenario_slug,
        "scenario_output_dir": str(scenario.scenario_output_dir),
        "scenario_error_plot": str(scenario.scenario_output_dir / base_experiment.ERROR_PLOT_FILENAME),
    }


def scenario_prefix_row(scenario: PopulationShapeScenario) -> dict[str, Any]:
    """Return all provenance columns for one scenario."""

    return {
        **population_shape_metadata_row(scenario.population_shape.slug),
        **agents_per_bucket_metadata_row(scenario.config),
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
    scenario: PopulationShapeScenario,
    *,
    include_raw_rationality_error: bool = False,
) -> None:
    """Run one population-shape x agents-per-bucket scenario."""

    print(
        "[population-shape] "
        f"Running scenario {scenario.scenario_index}: {scenario.scenario_slug} "
        f"({scenario.config.num_agents} agents/seed)",
        flush=True,
    )
    run_hjeeds_config(
        scenario.config,
        log_prefix=f"[population-shape] {scenario.scenario_slug}:",
        include_raw_rationality_error=include_raw_rationality_error,
    )
    print(
        "[population-shape] "
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


def _plot_lowest_bucket_metric(
    output_path: Path,
    rows: Sequence[dict[str, Any]],
    scenarios: Sequence[PopulationShapeScenario],
    metric_name: str,
    title: str,
    ylabel: str,
    missing_message: str,
) -> None:
    """Plot one lowest-bucket metric across agents-per-bucket values and shapes."""

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
    shape_slugs = [shape.slug for shape in POPULATION_SHAPE_SPECS]
    shape_labels = {shape.slug: shape.label for shape in POPULATION_SHAPE_SPECS}
    agents_values = sorted({scenario.config.agents_per_bucket for scenario in scenarios})
    x_positions = {agents_per_bucket: index for index, agents_per_bucket in enumerate(agents_values)}

    parsed_rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        if str(row.get("metric", "")) != metric_name:
            continue
        if int(float(row["count_bucket"])) != selected_bucket:
            continue
        mean = _as_float(row.get("mean"))
        if mean is None:
            continue
        key = (
            str(row["population_shape_slug"]),
            int(row["agents_per_bucket"]),
            str(row["method"]),
        )
        parsed_rows[key] = row

    figure, axes = plt.subplots(
        1,
        len(shape_slugs),
        figsize=(4.2 * len(shape_slugs), 4.2),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes_row = axes[0]
    methods = sorted(
        {
            str(row.get("method", ""))
            for row in rows
            if str(row.get("metric", "")) == metric_name and int(float(row["count_bucket"])) == selected_bucket
        },
        key=lambda method: (METHOD_ORDER.get(method, len(METHOD_ORDER)), method),
    )

    for axis_index, (axis, shape_slug) in enumerate(zip(axes_row, shape_slugs)):
        axis.set_title(shape_labels[shape_slug])
        axis.set_xlabel("Agents per bucket")
        if axis_index == 0:
            axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        axis.set_xticks(list(x_positions.values()))
        axis.set_xticklabels([str(value) for value in agents_values])

        drew_rows = False
        for method in methods:
            x_values: list[int] = []
            y_values: list[float] = []
            lower_errors: list[float] = []
            upper_errors: list[float] = []

            for agents_per_bucket in agents_values:
                row = parsed_rows.get((shape_slug, agents_per_bucket, method))
                if row is None:
                    continue
                mean = _as_float(row.get("mean"))
                if mean is None:
                    continue
                ci_lower = _as_float(row.get("ci_lower"))
                ci_upper = _as_float(row.get("ci_upper"))
                x_values.append(x_positions[agents_per_bucket])
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

    figure.suptitle(f"{title} at lowest observation bucket ({selected_bucket})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_lowest_bucket_comparisons(
    output_dir: Path,
    summary_by_bucket_rows: Sequence[dict[str, Any]],
    scenarios: Sequence[PopulationShapeScenario],
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
    scenarios: Sequence[PopulationShapeScenario],
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
        POPULATION_SHAPE_METADATA_HEADER
        + AGENTS_PER_BUCKET_METADATA_HEADER
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
    plot_lowest_bucket_comparisons(
        output_dir,
        all_bucket_rows,
        scenarios,
        include_raw_rationality_error=include_raw_rationality_error,
    )

    print(f"[population-shape] Aggregated results into {output_dir.resolve()}", flush=True)


def print_dry_run_summary(
    scenarios: Sequence[PopulationShapeScenario],
    output_dir: Path,
    *,
    include_raw_rationality_error: bool = False,
) -> None:
    """Report the population-shape workload without running inference."""

    shape_slugs = [shape.slug for shape in POPULATION_SHAPE_SPECS]
    agents_values = sorted({scenario.config.agents_per_bucket for scenario in scenarios})

    print("=== DRY RUN: Population Shape x Agents Per Bucket Sensitivity ===")
    print("No simulation or inference functions will be executed.")
    print()
    print(f"Population shapes: {', '.join(shape_slugs)}")
    print(f"Agents-per-bucket values: {agents_values}")
    print(f"Total scenarios: {len(scenarios)}")
    if scenarios:
        print(f"Seeds per scenario: {_seed_values_label(scenarios[0].config.seed_values)}")
        print(f"Count buckets: {scenarios[0].config.count_buckets}")
    print(f"Root output directory: {output_dir.resolve()}")
    print()
    print("Scenario folders:")
    for scenario in scenarios:
        print(f"  - [{scenario.scenario_index:02d}] {scenario.scenario_slug}/ -> {scenario.scenario_output_dir}")
    print()
    print("Root combined artifacts:")
    print(f"  - {output_dir / SCENARIOS_FILENAME}")
    print(f"  - {output_dir / COMBINED_AGENT_LEVEL_FILENAME}")
    print(f"  - {output_dir / COMBINED_SUMMARY_BY_BUCKET_FILENAME}")
    print(f"  - {output_dir / COMBINED_SUMMARY_OVERALL_FILENAME}")
    for metric_name, _title, _ylabel, _missing_message in error_metric_panels(include_raw_rationality_error):
        print(f"  - {output_dir / LOWEST_BUCKET_PLOT_TEMPLATE.format(metric=metric_name)}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the population-shape sensitivity sweep."""

    args = parse_args(argv)
    scenario_index = scenario_index_from_environment()
    if scenario_index is not None and (args.aggregate_results or args.plot_only):
        raise ValueError("Scenario-index environment mode cannot be combined with --aggregate-results or --plot-only.")

    scenarios = build_scenarios(args, DEFAULT_AGENTS_PER_BUCKET_VALUES)
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
