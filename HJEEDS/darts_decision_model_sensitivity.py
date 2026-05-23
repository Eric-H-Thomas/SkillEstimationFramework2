# This file has been fully edited by a human researcher as of 05/22/26 at 6:01 PM MDT.
"""Run H-JEEDS true decision-model sensitivity experiments.

This runner varies the simulator's true decision-making model while keeping
the H-JEEDS estimator's likelihood fixed to the default softmax assumption.
The default sweep is:

- true decision model: softmax, rational, flip, deceptive
- agents per bucket: 1, 2, 5, 10, 25
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
from HJEEDS.artifacts import ERROR_METRIC_PANELS, METHOD_ORDER
from HJEEDS.decision_models import (
    DECISION_MODEL_SPECS,
    DecisionModelSpec,
    decision_model_folder_slug,
    decision_model_metadata_row,
)


DEFAULT_OUTPUT_DIR = Path("HJEEDS/results/hierarchical_darts_decision_model_sensitivity")
DEFAULT_AGENTS_PER_BUCKET_VALUES = (1, 2, 5, 10, 25)
DEFAULT_COUNT_BUCKETS = base_experiment.DEFAULT_COUNT_BUCKETS

SCENARIOS_FILENAME = "decision_model_sensitivity_scenarios.csv"
COMBINED_AGENT_LEVEL_FILENAME = "decision_model_sensitivity_agent_level_results.csv"
COMBINED_SUMMARY_BY_BUCKET_FILENAME = "decision_model_sensitivity_summary_by_bucket.csv"
COMBINED_SUMMARY_OVERALL_FILENAME = "decision_model_sensitivity_summary_overall.csv"
LOWEST_BUCKET_PLOT_TEMPLATE = "decision_model_lowest_bucket_{metric}.png"

DECISION_MODEL_METADATA_HEADER = [
    "decision_model_slug",
    "decision_model_label",
    "decision_model_description",
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
class DecisionModelScenario:
    """One concrete true-decision-model x agents-per-bucket scenario."""

    scenario_index: int
    config: base_experiment.ExperimentConfig
    decision_model: DecisionModelSpec

    @property
    def scenario_slug(self) -> str:
        """Return a stable combined scenario identifier."""

        return f"{decision_model_folder_slug(self.decision_model.slug)}__{_agents_per_bucket_slug(self.config.agents_per_bucket)}"

    @property
    def scenario_output_dir(self) -> Path:
        """Return the directory where this scenario's normal artifacts will live."""

        return self.config.output_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for the decision-model ablation."""

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
        help="Root output directory for the decision-model sweep.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the decision-model workload and stop before simulation/inference.",
    )
    parser.add_argument(
        "--aggregate-results",
        action="store_true",
        help="Collect already-computed scenario folders into root combined CSVs.",
    )
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
    decision_model: DecisionModelSpec,
    agents_per_bucket: int,
) -> base_experiment.ExperimentConfig:
    """Build one base H-JEEDS config for a true-decision-model scenario."""

    output_dir = (
        Path(args.output_dir)
        / decision_model_folder_slug(decision_model.slug)
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
        true_decision_model_slug=decision_model.slug,
    )
    return base_experiment.build_config_from_args(base_args)


def build_scenarios(
    args: argparse.Namespace,
    agents_per_bucket_values: Sequence[int] = DEFAULT_AGENTS_PER_BUCKET_VALUES,
) -> tuple[DecisionModelScenario, ...]:
    """Build the decision-model x agents-per-bucket scenarios."""

    scenarios: list[DecisionModelScenario] = []
    for decision_model in DECISION_MODEL_SPECS:
        for agents_per_bucket in agents_per_bucket_values:
            scenarios.append(
                DecisionModelScenario(
                    scenario_index=len(scenarios),
                    config=build_config_for_scenario(args, decision_model, agents_per_bucket),
                    decision_model=decision_model,
                )
            )
    return tuple(scenarios)


def agents_per_bucket_metadata_row(config: base_experiment.ExperimentConfig) -> dict[str, Any]:
    """Return metadata for one agents-per-bucket value."""

    return {
        "agents_per_bucket_slug": _agents_per_bucket_slug(config.agents_per_bucket),
        "agents_per_bucket": config.agents_per_bucket,
        "scenario_num_agents": config.num_agents,
        "count_buckets": _count_bucket_label(config.count_buckets),
    }


def scenario_metadata_row(scenario: DecisionModelScenario) -> dict[str, Any]:
    """Return path metadata for one concrete scenario."""

    return {
        "scenario_index": scenario.scenario_index,
        "scenario_slug": scenario.scenario_slug,
        "scenario_output_dir": str(scenario.scenario_output_dir),
        "scenario_error_plot": str(scenario.scenario_output_dir / base_experiment.ERROR_PLOT_FILENAME),
    }


def scenario_prefix_row(scenario: DecisionModelScenario) -> dict[str, Any]:
    """Return all provenance columns for one scenario."""

    return {
        **decision_model_metadata_row(scenario.decision_model.slug),
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


def _as_float(value: Any) -> float | None:
    """Convert CSV-ish values to floats, treating blanks as missing."""

    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_single_scenario(scenario: DecisionModelScenario) -> None:
    """Run one decision-model x agents-per-bucket scenario."""

    config = scenario.config
    print(
        "[decision-model] "
        f"Running scenario {scenario.scenario_index}: {scenario.scenario_slug} "
        f"({config.num_agents} agents/seed)",
        flush=True,
    )

    seed_results: list[base_experiment.SeedResult] = []
    for seed_index, seed in enumerate(config.seed_values, start=1):
        print(
            "[decision-model] "
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
    base_experiment.plot_error_by_bucket(output_paths["error_plot"], summary_by_bucket_rows)
    print(
        "[decision-model] "
        f"Wrote scenario results to {scenario.scenario_output_dir.resolve()}",
        flush=True,
    )


def _plot_lowest_bucket_metric(
    output_path: Path,
    rows: Sequence[dict[str, Any]],
    scenarios: Sequence[DecisionModelScenario],
    metric_name: str,
    title: str,
    ylabel: str,
    missing_message: str,
) -> None:
    """Plot one lowest-bucket metric across agents-per-bucket values and decision models."""

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
    decision_model_slugs = [model.slug for model in DECISION_MODEL_SPECS]
    decision_model_labels = {model.slug: model.label for model in DECISION_MODEL_SPECS}
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
            str(row["decision_model_slug"]),
            int(row["agents_per_bucket"]),
            str(row["method"]),
        )
        parsed_rows[key] = row

    figure, axes = plt.subplots(
        1,
        len(decision_model_slugs),
        figsize=(4.2 * len(decision_model_slugs), 4.2),
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

    for axis_index, (axis, decision_model_slug) in enumerate(zip(axes_row, decision_model_slugs)):
        axis.set_title(decision_model_labels[decision_model_slug])
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
                row = parsed_rows.get((decision_model_slug, agents_per_bucket, method))
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

    figure.suptitle(f"{title} at lowest observation bucket ({selected_bucket})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_lowest_bucket_comparisons(
    output_dir: Path,
    summary_by_bucket_rows: Sequence[dict[str, Any]],
    scenarios: Sequence[DecisionModelScenario],
) -> None:
    """Write one root comparison plot per metric."""

    for metric_name, title, ylabel, missing_message in ERROR_METRIC_PANELS:
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
    scenarios: Sequence[DecisionModelScenario],
    output_dir: Path,
) -> None:
    """Collect already-computed decision-model scenario folders."""

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

        scenario_rows.append(prefix)
        all_agent_rows.extend({**prefix, **row} for row in agent_rows)
        all_bucket_rows.extend({**prefix, **row} for row in bucket_rows)
        all_overall_rows.extend({**prefix, **row} for row in overall_rows)

    combined_prefix_header = (
        DECISION_MODEL_METADATA_HEADER
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
    plot_lowest_bucket_comparisons(output_dir, all_bucket_rows, scenarios)

    print(f"[decision-model] Aggregated results into {output_dir.resolve()}", flush=True)


def print_dry_run_summary(
    scenarios: Sequence[DecisionModelScenario],
    output_dir: Path,
) -> None:
    """Report the decision-model workload without running inference."""

    decision_model_slugs = [model.slug for model in DECISION_MODEL_SPECS]
    agents_values = sorted({scenario.config.agents_per_bucket for scenario in scenarios})

    print("=== DRY RUN: True Decision Model x Agents Per Bucket Sensitivity ===")
    print("No simulation or inference functions will be executed.")
    print()
    print(f"True decision models: {', '.join(decision_model_slugs)}")
    print(f"Estimator decision model: softmax")
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
    print("Planned root combined artifacts:")
    print(f"  - {output_dir / SCENARIOS_FILENAME}")
    print(f"  - {output_dir / COMBINED_AGENT_LEVEL_FILENAME}")
    print(f"  - {output_dir / COMBINED_SUMMARY_BY_BUCKET_FILENAME}")
    print(f"  - {output_dir / COMBINED_SUMMARY_OVERALL_FILENAME}")
    for metric_name, _title, _ylabel, _missing_message in ERROR_METRIC_PANELS:
        print(f"  - {output_dir / LOWEST_BUCKET_PLOT_TEMPLATE.format(metric=metric_name)}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the decision-model sensitivity sweep."""

    args = parse_args(argv)
    scenario_index = scenario_index_from_environment()
    if scenario_index is not None and args.aggregate_results:
        raise ValueError("Scenario-index environment mode and --aggregate-results are mutually exclusive.")

    scenarios = build_scenarios(args)
    output_dir = Path(args.output_dir)

    if args.dry_run:
        print_dry_run_summary(scenarios, output_dir)
        return 0

    if args.aggregate_results:
        aggregate_existing_results(scenarios, output_dir)
        return 0

    if scenario_index is not None:
        if scenario_index >= len(scenarios):
            raise ValueError(
                f"scenario_index must be between 0 and {len(scenarios) - 1}. "
                f"Received {scenario_index}."
            )
        run_single_scenario(scenarios[scenario_index])
        return 0

    for scenario in scenarios:
        run_single_scenario(scenario)
    aggregate_existing_results(scenarios, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
