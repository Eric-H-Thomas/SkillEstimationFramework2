# This file has been fully edited by a human researcher as of 05/22/26 at 9:52 AM MDT.
"""Scaffold the H-JEEDS true population-correlation sensitivity ablation.

This runner will vary the simulator's true correlation between execution skill
and decision-making skill while keeping the estimator model and default
hyperpriors fixed. The planned default sweep is:

- true population correlation: -0.9, -0.5, 0.0, +0.5, +0.9
- agents per bucket: 1, 2, 5, 10, 25

Execution is implemented for each scenario. Aggregation remains a TODO stub
for now, while the dry-run path lets us review the planned workload before
filling in result collection and plotting.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence


# Ensure the repository root is importable when this file is executed directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HJEEDS import darts_hierarchical_vs_jeeds as base_experiment


DEFAULT_OUTPUT_DIR = Path("HJEEDS/results/hierarchical_darts_true_correlation_sensitivity")
DEFAULT_AGENTS_PER_BUCKET_VALUES = (1, 2, 5, 10, 25)
DEFAULT_COUNT_BUCKETS = base_experiment.DEFAULT_COUNT_BUCKETS

SCENARIOS_FILENAME = "true_correlation_sensitivity_scenarios.csv"
COMBINED_AGENT_LEVEL_FILENAME = "true_correlation_sensitivity_agent_level_results.csv"
COMBINED_SUMMARY_BY_BUCKET_FILENAME = "true_correlation_sensitivity_summary_by_bucket.csv"
COMBINED_SUMMARY_OVERALL_FILENAME = "true_correlation_sensitivity_summary_overall.csv"


@dataclass(frozen=True)
class TrueCorrelationSpec:
    """Metadata for one true population-correlation condition."""

    slug: str
    label: str
    correlation: float
    description: str


TRUE_CORRELATION_SPECS = (
    TrueCorrelationSpec(
        slug="r_neg_0_9",
        label="r = -0.9",
        correlation=-0.9,
        description="Strong negative true execution/decision skill correlation",
    ),
    TrueCorrelationSpec(
        slug="r_neg_0_5",
        label="r = -0.5",
        correlation=-0.5,
        description="Matched default true execution/decision skill correlation",
    ),
    TrueCorrelationSpec(
        slug="r_0_0",
        label="r = 0.0",
        correlation=0.0,
        description="No true execution/decision skill correlation",
    ),
    TrueCorrelationSpec(
        slug="r_pos_0_5",
        label="r = +0.5",
        correlation=0.5,
        description="Moderate positive true execution/decision skill correlation",
    ),
    TrueCorrelationSpec(
        slug="r_pos_0_9",
        label="r = +0.9",
        correlation=0.9,
        description="Strong positive true execution/decision skill correlation",
    ),
)


@dataclass(frozen=True)
class TrueCorrelationScenario:
    """One concrete true-correlation x agents-per-bucket scenario."""

    scenario_index: int
    config: base_experiment.ExperimentConfig
    true_correlation: TrueCorrelationSpec

    @property
    def scenario_slug(self) -> str:
        """Return a stable combined scenario identifier."""

        return f"{true_correlation_folder_slug(self.true_correlation)}__{_agents_per_bucket_slug(self.config.agents_per_bucket)}"

    @property
    def scenario_output_dir(self) -> Path:
        """Return the directory where this scenario's normal artifacts will live."""

        return self.config.output_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for the true-correlation ablation scaffold."""

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
        help="Root output directory for the true-correlation sweep.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the planned true-correlation workload and stop before simulation/inference.",
    )
    parser.add_argument(
        "--aggregate-results",
        action="store_true",
        help="TODO: collect already-computed scenario folders into root combined CSVs and plots.",
    )
    return parser.parse_args(argv)


def _agents_per_bucket_slug(agents_per_bucket: int) -> str:
    """Return a stable folder slug for one agents-per-bucket value."""

    return f"agents_per_bucket_{agents_per_bucket:03d}"


def _seed_values_label(seed_values: Sequence[int]) -> str:
    """Return a readable seed description for dry-run output."""

    if len(seed_values) <= 10:
        return str(tuple(seed_values))
    return f"{len(seed_values)} seeds ({seed_values[0]} through {seed_values[-1]})"


def true_correlation_folder_slug(true_correlation: TrueCorrelationSpec) -> str:
    """Return a stable folder slug for one true-correlation condition."""

    return f"true_correlation_{true_correlation.slug}"


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
    true_correlation: TrueCorrelationSpec,
    agents_per_bucket: int,
) -> base_experiment.ExperimentConfig:
    """Build one base H-JEEDS config for a true-correlation scenario."""

    # TODO: Decide whether this ablation should keep hyperprior m_r fixed at default or matched per condition
    # TODO: Current scaffold keeps estimator hyperpriors fixed and varies only the simulator truth
    output_dir = (
        Path(args.output_dir)
        / true_correlation_folder_slug(true_correlation)
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
    true_population = replace(config.true_population, correlation=true_correlation.correlation)
    return replace(config, true_population=true_population)


def build_scenarios(
    args: argparse.Namespace,
    agents_per_bucket_values: Sequence[int] = DEFAULT_AGENTS_PER_BUCKET_VALUES,
) -> tuple[TrueCorrelationScenario, ...]:
    """Build the planned true-correlation x agents-per-bucket scenarios."""

    scenarios: list[TrueCorrelationScenario] = []
    for true_correlation in TRUE_CORRELATION_SPECS:
        for agents_per_bucket in agents_per_bucket_values:
            scenarios.append(
                TrueCorrelationScenario(
                    scenario_index=len(scenarios),
                    config=build_config_for_scenario(args, true_correlation, agents_per_bucket),
                    true_correlation=true_correlation,
                )
            )
    return tuple(scenarios)


def run_single_scenario(scenario: TrueCorrelationScenario) -> None:
    """Run one true-correlation x agents-per-bucket scenario."""

    config = scenario.config
    print(
        "[true-correlation] "
        f"Running scenario {scenario.scenario_index}: {scenario.scenario_slug} "
        f"({config.num_agents} agents/seed)",
        flush=True,
    )

    seed_results: list[base_experiment.SeedResult] = []
    for seed_index, seed in enumerate(config.seed_values, start=1):
        print(
            "[true-correlation] "
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
        "[true-correlation] "
        f"Wrote scenario results to {scenario.scenario_output_dir.resolve()}",
        flush=True,
    )


def aggregate_existing_results(
    scenarios: Sequence[TrueCorrelationScenario],
    output_dir: Path,
) -> None:
    """Collect already-computed true-correlation scenario folders."""

    # TODO: Mirror darts_population_shape_sensitivity.aggregate_existing_results
    # TODO: Prefix rows with true-correlation and agents-per-bucket metadata
    # TODO: Add compact lowest-bucket comparison plots across true correlation values
    _ = (scenarios, output_dir)
    raise NotImplementedError("True-correlation sensitivity aggregation is scaffolded but not implemented yet.")


def print_dry_run_summary(
    scenarios: Sequence[TrueCorrelationScenario],
    output_dir: Path,
) -> None:
    """Report the planned true-correlation workload without running inference."""

    true_correlation_labels = [spec.label for spec in TRUE_CORRELATION_SPECS]
    agents_values = sorted({scenario.config.agents_per_bucket for scenario in scenarios})

    print("=== DRY RUN: True Population Correlation x Agents Per Bucket Sensitivity ===")
    print("No simulation or inference functions will be executed.")
    print()
    print(f"True correlations: {', '.join(true_correlation_labels)}")
    print("Estimator hyperpriors: default")
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the true-correlation sensitivity scaffold."""

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
