# This file has been fully edited by a human researcher as of 05/22/26 at 9:52 AM MDT.
"""Scaffold the H-JEEDS true decision-model sensitivity ablation.

This runner will vary the simulator's true decision-making model while keeping
the H-JEEDS estimator's likelihood fixed to the default softmax assumption.
The planned default sweep is:

- true decision model: softmax, rational, flip, deceptive
- agents per bucket: 1, 2, 5, 10, 25

Execution and aggregation are intentionally TODO stubs for now. The dry-run
path is implemented so we can review the planned workload before filling in the
simulator changes.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


# Ensure the repository root is importable when this file is executed directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HJEEDS import darts_hierarchical_vs_jeeds as base_experiment
from HJEEDS.decision_models import (
    DECISION_MODEL_SPECS,
    DecisionModelSpec,
    decision_model_folder_slug,
)


DEFAULT_OUTPUT_DIR = Path("HJEEDS/results/hierarchical_darts_decision_model_sensitivity")
DEFAULT_AGENTS_PER_BUCKET_VALUES = (1, 2, 5, 10, 25)
DEFAULT_COUNT_BUCKETS = base_experiment.DEFAULT_COUNT_BUCKETS

SCENARIOS_FILENAME = "decision_model_sensitivity_scenarios.csv"
COMBINED_AGENT_LEVEL_FILENAME = "decision_model_sensitivity_agent_level_results.csv"
COMBINED_SUMMARY_BY_BUCKET_FILENAME = "decision_model_sensitivity_summary_by_bucket.csv"
COMBINED_SUMMARY_OVERALL_FILENAME = "decision_model_sensitivity_summary_overall.csv"


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
    """Parse CLI options for the decision-model ablation scaffold."""

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
        help="Report the planned decision-model workload and stop before simulation/inference.",
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
    """Build the planned decision-model x agents-per-bucket scenarios."""

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


def aggregate_existing_results(
    scenarios: Sequence[DecisionModelScenario],
    output_dir: Path,
) -> None:
    """Collect already-computed decision-model scenario folders."""

    # TODO: Mirror darts_population_shape_sensitivity.aggregate_existing_results
    # TODO: Prefix rows with decision-model and agents-per-bucket metadata
    # TODO: Add compact lowest-bucket comparison plots across decision models
    _ = (scenarios, output_dir)
    raise NotImplementedError("Decision-model sensitivity aggregation is scaffolded but not implemented yet.")


def print_dry_run_summary(
    scenarios: Sequence[DecisionModelScenario],
    output_dir: Path,
) -> None:
    """Report the planned decision-model workload without running inference."""

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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the decision-model sensitivity scaffold."""

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
