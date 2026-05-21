# This file has been fully reviewed by a human researcher as of 05/21/26 at 10:25 AM MT.
"""Scaffold the compact H-JEEDS compound-stress ablation.

This runner is meant to show that H-JEEDS was also tested under a small number
of combined misspecification settings, without turning the paper into a giant
factorial ablation. The planned default sweep is:

- compound stress setting: default, moderate compound stress, strong compound stress
- agents per bucket: 1, 2, 5, 10, 25

Execution and aggregation are intentionally TODO stubs for now. The dry-run
path is implemented so we can review the planned workload before filling in the
simulator and hyperprior wiring.
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
from HJEEDS.decision_models import (
    SOFTMAX_DECISION_MODEL_SLUG,
    FLIP_DECISION_MODEL_SLUG,
    DECEPTIVE_DECISION_MODEL_SLUG,
    decision_model_metadata_row,
)
from HJEEDS.population_shapes import (
    DEFAULT_POPULATION_SHAPE_SLUG,
    OUTLIER_HEAVY_POPULATION_SHAPE_SLUG,
    BIMODAL_POPULATION_SHAPE_SLUG,
    population_shape_metadata_row,
)


DEFAULT_OUTPUT_DIR = Path("HJEEDS/results/hierarchical_darts_compound_stress_sensitivity")
DEFAULT_AGENTS_PER_BUCKET_VALUES = (1, 2, 5, 10, 25)
DEFAULT_COUNT_BUCKETS = base_experiment.DEFAULT_COUNT_BUCKETS

DEFAULT_HYPERPRIOR_CONDITION_SLUG = "default"
MODERATE_HYPERPRIOR_CONDITION_SLUG = "moderate_combined_misspecification"
STRONG_HYPERPRIOR_CONDITION_SLUG = "strong_combined_misspecification"

SCENARIOS_FILENAME = "compound_stress_sensitivity_scenarios.csv"
COMBINED_AGENT_LEVEL_FILENAME = "compound_stress_sensitivity_agent_level_results.csv"
COMBINED_SUMMARY_BY_BUCKET_FILENAME = "compound_stress_sensitivity_summary_by_bucket.csv"
COMBINED_SUMMARY_OVERALL_FILENAME = "compound_stress_sensitivity_summary_overall.csv"


@dataclass(frozen=True)
class CompoundStressSpec:
    """Metadata for one compound-stress condition."""

    slug: str
    label: str
    hyperprior_condition_slug: str
    population_shape_slug: str
    decision_model_slug: str
    true_correlation: float
    description: str


COMPOUND_STRESS_SPECS = (
    CompoundStressSpec(
        slug="default",
        label="Default",
        hyperprior_condition_slug=DEFAULT_HYPERPRIOR_CONDITION_SLUG,
        population_shape_slug=DEFAULT_POPULATION_SHAPE_SLUG,
        decision_model_slug=SOFTMAX_DECISION_MODEL_SLUG,
        true_correlation=-0.5,
        description="Matched default H-JEEDS simulator and estimator assumptions",
    ),
    CompoundStressSpec(
        slug="moderate_compound_stress",
        label="Moderate Compound Stress",
        hyperprior_condition_slug=MODERATE_HYPERPRIOR_CONDITION_SLUG,
        population_shape_slug=OUTLIER_HEAVY_POPULATION_SHAPE_SLUG,
        decision_model_slug=FLIP_DECISION_MODEL_SLUG,
        true_correlation=0.0,
        description="Moderate combined stress across hyperpriors, population shape, policy, and correlation",
    ),
    CompoundStressSpec(
        slug="strong_compound_stress",
        label="Strong Compound Stress",
        hyperprior_condition_slug=STRONG_HYPERPRIOR_CONDITION_SLUG,
        population_shape_slug=BIMODAL_POPULATION_SHAPE_SLUG,
        decision_model_slug=DECEPTIVE_DECISION_MODEL_SLUG,
        true_correlation=0.9,
        description="Strong combined stress with wrong-sign correlation and non-softmax deceptive behavior",
    ),
)


@dataclass(frozen=True)
class CompoundStressScenario:
    """One concrete compound-stress x agents-per-bucket scenario."""

    scenario_index: int
    config: base_experiment.ExperimentConfig
    compound_stress: CompoundStressSpec

    @property
    def scenario_slug(self) -> str:
        """Return a stable combined scenario identifier."""

        return f"compound_stress_{self.compound_stress.slug}__{_agents_per_bucket_slug(self.config.agents_per_bucket)}"

    @property
    def scenario_output_dir(self) -> Path:
        """Return the directory where this scenario's normal artifacts will live."""

        return self.config.output_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for the compound-stress ablation scaffold."""

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
        help="Root output directory for the compound-stress sweep.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the planned compound-stress workload and stop before simulation/inference.",
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


def compound_stress_metadata_row(compound_stress: CompoundStressSpec) -> dict[str, object]:
    """Return CSV metadata for one compound-stress condition."""

    population_metadata = population_shape_metadata_row(compound_stress.population_shape_slug)
    decision_metadata = decision_model_metadata_row(compound_stress.decision_model_slug)
    return {
        "compound_stress_slug": compound_stress.slug,
        "compound_stress_label": compound_stress.label,
        "compound_stress_description": compound_stress.description,
        "hyperprior_condition_slug": compound_stress.hyperprior_condition_slug,
        "true_correlation": compound_stress.true_correlation,
        **population_metadata,
        **decision_metadata,
    }


def build_config_for_scenario(
    args: argparse.Namespace,
    compound_stress: CompoundStressSpec,
    agents_per_bucket: int,
) -> base_experiment.ExperimentConfig:
    """Build one base H-JEEDS config for a compound-stress scenario."""

    # TODO: Apply compound_stress.hyperprior_condition_slug using prior_sensitivity.build_condition_hyperpriors
    # TODO: Thread compound_stress.decision_model_slug into simulator config once decision models are implemented
    # TODO: Confirm whether strong compound stress should use bimodal or outlier-heavy before final experiments
    output_dir = (
        Path(args.output_dir)
        / f"compound_stress_{compound_stress.slug}"
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
    true_population = replace(
        config.true_population,
        population_shape_slug=compound_stress.population_shape_slug,
        correlation=compound_stress.true_correlation,
    )
    return replace(config, true_population=true_population)


def build_scenarios(
    args: argparse.Namespace,
    agents_per_bucket_values: Sequence[int] = DEFAULT_AGENTS_PER_BUCKET_VALUES,
) -> tuple[CompoundStressScenario, ...]:
    """Build the planned compound-stress x agents-per-bucket scenarios."""

    scenarios: list[CompoundStressScenario] = []
    for compound_stress in COMPOUND_STRESS_SPECS:
        for agents_per_bucket in agents_per_bucket_values:
            scenarios.append(
                CompoundStressScenario(
                    scenario_index=len(scenarios),
                    config=build_config_for_scenario(args, compound_stress, agents_per_bucket),
                    compound_stress=compound_stress,
                )
            )
    return tuple(scenarios)


def run_single_scenario(scenario: CompoundStressScenario) -> None:
    """Run one compound-stress x agents-per-bucket scenario."""

    # TODO: Implement after hyperprior, population-shape, decision-model, and correlation runners are stable
    # TODO: Ensure each component of the compound stress appears in scenario metadata and result CSVs
    # TODO: Keep this runner compact so it remains a reviewer-facing stress test rather than an ablation soup
    raise NotImplementedError(
        "Compound-stress sensitivity execution is scaffolded but not implemented yet. "
        f"Requested scenario: {scenario.scenario_slug}."
    )


def aggregate_existing_results(
    scenarios: Sequence[CompoundStressScenario],
    output_dir: Path,
) -> None:
    """Collect already-computed compound-stress scenario folders."""

    # TODO: Mirror darts_population_shape_sensitivity.aggregate_existing_results
    # TODO: Prefix rows with compound-stress metadata and each component's metadata
    # TODO: Add one compact plot or table suitable for the main paper
    _ = (scenarios, output_dir)
    raise NotImplementedError("Compound-stress sensitivity aggregation is scaffolded but not implemented yet.")


def print_dry_run_summary(
    scenarios: Sequence[CompoundStressScenario],
    output_dir: Path,
) -> None:
    """Report the planned compound-stress workload without running inference."""

    stress_labels = [spec.label for spec in COMPOUND_STRESS_SPECS]
    agents_values = sorted({scenario.config.agents_per_bucket for scenario in scenarios})

    print("=== DRY RUN: Compound Stress x Agents Per Bucket Sensitivity ===")
    print("No simulation or inference functions will be executed.")
    print()
    print(f"Compound stress settings: {', '.join(stress_labels)}")
    print(f"Agents-per-bucket values: {agents_values}")
    print(f"Total scenarios: {len(scenarios)}")
    if scenarios:
        print(f"Seeds per scenario: {_seed_values_label(scenarios[0].config.seed_values)}")
        print(f"Count buckets: {scenarios[0].config.count_buckets}")
    print(f"Root output directory: {output_dir.resolve()}")
    print()
    print("Stress components:")
    for compound_stress in COMPOUND_STRESS_SPECS:
        metadata = compound_stress_metadata_row(compound_stress)
        print(
            f"  - {metadata['compound_stress_slug']}: "
            f"hyperprior={metadata['hyperprior_condition_slug']}, "
            f"population_shape={metadata['population_shape_slug']}, "
            f"decision_model={metadata['decision_model_slug']}, "
            f"true_correlation={metadata['true_correlation']}"
        )
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
    """Run the compound-stress sensitivity scaffold."""

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
