"""Run H-JEEDS experiments with observation counts assigned by true skill.

The primary condition gives the best demonstrators the largest observation
buckets. Matched randomized and reversed assignments separate that exposure
mechanism from ordinary count-bucket difficulty.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HJEEDS import darts_hierarchical_vs_jeeds as base_experiment
from HJEEDS.models import AgentTruth, ExperimentConfig
from HJEEDS.sampling import ObservationCountAssigner


DEFAULT_OUTPUT_DIR = Path("HJEEDS/results/hierarchical_darts_exposure_skill_bias")
ASSIGNMENT_DIAGNOSTICS_FILENAME = "assignment_diagnostics.csv"
SCENARIO_MODES = ("quality_aligned", "randomized", "quality_reversed")


@dataclass(frozen=True)
class ExposureScenario:
    mode: str
    config: ExperimentConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=base_experiment.parse_seed_argument, required=True)
    parser.add_argument("--num-seeds", type=int, default=base_experiment.DEFAULT_NUM_SEEDS)
    parser.add_argument("--count-buckets", default=','.join(map(str, base_experiment.DEFAULT_COUNT_BUCKETS)))
    parser.add_argument("--agents-per-bucket", type=int, default=base_experiment.DEFAULT_AGENTS_PER_BUCKET)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--environment", choices=("1d", "2d"), default="1d")
    parser.add_argument("--num-sigma-grid", type=int, default=base_experiment.DEFAULT_NUM_SIGMA_GRID)
    parser.add_argument("--num-lambda-grid", type=int, default=base_experiment.DEFAULT_NUM_LAMBDA_GRID)
    parser.add_argument("--sigma-min", type=float, default=base_experiment.DEFAULT_SIGMA_MIN)
    parser.add_argument("--sigma-max", type=float, default=base_experiment.DEFAULT_SIGMA_MAX)
    parser.add_argument("--lambda-min", type=float, default=base_experiment.DEFAULT_LAMBDA_MIN)
    parser.add_argument("--lambda-max", type=float, default=base_experiment.DEFAULT_LAMBDA_MAX)
    parser.add_argument("--delta", type=float, default=base_experiment.DEFAULT_DELTA)
    parser.add_argument("--min-success-regions", type=int, default=base_experiment.DEFAULT_MIN_SUCCESS_REGIONS)
    parser.add_argument("--max-success-regions", type=int, default=base_experiment.DEFAULT_MAX_SUCCESS_REGIONS)
    parser.add_argument("--min-region-width", type=float, default=base_experiment.DEFAULT_MIN_REGION_WIDTH)
    parser.add_argument("--modes", default=','.join(SCENARIO_MODES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--include-raw-rationality-error", action="store_true")
    return parser


def _base_config(args: argparse.Namespace, output_dir: Path) -> ExperimentConfig:
    count_buckets = tuple(int(piece.strip()) for piece in args.count_buckets.split(",") if piece.strip())
    base_args = argparse.Namespace(
        seed=args.seed,
        num_seeds=args.num_seeds,
        num_agents=len(count_buckets) * args.agents_per_bucket,
        count_buckets=args.count_buckets,
        agents_per_bucket=args.agents_per_bucket,
        delta=args.delta,
        environment=args.environment,
        num_sigma_grid=args.num_sigma_grid,
        num_lambda_grid=args.num_lambda_grid,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        output_dir=str(output_dir),
        min_success_regions=args.min_success_regions,
        max_success_regions=args.max_success_regions,
        min_region_width=args.min_region_width,
        dry_run=args.dry_run,
    )
    return base_experiment.build_config_from_args(base_args)


def _quality_order(truths: list[AgentTruth]) -> np.ndarray:
    """Rank agents by average execution- and decision-quality percentile."""

    sigma_order = np.argsort([truth.sigma_true for truth in truths], kind="stable")
    lambda_order = np.argsort([-truth.lambda_true for truth in truths], kind="stable")
    sigma_rank = np.empty(len(truths), dtype=float)
    lambda_rank = np.empty(len(truths), dtype=float)
    sigma_rank[sigma_order] = np.arange(len(truths), dtype=float)
    lambda_rank[lambda_order] = np.arange(len(truths), dtype=float)
    return np.argsort(-(sigma_rank * -1.0 + lambda_rank * -1.0), kind="stable")


def make_count_assigner(mode: str) -> ObservationCountAssigner:
    """Create a count assignment that is independent of simulation draws."""

    if mode not in SCENARIO_MODES:
        raise ValueError(f"Unknown exposure mode: {mode}")

    def assign(config: ExperimentConfig, truths: list[AgentTruth], seed: int) -> list[int]:
        counts = [bucket for bucket in config.count_buckets for _ in range(config.agents_per_bucket)]
        quality_order = _quality_order(truths)
        if mode == "quality_aligned":
            ordered_agents = quality_order[::-1]
        elif mode == "quality_reversed":
            ordered_agents = quality_order
        else:
            ordered_agents = np.random.default_rng(seed + 104729).permutation(len(truths))

        assigned = [0] * len(truths)
        for agent_index, count in zip(ordered_agents, counts):
            assigned[int(agent_index)] = count
        return assigned

    return assign


def _write_assignment_diagnostics(output_dir: Path, agent_results: Sequence[object]) -> None:
    rows = []
    for result in agent_results:
        result = result
        rows.append(
            {
                "seed": result.seed,
                "agent_id": result.agent_id,
                "count_bucket": result.count_bucket,
                "sigma_true": result.sigma_true,
                "log_lambda_true": result.log_lambda_true,
                "quality_score": -np.log(result.sigma_true) + result.log_lambda_true,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ASSIGNMENT_DIAGNOSTICS_FILENAME).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]) if rows else (
            "seed", "agent_id", "count_bucket", "sigma_true", "log_lambda_true", "quality_score"
        ))
        writer.writeheader()
        writer.writerows(rows)


def _run_scenario(scenario: ExposureScenario, include_raw_rationality_error: bool) -> None:
    config = scenario.config
    if config.dry_run:
        base_experiment.print_dry_run_summary(config)
        return
    seed_results = []
    assigner = make_count_assigner(scenario.mode)
    for seed in config.seed_values:
        seed_results.append(
            base_experiment.run_single_seed(
                config,
                seed,
                observation_count_assigner=assigner,
            )
        )

    output_paths = base_experiment.planned_output_paths(config.output_dir)
    agent_results = [result for seed_result in seed_results for result in seed_result.agent_results]
    bucket_rows, overall_rows = base_experiment.aggregate_results_across_seeds(seed_results)
    base_experiment.write_agent_level_csv(output_paths["agent_level_csv"], agent_results, config.environment)
    base_experiment.write_summary_csvs(config.output_dir, bucket_rows, overall_rows)
    base_experiment.plot_error_by_bucket(
        output_paths["error_plot"],
        bucket_rows,
        include_raw_rationality_error=include_raw_rationality_error,
    )
    _write_assignment_diagnostics(config.output_dir, agent_results)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    modes = tuple(mode.strip() for mode in args.modes.split(",") if mode.strip())
    unknown_modes = set(modes) - set(SCENARIO_MODES)
    if unknown_modes:
        raise ValueError(f"Unknown exposure modes: {sorted(unknown_modes)}")
    for mode in modes:
        output_dir = Path(args.output_dir) / mode
        config = _base_config(args, output_dir)
        scenario = ExposureScenario(mode=mode, config=config)
        if args.plot_only:
            base_experiment.regenerate_plots_from_existing_results(
                output_dir,
                include_raw_rationality_error=args.include_raw_rationality_error,
            )
        else:
            _run_scenario(scenario, args.include_raw_rationality_error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
