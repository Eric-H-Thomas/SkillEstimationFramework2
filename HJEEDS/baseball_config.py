# This file was written or edited by AI and still requires human review. Delete this comment when done.
"""CLI and configuration for baseball HJEEDS experiments."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .baseball_pitch import (
    DEFAULT_DELTA,
    DEFAULT_EXECUTION_SKILL_MAX,
    DEFAULT_EXECUTION_SKILL_MIN,
    build_execution_skill_grid,
    build_log_lambda_grid,
    resolve_agent_roster,
)
from .config import (
    DEFAULT_AGENTS_PER_BUCKET,
    DEFAULT_COUNT_BUCKETS,
    DEFAULT_HYPERPRIORS,
    DEFAULT_NUM_AGENTS,
    DEFAULT_NUM_SEEDS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    DEFAULT_TRUE_POPULATION,
    HyperpriorConfig,
    TruePopulationConfig,
    _parse_count_buckets,
    parse_seed_argument,
    planned_output_paths,
)
from .models import ExperimentConfig


CLI_DESCRIPTION = """Compare independent JEEDS vs hierarchical H-JEEDS on Statcast baseball data.

Each agent is a (pitcherID, pitchType) pair. Observations are real pitch locations;
per-pitch utility grids come from the OptimusPitch RNN (final_OP) and getUtility().
"""


DEFAULT_OUTPUT_DIR_BASEBALL = Path("HJEEDS/results/hierarchical_baseball")
DEFAULT_LAMBDA_MIN = 1e-3
DEFAULT_LAMBDA_MAX = 10 ** 3.6
DEFAULT_NUM_SIGMA_GRID = 21
DEFAULT_NUM_LAMBDA_GRID = 21
DEFAULT_PITCHER_IDS = (445276, 623433)
DEFAULT_PITCH_TYPES = ("FF",)
DEFAULT_COUNT_BUCKETS_BASEBALL = (5, 10)
DEFAULT_AGENTS_PER_BUCKET_BASEBALL = 1


@dataclass(frozen=True)
class BaseballExperimentConfig:
    """Experiment configuration for Statcast baseball HJEEDS."""

    base: ExperimentConfig
    pitcher_ids: tuple[int, ...]
    pitch_types: tuple[str, ...]
    max_pitches_per_agent: int | None
    agents: tuple[tuple[int, str], ...]

    @property
    def environment(self) -> str:
        return self.base.environment

    @property
    def seed_values(self) -> tuple[int, ...]:
        return self.base.seed_values


def build_baseball_skill_grids(config: BaseballExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Build execution-skill and log-lambda grids for baseball."""

    import numpy as np

    if config.base.num_sigma_grid == DEFAULT_NUM_SIGMA_GRID and (
        config.base.sigma_min == DEFAULT_EXECUTION_SKILL_MIN
        and config.base.sigma_max == DEFAULT_EXECUTION_SKILL_MAX
    ):
        sigma_grid = build_execution_skill_grid(config.base.delta)
    else:
        sigma_grid = np.linspace(
            config.base.sigma_min,
            config.base.sigma_max,
            config.base.num_sigma_grid,
            dtype=float,
        )
    log_lambda_grid = build_log_lambda_grid(
        lambda_min=config.base.lambda_min,
        lambda_max=config.base.lambda_max,
        num_lambda_grid=config.base.num_lambda_grid,
    )
    return sigma_grid, log_lambda_grid


def parse_baseball_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    parser.add_argument("--seed", type=parse_seed_argument, required=True)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument(
        "--count-buckets",
        type=str,
        default=",".join(str(bucket) for bucket in DEFAULT_COUNT_BUCKETS_BASEBALL),
    )
    parser.add_argument(
        "--agents-per-bucket",
        type=int,
        default=DEFAULT_AGENTS_PER_BUCKET_BASEBALL,
    )
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--num-sigma-grid", type=int, default=DEFAULT_NUM_SIGMA_GRID)
    parser.add_argument("--num-lambda-grid", type=int, default=DEFAULT_NUM_LAMBDA_GRID)
    parser.add_argument("--sigma-min", type=float, default=DEFAULT_EXECUTION_SKILL_MIN)
    parser.add_argument("--sigma-max", type=float, default=DEFAULT_EXECUTION_SKILL_MAX)
    parser.add_argument("--lambda-min", type=float, default=DEFAULT_LAMBDA_MIN)
    parser.add_argument("--lambda-max", type=float, default=DEFAULT_LAMBDA_MAX)
    parser.add_argument("--pitcher-ids", type=str, default=",".join(str(pid) for pid in DEFAULT_PITCHER_IDS))
    parser.add_argument("--pitch-types", type=str, default=",".join(DEFAULT_PITCH_TYPES))
    parser.add_argument(
        "--max-pitches-per-agent",
        type=int,
        default=None,
        help="Cap pitches per agent (newest first). Useful for smoke tests.",
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR_BASEBALL))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_baseball_config_from_args(args: argparse.Namespace) -> BaseballExperimentConfig:
    count_buckets = _parse_count_buckets(args.count_buckets)
    pitcher_ids = tuple(int(piece.strip()) for piece in args.pitcher_ids.split(",") if piece.strip())
    pitch_types = tuple(piece.strip() for piece in args.pitch_types.split(",") if piece.strip())
    if not pitcher_ids:
        raise ValueError("At least one pitcher id is required.")
    if not pitch_types:
        raise ValueError("At least one pitch type is required.")

    roster = resolve_agent_roster(pitcher_ids, pitch_types)
    num_agents = args.num_agents if args.num_agents is not None else len(roster)
    if num_agents != len(roster):
        raise ValueError(
            f"num_agents must equal len(pitcher_ids) * len(pitch_types) = {len(roster)}. "
            f"Received {num_agents}."
        )
    if len(count_buckets) * args.agents_per_bucket != num_agents:
        raise ValueError(
            "For baseball, assign one count bucket per agent: "
            f"len(count_buckets) * agents_per_bucket must equal {num_agents}."
        )

    output_dir = Path(args.output_dir)
    base = ExperimentConfig(
        environment="baseball",
        seed=args.seed,
        num_seeds=args.num_seeds,
        num_agents=num_agents,
        count_buckets=count_buckets,
        agents_per_bucket=args.agents_per_bucket,
        delta=args.delta,
        num_sigma_grid=args.num_sigma_grid,
        num_lambda_grid=args.num_lambda_grid,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        output_dir=output_dir,
        environment_grids={},
        dry_run=args.dry_run,
        min_success_regions=2,
        max_success_regions=6,
        min_region_width=0.25,
        hyperpriors=DEFAULT_HYPERPRIORS,
        true_population=TruePopulationConfig(
            mean_log_sigma=math.log(0.5),
            mean_log_lambda=math.log(1.0),
            tau_eta=0.35,
            tau_rho=1.0,
            correlation=-0.5,
        ),
    )
    config = BaseballExperimentConfig(
        base=base,
        pitcher_ids=pitcher_ids,
        pitch_types=pitch_types,
        max_pitches_per_agent=args.max_pitches_per_agent,
        agents=tuple((spec.pitcher_id, spec.pitch_type) for spec in roster),
    )
    build_baseball_skill_grids(config)
    return config


def print_baseball_dry_run_summary(config: BaseballExperimentConfig) -> None:
    paths = planned_output_paths(config.base.output_dir)
    sigma_grid, log_lambda_grid = build_baseball_skill_grids(config)

    print("=== DRY RUN: Hierarchical Baseball vs JEEDS ===")
    print("No RNN inference or estimation will run.")
    print()
    print(f"Environment: {config.environment}")
    print(f"Seeds: {config.seed_values}")
    print(f"Agents (pitcher, pitch type): {config.agents}")
    print(f"Count buckets: {config.base.count_buckets}")
    print(f"Max pitches per agent: {config.max_pitches_per_agent or 'all available'}")
    print(f"Delta: {config.base.delta}")
    print(f"Execution skill grid: {len(sigma_grid)} points [{sigma_grid[0]:.3f}, {sigma_grid[-1]:.3f}]")
    print(f"Log-lambda grid: {len(log_lambda_grid)} points")
    print(f"Output directory: {config.base.output_dir.resolve()}")
    print("Planned artifacts:")
    for label, path in paths.items():
        print(f"  - {label}: {path}")
    print()
    print("Planned pipeline:")
    print("  1. Load ProcessedData-From-GivenFiles.pkl")
    print("  2. For each agent, take newest N pitches per count bucket")
    print("  3. Build per-pitch RNN utility grids + EV surfaces")
    print("  4. Compute independent JEEDS log-likelihood grids")
    print("  5. Fit hierarchical MAP prior and rerun posteriors")
    print("  6. Write agent_level_results.csv")
