# This file was written or edited by AI and still requires human review. Delete this comment when done.
"""CLI and configuration for baseball HJEEDS experiments."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .baseball_hyperpriors import resolve_baseball_hyperpriors
from .baseball_pitch import (
    DEFAULT_DELTA,
    DEFAULT_EXECUTION_SKILL_MAX,
    DEFAULT_EXECUTION_SKILL_MIN,
    StatcastAgentSpec,
    build_execution_skill_grid,
    build_log_lambda_grid,
)
from .baseball_roster import (
    add_common_roster_arguments,
    add_hyperprior_arguments,
    load_statcast_for_roster,
    parse_pitch_types,
    resolve_baseball_roster,
    validate_roster_selection,
)
from .config import (
    DEFAULT_NUM_SEEDS,
    DEFAULT_OUTPUT_DIR,
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
DEFAULT_PITCHER_IDS = (543037, 453286)
DEFAULT_PITCH_TYPES = ("FF",)
DEFAULT_MIN_PITCHES_PER_AGENT = 100


@dataclass(frozen=True)
class BaseballExperimentConfig:
    """Experiment configuration for Statcast baseball HJEEDS."""

    base: ExperimentConfig
    season_year: int | None
    pitcher_ids: tuple[int, ...]
    pitch_types: tuple[str, ...]
    max_pitches_per_agent: int | None
    use_natural_pitch_counts: bool
    agent_specs: tuple[StatcastAgentSpec, ...]
    agents: tuple[tuple[int, str], ...]
    agent_pitch_counts: tuple[tuple[int, str, int], ...] = ()
    excluded_agents: tuple[tuple[int, str, int], ...] = ()

    @property
    def environment(self) -> str:
        return self.base.environment

    @property
    def seed_values(self) -> tuple[int, ...]:
        return self.base.seed_values


def build_baseball_skill_grids(config: BaseballExperimentConfig):
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
    parser.add_argument(
        "--seed",
        type=parse_seed_argument,
        required=False,
        default=None,
        help=(
            "Base seed (or 'default' for 12345). Kept for CLI compatibility; "
            "Statcast likelihoods are deterministic in seed."
        ),
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Keep at 1 for baseball: additional seeds do not change Statcast estimates.",
    )
    parser.add_argument(
        "--count-buckets",
        type=str,
        default="",
        help="Comma-separated synthetic pitch-count buckets (ignored with --use-natural-pitch-counts).",
    )
    parser.add_argument(
        "--agents-per-bucket",
        type=int,
        default=1,
        help="Agents assigned to each count bucket (ignored with --use-natural-pitch-counts).",
    )
    parser.add_argument(
        "--use-natural-pitch-counts",
        action="store_true",
        help="Use each agent's actual pitch count (optionally capped by --max-pitches-per-agent).",
    )
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
    add_common_roster_arguments(parser)
    add_hyperprior_arguments(parser)
    return parser.parse_args(argv)


def build_baseball_config_from_args(args: argparse.Namespace) -> BaseballExperimentConfig:
    pitch_types = parse_pitch_types(args.pitch_types)
    min_pitches_per_agent = (
        args.min_pitches_per_agent
        if args.min_pitches_per_agent is not None
        else DEFAULT_MIN_PITCHES_PER_AGENT
    )
    all_data = load_statcast_for_roster(args.season_year)

    validate_roster_selection(
        all_eligible_agents=args.all_eligible_agents,
        top_pitchers=args.top_pitchers,
        bbip_extremes=getattr(args, "bbip_extremes", None),
    )

    if getattr(args, "bbip_extremes", None) is not None:
        roster_selector = dict(
            all_eligible_agents=False,
            pitcher_ids=None,
            top_pitchers=None,
            bbip_extremes=args.bbip_extremes,
        )
    elif args.all_eligible_agents:
        roster_selector = dict(all_eligible_agents=True, pitcher_ids=None, top_pitchers=None, bbip_extremes=None)
    elif args.top_pitchers is not None:
        roster_selector = dict(all_eligible_agents=False, pitcher_ids=None, top_pitchers=args.top_pitchers, bbip_extremes=None)
    else:
        pitcher_ids = tuple(int(piece.strip()) for piece in args.pitcher_ids.split(",") if piece.strip())
        roster_selector = dict(all_eligible_agents=False, pitcher_ids=pitcher_ids, top_pitchers=None, bbip_extremes=None)

    roster = resolve_baseball_roster(
        all_data=all_data,
        season_year=args.season_year,
        pitch_types=pitch_types,
        min_pitches_per_agent=min_pitches_per_agent,
        max_agents=args.max_agents,
        **roster_selector,
    )

    use_natural = args.use_natural_pitch_counts or args.all_eligible_agents
    if use_natural:
        count_buckets = (0,)
        agents_per_bucket = len(roster.agent_specs)
    else:
        if not args.count_buckets:
            raise ValueError(
                "Provide --count-buckets or enable --use-natural-pitch-counts / --all-eligible-agents."
            )
        count_buckets = _parse_count_buckets(args.count_buckets)
        agents_per_bucket = args.agents_per_bucket
        if len(count_buckets) * agents_per_bucket != len(roster.agent_specs):
            raise ValueError(
                "For synthetic count buckets, len(count_buckets) * agents_per_bucket must equal "
                f"{len(roster.agent_specs)} agents. Use --use-natural-pitch-counts for full-population runs."
            )

    hyperpriors = resolve_baseball_hyperpriors(
        preset=args.hyperprior_preset,
        calibrated_path=Path(args.hyperprior_config) if args.hyperprior_config else None,
    )
    output_dir = Path(args.output_dir)
    base = ExperimentConfig(
        environment="baseball",
        seed=args.seed,
        num_seeds=args.num_seeds,
        num_agents=len(roster.agent_specs),
        count_buckets=count_buckets,
        agents_per_bucket=agents_per_bucket,
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
        hyperpriors=hyperpriors,
        true_population=TruePopulationConfig(
            mean_log_sigma=hyperpriors.mean_vector[0],
            mean_log_lambda=hyperpriors.mean_vector[1],
            tau_eta=math.exp(hyperpriors.log_tau_eta_mean),
            tau_rho=math.exp(hyperpriors.log_tau_rho_mean),
            correlation=math.tanh(hyperpriors.m_r),
        ),
    )
    config = BaseballExperimentConfig(
        base=base,
        season_year=args.season_year,
        pitcher_ids=roster.pitcher_ids,
        pitch_types=roster.pitch_types,
        max_pitches_per_agent=args.max_pitches_per_agent,
        use_natural_pitch_counts=use_natural,
        agent_specs=roster.agent_specs,
        agents=tuple((spec.pitcher_id, spec.pitch_type) for spec in roster.agent_specs),
        agent_pitch_counts=roster.agent_pitch_counts,
        excluded_agents=roster.excluded_agents,
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
    print(f"Season year: {config.season_year or 'all seasons in pickle'}")
    print(f"Seeds: {config.seed_values}")
    print(
        "Note: seed does not change Statcast likelihoods (execution PDFs use "
        "multivariate_normal.pdf; pitch order is newest-first by game_date)."
    )
    print(f"Agents: {len(config.agent_specs)}")
    print(f"Pitch types: {config.pitch_types}")
    print(f"Use natural pitch counts: {config.use_natural_pitch_counts}")
    if config.agent_pitch_counts:
        print("Agent pitch counts (first 10):")
        for pitcher_id, pitch_type, pitch_count in config.agent_pitch_counts[:10]:
            print(f"  pitcher={pitcher_id} pitch_type={pitch_type} count={pitch_count}")
        if len(config.agent_pitch_counts) > 10:
            print(f"  ... and {len(config.agent_pitch_counts) - 10} more")
    if config.excluded_agents:
        print("Excluded agents:")
        for pitcher_id, pitch_type, pitch_count in config.excluded_agents:
            print(f"  pitcher={pitcher_id} pitch_type={pitch_type} count={pitch_count}")
    print(f"Hyperprior preset centers (log sigma, log lambda): {config.base.hyperpriors.mean_vector}")
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
    print("  2. For each agent, take newest pitches (natural count or bucket cap)")
    print("  3. Build per-pitch RNN utility grids + EV surfaces")
    print("  4. Compute independent JEEDS log-likelihood grids")
    print("  5. Fit hierarchical MAP prior and rerun posteriors")
    print("  6. Write agent_level_results.csv")
