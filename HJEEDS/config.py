# This file still requires human verification. Delete this comment when done.
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .models import ExperimentConfig, HyperpriorConfig, TruePopulationConfig


# This module exists to answer one question: "What study are we about to run?"
# It centralizes the default experiment design, the command-line interface, and
# the conversion from raw CLI strings into a validated ``ExperimentConfig``.


CLI_DESCRIPTION = """Initial 1D darts hierarchical-vs-JEEDS experiments.

This module implements the first end-to-end synthetic experiment for comparing
independent JEEDS against a hierarchical empirical-Bayes extension. In
particular, this file:

1. Defines the configuration objects and command-line interface for the study.
2. Holds the data structures for true skills, simulated data, per-method
   estimates, and per-seed summaries.
3. Runs simulation, independent JEEDS inference, hierarchical inference,
   aggregation, and artifact writing.
4. Keeps a working ``--dry-run`` mode so we can validate parser/config wiring
   and inspect the intended workload before launching the numerical path.

The experiment compares:

- An independent JEEDS baseline that estimates each demonstrator separately
  with a uniform prior over the JEEDS skill grid.
- A hierarchical Bayesian alternative that learns a population-level prior over
  demonstrator skill and uses that prior to improve low-data estimates.

The main experimental twist is uneven observation counts: some demonstrators
will have only a handful of throws while others will have many. That is the
setting where the hierarchical model is expected to help most.

The implementation remains intentionally compact so later ablations can be
added without reorganizing the file.
"""


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# These defaults mirror the initial experiment plan discussed for the paper.
# They are kept as module-level constants so the intended study configuration is
# easy to inspect in one place.
DEFAULT_SIGMA_MIN = 0.5
DEFAULT_SIGMA_MAX = 4.5
DEFAULT_NUM_SIGMA_GRID = 21

DEFAULT_LAMBDA_MIN = 1e-3
DEFAULT_LAMBDA_MAX = 1e2
DEFAULT_NUM_LAMBDA_GRID = 21

DEFAULT_SIGMA_GRID = tuple(np.linspace(DEFAULT_SIGMA_MIN, DEFAULT_SIGMA_MAX, DEFAULT_NUM_SIGMA_GRID).tolist())
# JEEDS models lambda itself, not log(lambda). The historical code builds the
# lambda hypothesis grid with ``np.logspace`` because the candidate values span
# several orders of magnitude. That does not make base-10 log part of the
# model; it is only one way to generate a geometrically spaced grid. Using
# equally spaced natural logs and exponentiating would yield the same grid once
# the original-scale endpoints are fixed, so the experiment exposes
# ``lambda_min``/``lambda_max`` directly to avoid mixing log conventions in the
# user-facing interface.
DEFAULT_LAMBDA_GRID = tuple(
    np.exp(np.linspace(np.log(DEFAULT_LAMBDA_MIN), np.log(DEFAULT_LAMBDA_MAX), DEFAULT_NUM_LAMBDA_GRID)).tolist()
)

DEFAULT_COUNT_BUCKETS = (5, 10, 25, 100)
DEFAULT_AGENTS_PER_BUCKET = 5
DEFAULT_NUM_AGENTS = len(DEFAULT_COUNT_BUCKETS) * DEFAULT_AGENTS_PER_BUCKET
DEFAULT_DELTA = 0.1
DEFAULT_OUTPUT_DIR = Path("HJEEDS/results/hierarchical_darts")

DEFAULT_MIN_REGIONS = 2
DEFAULT_MAX_REGIONS = 6
DEFAULT_MIN_REGION_WIDTH = 0.25
DEFAULT_TRUE_POPULATION = TruePopulationConfig(
    mean_log_sigma=math.log(1.5),
    mean_log_lambda=math.log(1.0),
    tau_eta=0.35,
    tau_rho=1.0,
    correlation=-0.5,
)
DEFAULT_HYPERPRIORS = HyperpriorConfig(
    # For simulation studies, the default hyperprior centers are aligned with
    # the true data-generating population so the "unbiased" sensitivity condition
    # has centered hyperpriors. The prior variances still leave room
    # for empirical Bayes to move when the observed data disagree.
    mean_vector=(DEFAULT_TRUE_POPULATION.mean_log_sigma, DEFAULT_TRUE_POPULATION.mean_log_lambda),
    covariance_diagonal=(0.6**2, 3.0**2),
    log_tau_eta_mean=math.log(DEFAULT_TRUE_POPULATION.tau_eta),
    log_tau_eta_sd=0.5,
    log_tau_rho_mean=math.log(DEFAULT_TRUE_POPULATION.tau_rho),
    log_tau_rho_sd=0.5,
    m_r=math.atanh(DEFAULT_TRUE_POPULATION.correlation),
    s_r=0.75,
)

AGENT_LEVEL_FILENAME = "agent_level_results.csv"
SUMMARY_BY_BUCKET_FILENAME = "summary_by_bucket.csv"
SUMMARY_OVERALL_FILENAME = "summary_overall.csv"
ERROR_PLOT_FILENAME = "error_by_count_bucket.png"

# The CSV headers are declared in one place so artifact-writing code can stay
# dumb and predictable.  This also makes it easy for a human reviewer to see
# exactly what is expected in each output table.
AGENT_LEVEL_CSV_HEADER = [
    "seed",
    "agent_id",
    "count_bucket",
    "num_observations",
    "sigma_true",
    "lambda_true",
    "jeeds_posterior_mean_sigma",
    "jeeds_posterior_mean_lambda",
    "jeeds_map_sigma",
    "jeeds_map_lambda",
    "jeeds_status",
    "hierarchical_posterior_mean_sigma",
    "hierarchical_posterior_mean_lambda",
    "hierarchical_map_sigma",
    "hierarchical_map_lambda",
    "hierarchical_status",
    "notes",
]

SUMMARY_BY_BUCKET_CSV_HEADER = [
    "method",
    "metric",
    "count_bucket",
    "num_agents",
    "mean",
    "ci_lower",
    "ci_upper",
    "notes",
]

SUMMARY_OVERALL_CSV_HEADER = [
    "method",
    "metric",
    "num_agents",
    "mean",
    "ci_lower",
    "ci_upper",
    "notes",
]


def _parse_count_buckets(raw_value: str) -> tuple[int, ...]:
    """Parse a comma-separated bucket string such as ``"5,10,25,100"``."""

    # Users pass buckets as a compact comma-separated CLI string.  We normalize
    # whitespace first so downstream code only ever sees clean integer tuples.
    pieces = [piece.strip() for piece in raw_value.split(",") if piece.strip()]
    if not pieces:
        raise ValueError("At least one count bucket must be provided.")

    buckets = tuple(int(piece) for piece in pieces)
    if any(bucket <= 0 for bucket in buckets):
        raise ValueError(f"Count buckets must all be positive integers. Received: {buckets}")
    return buckets


def planned_output_paths(output_dir: Path) -> dict[str, Path]:
    """Return the artifact locations for this experiment family."""

    # Keeping the naming convention here prevents small path mismatches between
    # the dry-run summary, the writers, and the Slurm wrapper.
    return {
        "agent_level_csv": output_dir / AGENT_LEVEL_FILENAME,
        "summary_by_bucket_csv": output_dir / SUMMARY_BY_BUCKET_FILENAME,
        "summary_overall_csv": output_dir / SUMMARY_OVERALL_FILENAME,
        "error_plot": output_dir / ERROR_PLOT_FILENAME,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the experiment."""

    # The parser intentionally mirrors the study design from the paper:
    # observation buckets, skill-grid sizes, darts-surface settings, and
    # output location all stay visible at the command line.
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)

    parser.add_argument("--seed", type=int, default=12345, help="Base seed used to derive per-run seeds.")
    parser.add_argument(
        "--num-seeds",
        type=int,
        required=True,
        help="Number of random seeds to run. This argument is required so seed counts are always intentional.",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=DEFAULT_NUM_AGENTS,
        help="Total number of demonstrators in the synthetic population.",
    )
    parser.add_argument(
        "--count-buckets",
        type=str,
        default=",".join(str(bucket) for bucket in DEFAULT_COUNT_BUCKETS),
        help="Comma-separated observation-count buckets assigned across demonstrators.",
    )
    parser.add_argument(
        "--agents-per-bucket",
        type=int,
        default=DEFAULT_AGENTS_PER_BUCKET,
        help="How many demonstrators receive each observation-count bucket.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=DEFAULT_DELTA,
        help="Target-grid resolution used by the darts environment and later likelihood code.",
    )
    parser.add_argument(
        "--num-sigma-grid",
        type=int,
        default=DEFAULT_NUM_SIGMA_GRID,
        help="Number of execution-skill hypotheses on the JEEDS grid.",
    )
    parser.add_argument(
        "--num-lambda-grid",
        type=int,
        default=DEFAULT_NUM_LAMBDA_GRID,
        help="Number of decision-skill hypotheses on the JEEDS grid.",
    )
    parser.add_argument(
        "--sigma-min",
        type=float,
        default=DEFAULT_SIGMA_MIN,
        help="Minimum execution skill on the JEEDS hypothesis grid.",
    )
    parser.add_argument(
        "--sigma-max",
        type=float,
        default=DEFAULT_SIGMA_MAX,
        help="Maximum execution skill on the JEEDS hypothesis grid.",
    )
    parser.add_argument(
        "--lambda-min",
        type=float,
        default=DEFAULT_LAMBDA_MIN,
        help="Minimum decision-skill value on the JEEDS lambda grid.",
    )
    parser.add_argument(
        "--lambda-max",
        type=float,
        default=DEFAULT_LAMBDA_MAX,
        help="Maximum decision-skill value on the JEEDS lambda grid.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for CSV summaries and figures.",
    )
    parser.add_argument(
        "--min-regions",
        type=int,
        default=DEFAULT_MIN_REGIONS,
        help="Minimum number of reward regions when sampling a 1D darts surface.",
    )
    parser.add_argument(
        "--max-regions",
        type=int,
        default=DEFAULT_MAX_REGIONS,
        help="Maximum number of reward regions when sampling a 1D darts surface.",
    )
    parser.add_argument(
        "--min-region-width",
        type=float,
        default=DEFAULT_MIN_REGION_WIDTH,
        help="Minimum spacing between sampled reward-surface boundaries.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the intended workload and stop before simulation or inference.",
    )

    return parser.parse_args(argv)


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Translate parsed CLI args into a validated ``ExperimentConfig``."""

    from .sampling import build_skill_grids

    # Parse the structured fields once here so the rest of the code only needs
    # to reason about typed values, not raw CLI strings.
    count_buckets = _parse_count_buckets(args.count_buckets)

    config = ExperimentConfig(
        seed=args.seed,
        num_seeds=args.num_seeds,
        num_agents=args.num_agents,
        count_buckets=count_buckets,
        agents_per_bucket=args.agents_per_bucket,
        delta=args.delta,
        num_sigma_grid=args.num_sigma_grid,
        num_lambda_grid=args.num_lambda_grid,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
        min_regions=args.min_regions,
        max_regions=args.max_regions,
        min_region_width=args.min_region_width,
        hyperpriors=DEFAULT_HYPERPRIORS,
        true_population=DEFAULT_TRUE_POPULATION,
    )

    # The checks below catch design mistakes early, including in dry-run mode.
    # That is especially useful when launching long multi-seed sweeps on a
    # cluster, where a bad CLI should fail fast rather than after scheduling.
    if config.num_seeds <= 0:
        raise ValueError(f"num_seeds must be positive. Received {config.num_seeds}.")
    if config.agents_per_bucket <= 0:
        raise ValueError(f"agents_per_bucket must be positive. Received {config.agents_per_bucket}.")
    if config.num_agents <= 0:
        raise ValueError(f"num_agents must be positive. Received {config.num_agents}.")
    if config.expected_agent_count != config.num_agents:
        raise ValueError(
            "num_agents must equal len(count_buckets) * agents_per_bucket for this experiment. "
            f"Expected {config.expected_agent_count} from the bucket design, received {config.num_agents}."
        )

    # Validate grid construction up front so dry-run catches bad CLI choices.
    build_skill_grids(config)

    return config


def print_dry_run_summary(config: ExperimentConfig) -> None:
    """Print the workload and artifact plan without launching the experiment."""

    # Dry-run is deliberately verbose because it is the safest way to confirm
    # that a long run matches the intended paper condition before simulation
    # or optimization begins.
    paths = planned_output_paths(config.output_dir)

    print("=== DRY RUN: 1D Hierarchical Darts vs JEEDS ===")
    print("This dry run validates parser/config wiring and reports the intended workload.")
    print("No simulation or inference functions will be executed.")
    print()
    print(f"Seeds: {config.seed_values}")
    print(f"Agents: {config.num_agents}")
    print(f"Count buckets: {config.count_buckets}")
    print(f"Agents per bucket: {config.agents_per_bucket}")
    print(f"Delta: {config.delta}")
    print(f"Sigma grid size: {config.num_sigma_grid}")
    print(f"Lambda grid size: {config.num_lambda_grid}")
    print(f"Output directory: {config.output_dir.resolve()}")
    print("Planned artifacts:")
    for label, path in paths.items():
        print(f"  - {label}: {path}")
    print()
    print("Planned pipeline:")
    print("  1. Sample one fixed 1D darts reward surface per seed.")
    print("  2. Sample demonstrator true skills from the hierarchical population.")
    print("  3. Assign uneven observation counts across demonstrators.")
    print("  4. Simulate agent datasets from the JEEDS-consistent generative model.")
    print("  5. Run independent JEEDS on each agent.")
    print("  6. Fit hierarchical MAP hyperparameters and rerun posteriors.")
    print("  7. Aggregate errors by bucket and overall.")
    print("  8. Write CSV summaries and a two-panel figure.")
