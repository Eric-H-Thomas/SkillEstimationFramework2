"""Initial 1D darts hierarchical-vs-JEEDS experiments.

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

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# Ensure the repository root is importable when this file is executed directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class HyperpriorConfig:
    """Container for the population-level hyperpriors from the paper draft."""

    mean_vector: tuple[float, float]
    covariance_diagonal: tuple[float, float]
    a_eta: float
    a_rho: float
    m_r: float
    s_r: float

    @property
    def covariance_matrix(self) -> np.ndarray:
        """Return the 2x2 prior covariance matrix for the population mean."""

        return np.diag(self.covariance_diagonal)


@dataclass(frozen=True)
class TruePopulationConfig:
    """Ground-truth population used to simulate demonstrator skill profiles."""

    mean_log_sigma: float
    mean_log_lambda: float
    tau_eta: float
    tau_rho: float
    correlation: float

    @property
    def mean_vector(self) -> np.ndarray:
        """Return the population mean in log-skill space."""

        return np.array([self.mean_log_sigma, self.mean_log_lambda], dtype=float)

    @property
    def covariance_matrix(self) -> np.ndarray:
        """Return the 2x2 covariance matrix used to sample true skills."""

        cov = np.array(
            [
                [self.tau_eta**2, self.correlation * self.tau_eta * self.tau_rho],
                [self.correlation * self.tau_eta * self.tau_rho, self.tau_rho**2],
            ],
            dtype=float,
        )
        return cov


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
DEFAULT_NUM_SEEDS = 30
DEFAULT_DELTA = 0.1
DEFAULT_OUTPUT_DIR = Path("Testing/results/hierarchical_darts")

DEFAULT_MIN_REGIONS = 2
DEFAULT_MAX_REGIONS = 6
DEFAULT_MIN_REGION_WIDTH = 0.25
DEFAULT_HYPERPRIORS = HyperpriorConfig(
    mean_vector=(0.41, -1.15),
    covariance_diagonal=(0.6**2, 3.0**2),
    a_eta=0.31,
    a_rho=1.53,
    m_r=-0.31,
    s_r=0.75,
)
DEFAULT_TRUE_POPULATION = TruePopulationConfig(
    mean_log_sigma=math.log(1.5),
    mean_log_lambda=math.log(1.0),
    tau_eta=0.35,
    tau_rho=1.0,
    correlation=-0.5,
)

AGENT_LEVEL_FILENAME = "agent_level_results.csv"
SUMMARY_BY_BUCKET_FILENAME = "summary_by_bucket.csv"
SUMMARY_OVERALL_FILENAME = "summary_overall.csv"
ERROR_PLOT_FILENAME = "error_by_count_bucket.png"

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
@dataclass(frozen=True)
class ExperimentConfig:
    """All configuration needed to define one sweep of the experiment."""

    seed: int
    num_seeds: int
    num_agents: int
    count_buckets: tuple[int, ...]
    agents_per_bucket: int
    delta: float
    num_sigma_grid: int
    num_lambda_grid: int
    sigma_min: float
    sigma_max: float
    lambda_min: float
    lambda_max: float
    output_dir: Path
    dry_run: bool
    min_regions: int
    max_regions: int
    min_region_width: float
    hyperpriors: HyperpriorConfig
    true_population: TruePopulationConfig

    @property
    def expected_agent_count(self) -> int:
        """Return the implied number of agents from the bucket design."""

        return len(self.count_buckets) * self.agents_per_bucket

    @property
    def seed_values(self) -> tuple[int, ...]:
        """Return the concrete seed values that would be executed."""

        return tuple(self.seed + offset for offset in range(self.num_seeds))


@dataclass(frozen=True)
class AgentTruth:
    """Ground-truth skill profile for one demonstrator."""

    agent_id: int
    sigma_true: float
    lambda_true: float
    log_sigma_true: float
    log_lambda_true: float


@dataclass(frozen=True)
class AgentDataset:
    """Observed dataset for one demonstrator.

    The arrays are optional so helper-level tests can construct partial
    datasets, but the main simulation path populates both.
    """

    agent_id: int
    seed: int
    # This usually equals ``num_observations`` in the current experiment. We keep
    # it separate so later analyses can use bucket labels that collapse multiple
    # exact counts into one group.
    count_bucket: int
    num_observations: int
    reward_surface: tuple[float, ...]
    intended_targets: np.ndarray | None = None
    executed_actions: np.ndarray | None = None
    notes: str = ""


@dataclass(frozen=True)
class MethodEstimate:
    """Estimated skill output for one method on one demonstrator."""

    method_name: str
    posterior_mean_sigma: float | None = None
    posterior_mean_lambda: float | None = None
    map_sigma: float | None = None
    map_lambda: float | None = None
    status: str = "todo"
    notes: str = ""


@dataclass(frozen=True)
class AgentResult:
    """Combined truth and method estimates for one demonstrator."""

    seed: int
    agent_id: int
    count_bucket: int
    num_observations: int
    sigma_true: float
    lambda_true: float
    jeeds: MethodEstimate
    hierarchical: MethodEstimate
    notes: str = ""


@dataclass
class SeedResult:
    """Bundle all per-seed outputs for later aggregation."""

    seed: int
    reward_surface: tuple[float, ...]
    agent_truths: list[AgentTruth] = field(default_factory=list)
    agent_datasets: list[AgentDataset] = field(default_factory=list)
    agent_results: list[AgentResult] = field(default_factory=list)
    summary_by_bucket_rows: list[dict[str, Any]] = field(default_factory=list)
    summary_overall_rows: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""


def _parse_count_buckets(raw_value: str) -> tuple[int, ...]:
    """Parse a comma-separated bucket string such as ``"5,10,25,100"``."""

    pieces = [piece.strip() for piece in raw_value.split(",") if piece.strip()]
    if not pieces:
        raise ValueError("At least one count bucket must be provided.")

    buckets = tuple(int(piece) for piece in pieces)
    if any(bucket <= 0 for bucket in buckets):
        raise ValueError(f"Count buckets must all be positive integers. Received: {buckets}")
    return buckets


def planned_output_paths(output_dir: Path) -> dict[str, Path]:
    """Return the artifact locations for this experiment family."""

    return {
        "agent_level_csv": output_dir / AGENT_LEVEL_FILENAME,
        "summary_by_bucket_csv": output_dir / SUMMARY_BY_BUCKET_FILENAME,
        "summary_overall_csv": output_dir / SUMMARY_OVERALL_FILENAME,
        "error_plot": output_dir / ERROR_PLOT_FILENAME,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--seed", type=int, default=12345, help="Base seed used to derive per-run seeds.")
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS, help="Number of random seeds to run.")
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


# -----------------------------------------------------------------------------
# Sampling Helpers
# -----------------------------------------------------------------------------

def sample_reward_surface(rng: np.random.Generator, config: ExperimentConfig) -> tuple[float, ...]:
    """Sample a single 1D darts reward surface.

    We intentionally keep one reward surface fixed for all demonstrators within
    a seed. That isolates the skill-estimation question from environment
    variability and keeps the first comparison aligned with the paper plan.
    """

    # Import lazily so ``--dry-run`` can validate the experiment structure even
    # when the local Python environment is missing the heavier scientific
    # dependencies required by the full darts environment module.
    from Environments.Darts.RandomDarts import darts

    states = darts.generate_random_states(
        rng,
        config.min_regions,
        config.max_regions,
        1,
        min_width=config.min_region_width,
    )
    if not states:
        raise RuntimeError("Failed to sample a 1D darts reward surface.")
    return tuple(float(boundary) for boundary in states[0])


def build_skill_grids(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Construct the JEEDS execution-skill and decision-skill grids.

    We expose lambda bounds on the original scale and then build a
    geometrically spaced grid by interpolating evenly in natural-log space.
    This is equivalent to the historical ``np.logspace`` style once the
    original-scale endpoints are fixed.
    """

    if config.num_sigma_grid <= 0:
        raise ValueError("num_sigma_grid must be positive.")
    if config.num_lambda_grid <= 0:
        raise ValueError("num_lambda_grid must be positive.")
    if config.sigma_min <= 0.0 or config.sigma_max <= 0.0:
        raise ValueError("Execution skill bounds must be positive.")
    if config.lambda_min <= 0.0 or config.lambda_max <= 0.0:
        raise ValueError("Decision-skill bounds must be positive.")
    if config.sigma_min >= config.sigma_max:
        raise ValueError("sigma_min must be strictly less than sigma_max.")
    if config.lambda_min >= config.lambda_max:
        raise ValueError("lambda_min must be strictly less than lambda_max.")

    sigma_grid = np.linspace(config.sigma_min, config.sigma_max, config.num_sigma_grid, dtype=float)
    lambda_grid = np.exp(
        np.linspace(
            np.log(config.lambda_min),
            np.log(config.lambda_max),
            config.num_lambda_grid,
            dtype=float,
        )
    )
    return sigma_grid, lambda_grid


def sample_true_population_params(
    rng: np.random.Generator,
    config: ExperimentConfig,
    sigma_grid: np.ndarray,
    lambda_grid: np.ndarray,
) -> list[AgentTruth]:
    """Sample the true demonstrator skill profiles for one seed.

    The simulated truth is rejection-sampled to stay inside the estimator grid
    so early experiments are not dominated by grid truncation artifacts.
    """

    sigma_min, sigma_max = float(np.min(sigma_grid)), float(np.max(sigma_grid))
    lambda_min, lambda_max = float(np.min(lambda_grid)), float(np.max(lambda_grid))

    truths: list[AgentTruth] = []
    covariance = config.true_population.covariance_matrix
    mean_vector = config.true_population.mean_vector

    for agent_id in range(config.num_agents):
        # We keep the simulated truth inside the estimator support so later
        # comparisons do not get dominated by pure grid truncation artifacts.
        for _attempt in range(10_000):
            eta, rho = rng.multivariate_normal(mean_vector, covariance)
            sigma_true = math.exp(float(eta))
            lambda_true = math.exp(float(rho))

            if sigma_min <= sigma_true <= sigma_max and lambda_min <= lambda_true <= lambda_max:
                truths.append(
                    AgentTruth(
                        agent_id=agent_id,
                        sigma_true=sigma_true,
                        lambda_true=lambda_true,
                        log_sigma_true=float(eta),
                        log_lambda_true=float(rho),
                    )
                )
                break
        else:
            raise RuntimeError(
                "Failed to sample a true skill profile within the JEEDS grid support. "
                "Consider widening the grids or changing the true population."
            )

    return truths


def assign_observation_counts(config: ExperimentConfig) -> list[int]:
    """Assign the uneven per-demonstrator observation counts for one seed."""

    counts: list[int] = []
    for bucket in config.count_buckets:
        counts.extend([bucket] * config.agents_per_bucket)

    if len(counts) != config.num_agents:
        raise RuntimeError(
            "Observation-count assignment produced the wrong number of agents. "
            f"Expected {config.num_agents}, received {len(counts)}."
        )

    # We keep the ordering deterministic so it is obvious how
    # the bucket design maps to agent IDs. If we later want random bucket
    # assignments, we can do that explicitly and document the change.
    return counts


def simulate_agent_dataset(
    rng: np.random.Generator,
    seed: int,
    config: ExperimentConfig,
    reward_surface: tuple[float, ...],
    agent_truth: AgentTruth,
    num_observations: int,
    sigma_grid: np.ndarray,
    lambda_grid: np.ndarray,
) -> AgentDataset:
    """Simulate one demonstrator's observed throws.

    Simulate one demonstrator using the JEEDS-style generative story.

    The intended targets are sampled from a softmax policy over the 1D darts
    target grid, where expected values are computed under the demonstrator's
    true execution skill. Each intended target is then perturbed by wrapped
    Gaussian execution noise to produce the observed action.

    The simulator follows the same broad modeling assumptions used by the
    likelihood path so the generated data and estimators are matched in this
    first experiment.
    """

    # Import lazily so dry-run and helper tests can still run in lightweight
    # environments where the full darts/scipy stack is not installed.
    from Environments.Darts.RandomDarts import darts

    if num_observations <= 0:
        raise ValueError(f"num_observations must be positive. Received {num_observations}.")
    if agent_truth.sigma_true < float(np.min(sigma_grid)) or agent_truth.sigma_true > float(np.max(sigma_grid)):
        raise ValueError(
            f"True sigma {agent_truth.sigma_true} lies outside the provided sigma grid "
            f"[{float(np.min(sigma_grid))}, {float(np.max(sigma_grid))}]."
        )
    if agent_truth.lambda_true < float(np.min(lambda_grid)) or agent_truth.lambda_true > float(np.max(lambda_grid)):
        raise ValueError(
            f"True lambda {agent_truth.lambda_true} lies outside the provided lambda grid "
            f"[{float(np.min(lambda_grid))}, {float(np.max(lambda_grid))}]."
        )

    # Compute the true expected-value curve over the same 1D target grid shape
    # used elsewhere in the darts codebase. ``actions`` is the discrete set of
    # intended targets from which the demonstrator chooses.
    expected_values, actions = darts.compute_expected_value_curve(
        reward_surface,
        agent_truth.sigma_true,
        config.delta,
    )
    expected_values = np.asarray(expected_values, dtype=float)
    actions = np.asarray(actions, dtype=float)

    if expected_values.shape != actions.shape:
        raise RuntimeError(
            "Expected-value computation returned mismatched arrays for values and actions: "
            f"{expected_values.shape} vs {actions.shape}."
        )
    if expected_values.size == 0:
        raise RuntimeError("Expected-value computation returned an empty target grid.")

    # Softmax over expected values. We subtract the maximum before exponentiating
    # to keep the probabilities numerically stable for large lambda values.
    scaled_values = agent_truth.lambda_true * expected_values
    scaled_values -= np.max(scaled_values)
    target_probabilities = np.exp(scaled_values)
    target_probabilities /= np.sum(target_probabilities)

    intended_targets = rng.choice(actions, size=num_observations, p=target_probabilities)

    # Each intended target is executed with wrapped Gaussian noise using the
    # existing domain helper so the simulation matches the board geometry the
    # estimator will later assume.
    executed_actions = np.array(
        [
            darts.sample_noisy_action(rng, reward_surface, agent_truth.sigma_true, float(target))
            for target in intended_targets
        ],
        dtype=float,
    )

    return AgentDataset(
        agent_id=agent_truth.agent_id,
        seed=seed,
        count_bucket=num_observations,
        num_observations=num_observations,
        reward_surface=reward_surface,
        intended_targets=np.asarray(intended_targets, dtype=float),
        executed_actions=executed_actions,
        notes=(
            "Simulated with a JEEDS-style softmax policy over the 1D darts target grid "
            "and wrapped Gaussian execution noise."
        ),
    )


# -----------------------------------------------------------------------------
# Likelihood Helpers
# -----------------------------------------------------------------------------

def compute_agent_log_likelihood_grid(
    config: ExperimentConfig,
    reward_surface: tuple[float, ...],
    agent_dataset: AgentDataset,
    sigma_grid: np.ndarray,
    lambda_grid: np.ndarray,
) -> np.ndarray:
    """Compute the standalone JEEDS log-likelihood table for one agent.

    This mirrors the production JEEDS update rule in ``Estimators/joint.py``:
    for each execution-skill hypothesis we compute the expected-value curve and,
    for each decision-skill hypothesis, evaluate the per-observation update
    term obtained after marginalizing over latent intended targets.

    The result is a grid of log-likelihoods with shape
    ``(len(sigma_grid), len(lambda_grid))`` that later estimators can combine
    with different priors.
    """

    # Import lazily so the module remains lightweight to import in dry-run
    # mode. The full likelihood path depends on the darts environment helpers.
    from Environments.Darts.RandomDarts import darts

    if agent_dataset.executed_actions is None:
        raise ValueError("Agent dataset must contain executed_actions before likelihood evaluation.")
    if agent_dataset.executed_actions.shape[0] != agent_dataset.num_observations:
        raise ValueError(
            "Length of executed_actions does not match num_observations: "
            f"{agent_dataset.executed_actions.shape[0]} vs {agent_dataset.num_observations}."
        )

    executed_actions = np.asarray(agent_dataset.executed_actions, dtype=float)
    log_likelihood_grid = np.full((len(sigma_grid), len(lambda_grid)), -np.inf, dtype=float)

    reference_targets: np.ndarray | None = None

    for sigma_index, sigma_hypothesis in enumerate(sigma_grid):
        expected_values, target_actions = darts.compute_expected_value_curve(
            reward_surface,
            float(sigma_hypothesis),
            config.delta,
        )
        expected_values = np.asarray(expected_values, dtype=float)  # shape: (T,)
        target_actions = np.asarray(target_actions, dtype=float)  # shape: (T,)

        if expected_values.shape != target_actions.shape:
            raise RuntimeError(
                "Expected-value computation returned mismatched arrays for values and actions: "
                f"{expected_values.shape} vs {target_actions.shape}."
            )
        if expected_values.size == 0:
            raise RuntimeError("Expected-value computation returned an empty target grid.")

        # In 1D darts, the target grid should be shared across sigma hypotheses.
        # We verify that explicitly so later code can safely assume a common grid.
        if reference_targets is None:
            reference_targets = target_actions
        elif not np.allclose(reference_targets, target_actions):
            raise RuntimeError("Target grid changed across sigma hypotheses; expected a shared 1D darts grid.")

        # JEEDS currently uses a plain Gaussian density around each target when
        # evaluating P(x | t, sigma), even though the simulator wraps executed
        # actions back onto the board. We mirror that production update here so
        # this standalone likelihood table stays aligned with the existing code.
        action_differences = executed_actions[:, None] - target_actions[None, :]  # shape: (N, T)
        gaussian_scale = float(sigma_hypothesis)
        gaussian_coeff = 1.0 / (math.sqrt(2.0 * math.pi) * gaussian_scale)
        pdf_matrix = gaussian_coeff * np.exp(-0.5 * np.square(action_differences / gaussian_scale))  # shape: (N, T)

        # If any observation receives zero density for every possible target
        # under this sigma hypothesis, then the entire sigma row is impossible.
        if np.any(np.sum(pdf_matrix, axis=1) <= 0.0) or np.any(~np.isfinite(pdf_matrix)):
            continue

        for lambda_index, lambda_hypothesis in enumerate(lambda_grid):
            # Softmax normalization trick copied from the JEEDS update: shift by
            # the maximum exponent before taking exp to prevent overflow.
            shifted_values = expected_values * float(lambda_hypothesis)  # shape: (T,)
            shifted_values -= np.max(shifted_values)
            exponentiated_values = np.exp(shifted_values)
            normalization = np.sum(exponentiated_values)

            if normalization <= 0.0 or not np.isfinite(normalization):
                continue

            # For each observed action x_n, JEEDS uses
            #   [sum_t exp(lambda * V_t) * phi(x_n; t, sigma)] / [sum_t exp(lambda * V_t)]
            # as the observation likelihood contribution under (sigma, lambda).
            # The denominator is only the softmax normalizer. The Gaussian term
            # appears inside the numerator's sum, so there is no cancellation.
            weighted_pdfs = pdf_matrix @ exponentiated_values  # shape: (N,)
            observation_updates = weighted_pdfs / normalization  # shape: (N,)

            if np.any(observation_updates <= 0.0) or np.any(~np.isfinite(observation_updates)):
                continue

            log_likelihood_grid[sigma_index, lambda_index] = float(np.sum(np.log(observation_updates)))  # scalar

    return log_likelihood_grid


# -----------------------------------------------------------------------------
# Baseline Methods
# -----------------------------------------------------------------------------

def run_independent_jeeds_baseline(
    log_likelihood_grid: np.ndarray,
    sigma_grid: np.ndarray,
    lambda_grid: np.ndarray,
) -> MethodEstimate:
    """Infer one agent's skill under the independent JEEDS baseline.

    Use a uniform prior over the ``(sigma, lambda)`` grid, so the posterior is
    proportional to the likelihood table computed for this agent. The returned
    estimate reports both posterior means and the MAP grid cell.
    """

    expected_shape = (len(sigma_grid), len(lambda_grid))
    if log_likelihood_grid.shape != expected_shape:
        raise ValueError(
            "log_likelihood_grid has the wrong shape for the provided skill grids: "
            f"{log_likelihood_grid.shape} vs {expected_shape}."
        )

    finite_mask = np.isfinite(log_likelihood_grid)
    if not np.any(finite_mask):
        return MethodEstimate(
            method_name="jeeds",
            status="no_finite_likelihood",
            notes="All entries in the standalone JEEDS log-likelihood grid were non-finite.",
        )

    # Under a uniform prior, the log posterior differs from the log likelihood
    # by an additive constant. We subtract the global maximum before
    # exponentiating so the normalization is stable.
    stabilized_log_posterior = np.full_like(log_likelihood_grid, -np.inf, dtype=float)
    max_log_likelihood = float(np.max(log_likelihood_grid[finite_mask]))
    stabilized_log_posterior[finite_mask] = log_likelihood_grid[finite_mask] - max_log_likelihood

    posterior_unnormalized = np.zeros_like(log_likelihood_grid, dtype=float)
    posterior_unnormalized[finite_mask] = np.exp(stabilized_log_posterior[finite_mask])
    normalization = float(np.sum(posterior_unnormalized))

    if normalization <= 0.0 or not np.isfinite(normalization):
        return MethodEstimate(
            method_name="jeeds",
            status="invalid_posterior_normalization",
            notes="Failed to normalize the standalone JEEDS posterior under a uniform prior.",
        )

    posterior = posterior_unnormalized / normalization  # shape: (S, L)

    sigma_marginal = np.sum(posterior, axis=1)  # shape: (S,)
    lambda_marginal = np.sum(posterior, axis=0)  # shape: (L,)

    posterior_mean_sigma = float(np.dot(sigma_marginal, sigma_grid))
    posterior_mean_lambda = float(np.dot(lambda_marginal, lambda_grid))

    map_index = int(np.argmax(posterior))
    sigma_map_index, lambda_map_index = np.unravel_index(map_index, posterior.shape)
    map_sigma = float(sigma_grid[sigma_map_index])
    map_lambda = float(lambda_grid[lambda_map_index])

    return MethodEstimate(
        method_name="jeeds",
        posterior_mean_sigma=posterior_mean_sigma,
        posterior_mean_lambda=posterior_mean_lambda,
        map_sigma=map_sigma,
        map_lambda=map_lambda,
        status="ok",
        notes="Standalone JEEDS posterior computed with a uniform prior over the skill grid.",
    )


# -----------------------------------------------------------------------------
# Hierarchical Method
# -----------------------------------------------------------------------------

def fit_population_hyperparameters_map(
    config: ExperimentConfig,
    agent_log_likelihoods: Sequence[np.ndarray],
    sigma_grid: np.ndarray,
    lambda_grid: np.ndarray,
) -> dict[str, Any]:
    """Fit the population hyperparameters for the hierarchical model.

    This implementation performs a lightweight empirical-Bayes MAP fit over the
    population parameters by maximizing:

    1. The marginal likelihood of each agent's JEEDS grid under the current
       population prior, plus
    2. The paper's hyperpriors over ``mu``, ``tau_eta``, ``tau_rho``, and
       ``zeta_r``.

    We optimize in an unconstrained parameter space with SciPy:

    - ``tau_eta = exp(log_tau_eta)``
    - ``tau_rho = exp(log_tau_rho)``
    - ``r = tanh(zeta_r)``

    The objective is still approximate relative to the full hierarchical model
    in the paper draft because it relies on the discretized prior described in
    ``build_discrete_hierarchical_prior``. But this gives us a concrete,
    testable MAP fit that can drive the first end-to-end experiments.
    """
    from scipy import optimize

    sigma_grid = np.asarray(sigma_grid, dtype=float)
    lambda_grid = np.asarray(lambda_grid, dtype=float)
    expected_shape = (len(sigma_grid), len(lambda_grid))

    if sigma_grid.ndim != 1 or lambda_grid.ndim != 1:
        raise ValueError("sigma_grid and lambda_grid must both be one-dimensional arrays.")
    if sigma_grid.size == 0 or lambda_grid.size == 0:
        raise ValueError("sigma_grid and lambda_grid must both be non-empty.")

    validated_log_likelihoods: list[np.ndarray] = []
    for agent_index, grid in enumerate(agent_log_likelihoods):
        grid = np.asarray(grid, dtype=float)
        if grid.shape != expected_shape:
            raise ValueError(
                "Each agent log-likelihood grid must match the JEEDS skill-grid shape. "
                f"Agent {agent_index} had shape {grid.shape}; expected {expected_shape}."
            )
        validated_log_likelihoods.append(grid)

    m0 = np.asarray(config.hyperpriors.mean_vector, dtype=float)  # shape: (2,)
    s0 = np.asarray(config.hyperpriors.covariance_matrix, dtype=float)  # shape: (2, 2)
    s0_inverse = np.linalg.inv(s0)
    a_eta = float(config.hyperpriors.a_eta)
    a_rho = float(config.hyperpriors.a_rho)
    m_r = float(config.hyperpriors.m_r)
    s_r = float(config.hyperpriors.s_r)

    def unpack_parameters(parameter_vector: np.ndarray) -> dict[str, Any]:
        """Map unconstrained search coordinates into model parameters."""

        mu_eta, mu_rho, log_tau_eta, log_tau_rho, zeta_r = parameter_vector.tolist()
        tau_eta = math.exp(log_tau_eta)
        tau_rho = math.exp(log_tau_rho)
        correlation = math.tanh(zeta_r)
        covariance = np.array(
            [
                [tau_eta**2, correlation * tau_eta * tau_rho],
                [correlation * tau_eta * tau_rho, tau_rho**2],
            ],
            dtype=float,
        )
        return {
            "mu_eta": float(mu_eta),
            "mu_rho": float(mu_rho),
            "mu": np.array([mu_eta, mu_rho], dtype=float),
            "tau_eta": float(tau_eta),
            "tau_rho": float(tau_rho),
            "zeta_r": float(zeta_r),
            "r": float(correlation),
            "correlation": float(correlation),
            "covariance_matrix": covariance,
        }

    def evaluate_log_posterior(parameter_vector: np.ndarray) -> float:
        """Return the approximate population log posterior at one parameter point."""

        unpacked = unpack_parameters(parameter_vector)

        try:
            discrete_prior = build_discrete_hierarchical_prior(
                unpacked,
                sigma_grid=sigma_grid,
                lambda_grid=lambda_grid,
            )
        except (KeyError, RuntimeError, ValueError, np.linalg.LinAlgError):
            return -np.inf

        positive_prior_mask = discrete_prior > 0.0  # shape: (S, L)
        if not np.any(positive_prior_mask):
            return -np.inf

        log_prior_grid = np.full_like(discrete_prior, -np.inf, dtype=float)
        log_prior_grid[positive_prior_mask] = np.log(discrete_prior[positive_prior_mask])

        marginal_log_likelihood = 0.0
        for log_likelihood_grid in validated_log_likelihoods:
            supported_cells = np.isfinite(log_likelihood_grid) & positive_prior_mask  # shape: (S, L)
            if not np.any(supported_cells):
                return -np.inf

            log_joint_grid = log_likelihood_grid[supported_cells] + log_prior_grid[supported_cells]
            max_log_joint = float(np.max(log_joint_grid))
            if not np.isfinite(max_log_joint):
                return -np.inf

            marginal_log_likelihood += max_log_joint + math.log(
                float(np.sum(np.exp(log_joint_grid - max_log_joint)))
            )

        mu_centered = unpacked["mu"] - m0  # shape: (2,)
        mu_log_prior = -0.5 * float(mu_centered @ s0_inverse @ mu_centered)
        tau_log_prior = -0.5 * (unpacked["tau_eta"] / a_eta) ** 2 - 0.5 * (unpacked["tau_rho"] / a_rho) ** 2
        zeta_log_prior = -0.5 * ((unpacked["zeta_r"] - m_r) / s_r) ** 2

        total_log_posterior = marginal_log_likelihood + mu_log_prior + tau_log_prior + zeta_log_prior
        if not np.isfinite(total_log_posterior):
            return -np.inf
        return float(total_log_posterior)

    initial_parameter_vector = np.array(
        [
            m0[0],
            m0[1],
            math.log(max(a_eta, 1e-6)),
            math.log(max(a_rho, 1e-6)),
            m_r,
        ],
        dtype=float,
    )

    initial_score = evaluate_log_posterior(initial_parameter_vector)
    objective_evaluations = 0

    def objective(parameter_vector: np.ndarray) -> float:
        """Return the negative log posterior for SciPy minimization."""

        nonlocal objective_evaluations
        objective_evaluations += 1

        score = evaluate_log_posterior(parameter_vector)
        if not np.isfinite(score):
            # Powell's method can do awkward arithmetic with infinities while
            # bracketing one-dimensional searches, so use a very large finite
            # penalty for invalid population-parameter proposals.
            return 1e100
        return -float(score)

    optimization_result = optimize.minimize(
        objective,
        initial_parameter_vector,
        method="Powell",
        options={
            "maxiter": 300,
            "xtol": 1e-3,
            "ftol": 1e-3,
            "disp": False,
        },
    )

    final_parameter_vector = np.asarray(optimization_result.x, dtype=float)
    final_score = evaluate_log_posterior(final_parameter_vector)
    objective_evaluations += 1

    # If Powell returns an invalid point, keep the initialization rather than
    # silently emitting unusable hyperparameters.
    if not np.isfinite(final_score):
        final_parameter_vector = initial_parameter_vector.copy()
        final_score = initial_score

    fitted = unpack_parameters(final_parameter_vector)
    fitted["objective_value"] = float(final_score)
    fitted["initial_objective_value"] = float(initial_score)
    fitted["num_objective_evaluations"] = int(max(objective_evaluations, getattr(optimization_result, "nfev", 0)))
    fitted["num_optimizer_iterations"] = int(getattr(optimization_result, "nit", 0))
    fitted["optimization_method"] = "scipy.optimize.minimize(Powell)"
    fitted["converged"] = bool(optimization_result.success)
    fitted["optimizer_message"] = str(optimization_result.message)
    fitted["num_agents"] = len(validated_log_likelihoods)
    fitted["notes"] = (
        "Approximate empirical-Bayes MAP fit over a discretized hierarchical prior "
        "using scipy.optimize.minimize with Powell search in unconstrained coordinates."
    )
    return fitted


def build_discrete_hierarchical_prior(
    fitted_hyperparameters: dict[str, Any],
    sigma_grid: np.ndarray,
    lambda_grid: np.ndarray,
) -> np.ndarray:
    """Discretize the fitted log-space population prior onto the JEEDS grid.

    The fitted population model lives in ``(log sigma, log lambda)`` space, but
    JEEDS inference happens on a discrete grid in ``(sigma, lambda)`` space.
    This function bridges that gap by approximating the probability mass
    associated with each JEEDS grid cell and renormalizing the result to a
    proper discrete prior.

    Implementation note:
    We use a cell-mass approximation rather than exact rectangle integration.
    For each grid point, we:

    1. Construct the surrounding log-space cell boundaries using adjacent grid
       points.
    2. Evaluate the bivariate Normal density at the JEEDS grid point in log
       space.
    3. Multiply by the surrounding cell area in log space.

    This keeps the discretization fully NumPy-based and easy to inspect. If we
    later decide we need exact multivariate Normal rectangle probabilities, we
    can swap this approximation out without changing downstream code.
    """
    sigma_grid = np.asarray(sigma_grid, dtype=float)
    lambda_grid = np.asarray(lambda_grid, dtype=float)

    if sigma_grid.ndim != 1 or lambda_grid.ndim != 1:
        raise ValueError("sigma_grid and lambda_grid must both be one-dimensional arrays.")
    if sigma_grid.size == 0 or lambda_grid.size == 0:
        raise ValueError("sigma_grid and lambda_grid must both be non-empty.")
    if np.any(~np.isfinite(sigma_grid)) or np.any(~np.isfinite(lambda_grid)):
        raise ValueError("sigma_grid and lambda_grid must contain only finite values.")
    if np.any(sigma_grid <= 0.0) or np.any(lambda_grid <= 0.0):
        raise ValueError("sigma_grid and lambda_grid must be strictly positive for log-space discretization.")
    if np.any(np.diff(sigma_grid) <= 0.0) or np.any(np.diff(lambda_grid) <= 0.0):
        raise ValueError("sigma_grid and lambda_grid must be strictly increasing.")

    # Accept a couple of equivalent key layouts so this function stays easy to
    # reuse once the MAP-fitting step is implemented.
    if "mu" in fitted_hyperparameters:
        mu = np.asarray(fitted_hyperparameters["mu"], dtype=float)
    elif "mean_vector" in fitted_hyperparameters:
        mu = np.asarray(fitted_hyperparameters["mean_vector"], dtype=float)
    else:
        mu = np.array(
            [
                fitted_hyperparameters["mu_eta"],
                fitted_hyperparameters["mu_rho"],
            ],
            dtype=float,
        )

    if mu.shape != (2,):
        raise ValueError(f"Population mean must have shape (2,), received {mu.shape}.")

    if "covariance_matrix" in fitted_hyperparameters:
        covariance = np.asarray(fitted_hyperparameters["covariance_matrix"], dtype=float)
    elif "Sigma_z" in fitted_hyperparameters:
        covariance = np.asarray(fitted_hyperparameters["Sigma_z"], dtype=float)
    else:
        tau_eta = float(fitted_hyperparameters["tau_eta"])
        tau_rho = float(fitted_hyperparameters["tau_rho"])
        if "correlation" in fitted_hyperparameters:
            correlation = float(fitted_hyperparameters["correlation"])
        elif "r" in fitted_hyperparameters:
            correlation = float(fitted_hyperparameters["r"])
        elif "zeta_r" in fitted_hyperparameters:
            correlation = math.tanh(float(fitted_hyperparameters["zeta_r"]))
        else:
            raise KeyError(
                "fitted_hyperparameters must include one of: correlation, r, or zeta_r."
            )

        if tau_eta <= 0.0 or tau_rho <= 0.0:
            raise ValueError("tau_eta and tau_rho must both be strictly positive.")
        if not np.isfinite(correlation) or correlation <= -1.0 or correlation >= 1.0:
            raise ValueError("Correlation must be finite and lie strictly between -1 and 1.")

        covariance = np.array(
            [
                [tau_eta**2, correlation * tau_eta * tau_rho],
                [correlation * tau_eta * tau_rho, tau_rho**2],
            ],
            dtype=float,
        )

    if covariance.shape != (2, 2):
        raise ValueError(f"Population covariance must have shape (2, 2), received {covariance.shape}.")
    if np.any(~np.isfinite(covariance)):
        raise ValueError("Population covariance must contain only finite values.")

    determinant = float(np.linalg.det(covariance))
    if determinant <= 0.0 or not np.isfinite(determinant):
        raise ValueError("Population covariance must be positive definite.")

    inverse_covariance = np.linalg.inv(covariance)
    normalization_constant = 1.0 / (2.0 * math.pi * math.sqrt(determinant))

    def log_grid_edges(log_grid: np.ndarray) -> np.ndarray:
        """Construct surrounding cell boundaries in log space."""

        if log_grid.size == 1:
            return np.array([log_grid[0] - 0.5, log_grid[0] + 0.5], dtype=float)

        edges = np.empty(log_grid.size + 1, dtype=float)
        edges[1:-1] = 0.5 * (log_grid[:-1] + log_grid[1:])
        edges[0] = log_grid[0] - 0.5 * (log_grid[1] - log_grid[0])
        edges[-1] = log_grid[-1] + 0.5 * (log_grid[-1] - log_grid[-2])
        return edges

    log_sigma_grid = np.log(sigma_grid)  # shape: (S,)
    log_lambda_grid = np.log(lambda_grid)  # shape: (L,)

    sigma_edges = log_grid_edges(log_sigma_grid)  # shape: (S + 1,)
    lambda_edges = log_grid_edges(log_lambda_grid)  # shape: (L + 1,)
    sigma_cell_widths = np.diff(sigma_edges)  # shape: (S,)
    lambda_cell_widths = np.diff(lambda_edges)  # shape: (L,)

    log_sigma_mesh, log_lambda_mesh = np.meshgrid(
        log_sigma_grid,
        log_lambda_grid,
        indexing="ij",
    )
    log_points = np.stack([log_sigma_mesh, log_lambda_mesh], axis=-1)  # shape: (S, L, 2)
    centered_points = log_points - mu  # shape: (S, L, 2)

    # Evaluate the bivariate Normal density at each JEEDS grid point in
    # log-skill space.
    quadratic_form = np.einsum(
        "...i,ij,...j->...",
        centered_points,
        inverse_covariance,
        centered_points,
    )  # shape: (S, L)
    log_density = math.log(normalization_constant) - 0.5 * quadratic_form  # shape: (S, L)
    density = np.exp(log_density)  # shape: (S, L)

    cell_areas = sigma_cell_widths[:, None] * lambda_cell_widths[None, :]  # shape: (S, L)
    prior_mass = density * cell_areas  # shape: (S, L)

    if np.any(~np.isfinite(prior_mass)) or np.any(prior_mass < 0.0):
        raise RuntimeError("Discrete hierarchical prior contained invalid mass values.")

    total_mass = float(np.sum(prior_mass))
    if total_mass <= 0.0 or not np.isfinite(total_mass):
        raise RuntimeError("Discrete hierarchical prior had zero or non-finite total mass.")

    return prior_mass / total_mass  # shape: (S, L)


def run_hierarchical_estimator(
    log_likelihood_grid: np.ndarray,
    discrete_prior: np.ndarray,
    sigma_grid: np.ndarray,
    lambda_grid: np.ndarray,
) -> MethodEstimate:
    """Infer one agent's skill using the fitted hierarchical prior.

    Once the population-level MAP fit has been discretized onto the JEEDS
    hypothesis grid, posterior inference is almost identical to the
    independent JEEDS baseline. The only difference is that we now combine the
    agent-specific log likelihood table with a non-uniform prior over grid
    cells before normalizing.

    This function intentionally mirrors ``run_independent_jeeds_baseline`` so
    the two posterior computations stay easy to compare.
    """
    expected_shape = (len(sigma_grid), len(lambda_grid))
    if log_likelihood_grid.shape != expected_shape:
        raise ValueError(
            "log_likelihood_grid has the wrong shape for the provided skill grids: "
            f"{log_likelihood_grid.shape} vs {expected_shape}."
        )
    if discrete_prior.shape != expected_shape:
        raise ValueError(
            "discrete_prior has the wrong shape for the provided skill grids: "
            f"{discrete_prior.shape} vs {expected_shape}."
        )

    prior = np.asarray(discrete_prior, dtype=float)
    if np.any(~np.isfinite(prior)) or np.any(prior < 0.0):
        return MethodEstimate(
            method_name="hierarchical",
            status="invalid_prior_values",
            notes="Discrete hierarchical prior contained negative or non-finite values.",
        )

    prior_total = float(np.sum(prior))
    if prior_total <= 0.0 or not np.isfinite(prior_total):
        return MethodEstimate(
            method_name="hierarchical",
            status="invalid_prior_normalization",
            notes="Discrete hierarchical prior had zero or non-finite total mass.",
        )

    # Normalize defensively so the function can accept either a raw mass table
    # from the discretization step or an already normalized prior.
    prior = prior / prior_total  # shape: (S, L)
    positive_prior_mask = prior > 0.0  # shape: (S, L)

    if not np.any(positive_prior_mask):
        return MethodEstimate(
            method_name="hierarchical",
            status="no_prior_support",
            notes="Discrete hierarchical prior assigned zero mass to every grid cell.",
        )

    finite_likelihood_mask = np.isfinite(log_likelihood_grid)  # shape: (S, L)
    posterior_support_mask = finite_likelihood_mask & positive_prior_mask  # shape: (S, L)

    if not np.any(posterior_support_mask):
        return MethodEstimate(
            method_name="hierarchical",
            status="no_finite_posterior_support",
            notes=(
                "No grid cell had both positive hierarchical prior mass and a finite "
                "agent log likelihood."
            ),
        )

    log_prior = np.full_like(prior, -np.inf, dtype=float)
    log_prior[positive_prior_mask] = np.log(prior[positive_prior_mask])

    log_posterior = np.full_like(log_likelihood_grid, -np.inf, dtype=float)
    log_posterior[posterior_support_mask] = (
        log_likelihood_grid[posterior_support_mask] + log_prior[posterior_support_mask]
    )

    # As in the JEEDS baseline path, subtract the largest supported log weight
    # before exponentiating so the posterior normalization stays numerically
    # stable even when likelihoods are very sharp.
    stabilized_log_posterior = np.full_like(log_posterior, -np.inf, dtype=float)
    max_log_posterior = float(np.max(log_posterior[posterior_support_mask]))
    stabilized_log_posterior[posterior_support_mask] = (
        log_posterior[posterior_support_mask] - max_log_posterior
    )

    posterior_unnormalized = np.zeros_like(log_posterior, dtype=float)
    posterior_unnormalized[posterior_support_mask] = np.exp(
        stabilized_log_posterior[posterior_support_mask]
    )
    normalization = float(np.sum(posterior_unnormalized))

    if normalization <= 0.0 or not np.isfinite(normalization):
        return MethodEstimate(
            method_name="hierarchical",
            status="invalid_posterior_normalization",
            notes="Failed to normalize the hierarchical posterior after combining prior and likelihood.",
        )

    posterior = posterior_unnormalized / normalization  # shape: (S, L)

    sigma_marginal = np.sum(posterior, axis=1)  # shape: (S,)
    lambda_marginal = np.sum(posterior, axis=0)  # shape: (L,)

    posterior_mean_sigma = float(np.dot(sigma_marginal, sigma_grid))
    posterior_mean_lambda = float(np.dot(lambda_marginal, lambda_grid))

    map_index = int(np.argmax(posterior))
    sigma_map_index, lambda_map_index = np.unravel_index(map_index, posterior.shape)
    map_sigma = float(sigma_grid[sigma_map_index])
    map_lambda = float(lambda_grid[lambda_map_index])

    return MethodEstimate(
        method_name="hierarchical",
        posterior_mean_sigma=posterior_mean_sigma,
        posterior_mean_lambda=posterior_mean_lambda,
        map_sigma=map_sigma,
        map_lambda=map_lambda,
        status="ok",
        notes=(
            "Hierarchical posterior computed by combining the agent log-likelihood grid "
            "with the supplied discrete population prior."
        ),
    )


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------

def summarize_seed_results(seed_result: SeedResult) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize one seed's agent-level outputs into bucketed and overall rows.

    We summarize posterior-mean estimates for every method that completed with
    ``status == "ok"``. MAP estimates remain in the agent-level CSV as
    diagnostics, but posterior means are the primary estimates for the paper.
    """

    def add_metric(
        container: dict[tuple[str, str, int], list[float]],
        *,
        method: str,
        metric: str,
        count_bucket: int,
        value: float,
    ) -> None:
        """Accumulate one metric value under a method/metric/bucket key."""

        key = (method, metric, count_bucket)
        if key not in container:
            container[key] = []
        container[key].append(value)

    bucket_metrics: dict[tuple[str, str, int], list[float]] = {}
    overall_metrics: dict[tuple[str, str], list[float]] = {}

    for result in seed_result.agent_results:
        method_info = {
            "jeeds": result.jeeds,
            "hierarchical": result.hierarchical,
        }

        for method_name, estimate in method_info.items():
            # Only summarize methods that produced real posterior-mean
            # estimates. Failed methods keep their status in the agent-level
            # CSV, but they should not contribute numeric error rows.
            if estimate.status != "ok":
                continue
            if estimate.posterior_mean_sigma is None or estimate.posterior_mean_lambda is None:
                continue
            if result.lambda_true <= 0.0 or estimate.posterior_mean_lambda <= 0.0:
                continue

            abs_sigma_error = abs(estimate.posterior_mean_sigma - result.sigma_true)
            abs_log10_lambda_error = abs(
                math.log10(estimate.posterior_mean_lambda) - math.log10(result.lambda_true)
            )

            add_metric(
                bucket_metrics,
                method=method_name,
                metric="abs_sigma_error",
                count_bucket=result.count_bucket,
                value=abs_sigma_error,
            )
            add_metric(
                bucket_metrics,
                method=method_name,
                metric="abs_log10_lambda_error",
                count_bucket=result.count_bucket,
                value=abs_log10_lambda_error,
            )

            overall_sigma_key = (method_name, "abs_sigma_error")
            overall_lambda_key = (method_name, "abs_log10_lambda_error")
            overall_metrics.setdefault(overall_sigma_key, []).append(abs_sigma_error)
            overall_metrics.setdefault(overall_lambda_key, []).append(abs_log10_lambda_error)

    summary_by_bucket_rows: list[dict[str, Any]] = []
    for (method_name, metric_name, count_bucket), values in sorted(bucket_metrics.items()):
        summary_by_bucket_rows.append(
            {
                "method": method_name,
                "metric": metric_name,
                "count_bucket": count_bucket,
                "num_agents": len(values),
                "mean": float(np.mean(values)),
                "ci_lower": "",
                "ci_upper": "",
                "notes": (
                    "Seed-level mean over agents with valid posterior-mean estimates. "
                    "Confidence intervals are added during across-seed aggregation."
                ),
            }
        )

    summary_overall_rows: list[dict[str, Any]] = []
    for (method_name, metric_name), values in sorted(overall_metrics.items()):
        summary_overall_rows.append(
            {
                "method": method_name,
                "metric": metric_name,
                "num_agents": len(values),
                "mean": float(np.mean(values)),
                "ci_lower": "",
                "ci_upper": "",
                "notes": (
                    "Seed-level overall mean over agents with valid posterior-mean estimates. "
                    "Confidence intervals are added during across-seed aggregation."
                ),
            }
        )

    return summary_by_bucket_rows, summary_overall_rows


def aggregate_results_across_seeds(
    seed_results: Sequence[SeedResult],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate seed-level summaries into final experiment tables.

    Combine per-seed summary rows by averaging the seed-level means, then add
    normal-approximation 95% confidence intervals across seeds.
    """

    def mean_confidence_interval(
        values: Sequence[float],
    ) -> tuple[float, float]:
        """Return a closed-form 95% CI for the mean of ``values``.

        We treat seeds as the unit of replication and compute the CI from the
        standard error of the seed-level means. With the number of seeds we
        expect to run, the normal approximation should be reasonable.
        """

        values_array = np.asarray(values, dtype=float)
        if values_array.size == 0:
            raise ValueError("Cannot compute a confidence interval from an empty set of values.")
        if values_array.size == 1:
            scalar_value = float(values_array[0])
            return scalar_value, scalar_value

        mean_value = float(np.mean(values_array))
        sample_std = float(np.std(values_array, ddof=1))
        standard_error = sample_std / math.sqrt(values_array.size)
        half_width = 1.96 * standard_error
        return mean_value - half_width, mean_value + half_width

    def clamp_ci_to_metric_support(metric_name: str, ci_lower: float, ci_upper: float) -> tuple[float, float]:
        """Keep reported CIs inside the known support of the metric."""

        if metric_name.startswith("abs_"):
            ci_lower = max(0.0, ci_lower)
        return ci_lower, ci_upper

    if not seed_results:
        return [], []

    bucket_groups: dict[tuple[str, str, int], dict[str, Any]] = {}
    overall_groups: dict[tuple[str, str], dict[str, Any]] = {}

    for seed_result in seed_results:
        for row in seed_result.summary_by_bucket_rows:
            key = (str(row["method"]), str(row["metric"]), int(row["count_bucket"]))
            if key not in bucket_groups:
                bucket_groups[key] = {
                    "means": [],
                    "num_agents": 0,
                    "num_seeds": 0,
                }
            bucket_groups[key]["means"].append(float(row["mean"]))
            bucket_groups[key]["num_agents"] += int(row["num_agents"])
            bucket_groups[key]["num_seeds"] += 1

        for row in seed_result.summary_overall_rows:
            key = (str(row["method"]), str(row["metric"]))
            if key not in overall_groups:
                overall_groups[key] = {
                    "means": [],
                    "num_agents": 0,
                    "num_seeds": 0,
                }
            overall_groups[key]["means"].append(float(row["mean"]))
            overall_groups[key]["num_agents"] += int(row["num_agents"])
            overall_groups[key]["num_seeds"] += 1

    summary_by_bucket_rows: list[dict[str, Any]] = []
    for (method_name, metric_name, count_bucket), info in sorted(bucket_groups.items()):
        mean_values = info["means"]
        ci_lower, ci_upper = mean_confidence_interval(mean_values)
        ci_lower, ci_upper = clamp_ci_to_metric_support(metric_name, ci_lower, ci_upper)
        summary_by_bucket_rows.append(
            {
                "method": method_name,
                "metric": metric_name,
                "count_bucket": count_bucket,
                "num_agents": info["num_agents"],
                "mean": float(np.mean(mean_values)),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "notes": (
                    "Across-seed mean of seed-level summary means with a normal-approximation "
                    f"95% CI over {info['num_seeds']} seeds."
                ),
            }
        )

    summary_overall_rows: list[dict[str, Any]] = []
    for (method_name, metric_name), info in sorted(overall_groups.items()):
        mean_values = info["means"]
        ci_lower, ci_upper = mean_confidence_interval(mean_values)
        ci_lower, ci_upper = clamp_ci_to_metric_support(metric_name, ci_lower, ci_upper)
        summary_overall_rows.append(
            {
                "method": method_name,
                "metric": metric_name,
                "num_agents": info["num_agents"],
                "mean": float(np.mean(mean_values)),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "notes": (
                    "Across-seed mean of seed-level summary means with a normal-approximation "
                    f"95% CI over {info['num_seeds']} seeds."
                ),
            }
        )

    return summary_by_bucket_rows, summary_overall_rows


# -----------------------------------------------------------------------------
# Artifacts
# -----------------------------------------------------------------------------

def _value_or_blank(value: Any) -> Any:
    """Return CSV-friendly blanks for missing estimate values."""

    return "" if value is None else value


def _agent_result_to_row(result: AgentResult) -> dict[str, Any]:
    """Flatten an ``AgentResult`` into the agent-level CSV schema."""

    return {
        "seed": result.seed,
        "agent_id": result.agent_id,
        "count_bucket": result.count_bucket,
        "num_observations": result.num_observations,
        "sigma_true": result.sigma_true,
        "lambda_true": result.lambda_true,
        "jeeds_posterior_mean_sigma": _value_or_blank(result.jeeds.posterior_mean_sigma),
        "jeeds_posterior_mean_lambda": _value_or_blank(result.jeeds.posterior_mean_lambda),
        "jeeds_map_sigma": _value_or_blank(result.jeeds.map_sigma),
        "jeeds_map_lambda": _value_or_blank(result.jeeds.map_lambda),
        "jeeds_status": result.jeeds.status,
        "hierarchical_posterior_mean_sigma": _value_or_blank(result.hierarchical.posterior_mean_sigma),
        "hierarchical_posterior_mean_lambda": _value_or_blank(result.hierarchical.posterior_mean_lambda),
        "hierarchical_map_sigma": _value_or_blank(result.hierarchical.map_sigma),
        "hierarchical_map_lambda": _value_or_blank(result.hierarchical.map_lambda),
        "hierarchical_status": result.hierarchical.status,
        "notes": result.notes,
    }


def write_agent_level_csv(output_path: Path, agent_results: Sequence[AgentResult]) -> None:
    """Write the agent-level CSV schema."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGENT_LEVEL_CSV_HEADER)
        writer.writeheader()
        for result in agent_results:
            writer.writerow(_agent_result_to_row(result))


def _write_dict_rows(output_path: Path, header: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    """Internal helper for writing summary CSVs with a fixed schema."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in header})


def write_summary_csvs(
    output_dir: Path,
    summary_by_bucket_rows: Sequence[dict[str, Any]],
    summary_overall_rows: Sequence[dict[str, Any]],
) -> None:
    """Write the bucketed and overall summary CSV schemas."""

    paths = planned_output_paths(output_dir)
    _write_dict_rows(paths["summary_by_bucket_csv"], SUMMARY_BY_BUCKET_CSV_HEADER, summary_by_bucket_rows)
    _write_dict_rows(paths["summary_overall_csv"], SUMMARY_OVERALL_CSV_HEADER, summary_overall_rows)


def plot_error_by_bucket(output_path: Path, summary_by_bucket_rows: Sequence[dict[str, Any]]) -> None:
    """Create the two-panel figure for sigma and log-lambda error.

    The input rows are the across-seed summaries, so the figure plots the
    reported mean and 95% CI for each method at each observation-count bucket.
    """

    # Import matplotlib lazily so dry-run and non-plotting helper tests do not
    # need to import the plotting stack. Force a non-interactive backend so the
    # same script works on headless SLURM nodes.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    def as_float(value: Any) -> float | None:
        """Convert CSV-ish values to floats, treating blanks as missing."""

        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def as_bucket(value: Any) -> int | None:
        """Convert a summary row's bucket value to an integer label."""

        numeric_value = as_float(value)
        if numeric_value is None:
            return None
        return int(numeric_value)

    metric_panels = [
        (
            "abs_sigma_error",
            "Execution Skill Error by Count Bucket",
            r"|$\hat{\sigma} - \sigma$|",
        ),
        (
            "abs_log10_lambda_error",
            "Decision Skill Error by Count Bucket",
            r"|$\log_{10}\hat{\lambda} - \log_{10}\lambda$|",
        ),
    ]
    method_order = {
        "jeeds": 0,
        "hierarchical": 1,
    }

    parsed_rows: list[dict[str, Any]] = []
    for row in summary_by_bucket_rows:
        bucket = as_bucket(row.get("count_bucket"))
        mean = as_float(row.get("mean"))
        if bucket is None or mean is None:
            continue
        parsed_rows.append(
            {
                "method": str(row.get("method", "")),
                "metric": str(row.get("metric", "")),
                "count_bucket": bucket,
                "mean": mean,
                "ci_lower": as_float(row.get("ci_lower")),
                "ci_upper": as_float(row.get("ci_upper")),
            }
        )

    bucket_values = sorted({row["count_bucket"] for row in parsed_rows})
    bucket_positions = {bucket: index for index, bucket in enumerate(bucket_values)}

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for axis, (metric_name, title, ylabel) in zip(axes, metric_panels):
        axis.set_title(title)
        axis.set_xlabel("Observation count bucket")
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        metric_rows = [row for row in parsed_rows if row["metric"] == metric_name]
        methods = sorted(
            {row["method"] for row in metric_rows},
            key=lambda method: (method_order.get(method, len(method_order)), method),
        )

        if metric_rows and bucket_values:
            for method in methods:
                method_rows = sorted(
                    [row for row in metric_rows if row["method"] == method],
                    key=lambda row: bucket_positions[row["count_bucket"]],
                )

                x_values: list[int] = []
                y_values: list[float] = []
                lower_errors: list[float] = []
                upper_errors: list[float] = []

                for row in method_rows:
                    mean = row["mean"]
                    ci_lower = row["ci_lower"]
                    ci_upper = row["ci_upper"]

                    x_values.append(bucket_positions[row["count_bucket"]])
                    y_values.append(mean)
                    lower_errors.append(max(0.0, mean - ci_lower) if ci_lower is not None else 0.0)
                    upper_errors.append(max(0.0, ci_upper - mean) if ci_upper is not None else 0.0)

                axis.errorbar(
                    x_values,
                    y_values,
                    yerr=np.array([lower_errors, upper_errors], dtype=float),
                    marker="o",
                    capsize=4,
                    linewidth=2,
                    label=method,
                )

            axis.set_xticks(list(bucket_positions.values()))
            axis.set_xticklabels([str(bucket) for bucket in bucket_values])
            axis.legend(title="Method")
        else:
            axis.text(
                0.5,
                0.5,
                f"No rows for {metric_name}",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


# -----------------------------------------------------------------------------
# CLI Entry Point / Orchestration
# -----------------------------------------------------------------------------

def run_single_seed(config: ExperimentConfig, seed: int) -> SeedResult:
    """Run the experiment pipeline for one seed.

    The structure below is intentionally explicit so it is easy to see the
    order in which each seed's simulation and inference steps execute.
    """

    rng = np.random.default_rng(seed)
    reward_surface = sample_reward_surface(rng, config)
    sigma_grid, lambda_grid = build_skill_grids(config)
    agent_truths = sample_true_population_params(rng, config, sigma_grid, lambda_grid)
    observation_counts = assign_observation_counts(config)

    seed_result = SeedResult(
        seed=seed,
        reward_surface=reward_surface,
        agent_truths=agent_truths,
        notes="Seed result includes standalone JEEDS estimates and hierarchical empirical-Bayes estimates.",
    )

    agent_records: list[tuple[AgentTruth, AgentDataset, np.ndarray, MethodEstimate]] = []

    for agent_truth, num_observations in zip(agent_truths, observation_counts):
        dataset = simulate_agent_dataset(
            rng=rng,
            seed=seed,
            config=config,
            reward_surface=reward_surface,
            agent_truth=agent_truth,
            num_observations=num_observations,
            sigma_grid=sigma_grid,
            lambda_grid=lambda_grid,
        )
        seed_result.agent_datasets.append(dataset)

        log_likelihood_grid = compute_agent_log_likelihood_grid(
            config=config,
            reward_surface=reward_surface,
            agent_dataset=dataset,
            sigma_grid=sigma_grid,
            lambda_grid=lambda_grid,
        )
        jeeds_estimate = run_independent_jeeds_baseline(
            log_likelihood_grid=log_likelihood_grid,
            sigma_grid=sigma_grid,
            lambda_grid=lambda_grid,
        )

        agent_records.append((agent_truth, dataset, log_likelihood_grid, jeeds_estimate))

    fitted_hyperparameters = fit_population_hyperparameters_map(
        config=config,
        agent_log_likelihoods=[record[2] for record in agent_records],
        sigma_grid=sigma_grid,
        lambda_grid=lambda_grid,
    )
    discrete_hierarchical_prior = build_discrete_hierarchical_prior(
        fitted_hyperparameters=fitted_hyperparameters,
        sigma_grid=sigma_grid,
        lambda_grid=lambda_grid,
    )

    seed_result.notes = (
        "Seed result includes standalone JEEDS estimates and hierarchical empirical-Bayes estimates. "
        f"Hierarchical population fit status: converged={fitted_hyperparameters.get('converged')}; "
        f"objective={fitted_hyperparameters.get('objective_value')}."
    )

    for agent_truth, dataset, log_likelihood_grid, jeeds_estimate in agent_records:
        hierarchical_estimate = run_hierarchical_estimator(
            log_likelihood_grid=log_likelihood_grid,
            discrete_prior=discrete_hierarchical_prior,
            sigma_grid=sigma_grid,
            lambda_grid=lambda_grid,
        )

        seed_result.agent_results.append(
            AgentResult(
                seed=seed,
                agent_id=agent_truth.agent_id,
                count_bucket=dataset.count_bucket,
                num_observations=dataset.num_observations,
                sigma_true=agent_truth.sigma_true,
                lambda_true=agent_truth.lambda_true,
                jeeds=jeeds_estimate,
                hierarchical=hierarchical_estimate,
                notes=(
                    "Agent result contains standalone JEEDS and hierarchical empirical-Bayes estimates."
                ),
            )
        )

    seed_result.summary_by_bucket_rows, seed_result.summary_overall_rows = summarize_seed_results(seed_result)
    return seed_result


def main(argv: Sequence[str] | None = None) -> int:

    args = parse_args(argv)
    config = build_config_from_args(args)

    if config.dry_run:
        print_dry_run_summary(config)
        return 0

    seed_results: list[SeedResult] = []
    for seed_index, seed in enumerate(config.seed_values, start=1):
        print(f"[hier-darts] Running seed {seed_index}/{config.num_seeds}: {seed}", flush=True)
        seed_results.append(run_single_seed(config, seed))

    output_paths = planned_output_paths(config.output_dir)
    all_agent_results = [result for seed_result in seed_results for result in seed_result.agent_results]
    summary_by_bucket_rows, summary_overall_rows = aggregate_results_across_seeds(seed_results)

    write_agent_level_csv(output_paths["agent_level_csv"], all_agent_results)
    write_summary_csvs(config.output_dir, summary_by_bucket_rows, summary_overall_rows)
    plot_error_by_bucket(output_paths["error_plot"], summary_by_bucket_rows)
    print(f"[hier-darts] Wrote results to {config.output_dir.resolve()}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
