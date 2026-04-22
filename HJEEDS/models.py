# This file still requires human verification. Delete this comment when done.
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# These dataclasses are the shared vocabulary of the package.  They make the
# experiment pipeline easier to read because each stage passes named objects
# instead of loose dictionaries or tuples.


@dataclass(frozen=True)
class HyperpriorConfig:
    """Container for the population-level hyperpriors from the paper draft."""

    mean_vector: tuple[float, float]
    covariance_diagonal: tuple[float, float]
    log_tau_eta_mean: float
    log_tau_eta_sd: float
    log_tau_rho_mean: float
    log_tau_rho_sd: float
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


@dataclass(frozen=True)
class ExperimentConfig:
    """All configuration needed to define one sweep of the experiment."""

    # The fields below capture both the experimental design
    # (number of seeds, count buckets, grid size) and the modeling assumptions
    # (true population and hyperpriors).  Keeping them together means every
    # function downstream can operate on a single immutable config object.
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

    # The code keeps both original-scale and log-scale versions because the
    # simulator works on the original scale while the hierarchical population
    # model is defined in log-skill space.
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

    # ``status`` lets downstream code distinguish valid estimates from failed
    # inference paths without throwing away the rest of the agent's record.
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

    # This object accumulates everything produced for one random seed so the
    # final aggregation step can treat seeds as the unit of replication.
    seed: int
    reward_surface: tuple[float, ...]
    agent_truths: list[AgentTruth] = field(default_factory=list)
    agent_datasets: list[AgentDataset] = field(default_factory=list)
    agent_results: list[AgentResult] = field(default_factory=list)
    summary_by_bucket_rows: list[dict[str, Any]] = field(default_factory=list)
    summary_overall_rows: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""
