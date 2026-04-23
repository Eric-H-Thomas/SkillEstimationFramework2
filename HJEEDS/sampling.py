# This file has been fully verified by a human researcher as of 04/23/26 at 9:49 AM MT.

from __future__ import annotations
import math
import numpy as np
from .models import AgentDataset, AgentTruth, ExperimentConfig


# This module is responsible for the synthetic-data side of the experiment:
# sampling the reward surfaces, building the skill grids, sampling the
# latent demonstrator population, and generating observed throws that match
# the JEEDS generative assumptions.


def sample_reward_surface(rng: np.random.Generator, config: ExperimentConfig) -> tuple[float, ...]:
    """Sample a single 1D darts reward surface.

    We intentionally keep one reward surface fixed for all demonstrators within
    a seed. That isolates the skill-estimation question from environment
    variability.
    """

    # Import lazily so ``--dry-run`` can validate the experiment structure even
    # when the local Python environment is missing the heavier scientific
    # dependencies required by the full darts environment module.
    from Environments.Darts.RandomDarts import darts

    # The darts helper returns a list because the original environment code can
    # generate multiple random boards at once.  Here we always request exactly
    # one board and then freeze it for the whole seed.
    states = darts.generate_random_states(
        rng,
        config.min_success_regions,
        config.max_success_regions,
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

    # Execution skill is sampled on a linear grid while decision skill uses a
    # geometric grid.  That asymmetry mirrors the legacy JEEDS setup, where
    # lambda spans several orders of magnitude more naturally than sigma.
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
    log_sigma_min, log_sigma_max = math.log(sigma_min), math.log(sigma_max)
    log_lambda_min, log_lambda_max = math.log(lambda_min), math.log(lambda_max)

    truths: list[AgentTruth] = []
    covariance = config.true_population.covariance_matrix
    mean_vector = config.true_population.mean_vector

    for agent_id in range(config.num_agents):
        # We keep the simulated truth inside the estimator support so later
        # comparisons do not get dominated by pure grid truncation artifacts.
        for _attempt in range(10_000):
            eta, rho = rng.multivariate_normal(mean_vector, covariance)

            if log_sigma_min <= float(eta) <= log_sigma_max and log_lambda_min <= float(rho) <= log_lambda_max:
                truths.append(
                    AgentTruth(
                        agent_id=agent_id,
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

    # The bucket design is deterministic by construction: the first
    # ``agents_per_bucket`` agents get the first bucket, the next block gets
    # the next bucket, and so on.  That makes inspection/debugging easier.
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

    # This is the decision-making part of the generative model from the paper:
    # convert expected values into probabilities over intended targets using a
    # lambda-controlled softmax.
    #
    # We subtract the maximum before exponentiating to keep the probabilities
    # numerically stable for large lambda values.
    scaled_values = agent_truth.lambda_true * expected_values
    scaled_values -= np.max(scaled_values)
    target_probabilities = np.exp(scaled_values)
    target_probabilities /= np.sum(target_probabilities)

    intended_targets = rng.choice(actions, size=num_observations, p=target_probabilities)

    # This is the execution-skill part of the model: after a target is chosen,
    # the observed action is a noisy perturbation of that intended target.
    #
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
