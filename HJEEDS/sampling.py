# This file has been fully edited by a human researcher as of 05/21/26 at 12:22 PM MDT.

from __future__ import annotations
import math
import numpy as np
from . import darts_environment as darts
from .decision_models import sample_intended_targets_for_decision_model
from .models import AgentDataset, AgentTruth, ExperimentConfig
from .population_shapes import sample_log_skill_profile


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

    # The darts helper returns a list because the environment code can
    # generate multiple random boards at once.  Here we always request exactly
    # one board and then freeze it for the whole seed.
    reward_surfaces = darts.generate_random_states(
        rng=rng,
        min_success_region_count=config.min_success_regions,
        max_success_region_count=config.max_success_regions,
        num_reward_surfaces=1,
        min_boundary_spacing=config.min_region_width,
    )
    if not reward_surfaces:
        raise RuntimeError("Failed to sample a 1D darts reward surface.")
    return tuple(float(boundary) for boundary in reward_surfaces[0])


def build_skill_grids(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Construct the JEEDS execution-skill grid and canonical log-lambda grid.

    The user still specifies the lambda bounds on the original scale because
    those are the values used inside the softmax policy. Internally, however,
    H-JEEDS carries the decision-skill axis in natural-log space so posterior
    summaries and hierarchical priors share the same geometry.
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

    # Execution skill still uses a linear raw-sigma grid. Decision skill is
    # stored as equally spaced log-lambda values, which correspond to a
    # geometrically spaced raw-lambda grid once exponentiated.
    sigma_grid = np.linspace(config.sigma_min, config.sigma_max, config.num_sigma_grid, dtype=float)
    log_lambda_grid = np.linspace(
        np.log(config.lambda_min),
        np.log(config.lambda_max),
        config.num_lambda_grid,
        dtype=float,
    )
    return sigma_grid, log_lambda_grid


def sample_true_population_params(
    rng: np.random.Generator,
    config: ExperimentConfig,
    sigma_grid: np.ndarray,
    log_lambda_grid: np.ndarray,
) -> list[AgentTruth]:
    """Sample the true demonstrator skill profiles for one seed.

    The simulated truth is rejection-sampled to stay inside the estimator grid
    so early experiments are not dominated by grid truncation artifacts.
    """

    sigma_min, sigma_max = float(np.min(sigma_grid)), float(np.max(sigma_grid))
    log_sigma_min, log_sigma_max = math.log(sigma_min), math.log(sigma_max)
    log_lambda_min = float(np.min(log_lambda_grid))
    log_lambda_max = float(np.max(log_lambda_grid))

    truths: list[AgentTruth] = []
    for agent_id in range(config.num_agents):
        # We keep the simulated truth inside the estimator support so later
        # comparisons do not get dominated by pure grid truncation artifacts.
        for _attempt in range(10_000):
            eta, rho = sample_log_skill_profile(rng, config.true_population)

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
    log_lambda_grid: np.ndarray,
) -> AgentDataset:
    """Simulate one demonstrator's observed throws.

    Simulate one demonstrator using the JEEDS-style generative story.

    The intended targets are sampled from a softmax policy over the 1D darts
    target grid, where expected values are computed under the demonstrator's
    true execution skill. Each intended target is then perturbed by Gaussian
    execution noise to produce the observed action.

    The simulator follows the same broad modeling assumptions used by the
    likelihood path so the generated data and estimators are matched in this
    first experiment.
    """

    if num_observations <= 0:
        raise ValueError(f"num_observations must be positive. Received {num_observations}.")
    if agent_truth.sigma_true < float(np.min(sigma_grid)) or agent_truth.sigma_true > float(np.max(sigma_grid)):
        raise ValueError(
            f"True sigma {agent_truth.sigma_true} lies outside the provided sigma grid "
            f"[{float(np.min(sigma_grid))}, {float(np.max(sigma_grid))}]."
        )
    if (
        agent_truth.log_lambda_true < float(np.min(log_lambda_grid))
        or agent_truth.log_lambda_true > float(np.max(log_lambda_grid))
    ):
        raise ValueError(
            f"True log_lambda {agent_truth.log_lambda_true} lies outside the provided log-lambda grid "
            f"[{float(np.min(log_lambda_grid))}, {float(np.max(log_lambda_grid))}]."
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

    intended_targets = sample_intended_targets_for_decision_model(
        rng=rng,
        decision_model_slug=config.true_decision_model_slug,
        actions=actions,
        expected_values=expected_values,
        lambda_true=agent_truth.lambda_true,
        num_observations=num_observations,
    )

    # This is the execution-skill part of the model: after a target is chosen,
    # the observed action is a noisy perturbation of that intended target.  The
    # final-paper 1D darts model does not wrap executions at the board edge, so
    # noisy actions can land outside [-BOARD_LIMIT, BOARD_LIMIT].
    executed_actions = np.array(
        [
            darts.sample_noisy_action(
                rng=rng,
                execution_noise_sd=agent_truth.sigma_true,
                intended_target=float(target),
            )
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
            f"Simulated with the {config.true_decision_model_slug} true decision model over the "
            "1D darts target grid and non-wrapped Gaussian execution noise."
        ),
    )
