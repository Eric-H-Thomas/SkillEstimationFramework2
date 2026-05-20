# This file has been fully verified by a human researcher as of 05/08/26 at 9:59 AM MT.
from __future__ import annotations

import math

import numpy as np


# Helpers for the JEEDS rationality-percentage metric.  The value reported here
# is in percentage points: 0 means uniform-random target choice and 100 means
# always choosing a maximum-expected-value target for the supplied reward
# surface and execution skill.


def rationality_percent_from_expected_values(
    expected_values: np.ndarray,
    log_lambda: float,
) -> float | None:
    """Map a log-lambda value to rationality percentage points."""

    # ``expected_values`` is the EV curve over the discrete aiming grid for a
    # fixed reward surface and execution-noise value.  Each element answers:
    # "if the agent intentionally aims at this target, what reward should they
    # expect after execution noise is applied?"
    expected_values = np.asarray(expected_values, dtype=float)
    if expected_values.ndim != 1 or expected_values.size == 0:
        raise ValueError("expected_values must be a non-empty one-dimensional array.")
    if np.any(~np.isfinite(expected_values)):
        raise ValueError("expected_values must contain only finite values.")
    if not math.isfinite(log_lambda):
        return None

    maximum_reward = float(np.max(expected_values))
    uniform_reward = float(np.mean(expected_values))

    # The paper's denominator is the reward gap between perfect target choice
    # and uniformly random target choice.  If every target has the same EV, that
    # gap is zero and rationality percentage is not identifiable for this board.
    denominator = maximum_reward - uniform_reward
    if denominator <= 0.0 or not math.isfinite(denominator):
        return None

    # H-JEEDS stores decision skill canonically as log(lambda).  The softmax
    # policy itself still needs raw lambda, so this is the intentional boundary
    # where we return to the original scale.
    raw_lambda = math.exp(log_lambda)

    # Convert the decision skill into the agent's full target-choice policy.
    # Subtracting the maximum is the standard numerically stable softmax trick:
    # it keeps the same probabilities while avoiding unnecessarily huge
    # exponentials.
    shifted_values = raw_lambda * expected_values
    shifted_values -= np.max(shifted_values)
    target_weights = np.exp(shifted_values)
    normalization = float(np.sum(target_weights))
    if normalization <= 0.0 or not math.isfinite(normalization):
        return None

    # Because the full policy is known in simulation, use the distribution-level
    # version of the paper's r_A: expected reward under all target probabilities,
    # not just the reward of one sampled intended target.
    target_probabilities = target_weights / normalization
    agent_reward = float(np.dot(target_probabilities, expected_values))
    return 100.0 * (agent_reward - uniform_reward) / denominator


def compute_expected_values_for_rationality(
    reward_surface: tuple[float, ...],
    sigma: float,
    delta: float,
    environment: str = "1d",
) -> np.ndarray:
    """Return the expected-value curve used to evaluate rationality percentage."""

    # Import lazily for the same reason as the simulation and likelihood paths:
    # dry-run/config code should not need to import the full darts environment.
    from HJEEDS.environment_adapters import get_environment_domain

    domain = get_environment_domain(environment)
    expected_values, _actions = domain.compute_expected_value_curve(reward_surface, sigma, delta)
    return np.asarray(expected_values, dtype=float)
