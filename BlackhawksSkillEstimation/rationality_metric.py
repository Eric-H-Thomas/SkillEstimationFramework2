"""JEEDS rationality-percentage metric (0–100 scale) for Blackhawks hockey.

Percentage points: 0 means uniform-random target choice and 100 means always
choosing a maximum-expected-value target for the supplied reward surface and
execution skill. Ported from H-JEEDS ``rationality.py`` (main branch).
"""
from __future__ import annotations

import math

import numpy as np


def rationality_percent_from_expected_values(
    expected_values: np.ndarray,
    log_lambda: float,
) -> float | None:
    """Map a log-lambda value to rationality percentage points."""
    expected_values = np.asarray(expected_values, dtype=float)
    if expected_values.ndim != 1 or expected_values.size == 0:
        raise ValueError("expected_values must be a non-empty one-dimensional array.")
    if np.any(~np.isfinite(expected_values)):
        raise ValueError("expected_values must contain only finite values.")
    if not math.isfinite(log_lambda):
        return None

    maximum_reward = float(np.max(expected_values))
    uniform_reward = float(np.mean(expected_values))

    denominator = maximum_reward - uniform_reward
    if denominator <= 0.0 or not math.isfinite(denominator):
        return None

    raw_lambda = math.exp(log_lambda)

    shifted_values = raw_lambda * expected_values
    shifted_values -= np.max(shifted_values)
    target_weights = np.exp(shifted_values)
    normalization = float(np.sum(target_weights))
    if normalization <= 0.0 or not math.isfinite(normalization):
        return None

    target_probabilities = target_weights / normalization
    agent_reward = float(np.dot(target_probabilities, expected_values))
    return 100.0 * (agent_reward - uniform_reward) / denominator


def rationality_percent_from_expected_values_raw_lambda(
    expected_values: np.ndarray,
    lambda_raw: float,
) -> float | None:
    """Map raw lambda (hockey JEEDS scale) to rationality percentage points."""
    if not math.isfinite(lambda_raw) or lambda_raw <= 0.0:
        return None
    return rationality_percent_from_expected_values(expected_values, math.log(lambda_raw))
