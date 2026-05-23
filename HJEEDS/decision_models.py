# This file has been fully edited by a human researcher as of 05/22/26 at 6:01 PM MDT.
"""Decision-model metadata for H-JEEDS simulator misspecification studies."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .rationality import rationality_percent_from_expected_values


SOFTMAX_DECISION_MODEL_SLUG = "softmax"
RATIONAL_DECISION_MODEL_SLUG = "rational"
FLIP_DECISION_MODEL_SLUG = "flip"
DECEPTIVE_DECISION_MODEL_SLUG = "deceptive"


@dataclass(frozen=True)
class DecisionModelSpec:
    """Metadata for one true decision-making model condition."""

    slug: str
    label: str
    description: str


DECISION_MODEL_SPECS = (
    DecisionModelSpec(
        slug=SOFTMAX_DECISION_MODEL_SLUG,
        label="Softmax",
        description="Matched softmax policy used by the default H-JEEDS simulator",
    ),
    DecisionModelSpec(
        slug=RATIONAL_DECISION_MODEL_SLUG,
        label="Rational",
        description="Deterministic optimal target choice for the agent's execution skill",
    ),
    DecisionModelSpec(
        slug=FLIP_DECISION_MODEL_SLUG,
        label="Flip",
        description="Epsilon-greedy style policy adapted from the original JEEDS experiments",
    ),
    DecisionModelSpec(
        slug=DECEPTIVE_DECISION_MODEL_SLUG,
        label="Deceptive",
        description="Structured non-random suboptimal policy adapted from the original JEEDS experiments",
    ),
)

DECISION_MODEL_BY_SLUG = {model.slug: model for model in DECISION_MODEL_SPECS}
DEFAULT_DECISION_MODEL_SLUGS = tuple(model.slug for model in DECISION_MODEL_SPECS)


def get_decision_model_spec(decision_model_slug: str) -> DecisionModelSpec:
    """Return metadata for one decision-making model."""

    try:
        return DECISION_MODEL_BY_SLUG[decision_model_slug]
    except KeyError as exc:
        allowed = ", ".join(DEFAULT_DECISION_MODEL_SLUGS)
        raise ValueError(
            f"Unknown decision model '{decision_model_slug}'. Expected one of: {allowed}."
        ) from exc


def decision_model_folder_slug(decision_model_slug: str) -> str:
    """Return a stable folder slug for one decision-making model."""

    _ = get_decision_model_spec(decision_model_slug) # Validate decision-model slug
    return f"decision_model_{decision_model_slug}"


def decision_model_metadata_row(decision_model_slug: str) -> dict[str, str]:
    """Return CSV metadata for one decision-making model."""

    decision_model = get_decision_model_spec(decision_model_slug)
    return {
        "decision_model_slug": decision_model.slug,
        "decision_model_label": decision_model.label,
        "decision_model_description": decision_model.description,
    }


# Convert the positive H-JEEDS lambda scale into the probability scale used by Flip
def _lambda_to_rationality_probability(expected_values: np.ndarray, lambda_true: float) -> float:
    """Map raw lambda to a paper-aligned rational-action probability."""

    # Treat lambda=0 as the uniform-random policy endpoint
    if lambda_true == 0.0:
        return 0.0
    # Reuse the paper's rationality-percent metric, which expects log(lambda)
    rationality_percent = rationality_percent_from_expected_values(expected_values, math.log(lambda_true))
    # If the reward gap is zero, rationality is not identifiable and P does not affect reward quality
    if rationality_percent is None:
        return 0.0
    # Convert percentage points into a unit-interval probability and guard against tiny numerical spillover
    return float(np.clip(rationality_percent / 100.0, 0.0, 1.0))


# Find the target actions that are tied for maximal expected reward
def _optimal_actions(actions: np.ndarray, expected_values: np.ndarray) -> np.ndarray:
    """Return actions tied for maximal expected value."""

    # Compute the highest expected reward available on the target grid
    best_expected_value = np.max(expected_values)
    # Mark every grid action whose expected reward is tied with the maximum up to floating-point noise
    optimal_mask = np.isclose(expected_values, best_expected_value, rtol=1e-12, atol=1e-12)
    # Keep only the actions whose corresponding mask value is True
    optimal_actions = actions[optimal_mask]
    # Fail loudly if numerical weirdness somehow produced no optimal action
    if optimal_actions.size == 0:
        raise RuntimeError("No optimal actions found.")
    # Return the possibly multi-action set of rational choices
    return optimal_actions


# Find deceptive actions that are acceptable but as far from optimal actions as possible
def _deceptive_actions(
    actions: np.ndarray,
    expected_values: np.ndarray,
    rationality_probability: float,
) -> np.ndarray:
    """Return acceptable actions farthest from the rational target set."""

    # At full rationality, the acceptable set should only contain optimal actions
    if rationality_probability >= 1.0:
        return _optimal_actions(actions, expected_values)
    # Compute the mean expected reward over uniformly random target selection
    mean_expected_value = float(np.mean(expected_values))
    # Compute the best expected reward available on the target grid
    best_expected_value = float(np.max(expected_values))
    # Interpolate between mean reward and best reward using the rationality-percent scale
    minimum_acceptable_value = mean_expected_value + rationality_probability * (
        best_expected_value - mean_expected_value
    )
    # Keep actions whose expected reward is acceptable under the deceptive threshold
    acceptable_mask = expected_values >= minimum_acceptable_value
    # Fail clearly if the thresholding logic produces no acceptable actions
    if not np.any(acceptable_mask):
        raise RuntimeError(
            "Deceptive decision model found no acceptable actions for "
            f"rationality_probability={rationality_probability} and "
            f"minimum_acceptable_value={minimum_acceptable_value}."
        )
    # Extract the actions that satisfy the acceptable reward threshold
    acceptable_actions = actions[acceptable_mask]
    # Extract the rational target set used as the distance reference
    optimal_actions = _optimal_actions(actions, expected_values)
    # Compute every acceptable action's distance to its nearest rational action
    distances_to_optimal = np.min(np.abs(acceptable_actions[:, None] - optimal_actions[None, :]), axis=1)
    # Find the largest distance among acceptable actions
    farthest_distance = float(np.max(distances_to_optimal))
    # Mark every acceptable action tied for farthest distance up to floating-point noise
    farthest_mask = np.isclose(distances_to_optimal, farthest_distance, rtol=1e-12, atol=1e-12)
    # Return all tied deceptive choices so the caller can sample among them
    return acceptable_actions[farthest_mask]


def sample_intended_targets_for_decision_model(
    *,
    rng: np.random.Generator,
    decision_model_slug: str,
    actions: np.ndarray,
    expected_values: np.ndarray,
    lambda_true: float,
    num_observations: int,
) -> np.ndarray:
    """Sample intended targets from one true decision-making model."""

    decision_model = get_decision_model_spec(decision_model_slug)
    actions = np.asarray(actions, dtype=float)
    expected_values = np.asarray(expected_values, dtype=float)

    if num_observations <= 0:
        raise ValueError(f"num_observations must be positive. Received {num_observations}.")
    if actions.shape != expected_values.shape:
        raise ValueError(
            "actions and expected_values must have the same shape. "
            f"Received {actions.shape} and {expected_values.shape}."
        )
    if actions.size == 0:
        raise ValueError("actions and expected_values must be nonempty.")
    if not np.all(np.isfinite(actions)):
        raise ValueError("actions must contain only finite values.")
    if not np.all(np.isfinite(expected_values)):
        raise ValueError("expected_values must contain only finite values.")
    if not np.isfinite(lambda_true) or lambda_true < 0.0:
        raise ValueError(f"lambda_true must be finite and nonnegative. Received {lambda_true}.")

    if decision_model.slug == SOFTMAX_DECISION_MODEL_SLUG:
        # This is the decision-making part of the generative model from the paper:
        # convert expected values into probabilities over intended targets using a
        # lambda-controlled softmax.
        #
        # We subtract the maximum before exponentiating to keep the probabilities
        # numerically stable for large lambda values.
        scaled_values = float(lambda_true) * expected_values
        scaled_values -= np.max(scaled_values)
        target_probabilities = np.exp(scaled_values)
        target_probabilities /= np.sum(target_probabilities)
        return rng.choice(actions, size=num_observations, p=target_probabilities)

    if decision_model.slug == RATIONAL_DECISION_MODEL_SLUG:
        return rng.choice(_optimal_actions(actions, expected_values), size=num_observations)

    # FlipAgent: rational with probability P, otherwise random
    if decision_model.slug == FLIP_DECISION_MODEL_SLUG:
        # Convert lambda into the paper's rationality-percent value and use that as the Flip probability P
        probability_rational = _lambda_to_rationality_probability(expected_values, float(lambda_true))
        # Pre-sample the rational target for every observation
        rational_targets = rng.choice(_optimal_actions(actions, expected_values), size=num_observations)
        # Pre-sample the random target for every observation
        random_targets = rng.choice(actions, size=num_observations)
        # Draw one Bernoulli-style decision per observation to choose rational vs random behavior
        rational_draws = rng.random(num_observations) < probability_rational
        # Select the rational target where the draw is True, otherwise select the random target
        return np.where(rational_draws, rational_targets, random_targets)

    # Deceptive agent: choose acceptable-reward actions far from rational targets
    if decision_model.slug == DECEPTIVE_DECISION_MODEL_SLUG:
        # Convert lambda into the paper's rationality-percent value for the acceptable-reward threshold
        rationality_probability = _lambda_to_rationality_probability(expected_values, float(lambda_true))
        # Find acceptable actions farthest from the rational target set
        deceptive_actions = _deceptive_actions(actions, expected_values, rationality_probability)
        # Sample uniformly among tied deceptive actions
        return rng.choice(deceptive_actions, size=num_observations)

    raise ValueError(f"Unsupported decision model: {decision_model.slug}.")
