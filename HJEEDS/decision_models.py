# This file has been fully edited by a human researcher as of 05/21/26 at 10:51 AM MDT.
"""Decision-model metadata for H-JEEDS simulator misspecification studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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

    # TODO: Implement rational, flip, and deceptive policies with definitions matching the JEEDS paper
    # TODO: Decide whether flip/deceptive use lambda_true directly or a calibrated transform of log-lambda
    raise NotImplementedError(
        "Decision-model sampling is scaffolded but not implemented yet. "
        f"Requested true model: {decision_model.slug}."
    )
