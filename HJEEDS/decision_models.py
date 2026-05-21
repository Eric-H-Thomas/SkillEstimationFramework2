# This file has been fully reviewed by a human researcher as of 05/20/26 at 1:36 PM MT.
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

    # TODO: Move the current softmax sampling code out of sampling.py and into this dispatcher
    # TODO: Implement rational, flip, and deceptive policies with definitions matching the JEEDS paper
    # TODO: Decide whether flip/deceptive use lambda_true directly or a calibrated transform of log-lambda
    # TODO: Add tests showing softmax reproduces the current simulator exactly before changing callers
    _ = (rng, actions, expected_values, lambda_true, num_observations)
    decision_model = get_decision_model_spec(decision_model_slug)
    raise NotImplementedError(
        "Decision-model sampling is scaffolded but not implemented yet. "
        f"Requested true model: {decision_model.slug}."
    )
