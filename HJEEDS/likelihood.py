# This file has been fully verified by a human researcher as of 05/22/26 at 2:29 PM MT.

from __future__ import annotations
import math
import numpy as np
from . import darts_environment as darts
from .models import AgentDataset, ExperimentConfig

# This module converts one demonstrator's observed actions into the JEEDS
# log-likelihood grid over all candidate ``(sigma, log lambda)`` grid cells.
# Raw lambda values are only recovered locally when the softmax policy needs
# them. The resulting grid is the shared input to both the independent baseline
# and the hierarchical variant.


def compute_agent_log_likelihood_grid(
    config: ExperimentConfig,
    reward_surface: tuple[float, ...],
    agent_dataset: AgentDataset,
    sigma_grid: np.ndarray,
    log_lambda_grid: np.ndarray,
) -> np.ndarray:
    """Compute the standalone JEEDS log-likelihood table for one agent.

    This mirrors the production JEEDS update rule in ``Estimators/joint.py``:
    for each execution-skill hypothesis we compute the expected-value curve and,
    for each decision-skill hypothesis, evaluate the per-observation update
    term obtained after marginalizing over latent intended targets.

    The result is a grid of log-likelihoods with shape
    ``(len(sigma_grid), len(log_lambda_grid))``. The columns are indexed by
    log-lambda hypotheses, even though the softmax inside the likelihood still
    needs the corresponding raw lambda values.
    """

    if agent_dataset.executed_actions is None:
        raise ValueError("Agent dataset must contain executed_actions before likelihood evaluation.")
    if agent_dataset.executed_actions.shape[0] != agent_dataset.num_observations:
        raise ValueError(
            "Length of executed_actions does not match num_observations: "
            f"{agent_dataset.executed_actions.shape[0]} vs {agent_dataset.num_observations}."
        )

    executed_actions = np.asarray(agent_dataset.executed_actions, dtype=object if config.environment == "2d" else float)
    # Start with ``-inf`` everywhere so impossible parameter pairs naturally
    # remain excluded unless a later calculation supplies a valid likelihood.
    log_lambda_grid = np.asarray(log_lambda_grid, dtype=float)
    raw_lambda_grid = np.exp(log_lambda_grid)
    log_likelihood_grid = np.full((len(sigma_grid), len(log_lambda_grid)), -np.inf, dtype=float)

    reference_targets: np.ndarray | None = None
    cached_action_differences: np.ndarray | None = None

    # Only create the 2D domain adapter if needed
    domain = None
    if config.environment == "2d":
        from .environment_adapters import get_environment_domain
        domain = get_environment_domain(config.environment)

    for sigma_index, sigma_hypothesis in enumerate(sigma_grid):
        # First hold execution skill fixed. Under that sigma hypothesis we can
        # compute the target grid and Gaussian noise model once, then
        # reuse those pieces across all log-lambda grid cells.
        if config.environment == "1d":
            expected_values, target_actions = darts.compute_expected_value_curve(
                reward_surface,
                float(sigma_hypothesis),
                config.delta,
            )
        else:
            expected_values, target_actions = domain.compute_expected_value_curve(
                reward_surface,
                float(sigma_hypothesis),
                config.delta,
            )

        expected_values = np.asarray(expected_values, dtype=float)  # shape: (T,)
        target_actions = np.asarray(target_actions, dtype=object if config.environment == "2d" else float)  # shape: (T,)

        # Basic sanity checks for expected-values / target grid
        if expected_values.size == 0:
            raise RuntimeError("Expected-value computation returned an empty target grid.")
        if expected_values.shape != target_actions.shape:
            raise RuntimeError(
                "Expected-value computation returned mismatched arrays for values and actions: "
                f"{expected_values.shape} vs {target_actions.shape}."
            )

        # In 1D darts, the target grid should be shared across sigma hypotheses.
        # We verify that explicitly so later code can safely assume a common grid.
        if reference_targets is None:
            reference_targets = target_actions
        elif config.environment == "1d":
            if not np.allclose(reference_targets, target_actions):
                raise RuntimeError("Target grid changed across sigma hypotheses; expected a shared 1D darts grid.")
        elif not np.array_equal(reference_targets, target_actions):
            raise RuntimeError("Target grid changed across sigma hypotheses; expected a shared 2D darts grid.")

        if config.environment == "1d":
            action_differences = executed_actions[:, None] - target_actions[None, :]  # shape: (N, T)
        elif config.environment == "2d":
            if cached_action_differences is None:
                cached_action_differences = np.array(
                    [
                        [
                            domain.compute_action_difference(exec_action, target_action)
                            for target_action in reference_targets
                        ]
                        for exec_action in executed_actions
                    ],
                    dtype=float,
                )
            action_differences = cached_action_differences
        else:
            raise ValueError(f"Unknown environment: {config.environment}")

        gaussian_scale = float(sigma_hypothesis)
        if config.environment == "2d":
            gaussian_coeff = 1.0 / (2.0 * math.pi * gaussian_scale**2)
        else:
            gaussian_coeff = 1.0 / (math.sqrt(2.0 * math.pi) * gaussian_scale)
        pdf_matrix = gaussian_coeff * np.exp(-0.5 * np.square(action_differences / gaussian_scale))  # shape: (N, T)

        # If any observation receives zero density for every possible target
        # under this sigma hypothesis, then the entire sigma row is impossible.
        if np.any(np.sum(pdf_matrix, axis=1) <= 0.0) or np.any(~np.isfinite(pdf_matrix)):
            continue

        for lambda_index, lambda_hypothesis in enumerate(raw_lambda_grid):
            # The grid cell is identified by its log-lambda coordinate, but the
            # softmax itself still requires the raw lambda value recovered by
            # exponentiating that coordinate.
            shifted_values = expected_values * float(lambda_hypothesis)  # shape: (T,)
            shifted_values -= np.max(shifted_values)
            exponentiated_values = np.exp(shifted_values)
            normalization = np.sum(exponentiated_values)

            if normalization <= 0.0 or not np.isfinite(normalization):
                continue

            # For each observed action x_n, JEEDS uses
            #   [sum_t exp(lambda * V_t) * phi(x_n; t, sigma)] / [sum_t exp(lambda * V_t)]
            # as the observation likelihood contribution under this
            # ``(sigma, log lambda)`` grid cell.
            # The denominator is only the softmax normalizer. The Gaussian term
            # appears inside the numerator's sum, so there is no cancellation.
            weighted_pdfs = np.sum(pdf_matrix * exponentiated_values[None, :], axis=1)  # shape: (N,)
            observation_updates = weighted_pdfs / normalization  # shape: (N,)

            if np.any(observation_updates <= 0.0) or np.any(~np.isfinite(observation_updates)):
                continue

            log_likelihood_grid[sigma_index, lambda_index] = float(np.sum(np.log(observation_updates)))  # scalar

    return log_likelihood_grid
