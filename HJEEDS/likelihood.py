# This file has been fully verified by a human researcher as of 04/23/26 at 1:15 PM MT.

from __future__ import annotations
import math
import numpy as np
from .models import AgentDataset, ExperimentConfig

# This module converts one demonstrator's observed actions into the JEEDS
# log-likelihood grid over all candidate ``(sigma, lambda)`` pairs.  That grid
# is the shared input to both the independent baseline and the hierarchical
# variant.


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
    # Start with ``-inf`` everywhere so impossible parameter pairs naturally
    # remain excluded unless a later calculation supplies a valid likelihood.
    log_likelihood_grid = np.full((len(sigma_grid), len(lambda_grid)), -np.inf, dtype=float)

    reference_targets: np.ndarray | None = None

    for sigma_index, sigma_hypothesis in enumerate(sigma_grid):
        # First hold execution skill fixed.  Under that sigma hypothesis we can
        # compute the target grid, wrapped Gaussian noise model, and then later
        # evaluate all lambda hypotheses against the same precomputed pieces.
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

        # The simulator wraps executed actions back onto the 1D board, so the
        # likelihood must measure the shortest wrapped distance between each
        # executed action and candidate target. For example, on [-10, 10], an
        # executed action near -10 can be close to a target near +10 because the
        # board edge wraps around.
        raw_action_differences = executed_actions[:, None] - target_actions[None, :]  # shape: (N, T)
        board_limit = float(darts.BOARD_LIMIT)
        board_width = 2.0 * board_limit
        action_differences = np.where(
            raw_action_differences > board_limit,
            raw_action_differences - board_width,
            raw_action_differences,
        )  # shape: (N, T)
        action_differences = np.where(
            action_differences < -board_limit,
            action_differences + board_width,
            action_differences,
        )  # shape: (N, T)
        gaussian_scale = float(sigma_hypothesis)
        gaussian_coeff = 1.0 / (math.sqrt(2.0 * math.pi) * gaussian_scale)
        pdf_matrix = gaussian_coeff * np.exp(-0.5 * np.square(action_differences / gaussian_scale))  # shape: (N, T)

        # If any observation receives zero density for every possible target
        # under this sigma hypothesis, then the entire sigma row is impossible.
        if np.any(np.sum(pdf_matrix, axis=1) <= 0.0) or np.any(~np.isfinite(pdf_matrix)):
            continue

        for lambda_index, lambda_hypothesis in enumerate(lambda_grid):
            # Then hold decision-making skill fixed and evaluate the JEEDS
            # marginal likelihood for this full ``(sigma, lambda)`` cell.
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
            weighted_pdfs = np.sum(pdf_matrix * exponentiated_values[None, :], axis=1)  # shape: (N,)
            observation_updates = weighted_pdfs / normalization  # shape: (N,)

            if np.any(observation_updates <= 0.0) or np.any(~np.isfinite(observation_updates)):
                continue

            log_likelihood_grid[sigma_index, lambda_index] = float(np.sum(np.log(observation_updates)))  # scalar

    return log_likelihood_grid
