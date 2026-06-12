# This file was written or edited by AI and still requires human review. Delete this comment when done.
"""Baseball log-likelihood grid for HJEEDS (``joint.py`` baseball-multi parity)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from Estimators.utils import computePDF

from .baseball_pitch import PitchObservation, baseball_execution_skill_key


def compute_baseball_log_likelihood_grid(
    pitch_observations: Sequence[PitchObservation],
    possible_targets_feet: np.ndarray,
    all_covs: dict[str, np.ndarray],
    sigma_grid: np.ndarray,
    log_lambda_grid: np.ndarray,
    *,
    delta: float,
) -> np.ndarray:
    """Compute the JEEDS log-likelihood table for one Statcast agent.

    Mirrors ``JointMethodQRE._compute_pdfs_and_evs`` + ``_perform_update`` for
    ``baseball-multi``: per-pitch precomputed EV surfaces, ``computePDF`` at the
    observed action, and ``pdfs * delta**2`` scaling.
    """

    if not pitch_observations:
        raise ValueError("At least one pitch observation is required.")

    executed_actions = np.asarray([obs.executed_action for obs in pitch_observations], dtype=float)
    log_lambda_grid = np.asarray(log_lambda_grid, dtype=float)
    raw_lambda_grid = np.exp(log_lambda_grid)
    log_likelihood_grid = np.full((len(sigma_grid), len(log_lambda_grid)), -np.inf, dtype=float)
    target_means = np.asarray(possible_targets_feet, dtype=float)

    for sigma_index, sigma_hypothesis in enumerate(sigma_grid):
        key = baseball_execution_skill_key(float(sigma_hypothesis))
        if key not in all_covs:
            continue
        cov = all_covs[key]
        cov_stack = np.array([cov] * len(target_means))

        pdfs_per_observation: list[np.ndarray] = []
        evs_per_observation: list[np.ndarray] = []
        for observation_index, observation in enumerate(pitch_observations):
            if key not in observation.evs_per_execution_skill:
                pdfs_per_observation = []
                break
            evs = observation.evs_per_execution_skill[key].reshape(-1)
            action = executed_actions[observation_index]
            pdfs = computePDF(x=action, means=target_means, covs=cov_stack)
            pdfs = np.multiply(pdfs, np.square(delta))
            if np.sum(pdfs) == 0.0 or not np.all(np.isfinite(pdfs)):
                pdfs_per_observation = []
                break
            pdfs_per_observation.append(pdfs)
            evs_per_observation.append(evs)

        if len(pdfs_per_observation) != len(pitch_observations):
            continue

        for lambda_index, lambda_hypothesis in enumerate(raw_lambda_grid):
            log_likelihood = 0.0
            valid = True
            for pdfs, evs in zip(pdfs_per_observation, evs_per_observation):
                shifted = evs * float(lambda_hypothesis)
                max_shifted = float(np.max(shifted))
                exponentiated = np.exp(shifted - max_shifted)
                denominator = float(np.sum(exponentiated))
                if denominator <= 0.0 or not np.isfinite(denominator):
                    valid = False
                    break
                numerator = float(np.sum(exponentiated * pdfs))
                update = numerator / denominator
                if update <= 0.0 or not np.isfinite(update):
                    valid = False
                    break
                log_likelihood += float(np.log(update))
            if valid:
                log_likelihood_grid[sigma_index, lambda_index] = log_likelihood

    return log_likelihood_grid
