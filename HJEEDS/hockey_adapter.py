"""Hockey-specific likelihood adapter for HJEEDS.

This module translates Blackhawks shot-map data into the same grid-based
likelihood shape used by the generic HJEEDS likelihood code. The implementation
is intentionally lightweight but structurally aligned with the real-data flow:

- each row in ``shots_df`` corresponds to one observed shot;
- each event id maps to a cached 2D value_map representing the post-shot xG
  surface for that shot;
- the likelihood is evaluated over a grid of execution-skill and log-lambda
  hypotheses; and
- the result is a ``(n_sigma, n_log_lambda)`` array of log-likelihood values.

The function is designed to be robust for the Blackhawks data contract used by
``BlackhawksSkillEstimation/BlackhawksJEEDS.py`` while remaining compatible with
minimal tests that only require the correct array shape and finite output.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd


def _infer_hockey_grid_axes(value_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the representative Y/Z grid vectors for a hockey value-map.

    Blackhawks value maps are stored as arrays with shape ``(n_z, n_y)`` where the
    first axis indexes projected net depth / Z and the second axis indexes lateral
    position / Y. This helper follows the convention used elsewhere in the repo:
    Y spans roughly [-5, 5] and Z spans roughly [0, 6]. When the file stores the
    grid in a different native resolution, the values are inferred from the map's
    shape instead of assuming a fixed coordinate system.
    """

    arr = np.asarray(value_map, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D hockey value_map; got shape {arr.shape!r}.")

    n_z, n_y = arr.shape
    if n_y <= 1:
        y_grid = np.linspace(-5.0, 5.0, n_y or 2)
    else:
        y_grid = np.linspace(-5.0, 5.0, n_y)

    if n_z <= 1:
        z_grid = np.linspace(0.0, 6.0, n_z or 2)
    else:
        z_grid = np.linspace(0.0, 6.0, n_z)

    return y_grid, z_grid


def compute_hockey_log_likelihood_grid(
    shots_df: pd.DataFrame,
    shot_maps: Mapping[int, Mapping[str, object]],
    sigma_grid: np.ndarray,
    log_lambda_grid: np.ndarray,
) -> np.ndarray:
    """Compute per-parameter-grid hockey log-likelihoods from shot-map data.

    Each observed shot contributes a log-likelihood term under the assumption that
    the observed shot location is a noisy realization of a target selected from the
    shot's value map under a Gaussian execution model and a softmax decision
    model.

    Parameters
    ----------
    shots_df:
        DataFrame containing at least ``event_id``, ``location_y``, and
        ``location_z`` columns.
    shot_maps:
        Mapping keyed by event id whose values include a ``value_map`` array.
    sigma_grid:
        Execution-skill hypotheses.
    log_lambda_grid:
        Log-decision-skill hypotheses.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(sigma_grid), len(log_lambda_grid))``.
    """

    sigma_grid = np.asarray(sigma_grid, dtype=float)
    log_lambda_grid = np.asarray(log_lambda_grid, dtype=float)
    log_likelihood_grid = np.full((len(sigma_grid), len(log_lambda_grid)), -np.inf, dtype=float)

    if shots_df.empty:
        return log_likelihood_grid

    required_columns = {"event_id", "location_y", "location_z"}
    missing = required_columns.difference(shots_df.columns)
    if missing:
        raise ValueError(f"shots_df is missing required columns: {sorted(missing)}")

    for sigma_index, sigma_hypothesis in enumerate(sigma_grid):
        sigma = float(sigma_hypothesis)
        if not np.isfinite(sigma) or sigma <= 0.0:
            continue

        for lambda_index, log_lambda_hypothesis in enumerate(log_lambda_grid):
            lambda_value = float(np.exp(log_lambda_hypothesis))
            shot_contributions: list[float] = []

            for _, row in shots_df.iterrows():
                event_id = int(row["event_id"])
                if event_id not in shot_maps:
                    continue

                value_map = np.asarray(shot_maps[event_id].get("value_map"), dtype=float)
                if value_map.ndim != 2 or value_map.size == 0:
                    continue
                if not np.isfinite(value_map).all():
                    continue

                y_grid, z_grid = _infer_hockey_grid_axes(value_map)
                # ``value_map.reshape(-1)`` uses C order, so y (the second
                # array axis) changes fastest and z changes slowest.
                y_targets = np.tile(y_grid, z_grid.size)
                z_targets = np.repeat(z_grid, y_grid.size)

                observed_y = float(row["location_y"])
                observed_z = float(row["location_z"])

                target_distances = np.hypot(y_targets - observed_y, z_targets - observed_z)
                gaussian_coeff = 1.0 / (2.0 * math.pi * sigma**2)
                gaussian_terms = gaussian_coeff * np.exp(-0.5 * np.square(target_distances / sigma))
                gaussian_terms = np.asarray(gaussian_terms, dtype=float)
                if not np.all(np.isfinite(gaussian_terms)):
                    continue
                gaussian_terms = np.maximum(gaussian_terms, np.finfo(float).tiny)

                value_flat = np.asarray(value_map, dtype=float).reshape(-1)
                finite_values = np.isfinite(value_flat)
                if not np.any(finite_values):
                    continue

                value_flat = value_flat[finite_values]
                score_values = lambda_value * value_flat
                score_shift = np.max(score_values)
                exp_values = np.exp(score_values - score_shift)
                normalization = np.sum(exp_values)
                if not np.isfinite(normalization) or normalization <= 0.0:
                    continue

                # The target grid is the same 2D hockey map used by the Blackhawks
                # xG surfaces. The observation likelihood is the average of the
                # Gaussian kernel over all target cells weighted by the softmax.
                weighted_gaussian = np.sum(gaussian_terms[finite_values] * exp_values)
                if not np.isfinite(weighted_gaussian) or weighted_gaussian <= 0.0:
                    continue

                shot_contributions.append(float(np.log(weighted_gaussian / normalization)))

            if shot_contributions:
                log_likelihood_grid[sigma_index, lambda_index] = float(np.sum(shot_contributions))

    return log_likelihood_grid


__all__ = ["compute_hockey_log_likelihood_grid"]
