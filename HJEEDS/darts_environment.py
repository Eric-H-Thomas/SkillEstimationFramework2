
# Paper correspondence: Main `subsec:one_d_darts`; Supplement `app:one_d_darts_environment`.
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


# HJEEDS uses the final-paper 1D darts geometry: actions live on the real line,
# the board occupies [-BOARD_LIMIT, BOARD_LIMIT], in-board rewards alternate
# between LOW_REWARD and HIGH_REWARD, and noisy executions outside the board
# receive OUTSIDE_REWARD rather than wrapping to the other side.
BOARD_LIMIT = 10.0
OUTSIDE_REWARD = 0.0
LOW_REWARD = 1.0
HIGH_REWARD = 2.0


def generate_random_states(
    rng: np.random.Generator,
    min_success_region_count: int,
    max_success_region_count: int,
    num_reward_surfaces: int,
    min_boundary_spacing: float = 0.0,
) -> list[list[float]]:
    """Generate random 1D reward-surface boundaries."""

    if num_reward_surfaces <= 0:
        raise ValueError(f"num_reward_surfaces must be positive. Received {num_reward_surfaces}.")
    if min_success_region_count <= 0 or max_success_region_count <= 0:
        raise ValueError(
            "Success-region count bounds must be positive. "
            f"Received min={min_success_region_count}, max={max_success_region_count}."
        )
    if min_success_region_count > max_success_region_count:
        raise ValueError(
            "min_success_region_count must be less than or equal to "
            "max_success_region_count. "
            f"Received min={min_success_region_count}, max={max_success_region_count}."
        )
    if min_boundary_spacing < 0.0:
        raise ValueError(f"min_boundary_spacing must be non-negative. Received {min_boundary_spacing}.")

    reward_surfaces: list[list[float]] = []
    for _ in range(num_reward_surfaces):
        # Each high-reward region needs a left and right boundary, so a sampled
        # count of K success regions becomes 2K alternating reward boundaries.
        sampled_success_region_count = int(
            rng.integers(min_success_region_count, max_success_region_count + 1)
        )
        target_boundary_count = sampled_success_region_count * 2
        reward_boundaries: list[float] = []

        while len(reward_boundaries) < target_boundary_count:
            candidate_boundary = float(rng.uniform(-BOARD_LIMIT, BOARD_LIMIT))
            has_enough_spacing = all(
                abs(candidate_boundary - existing_boundary) >= min_boundary_spacing
                for existing_boundary in reward_boundaries
            )
            if has_enough_spacing:
                reward_boundaries.append(candidate_boundary)

        reward_surfaces.append(np.sort(reward_boundaries).astype(float).tolist())

    return reward_surfaces


def get_reward_for_action(reward_boundaries: Sequence[float], executed_action: float) -> float:
    """Return the reward for a non-wrapped executed action."""

    if executed_action < -BOARD_LIMIT or executed_action > BOARD_LIMIT:
        return OUTSIDE_REWARD

    is_low_region = True
    for reward_boundary in reward_boundaries:
        if executed_action < float(reward_boundary):
            break
        is_low_region = not is_low_region

    return LOW_REWARD if is_low_region else HIGH_REWARD


def sample_noisy_action(
    rng: np.random.Generator,
    execution_noise_sd: float,
    intended_target: float,
) -> float:
    """Return the non-wrapped executed action after Gaussian execution noise."""

    if execution_noise_sd <= 0.0 or not math.isfinite(execution_noise_sd):
        raise ValueError(f"execution_noise_sd must be positive and finite. Received {execution_noise_sd}.")
    return float(intended_target + rng.normal(0.0, execution_noise_sd))


def compute_expected_value_curve(
    reward_boundaries: Sequence[float],
    execution_noise_sd: float,
    grid_resolution: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact expected rewards on the discretized non-wrapped target grid.

    For target action ``t``, this computes the convolution

        EV(t) = integral R(x) Normal(x; t, execution_noise_sd) dx

    where ``R(x)`` is 1 in low in-board regions, 2 in high in-board regions,
    and 0 outside the physical board. Because the reward is piecewise constant,
    each interval's Gaussian probability mass can be computed exactly from the
    Normal CDF. ``grid_resolution`` therefore controls only the legal target
    grid; it does not introduce numerical integration or endpoint bias.
    """

    if execution_noise_sd <= 0.0 or not math.isfinite(execution_noise_sd):
        raise ValueError(f"execution_noise_sd must be positive and finite. Received {execution_noise_sd}.")
    if grid_resolution <= 0.0 or not math.isfinite(grid_resolution):
        raise ValueError(f"grid_resolution must be positive and finite. Received {grid_resolution}.")

    boundaries = np.asarray(reward_boundaries, dtype=float)
    if boundaries.ndim != 1 or np.any(~np.isfinite(boundaries)):
        raise ValueError("reward_boundaries must be a finite one-dimensional sequence.")
    if np.any(boundaries < -BOARD_LIMIT) or np.any(boundaries > BOARD_LIMIT):
        raise ValueError(
            f"reward_boundaries must lie within [-{BOARD_LIMIT}, {BOARD_LIMIT}]."
        )
    if np.any(np.diff(boundaries) < 0.0):
        raise ValueError("reward_boundaries must be sorted in nondecreasing order.")

    # The target board is [-10, 10], so its width is 2 * BOARD_LIMIT.
    # ``intervals_per_board`` controls the legal intended-target grid only.
    intervals_per_board = int(round(2.0 * BOARD_LIMIT / grid_resolution))
    if intervals_per_board < 1:
        raise ValueError(f"grid_resolution={grid_resolution} is too large to build the darts target grid.")

    # Keep zero as an actual legal target rather than a gap between two target
    # points. That requires an even number of intervals (and therefore an odd
    # number of grid points). If rounding produced an odd count, nudge it up by
    # one; the realized target spacing remains close to grid_resolution.
    if intervals_per_board % 2 != 0:
        intervals_per_board += 1

    target_actions = np.linspace(
        -BOARD_LIMIT,
        BOARD_LIMIT,
        intervals_per_board + 1,
        dtype=float,
    )

    # Split the physical board into its constant-reward intervals. A boundary
    # toggles the reward between LOW_REWARD and HIGH_REWARD; zero-width
    # intervals caused by a boundary at an endpoint or a repeated boundary are
    # harmless and preserve the same toggling semantics as get_reward_for_action.
    interval_edges = np.concatenate(([-BOARD_LIMIT], boundaries, [BOARD_LIMIT]))
    interval_rewards = np.where(
        np.arange(interval_edges.size - 1) % 2 == 0,
        LOW_REWARD,
        HIGH_REWARD,
    ).astype(float)

    # For X ~ Normal(target, sigma), P(a <= X <= b) is exactly
    # Phi((b-target)/sigma) - Phi((a-target)/sigma). Off-board mass contributes
    # OUTSIDE_REWARD=0, so no tail approximation or truncation is needed.
    from scipy.special import ndtr

    standardized_edges = (
        interval_edges[:, None] - target_actions[None, :]
    ) / execution_noise_sd
    interval_probability_mass = np.diff(ndtr(standardized_edges), axis=0)
    expected_values = np.sum(
        interval_rewards[:, None] * interval_probability_mass,
        axis=0,
    )

    return np.asarray(expected_values, dtype=float), target_actions
