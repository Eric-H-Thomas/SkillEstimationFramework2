# This file has been fully verified by a human researcher as of 05/14/26 at 5:00 PM MT.

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
    """Return expected rewards for all target actions on the non-wrapped board.

    For target action ``t``, this computes the convolution

        EV(t) = integral R(x) Normal(x; t, execution_noise_sd) dx

    where ``R(x)`` is 1 in low in-board regions, 2 in high in-board regions,
    and 0 outside the physical board.  The extended ``[-3m, 3m]`` grid leaves
    room for Gaussian spillover at both board edges while the returned target
    grid is cropped back to ``[-m, m]``.
    """

    if execution_noise_sd <= 0.0 or not math.isfinite(execution_noise_sd):
        raise ValueError(f"execution_noise_sd must be positive and finite. Received {execution_noise_sd}.")
    if grid_resolution <= 0.0 or not math.isfinite(grid_resolution):
        raise ValueError(f"grid_resolution must be positive and finite. Received {grid_resolution}.")

    # The target board is [-10, 10], so its width is 2 * BOARD_LIMIT.
    # ``intervals_per_board`` is the number of small integration intervals
    # across that physical board.  For example, if grid_resolution = 0.1, then
    # the board has width 20 and we use about 200 intervals.
    #
    # Important mental-model note: np.linspace with N intervals creates N + 1
    # grid points.  In the later Riemann-style approximation, each grid point
    # acts like the center of its own rectangle/bin with width ``grid_step``.
    # So if you are picturing rectangles, picture one rectangle per grid point,
    # not one rectangle per interval.
    intervals_per_board = int(round(2.0 * BOARD_LIMIT / grid_resolution))
    if intervals_per_board < 1:
        raise ValueError(f"grid_resolution={grid_resolution} is too large to build the extended darts grid.")

    # A symmetric grid needs zero to be one of the actual grid points, not a
    # gap between two points.  That is what keeps the Gaussian kernel centered
    # exactly at zero when we later convolve.  To get zero as a point on the
    # full [-3m, 3m] grid, we want an odd number of grid points, which means an
    # even number of intervals.  If the rounded interval count is odd, nudge it
    # up by one.  The realized grid spacing will still be very close to
    # grid_resolution.
    if intervals_per_board % 2 != 0:
        intervals_per_board += 1

    # We do the convolution on a larger grid than the actual board:
    #
    #   actual target board:      [-m,  m]
    #   convolution work grid:  [-3m, 3m]
    #
    # The extra space matters because Gaussian execution noise can spill past
    # the board edges.  Non-wrapped darts does not fold that mass back onto the
    # board.  Instead, off-board executed actions get reward 0.  By explicitly
    # representing a wide off-board region with reward 0, the convolution can
    # correctly lower expected values near the edges.
    num_intervals = 3 * intervals_per_board
    extended_grid = np.linspace(-3.0 * BOARD_LIMIT, 3.0 * BOARD_LIMIT, num_intervals + 1, dtype=float)

    # This is the actual spacing between adjacent grid points.  It will usually
    # equal grid_resolution, but using the realized spacing is safer because
    # np.linspace defines the endpoints exactly and the interval count may have
    # been nudged to keep the grid centered.
    grid_step = float(extended_grid[1] - extended_grid[0])

    # Discretize the reward function R(x) on the extended grid.  Inside the
    # board, R(x) alternates between LOW_REWARD and HIGH_REWARD as the action
    # crosses each boundary.  Outside [-BOARD_LIMIT, BOARD_LIMIT], R(x) is 0.
    #
    # This vector is the left-hand object in the convolution: the environment's
    # deterministic reward landscape before execution noise is applied.
    reward_values = np.array(
        [get_reward_for_action(reward_boundaries, float(executed_action)) for executed_action in extended_grid],
        dtype=float,
    )

    # Build a zero-centered Gaussian density on the same grid.  Conceptually,
    # this is the distribution of execution error:
    #
    #   epsilon ~ Normal(0, execution_noise_sd)
    #   executed_action = intended_target + epsilon
    #
    # We evaluate the density at each possible error size in ``extended_grid``.
    # The kernel is centered at 0, so high mass near 0 means "most throws land
    # close to the intended target."  Larger ``execution_noise_sd`` spreads this
    # kernel out and makes the agent's execution less precise.
    gaussian_kernel = (
        np.exp(-0.5 * np.square(extended_grid / execution_noise_sd))
        / (math.sqrt(2.0 * math.pi) * execution_noise_sd)
    )

    # The continuous expected value at target t is
    #
    #   EV(t) = integral R(x) * Normal(x; mean=t, sd=execution_noise_sd) dx.
    #
    # Equivalently, this is the reward landscape R convolved with the
    # zero-centered execution-error density.  Numerically, the integral becomes
    # a sum over grid cells:
    #
    #   sum_x R(x) * gaussian_density(x - t) * grid_step
    #
    # Multiplying the density by ``grid_step`` converts the Gaussian density
    # values into approximate probability mass per grid cell.  Without that
    # factor, the scale of the convolution would depend on the arbitrary grid
    # resolution.
    convolved_values = np.convolve(reward_values, gaussian_kernel * grid_step, "same")

    # ``convolved_values`` is defined over the whole work grid [-3m, 3m], but
    # legal intended targets are only on the physical board [-m, m].  The middle
    # third of the work grid corresponds exactly to that board.  We crop back to
    # the target board after doing the convolution so callers get one EV value
    # for each possible intended target.
    left = intervals_per_board
    right = (2 * intervals_per_board) + 1

    return convolved_values[left:right], extended_grid[left:right]
