"""Synthetic hockey shooters on a real player's shot situations.

The donor contributes shooter locations and per-event xG maps. A known
``(sigma, lambda)`` pair contributes the aim and the miss. Situations are
drawn independently with replacement from the donor pool: repeating an xG map
is another shot from that situation, not a copy of a previous observation.

Within one seed the situation sequence is shared across truth pairs, so
differences in the log-lambda posterior are not an artifact of which maps were
drawn. Executed angles stay in the angular coordinates the hockey likelihood
uses. They are not written back to goal-line ``(y, z)``, because that
conversion clips elevation and can drop shots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, NamedTuple, Sequence

import numpy as np

from BlackhawksSkillEstimation.BlackhawksJEEDS import load_player_data
from BlackhawksSkillEstimation.player_subsample_stability import (
    discover_cached_seasons,
    filter_shot_group,
    sort_chronologically,
)
from HJEEDS.estimation import build_discrete_hierarchical_prior
from HJEEDS.hockey_adapter import (
    HockeySituation,
    _blurred_utility,
    _execution_kernel,
    lambda_log_contributions_batch,
    lambda_log_contributions_many_surfaces,
    prepare_hockey_situations,
)


def draw_situation_indices(
    num_situations: int,
    num_shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw i.i.d. situation indexes with replacement.

    ``num_shots`` may exceed ``num_situations``. Each index is one independent
    draw from the empirical distribution of the donor pool.
    """

    pool_size = int(num_situations)
    shots = int(num_shots)
    if pool_size < 1:
        raise ValueError("Need at least one donor situation.")
    if shots < 1:
        raise ValueError(f"num_shots must be positive. Received {num_shots}.")
    return rng.integers(0, pool_size, size=shots, endpoint=False)


def sample_target_indices(
    values: np.ndarray,
    lambda_true: float,
    rng: np.random.Generator,
    size: int,
) -> np.ndarray:
    """Draw intended-target indexes from ``softmax(lambda * values)``."""

    if int(size) < 1:
        raise ValueError(f"size must be positive. Received {size}.")
    if not np.isfinite(lambda_true) or lambda_true < 0.0:
        raise ValueError(f"lambda_true must be finite and nonnegative. Received {lambda_true}.")

    flattened = np.asarray(values, dtype=float).reshape(-1)
    if flattened.size == 0:
        raise ValueError("values must be nonempty.")
    scaled = float(lambda_true) * flattened
    finite = np.isfinite(scaled)
    if not np.any(finite):
        raise ValueError("values must contain at least one finite entry.")
    scaled = np.where(finite, scaled, -np.inf)
    scaled = scaled - np.max(scaled[finite])
    probabilities = np.exp(scaled)
    probabilities = np.where(np.isfinite(probabilities), probabilities, 0.0)
    total = float(np.sum(probabilities))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("Softmax probabilities could not be normalized.")
    probabilities /= total
    return rng.choice(probabilities.size, size=int(size), p=probabilities)


def simulate_executed_actions(
    situations: Sequence[HockeySituation],
    situation_indices: np.ndarray,
    sigma_true: float,
    lambda_true: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample an angular execution for each draw in ``situation_indices``.

    The intended target is a cell of the donor's angular grid, chosen with the
    softmax of the value surface blurred at ``sigma_true``. The executed angle
    is that cell plus isotropic Gaussian noise of sd ``sigma_true``. Draws that
    share a situation reuse one blurred surface and consume ``rng`` in shot order.
    """

    if not np.isfinite(sigma_true) or sigma_true <= 0.0:
        raise ValueError(f"sigma_true must be finite and positive. Received {sigma_true}.")

    indices = np.asarray(situation_indices, dtype=int)
    if indices.ndim != 1:
        raise ValueError("situation_indices must be a one-dimensional array.")
    executed = np.empty((indices.size, 2), dtype=float)
    blurred_values: dict[int, np.ndarray] = {}
    for shot_number, situation_index in enumerate(indices.tolist()):
        if situation_index < 0 or situation_index >= len(situations):
            raise IndexError(f"Situation index {situation_index} is outside a pool of {len(situations)}.")
        values = blurred_values.get(situation_index)
        situation = situations[situation_index]
        if values is None:
            values = _blurred_utility(situation.utility_grid, float(sigma_true), situation.bin_size)
            if values.size != situation.target_angles.shape[0]:
                raise ValueError(
                    "Blurred utility length does not match the angular target list: "
                    f"{values.size} vs {situation.target_angles.shape[0]}."
                )
            blurred_values[situation_index] = values
        chosen = sample_target_indices(values, lambda_true, rng, size=1)[0]
        target = situation.target_angles[int(chosen)]
        executed[shot_number] = target + rng.normal(0.0, float(sigma_true), size=2)
    return executed


def shot_log_likelihoods(
    target_angles: np.ndarray,
    utility_grid: np.ndarray,
    bin_size: float,
    executed_actions: np.ndarray,
    sigma_grid: np.ndarray,
    log_lambda_grid: np.ndarray,
) -> np.ndarray:
    """Per-shot log likelihoods on the hockey ``(sigma, log lambda)`` grid.

    Returns shape ``(n_shots, n_sigma, n_log_lambda)``. Invalid cells are NaN,
    matching the shots ``_accumulate_vectorized`` skips rather than a ``-inf``
    that would poison a later sum.
    """

    executed = np.asarray(executed_actions, dtype=float)
    if executed.ndim == 1:
        executed = executed.reshape(1, 2)
    if executed.ndim != 2 or executed.shape[1] != 2:
        raise ValueError(f"executed_actions must have shape (n_shots, 2); got {executed.shape}.")

    sigma_values = np.asarray(sigma_grid, dtype=float)
    log_lambda_values = np.asarray(log_lambda_grid, dtype=float)
    lambdas = np.exp(log_lambda_values)
    targets = np.asarray(target_angles, dtype=float)
    rows = np.full((executed.shape[0], sigma_values.size, log_lambda_values.size), np.nan, dtype=float)
    for sigma_index, sigma in enumerate(sigma_values):
        sigma = float(sigma)
        if not np.isfinite(sigma) or sigma <= 0.0:
            continue
        values = _blurred_utility(utility_grid, sigma, bin_size)
        distances = np.hypot(
            targets[None, :, 0] - executed[:, None, 0],
            targets[None, :, 1] - executed[:, None, 1],
        )
        gaussian_terms = _execution_kernel(distances, sigma)
        if gaussian_terms is None:
            continue
        rows[:, sigma_index, :] = lambda_log_contributions_batch(values, gaussian_terms, lambdas)
    return rows


def sequence_log_likelihoods(
    situations: Sequence[HockeySituation],
    situation_indices: np.ndarray,
    executed_actions: np.ndarray,
    sigma_grid: np.ndarray,
    log_lambda_grid: np.ndarray,
) -> np.ndarray:
    """Per-shot log likelihoods, blurring each situation once per sigma.

    Shots from different xG maps are scored together. The blurred surface
    depends only on the situation and the sigma hypothesis, so repeated draws
    of one map share that blur and only the execution kernel changes.
    """

    indices = np.asarray(situation_indices, dtype=int)
    executed = np.asarray(executed_actions, dtype=float)
    if indices.ndim != 1 or executed.ndim != 2 or executed.shape != (indices.size, 2):
        raise ValueError(
            "situation_indices must be (n_shots,) and executed_actions (n_shots, 2). "
            f"Received {indices.shape} and {executed.shape}."
        )
    if indices.size == 0:
        raise ValueError("situation_indices must be nonempty.")

    sigma_values = np.asarray(sigma_grid, dtype=float)
    log_lambda_values = np.asarray(log_lambda_grid, dtype=float)
    lambdas = np.exp(log_lambda_values)
    rows = np.full((indices.size, sigma_values.size, log_lambda_values.size), np.nan, dtype=float)
    groups = {
        int(situation_index): np.flatnonzero(indices == situation_index)
        for situation_index in np.unique(indices)
    }
    target_count = situations[int(indices[0])].target_angles.shape[0]
    blurred_by_situation = np.empty((indices.size, target_count), dtype=float)
    kernels = np.empty((indices.size, target_count), dtype=float)

    for sigma_index, sigma in enumerate(sigma_values):
        sigma = float(sigma)
        if not np.isfinite(sigma) or sigma <= 0.0:
            continue
        usable = np.ones(indices.size, dtype=bool)
        for situation_index, shot_numbers in groups.items():
            situation = situations[situation_index]
            if situation.target_angles.shape[0] != target_count:
                raise ValueError("Every situation in a sequence must share the angular target count.")
            blurred = _blurred_utility(situation.utility_grid, sigma, situation.bin_size)
            distances = np.hypot(
                situation.target_angles[None, :, 0] - executed[shot_numbers, None, 0],
                situation.target_angles[None, :, 1] - executed[shot_numbers, None, 1],
            )
            gaussian_terms = _execution_kernel(distances, sigma)
            if gaussian_terms is None or blurred.size != target_count or not np.isfinite(blurred).all():
                usable[shot_numbers] = False
                continue
            blurred_by_situation[shot_numbers] = blurred
            kernels[shot_numbers] = gaussian_terms
        if not np.any(usable):
            continue
        contributions = lambda_log_contributions_many_surfaces(
            blurred_by_situation[usable],
            kernels[usable],
            lambdas,
        )
        rows[np.flatnonzero(usable), sigma_index, :] = contributions
    return rows


def prefix_log_likelihoods(
    per_shot_log_likelihoods: np.ndarray,
    prefix_lengths: Sequence[int],
) -> dict[int, np.ndarray]:
    """Cumulative log-likelihood grids at the requested shot counts.

    NaN cells are skipped, as in the hockey adapter. A cell that no shot in the
    prefix informed stays ``-inf``.
    """

    per_shot = np.asarray(per_shot_log_likelihoods, dtype=float)
    if per_shot.ndim != 3:
        raise ValueError(f"per-shot log likelihoods must have shape (n, n_sigma, n_lambda); got {per_shot.shape}.")
    num_shots = per_shot.shape[0]
    lengths = [int(length) for length in prefix_lengths]
    if any(length < 1 or length > num_shots for length in lengths):
        raise ValueError(f"Prefix lengths must lie in 1..{num_shots}. Received {lengths}.")

    safe = np.where(np.isfinite(per_shot), per_shot, 0.0)
    valid = np.isfinite(per_shot)
    cumulative_total = np.cumsum(safe, axis=0)
    cumulative_count = np.cumsum(valid, axis=0)
    grids: dict[int, np.ndarray] = {}
    for length in lengths:
        total = cumulative_total[length - 1]
        count = cumulative_count[length - 1]
        grid = np.full(total.shape, -np.inf, dtype=float)
        informed = count > 0
        grid[informed] = total[informed]
        grids[length] = grid
    return grids


def uniform_skill_prior(sigma_grid: np.ndarray, log_lambda_grid: np.ndarray) -> np.ndarray:
    """Uniform mass on the ``(sigma, log lambda)`` cells."""

    prior = np.ones((len(sigma_grid), len(log_lambda_grid)), dtype=float)
    prior /= np.sum(prior)
    return prior


def normal_skill_prior(
    sigma_grid: np.ndarray,
    log_lambda_grid: np.ndarray,
    *,
    mean_log_sigma: float,
    mean_log_lambda: float,
    sd_log_sigma: float,
    sd_log_lambda: float,
) -> np.ndarray:
    """Independent Normal prior on ``(log sigma, log lambda)``, discretized onto the grid."""

    return build_discrete_hierarchical_prior(
        {
            "mu": np.array([mean_log_sigma, mean_log_lambda], dtype=float),
            "tau_eta": float(sd_log_sigma),
            "tau_rho": float(sd_log_lambda),
            "correlation": 0.0,
        },
        np.asarray(sigma_grid, dtype=float),
        np.asarray(log_lambda_grid, dtype=float),
    )


def _marginal_mean_sd(grid: np.ndarray, marginal: np.ndarray) -> tuple[float, float]:
    weights = np.asarray(marginal, dtype=float)
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return float("nan"), float("nan")
    weights = weights / total
    mean = float(np.dot(weights, grid))
    second = float(np.dot(weights, np.square(grid)))
    variance = max(second - mean * mean, 0.0)
    return mean, float(np.sqrt(variance))


def equal_tailed_interval(
    grid: np.ndarray,
    marginal: np.ndarray,
    *,
    lower: float = 0.025,
    upper: float = 0.975,
) -> tuple[float, float]:
    """Equal-tailed central interval on a discrete one-dimensional grid."""

    weights = np.asarray(marginal, dtype=float)
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return float("nan"), float("nan")
    cdf = np.cumsum(weights / total)
    low_index = int(np.searchsorted(cdf, lower, side="left"))
    high_index = int(np.searchsorted(cdf, upper, side="left"))
    low_index = min(low_index, grid.size - 1)
    high_index = min(max(high_index, low_index), grid.size - 1)
    return float(grid[low_index]), float(grid[high_index])


def summarize_posterior(
    log_likelihood_grid: np.ndarray,
    discrete_prior: np.ndarray,
    sigma_grid: np.ndarray,
    log_lambda_grid: np.ndarray,
) -> dict[str, float | str]:
    """Posterior means and the central 95% interval of ``log lambda``.

    The normalization matches ``run_hierarchical_estimator``: add ``log prior``
    to the log likelihood on cells where both are usable, then renormalize.
    """

    sigma_values = np.asarray(sigma_grid, dtype=float)
    log_lambda_values = np.asarray(log_lambda_grid, dtype=float)
    empty = {
        "status": "no_finite_posterior",
        "posterior_mean_sigma": float("nan"),
        "posterior_mean_log_lambda": float("nan"),
        "posterior_sd_log_lambda": float("nan"),
        "ci_low_log_lambda": float("nan"),
        "ci_high_log_lambda": float("nan"),
    }
    log_likelihood = np.asarray(log_likelihood_grid, dtype=float)
    prior = np.asarray(discrete_prior, dtype=float)
    expected = (sigma_values.size, log_lambda_values.size)
    if log_likelihood.shape != expected or prior.shape != expected:
        raise ValueError(
            "log_likelihood_grid and discrete_prior must both have shape "
            f"{expected}. Received {log_likelihood.shape} and {prior.shape}."
        )
    if np.any(~np.isfinite(prior)) or np.any(prior < 0.0):
        empty["status"] = "invalid_prior"
        return empty
    prior_total = float(np.sum(prior))
    if prior_total <= 0.0 or not np.isfinite(prior_total):
        empty["status"] = "invalid_prior"
        return empty
    prior = prior / prior_total
    support = np.isfinite(log_likelihood) & (prior > 0.0)
    if not np.any(support):
        return empty

    log_posterior = np.full(expected, -np.inf, dtype=float)
    log_posterior[support] = log_likelihood[support] + np.log(prior[support])
    stabilized = log_posterior - np.max(log_posterior[support])
    unnormalized = np.zeros(expected, dtype=float)
    unnormalized[support] = np.exp(stabilized[support])
    normalizer = float(np.sum(unnormalized))
    if normalizer <= 0.0 or not np.isfinite(normalizer):
        return empty
    posterior = unnormalized / normalizer
    marginal_sigma = np.sum(posterior, axis=1)
    marginal_log_lambda = np.sum(posterior, axis=0)
    mean_sigma, _ = _marginal_mean_sd(sigma_values, marginal_sigma)
    mean_log_lambda, sd_log_lambda = _marginal_mean_sd(log_lambda_values, marginal_log_lambda)
    ci_low, ci_high = equal_tailed_interval(log_lambda_values, marginal_log_lambda)
    return {
        "status": "ok",
        "posterior_mean_sigma": mean_sigma,
        "posterior_mean_log_lambda": mean_log_lambda,
        "posterior_sd_log_lambda": sd_log_lambda,
        "ci_low_log_lambda": ci_low,
        "ci_high_log_lambda": ci_high,
    }


def departure_flags(
    *,
    posterior_mean_log_lambda: float,
    posterior_sd_log_lambda: float,
    ci_low_log_lambda: float,
    ci_high_log_lambda: float,
    prior_mean_log_lambda: float,
    prior_sd_log_lambda: float,
    true_log_lambda: float,
) -> dict[str, bool]:
    """Whether this prefix has left the prior, and whether it has recovered the truth.

    ``shifted`` requires the posterior mean to have moved toward the truth by
    more than half a prior standard deviation. ``excluded_prior_mean`` is the
    sample-size marker for leaving the prior: the central 95% interval no
    longer contains the prior mean. ``recovered`` requires that interval to
    cover the truth and the posterior sd to be under half the prior sd.
    """

    values = (
        posterior_mean_log_lambda,
        posterior_sd_log_lambda,
        ci_low_log_lambda,
        ci_high_log_lambda,
        prior_mean_log_lambda,
        prior_sd_log_lambda,
        true_log_lambda,
    )
    if not all(np.isfinite(value) for value in values):
        return {"shifted": False, "excluded_prior_mean": False, "recovered": False}

    delta = float(posterior_mean_log_lambda) - float(prior_mean_log_lambda)
    toward_truth = float(true_log_lambda) - float(prior_mean_log_lambda)
    shifted = abs(delta) > 0.5 * float(prior_sd_log_lambda) and delta * toward_truth > 0.0
    excluded = float(ci_high_log_lambda) < float(prior_mean_log_lambda) or float(ci_low_log_lambda) > float(
        prior_mean_log_lambda
    )
    recovered = (
        float(ci_low_log_lambda) <= float(true_log_lambda) <= float(ci_high_log_lambda)
        and float(posterior_sd_log_lambda) < 0.5 * float(prior_sd_log_lambda)
    )
    return {
        "shifted": bool(shifted),
        "excluded_prior_mean": bool(excluded),
        "recovered": bool(recovered),
    }


class DonorPool(NamedTuple):
    """Usable situations from one player's cached shot pool."""

    situations: list[HockeySituation]
    rows_in: int
    seasons: tuple[int, ...]


def load_donor_situations(
    player_id: int,
    *,
    seasons: Sequence[int] | None = None,
    data_dir: Path | str = Path("Data/Hockey"),
    shot_group: str = "wristshot_snapshot",
    max_situations: int | None = None,
) -> DonorPool:
    """Load one player's usable shot situations, in chronological order.

    Duplicate event ids keep the earliest row. ``max_situations`` truncates
    that ordered pool for a smoke test; the full pool is the experiment.
    ``rows_in`` counts rows after the shot-group filter and before the angular
    transform drops any that it cannot use.
    """

    data_path = Path(data_dir)
    chosen_seasons = list(seasons) if seasons is not None else discover_cached_seasons(player_id, data_path)
    if not chosen_seasons:
        raise FileNotFoundError(f"No cached seasons for player {player_id} under {data_path}.")

    shots_df, shot_maps = load_player_data(player_id, chosen_seasons, data_path)
    filtered = sort_chronologically(filter_shot_group(shots_df, shot_group))
    filtered = filtered.drop_duplicates(subset=["event_id"], keep="first").reset_index(drop=True)
    if max_situations is not None:
        if int(max_situations) < 1:
            raise ValueError(f"max_situations must be positive. Received {max_situations}.")
        filtered = filtered.iloc[: int(max_situations)].reset_index(drop=True)
    situations = prepare_hockey_situations(filtered, shot_maps)
    if not situations:
        raise ValueError(
            f"Player {player_id} has no usable {shot_group} situations in seasons {chosen_seasons}."
        )
    return DonorPool(
        situations=situations,
        rows_in=int(len(filtered)),
        seasons=tuple(int(season) for season in chosen_seasons),
    )


def prior_log_lambda_marginal(discrete_prior: np.ndarray, log_lambda_grid: np.ndarray) -> tuple[float, float]:
    """Mean and sd of ``log lambda`` under a discrete joint prior."""

    prior = np.asarray(discrete_prior, dtype=float)
    marginal = np.sum(prior, axis=0)
    return _marginal_mean_sd(np.asarray(log_lambda_grid, dtype=float), marginal)


def evaluate_prefixes(
    per_shot_log_likelihoods: np.ndarray,
    situation_indices: np.ndarray,
    prefix_lengths: Sequence[int],
    priors: Mapping[str, np.ndarray],
    sigma_grid: np.ndarray,
    log_lambda_grid: np.ndarray,
    *,
    true_sigma: float,
    true_lambda: float,
) -> list[dict[str, object]]:
    """Posterior summaries for every prior at every prefix of one simulated sequence."""

    true_log_lambda = float(np.log(true_lambda))
    indices = np.asarray(situation_indices, dtype=int)
    prefix_grids = prefix_log_likelihoods(per_shot_log_likelihoods, prefix_lengths)
    prior_stats = {
        name: prior_log_lambda_marginal(prior, log_lambda_grid) for name, prior in priors.items()
    }
    rows: list[dict[str, object]] = []
    for length, log_likelihood in prefix_grids.items():
        unique_situations = int(np.unique(indices[:length]).size)
        for prior_name, prior in priors.items():
            summary = summarize_posterior(log_likelihood, prior, sigma_grid, log_lambda_grid)
            prior_mean, prior_sd = prior_stats[prior_name]
            flags = departure_flags(
                posterior_mean_log_lambda=float(summary["posterior_mean_log_lambda"]),
                posterior_sd_log_lambda=float(summary["posterior_sd_log_lambda"]),
                ci_low_log_lambda=float(summary["ci_low_log_lambda"]),
                ci_high_log_lambda=float(summary["ci_high_log_lambda"]),
                prior_mean_log_lambda=prior_mean,
                prior_sd_log_lambda=prior_sd,
                true_log_lambda=true_log_lambda,
            )
            rows.append(
                {
                    "true_sigma": float(true_sigma),
                    "true_lambda": float(true_lambda),
                    "true_log_lambda": true_log_lambda,
                    "prior_name": prior_name,
                    "prior_mean_log_lambda": prior_mean,
                    "prior_sd_log_lambda": prior_sd,
                    "n_shots": int(length),
                    "num_unique_situations": unique_situations,
                    **summary,
                    **flags,
                }
            )
    return rows


def simulate_truth_sequence(
    situations: Sequence[HockeySituation],
    situation_indices: np.ndarray,
    *,
    sigma_true: float,
    lambda_true: float,
    sigma_grid: np.ndarray,
    log_lambda_grid: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate one truth on a fixed situation sequence and return per-shot log likelihoods."""

    executed = simulate_executed_actions(situations, situation_indices, sigma_true, lambda_true, rng)
    return sequence_log_likelihoods(
        situations,
        situation_indices,
        executed,
        sigma_grid,
        log_lambda_grid,
    )


def situation_rng(base_seed: int, seed: int) -> np.random.Generator:
    """RNG for the shared situation sequence of one seed."""

    return np.random.default_rng([int(base_seed), int(seed)])


def execution_rng(base_seed: int, seed: int, sigma_index: int, lambda_index: int) -> np.random.Generator:
    """RNG for executions of one truth. Independent of which other truths are run."""

    return np.random.default_rng([int(base_seed), int(seed), int(sigma_index) + 1, int(lambda_index) + 1])
