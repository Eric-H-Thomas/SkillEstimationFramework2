"""Blackhawks-to-HJEEDS runner for real hockey shot data.

This module is the first concrete bridge between the repo's cached Blackhawks
shot data and the generic hierarchical JEEDS likelihood machinery. It is a
lightweight but real-data-aware wrapper around
:func:`HJEEDS.hockey_adapter.compute_hockey_log_likelihood_grid` and the shared
uniform-prior posterior code in :mod:`HJEEDS.estimation`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from BlackhawksSkillEstimation.BlackhawksJEEDS import SHOT_TYPE_GROUPS, load_player_data
from HJEEDS.estimation import (
    build_discrete_hierarchical_prior,
    fit_population_hyperparameters_map,
    run_independent_jeeds_baseline,
)
from HJEEDS.hockey_adapter import compute_hockey_log_likelihood_grid
from HJEEDS.models import ExperimentConfig, HyperpriorConfig, TruePopulationConfig


DEFAULT_HOCKEY_SIGMA_GRID = np.linspace(0.004, 0.25, 50, dtype=float)
# Historical Blackhawks JEEDS used log10(lambda) in [-1, 3], i.e. lambda in
# [0.1, 1000]. HJEEDS stores the equivalent grid in natural-log coordinates.
DEFAULT_HOCKEY_LOG_LAMBDA_GRID = np.linspace(np.log(0.1), np.log(1000.0), 100, dtype=float)

# These are preliminary, manually chosen hockey centers for the first population
# run. They summarize prior Blackhawks JEEDS output but are not fitted from this
# HJEEDS run: sigma=0.075 and lambda=10, represented in natural-log coordinates.
HOCKEY_PRELIMINARY_MEAN_LOG_SIGMA = float(np.log(0.075))
HOCKEY_PRELIMINARY_MEAN_LOG_LAMBDA = float(np.log(10.0))
HOCKEY_PRELIMINARY_HYPERPRIORS = HyperpriorConfig(
    mean_vector=(HOCKEY_PRELIMINARY_MEAN_LOG_SIGMA, HOCKEY_PRELIMINARY_MEAN_LOG_LAMBDA),
    covariance_diagonal=(0.5**2, 1.0**2),
    log_tau_eta_mean=np.log(0.35),
    log_tau_eta_sd=0.5,
    log_tau_rho_mean=np.log(0.8),
    log_tau_rho_sd=0.5,
    m_r=0.0,
    s_r=0.75,
)


def _filter_shot_group(df: pd.DataFrame, shot_group: str | None) -> pd.DataFrame:
    """Apply the same shot-type filter used by the Blackhawks JEEDS runner."""
    if not shot_group:
        return df.copy()

    if shot_group not in SHOT_TYPE_GROUPS:
        valid = ", ".join(sorted(SHOT_TYPE_GROUPS))
        raise ValueError(f"Unknown shot_group '{shot_group}'. Valid groups: {valid}")

    _, allowed_types, include_null = SHOT_TYPE_GROUPS[shot_group]
    df_lc = df.rename(columns=str.lower)
    if "shot_type" not in df_lc.columns:
        return df_lc

    shot_series = df_lc["shot_type"].where(pd.notna(df_lc["shot_type"]))
    shot_lower = shot_series.astype(str).str.lower()
    mask = shot_lower.isin(allowed_types)
    if include_null:
        mask = shot_series.isna() | mask
    return df_lc[mask].reset_index(drop=True)


def _resolve_sigma_and_lambda_grids(
    sigma_grid: Sequence[float] | np.ndarray | None,
    log_lambda_grid: Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    sigma = np.asarray(sigma_grid if sigma_grid is not None else DEFAULT_HOCKEY_SIGMA_GRID, dtype=float)
    log_lambda = np.asarray(
        log_lambda_grid if log_lambda_grid is not None else DEFAULT_HOCKEY_LOG_LAMBDA_GRID,
        dtype=float,
    )
    if sigma.ndim != 1 or log_lambda.ndim != 1:
        raise ValueError("sigma_grid and log_lambda_grid must both be 1D arrays.")
    if sigma.size == 0 or log_lambda.size == 0:
        raise ValueError("sigma_grid and log_lambda_grid must both be non-empty.")
    return sigma, log_lambda


def run_hockey_jeeds_for_shots(
    shots_df: pd.DataFrame,
    shot_maps: Mapping[int, Mapping[str, object]],
    sigma_grid: Sequence[float] | np.ndarray | None = None,
    log_lambda_grid: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Run a uniform-prior HJEEDS pass on a Blackhawks shot pool.

    Returns a dictionary with both the generic posterior summary and the full
    grid/metadata needed by later hierarchical calibration code.
    """
    sigma_grid_arr, log_lambda_grid_arr = _resolve_sigma_and_lambda_grids(sigma_grid, log_lambda_grid)
    shots_df = shots_df.copy()

    if shots_df.empty:
        return {
            "status": "no_data",
            "num_shots": 0,
            "execution_skill": None,
            "rationality": None,
            "posterior_mean_sigma": None,
            "posterior_mean_log_lambda": None,
            "map_sigma": None,
            "map_log_lambda": None,
            "sigma_grid": sigma_grid_arr,
            "log_lambda_grid": log_lambda_grid_arr,
            "notes": "No shots were available for HJEEDS inference.",
        }

    required_columns = {"event_id", "location_y", "location_z"}
    missing = required_columns.difference(shots_df.columns)
    if missing:
        raise ValueError(f"shots_df is missing required columns: {sorted(missing)}")

    log_likelihood_grid = compute_hockey_log_likelihood_grid(
        shots_df=shots_df,
        shot_maps=shot_maps,
        sigma_grid=sigma_grid_arr,
        log_lambda_grid=log_lambda_grid_arr,
    )
    estimate = run_independent_jeeds_baseline(
        log_likelihood_grid=log_likelihood_grid,
        sigma_grid=sigma_grid_arr,
        log_lambda_grid=log_lambda_grid_arr,
    )

    result: dict[str, Any] = {
        "status": estimate.status,
        "num_shots": int(len(shots_df)),
        "execution_skill": estimate.posterior_mean_sigma,
        "rationality": estimate.posterior_mean_log_lambda,
        "posterior_mean_sigma": estimate.posterior_mean_sigma,
        "posterior_mean_log_lambda": estimate.posterior_mean_log_lambda,
        "map_sigma": estimate.map_sigma,
        "map_log_lambda": estimate.map_log_lambda,
        "sigma_grid": sigma_grid_arr,
        "log_lambda_grid": log_lambda_grid_arr,
        "log_likelihood_grid": log_likelihood_grid,
        "notes": estimate.notes,
    }
    return result


def build_hockey_experiment_config(
    *,
    sigma_grid: Sequence[float] | np.ndarray,
    log_lambda_grid: Sequence[float] | np.ndarray,
    num_agents: int,
    hyperpriors: HyperpriorConfig | None = None,
) -> ExperimentConfig:
    """Build the minimal config needed to fit a shared Blackhawks population prior."""
    sigma_arr = np.asarray(sigma_grid, dtype=float)
    log_lambda_arr = np.asarray(log_lambda_grid, dtype=float)
    if sigma_arr.ndim != 1 or log_lambda_arr.ndim != 1:
        raise ValueError("sigma_grid and log_lambda_grid must both be one-dimensional.")

    chosen_hyperpriors = hyperpriors or HOCKEY_PRELIMINARY_HYPERPRIORS
    return ExperimentConfig(
        environment="hockey",
        seed=0,
        num_seeds=1,
        num_agents=int(num_agents),
        count_buckets=(1,),
        agents_per_bucket=max(1, int(num_agents)),
        delta=0.1,
        num_sigma_grid=int(sigma_arr.size),
        num_lambda_grid=int(log_lambda_arr.size),
        sigma_min=float(np.min(sigma_arr)),
        sigma_max=float(np.max(sigma_arr)),
        lambda_min=float(np.min(np.exp(log_lambda_arr))),
        lambda_max=float(np.max(np.exp(log_lambda_arr))),
        output_dir=Path("HJEEDS/results/hockey"),
        dry_run=False,
        min_success_regions=1,
        max_success_regions=1,
        min_region_width=0.0,
        hyperpriors=chosen_hyperpriors,
        true_population=TruePopulationConfig(
            mean_log_sigma=float(np.mean(np.log(sigma_arr))),
            mean_log_lambda=float(np.mean(log_lambda_arr)),
            tau_eta=0.35,
            tau_rho=0.8,
            correlation=0.0,
            population_shape_slug="hockey",
        ),
    )


def compute_hockey_player_log_likelihood_grid(
    player_id: int,
    seasons: Sequence[int],
    *,
    data_dir: str | Path = Path("Data/Hockey"),
    shot_group: str | None = "wristshot_snapshot",
    sigma_grid: Sequence[float] | np.ndarray | None = None,
    log_lambda_grid: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Compute the raw per-player HJEEDS log-likelihood grid from cached Blackhawks data."""
    sigma_arr, log_lambda_arr = _resolve_sigma_and_lambda_grids(sigma_grid, log_lambda_grid)
    shots_df, shot_maps = load_player_data(player_id=player_id, seasons=list(seasons), data_dir=data_dir)
    shots_df = _filter_shot_group(shots_df, shot_group).copy()
    if shots_df.empty:
        raise ValueError(f"No shots available for player {player_id} after filtering '{shot_group}'.")
    return compute_hockey_log_likelihood_grid(
        shots_df=shots_df,
        shot_maps=shot_maps,
        sigma_grid=sigma_arr,
        log_lambda_grid=log_lambda_arr,
    )


def fit_hockey_population_hyperparameters(
    player_log_likelihoods: Sequence[np.ndarray],
    sigma_grid: Sequence[float] | np.ndarray | None = None,
    log_lambda_grid: Sequence[float] | np.ndarray | None = None,
    *,
    hyperpriors: HyperpriorConfig | None = None,
) -> dict[str, Any]:
    """Fit a shared population prior across a cohort of Blackhawks players."""
    sigma_arr, log_lambda_arr = _resolve_sigma_and_lambda_grids(sigma_grid, log_lambda_grid)
    config = build_hockey_experiment_config(
        sigma_grid=sigma_arr,
        log_lambda_grid=log_lambda_arr,
        num_agents=len(player_log_likelihoods),
        hyperpriors=hyperpriors,
    )
    return fit_population_hyperparameters_map(
        config=config,
        agent_log_likelihoods=[np.asarray(grid, dtype=float) for grid in player_log_likelihoods],
        sigma_grid=sigma_arr,
        log_lambda_grid=log_lambda_arr,
    )


def fit_hockey_hyperprior_for_players(
    player_ids: Sequence[int],
    seasons: Sequence[int],
    *,
    data_dir: str | Path = Path("Data/Hockey"),
    shot_group: str | None = "wristshot_snapshot",
    sigma_grid: Sequence[float] | np.ndarray | None = None,
    log_lambda_grid: Sequence[float] | np.ndarray | None = None,
    hyperpriors: HyperpriorConfig | None = None,
) -> dict[str, Any]:
    """Load a cohort, compute each player's log-likelihood grid, and fit a shared prior."""
    sigma_arr, log_lambda_arr = _resolve_sigma_and_lambda_grids(sigma_grid, log_lambda_grid)
    player_grids = []
    for player_id in player_ids:
        player_grids.append(
            compute_hockey_player_log_likelihood_grid(
                player_id=player_id,
                seasons=seasons,
                data_dir=data_dir,
                shot_group=shot_group,
                sigma_grid=sigma_arr,
                log_lambda_grid=log_lambda_arr,
            )
        )

    fitted = fit_hockey_population_hyperparameters(
        player_log_likelihoods=player_grids,
        sigma_grid=sigma_arr,
        log_lambda_grid=log_lambda_arr,
        hyperpriors=hyperpriors,
    )
    fitted["player_ids"] = [int(pid) for pid in player_ids]
    fitted["shot_group"] = shot_group
    fitted["seasons"] = [int(s) for s in seasons]
    fitted["discrete_prior"] = build_discrete_hierarchical_prior(
        fitted_hyperparameters=fitted,
        sigma_grid=sigma_arr,
        log_lambda_grid=log_lambda_arr,
    )
    return fitted


def run_hockey_jeeds_for_player(
    player_id: int,
    seasons: Sequence[int] | None = None,
    *,
    data_dir: str | Path = Path("Data/Hockey"),
    shot_group: str | None = "wristshot_snapshot",
    sigma_grid: Sequence[float] | np.ndarray | None = None,
    log_lambda_grid: Sequence[float] | np.ndarray | None = None,
    offline_data: tuple[pd.DataFrame, dict[int, dict[str, object]]] | None = None,
) -> dict[str, Any]:
    """Run a single-player Blackhawks HJEEDS estimate from cached parquet/npz data."""
    if offline_data is not None:
        shots_df, shot_maps = offline_data
    else:
        if seasons is None:
            raise ValueError("Either seasons or offline_data must be provided.")
        shots_df, shot_maps = load_player_data(player_id=player_id, seasons=list(seasons), data_dir=data_dir)

    shots_df = shots_df.copy()
    shots_df = _filter_shot_group(shots_df, shot_group)

    if shots_df.empty:
        return {
            "player_id": player_id,
            "status": "no_data",
            "num_shots": 0,
            "execution_skill": None,
            "rationality": None,
            "posterior_mean_sigma": None,
            "posterior_mean_log_lambda": None,
            "notes": "No shots remained after applying the requested shot-type filter.",
        }

    result = run_hockey_jeeds_for_shots(
        shots_df=shots_df,
        shot_maps=shot_maps,
        sigma_grid=sigma_grid,
        log_lambda_grid=log_lambda_grid,
    )
    result["player_id"] = player_id
    result["seasons"] = list(seasons) if seasons is not None else None
    result["shot_group"] = shot_group
    return result


__all__ = [
    "DEFAULT_HOCKEY_SIGMA_GRID",
    "DEFAULT_HOCKEY_LOG_LAMBDA_GRID",
    "HOCKEY_PRELIMINARY_HYPERPRIORS",
    "HOCKEY_PRELIMINARY_MEAN_LOG_LAMBDA",
    "HOCKEY_PRELIMINARY_MEAN_LOG_SIGMA",
    "build_hockey_experiment_config",
    "compute_hockey_player_log_likelihood_grid",
    "fit_hockey_population_hyperparameters",
    "fit_hockey_hyperprior_for_players",
    "run_hockey_jeeds_for_shots",
    "run_hockey_jeeds_for_player",
]
