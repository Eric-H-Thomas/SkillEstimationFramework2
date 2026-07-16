# This file has been fully reviewed by a human researcher as of 07/16/26 at 1:45 PM MDT.
"""Parity check: HJEEDS baseball likelihood vs ``JointMethodQRE`` baseball-multi.

Exercises production ``_compute_pdfs_and_evs`` and ``_perform_update`` (not a
reimplementation of the update). Log-likelihood is recovered as
``log(unnormalized_posterior / prior)`` so normalization in ``add_observation``
does not enter the comparison — matching the HJEEDS likelihood grid, which
stores summed ``log(update)`` terms before any prior.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Estimators.joint import JointMethodQRE
from HJEEDS.baseball_likelihood import compute_baseball_log_likelihood_grid
from HJEEDS.baseball_pitch import (
    DEFAULT_DELTA,
    BaseballRuntime,
    PitchObservation,
    build_baseball_runtime,
    build_execution_skill_grid,
    build_log_lambda_grid,
    build_pitch_observation,
    load_processed_statcast,
)


def _spaces_stub(runtime: BaseballRuntime) -> SimpleNamespace:
    """Minimal spaces object for ``JointMethodQRE._compute_pdfs_and_evs``.

    ``get_key`` matches ``SpacesBaseball.get_key`` / ``baseball_execution_skill_key``.
    """

    return SimpleNamespace(
        delta=runtime.grids.delta,
        all_covs=runtime.all_covs,
        possibleTargetsFeet=runtime.grids.possible_targets_feet,
        get_key=lambda info, r=0.0: "|".join(map(str, info)) + f"|{r}",
    )


def _log_update_from_posterior(prior: np.ndarray, posterior: np.ndarray) -> np.ndarray:
    """Map unnormalized posterior mass to per-cell ``log(update)`` vs ``prior``."""

    with np.errstate(divide="ignore", invalid="ignore"):
        log_update = np.log(posterior) - np.log(prior)
    return np.where(
        np.isfinite(log_update) & (posterior > 0.0) & (prior > 0.0),
        log_update,
        -np.inf,
    )


def _apply_qre_updates(
    estimator: JointMethodQRE,
    observations: list[PitchObservation],
    runtime: BaseballRuntime,
    delta: float,
) -> np.ndarray:
    """Run production PDF/EV + ``_perform_update``; return ``log(post/prior)``.

    Does **not** normalize between pitches (unlike ``add_observation``), so the
    result equals the sum of per-pitch ``log(update)`` terms.
    """

    estimator.reset()
    prior = np.asarray(estimator.current_probs, dtype=float).copy()
    spaces = _spaces_stub(runtime)

    for observation in observations:
        other_args = {"infoPerRow": {"evsPerXskill": observation.evs_per_execution_skill}}
        pdfs_per_execution_skill, evs_per_execution_skill = estimator._compute_pdfs_and_evs(
            spaces,
            np.asarray(observation.executed_action, dtype=float),
            None,
            delta,
            other_args,
        )
        estimator._perform_update(pdfs_per_execution_skill, evs_per_execution_skill)

    posterior = np.asarray(estimator.current_probs, dtype=float)
    return _log_update_from_posterior(prior, posterior)


def _assert_grids_match(hjeeds_grid: np.ndarray, joint_grid: np.ndarray) -> None:
    finite_hjeeds = np.isfinite(hjeeds_grid)
    finite_joint = np.isfinite(joint_grid)
    assert np.array_equal(finite_hjeeds, finite_joint), (
        f"Finite masks differ: hjeeds={finite_hjeeds.sum()} joint={finite_joint.sum()}"
    )
    np.testing.assert_allclose(
        hjeeds_grid[finite_hjeeds],
        joint_grid[finite_joint],
        rtol=1e-10,
        atol=1e-10,
    )


def _build_estimator(
    sigma_grid: np.ndarray,
    log_lambda_grid: np.ndarray,
) -> JointMethodQRE:
    estimator = JointMethodQRE(
        list(sigma_grid),
        len(log_lambda_grid),
        "baseball-multi",
    )
    # HJEEDS stores log-lambda; joint.py stores raw lambda via logspace(-3, 3.6, n).
    np.testing.assert_allclose(
        np.asarray(estimator.execution_skills, dtype=float),
        np.asarray(sigma_grid, dtype=float),
    )
    np.testing.assert_allclose(
        np.asarray(estimator.rationality_levels, dtype=float),
        np.exp(np.asarray(log_lambda_grid, dtype=float)),
        rtol=1e-12,
        atol=1e-12,
    )
    return estimator


def verify_single_pitch_likelihood_matches_joint(
    runtime: BaseballRuntime,
    observation: PitchObservation,
    sigma_grid: np.ndarray,
    log_lambda_grid: np.ndarray,
    delta: float,
) -> None:
    hjeeds_grid = compute_baseball_log_likelihood_grid(
        pitch_observations=[observation],
        possible_targets_feet=runtime.grids.possible_targets_feet,
        all_covs=runtime.all_covs,
        sigma_grid=np.asarray(sigma_grid, dtype=float),
        log_lambda_grid=log_lambda_grid,
        delta=delta,
    )

    estimator = _build_estimator(sigma_grid, log_lambda_grid)
    joint_grid = _apply_qre_updates(estimator, [observation], runtime, delta)
    _assert_grids_match(hjeeds_grid, joint_grid)


def verify_multi_pitch_likelihood_matches_joint(
    runtime: BaseballRuntime,
    observations: list[PitchObservation],
    sigma_grid: np.ndarray,
    log_lambda_grid: np.ndarray,
    delta: float,
) -> None:
    """Multi-pitch HJEEDS grid must match sequential unnormalized QRE updates."""

    hjeeds_grid = compute_baseball_log_likelihood_grid(
        pitch_observations=observations,
        possible_targets_feet=runtime.grids.possible_targets_feet,
        all_covs=runtime.all_covs,
        sigma_grid=np.asarray(sigma_grid, dtype=float),
        log_lambda_grid=log_lambda_grid,
        delta=delta,
    )

    estimator = _build_estimator(sigma_grid, log_lambda_grid)
    joint_grid = _apply_qre_updates(estimator, observations, runtime, delta)
    _assert_grids_match(hjeeds_grid, joint_grid)


def main() -> int:
    rng = np.random.default_rng(0)
    delta = DEFAULT_DELTA
    # Small grids keep the check fast; endpoints still match JointMethodQRE baseball-multi.
    sigma_grid = build_execution_skill_grid(delta)[:8]
    log_lambda_grid = build_log_lambda_grid(num_lambda_grid=5)
    runtime = build_baseball_runtime(rng, sigma_grid, delta=delta)

    all_data = load_processed_statcast()
    observations = [
        build_pitch_observation(all_data.iloc[1000], runtime, sigma_grid),
        build_pitch_observation(all_data.iloc[2500], runtime, sigma_grid),
    ]

    verify_single_pitch_likelihood_matches_joint(
        runtime, observations[0], sigma_grid, log_lambda_grid, delta,
    )
    verify_multi_pitch_likelihood_matches_joint(
        runtime, observations, sigma_grid, log_lambda_grid, delta,
    )
    print("HJEEDS baseball likelihood parity check passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
