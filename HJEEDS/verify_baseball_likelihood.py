"""Verify that the HJEEDS baseball pipeline produces the same log-likelihood as the regular baseball environment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Estimators.joint import JointMethodQRE
from Estimators.utils import computePDF
from HJEEDS.baseball_likelihood import compute_baseball_log_likelihood_grid
from HJEEDS.baseball_pitch import (
    baseball_execution_skill_key,
    build_baseball_runtime,
    build_execution_skill_grid,
    build_log_lambda_grid,
    build_pitch_observation,
    load_processed_statcast,
)


def _joint_log_likelihood_for_observation(
    estimator: JointMethodQRE,
    noisy_action,
    possible_targets_feet,
    evs_per_xskill: dict,
    all_covs: dict,
    delta: float,
) -> np.ndarray:
    """Rebuild one observation's contribution across the full grid via joint updates."""

    log_likelihood = np.full(
        (estimator.num_execution_skills, estimator.num_rationality_levels),
        -np.inf,
        dtype=float,
    )
    for exec_index, execution_skill in enumerate(estimator.execution_skills):
        key = baseball_execution_skill_key(float(execution_skill))
        evs = evs_per_xskill[key].reshape(-1)
        cov = all_covs[key]
        pdfs = computePDF(
            x=np.asarray(noisy_action, dtype=float),
            means=possible_targets_feet,
            covs=np.array([cov] * len(possible_targets_feet)),
        )
        pdfs = np.multiply(pdfs, np.square(delta))
        if np.sum(pdfs) == 0.0:
            continue
        for rat_index, rationality in enumerate(estimator.rationality_levels):
            shifted = evs * rationality
            max_shifted = np.max(shifted)
            exponentiated = np.exp(shifted - max_shifted)
            denominator = np.sum(exponentiated)
            if denominator <= 0.0:
                continue
            update = np.sum(exponentiated * pdfs) / denominator
            if update > 0.0:
                log_likelihood[exec_index, rat_index] = float(np.log(update))
    return log_likelihood


def verify_single_pitch_likelihood_matches_joint() -> None:
    rng = np.random.default_rng(0)
    delta = 0.0417
    sigma_grid = build_execution_skill_grid(delta)[:8]
    log_lambda_grid = build_log_lambda_grid(num_lambda_grid=5)
    runtime = build_baseball_runtime(rng, sigma_grid, delta=delta)

    all_data = load_processed_statcast()
    sample_row = all_data.iloc[1000]
    observation = build_pitch_observation(sample_row, runtime, sigma_grid)
    noisy_action = observation.executed_action

    hjeeds_grid = compute_baseball_log_likelihood_grid(
        pitch_observations=[observation],
        possible_targets_feet=runtime.grids.possible_targets_feet,
        all_covs=runtime.all_covs,
        sigma_grid=np.asarray(sigma_grid, dtype=float),
        log_lambda_grid=log_lambda_grid,
        delta=delta,
    )

    estimator = JointMethodQRE(
        list(sigma_grid),
        len(log_lambda_grid),
        "baseball-multi",
    )
    joint_grid = _joint_log_likelihood_for_observation(
        estimator,
        noisy_action,
        runtime.grids.possible_targets_feet,
        observation.evs_per_execution_skill,
        runtime.all_covs,
        delta,
    )

    finite_hjeeds = np.isfinite(hjeeds_grid)
    finite_joint = np.isfinite(joint_grid)
    assert np.array_equal(finite_hjeeds, finite_joint)
    np.testing.assert_allclose(
        hjeeds_grid[finite_hjeeds],
        joint_grid[finite_joint],
        rtol=1e-10,
        atol=1e-10,
    )


def main() -> int:
    verify_single_pitch_likelihood_matches_joint()
    print("HJEEDS baseball likelihood parity check passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
