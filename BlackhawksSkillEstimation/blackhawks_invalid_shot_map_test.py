"""Guards against all-NaN Blackhawks xG maps poisoning JEEDS/MCSE.

2025-26 for player 950160 has three all-NaN value maps. The first one in the
wristshot/snapshot stream (event 3026775021015) zeroed the joint posterior at
shot 85: MAP pinned to the first grid cell (0.004 rad, lambda=0.1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from BlackhawksSkillEstimation.BlackhawksJEEDS import (
    transform_shots_for_jeeds,
    value_map_is_finite,
)
from BlackhawksSkillEstimation.BlackhawksMCSE import transform_shots_for_mcse
from Estimators.joint import JointMethodQRE


def _one_good_one_bad_shot() -> tuple[pd.DataFrame, dict[int, dict[str, np.ndarray]]]:
    """Two shots at the same location, one with a usable map and one all-NaN."""
    df = pd.DataFrame(
        [
            {
                "event_id": 1,
                "start_x": 70.0,
                "start_y": 0.0,
                "location_y": 0.0,
                "location_z": 2.0,
                "shot_type": "snapshot",
            },
            {
                "event_id": 2,
                "start_x": 70.0,
                "start_y": 0.0,
                "location_y": 0.0,
                "location_z": 2.0,
                "shot_type": "snapshot",
            },
        ]
    )
    shot_maps = {
        1: {"value_map": np.linspace(0.01, 0.2, 31 * 51, dtype=float).reshape(31, 51)},
        2: {"value_map": np.full((31, 51), np.nan)},
    }
    return df, shot_maps


def test_value_map_is_finite_rejects_all_nan_and_empty() -> None:
    assert value_map_is_finite(np.ones((31, 51))) is True
    assert value_map_is_finite(np.full((31, 51), np.nan)) is False
    assert value_map_is_finite(np.array([[1.0, np.nan], [0.2, 0.3]])) is False
    assert value_map_is_finite(np.zeros((0, 51))) is False
    assert value_map_is_finite(None) is False
    assert value_map_is_finite(np.ones(51)) is False


def test_transform_skips_all_nan_map_and_keeps_finite_neighbor() -> None:
    """A finite map is kept; an all-NaN sibling with the same location is dropped."""
    df, shot_maps = _one_good_one_bad_shot()
    inputs = transform_shots_for_jeeds(df, shot_maps=shot_maps, candidate_skills=[0.05])
    assert inputs.skipped_invalid_map == 1
    assert len(inputs.actions) == 1


def test_mcse_transform_skips_all_nan_map_and_keeps_finite_neighbor() -> None:
    """MCSE has its own transform loop, so it needs its own guard coverage."""
    df, shot_maps = _one_good_one_bad_shot()
    inputs = transform_shots_for_mcse(df, shot_maps=shot_maps)
    assert inputs.skipped_invalid_map == 1
    assert len(inputs.actions) == 1


def test_nan_ev_update_does_not_poison_jeeds_posterior(tmp_path, monkeypatch) -> None:
    """If every hypothesis has NaN EVs, restore the previous posterior instead of NaNs."""
    estimator = JointMethodQRE(
        [0.004, 0.25],
        4,
        "hockey-multi",
        times_base_dir=str(tmp_path),
    )
    prior = np.array(estimator.current_probs, copy=True)

    def _nan_pdfs_and_evs(*_args, **_kwargs):
        n_targets = 8
        pdfs = {skill: np.ones(n_targets) for skill in estimator.execution_skills}
        evs = {skill: np.full(n_targets, np.nan) for skill in estimator.execution_skills}
        return pdfs, evs

    monkeypatch.setattr(estimator, "_compute_pdfs_and_evs", _nan_pdfs_and_evs)

    class _Spaces:
        delta = (0.01, 0.01)

    estimator.add_observation(
        np.random.default_rng(0),
        _Spaces(),
        state=None,
        action=[0.1, 0.1],
        resultsFolder=str(tmp_path),
        tag="nan-ev-guard",
        infoPerRow={"evsPerXskill": {}},
    )

    np.testing.assert_allclose(estimator.current_probs, prior)
    assert np.isfinite(estimator.current_probs).all()
    ees = estimator.estimates_execution_skills[estimator.names[1]][-1]
    eps = estimator.estimates_rationality_levels[estimator.names[1]][-1]
    assert np.isfinite(ees) and np.isfinite(eps)
