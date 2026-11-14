import numpy as np
import pandas as pd
import pytest

from BlackhawksSkillEstimation.personal_maxg import (
    empirical_sigma_grid,
    maxg_value,
    player_jeeds_profile,
    policy_expectation,
    realized_shot_xg,
    results_shot_group,
    score_surfaces,
    wide_sigma_grid,
)
from HJEEDS.hockey_adapter import blur_utility


def test_policy_expectation_is_at_most_maxg_and_approaches_it():
    raw = np.array([[0.05, 0.20], [0.90, 0.15]], dtype=float)
    peak = maxg_value(raw)
    uniform = policy_expectation(raw, lambda_value=0.0)
    peaked = policy_expectation(raw, lambda_value=1e8)

    assert uniform <= peak + 1e-9
    assert peaked <= peak + 1e-9
    assert uniform < peaked
    assert peaked == pytest.approx(peak, abs=1e-6)

    max_low, policy_low = score_surfaces([raw, raw], lambda_value=0.0)
    max_high, policy_high = score_surfaces([raw, raw], lambda_value=1e8)
    assert max_low == pytest.approx(2.0 * peak)
    assert max_high == pytest.approx(max_low)
    assert policy_low < policy_high
    assert policy_high == pytest.approx(max_high, abs=1e-6)
    assert policy_low <= max_low + 1e-9


def test_blurred_policy_expectation_approaches_maxg():
    utility = np.zeros((9, 9), dtype=float)
    utility[4, 4] = 1.0
    raw = blur_utility(utility, sigma=0.05, bin_size=0.02)
    peak = maxg_value(raw)
    soft = policy_expectation(raw, lambda_value=1.0)
    sharp = policy_expectation(raw, lambda_value=1e6)

    assert soft <= peak + 1e-9
    assert sharp <= peak + 1e-9
    assert soft < sharp
    assert sharp == pytest.approx(peak, abs=1e-5)


def test_empirical_sigma_grid_spans_usable_jeeds_means():
    frame = pd.DataFrame(
        {
            "jeeds_status": ["ok", "ok", "failed"],
            "jeeds_mean_sigma": [0.04, 0.07, 0.90],
        }
    )
    grid = empirical_sigma_grid(frame, 10)
    assert grid.size == 10
    assert grid[0] == pytest.approx(0.04)
    assert grid[-1] == pytest.approx(0.07)
    assert np.all(np.diff(grid) > 0)


def test_wide_sigma_grid_endpoints():
    grid = wide_sigma_grid(10)
    assert grid.size == 10
    assert grid[0] == pytest.approx(0.01)
    assert grid[-1] == pytest.approx(0.20)
    assert np.all(np.diff(grid) > 0)


def test_player_jeeds_profile_reads_the_pooled_hjeeds_table():
    frame = pd.DataFrame(
        [
            {
                "player_id": 7,
                "jeeds_status": "ok",
                "jeeds_mean_sigma": 0.06,
                "jeeds_mean_log_lambda": 1.0,
                "hierarchical_mean_sigma": 0.04,
                "hierarchical_mean_log_lambda": 2.0,
            },
            {
                "player_id": 8,
                "jeeds_status": "no_data",
                "jeeds_mean_sigma": 0.2,
                "jeeds_mean_log_lambda": 0.5,
            },
        ]
    )

    sigma, lambda_value = player_jeeds_profile(frame, 7)
    assert sigma == pytest.approx(0.06)
    assert lambda_value == pytest.approx(np.exp(1.0))
    with pytest.raises(RuntimeError, match="no pooled JEEDS estimate"):
        player_jeeds_profile(frame, 8)


def test_results_shot_group_reads_the_run_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text('{"shot_group": "wristshot_snapshot"}\n', encoding="utf-8")
    assert results_shot_group(tmp_path / "agent_level_results.csv") == "wristshot_snapshot"
    assert results_shot_group(tmp_path / "missing" / "agent_level_results.csv") is None


def test_realized_shot_xg_reads_the_nearest_cell():
    utility = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    targets = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    assert realized_shot_xg(utility, targets, np.array([0.9, 0.1])) == pytest.approx(0.2)
