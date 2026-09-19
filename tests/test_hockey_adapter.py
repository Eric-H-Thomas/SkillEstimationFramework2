import numpy as np
import pandas as pd

from HJEEDS.hockey_adapter import compute_hockey_log_likelihood_grid


def test_compute_hockey_log_likelihood_grid_has_expected_shape():
    rng = np.random.default_rng(7)
    value_map = rng.random((16, 18), dtype=float)
    value_map = np.asarray(value_map, dtype=float)

    shots_df = pd.DataFrame(
        [
            {
                "event_id": 101,
                "start_x": 89.5,
                "start_y": 0.6,
                "location_y": 0.2,
                "location_z": 1.4,
            },
            {
                "event_id": 102,
                "start_x": 88.6,
                "start_y": 1.2,
                "location_y": -0.5,
                "location_z": 1.1,
            },
        ]
    )
    shot_maps = {
        101: {"value_map": value_map},
        102: {"value_map": value_map},
    }

    sigma_grid = np.array([0.02, 0.08], dtype=float)
    log_lambda_grid = np.log(np.array([0.1, 1.0, 10.0], dtype=float))

    grid = compute_hockey_log_likelihood_grid(
        shots_df=shots_df,
        shot_maps=shot_maps,
        sigma_grid=sigma_grid,
        log_lambda_grid=log_lambda_grid,
    )

    assert grid.shape == (len(sigma_grid), len(log_lambda_grid))
    assert np.isfinite(grid).any()


def test_compute_hockey_log_likelihood_grid_matches_c_order_map_coordinates():
    value_map = np.zeros((2, 3), dtype=float)
    value_map[0, 2] = 10.0
    shots_df = pd.DataFrame(
        [{"event_id": 301, "location_y": 5.0, "location_z": 0.0}]
    )

    grid = compute_hockey_log_likelihood_grid(
        shots_df=shots_df,
        shot_maps={301: {"value_map": value_map}},
        sigma_grid=np.array([0.1], dtype=float),
        log_lambda_grid=np.array([np.log(0.1)], dtype=float),
    )

    # The high-value cell is value_map[0, 2], which flattens to the third
    # entry and therefore corresponds to y=5, z=0.
    assert grid[0, 0] > 0.0


def test_run_hockey_jeeds_for_player_uses_blackhawks_contract():
    rng = np.random.default_rng(11)
    value_map = rng.random((12, 14), dtype=float)
    value_map = np.asarray(value_map, dtype=float)

    shots_df = pd.DataFrame(
        [
            {
                "event_id": 201,
                "season": 20232024,
                "game_id": 100,
                "shot_type": "wristshot",
                "location_y": 0.1,
                "location_z": 1.2,
            },
            {
                "event_id": 202,
                "season": 20232024,
                "game_id": 100,
                "shot_type": "snapshot",
                "location_y": -0.4,
                "location_z": 1.0,
            },
            {
                "event_id": 203,
                "season": 20232024,
                "game_id": 101,
                "shot_type": "wristshot",
                "location_y": 0.8,
                "location_z": 1.6,
            },
        ]
    )
    shot_maps = {
        201: {"value_map": value_map},
        202: {"value_map": value_map},
        203: {"value_map": value_map},
    }

    from HJEEDS.hockey_pipeline import run_hockey_jeeds_for_player

    result = run_hockey_jeeds_for_player(
        player_id=950160,
        seasons=[20232024],
        offline_data=(shots_df, shot_maps),
        shot_group="wristshot_snapshot",
        sigma_grid=np.array([0.02, 0.08], dtype=float),
        log_lambda_grid=np.log(np.array([0.1, 1.0, 10.0], dtype=float)),
    )

    assert result["status"] == "ok"
    assert result["player_id"] == 950160
    assert result["seasons"] == [20232024]
    assert result["shot_group"] == "wristshot_snapshot"
    assert result["num_shots"] == 3
    assert np.isfinite(result["posterior_mean_sigma"])
    assert np.isfinite(result["posterior_mean_log_lambda"])
