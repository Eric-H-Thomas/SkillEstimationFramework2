import numpy as np
import pandas as pd
import pytest

from BlackhawksSkillEstimation.analysis.plot_hjeeds_grid_resolution import (
    _snap_fraction,
    resolution_table,
)


def _cell_frame(n_sigma, n_lambda, offset):
    rows = []
    for player_id, base in ((1, 0.04), (2, 0.06)):
        for seed in range(12):
            value = base + 0.001 * (seed % 4)
            rows.append({
                "player_id": player_id,
                "n_shots": 100,
                "seed": seed,
                "jeeds_status": "ok",
                "hierarchical_status": "ok",
                "jeeds_mean_sigma": value + offset,
                "jeeds_mean_log_lambda": 1.0 + value,
                "hierarchical_mean_sigma": value,
                "hierarchical_mean_log_lambda": 1.2 + value,
                "n_sigma": n_sigma,
                "n_lambda": n_lambda,
            })
    return pd.DataFrame(rows)


def _manifest(n_sigma, n_lambda):
    return {
        "grid": {
            "sigma_grid": np.linspace(0.004, 0.25, n_sigma).tolist(),
            "log_lambda_grid": np.linspace(np.log(0.1), np.log(1000.0), n_lambda).tolist(),
        }
    }


def test_shift_is_paired_and_scaled_by_reference_seed_sd():
    coarse = _cell_frame(2, 2, offset=0.0005)
    fine = _cell_frame(3, 3, offset=0.0)
    data = pd.concat([coarse, fine], ignore_index=True)
    manifests = {(2, 2): _manifest(2, 2), (3, 3): _manifest(3, 3)}

    table = resolution_table(data, manifests)

    reference_sd = fine[fine.player_id == 1]["jeeds_mean_sigma"].std(ddof=1)
    coarse_row = table[(table.grid == "2x2") & (table.quantity == "JEEDS execution sigma")].iloc[0]
    fine_row = table[(table.grid == "3x3") & (table.quantity == "JEEDS execution sigma")].iloc[0]
    assert coarse_row["median_shift_seed_sd"] == pytest.approx(0.0005 / reference_sd)
    assert coarse_row["paired_rows"] == 24
    assert fine_row["is_reference"]
    assert fine_row["median_shift_seed_sd"] == pytest.approx(0.0)
    other = table[(table.grid == "2x2") & (table.quantity == "JEEDS log lambda")].iloc[0]
    assert other["median_shift_seed_sd"] == pytest.approx(0.0)


def test_snap_fraction_detects_estimates_on_grid_points():
    grid = np.linspace(0.0, 1.0, 11)
    on_grid = np.repeat(grid[3:5], 10)
    between = grid[3:5] + 0.05
    assert _snap_fraction(on_grid, grid) == pytest.approx(1.0)
    assert _snap_fraction(between, grid) == pytest.approx(0.0)
