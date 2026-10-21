import numpy as np
import pandas as pd
import pytest

from BlackhawksSkillEstimation.analysis.summarize_hjeeds_grid_resolution import comparison_rows
from HJEEDS.hockey_pipeline import make_hockey_log_lambda_grid, make_hockey_sigma_grid
from HJEEDS.run_hockey_subsample import DEFAULT_SEASONS, build_parser, resolve_skill_grids


def test_count_grids_use_historical_endpoints():
    sigma = make_hockey_sigma_grid(250)
    log_lambda = make_hockey_log_lambda_grid(50)
    assert sigma[0] == pytest.approx(0.004)
    assert sigma[-1] == pytest.approx(0.25)
    assert len(sigma) == 250
    assert log_lambda[0] == pytest.approx(np.log(0.1))
    assert log_lambda[-1] == pytest.approx(np.log(1000.0))
    assert len(log_lambda) == 50


def test_subsample_defaults_to_five_seasons_and_count_grids():
    parser = build_parser()
    args = parser.parse_args(["preflight"])
    assert args.seasons == list(DEFAULT_SEASONS)
    sigma, log_lambda = resolve_skill_grids(args)
    assert len(sigma) == 50
    assert len(log_lambda) == 100

    counted = parser.parse_args(["preflight", "--num-sigma-grid", "3", "--num-log-lambda-grid", "4"])
    sigma, log_lambda = resolve_skill_grids(counted)
    assert len(sigma) == 3
    assert sigma[0] == pytest.approx(0.004)
    assert sigma[-1] == pytest.approx(0.25)
    assert len(log_lambda) == 4

    explicit = parser.parse_args(["preflight", "--sigma-grid", "0.01", "0.2"])
    sigma, _ = resolve_skill_grids(explicit)
    np.testing.assert_allclose(sigma, [0.01, 0.2])

    both = parser.parse_args(["preflight", "--sigma-grid", "0.01", "0.2", "--num-sigma-grid", "5"])
    with pytest.raises(ValueError, match="--sigma-grid"):
        resolve_skill_grids(both)


def test_grid_resolution_summary_splits_target_from_roster_median():
    source = pd.DataFrame(
        [
            {"player_id": 950160, "n_shots": 400, "jeeds_status": "ok", "jeeds_mean_sigma": 0.10,
             "jeeds_mean_log_lambda": 1.0, "hierarchical_status": "ok", "hierarchical_mean_sigma": 0.20,
             "hierarchical_mean_log_lambda": 2.0},
            {"player_id": 950160, "n_shots": 400, "jeeds_status": "ok", "jeeds_mean_sigma": 0.30,
             "jeeds_mean_log_lambda": 3.0, "hierarchical_status": "ok", "hierarchical_mean_sigma": 0.20,
             "hierarchical_mean_log_lambda": 2.0},
            {"player_id": 1, "n_shots": 400, "jeeds_status": "ok", "jeeds_mean_sigma": 0.10,
             "jeeds_mean_log_lambda": 2.0, "hierarchical_status": "ok", "hierarchical_mean_sigma": 0.50,
             "hierarchical_mean_log_lambda": 1.0},
            {"player_id": 1, "n_shots": 400, "jeeds_status": "ok", "jeeds_mean_sigma": 0.10,
             "jeeds_mean_log_lambda": 2.0, "hierarchical_status": "ok", "hierarchical_mean_sigma": 0.50,
             "hierarchical_mean_log_lambda": 1.0},
        ]
    )
    rows = comparison_rows(
        source,
        season_pool=5,
        seasons=[20212022, 20252026],
        n_sigma=50,
        n_lambda=100,
        target_player_id=950160,
        n_shots_values=[400],
    )
    target = next(row for row in rows if row["scope"] == "target" and row["metric"] == "jeeds_log_lambda")
    roster = next(row for row in rows if row["scope"] == "roster_median" and row["metric"] == "jeeds_log_lambda")
    assert target["cv"] == pytest.approx(np.sqrt(2.0) / 2.0)
    assert target["stability_score"] == pytest.approx(1.0 / (1.0 + np.sqrt(2.0) / 2.0))
    assert roster["n_players"] == 1
    assert roster["cv"] == pytest.approx(0.0)
    assert roster["stability_score"] == pytest.approx(1.0)
