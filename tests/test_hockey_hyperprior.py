import numpy as np

from HJEEDS.hockey_pipeline import fit_hockey_population_hyperparameters


def test_fit_hockey_population_hyperparameters_runs_on_two_players():
    sigma_grid = np.array([0.02, 0.08, 0.16], dtype=float)
    log_lambda_grid = np.log(np.array([0.2, 1.0, 5.0], dtype=float))

    grid_a = np.full((len(sigma_grid), len(log_lambda_grid)), -np.inf, dtype=float)
    grid_b = np.full((len(sigma_grid), len(log_lambda_grid)), -np.inf, dtype=float)
    grid_a[0, 0] = -2.0
    grid_a[1, 1] = -1.1
    grid_b[1, 1] = -1.2
    grid_b[2, 2] = -0.9

    fitted = fit_hockey_population_hyperparameters(
        [grid_a, grid_b],
        sigma_grid=sigma_grid,
        log_lambda_grid=log_lambda_grid,
    )

    assert fitted["num_agents"] == 2
    assert fitted["mu"].shape == (2,)
    assert fitted["covariance_matrix"].shape == (2, 2)
    assert np.isfinite(fitted["objective_value"])
