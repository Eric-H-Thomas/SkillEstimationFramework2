import numpy as np
import pandas as pd

from HJEEDS.estimation import run_hierarchical_estimator, run_independent_jeeds_baseline
from HJEEDS.hockey_adapter import (
    HockeySituation,
    _accumulate_vectorized,
    _lambda_log_contributions,
    lambda_log_contributions_batch,
    lambda_log_contributions_many_surfaces,
    prepare_hockey_situations,
)
from HJEEDS.hockey_synthetic_player import (
    departure_flags,
    draw_situation_indices,
    evaluate_prefixes,
    execution_rng,
    normal_skill_prior,
    prefix_log_likelihoods,
    sample_target_indices,
    sequence_log_likelihoods,
    shot_log_likelihoods,
    simulate_executed_actions,
    situation_rng,
    summarize_posterior,
    uniform_skill_prior,
)
from HJEEDS.run_hockey_rationality_sample_size import majority_crossings, resolve_prefixes


def _toy_situation() -> HockeySituation:
    return HockeySituation(
        event_id=1,
        target_angles=np.array(
            [[0.0, 0.0], [0.0, 0.3], [0.3, 0.0], [0.3, 0.3]],
            dtype=float,
        ),
        executed_action=np.array([0.3, 0.3], dtype=float),
        utility_grid=np.array([[0.0, 0.0], [0.0, 6.0]], dtype=float),
        bin_size=0.15,
    )


def test_target_distribution_follows_lambda():
    values = np.array([0.0, 0.0, 1.0, 0.2])
    high = sample_target_indices(values, lambda_true=40.0, rng=np.random.default_rng(0), size=300)
    low = sample_target_indices(values, lambda_true=0.0, rng=np.random.default_rng(1), size=800)

    assert np.mean(high == 2) > 0.95
    counts = np.bincount(low, minlength=4) / low.size
    assert np.all(np.abs(counts - 0.25) < 0.08)


def test_situation_draws_allow_replacement_past_the_pool_size():
    indices = draw_situation_indices(2, 40, np.random.default_rng(0))

    assert indices.shape == (40,)
    assert indices.min() >= 0
    assert indices.max() <= 1
    assert len(set(indices.tolist())) < len(indices)


def test_one_seed_shares_situations_and_not_executions():
    first = draw_situation_indices(5, 20, situation_rng(7, 3))
    second = draw_situation_indices(5, 20, situation_rng(7, 3))
    assert np.array_equal(first, second)

    lambda_zero = execution_rng(7, 3, 0, 0).random(6)
    lambda_one = execution_rng(7, 3, 0, 1).random(6)
    assert not np.array_equal(lambda_zero, lambda_one)


def test_repeated_situation_is_not_a_copied_likelihood_row():
    situation = _toy_situation()
    indices = np.array([0, 0])
    executed = np.array([[0.0, 0.0], [0.3, 0.3]], dtype=float)
    sigma_grid = np.array([0.05, 0.1], dtype=float)
    log_lambda_grid = np.log(np.array([0.5, 5.0, 20.0]))
    rows = sequence_log_likelihoods([situation], indices, executed, sigma_grid, log_lambda_grid)

    assert rows.shape == (2, 2, 3)
    assert np.isfinite(rows).all()
    assert not np.allclose(rows[0], rows[1])


def test_per_shot_rows_match_the_hockey_accumulator():
    situation = _toy_situation()
    sigma_grid = np.array([0.05, 0.1], dtype=float)
    log_lambda_grid = np.log(np.array([0.5, 5.0, 20.0]))
    executed = simulate_executed_actions(
        [situation],
        np.array([0, 0]),
        sigma_true=0.05,
        lambda_true=8.0,
        rng=np.random.default_rng(4),
    )
    rows = sequence_log_likelihoods([situation], np.array([0, 0]), executed, sigma_grid, log_lambda_grid)
    combined = prefix_log_likelihoods(rows, [1, 2])
    for length in (1, 2):
        reference = _accumulate_vectorized(
            [
                (situation.target_angles, action, situation.utility_grid, situation.bin_size)
                for action in executed[:length]
            ],
            sigma_grid,
            log_lambda_grid,
        )
        informed = np.isfinite(rows[:length]).all(axis=0)
        assert np.array_equal(informed, np.isfinite(reference))
        np.testing.assert_allclose(combined[length][informed], reference[informed])


def test_many_surface_contributions_match_the_single_surface_batch():
    rng = np.random.default_rng(2)
    values = rng.random((5, 6))
    weights = rng.random((5, 6))
    values[1, 2] = np.nan
    lambdas = np.array([0.2, 1.0, np.nan, 8.0])
    many = lambda_log_contributions_many_surfaces(values, weights, lambdas, chunk_size=2)

    for shot in range(values.shape[0]):
        single = lambda_log_contributions_batch(values[shot], weights[shot][None, :], lambdas)[0]
        np.testing.assert_allclose(many[shot], single, equal_nan=True)


def test_batch_lambda_contributions_match_the_scalar_loop():
    values = np.array([0.1, 0.4, 0.2, np.nan])
    weights = np.array(
        [
            [0.2, 0.3, 0.4, 0.1],
            [0.5, 0.1, 0.2, 0.2],
        ]
    )
    lambdas = np.array([0.1, 1.0, 10.0])
    batch = lambda_log_contributions_batch(values, weights, lambdas)

    for shot in range(weights.shape[0]):
        scalar = _lambda_log_contributions(values, weights[shot], lambdas)
        np.testing.assert_allclose(batch[shot], scalar, equal_nan=True)


def test_posterior_means_match_the_existing_estimators():
    rng = np.random.default_rng(0)
    sigma_grid = np.linspace(0.02, 0.12, 4)
    log_lambda_grid = np.linspace(np.log(0.5), np.log(20.0), 5)
    log_likelihood = rng.normal(size=(4, 5))
    log_likelihood[0, 1] = -np.inf

    uniform = uniform_skill_prior(sigma_grid, log_lambda_grid)
    uniform_summary = summarize_posterior(log_likelihood, uniform, sigma_grid, log_lambda_grid)
    uniform_estimate = run_independent_jeeds_baseline(log_likelihood, sigma_grid, log_lambda_grid)
    np.testing.assert_allclose(
        uniform_summary["posterior_mean_log_lambda"],
        uniform_estimate.posterior_mean_log_lambda,
    )
    np.testing.assert_allclose(
        uniform_summary["posterior_mean_sigma"],
        uniform_estimate.posterior_mean_sigma,
    )

    normal = normal_skill_prior(
        sigma_grid,
        log_lambda_grid,
        mean_log_sigma=np.log(0.075),
        mean_log_lambda=np.log(10.0),
        sd_log_sigma=0.5,
        sd_log_lambda=1.0,
    )
    normal_summary = summarize_posterior(log_likelihood, normal, sigma_grid, log_lambda_grid)
    normal_estimate = run_hierarchical_estimator(log_likelihood, normal, sigma_grid, log_lambda_grid)
    np.testing.assert_allclose(
        normal_summary["posterior_mean_log_lambda"],
        normal_estimate.posterior_mean_log_lambda,
    )
    np.testing.assert_allclose(
        normal_summary["posterior_mean_sigma"],
        normal_estimate.posterior_mean_sigma,
    )


def test_high_lambda_posterior_sits_above_low_lambda():
    situation = _toy_situation()
    sigma_grid = np.array([0.03, 0.05, 0.1], dtype=float)
    log_lambda_grid = np.linspace(np.log(0.2), np.log(80.0), 17)
    prior = {"uniform": uniform_skill_prior(sigma_grid, log_lambda_grid)}
    indices = np.zeros(50, dtype=int)
    means = {}
    for lambda_true, seed in ((0.4, 4), (40.0, 40)):
        executed = simulate_executed_actions(
            [situation],
            indices,
            sigma_true=0.05,
            lambda_true=lambda_true,
            rng=np.random.default_rng(seed),
        )
        rows = shot_log_likelihoods(
            situation.target_angles,
            situation.utility_grid,
            situation.bin_size,
            executed,
            sigma_grid,
            log_lambda_grid,
        )
        evaluated = evaluate_prefixes(
            rows,
            indices,
            [50],
            prior,
            sigma_grid,
            log_lambda_grid,
            true_sigma=0.05,
            true_lambda=lambda_true,
        )
        assert evaluated[0]["status"] == "ok"
        means[lambda_true] = float(evaluated[0]["posterior_mean_log_lambda"])

    assert means[40.0] > means[0.4] + 0.5


def test_departure_flags_require_movement_toward_the_truth():
    left_the_prior = departure_flags(
        posterior_mean_log_lambda=np.log(40.0),
        posterior_sd_log_lambda=0.2,
        ci_low_log_lambda=np.log(15.0),
        ci_high_log_lambda=np.log(90.0),
        prior_mean_log_lambda=np.log(10.0),
        prior_sd_log_lambda=1.0,
        true_log_lambda=np.log(50.0),
    )
    assert left_the_prior == {"shifted": True, "excluded_prior_mean": True, "recovered": True}

    wrong_direction = departure_flags(
        posterior_mean_log_lambda=np.log(1.0),
        posterior_sd_log_lambda=0.2,
        ci_low_log_lambda=np.log(0.4),
        ci_high_log_lambda=np.log(2.0),
        prior_mean_log_lambda=np.log(10.0),
        prior_sd_log_lambda=1.0,
        true_log_lambda=np.log(50.0),
    )
    assert wrong_direction["shifted"] is False
    assert wrong_direction["excluded_prior_mean"] is True
    assert wrong_direction["recovered"] is False


def test_majority_crossing_uses_more_than_half_the_seeds():
    rows = []
    flags_by_seed = {
        0: {10: False, 20: True},
        1: {10: False, 20: True},
        2: {10: False, 20: False},
    }
    for seed, flags in flags_by_seed.items():
        for n_shots, excluded in flags.items():
            rows.append(
                {
                    "status": "ok",
                    "true_sigma": 0.075,
                    "true_lambda": 100.0,
                    "prior_name": "uniform",
                    "n_shots": n_shots,
                    "seed": seed,
                    "shifted": False,
                    "excluded_prior_mean": excluded,
                    "recovered": False,
                }
            )

    summary = majority_crossings(rows)
    assert summary == [
        {
            "true_sigma": 0.075,
            "true_lambda": 100.0,
            "prior_name": "uniform",
            "shifted": None,
            "excluded_prior_mean": 20,
            "recovered": None,
        }
    ]


def test_default_prefixes_stop_at_the_simulated_length():
    assert resolve_prefixes(200, None) == [10, 25, 50, 100, 200]
    assert resolve_prefixes(30, None) == [10, 25, 30]


def test_prepared_angular_shot_matches_the_accumulator():
    rng = np.random.default_rng(7)
    value_map = rng.random((12, 14), dtype=float)
    shots_df = pd.DataFrame(
        [
            {
                "event_id": 101,
                "start_x": 75.5,
                "start_y": 0.6,
                "location_y": 0.2,
                "location_z": 1.4,
            }
        ]
    )
    situations = prepare_hockey_situations(shots_df, {101: {"value_map": value_map}})
    assert len(situations) == 1
    situation = situations[0]
    executed = np.array([situation.executed_action], dtype=float)
    sigma_grid = np.array([0.05, 0.12], dtype=float)
    log_lambda_grid = np.log(np.array([0.5, 4.0, 30.0]))
    row = shot_log_likelihoods(
        situation.target_angles,
        situation.utility_grid,
        situation.bin_size,
        executed,
        sigma_grid,
        log_lambda_grid,
    )[0]
    reference = _accumulate_vectorized(
        [(situation.target_angles, situation.executed_action, situation.utility_grid, situation.bin_size)],
        sigma_grid,
        log_lambda_grid,
    )
    informed = np.isfinite(row)
    assert np.array_equal(informed, np.isfinite(reference))
    np.testing.assert_allclose(row[informed], reference[informed])
