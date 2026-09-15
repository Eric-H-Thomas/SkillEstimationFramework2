# Paper correspondence: Main `sec:model` and `subsec:inference`.
"""Regression tests for population fitting and simulated decision-skill truth."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.special import ndtr

from HJEEDS.config import build_config_from_args, parse_args
from HJEEDS.decision_models import (
    DECEPTIVE_DECISION_MODEL_SLUG,
    FLIP_DECISION_MODEL_SLUG,
    RATIONAL_DECISION_MODEL_SLUG,
    SOFTMAX_DECISION_MODEL_SLUG,
)
from HJEEDS.estimation import fit_population_hyperparameters_map
from HJEEDS.pipeline import _true_rationality_percent
from HJEEDS.rationality import rationality_percent_from_expected_values


class PopulationFitRegressionTests(unittest.TestCase):
    """Protect optimizer behavior that materially affects H-JEEDS estimates."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = build_config_from_args(
            parse_args(["--seed", "default", "--num-seeds", "1"])
        )
        cls.sigma_grid = np.array([0.5, 1.0, 1.5, 2.5, 4.5], dtype=float)
        cls.log_lambda_grid = np.array([-4.0, -2.0, 0.0, 2.0, 4.0], dtype=float)

        log_sigma_mesh, log_lambda_mesh = np.meshgrid(
            np.log(cls.sigma_grid),
            cls.log_lambda_grid,
            indexing="ij",
        )
        cls.informative_grid = (
            -0.5 * np.square((log_sigma_mesh - np.log(2.5)) / 0.3)
            -0.5 * np.square((log_lambda_mesh - 2.0) / 0.5)
        )

    def _fit(self, grids: list[np.ndarray]) -> dict[str, object]:
        return fit_population_hyperparameters_map(
            self.config,
            grids,
            self.sigma_grid,
            self.log_lambda_grid,
        )

    def test_fit_is_invariant_to_agent_specific_additive_constants(self) -> None:
        unshifted = [self.informative_grid.copy() for _ in range(8)]
        shifted = [
            self.informative_grid + (agent_index + 1) * 1_000_000.0
            for agent_index in range(8)
        ]

        fit_unshifted = self._fit(unshifted)
        fit_shifted = self._fit(shifted)

        for key in ("mu_eta", "mu_rho", "tau_eta", "tau_rho", "r", "objective_value"):
            with self.subTest(key=key):
                self.assertAlmostEqual(
                    float(fit_unshifted[key]),
                    float(fit_shifted[key]),
                    places=8,
                )

    def test_optimizer_converges_and_never_degrades_initial_objective(self) -> None:
        fitted = self._fit([self.informative_grid.copy() for _ in range(8)])

        self.assertTrue(fitted["converged"])
        self.assertEqual(fitted["selected_solution"], "optimizer")
        self.assertGreater(int(fitted["num_optimizer_iterations"]), 1)
        self.assertGreater(float(fitted["objective_improvement"]), 0.0)
        self.assertGreaterEqual(
            float(fitted["objective_value"]),
            float(fitted["initial_objective_value"]),
        )

    def test_invalid_overflowing_optimizer_result_falls_back_to_initialization(self) -> None:
        invalid_result = SimpleNamespace(
            x=np.array([0.0, 0.0, 1_000.0, 1_000.0, 0.0]),
            success=True,
            message="synthetic invalid optimizer result",
            nfev=1,
            nit=1,
        )

        with patch("scipy.optimize.minimize", return_value=invalid_result):
            fitted = self._fit([self.informative_grid.copy()])

        self.assertEqual(fitted["selected_solution"], "initial")
        self.assertFalse(fitted["converged"])
        self.assertIsNone(fitted["optimizer_objective_value"])
        self.assertEqual(fitted["objective_improvement"], 0.0)
        self.assertEqual(fitted["objective_value"], fitted["initial_objective_value"])


class RationalityTruthRegressionTests(unittest.TestCase):
    """Distinguish deterministic rational behavior from latent lambda labels."""

    def test_rational_policy_truth_is_always_one_hundred_percent(self) -> None:
        expected_values = np.array([0.2, 0.8, 1.4], dtype=float)

        for log_lambda in (-5.0, 0.0, 5.0):
            with self.subTest(log_lambda=log_lambda):
                self.assertEqual(
                    _true_rationality_percent(
                        expected_values,
                        log_lambda,
                        RATIONAL_DECISION_MODEL_SLUG,
                    ),
                    100.0,
                )

    def test_other_policy_truth_semantics_are_unchanged(self) -> None:
        expected_values = np.array([0.2, 0.8, 1.4], dtype=float)
        log_lambda = 0.7
        expected = rationality_percent_from_expected_values(expected_values, log_lambda)

        for slug in (
            SOFTMAX_DECISION_MODEL_SLUG,
            FLIP_DECISION_MODEL_SLUG,
            DECEPTIVE_DECISION_MODEL_SLUG,
        ):
            with self.subTest(slug=slug):
                self.assertEqual(
                    _true_rationality_percent(expected_values, log_lambda, slug),
                    expected,
                )


class ExactOneDExpectedValueTests(unittest.TestCase):
    """Protect exact Normal-CDF integration at board and reward boundaries."""

    def test_constant_reward_surface_includes_exact_off_board_tail_loss(self) -> None:
        from HJEEDS.darts_environment import BOARD_LIMIT, compute_expected_value_curve

        sigma = 1.7
        values, actions = compute_expected_value_curve((), sigma, 0.1)
        expected = ndtr((BOARD_LIMIT - actions) / sigma) - ndtr(
            (-BOARD_LIMIT - actions) / sigma
        )

        np.testing.assert_allclose(values, expected, rtol=0.0, atol=2e-15)

    def test_alternating_reward_intervals_use_exact_cdf_mass(self) -> None:
        from HJEEDS.darts_environment import BOARD_LIMIT, compute_expected_value_curve

        sigma = 1.3
        values, actions = compute_expected_value_curve((-1.0, 2.0), sigma, 0.2)
        edges = np.array([-BOARD_LIMIT, -1.0, 2.0, BOARD_LIMIT])
        rewards = np.array([1.0, 2.0, 1.0])
        masses = np.diff(ndtr((edges[:, None] - actions[None, :]) / sigma), axis=0)
        expected = np.sum(rewards[:, None] * masses, axis=0)

        np.testing.assert_allclose(values, expected, rtol=0.0, atol=2e-15)


if __name__ == "__main__":
    unittest.main()
