# Paper correspondence: Main `subsec:one_d_darts`; Supplement `app:one_d_darts_environment`.
"""Regression tests for the paper's non-wrapped 1D environment routing."""

from __future__ import annotations

import unittest

import numpy as np

from HJEEDS import darts_environment
from HJEEDS.environment_adapters import OneDDartsEnvironment


class OneDDartsEnvironmentConsistencyTests(unittest.TestCase):
    """Ensure simulation adapters use the same geometry as inference."""

    def setUp(self) -> None:
        self.adapter = OneDDartsEnvironment()
        self.reward_surface = (-7.0, -5.0, -1.0, 2.0, 6.0, 9.0)

    def test_expected_value_curve_matches_paper_environment(self) -> None:
        expected_values, actions = darts_environment.compute_expected_value_curve(
            self.reward_surface,
            execution_noise_sd=1.7,
            grid_resolution=0.1,
        )
        adapter_values, adapter_actions = self.adapter.compute_expected_value_curve(
            self.reward_surface,
            sigma=1.7,
            delta=0.1,
        )

        np.testing.assert_array_equal(adapter_actions, actions)
        np.testing.assert_allclose(adapter_values, expected_values, rtol=0.0, atol=0.0)

    def test_noisy_action_matches_non_wrapped_paper_environment(self) -> None:
        direct_rng = np.random.default_rng(8128)
        adapter_rng = np.random.default_rng(8128)

        direct_action = darts_environment.sample_noisy_action(
            direct_rng,
            execution_noise_sd=2.0,
            intended_target=9.5,
        )
        adapter_action = self.adapter.sample_noisy_action(
            adapter_rng,
            self.reward_surface,
            sigma=2.0,
            intended_action=9.5,
        )

        self.assertEqual(adapter_action, direct_action)

    def test_action_distance_does_not_wrap_at_board_edges(self) -> None:
        self.assertEqual(self.adapter.compute_action_difference(9.0, -9.0), 18.0)
        self.assertEqual(self.adapter.compute_action_difference(-3.0, 2.5), 5.5)


if __name__ == "__main__":
    unittest.main()
