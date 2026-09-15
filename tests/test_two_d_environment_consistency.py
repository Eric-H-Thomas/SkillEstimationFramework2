# Paper correspondence: Main `subsec:two_d_darts`.
"""Regression tests for independent 2D darts execution noise.

``Environments/Darts/RandomDarts/two_d_darts.draw_noise_sample`` seeds a frozen
``multivariate_normal`` from ``rng.bit_generator._seed_seq.entropy``, which is fixed
for the lifetime of a generator. Rebuilding that frozen distribution per call and
taking its first variate therefore returns the same displacement for every throw,
so a whole run shares one execution-noise vector scaled by each agent's sigma.
These tests pin the adapter to per-observation draws taken from the supplied
generator instead.
"""

from __future__ import annotations

import unittest

import numpy as np

from HJEEDS.environment_adapters import TwoDDartsEnvironment


class TwoDDartsExecutionNoiseTests(unittest.TestCase):
    """Ensure 2D execution noise is drawn independently per observation."""

    def setUp(self) -> None:
        self.adapter = TwoDDartsEnvironment()
        # Standard dartboard slice ordering; execution noise must not depend on it.
        self.reward_surface = (25, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10, 6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 11)

    def test_successive_draws_differ(self) -> None:
        rng = np.random.default_rng(4242)
        draws = [
            self.adapter.sample_noisy_action(rng, self.reward_surface, 30.0, (0.0, 0.0))
            for _ in range(32)
        ]
        self.assertEqual(len(set(draws)), len(draws))

    def test_advancing_the_generator_changes_the_draw(self) -> None:
        first = np.random.default_rng(4242)
        second = np.random.default_rng(4242)
        second.random(64)

        first_action = self.adapter.sample_noisy_action(first, self.reward_surface, 30.0, (0.0, 0.0))
        second_action = self.adapter.sample_noisy_action(second, self.reward_surface, 30.0, (0.0, 0.0))
        self.assertNotEqual(first_action, second_action)

    def test_noise_is_isotropic_with_requested_scale(self) -> None:
        sigma = 12.5
        rng = np.random.default_rng(90210)
        offsets = np.array(
            [
                self.adapter.sample_noisy_action(rng, self.reward_surface, sigma, (0.0, 0.0))
                for _ in range(20_000)
            ],
            dtype=float,
        )

        # Matches the legacy isotropic covariance diag(sigma**2, sigma**2).
        np.testing.assert_allclose(offsets.mean(axis=0), [0.0, 0.0], atol=sigma * 0.05)
        np.testing.assert_allclose(offsets.std(axis=0, ddof=1), [sigma, sigma], rtol=0.05)
        correlation = float(np.corrcoef(offsets[:, 0], offsets[:, 1])[0, 1])
        self.assertLess(abs(correlation), 0.05)

    def test_draw_is_reproducible_for_a_given_seed(self) -> None:
        expected = self.adapter.sample_noisy_action(
            np.random.default_rng(7), self.reward_surface, 20.0, (5.0, -3.0)
        )
        repeated = self.adapter.sample_noisy_action(
            np.random.default_rng(7), self.reward_surface, 20.0, (5.0, -3.0)
        )
        self.assertEqual(expected, repeated)

    def test_noise_ignores_the_reward_surface(self) -> None:
        shuffled = tuple(reversed(self.reward_surface))
        baseline = self.adapter.sample_noisy_action(
            np.random.default_rng(11), self.reward_surface, 15.0, (0.0, 0.0)
        )
        alternative = self.adapter.sample_noisy_action(
            np.random.default_rng(11), shuffled, 15.0, (0.0, 0.0)
        )
        self.assertEqual(baseline, alternative)

    def test_action_difference_is_euclidean_and_nonnegative(self) -> None:
        rng = np.random.default_rng(5)
        for _ in range(200):
            first = tuple(rng.uniform(-170.0, 170.0, size=2))
            second = tuple(rng.uniform(-170.0, 170.0, size=2))
            distance = self.adapter.compute_action_difference(first, second)
            self.assertGreaterEqual(distance, 0.0)
            self.assertAlmostEqual(
                distance,
                float(np.hypot(first[0] - second[0], first[1] - second[1])),
                places=6,
            )


if __name__ == "__main__":
    unittest.main()
