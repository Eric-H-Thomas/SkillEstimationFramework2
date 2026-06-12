# This file has been fully verified by a human researcher as of 6/12/26 at 3:14 PM MDT.
"""Environment-specific domain adapters for HJEEDS likelihood and simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EnvironmentDomain(ABC):
    """Abstract interface for environment-specific operations."""

    @abstractmethod
    def compute_expected_value_curve(
        self,
        reward_surface: tuple[float, ...],
        sigma: float,
        delta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute EV curve for all actions under given execution skill."""

    @abstractmethod
    def sample_noisy_action(
        self,
        rng: np.random.Generator,
        reward_surface: tuple[float, ...],
        sigma: float,
        intended_action: float | tuple[float, float],
    ) -> float | tuple[float, float]:
        """Draw noisy execution of intended action; return observed action."""

    @abstractmethod
    def compute_action_difference(
        self,
        action_1: float | tuple[float, float],
        action_2: float | tuple[float, float],
        reward_surface: tuple[float, ...] | None = None,
    ) -> float:
        """Return distance between two actions."""


class OneDDartsEnvironment(EnvironmentDomain):
    """1D darts environment adapter."""

    def compute_expected_value_curve(self, reward_surface, sigma, delta):
        from Environments.Darts.RandomDarts import darts

        return darts.compute_expected_value_curve(reward_surface, sigma, delta)

    def sample_noisy_action(self, rng, reward_surface, sigma, intended_action):
        from Environments.Darts.RandomDarts import darts

        return darts.sample_noisy_action(rng, reward_surface, sigma, intended_action)

    def compute_action_difference(self, action_1, action_2, reward_surface=None):
        from Environments.Darts.RandomDarts import darts

        return darts.calculate_wrapped_action_difference(action_1, action_2)


class TwoDDartsEnvironment(EnvironmentDomain):
    """2D darts environment adapter."""

    def compute_expected_value_curve(self, reward_surface, sigma, delta):
        from Environments.Darts.RandomDarts import two_d_darts

        rng = np.random.default_rng(0)
        XYs, EVs, _ = two_d_darts.compute_expected_value_curve(rng, reward_surface, sigma, delta)
        on_board_xs, on_board_ys, on_board_evs = two_d_darts.getInfoOnBoardOnly(XYs, EVs)
        actions = np.empty(len(on_board_xs), dtype=object)
        actions[:] = [
            (float(x_coord), float(y_coord))
            for x_coord, y_coord in zip(on_board_xs, on_board_ys)
        ]
        return np.asarray(on_board_evs, dtype=float), actions

    def sample_noisy_action(self, rng, reward_surface, sigma, intended_action):
        from Environments.Darts.RandomDarts import two_d_darts

        result = two_d_darts.sample_noisy_action(rng, reward_surface, sigma, intended_action)
        return tuple(result)

    def compute_action_difference(self, action_1, action_2, reward_surface=None):
        from Environments.Darts.RandomDarts import two_d_darts

        return float(two_d_darts.calculate_action_difference(action_1, action_2))


class BaseballMultiEnvironment(EnvironmentDomain):
    """Statcast baseball uses ``baseball_pitch`` directly; this stub satisfies the factory."""

    def compute_expected_value_curve(self, reward_surface, sigma, delta):
        raise NotImplementedError("Use HJEEDS.baseball_pitch for Statcast utility surfaces.")

    def sample_noisy_action(self, rng, reward_surface, sigma, intended_action):
        from HJEEDS.baseball_pitch import sample_noisy_action

        return sample_noisy_action(rng, intended_action, sigma)

    def compute_action_difference(self, action_1, action_2, reward_surface=None) -> float:
        a1 = np.asarray(action_1, dtype=float)
        a2 = np.asarray(action_2, dtype=float)
        return float(np.linalg.norm(a1 - a2))


def get_environment_domain(environment: str) -> EnvironmentDomain:
    """Factory function to get the appropriate domain adapter."""

    if environment == "1d":
        return OneDDartsEnvironment()
    if environment == "2d":
        return TwoDDartsEnvironment()
    if environment == "baseball":
        return BaseballMultiEnvironment()
    raise ValueError(f"Unknown environment: {environment}. Choose '1d', '2d', or 'baseball'.")