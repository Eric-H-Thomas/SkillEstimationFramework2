"""Per-shot space objects for MCSE (PFE) on Blackhawks angular xG surfaces."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.signal import convolve2d

from Environments.Hockey import hockey as hockey_domain


class BlackhawksPFESpaces:
    """Space cache for ``QREMethod_Multi_Particles`` on Blackhawks shot data.

    Mirrors the interface used by the ``hockey-multi`` branch in
    ``joint_pfe.py`` and ``SpacesHockey.updateSpaceParticles``.
    """

    domainName = "hockey-multi"
    domain = hockey_domain

    def __init__(
        self,
        y_grid: np.ndarray,
        z_grid: np.ndarray,
        grid_targets_angular: np.ndarray,
    ) -> None:
        self.y_grid = np.asarray(y_grid, dtype=float)
        self.z_grid = np.asarray(z_grid, dtype=float)
        self.grid = np.asarray(grid_targets_angular, dtype=float)
        self.possibleTargets = self.grid.reshape(-1, 2)

        dy = float(np.diff(self.y_grid).mean()) if len(self.y_grid) > 1 else 1.0
        dz = float(np.diff(self.z_grid).mean()) if len(self.z_grid) > 1 else 1.0
        self.delta = (dy, dz)

        middle = max(0, int(len(self.y_grid) / 2) - 1)
        self.mean = [float(self.y_grid[middle]), float(self.z_grid[middle])]

        self.pdfsPerXskill: dict[str, np.ndarray] = {}
        self.evsPerXskill: dict[str, np.ndarray] = {}

    @staticmethod
    def get_key(info: Sequence[float], r: float) -> str:
        return "|".join(map(str, info)) + f"|{r}"

    def updateSpaceParticles(self, rng, each, state, info, wid=None) -> None:
        cov_matrix = self.domain.getCovMatrix(each[:-2], each[-2])
        key = self.get_key(each[:-2], each[-2])

        if key not in self.pdfsPerXskill:
            self.pdfsPerXskill[key] = self.domain.getNormalDistribution(
                rng,
                cov_matrix,
                self.delta,
                self.mean,
                self.grid,
            )

        if key not in self.evsPerXskill:
            zs = info["Zs"]
            self.evsPerXskill[key] = convolve2d(
                zs,
                self.pdfsPerXskill[key],
                mode="same",
                fillvalue=0.0,
            )

    def deleteSpaceParticles(self, each, state) -> None:
        # ``each`` is particle without lambda: [x_y, x_z, rho]
        if len(each) >= 3:
            key = self.get_key(each[:2], each[2])
        else:
            key = self.get_key(each[:-2], each[-2])
        try:
            del self.pdfsPerXskill[key]
            del self.evsPerXskill[key]
        except KeyError:
            pass
