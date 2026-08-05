"""Per-shot space objects for MCSE (PFE) on Blackhawks angular xG surfaces."""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np
from scipy.signal import convolve2d

from Environments.Hockey import hockey as hockey_domain

# Shares the BH_EV_NORMALIZE switch with BlackhawksJEEDS so an A/B run applies
# the same convention to both estimators.
EV_NORMALIZE = os.environ.get("BH_EV_NORMALIZE", "1") not in ("0", "", "false", "False")


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
            evs = convolve2d(
                zs,
                self.pdfsPerXskill[key],
                mode="same",
                fillvalue=0.0,
            )
            if EV_NORMALIZE:
                # Keep lambda on the same dimensionless scale JEEDS uses when
                # BH_EV_NORMALIZE is on, so the two estimators stay comparable.
                # Without it the EV scale shrinks as execution skill worsens and
                # lambda absorbs the difference, pinning it against the grid cap.
                ev_scale = float(np.max(evs) - np.mean(evs))
                if ev_scale > 1e-12:
                    evs = evs / ev_scale
            self.evsPerXskill[key] = evs

    def clear_particle_caches(self) -> None:
        """Drop PDF/EV caches for this shot (safe after the observation is done)."""
        self.pdfsPerXskill.clear()
        self.evsPerXskill.clear()

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
