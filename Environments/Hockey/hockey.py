"""High-level utilities for analyzing hockey experiments.

This module is responsible for producing the expected-value (EV) heatmaps used
throughout the ``hockey-multi`` environment.  It is invoked by the
``runExpHockey.py`` and ``runExpGiven.py`` drivers after they serialize
experiment data under ``Experiments/hockey-multi``.  The script dynamically
loads :mod:`setupSpaces` so it can instantiate ``SpacesHockey`` and apply the
same discretized target grid that is used during simulation, ensuring the plots
line up with the environment configuration.

Functions
---------
``get_domain_name``
    Framework hook that identifies the domain name as ``"hockey-multi"``.
``getCovMatrix``
    Builds a covariance matrix given per-axis standard deviations and a shared
    correlation coefficient.
``draw_noise_sample``
    Creates a seeded multivariate normal distribution used for sampling noisy
    shot executions.
``sample_noisy_action``
    Adds Gaussian noise to a commanded action vector, optionally using a cached
    distribution.
``getNormalDistribution``
    Evaluates the probability density over the angular grid (and optionally
    persists debug contour plots).
``plotEVs``
    Entry point when running the module as a script; loads stored experiment
    data, evaluates EV surfaces for several ``xskill`` settings, and exports the
    resulting heatmaps per player attempt.
"""

from __future__ import annotations

import os
import pickle
import sys
from importlib.machinery import SourceFileLoader
from itertools import product
from random import sample
from typing import Any, Iterable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from scipy.signal import convolve2d
from scipy.stats import multivariate_normal

# ---------------------------------------------------------------------------
#  Dynamic module setup
# ---------------------------------------------------------------------------
# The legacy framework dynamically imports ``setupSpaces`` and exposes the
# resulting module under the name ``spaces`` so that downstream code can access
# ``spaces.SpacesHockey``.  The domain module performs a similar trick to
# provide ``get_domain_name`` when the environment is dynamically loaded.
script_path = os.path.realpath(__file__)
main_folder_name = script_path.split(f"Environments{os.sep}Hockey{os.sep}hockey.py")[0]

module = SourceFileLoader("setupSpaces.py", f"{main_folder_name}setupSpaces.py").load_module()
sys.modules["spaces"] = module
sys.modules["domain"] = sys.modules[__name__]


# ---------------------------------------------------------------------------
#  Domain helpers
# ---------------------------------------------------------------------------

def get_domain_name() -> str:
    """Return the name of this domain (required by the framework)."""

    return "hockey-multi"


def getCovMatrix(stdDevs: Iterable[float], rho: float) -> np.ndarray:
    """Build a covariance matrix from standard deviations and correlation.

    Parameters
    ----------
    stdDevs:
        A sequence of standard deviations for each dimension.
    rho:
        The correlation value between direction and elevation to apply to
        off-diagonal entries.
    """

    stdDevs = list(stdDevs)
    covMatrix = np.zeros((len(stdDevs), len(stdDevs)))
    np.fill_diagonal(covMatrix, np.square(stdDevs))

    # Fill the upper and lower triangles with the shared correlation value.
    for i in range(len(stdDevs)):
        for j in range(i + 1, len(stdDevs)):
            covMatrix[i, j] = np.prod(stdDevs) * rho
            covMatrix[j, i] = covMatrix[i, j]

    return covMatrix


def draw_noise_sample(
    rng: np.random.Generator,
    mean: Iterable[float] = (0.0, 0.0),
    X: Union[float, np.ndarray] = 0.0,
):
    """Create a multivariate normal distribution seeded from ``rng``.

    The legacy code stored the resulting distribution object to reuse when
    repeatedly sampling noisy actions.  This helper replicates that behavior and
    ensures deterministic sampling by explicitly deriving the seed from the
    generator.
    """

    entropy = rng.bit_generator._seed_seq.entropy  # type: ignore[attr-defined]
    seed = entropy[0] if isinstance(entropy, np.ndarray) else entropy
    return multivariate_normal(mean=mean, cov=X, seed=seed)


def sample_noisy_action(
    rng: np.random.Generator,
    mean: Iterable[float],
    L: float,
    a: Iterable[float],
    noiseModel: Optional[Any] = None,
) -> List[float]:
    """Draw a noisy action according to the provided noise model.

    ``L`` represents the square root of the variance (legacy naming).  When a
    noise model is not provided this function constructs a multivariate normal
    distribution and samples from it using the provided RNG.
    """

    if noiseModel is None:
        noiseModel = draw_noise_sample(rng, mean, L**2)

    noise = noiseModel.rvs(random_state=rng)
    return [a[0] + noise[0], a[1] + noise[1]]


def getNormalDistribution(
    rng: np.random.Generator,
    covMatrix: np.ndarray,
    resolution: Iterable[float],
    mean: Iterable[float],
    grid: np.ndarray,
    saveAt: Optional[str] = None,
    x: Optional[Any] = None,
) -> np.ndarray:
    """Return the normalized PDF evaluated over ``grid``.

    Optionally saves contour plots for debugging/visualization purposes.
    """

    distribution = draw_noise_sample(rng, mean, covMatrix)
    density = distribution.pdf(grid)
    density /= np.sum(density)

    if saveAt is not None:
        plt.contourf(grid[:, :, 0], grid[:, :, 1], density)
        plt.savefig(f"{saveAt}{os.sep}pdfs{os.sep}xskill{x}.jpg", bbox_inches="tight")
        plt.close()
        plt.clf()

    return density


# ---------------------------------------------------------------------------
#  Plot generation
# ---------------------------------------------------------------------------

def plotEVs() -> None:
    """Generate EV plots for the requested player and shot type."""

    try:
        experimentFolder = sys.argv[1]
        playerID = sys.argv[2]
        typeShot = sys.argv[3]
    except Exception:
        print(
            "Need to specify the name of the folder for the experiment (located under\n"
            "'Experiments/hockey-multi/'), the ID of the player and type of shot as\n"
            "command line argument."
        )
        sys.exit(1)

    mainFolder = f"Experiments{os.sep}hockey-multi{os.sep}{experimentFolder}{os.sep}Data{os.sep}"
    saveAt = (
        f"{mainFolder}Plots{os.sep}AngularHeatmapsPerXskill{os.sep}"
        f"Player{playerID}{os.sep}{typeShot}"
    )

    # Ensure the output directory structure exists.
    folders = [
        f"{mainFolder}Plots{os.sep}",
        f"{mainFolder}Plots{os.sep}AngularHeatmapsPerXskill{os.sep}",
        f"{mainFolder}Plots{os.sep}AngularHeatmapsPerXskill{os.sep}Player{playerID}{os.sep}",
        saveAt,
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    # Load the pickled experiment data.
    folder = f"{mainFolder}AngularHeatmaps{os.sep}"
    fileName = f"angular_heatmap_data_player_{playerID}_type_shot_{typeShot}.pkl"
    try:
        with open(folder + fileName, "rb") as infile:
            data = pickle.load(infile)
    except Exception as exc:
        print(exc)
        print("Can't load data for that player.")
        sys.exit(1)

    cmapStr = "gist_rainbow"
    c = 0.4
    n = plt.cm.jet.N
    cmap = (1.0 - c) * plt.get_cmap(cmapStr)(np.linspace(0.0, 1.0, n)) + c * np.ones((n, 4))
    cmap = ListedColormap(cmap)

    # Build the hockey target grid using the spaces helper.
    delta = None
    spaces = sys.modules["spaces"].SpacesHockey([], 1, sys.modules["domain"], delta)
    Y = spaces.targetsY
    Z = spaces.targetsZ
    targetsUtilityGridY, targetsUtilityGridZ = np.meshgrid(Y, Z)
    targetsUtilityGridYZ = np.stack((targetsUtilityGridY, targetsUtilityGridZ), axis=-1)
    shape = targetsUtilityGridYZ.shape
    listedTargetsUtilityGridYZ = targetsUtilityGridYZ.reshape((shape[0] * shape[1], shape[2]))

    # Candidate x-skills (radians) and correlation values.
    minX = 0.004
    maxX = np.pi / 4
    tempXskills = np.linspace(minX, maxX, 10)
    xskills = np.concatenate((np.linspace(minX, tempXskills[1], 10), np.array(tempXskills[2:])))
    rhos = [0.0]

    rng = np.random.default_rng(1000)
    allInfo = list(product(xskills, xskills))
    allInfo = list(product(allInfo, rhos))
    for idx in range(len(allInfo)):
        allInfo[idx] = list(eval(str(allInfo[idx]).replace(")", "").replace("(", "")))
    allInfo = np.round(allInfo, 4)

    # Legacy behavior: use only diagonal covariance combos.
    allInfo = []
    for xi in xskills:
        allInfo.append([xi, xi, 0.0])

    # Restrict to a random subset of player attempts for faster plotting.
    rows = list(data.keys())
    try:
        sampleRows = sample(rows, 20)
    except Exception:
        sampleRows = rows
    data = {k: data[k] for k in sampleRows}

    saveAtOriginal = saveAt
    for index in data:
        saveAt = saveAtOriginal + os.sep + str(index)
        os.makedirs(f"{saveAt}{os.sep}pdfs", exist_ok=True)

        heatmap = data[index]["heat_map"]
        shape = heatmap.shape
        listedUtilities = heatmap.reshape((shape[0] * shape[1], 1))
        Zs = data[index]["gridUtilitiesComputed"]
        gridTargetsAngular = data[index]["gridTargetsAngular"]
        listedTargetsAngular = data[index]["listedTargetsAngular"]
        executedAction = [data[index]["shot_location"][0], data[index]["shot_location"][1]]
        executedActionAngular = data[index]["executedActionAngular"]
        dirs, elevations = data[index]["dirs"], data[index]["elevations"]

        middle = int(len(dirs) / 2) - 1
        mean = [dirs[middle], elevations[middle]]
        spaces.delta = [abs(dirs[0] - dirs[1]), abs(elevations[0] - elevations[1])]
        playerLocation = [data[index]["start_x"], data[index]["start_y"]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        norm = plt.Normalize(0.0, np.max(listedUtilities))
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        ax1.scatter(
            listedTargetsUtilityGridYZ[:, 0],
            listedTargetsUtilityGridYZ[:, 1],
            c=cmap(norm(listedUtilities)),
        )
        ax1.scatter(executedAction[0], executedAction[1], c="black", marker="x")
        ax1.set_title("Given Heatmap - YZ")
        fig.colorbar(sm, ax=ax1)

        ax2.scatter(
            listedTargetsAngular[:, 0],
            listedTargetsAngular[:, 1],
            c=cmap(norm(Zs.flatten())),
        )
        ax2.scatter(executedActionAngular[0], executedActionAngular[1], c="black", marker="x")
        ax2.set_title("Computed Heatmap - Angular")
        fig.colorbar(sm, ax=ax2)
        plt.suptitle(f"Player Location: {playerLocation}")
        plt.tight_layout()
        plt.savefig(f"{saveAt}{os.sep}heatmaps.jpg", bbox_inches="tight")
        plt.close()

        norm = plt.Normalize(np.min(Zs), np.max(Zs))
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        for iii, x in enumerate(allInfo):
            covMatrix = getCovMatrix([x[0], x[1]], x[2])
            key = spaces.get_key([x[0], x[1]], x[2])
            spaces.pdfsPerXskill[key] = getNormalDistribution(
                rng,
                covMatrix,
                spaces.delta,
                mean,
                gridTargetsAngular,
                saveAt,
                key,
            )

            EVs = convolve2d(Zs, spaces.pdfsPerXskill[key], mode="same", fillvalue=0.0)
            maxEV = np.max(EVs)
            ii = np.unravel_index(EVs.argmax(), EVs.flatten().shape)

            fig, ax = plt.subplots()
            cbar = fig.colorbar(sm, ax=ax)
            cbar.ax.get_yaxis().labelpad = 15
            cbar.ax.set_ylabel("Expected Utilities", rotation=270)

            ax.scatter(
                listedTargetsAngular[:, 0],
                listedTargetsAngular[:, 1],
                c=cmap(norm(EVs.flatten())),
            )
            ax.scatter(
                listedTargetsAngular[:, 0][ii],
                listedTargetsAngular[:, 1][ii],
                color="black",
                marker="X",
                s=60,
                edgecolors="black",
                label="Max Expected Utility",
            )
            ax.set_title(f"xskill: {key}")
            plt.savefig(f"{saveAt}{os.sep}{iii}-xskill{key}.jpg", bbox_inches="tight")
            plt.close()
            plt.clf()


if __name__ == "__main__":
    plotEVs()
