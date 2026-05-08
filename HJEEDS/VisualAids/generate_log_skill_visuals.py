#!/usr/bin/env python
"""Generate visual aids for the H-JEEDS log-skill population model."""

# TODO: This file still requires human verification. Do not use in research until it has been verified. Remove this
#   comment when done.

from __future__ import annotations

import os
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUTPUT_DIR.parents[1]
MATPLOTLIB_CACHE_DIR = Path("/private/tmp/hjeeds_visualaids_matplotlib_cache")

MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from HJEEDS.config import (
    DEFAULT_HYPERPRIORS,
    DEFAULT_LAMBDA_MAX,
    DEFAULT_LAMBDA_MIN,
    DEFAULT_SIGMA_MAX,
    DEFAULT_SIGMA_MIN,
    DEFAULT_TRUE_POPULATION,
)


SURFACE_STRIDE = 2
POPULATION_COLOR = "#0B84A5"
MEAN_PRIOR_COLOR = "#F25C05"
POPULATION_EDGE_COLOR = "#45515C"
MEAN_PRIOR_EDGE_COLOR = "#7A3D00"


def bivariate_normal_density(
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """Evaluate a bivariate Normal density on a 2D mesh."""

    determinant = float(np.linalg.det(covariance))
    if determinant <= 0.0:
        raise ValueError("Covariance matrix must be positive definite.")

    inverse_covariance = np.linalg.inv(covariance)
    points = np.stack([x_mesh, y_mesh], axis=-1)
    centered = points - mean
    quadratic = np.einsum("...i,ij,...j->...", centered, inverse_covariance, centered)
    normalization = 1.0 / (2.0 * np.pi * np.sqrt(determinant))
    return normalization * np.exp(-0.5 * quadratic)


def build_log_skill_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Return a dense mesh over the same log-skill support used by H-JEEDS."""

    log_sigma_values = np.linspace(np.log(DEFAULT_SIGMA_MIN), np.log(DEFAULT_SIGMA_MAX), 100)
    log_lambda_values = np.linspace(np.log(DEFAULT_LAMBDA_MIN), np.log(DEFAULT_LAMBDA_MAX), 112)
    return np.meshgrid(log_sigma_values, log_lambda_values, indexing="xy")


def configure_axis(ax: plt.Axes, z_max: float, z_label: str) -> None:
    """Apply shared labels, limits, and camera angle to the 3D plots."""

    ax.set_xlabel(r"$\log \sigma$ (execution noise)", labelpad=8, fontsize=12)
    ax.set_ylabel(r"$\log \lambda$ (decision skill)", labelpad=8, fontsize=12)
    ax.set_zlabel(z_label, labelpad=7, fontsize=12)
    ax.set_xlim(np.log(DEFAULT_SIGMA_MIN), np.log(DEFAULT_SIGMA_MAX))
    ax.set_ylim(np.log(DEFAULT_LAMBDA_MIN), np.log(DEFAULT_LAMBDA_MAX))
    ax.set_zlim(0.0, z_max * 1.06)
    ax.view_init(elev=28, azim=-54)
    ax.set_box_aspect((1.15, 1.35, 0.72), zoom=0.86)
    ax.tick_params(axis="both", which="major", labelsize=11, pad=2)
    ax.tick_params(axis="z", which="major", labelsize=11, pad=2)
    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor("#9E9E9E")
        axis._axinfo["grid"]["color"] = (0.78, 0.78, 0.78, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.55


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save one figure as both a high-resolution PNG and a vector PDF."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(
            OUTPUT_DIR / f"{stem}{suffix}",
            dpi=340,
            bbox_inches="tight",
            pad_inches=0.18,
            facecolor="white",
        )
    plt.close(fig)


def plot_population_distribution(
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    population_density: np.ndarray,
) -> None:
    """Plot the population-level density over individual log-skill profiles."""

    fig = plt.figure(figsize=(9.8, 7.0), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        x_mesh,
        y_mesh,
        population_density,
        cmap="winter",
        edgecolor="#16515C",
        linewidth=0.28,
        antialiased=True,
        alpha=1.0,
        rstride=SURFACE_STRIDE,
        cstride=SURFACE_STRIDE,
        shade=False,
    )

    configure_axis(ax, float(np.max(population_density)), "density")
    ax.set_title(
        r"Population-level log-skill distribution: "
        r"$z_i \mid \mu, \Sigma_z$",
        pad=16,
        fontsize=15,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.9)
    save_figure(fig, "population_log_skill_density")


def plot_population_with_mean_prior(
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    population_density: np.ndarray,
    mean_prior_density: np.ndarray,
) -> None:
    """Overlay the skill-profile density and the prior over its center."""

    population_height = population_density / float(np.max(population_density))
    mean_prior_height = mean_prior_density / float(np.max(mean_prior_density)) * 0.72

    fig = plt.figure(figsize=(9.8, 7.0), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        x_mesh,
        y_mesh,
        mean_prior_height,
        cmap="autumn",
        edgecolor=MEAN_PRIOR_EDGE_COLOR,
        linewidth=0.26,
        antialiased=True,
        alpha=0.86,
        rstride=SURFACE_STRIDE,
        cstride=SURFACE_STRIDE,
        shade=False,
    )
    ax.plot_surface(
        x_mesh,
        y_mesh,
        population_height,
        cmap="winter",
        edgecolor="#16515C",
        linewidth=0.28,
        antialiased=True,
        alpha=0.96,
        rstride=SURFACE_STRIDE,
        cstride=SURFACE_STRIDE,
        shade=False,
    )

    configure_axis(ax, 1.0, "relative density height")
    ax.set_title(
        r"Population density and prior over its center",
        pad=16,
        fontsize=15,
    )
    legend_handles = [
        Line2D([0], [0], color="#008B8B", lw=7, alpha=0.95),
        Line2D([0], [0], color=MEAN_PRIOR_COLOR, lw=7, alpha=0.95),
    ]
    ax.legend(
        legend_handles,
        [r"Skill profiles: $z_i \mid \mu, \Sigma_z$", r"Mean prior: $\mu \sim N(m_0, S_0)$"],
        loc="upper left",
        frameon=False,
        fontsize=13,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.9)
    save_figure(fig, "population_and_mean_prior_overlay")


def main() -> None:
    x_mesh, y_mesh = build_log_skill_mesh()

    population_mean = DEFAULT_TRUE_POPULATION.mean_vector
    population_covariance = DEFAULT_TRUE_POPULATION.covariance_matrix
    mean_prior_center = np.asarray(DEFAULT_HYPERPRIORS.mean_vector, dtype=float)
    mean_prior_covariance = DEFAULT_HYPERPRIORS.covariance_matrix

    population_density = bivariate_normal_density(
        x_mesh,
        y_mesh,
        population_mean,
        population_covariance,
    )
    mean_prior_density = bivariate_normal_density(
        x_mesh,
        y_mesh,
        mean_prior_center,
        mean_prior_covariance,
    )

    plot_population_distribution(x_mesh, y_mesh, population_density)
    plot_population_with_mean_prior(
        x_mesh,
        y_mesh,
        population_density,
        mean_prior_density,
    )


if __name__ == "__main__":
    main()
