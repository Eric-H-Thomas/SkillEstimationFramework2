#!/usr/bin/env python
"""Generate the 3x3 H-JEEDS bias-confidence prior visual aids."""

# TODO: This file still requires human verification. Do not use in research until it has been verified. Remove this
#   comment when done.

from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "BiasConfidenceVisuals"
REPO_ROOT = OUTPUT_DIR.parents[2]
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

from HJEEDS.config import DEFAULT_HYPERPRIORS, DEFAULT_TRUE_POPULATION
from HJEEDS.darts_hierarchical_prior_sensitivity import (
    DEFAULT_SIGNIFICANT_BIAS_SD_UNITS,
    DEFAULT_SLIGHT_BIAS_SD_UNITS,
    DEFAULT_STRONG_STD_MULTIPLIER,
    DEFAULT_WEAK_STD_MULTIPLIER,
    PriorSensitivityCondition,
    build_condition_hyperpriors,
)
from HJEEDS.VisualAids.generate_log_skill_visuals import (
    MEAN_PRIOR_COLOR,
    MEAN_PRIOR_EDGE_COLOR,
    POPULATION_EDGE_COLOR,
    SURFACE_STRIDE,
    bivariate_normal_density,
    build_log_skill_mesh,
    configure_axis,
)


METADATA_FILENAME = "bias_confidence_visuals_metadata.csv"


def build_visual_conditions() -> tuple[PriorSensitivityCondition, ...]:
    """Return the paper's nine confidence x bias combinations."""

    confidence_levels = [
        ("weak", "weak", DEFAULT_WEAK_STD_MULTIPLIER),
        ("default", "default", 1.0),
        ("strong", "strong", DEFAULT_STRONG_STD_MULTIPLIER),
    ]
    bias_levels = [
        ("unbiased", "unbiased", 0.0),
        ("slightly biased", "slightly_biased", DEFAULT_SLIGHT_BIAS_SD_UNITS),
        (
            "significantly biased",
            "significantly_biased",
            DEFAULT_SIGNIFICANT_BIAS_SD_UNITS,
        ),
    ]

    return tuple(
        PriorSensitivityCondition(
            confidence_label=confidence_label,
            confidence_slug=confidence_slug,
            bias_label=bias_label,
            bias_slug=bias_slug,
            confidence_std_multiplier=float(confidence_multiplier),
            bias_sd_units=float(bias_sd_units),
        )
        for confidence_label, confidence_slug, confidence_multiplier in confidence_levels
        for bias_label, bias_slug, bias_sd_units in bias_levels
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save one condition as PNG and PDF."""

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


def plot_condition_overlay(
    condition: PriorSensitivityCondition,
    population_density: np.ndarray,
    mean_prior_density: np.ndarray,
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
) -> None:
    """Plot one bias-confidence overlay."""

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
        edgecolor=POPULATION_EDGE_COLOR,
        linewidth=0.28,
        antialiased=True,
        alpha=0.96,
        rstride=SURFACE_STRIDE,
        cstride=SURFACE_STRIDE,
        shade=False,
    )

    configure_axis(ax, 1.0, "relative density height")
    ax.set_title(
        f"{condition.confidence_label.title()} confidence, {condition.bias_label} prior mean",
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
    save_figure(fig, condition.condition_slug)


def write_metadata(rows: list[dict[str, float | str]]) -> None:
    """Write the concrete prior settings used by each visual."""

    if not rows:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / METADATA_FILENAME).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    x_mesh, y_mesh = build_log_skill_mesh()
    population_density = bivariate_normal_density(
        x_mesh,
        y_mesh,
        DEFAULT_TRUE_POPULATION.mean_vector,
        DEFAULT_TRUE_POPULATION.covariance_matrix,
    )

    metadata_rows: list[dict[str, float | str]] = []
    for condition in build_visual_conditions():
        hyperpriors = build_condition_hyperpriors(DEFAULT_HYPERPRIORS, condition)
        mean_prior_center = np.asarray(hyperpriors.mean_vector, dtype=float)
        mean_prior_covariance = hyperpriors.covariance_matrix
        mean_prior_density = bivariate_normal_density(
            x_mesh,
            y_mesh,
            mean_prior_center,
            mean_prior_covariance,
        )
        plot_condition_overlay(
            condition,
            population_density,
            mean_prior_density,
            x_mesh,
            y_mesh,
        )

        metadata_rows.append(
            {
                "condition_slug": condition.condition_slug,
                "confidence_level": condition.confidence_label,
                "bias_level": condition.bias_label,
                "confidence_std_multiplier": condition.confidence_std_multiplier,
                "bias_sd_units": condition.bias_sd_units,
                "mean_prior_log_sigma_center": mean_prior_center[0],
                "mean_prior_log_lambda_center": mean_prior_center[1],
                "mean_prior_sigma_center": math.exp(float(mean_prior_center[0])),
                "mean_prior_lambda_center": math.exp(float(mean_prior_center[1])),
                "mean_prior_log_sigma_sd": math.sqrt(float(hyperpriors.covariance_diagonal[0])),
                "mean_prior_log_lambda_sd": math.sqrt(float(hyperpriors.covariance_diagonal[1])),
            }
        )

    write_metadata(metadata_rows)


if __name__ == "__main__":
    main()
