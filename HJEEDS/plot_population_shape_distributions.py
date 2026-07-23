# This file was AI-generated and still requires human review. Remove this comment when done.
"""Visualize the three true population shapes used in the sensitivity study."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from HJEEDS.config import (
    DEFAULT_LAMBDA_MAX,
    DEFAULT_LAMBDA_MIN,
    DEFAULT_SIGMA_MAX,
    DEFAULT_SIGMA_MIN,
    DEFAULT_TRUE_POPULATION,
)
from HJEEDS.plot_population_shape_robustness import SHAPE_COLORS, SHAPE_LABELS, SHAPE_ORDER
from HJEEDS.population_shapes import (
    BIMODAL_BETWEEN_VARIANCE_FRACTION,
)
from HJEEDS.sensitivity_plot_common import (
    GRID_COLOR,
    TEXT_COLOR,
    blend,
    configure_matplotlib,
    save_figure_bundle,
)


DEFAULT_OUTPUT_STEM = Path(
    "HJEEDS/results/hjeeds_paper_500_seeds/population_shape/population_shape_distributions"
)
VISUALIZATION_COLORS = {
    **SHAPE_COLORS,
    "uniform": "#A56DE2",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--dpi", type=int, default=450)
    return parser.parse_args(argv)


def _gaussian_density(
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """Evaluate a bivariate Gaussian density on a mesh."""

    centered = np.stack((x_mesh - mean[0], y_mesh - mean[1]), axis=-1)
    inverse = np.linalg.inv(covariance)
    exponent = -0.5 * np.einsum("...i,ij,...j->...", centered, inverse, centered)
    normalizer = 2.0 * math.pi * math.sqrt(float(np.linalg.det(covariance)))
    return np.exp(exponent) / normalizer


def _shape_density(shape_slug: str, x_mesh: np.ndarray, y_mesh: np.ndarray) -> np.ndarray:
    """Evaluate one experiment population density in native log-skill space."""

    mean = DEFAULT_TRUE_POPULATION.mean_vector
    covariance = DEFAULT_TRUE_POPULATION.covariance_matrix
    if shape_slug == "default":
        return _gaussian_density(x_mesh, y_mesh, mean, covariance)

    if shape_slug == "uniform":
        centered = np.stack((x_mesh - mean[0], y_mesh - mean[1]), axis=0)
        unit_coordinates = np.linalg.solve(
            np.linalg.cholesky(covariance),
            centered.reshape(2, -1),
        ).reshape(centered.shape)
        return np.all(np.abs(unit_coordinates) <= math.sqrt(3.0), axis=0).astype(float)

    if shape_slug == "bimodal":
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        largest_index = int(np.argmax(eigenvalues))
        offset = (
            math.sqrt(BIMODAL_BETWEEN_VARIANCE_FRACTION * float(eigenvalues[largest_index]))
            * eigenvectors[:, largest_index]
        )
        component_covariance = covariance - np.outer(offset, offset)
        return 0.5 * (
            _gaussian_density(x_mesh, y_mesh, mean - offset, component_covariance)
            + _gaussian_density(x_mesh, y_mesh, mean + offset, component_covariance)
        )

    raise ValueError(f"Unknown population shape: {shape_slug}")


def _configure_axis(
    axis,
    shape_slug: str,
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    density: np.ndarray,
) -> None:
    """Draw one three-dimensional population-density surface."""

    from matplotlib import colors

    base_color = VISUALIZATION_COLORS[shape_slug]
    relative_density = density / float(np.max(density))
    color_map = colors.LinearSegmentedColormap.from_list(
        f"{shape_slug}_density",
        [
            blend(base_color, "#FFFFFF", 0.94),
            blend(base_color, "#FFFFFF", 0.68),
            base_color,
            blend(base_color, TEXT_COLOR, 0.2),
            blend(base_color, TEXT_COLOR, 0.42),
        ],
    )
    face_colors = color_map(colors.PowerNorm(gamma=0.56, vmin=0.0, vmax=1.0)(relative_density))
    mesh_color = colors.to_rgba(blend(base_color, TEXT_COLOR, 0.62), alpha=0.44)
    axis.plot_surface(
        x_mesh,
        y_mesh,
        relative_density,
        rstride=3,
        cstride=3,
        facecolors=face_colors,
        edgecolor=mesh_color,
        linewidth=0.32,
        antialiased=True,
        shade=False,
    )

    axis.set_xlim(math.log(DEFAULT_SIGMA_MIN), math.log(DEFAULT_SIGMA_MAX))
    axis.set_ylim(math.log(DEFAULT_LAMBDA_MIN), math.log(DEFAULT_LAMBDA_MAX))
    axis.set_zlim(0.0, 1.05)
    sigma_ticks = (0.5, 1.5, 3.0, 4.5)
    lambda_ticks = (0.001, 0.1, 1.0, 10.0, 100.0)
    axis.set_xticks(np.log(sigma_ticks), [f"{value:g}" for value in sigma_ticks])
    axis.set_yticks(np.log(lambda_ticks), [f"{value:g}" for value in lambda_ticks])
    axis.set_zticks((0.0, 0.5, 1.0), ("0", "0.5", "1"))
    axis.set_xlabel(r"Execution noise, $\sigma$", color=TEXT_COLOR, labelpad=5.0)
    axis.set_ylabel(r"Decision skill, $\lambda$", color=TEXT_COLOR, labelpad=7.0)
    axis.set_zlabel("Relative density", color=TEXT_COLOR, labelpad=4.0)
    axis.set_title(
        SHAPE_LABELS[shape_slug],
        fontsize=10.8,
        fontweight="bold",
        color=TEXT_COLOR,
        y=0.965,
        pad=0.0,
    )
    axis.view_init(elev=29.0, azim=-120.0)
    axis.set_box_aspect((1.4, 1.08, 0.68))
    for axis_name in ("x", "y", "z"):
        axis.tick_params(axis=axis_name, colors=TEXT_COLOR, labelsize=6.4, pad=0.5)
    for coordinate_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        coordinate_axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        coordinate_axis.pane.set_edgecolor("#B8B3BF")
        coordinate_axis._axinfo["grid"]["color"] = colors.to_rgba(GRID_COLOR, 0.68)
        coordinate_axis._axinfo["grid"]["linewidth"] = 0.45


def render(output_stem: Path, dpi: int) -> None:
    """Render the shared-scale three-panel population-shape figure."""

    configure_matplotlib()
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    log_sigma = np.linspace(math.log(DEFAULT_SIGMA_MIN), math.log(DEFAULT_SIGMA_MAX), 121)
    log_lambda = np.linspace(math.log(DEFAULT_LAMBDA_MIN), math.log(DEFAULT_LAMBDA_MAX), 151)
    x_mesh, y_mesh = np.meshgrid(log_sigma, log_lambda, indexing="xy")

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14.2, 4.8),
        subplot_kw={"projection": "3d"},
    )
    for axis, shape_slug in zip(np.ravel(axes), SHAPE_ORDER):
        _configure_axis(
            axis,
            shape_slug,
            x_mesh,
            y_mesh,
            _shape_density(shape_slug, x_mesh, y_mesh),
        )

    figure.suptitle(
        "True population-shape conditions",
        x=0.5,
        y=0.975,
        fontsize=14.0,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.035, wspace=-0.04)
    save_figure_bundle(figure, output_stem, dpi)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""

    args = parse_args(argv)
    render(args.output_stem, args.dpi)
    print(f"Wrote population-shape distribution figure to {args.output_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
