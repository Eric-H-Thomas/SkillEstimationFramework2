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
from HJEEDS.plot_population_shape_robustness import SHAPE_LABELS, SHAPE_ORDER
from HJEEDS.population_shapes import (
    BIMODAL_BETWEEN_VARIANCE_FRACTION,
)
from HJEEDS.sensitivity_plot_common import (
    GRID_COLOR,
    NUMERIC_3_COLORS,
    TEXT_COLOR,
    blend,
    configure_matplotlib,
    save_figure_bundle,
)


DEFAULT_OUTPUT_STEM = Path(
    "HJEEDS/results/hjeeds_paper_500_seeds/population_shape/population_shape_distributions"
)
VISUALIZATION_COLORS = {
    shape_slug: color for shape_slug, color in zip(SHAPE_ORDER, NUMERIC_3_COLORS)
}
DENSITY_COLORS = (
    blend(NUMERIC_3_COLORS[0], "#FFFFFF", 0.90),
    blend(NUMERIC_3_COLORS[1], "#FFFFFF", 0.45),
    NUMERIC_3_COLORS[2],
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument(
        "--single-column",
        action="store_true",
        help="Render a compact two-dimensional figure for one AAAI text column.",
    )
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


def _render_single_column(
    output_stem: Path,
    dpi: int,
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
) -> None:
    """Render readable normalized-density panels at their final column width."""

    import matplotlib.pyplot as plt
    from matplotlib import colors

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 7.6,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    density_colormap = colors.LinearSegmentedColormap.from_list(
        "hjeeds_paper_density",
        DENSITY_COLORS,
    )
    # Keep the panels tall enough to show the Gaussian modes clearly while
    # moving the shared density scale beside them to reduce the total height.
    figure, axes = plt.subplots(3, 1, figsize=(3.35, 2.82), sharex=True, sharey=True)
    density_image = None
    for axis, shape_slug in zip(np.ravel(axes), SHAPE_ORDER):
        density = _shape_density(shape_slug, x_mesh, y_mesh)
        relative_density = density / float(np.max(density))
        density_image = axis.pcolormesh(
            x_mesh,
            y_mesh,
            relative_density,
            shading="auto",
            cmap=density_colormap,
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )
        axis.contour(
            x_mesh,
            y_mesh,
            relative_density,
            levels=(0.5,) if shape_slug == "uniform" else (0.25, 0.5, 0.75),
            colors="#FFFFFF",
            linewidths=0.42,
            alpha=0.72,
        )
        axis.set_title(
            SHAPE_LABELS[shape_slug],
            loc="left",
            pad=2.0,
            fontweight="bold",
            color=TEXT_COLOR,
        )
        axis.tick_params(axis="both", colors=TEXT_COLOR, length=2.3, width=0.6, pad=1.5)
        for spine in axis.spines.values():
            spine.set_color("#AAA4B3")
            spine.set_linewidth(0.55)

    sigma_ticks = (0.5, 1.5, 4.5)
    lambda_ticks = (0.001, 1.0, 100.0)
    axes[-1].set_xticks(np.log(sigma_ticks), [f"{value:g}" for value in sigma_ticks])
    axes[-1].set_xlabel(r"Execution noise, $\sigma$", color=TEXT_COLOR, labelpad=2.5)
    for axis in axes:
        axis.set_yticks(np.log(lambda_ticks), [f"{value:g}" for value in lambda_ticks])
    figure.supylabel(r"Decision skill, $\lambda$", x=0.055, color=TEXT_COLOR, fontsize=7.2)

    # Narrow the density panels slightly to reserve a dedicated right-side band
    # for the shared color scale. This recovers the bottom band previously used
    # by the horizontal colorbar without flattening the three density panels.
    figure.subplots_adjust(left=0.19, right=0.79, top=0.95, bottom=0.15, hspace=0.30)
    if density_image is not None:
        colorbar_axis = figure.add_axes((0.835, 0.19, 0.035, 0.68))
        colorbar = figure.colorbar(
            density_image,
            cax=colorbar_axis,
            orientation="vertical",
            ticks=(0.0, 0.5, 1.0),
        )
        colorbar.set_label("Relative density", color=TEXT_COLOR, fontsize=6.7, labelpad=3.0)
        colorbar.ax.tick_params(labelsize=6.0, colors=TEXT_COLOR, length=2.0, pad=1.5)
        colorbar.outline.set_linewidth(0.5)
        colorbar.outline.set_edgecolor("#AAA4B3")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    save_figure_bundle(figure, output_stem, dpi)
    plt.close(figure)


def render(output_stem: Path, dpi: int, *, single_column: bool = False) -> None:
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

    if single_column:
        _render_single_column(output_stem, dpi, x_mesh, y_mesh)
        return

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
    render(args.output_stem, args.dpi, single_column=args.single_column)
    print(f"Wrote population-shape distribution figure to {args.output_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
