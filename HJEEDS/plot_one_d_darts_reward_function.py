# This file was AI-generated and still requires human review. Remove this comment when done.
"""Render a polished example 1D-Darts reward function figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from HJEEDS.sensitivity_plot_common import (
    CHARCOAL,
    GRID_COLOR,
    TEXT_COLOR,
    configure_matplotlib as _configure_matplotlib,
    save_figure_bundle as _save_figure_bundle,
)


DEFAULT_OUTPUT_STEM = Path("HJEEDS/results/figures/one_d_darts_reward_function_polished")

BOARD_LIMIT = 10.0
X_LIMIT = 15.0
LOW_REWARD = 1.0
HIGH_REWARD = 2.0

# Recovered from the current example figure so this is a polished version of
# the same reward surface rather than a newly sampled surface.
REWARD_BOUNDARIES = (
    -7.03,
    -5.24,
    -4.42,
    -3.63,
    0.19,
    1.98,
    3.28,
    9.17,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument(
        "--y-tick-step",
        type=float,
        default=0.5,
        help="Spacing between y-axis tick marks.",
    )
    return parser.parse_args(argv)


def build_step_arrays() -> tuple[list[float], list[float]]:
    """Build post-step x/y arrays for the example reward surface."""

    x_values = [-X_LIMIT, -BOARD_LIMIT, *REWARD_BOUNDARIES, BOARD_LIMIT, X_LIMIT]
    y_values = [0.0]
    current_reward = LOW_REWARD
    for _ in range(len(REWARD_BOUNDARIES) + 1):
        y_values.append(current_reward)
        current_reward = HIGH_REWARD if current_reward == LOW_REWARD else LOW_REWARD
    y_values.extend([0.0, 0.0])
    return x_values, y_values


def render(output_stem: Path, dpi: int, y_tick_step: float) -> None:
    """Render and save the reward-surface figure."""

    _configure_matplotlib()

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
        }
    )

    x_values, y_values = build_step_arrays()
    figure, axis = plt.subplots(figsize=(5.45, 2.25), constrained_layout=True)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")

    axis.fill_between(
        x_values,
        y_values,
        0.0,
        step="post",
        color="#2C6E9F",
        alpha=0.11,
        linewidth=0.0,
    )
    axis.step(
        x_values,
        y_values,
        where="post",
        color="#2C6E9F",
        linewidth=2.4,
        solid_capstyle="butt",
        solid_joinstyle="miter",
        label="Reward",
    )

    axis.set_xlim(-X_LIMIT, X_LIMIT)
    axis.set_ylim(-0.05, 2.1)
    axis.set_xlabel("Action", color=TEXT_COLOR, labelpad=4)
    axis.set_ylabel("Reward", color=TEXT_COLOR, labelpad=6)
    axis.set_xticks([-15, -10, -5, 0, 5, 10, 15])
    y_ticks = []
    tick = 0.0
    while tick <= HIGH_REWARD + 1e-9:
        y_ticks.append(round(tick, 10))
        tick += y_tick_step
    axis.set_yticks(y_ticks)
    axis.tick_params(axis="both", colors=TEXT_COLOR, length=3.5, width=0.8)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.65, alpha=0.75)
    axis.set_axisbelow(True)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for spine_name in ("left", "bottom"):
        axis.spines[spine_name].set_color(CHARCOAL)
        axis.spines[spine_name].set_linewidth(0.9)

    _save_figure_bundle(figure, output_stem, dpi=dpi)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    """Render the figure from the command line."""

    args = parse_args(argv)
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    render(args.output_stem, args.dpi, args.y_tick_step)
    print(f"Wrote {args.output_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
