# This file was AI-generated and still requires human review. Remove this comment when done.
"""Create a mirrored improvement plot for the high-data-anchor study.

The figure plots percent improvement of H-JEEDS over JEEDS for the low-data
agents in the anchor-availability experiment. Execution-skill improvement is
mirrored to the left of zero; decision-making improvement, using
rationality-percentage error, is mirrored to the right.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from HJEEDS.sensitivity_plot_common import (
    CHARCOAL,
    GRID_COLOR,
    NEGATIVE_TEXT_COLOR,
    NUMERIC_6_COLORS,
    TEXT_COLOR,
    compact_x_limits as _compact_x_limits,
    configure_matplotlib as _configure_matplotlib,
    format_axis_tick as _format_axis_tick,
    mirrored_bar_arrays as _mirrored_bar_arrays,
    missing_bar_label_latex as _missing_bar_label,
    save_figure_bundle as _save_figure_bundle,
    seed_observation_from_agent_row as _seed_observation_from_agent_row,
    summarize_seed_improvements as _summarize_seed_improvements,
)


DEFAULT_RESULTS_DIR = Path("HJEEDS/results/hjeeds_paper_500_seeds/anchor_availability")
DEFAULT_AGENT_LEVEL_CSV = DEFAULT_RESULTS_DIR / "anchor_availability_sensitivity_agent_level_results.csv"
DEFAULT_OUTPUT_STEM = DEFAULT_RESULTS_DIR / "anchor_availability_low_data_improvement_bars"
DEFAULT_LOW_DATA_OBSERVATIONS = 1

ANCHOR_COUNTS = (0, 1, 2, 5, 10, 25)
ANCHOR_COLORS = {
    0: NUMERIC_6_COLORS[0],
    1: NUMERIC_6_COLORS[1],
    2: NUMERIC_6_COLORS[2],
    5: NUMERIC_6_COLORS[3],
    10: NUMERIC_6_COLORS[4],
    25: NUMERIC_6_COLORS[5],
}


@dataclass(frozen=True)
class AnchorRow:
    """One plotted row of seed-aggregated improvement values."""

    anchor_count: int
    anchor_label: str
    execution_mean: float
    execution_ci_lower: float
    execution_ci_upper: float
    decision_mean: float
    decision_ci_lower: float
    decision_ci_upper: float
    average_mean: float
    num_seeds: int
    num_low_data_agents_per_seed: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-level-csv", type=Path, default=DEFAULT_AGENT_LEVEL_CSV)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument(
        "--low-data-observations",
        type=int,
        default=DEFAULT_LOW_DATA_OBSERVATIONS,
        help="Observation-count bucket to evaluate as the low-data agents.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=450,
        help="Raster output resolution for the PNG file.",
    )
    parser.add_argument(
        "--show-negative-bars",
        action="store_true",
        help="Draw negative-mean improvement bars instead of printing negative percentages.",
    )
    parser.add_argument(
        "--x-limits",
        type=float,
        nargs=2,
        metavar=("X_MIN", "X_MAX"),
        default=None,
        help="Optional x-axis limits. Defaults to compact asymmetric limits based on plotted intervals.",
    )
    return parser.parse_args(argv)


def _anchor_label(anchor_count: int) -> str:
    """Return a compact row label for one anchor count."""

    noun = "anchor" if anchor_count == 1 else "anchors"
    return f"{anchor_count} {noun}"


def anchor_color(anchor_count: int) -> str:
    """Return the plotted color for one anchor-count condition."""

    return ANCHOR_COLORS.get(anchor_count, "#B8B8B8")


def compute_improvement_rows(
    *,
    agent_level_csv: Path,
    low_data_observations: int,
) -> list[AnchorRow]:
    """Compute seed-level percent improvements for the low-data agents."""

    observations = []
    with agent_level_csv.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["count_bucket"]) != low_data_observations:
                continue
            if row.get("jeeds_status") != "ok" or row.get("hierarchical_status") != "ok":
                continue

            anchor_count = int(row["anchor_agent_count"])
            if anchor_count not in ANCHOR_COUNTS:
                continue

            observation = _seed_observation_from_agent_row(row, anchor_count)
            if observation is not None:
                observations.append(observation)

    summaries = _summarize_seed_improvements(observations)
    rows: list[AnchorRow] = []
    for anchor_count in ANCHOR_COUNTS:
        summary = summaries.get(anchor_count)
        if summary is None:
            continue
        rows.append(
            AnchorRow(
                anchor_count=anchor_count,
                anchor_label=_anchor_label(anchor_count),
                execution_mean=summary.execution_mean,
                execution_ci_lower=summary.execution_ci_lower,
                execution_ci_upper=summary.execution_ci_upper,
                decision_mean=summary.decision_mean,
                decision_ci_lower=summary.decision_ci_lower,
                decision_ci_upper=summary.decision_ci_upper,
                average_mean=summary.average_mean,
                num_seeds=summary.num_seeds,
                num_low_data_agents_per_seed=summary.num_agents_per_seed,
            )
        )

    return rows


def write_plot_data(output_csv: Path, rows: Sequence[AnchorRow], low_data_observations: int) -> None:
    """Write plotted values so the figure can be audited."""

    fieldnames = [
        "anchor_agent_count",
        "anchor_label",
        "low_data_observations",
        "num_seeds",
        "num_low_data_agents_per_seed",
        "execution_improvement_mean",
        "execution_improvement_ci_lower",
        "execution_improvement_ci_upper",
        "decision_improvement_mean",
        "decision_improvement_ci_lower",
        "decision_improvement_ci_upper",
        "average_improvement_mean",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "anchor_agent_count": row.anchor_count,
                    "anchor_label": row.anchor_label,
                    "low_data_observations": low_data_observations,
                    "num_seeds": row.num_seeds,
                    "num_low_data_agents_per_seed": row.num_low_data_agents_per_seed,
                    "execution_improvement_mean": row.execution_mean,
                    "execution_improvement_ci_lower": row.execution_ci_lower,
                    "execution_improvement_ci_upper": row.execution_ci_upper,
                    "decision_improvement_mean": row.decision_mean,
                    "decision_improvement_ci_lower": row.decision_ci_lower,
                    "decision_improvement_ci_upper": row.decision_ci_upper,
                    "average_improvement_mean": row.average_mean,
                }
            )


def plot_anchor_availability(
    *,
    rows: Sequence[AnchorRow],
    output_stem: Path,
    low_data_observations: int,
    dpi: int,
    hide_negative_bars: bool,
    x_limits: tuple[float, float] | None,
) -> None:
    """Render the mirrored high-data-anchor availability plot."""

    _configure_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    if not rows:
        raise ValueError("No rows available to plot.")

    if x_limits is not None:
        x_min, x_max = x_limits
        if x_min >= 0.0 or x_max <= 0.0 or x_min >= x_max:
            raise ValueError(f"x-axis limits must bracket zero in increasing order, got {x_limits}")
    else:
        x_min, x_max = _compact_x_limits(rows, hide_negative_bars)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.titlesize": 10.5,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    y_positions = np.arange(len(rows), dtype=float)
    colors = [anchor_color(row.anchor_count) for row in rows]
    execution_values, decision_values, execution_xerr, decision_xerr = _mirrored_bar_arrays(rows, hide_negative_bars)

    figure, axis = plt.subplots(figsize=(7.25, 4.0))
    bar_height = 0.64
    axis.barh(
        y_positions,
        execution_values,
        height=bar_height,
        color=colors,
        edgecolor="#F7F5F8",
        linewidth=0.45,
        zorder=2,
    )
    axis.barh(
        y_positions,
        decision_values,
        height=bar_height,
        color=colors,
        edgecolor="#F7F5F8",
        linewidth=0.45,
        zorder=2,
    )

    axis.errorbar(
        execution_values,
        y_positions,
        xerr=execution_xerr,
        fmt="none",
        ecolor=TEXT_COLOR,
        elinewidth=0.68,
        capsize=1.8,
        capthick=0.68,
        alpha=0.76,
        zorder=4,
    )
    axis.errorbar(
        decision_values,
        y_positions,
        xerr=decision_xerr,
        fmt="none",
        ecolor=TEXT_COLOR,
        elinewidth=0.68,
        capsize=1.8,
        capthick=0.68,
        alpha=0.76,
        zorder=4,
    )

    for y_value, row in zip(y_positions, rows):
        if hide_negative_bars and row.execution_mean < 0.0:
            axis.text(
                -0.45,
                y_value,
                _missing_bar_label(row.execution_mean, row.execution_ci_lower, row.execution_ci_upper),
                ha="right",
                va="center",
                fontsize=5.6,
                color=NEGATIVE_TEXT_COLOR,
                fontweight="bold",
                zorder=5,
            )
        if hide_negative_bars and row.decision_mean < 0.0:
            axis.text(
                0.65,
                y_value,
                _missing_bar_label(row.decision_mean, row.decision_ci_lower, row.decision_ci_upper),
                ha="left",
                va="center",
                fontsize=5.6,
                color=NEGATIVE_TEXT_COLOR,
                fontweight="bold",
                zorder=5,
            )

    axis.axvline(0.0, color=TEXT_COLOR, linewidth=0.9, zorder=6)
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(-0.7, float(y_positions[-1]) + 0.7)
    axis.invert_yaxis()
    axis.set_yticks(y_positions)
    axis.set_yticklabels([row.anchor_label for row in rows])
    tick_step = 5 if (x_max - x_min) <= 50 else 10
    axis.xaxis.set_major_locator(ticker.MultipleLocator(tick_step))
    axis.xaxis.set_major_formatter(ticker.FuncFormatter(_format_axis_tick))
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.45, alpha=0.75, zorder=1)
    axis.tick_params(axis="x", length=2.5, colors=TEXT_COLOR)
    axis.tick_params(axis="y", length=0, colors=TEXT_COLOR, pad=4.0)
    for tick_label in axis.get_yticklabels():
        tick_label.set_fontweight("bold")
        tick_label.set_color(TEXT_COLOR)

    for spine in ("left", "right", "top"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color("#AAA4B3")
    axis.spines["bottom"].set_linewidth(0.6)

    axis.set_xlabel("Percent improvement over JEEDS in absolute error", color=CHARCOAL, labelpad=7.0)
    figure.suptitle(
        "High-data-anchor availability",
        y=0.972,
        fontsize=13.0,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    axis.text(
        (x_min + 0.0) / 2.0,
        1.022,
        "Execution",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=7.6,
        color=TEXT_COLOR,
        fontweight="bold",
    )
    axis.text(
        x_max / 2.0,
        1.022,
        "Decision",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=7.6,
        color=TEXT_COLOR,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.03,
        f"Evaluated on 25 low-data agents with {low_data_observations} observation each; anchors have 25 observations each.",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=CHARCOAL,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    write_plot_data(output_stem.with_suffix(".csv"), rows, low_data_observations)
    figure.subplots_adjust(left=0.16, right=0.985, top=0.82, bottom=0.19)
    _save_figure_bundle(figure, output_stem, dpi)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""

    args = parse_args(argv)
    rows = compute_improvement_rows(
        agent_level_csv=args.agent_level_csv,
        low_data_observations=args.low_data_observations,
    )
    plot_anchor_availability(
        rows=rows,
        output_stem=args.output_stem,
        low_data_observations=args.low_data_observations,
        dpi=args.dpi,
        hide_negative_bars=not args.show_negative_bars,
        x_limits=tuple(args.x_limits) if args.x_limits is not None else None,
    )
    print(
        f"Wrote {len(rows)} anchor-availability settings to {args.output_stem.with_suffix('.png')} "
        f"and {args.output_stem.with_suffix('.svg')}"
    )


if __name__ == "__main__":
    main()
