# This file was AI-generated and still requires human review. Remove this comment when done.
"""Create mirrored improvement plots for the population-shape study.

The figure plots percent improvement of H-JEEDS over JEEDS at a fixed
agents-per-bucket value. Execution-skill improvement is mirrored to the left
of zero; decision-making improvement, using rationality-percentage error, is
mirrored to the right.
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
    CATEGORICAL_COLORS,
    CHARCOAL,
    GRID_COLOR,
    NEGATIVE_TEXT_COLOR,
    TEXT_COLOR,
    compact_x_limits as _compact_x_limits,
    configure_matplotlib as _configure_matplotlib,
    format_axis_tick as _format_axis_tick,
    mirrored_bar_arrays as _mirrored_bar_arrays,
    missing_bar_label_latex as _missing_bar_label,
    save_figure_bundle as _save_figure_bundle,
    seed_observation_from_agent_row as _seed_observation_from_agent_row,
    summary_fields as _summary_fields,
    summarize_seed_improvements as _summarize_seed_improvements,
    tick_step as _tick_step,
)


DEFAULT_RESULTS_DIR = Path("HJEEDS/results/hjeeds_paper_500_seeds/population_shape")
DEFAULT_AGENT_LEVEL_CSV = DEFAULT_RESULTS_DIR / "population_shape_sensitivity_agent_level_results.csv"
DEFAULT_AGENTS_PER_BUCKET = 5
DEFAULT_COUNT_BUCKETS = (5, 10, 25, 100, 1000)

SHAPE_ORDER = ("default", "uniform", "bimodal")
SHAPE_LABELS = {
    "default": "Default Gaussian",
    "uniform": "Uniform",
    "bimodal": "Bimodal",
}
SHAPE_COLORS = {
    "default": CHARCOAL,
    "uniform": CATEGORICAL_COLORS[0],
    "bimodal": CATEGORICAL_COLORS[1],
}


@dataclass(frozen=True)
class ShapeRow:
    """One plotted row of seed-aggregated improvement values."""

    shape_slug: str
    shape_label: str
    execution_mean: float
    execution_ci_lower: float
    execution_ci_upper: float
    decision_mean: float
    decision_ci_lower: float
    decision_ci_upper: float
    average_mean: float
    num_seeds: int
    num_agents_per_seed: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-level-csv", type=Path, default=DEFAULT_AGENT_LEVEL_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--agents-per-bucket", type=int, default=DEFAULT_AGENTS_PER_BUCKET)
    parser.add_argument(
        "--count-buckets",
        default=",".join(str(bucket) for bucket in DEFAULT_COUNT_BUCKETS),
        help="Comma-separated observation-count buckets to plot.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=450,
        help="Raster output resolution for PNG files.",
    )
    parser.add_argument(
        "--show-negative-bars",
        action="store_true",
        help="Draw negative-mean improvement bars instead of printing negative percentages.",
    )
    return parser.parse_args(argv)


def _parse_count_buckets(raw_value: str) -> tuple[int, ...]:
    """Parse a comma-separated count-bucket list."""

    buckets = tuple(int(piece.strip()) for piece in raw_value.split(",") if piece.strip())
    if not buckets:
        raise ValueError("At least one count bucket must be provided.")
    if any(bucket <= 0 for bucket in buckets):
        raise ValueError(f"Count buckets must be positive integers. Received: {buckets}")
    return buckets


def compute_improvement_rows(
    *,
    agent_level_csv: Path,
    agents_per_bucket: int,
    count_bucket: int,
) -> list[ShapeRow]:
    """Compute seed-level percent improvements for each plotted shape."""

    observations = []
    with agent_level_csv.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["agents_per_bucket"]) != agents_per_bucket or int(row["count_bucket"]) != count_bucket:
                continue
            if row.get("jeeds_status") != "ok" or row.get("hierarchical_status") != "ok":
                continue

            shape_slug = str(row["population_shape_slug"])
            if shape_slug not in SHAPE_ORDER:
                continue

            observation = _seed_observation_from_agent_row(row, shape_slug)
            if observation is not None:
                observations.append(observation)

    summaries = _summarize_seed_improvements(observations)
    rows: list[ShapeRow] = []
    for shape_slug in SHAPE_ORDER:
        summary = summaries.get(shape_slug)
        if summary is None:
            continue
        rows.append(
            ShapeRow(
                shape_slug=shape_slug,
                shape_label=SHAPE_LABELS[shape_slug],
                **_summary_fields(summary),
            )
        )

    return rows


def compute_overall_improvement_rows(*, agent_level_csv: Path, agents_per_bucket: int) -> list[ShapeRow]:
    """Compute percent improvements after averaging over all observation-count buckets."""

    observations = []
    with agent_level_csv.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["agents_per_bucket"]) != agents_per_bucket:
                continue
            if row.get("jeeds_status") != "ok" or row.get("hierarchical_status") != "ok":
                continue

            shape_slug = str(row["population_shape_slug"])
            if shape_slug not in SHAPE_ORDER:
                continue

            observation = _seed_observation_from_agent_row(row, shape_slug)
            if observation is not None:
                observations.append(observation)

    summaries = _summarize_seed_improvements(observations)
    rows: list[ShapeRow] = []
    for shape_slug in SHAPE_ORDER:
        summary = summaries.get(shape_slug)
        if summary is None:
            continue
        rows.append(
            ShapeRow(
                shape_slug=shape_slug,
                shape_label=SHAPE_LABELS[shape_slug],
                **_summary_fields(summary),
            )
        )

    return rows


def _write_plot_data(output_csv: Path, rows: Sequence[ShapeRow], count_bucket: int | str, agents_per_bucket: int) -> None:
    """Write plotted values so the figure can be audited."""

    fieldnames = [
        "population_shape_slug",
        "population_shape_label",
        "agents_per_bucket",
        "count_bucket",
        "num_seeds",
        "num_agents_per_seed",
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
                    "population_shape_slug": row.shape_slug,
                    "population_shape_label": row.shape_label,
                    "agents_per_bucket": agents_per_bucket,
                    "count_bucket": count_bucket,
                    "num_seeds": row.num_seeds,
                    "num_agents_per_seed": row.num_agents_per_seed,
                    "execution_improvement_mean": row.execution_mean,
                    "execution_improvement_ci_lower": row.execution_ci_lower,
                    "execution_improvement_ci_upper": row.execution_ci_upper,
                    "decision_improvement_mean": row.decision_mean,
                    "decision_improvement_ci_lower": row.decision_ci_lower,
                    "decision_improvement_ci_upper": row.decision_ci_upper,
                    "average_improvement_mean": row.average_mean,
                }
            )


def _draw_shape_axis(
    *,
    axis,
    rows: Sequence[ShapeRow],
    count_bucket: int | str,
    x_limits: tuple[float, float],
    hide_negative_bars: bool,
    panel: bool,
) -> None:
    """Draw one mirrored population-shape plot on an existing axis."""

    import matplotlib.ticker as ticker

    if not rows:
        raise ValueError("No rows available to plot.")

    x_min, x_max = x_limits
    y_positions = np.arange(len(rows), dtype=float)
    colors = [SHAPE_COLORS[row.shape_slug] for row in rows]
    execution_values, decision_values, execution_xerr, decision_xerr = _mirrored_bar_arrays(rows, hide_negative_bars)

    bar_height = 0.58 if panel else 0.64
    axis.barh(
        y_positions,
        execution_values,
        height=bar_height,
        color=colors,
        edgecolor="#F7F5F8",
        linewidth=0.5,
        zorder=2,
    )
    axis.barh(
        y_positions,
        decision_values,
        height=bar_height,
        color=colors,
        edgecolor="#F7F5F8",
        linewidth=0.5,
        zorder=2,
    )

    axis.errorbar(
        execution_values,
        y_positions,
        xerr=execution_xerr,
        fmt="none",
        ecolor=TEXT_COLOR,
        elinewidth=0.7 if not panel else 0.58,
        capsize=1.9 if not panel else 1.4,
        capthick=0.7 if not panel else 0.58,
        alpha=0.76,
        zorder=4,
    )
    axis.errorbar(
        decision_values,
        y_positions,
        xerr=decision_xerr,
        fmt="none",
        ecolor=TEXT_COLOR,
        elinewidth=0.7 if not panel else 0.58,
        capsize=1.9 if not panel else 1.4,
        capthick=0.7 if not panel else 0.58,
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
                fontsize=5.6 if not panel else 4.8,
                color=NEGATIVE_TEXT_COLOR,
                fontweight="bold",
                zorder=5,
            )
        if hide_negative_bars and row.decision_mean < 0.0:
            axis.text(
                0.45,
                y_value,
                _missing_bar_label(row.decision_mean, row.decision_ci_lower, row.decision_ci_upper),
                ha="left",
                va="center",
                fontsize=5.6 if not panel else 4.8,
                color=NEGATIVE_TEXT_COLOR,
                fontweight="bold",
                zorder=5,
            )

    axis.axvline(0.0, color=TEXT_COLOR, linewidth=0.9 if not panel else 0.78, zorder=6)
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(-0.6, float(y_positions[-1]) + 0.6)
    axis.invert_yaxis()
    axis.set_yticks(y_positions)
    axis.set_yticklabels([row.shape_label for row in rows])
    axis.xaxis.set_major_locator(ticker.MultipleLocator(_tick_step(x_min, x_max)))
    axis.xaxis.set_major_formatter(ticker.FuncFormatter(_format_axis_tick))
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.45 if not panel else 0.38, alpha=0.72, zorder=1)
    axis.tick_params(axis="x", length=2.0, colors=TEXT_COLOR, labelsize=6.8 if not panel else 5.9, pad=1.0)
    axis.tick_params(axis="y", length=0, colors=TEXT_COLOR, labelsize=7.0 if not panel else 6.0, pad=2.0)
    for tick_label in axis.get_yticklabels():
        tick_label.set_fontweight("bold")

    for spine in ("left", "right", "top"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color("#AAA4B3")
    axis.spines["bottom"].set_linewidth(0.6 if not panel else 0.5)

    if panel:
        axis.set_title(
            f"{count_bucket} observations/agent",
            fontsize=8.8,
            fontweight="bold",
            color=TEXT_COLOR,
            pad=15.0,
        )
    axis.text(
        (x_min + 0.0) / 2.0,
        1.035 if not panel else 1.03,
        "Execution",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=7.2 if not panel else 6.4,
        color=TEXT_COLOR,
        fontweight="bold",
    )
    axis.text(
        x_max / 2.0,
        1.035 if not panel else 1.03,
        "Decision",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=7.2 if not panel else 6.4,
        color=TEXT_COLOR,
        fontweight="bold",
    )


def plot_single_bucket(
    *,
    rows: Sequence[ShapeRow],
    count_bucket: int | str,
    agents_per_bucket: int,
    output_stem: Path,
    dpi: int,
    hide_negative_bars: bool,
    x_limits: tuple[float, float] | None = None,
) -> None:
    """Render one observation-count bucket plot."""

    _configure_matplotlib()
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 11.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    inferred_limits = _compact_x_limits(rows, hide_negative_bars) if x_limits is None else x_limits
    left_margin, right_margin = 0.2, 0.985
    figure, axis = plt.subplots(figsize=(7.2, 3.15))
    _draw_shape_axis(
        axis=axis,
        rows=rows,
        count_bucket=count_bucket,
        x_limits=inferred_limits,
        hide_negative_bars=hide_negative_bars,
        panel=False,
    )
    axis.set_xlabel("Percent improvement over JEEDS in absolute error", fontsize=7.4, color=CHARCOAL, labelpad=7.0)
    title = (
        "Population-shape sensitivity (all agents)"
        if count_bucket == "all"
        else f"Population-shape sensitivity ({count_bucket} observations/agent)"
    )
    figure.suptitle(
        title,
        x=(left_margin + right_margin) / 2.0,
        y=0.94,
        fontsize=11.5,
        fontweight="bold",
        color=TEXT_COLOR,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    _write_plot_data(output_stem.with_suffix(".csv"), rows, count_bucket, agents_per_bucket)
    figure.subplots_adjust(left=left_margin, right=right_margin, top=0.77, bottom=0.2)
    _save_figure_bundle(figure, output_stem, dpi)
    plt.close(figure)


def plot_panel_figure(
    *,
    rows_by_bucket: dict[int, list[ShapeRow]],
    count_buckets: Sequence[int],
    output_stem: Path,
    dpi: int,
    hide_negative_bars: bool,
) -> None:
    """Render a five-panel population-shape sensitivity figure."""

    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 8.8,
            "axes.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x_limit_pairs = [_compact_x_limits(rows_by_bucket[bucket], hide_negative_bars) for bucket in count_buckets]
    shared_x_limits = (min(x_min for x_min, _x_max in x_limit_pairs), max(x_max for _x_min, x_max in x_limit_pairs))

    figure, axes = plt.subplots(3, 2, figsize=(12.4, 10.2))
    flat_axes = list(axes.ravel())
    for axis, count_bucket in zip(flat_axes, count_buckets):
        _draw_shape_axis(
            axis=axis,
            rows=rows_by_bucket[count_bucket],
            count_bucket=count_bucket,
            x_limits=shared_x_limits,
            hide_negative_bars=hide_negative_bars,
            panel=True,
        )

    for axis in flat_axes[len(count_buckets) :]:
        axis.axis("off")

    figure.suptitle(
        "Population-shape sensitivity",
        x=0.5,
        y=0.987,
        fontsize=16.0,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    handles = [
        Patch(facecolor=SHAPE_COLORS[shape_slug], edgecolor="none", label=SHAPE_LABELS[shape_slug])
        for shape_slug in SHAPE_ORDER
    ]
    legend = figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(SHAPE_ORDER),
        frameon=False,
        columnspacing=1.3,
        handlelength=1.15,
        handleheight=0.9,
        borderaxespad=0.0,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    figure.text(
        0.5,
        0.022,
        "All panels use 5 agents per observation-count bucket. Values are percent improvement over JEEDS in absolute error.",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=CHARCOAL,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.subplots_adjust(left=0.11, right=0.985, top=0.885, bottom=0.065, hspace=0.68, wspace=0.38)
    _save_figure_bundle(figure, output_stem, dpi)
    plt.close(figure)


def render_all(
    *,
    agent_level_csv: Path,
    output_dir: Path,
    agents_per_bucket: int,
    count_buckets: Sequence[int],
    dpi: int,
    hide_negative_bars: bool,
) -> None:
    """Render all population-shape plots requested for the paper."""

    rows_by_bucket = {
        count_bucket: compute_improvement_rows(
            agent_level_csv=agent_level_csv,
            agents_per_bucket=agents_per_bucket,
            count_bucket=count_bucket,
        )
        for count_bucket in count_buckets
    }

    for count_bucket in count_buckets:
        output_stem = output_dir / f"bucket_{count_bucket:03d}_population_shape_improvement_bars"
        plot_single_bucket(
            rows=rows_by_bucket[count_bucket],
            count_bucket=count_bucket,
            agents_per_bucket=agents_per_bucket,
            output_stem=output_stem,
            dpi=dpi,
            hide_negative_bars=hide_negative_bars,
        )

    panel_stem = output_dir / "population_shape_sensitivity_panels"
    plot_panel_figure(
        rows_by_bucket=rows_by_bucket,
        count_buckets=count_buckets,
        output_stem=panel_stem,
        dpi=dpi,
        hide_negative_bars=hide_negative_bars,
    )

    overall_rows = compute_overall_improvement_rows(
        agent_level_csv=agent_level_csv,
        agents_per_bucket=agents_per_bucket,
    )
    plot_single_bucket(
        rows=overall_rows,
        count_bucket="all",
        agents_per_bucket=agents_per_bucket,
        output_stem=output_dir / "population_shape_all_agents_improvement_bars",
        dpi=dpi,
        hide_negative_bars=hide_negative_bars,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""

    args = parse_args(argv)
    count_buckets = _parse_count_buckets(args.count_buckets)
    render_all(
        agent_level_csv=args.agent_level_csv,
        output_dir=args.output_dir,
        agents_per_bucket=args.agents_per_bucket,
        count_buckets=count_buckets,
        dpi=args.dpi,
        hide_negative_bars=not args.show_negative_bars,
    )
    print(f"Wrote population-shape plots to {args.output_dir}")


if __name__ == "__main__":
    main()
