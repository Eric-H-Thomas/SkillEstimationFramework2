# This file was AI-generated and still requires human review. Remove this comment when done.
"""Create a multi-panel agents-per-bucket sensitivity figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from HJEEDS.plot_agents_per_bucket_robustness import (
    AGENTS_BASE_COLORS,
    CHARCOAL,
    DEFAULT_AGENT_LEVEL_CSV,
    DEFAULT_RESULTS_DIR,
    GRID_COLOR,
    NEGATIVE_TEXT_COLOR,
    TEXT_COLOR,
    ImprovementRow,
    _compact_x_limits,
    _format_axis_tick,
    _group_centers,
    _grouped_y_positions,
    _missing_bar_label,
    bar_color,
    compute_improvement_rows,
)
from HJEEDS.sensitivity_plot_common import (
    configure_matplotlib as _configure_matplotlib,
    mirrored_bar_arrays as _mirrored_bar_arrays,
    save_figure_bundle as _save_figure_bundle,
    tick_step as _tick_step,
)


DEFAULT_OUTPUT_STEM = DEFAULT_RESULTS_DIR / "agents_per_bucket_sensitivity_panels"
DEFAULT_COUNT_BUCKETS = (5, 10, 25, 100, 1000)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-level-csv", type=Path, default=DEFAULT_AGENT_LEVEL_CSV)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument(
        "--count-buckets",
        default=",".join(str(bucket) for bucket in DEFAULT_COUNT_BUCKETS),
        help="Comma-separated observation-count buckets to include as panels.",
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
    return parser.parse_args(argv)


def _parse_count_buckets(raw_value: str) -> tuple[int, ...]:
    """Parse a comma-separated count-bucket list."""

    buckets = tuple(int(piece.strip()) for piece in raw_value.split(",") if piece.strip())
    if not buckets:
        raise ValueError("At least one count bucket must be provided.")
    if any(bucket <= 0 for bucket in buckets):
        raise ValueError(f"Count buckets must be positive integers. Received: {buckets}")
    return buckets


def _panel_title(bucket: int) -> str:
    """Return the title for one panel."""

    return f"{bucket} observations/agent"


def _draw_panel(
    *,
    axis,
    rows: Sequence[ImprovementRow],
    count_bucket: int,
    hide_negative_bars: bool,
    x_limits: tuple[float, float],
) -> None:
    """Draw one mirrored agents-per-bucket panel on an existing axis."""

    import matplotlib.ticker as ticker
    from matplotlib.transforms import blended_transform_factory

    x_min, x_max = x_limits
    y_positions, group_boundaries = _grouped_y_positions(rows, group_gap=0.72)
    group_centers = _group_centers(rows, y_positions)
    colors = [bar_color(row.condition) for row in rows]
    labels = [row.condition.condition_code for row in rows]
    execution_values, decision_values, execution_xerr, decision_xerr = _mirrored_bar_arrays(rows, hide_negative_bars)

    bar_height = 0.62
    axis.barh(
        y_positions,
        execution_values,
        height=bar_height,
        color=colors,
        edgecolor="#F7F5F8",
        linewidth=0.35,
        zorder=2,
    )
    axis.barh(
        y_positions,
        decision_values,
        height=bar_height,
        color=colors,
        edgecolor="#F7F5F8",
        linewidth=0.35,
        zorder=2,
    )

    axis.errorbar(
        execution_values,
        y_positions,
        xerr=execution_xerr,
        fmt="none",
        ecolor=TEXT_COLOR,
        elinewidth=0.55,
        capsize=1.25,
        capthick=0.55,
        alpha=0.72,
        zorder=4,
    )
    axis.errorbar(
        decision_values,
        y_positions,
        xerr=decision_xerr,
        fmt="none",
        ecolor=TEXT_COLOR,
        elinewidth=0.55,
        capsize=1.25,
        capthick=0.55,
        alpha=0.72,
        zorder=4,
    )

    for boundary in group_boundaries:
        axis.axhline(boundary, color="#FFFFFF", linewidth=2.6, zorder=3)
        axis.axhline(boundary, color="#D8D3DD", linewidth=0.4, zorder=3)

    group_label_transform = blended_transform_factory(axis.transAxes, axis.transData)
    group_label_x = -0.072
    label_x = 0.006
    for group, group_y in group_centers:
        axis.text(
            group_label_x,
            group_y,
            str(group),
            ha="left",
            va="center",
            transform=group_label_transform,
            fontsize=6.0,
            color=TEXT_COLOR,
            fontweight="bold",
            clip_on=False,
            zorder=5,
        )
    for y_value, label in zip(y_positions, labels):
        axis.text(
            label_x,
            y_value,
            label,
            ha="left",
            va="center",
            transform=group_label_transform,
            fontsize=5.2,
            color=TEXT_COLOR,
            fontweight="bold",
            clip_on=False,
            zorder=5,
        )

    for y_value, row in zip(y_positions, rows):
        if hide_negative_bars and row.execution_mean < 0.0:
            axis.text(
                -0.45,
                y_value,
                _missing_bar_label(row.execution_mean, row.execution_ci_lower, row.execution_ci_upper),
                ha="right",
                va="center",
                fontsize=4.85,
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
                fontsize=4.85,
                color=NEGATIVE_TEXT_COLOR,
                fontweight="bold",
                zorder=5,
            )

    axis.axvline(0.0, color=TEXT_COLOR, linewidth=0.82, zorder=6)
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(-0.7, float(y_positions[-1]) + 0.7)
    axis.invert_yaxis()
    axis.set_yticks([])
    axis.xaxis.set_major_locator(ticker.MultipleLocator(_tick_step(x_min, x_max)))
    axis.xaxis.set_major_formatter(ticker.FuncFormatter(_format_axis_tick))
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.4, alpha=0.72, zorder=1)
    axis.tick_params(axis="x", length=2.0, colors=TEXT_COLOR, labelsize=6.1, pad=1.0)
    axis.tick_params(axis="y", length=0)

    for spine in ("left", "right", "top"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color("#AAA4B3")
    axis.spines["bottom"].set_linewidth(0.55)

    axis.set_title(_panel_title(count_bucket), fontsize=8.4, fontweight="bold", color=TEXT_COLOR, pad=17.0)
    axis.text(
        (x_min + 0.0) / 2.0,
        1.015,
        "Execution",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=6.3,
        color=TEXT_COLOR,
        fontweight="bold",
    )
    axis.text(
        x_max / 2.0,
        1.015,
        "Decision",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=6.3,
        color=TEXT_COLOR,
        fontweight="bold",
    )


def plot_panels(
    *,
    agent_level_csv: Path,
    count_buckets: Sequence[int],
    output_stem: Path,
    dpi: int,
    hide_negative_bars: bool,
) -> None:
    """Render the full multi-panel agents-per-bucket sensitivity figure."""

    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    rows_by_bucket: dict[int, list[ImprovementRow]] = {}
    for count_bucket in count_buckets:
        rows, selected_bucket = compute_improvement_rows(
            agent_level_csv=agent_level_csv,
            count_bucket=count_bucket,
        )
        rows_by_bucket[selected_bucket] = rows

    x_limit_pairs = [_compact_x_limits(rows, hide_negative_bars) for rows in rows_by_bucket.values()]
    shared_x_limits = (min(x_min for x_min, _x_max in x_limit_pairs), max(x_max for _x_min, x_max in x_limit_pairs))

    figure, axes = plt.subplots(3, 2, figsize=(13.2, 15.25))
    flat_axes = list(axes.ravel())
    for axis, count_bucket in zip(flat_axes, count_buckets):
        _draw_panel(
            axis=axis,
            rows=rows_by_bucket[count_bucket],
            count_bucket=count_bucket,
            hide_negative_bars=hide_negative_bars,
            x_limits=shared_x_limits,
        )

    for axis in flat_axes[len(count_buckets) :]:
        axis.axis("off")

    figure.suptitle(
        "Agents-per-bucket sensitivity",
        y=0.987,
        fontsize=17.0,
        fontweight="bold",
        color=TEXT_COLOR,
    )

    agent_handles = [
        Patch(facecolor=AGENTS_BASE_COLORS[value], edgecolor="none", label=f"{value} agent{'s' if value != 1 else ''}/bucket")
        for value in sorted(AGENTS_BASE_COLORS)
    ]
    legend = figure.legend(
        handles=agent_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.961),
        ncol=5,
        frameon=False,
        columnspacing=1.25,
        handlelength=1.15,
        handleheight=0.9,
        borderaxespad=0.0,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    figure.text(
        0.5,
        0.02,
        "Labels: DEF = default; MOD = moderate misspecification; STR = strong misspecification. "
        "Shade darkens with misspecification severity.",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=CHARCOAL,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.subplots_adjust(left=0.065, right=0.985, top=0.922, bottom=0.06, hspace=0.34, wspace=0.16)
    _save_figure_bundle(figure, output_stem, dpi)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""

    args = parse_args(argv)
    plot_panels(
        agent_level_csv=args.agent_level_csv,
        count_buckets=_parse_count_buckets(args.count_buckets),
        output_stem=args.output_stem,
        dpi=args.dpi,
        hide_negative_bars=not args.show_negative_bars,
    )
    print(
        f"Wrote agents-per-bucket panel figure to {args.output_stem.with_suffix('.png')} "
        f"and {args.output_stem.with_suffix('.svg')}"
    )


if __name__ == "__main__":
    main()
