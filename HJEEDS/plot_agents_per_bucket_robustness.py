# This file was AI-generated and still requires human review. Remove this comment when done.
"""Create mirrored improvement plots for the agents-per-bucket study.

The figure plots percent improvement of H-JEEDS over JEEDS for one
observation-count bucket. Execution-skill improvement is mirrored to the
left of zero; decision-making improvement, using rationality-percentage
error, is mirrored to the right.
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
    NUMERIC_5_COLORS,
    TEXT_COLOR,
    blend as _blend,
    compact_x_limits as _compact_x_limits,
    configure_matplotlib as _configure_matplotlib,
    format_axis_tick as _format_axis_tick,
    group_centers as _shared_group_centers,
    grouped_y_positions as _shared_grouped_y_positions,
    mirrored_bar_arrays as _mirrored_bar_arrays,
    missing_bar_label_latex as _missing_bar_label,
    readable_label_color,
    save_figure_bundle as _save_figure_bundle,
    seed_observation_from_agent_row as _seed_observation_from_agent_row,
    summary_fields as _summary_fields,
    summarize_seed_improvements as _summarize_seed_improvements,
)


DEFAULT_RESULTS_DIR = Path("HJEEDS/results/hjeeds_paper_500_seeds/agents_per_bucket")
DEFAULT_AGENT_LEVEL_CSV = DEFAULT_RESULTS_DIR / "agents_per_bucket_sensitivity_agent_level_results.csv"
DEFAULT_OUTPUT_STEM = DEFAULT_RESULTS_DIR / "agents_per_bucket_lowest_bucket_improvement_bars"

AGENTS_BASE_COLORS = {
    1: NUMERIC_5_COLORS[0],
    2: NUMERIC_5_COLORS[1],
    5: NUMERIC_5_COLORS[2],
    10: NUMERIC_5_COLORS[3],
    25: NUMERIC_5_COLORS[4],
}

CONDITION_ORDER = {
    "default": 0,
    "moderate_combined_misspecification": 1,
    "strong_combined_misspecification": 2,
}

CONDITION_LABELS = {
    "default": "Default",
    "moderate_combined_misspecification": "Moderate misspec.",
    "strong_combined_misspecification": "Strong misspec.",
}

CONDITION_CODES = {
    "default": "DEF",
    "moderate_combined_misspecification": "MOD",
    "strong_combined_misspecification": "STR",
}

@dataclass(frozen=True)
class AgentsCondition:
    """Metadata for one plotted agents-per-bucket condition."""

    agents_per_bucket: int
    condition_slug: str
    condition_label: str
    condition_code: str


@dataclass(frozen=True)
class ImprovementRow:
    """One plotted row of seed-aggregated improvement values."""

    condition: AgentsCondition
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
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument(
        "--count-bucket",
        type=int,
        default=None,
        help="Observation-count bucket to plot. Defaults to the minimum bucket in the CSV.",
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
        "--label-placement",
        choices=("center", "left"),
        default="left",
        help="Place condition labels near the center line or in a left-side label column.",
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


def _selected_bucket(agent_level_csv: Path, requested_bucket: int | None) -> int:
    """Return the selected observation-count bucket."""

    if requested_bucket is not None:
        return requested_bucket

    buckets: set[int] = set()
    with agent_level_csv.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            buckets.add(int(row["count_bucket"]))
    if not buckets:
        raise ValueError(f"No count buckets found in {agent_level_csv}.")
    return min(buckets)


def compute_improvement_rows(
    *,
    agent_level_csv: Path,
    count_bucket: int | None,
) -> tuple[list[ImprovementRow], int]:
    """Compute seed-level percent improvements for each plotted condition."""

    selected_bucket = _selected_bucket(agent_level_csv, count_bucket)

    observations = []
    metadata: dict[tuple[int, str], AgentsCondition] = {}
    with agent_level_csv.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["count_bucket"]) != selected_bucket:
                continue
            if row.get("jeeds_status") != "ok" or row.get("hierarchical_status") != "ok":
                continue

            condition_slug = str(row["condition_slug"])
            if condition_slug not in CONDITION_ORDER:
                continue

            agents_per_bucket = int(row["agents_per_bucket"])
            condition_key = (agents_per_bucket, condition_slug)
            observation = _seed_observation_from_agent_row(row, condition_key)
            if observation is None:
                continue

            metadata[condition_key] = AgentsCondition(
                agents_per_bucket=agents_per_bucket,
                condition_slug=condition_slug,
                condition_label=CONDITION_LABELS[condition_slug],
                condition_code=CONDITION_CODES[condition_slug],
            )
            observations.append(observation)

    rows: list[ImprovementRow] = []
    for key, summary in _summarize_seed_improvements(observations).items():
        rows.append(
            ImprovementRow(
                condition=metadata[key],
                **_summary_fields(summary),
            )
        )

    rows.sort(
        key=lambda row: (
            row.condition.agents_per_bucket,
            CONDITION_ORDER[row.condition.condition_slug],
        )
    )
    return rows, selected_bucket


def compute_overall_improvement_rows(*, agent_level_csv: Path) -> list[ImprovementRow]:
    """Compute percent improvements after averaging over all observation-count buckets."""

    observations = []
    metadata: dict[tuple[int, str], AgentsCondition] = {}
    with agent_level_csv.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("jeeds_status") != "ok" or row.get("hierarchical_status") != "ok":
                continue

            condition_slug = str(row["condition_slug"])
            if condition_slug not in CONDITION_ORDER:
                continue

            agents_per_bucket = int(row["agents_per_bucket"])
            condition_key = (agents_per_bucket, condition_slug)
            observation = _seed_observation_from_agent_row(row, condition_key)
            if observation is None:
                continue

            metadata[condition_key] = AgentsCondition(
                agents_per_bucket=agents_per_bucket,
                condition_slug=condition_slug,
                condition_label=CONDITION_LABELS[condition_slug],
                condition_code=CONDITION_CODES[condition_slug],
            )
            observations.append(observation)

    rows: list[ImprovementRow] = []
    for key, summary in _summarize_seed_improvements(observations).items():
        rows.append(
            ImprovementRow(
                condition=metadata[key],
                **_summary_fields(summary),
            )
        )

    rows.sort(
        key=lambda row: (
            row.condition.agents_per_bucket,
            CONDITION_ORDER[row.condition.condition_slug],
        )
    )
    return rows


def bar_color(condition: AgentsCondition) -> str:
    """Return the row color with agents-per-bucket hue and condition shade."""

    base = AGENTS_BASE_COLORS.get(condition.agents_per_bucket, "#B8B8B8")
    if condition.condition_slug == "default":
        return _blend(base, "#FFFFFF", 0.42)
    if condition.condition_slug == "strong_combined_misspecification":
        return _blend(base, TEXT_COLOR, 0.28)
    return base


def label_color(fill_color: str) -> str:
    """Return a readable condition-label color for one bar fill."""

    return readable_label_color(fill_color, threshold=0.28)


def _grouped_y_positions(rows: Sequence[ImprovementRow], group_gap: float = 0.8) -> tuple[np.ndarray, list[float]]:
    """Return y positions with extra space between agents-per-bucket groups."""

    return _shared_grouped_y_positions([row.condition.agents_per_bucket for row in rows], group_gap)


def _group_centers(rows: Sequence[ImprovementRow], y_positions: Sequence[float]) -> list[tuple[int, float]]:
    """Return y centers for each agents-per-bucket group."""

    centers = _shared_group_centers([row.condition.agents_per_bucket for row in rows], y_positions, sort=True)
    return [(int(group), center) for group, center in centers]


def _bucket_title_phrase(bucket: int) -> str:
    """Return a compact figure-title phrase for one observation-count bucket."""

    if bucket == 5:
        return "the lowest-data bucket"
    return f"the {bucket}-observation bucket"


def _write_plot_data(output_csv: Path, rows: Sequence[ImprovementRow], count_bucket: int) -> None:
    """Write plotted values so the figure can be audited."""

    fieldnames = [
        "agents_per_bucket",
        "condition_slug",
        "condition_label",
        "condition_code",
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
                    "agents_per_bucket": row.condition.agents_per_bucket,
                    "condition_slug": row.condition.condition_slug,
                    "condition_label": row.condition.condition_label,
                    "condition_code": row.condition.condition_code,
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


def plot_improvement_bars(
    *,
    rows: Sequence[ImprovementRow],
    count_bucket: int,
    output_stem: Path,
    dpi: int,
    hide_negative_bars: bool,
    label_placement: str,
    x_limits: tuple[float, float] | None,
) -> None:
    """Render the mirrored agents-per-bucket improvement plot."""

    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    from matplotlib.patches import Patch
    from matplotlib.transforms import blended_transform_factory

    if not rows:
        raise ValueError("No rows available to plot.")

    if x_limits is not None:
        x_min, x_max = x_limits
        if x_min >= 0.0 or x_max <= 0.0 or x_min >= x_max:
            raise ValueError(f"x-axis limits must bracket zero in increasing order, got {x_limits}")
    else:
        x_min, x_max = _compact_x_limits(rows, hide_negative_bars)

    y_positions, group_boundaries = _grouped_y_positions(rows)
    group_centers = _group_centers(rows, y_positions)
    colors = [bar_color(row.condition) for row in rows]
    labels = [row.condition.condition_code for row in rows]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 9.0,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 6.4,
            "legend.title_fontsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure_height = max(5.65, 0.19 * len(rows) + 2.95)
    figure, axis = plt.subplots(figsize=(7.35, figure_height))
    bar_height = 0.66
    execution_values, decision_values, execution_xerr, decision_xerr = _mirrored_bar_arrays(rows, hide_negative_bars)

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
        elinewidth=0.62,
        capsize=1.6,
        capthick=0.62,
        alpha=0.74,
        zorder=4,
    )
    axis.errorbar(
        decision_values,
        y_positions,
        xerr=decision_xerr,
        fmt="none",
        ecolor=TEXT_COLOR,
        elinewidth=0.62,
        capsize=1.6,
        capthick=0.62,
        alpha=0.74,
        zorder=4,
    )

    for boundary in group_boundaries:
        axis.axhline(boundary, color="#FFFFFF", linewidth=3.0, zorder=3)
        axis.axhline(boundary, color="#D8D3DD", linewidth=0.45, zorder=3)

    if label_placement == "left":
        group_label_transform = blended_transform_factory(axis.transAxes, axis.transData)
        group_label_x = -0.072
        label_x = -0.012
        for group, group_y in group_centers:
            axis.text(
                group_label_x,
                group_y,
                str(group),
                ha="center",
                va="center",
                transform=group_label_transform,
                fontsize=6.4,
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
                ha="right",
                va="center",
                transform=group_label_transform,
                fontsize=5.7,
                color=TEXT_COLOR,
                fontweight="bold",
                clip_on=False,
                zorder=5,
            )

    for y_value, row, label, color in zip(y_positions, rows, labels, colors):
        if hide_negative_bars and row.execution_mean < 0.0:
            axis.text(
                -0.45,
                y_value,
                _missing_bar_label(row.execution_mean, row.execution_ci_lower, row.execution_ci_upper),
                ha="right",
                va="center",
                fontsize=5.1,
                color=NEGATIVE_TEXT_COLOR,
                fontweight="bold",
                zorder=5,
            )
        elif label_placement == "center":
            axis.text(
                -1.25,
                y_value,
                label,
                ha="right",
                va="center",
                fontsize=5.1,
                color=label_color(color),
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
                fontsize=5.1,
                color=NEGATIVE_TEXT_COLOR,
                fontweight="bold",
                zorder=5,
            )
        elif label_placement == "center":
            axis.text(
                1.25,
                y_value,
                label,
                ha="left",
                va="center",
                fontsize=5.1,
                color=label_color(color),
                fontweight="bold",
                zorder=5,
            )

    axis.axvline(0.0, color=TEXT_COLOR, linewidth=0.9, zorder=6)
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(-0.7, float(y_positions[-1]) + 0.7)
    axis.invert_yaxis()
    axis.set_yticks([])
    tick_step = 5 if (x_max - x_min) <= 50 else 10
    axis.xaxis.set_major_locator(ticker.MultipleLocator(tick_step))
    axis.xaxis.set_major_formatter(ticker.FuncFormatter(_format_axis_tick))
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.45, alpha=0.75, zorder=1)
    axis.tick_params(axis="x", length=2.5, colors=TEXT_COLOR)
    axis.tick_params(axis="y", length=0)

    for spine in ("left", "right", "top"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color("#AAA4B3")
    axis.spines["bottom"].set_linewidth(0.6)

    axis.set_xlabel("Percent improvement over JEEDS in absolute error")
    figure.suptitle(
        f"Agents-per-bucket sensitivity ({count_bucket} observations/agent)",
        y=0.982,
        fontsize=9.2,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    axis.text(
        (x_min + 0.0) / 2.0,
        1.01,
        "Execution skill",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=7.7,
        color=TEXT_COLOR,
        fontweight="bold",
    )
    axis.text(
        x_max / 2.0,
        1.01,
        "Decision-making skill",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=7.7,
        color=TEXT_COLOR,
        fontweight="bold",
    )

    agent_handles = [
        Patch(facecolor=AGENTS_BASE_COLORS[value], edgecolor="none", label=f"{value} agent{'s' if value != 1 else ''}/bucket")
        for value in sorted(AGENTS_BASE_COLORS)
    ]
    legend = figure.legend(
        handles=agent_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=5,
        frameon=False,
        columnspacing=1.05,
        handlelength=1.1,
        handleheight=0.9,
        borderaxespad=0.0,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    figure.text(
        0.5,
        0.03,
        "Labels: DEF = default; MOD = moderate misspec.; STR = strong misspec. "
        "Shade darkens with misspecification severity.",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=CHARCOAL,
    )
    figure.text(
        0.5,
        0.012,
        "Note: each bucket plot uses its own x-axis scale.",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=CHARCOAL,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.subplots_adjust(left=0.12, right=0.985, top=0.86, bottom=0.125)
    _save_figure_bundle(figure, output_stem, dpi)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""

    args = parse_args(argv)
    rows, count_bucket = compute_improvement_rows(
        agent_level_csv=args.agent_level_csv,
        count_bucket=args.count_bucket,
    )
    _write_plot_data(args.output_stem.with_suffix(".csv"), rows, count_bucket)
    plot_improvement_bars(
        rows=rows,
        count_bucket=count_bucket,
        output_stem=args.output_stem,
        dpi=args.dpi,
        hide_negative_bars=not args.show_negative_bars,
        label_placement=args.label_placement,
        x_limits=tuple(args.x_limits) if args.x_limits is not None else None,
    )
    print(
        f"Wrote {len(rows)} settings to {args.output_stem.with_suffix('.png')} "
        f"and {args.output_stem.with_suffix('.svg')}"
    )


if __name__ == "__main__":
    main()
