# This file was AI-generated and still requires human review. Remove this comment when done.
"""Create mirrored improvement plots for remaining H-JEEDS sensitivity studies."""

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
    NUMERIC_3_COLORS,
    NUMERIC_5_COLORS,
    TEXT_COLOR,
    compact_x_limits as _shared_compact_x_limits,
    configure_matplotlib as _configure_matplotlib,
    format_axis_tick as _format_axis_tick,
    grouped_y_positions as _shared_grouped_y_positions,
    labeled_group_centers as _shared_labeled_group_centers,
    mirrored_bar_arrays as _mirrored_bar_arrays,
    missing_bar_label_plain as _missing_bar_label,
    save_figure_bundle as _save_figure_bundle,
    seed_observation_from_agent_row as _seed_observation_from_agent_row,
    summary_fields as _summary_fields,
    summarize_seed_improvements as _summarize_seed_improvements,
)


BASE_RESULTS_DIR = Path("HJEEDS/results/hjeeds_paper_500_seeds")
LOWEST_COUNT_BUCKET = 5
AGENTS_PER_BUCKET_ORDER = (1, 2, 5, 10, 25)


@dataclass(frozen=True)
class PlotConfig:
    """Configuration for one sensitivity plot."""

    experiment_slug: str
    title: str
    agent_level_csv: Path
    output_stem: Path
    factor_slug_column: str
    factor_label_column: str
    factor_order: tuple[str, ...]
    factor_labels: dict[str, str]
    factor_colors: dict[str, str]
    crossed_with_agents_per_bucket: bool


@dataclass(frozen=True)
class ImprovementRow:
    """One plotted row of seed-aggregated improvement values."""

    factor_slug: str
    factor_label: str
    subgroup_label: str
    execution_mean: float
    execution_ci_lower: float
    execution_ci_upper: float
    decision_mean: float
    decision_ci_lower: float
    decision_ci_upper: float
    average_mean: float
    num_seeds: int
    num_agents_per_seed: int


PLOT_CONFIGS = (
    PlotConfig(
        experiment_slug="decision_model",
        title="Decision-model misspecification",
        agent_level_csv=BASE_RESULTS_DIR / "decision_model" / "decision_model_sensitivity_agent_level_results.csv",
        output_stem=BASE_RESULTS_DIR / "decision_model" / "decision_model_lowest_bucket_improvement_bars",
        factor_slug_column="decision_model_slug",
        factor_label_column="decision_model_label",
        factor_order=("softmax", "flip", "rational", "deceptive"),
        factor_labels={
            "softmax": "Softmax",
            "flip": "Flip",
            "rational": "Rational",
            "deceptive": "Deceptive",
        },
        factor_colors={
            "softmax": CHARCOAL,
            "flip": CATEGORICAL_COLORS[0],
            "rational": CATEGORICAL_COLORS[1],
            "deceptive": CATEGORICAL_COLORS[2],
        },
        crossed_with_agents_per_bucket=True,
    ),
    PlotConfig(
        experiment_slug="true_correlation",
        title="True population correlation",
        agent_level_csv=BASE_RESULTS_DIR / "true_correlation" / "true_correlation_sensitivity_agent_level_results.csv",
        output_stem=BASE_RESULTS_DIR / "true_correlation" / "true_correlation_lowest_bucket_improvement_bars",
        factor_slug_column="true_correlation_slug",
        factor_label_column="true_correlation_label",
        factor_order=("r_neg_0_9", "r_neg_0_5", "r_0_0", "r_pos_0_5", "r_pos_0_9"),
        factor_labels={
            "r_neg_0_9": "r = -0.9",
            "r_neg_0_5": "r = -0.5",
            "r_0_0": "r = 0.0",
            "r_pos_0_5": "r = +0.5",
            "r_pos_0_9": "r = +0.9",
        },
        factor_colors={
            "r_neg_0_9": NUMERIC_5_COLORS[0],
            "r_neg_0_5": NUMERIC_5_COLORS[1],
            "r_0_0": NUMERIC_5_COLORS[2],
            "r_pos_0_5": NUMERIC_5_COLORS[3],
            "r_pos_0_9": NUMERIC_5_COLORS[4],
        },
        crossed_with_agents_per_bucket=True,
    ),
    PlotConfig(
        experiment_slug="grid_resolution",
        title="Grid resolution sensitivity",
        agent_level_csv=BASE_RESULTS_DIR / "grid_resolution" / "grid_resolution_sensitivity_agent_level_results.csv",
        output_stem=BASE_RESULTS_DIR / "grid_resolution" / "grid_resolution_lowest_bucket_improvement_bars",
        factor_slug_column="grid_resolution_slug",
        factor_label_column="grid_resolution_label",
        factor_order=("grid_011x011", "grid_021x021", "grid_041x041"),
        factor_labels={
            "grid_011x011": "11 x 11",
            "grid_021x021": "21 x 21",
            "grid_041x041": "41 x 41",
        },
        factor_colors={
            "grid_011x011": NUMERIC_3_COLORS[0],
            "grid_021x021": NUMERIC_3_COLORS[1],
            "grid_041x041": NUMERIC_3_COLORS[2],
        },
        crossed_with_agents_per_bucket=False,
    ),
    PlotConfig(
        experiment_slug="compound_stress",
        title="Compound stress test",
        agent_level_csv=BASE_RESULTS_DIR / "compound_stress" / "compound_stress_sensitivity_agent_level_results.csv",
        output_stem=BASE_RESULTS_DIR / "compound_stress" / "compound_stress_lowest_bucket_improvement_bars",
        factor_slug_column="compound_stress_slug",
        factor_label_column="compound_stress_label",
        factor_order=("default", "moderate_compound_stress", "strong_compound_stress"),
        factor_labels={
            "default": "Default",
            "moderate_compound_stress": "Moderate stress",
            "strong_compound_stress": "Strong stress",
        },
        factor_colors={
            "default": CHARCOAL,
            "moderate_compound_stress": CATEGORICAL_COLORS[0],
            "strong_compound_stress": CATEGORICAL_COLORS[1],
        },
        crossed_with_agents_per_bucket=True,
    ),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        default=",".join(config.experiment_slug for config in PLOT_CONFIGS),
        help="Comma-separated subset of experiments to render.",
    )
    parser.add_argument(
        "--count-bucket",
        type=int,
        default=LOWEST_COUNT_BUCKET,
        help="Observation-count bucket to plot.",
    )
    parser.add_argument(
        "--agents-per-bucket",
        type=int,
        choices=AGENTS_PER_BUCKET_ORDER,
        help="For crossed studies, plot only this agents-per-bucket condition.",
    )
    parser.add_argument(
        "--average-all-buckets",
        action="store_true",
        help="Average over every observation-count bucket instead of selecting one bucket.",
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


def _grouped_y_positions(rows: Sequence[ImprovementRow], group_gap: float) -> tuple[np.ndarray, list[float]]:
    """Return y positions with extra space between condition groups."""

    return _shared_grouped_y_positions([row.factor_slug for row in rows], group_gap)


def _group_centers(rows: Sequence[ImprovementRow], y_positions: Sequence[float]) -> list[tuple[str, str, float]]:
    """Return y centers for each condition group."""

    return [
        (str(group), label, center)
        for group, label, center in _shared_labeled_group_centers(
            [row.factor_slug for row in rows],
            [row.factor_label for row in rows],
            y_positions,
        )
    ]


def compute_improvement_rows(
    config: PlotConfig,
    count_bucket: int | None,
    agents_per_bucket: int | None = None,
) -> list[ImprovementRow]:
    """Compute seed-level percent improvements for one experiment."""

    observations = []
    factor_labels_from_csv: dict[str, str] = {}

    with config.agent_level_csv.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            if count_bucket is not None and int(row["count_bucket"]) != count_bucket:
                continue
            if row.get("jeeds_status") != "ok" or row.get("hierarchical_status") != "ok":
                continue

            factor_slug = str(row[config.factor_slug_column])
            if factor_slug not in config.factor_order:
                continue

            if config.crossed_with_agents_per_bucket:
                row_agents_per_bucket = int(row["agents_per_bucket"])
                subgroup_label = str(row_agents_per_bucket)
                if row_agents_per_bucket not in AGENTS_PER_BUCKET_ORDER:
                    continue
                if agents_per_bucket is not None and row_agents_per_bucket != agents_per_bucket:
                    continue
            else:
                subgroup_label = ""

            key = (factor_slug, subgroup_label)
            observation = _seed_observation_from_agent_row(row, key)
            if observation is None:
                continue

            factor_labels_from_csv[factor_slug] = str(row.get(config.factor_label_column, ""))
            observations.append(observation)

    summaries = _summarize_seed_improvements(observations)
    rows: list[ImprovementRow] = []
    for factor_slug in config.factor_order:
        if config.crossed_with_agents_per_bucket:
            subgroup_order = (
                (str(agents_per_bucket),)
                if agents_per_bucket is not None
                else tuple(str(value) for value in AGENTS_PER_BUCKET_ORDER)
            )
        else:
            subgroup_order = ("",)
        for subgroup_label in subgroup_order:
            summary_key = (factor_slug, subgroup_label)
            summary = summaries.get(summary_key)
            if summary is None:
                continue
            summary_values = _summary_fields(summary)
            rows.append(
                ImprovementRow(
                    factor_slug=factor_slug,
                    factor_label=config.factor_labels.get(
                        factor_slug,
                        factor_labels_from_csv.get(factor_slug, factor_slug),
                    ),
                    subgroup_label=subgroup_label,
                    **summary_values,
                )
            )

    return rows


def write_plot_data(output_csv: Path, rows: Sequence[ImprovementRow], count_bucket: int | str) -> None:
    """Write plotted values so the figure can be audited."""

    fieldnames = [
        "factor_slug",
        "factor_label",
        "subgroup_label",
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
                    "factor_slug": row.factor_slug,
                    "factor_label": row.factor_label,
                    "subgroup_label": row.subgroup_label,
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


def plot_rows(
    *,
    config: PlotConfig,
    rows: Sequence[ImprovementRow],
    count_bucket: int | str,
    dpi: int,
    hide_negative_bars: bool,
    group_by_agents_per_bucket: bool,
    output_stem: Path,
    agents_per_bucket: int | None,
) -> None:
    """Render one mirrored sensitivity plot."""

    _configure_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.transforms import blended_transform_factory

    if not rows:
        raise ValueError(f"No rows available to plot for {config.experiment_slug}.")

    x_min, x_max = _shared_compact_x_limits(rows, hide_negative_bars, negative_placeholder=12.0)
    y_positions, group_boundaries = _grouped_y_positions(
        rows,
        group_gap=0.8 if group_by_agents_per_bucket else 0.0,
    )
    group_centers = _group_centers(rows, y_positions)
    colors = [config.factor_colors.get(row.factor_slug, "#B8B8B8") for row in rows]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 10.0,
            "axes.labelsize": 7.7,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure_height = max(3.2, 0.22 * len(rows) + 2.8)
    figure, axis = plt.subplots(figsize=(8.1, figure_height))
    bar_height = 0.64 if len(rows) <= 15 else 0.58
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
        elinewidth=0.58,
        capsize=1.4,
        capthick=0.58,
        alpha=0.74,
        zorder=4,
    )
    axis.errorbar(
        decision_values,
        y_positions,
        xerr=decision_xerr,
        fmt="none",
        ecolor=TEXT_COLOR,
        elinewidth=0.58,
        capsize=1.4,
        capthick=0.58,
        alpha=0.74,
        zorder=4,
    )

    for boundary in group_boundaries:
        axis.axhline(boundary, color="#FFFFFF", linewidth=2.8, zorder=3)
        axis.axhline(boundary, color="#D8D3DD", linewidth=0.42, zorder=3)

    if group_by_agents_per_bucket:
        label_transform = blended_transform_factory(axis.transAxes, axis.transData)
        group_label_x = -0.064
        subgroup_label_x = -0.026
        for group_slug, group_label, group_y in group_centers:
            axis.text(
                group_label_x,
                group_y,
                group_label,
                ha="right",
                va="center",
                transform=label_transform,
                fontsize=6.5,
                color=TEXT_COLOR,
                fontweight="bold",
                clip_on=False,
                zorder=5,
            )
        for y_value, row in zip(y_positions, rows):
            axis.text(
                subgroup_label_x,
                y_value,
                row.subgroup_label,
                ha="left",
                va="center",
                transform=label_transform,
                fontsize=5.8,
                color=TEXT_COLOR,
                fontweight="bold",
                clip_on=False,
                zorder=5,
            )
    else:
        axis.set_yticks(y_positions)
        axis.set_yticklabels([row.factor_label for row in rows])
        axis.tick_params(axis="y", length=0, colors=TEXT_COLOR, labelsize=7.4, pad=4.0)
        for tick_label, row in zip(axis.get_yticklabels(), rows):
            tick_label.set_fontweight("bold")
            tick_label.set_color(TEXT_COLOR)

    for y_value, row in zip(y_positions, rows):
        if hide_negative_bars and row.execution_mean < 0.0:
            axis.text(
                -0.65,
                y_value,
                _missing_bar_label(row.execution_mean, row.execution_ci_lower, row.execution_ci_upper),
                ha="right",
                va="center",
                fontsize=4.95,
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
                fontsize=4.95,
                color=NEGATIVE_TEXT_COLOR,
                fontweight="bold",
                zorder=5,
            )

    axis.axvline(0.0, color=TEXT_COLOR, linewidth=0.9, zorder=6)
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(-0.7, float(y_positions[-1]) + 0.7)
    axis.invert_yaxis()
    if group_by_agents_per_bucket:
        axis.set_yticks([])
        axis.tick_params(axis="y", length=0)

    tick_step = 5 if (x_max - x_min) <= 50 else 10
    axis.xaxis.set_major_locator(ticker.MultipleLocator(tick_step))
    axis.xaxis.set_major_formatter(ticker.FuncFormatter(_format_axis_tick))
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.45, alpha=0.75, zorder=1)
    axis.tick_params(axis="x", length=2.5, colors=TEXT_COLOR)

    for spine in ("left", "right", "top"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color("#AAA4B3")
    axis.spines["bottom"].set_linewidth(0.6)

    axis.set_xlabel("Percent improvement over JEEDS in absolute error", color=CHARCOAL, labelpad=7.0)
    title = (
        f"{config.title} (all agents)"
        if count_bucket == "all"
        else f"{config.title} ({count_bucket} observations/agent)"
    )
    figure.suptitle(
        title,
        y=0.976,
        fontsize=11.6,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    axis.text(
        (x_min + 0.0) / 2.0,
        1.012,
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
        1.012,
        "Decision",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=7.6,
        color=TEXT_COLOR,
        fontweight="bold",
    )

    if group_by_agents_per_bucket:
        figure.text(
            0.5,
            0.026,
            "Small row labels indicate agents per observation-count bucket.",
            ha="center",
            va="bottom",
            fontsize=6.3,
            color=CHARCOAL,
        )
    elif agents_per_bucket is not None:
        figure.text(
            0.5,
            0.026,
            f"{agents_per_bucket} agents per observation-count bucket.",
            ha="center",
            va="bottom",
            fontsize=6.3,
            color=CHARCOAL,
        )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    write_plot_data(output_stem.with_suffix(".csv"), rows, count_bucket)
    left_margin = 0.18 if group_by_agents_per_bucket else 0.16
    bottom_margin = 0.12 if group_by_agents_per_bucket else 0.16
    figure.subplots_adjust(left=left_margin, right=0.985, top=0.885, bottom=bottom_margin)
    _save_figure_bundle(figure, output_stem, dpi)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""

    args = parse_args(argv)
    requested = {piece.strip() for piece in args.experiments.split(",") if piece.strip()}
    configs = [config for config in PLOT_CONFIGS if config.experiment_slug in requested]
    unknown = requested - {config.experiment_slug for config in PLOT_CONFIGS}
    if unknown:
        raise ValueError(f"Unknown experiment slug(s): {', '.join(sorted(unknown))}")

    count_bucket = None if args.average_all_buckets else args.count_bucket
    count_bucket_label: int | str = "all" if count_bucket is None else count_bucket

    for config in configs:
        agents_per_bucket = args.agents_per_bucket if config.crossed_with_agents_per_bucket else None
        rows = compute_improvement_rows(
            config,
            count_bucket,
            agents_per_bucket,
        )
        group_by_agents_per_bucket = (
            config.crossed_with_agents_per_bucket
            and agents_per_bucket is None
        )
        output_stem = config.output_stem
        if count_bucket is None:
            base_name = output_stem.name.removesuffix("_lowest_bucket_improvement_bars")
            qualifiers = []
            if agents_per_bucket is not None:
                qualifiers.append(f"agents_per_bucket_{agents_per_bucket:03d}")
            qualifiers.append("all_agents")
            output_stem = output_stem.with_name("_".join((base_name, *qualifiers, "improvement_bars")))
        elif agents_per_bucket is not None:
            base_name = output_stem.name.removesuffix("_improvement_bars")
            output_stem = output_stem.with_name(f"{base_name}_agents_per_bucket_{agents_per_bucket:03d}_improvement_bars")
        plot_rows(
            config=config,
            rows=rows,
            count_bucket=count_bucket_label,
            dpi=args.dpi,
            hide_negative_bars=not args.show_negative_bars,
            group_by_agents_per_bucket=group_by_agents_per_bucket,
            output_stem=output_stem,
            agents_per_bucket=agents_per_bucket,
        )
        print(f"Wrote {config.experiment_slug} plot to {output_stem.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
