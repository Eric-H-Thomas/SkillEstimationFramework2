# This file was AI-generated and still requires human review. Remove this comment when done.
"""Create the appendix plot for H-JEEDS hyperprior robustness.

The figure summarizes the lowest-data bucket by plotting percent improvement
of H-JEEDS over JEEDS for each hyperprior-robustness condition. The default
hyperprior appears once in each focus-area group. Execution-skill improvement
is mirrored to the left of zero; decision-skill improvement, using the
rationality-percentage error metric, is mirrored to the right.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from HJEEDS.sensitivity_plot_common import (
    CHARCOAL,
    CATEGORICAL_COLORS,
    GRID_COLOR,
    NEGATIVE_TEXT_COLOR,
    TEXT_COLOR,
    blend as _blend,
    compact_x_limits as _shared_compact_x_limits,
    configure_matplotlib as _configure_matplotlib,
    format_axis_tick as _format_axis_tick,
    grouped_y_positions as _shared_grouped_y_positions,
    mirrored_bar_arrays as _mirrored_bar_arrays,
    missing_bar_label_latex as _missing_bar_label,
    save_figure_bundle as _save_figure_bundle,
    seed_observation_from_agent_row as _seed_observation_from_agent_row,
    summary_fields as _summary_fields,
    summarize_seed_improvements as _summarize_seed_improvements,
)


DEFAULT_RESULTS_DIR = Path("HJEEDS/results/hjeeds_paper_500_seeds/hyperprior_robustness")
DEFAULT_AGENT_LEVEL_CSV = DEFAULT_RESULTS_DIR / "prior_sensitivity_agent_level_results.csv"
DEFAULT_CONDITIONS_CSV = DEFAULT_RESULTS_DIR / "prior_sensitivity_conditions.csv"
DEFAULT_OUTPUT_STEM = DEFAULT_RESULTS_DIR / "prior_sensitivity_lowest_bucket_improvement_bars"

HYPERPRIOR_KEY_COLUMNS = (
    "hyperprior_mu_eta",
    "hyperprior_mu_rho",
    "hyperprior_mu_eta_sd",
    "hyperprior_mu_rho_sd",
    "hyperprior_log_tau_eta_mean",
    "hyperprior_log_tau_eta_sd",
    "hyperprior_log_tau_rho_mean",
    "hyperprior_log_tau_rho_sd",
    "hyperprior_m_r",
    "hyperprior_s_r",
)

FOCUS_BASE_COLORS = {
    "average_skill": CATEGORICAL_COLORS[0],
    "population_spread": CATEGORICAL_COLORS[1],
    "correlation": CATEGORICAL_COLORS[2],
    "combined": CATEGORICAL_COLORS[3],
}

FOCUS_LABELS = {
    "average_skill": "Average population skill",
    "population_spread": "Population spread",
    "correlation": "Execution/decision correlation",
    "combined": "Combined",
}

FOCUS_LEGEND_LABELS = {
    "average_skill": "Avg. population skill",
    "population_spread": "Population spread",
    "correlation": "Correlation",
    "combined": "Combined",
}

BIAS_CODES = {
    "strong_reverse_misspecification": "SR",
    "moderate_reverse_misspecification": "MR",
    "unbiased": "U",
    "moderate_adverse_misspecification": "MA",
    "strong_adverse_misspecification": "SA",
}

CONFIDENCE_CODES = {
    "weak": "w",
    "default": "d",
    "strong": "s",
}


@dataclass(frozen=True)
class CanonicalCondition:
    """Metadata for one plotted hyperprior condition."""

    hyperprior_key: tuple[str, ...]
    condition_slug: str
    condition_label: str
    focus_slug: str
    focus_label: str
    bias_slug: str
    bias_label: str
    confidence_slug: str
    confidence_label: str
    is_default: bool


@dataclass(frozen=True)
class ImprovementRow:
    """One plotted row of seed-aggregated improvement values."""

    condition: CanonicalCondition
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
    parser.add_argument("--conditions-csv", type=Path, default=DEFAULT_CONDITIONS_CSV)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument(
        "--lowest-bucket",
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
        "--hide-negative-bars",
        action="store_true",
        help="Suppress negative-mean improvement bars and print the negative percentage instead.",
    )
    parser.add_argument(
        "--label-placement",
        choices=("center", "left"),
        default="center",
        help="Place condition labels near the center line or in a single left-side label column.",
    )
    parser.add_argument(
        "--x-limits",
        type=float,
        nargs=2,
        metavar=("X_MIN", "X_MAX"),
        default=None,
        help="Optional x-axis limits. Defaults to compact asymmetric limits based on plotted intervals.",
    )
    parser.add_argument(
        "--symmetric-x-axis",
        action="store_true",
        help="Use the earlier symmetric x-axis limits around zero.",
    )
    return parser.parse_args(argv)


def _read_dict_rows(path: Path) -> list[dict[str, Any]]:
    """Read a CSV file into dictionaries."""

    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _hyperprior_key(row: dict[str, Any]) -> tuple[str, ...]:
    """Return the columns that define a unique hyperprior setting."""

    return tuple(str(row[column]) for column in HYPERPRIOR_KEY_COLUMNS)


def _is_default_condition(condition_slug: str) -> bool:
    """Return true for the repeated unbiased/default baseline condition."""

    return condition_slug.endswith("__unbiased__default")


def plot_conditions(conditions_csv: Path) -> dict[str, CanonicalCondition]:
    """Return metadata for every plotted condition."""

    condition_rows = _read_dict_rows(conditions_csv)
    conditions: dict[str, CanonicalCondition] = {}
    for row in condition_rows:
        condition_slug = str(row["condition_slug"])
        is_default = _is_default_condition(condition_slug)
        conditions[condition_slug] = CanonicalCondition(
            hyperprior_key=_hyperprior_key(row),
            condition_slug=condition_slug,
            condition_label="default/baseline" if is_default else str(row["condition_label"]),
            focus_slug=str(row["focus_slug"]),
            focus_label=str(row["focus_label"]),
            bias_slug=str(row["bias_slug"]),
            bias_label=str(row["bias_label"]),
            confidence_slug=str(row["confidence_slug"]),
            confidence_label=str(row["confidence_label"]),
            is_default=is_default,
        )

    return conditions


def _lowest_bucket(agent_level_csv: Path, requested_bucket: int | None) -> int:
    """Return the selected lowest-data bucket."""

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
    conditions_csv: Path,
    lowest_bucket: int | None,
) -> tuple[list[ImprovementRow], int]:
    """Compute seed-level percent improvements for each plotted condition."""

    conditions = plot_conditions(conditions_csv)
    selected_bucket = _lowest_bucket(agent_level_csv, lowest_bucket)

    observations = []
    with agent_level_csv.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["count_bucket"]) != selected_bucket:
                continue
            if row.get("jeeds_status") != "ok" or row.get("hierarchical_status") != "ok":
                continue

            condition_slug = str(row["condition_slug"])
            if condition_slug not in conditions:
                continue
            observation = _seed_observation_from_agent_row(row, condition_slug)
            if observation is not None:
                observations.append(observation)

    rows: list[ImprovementRow] = []
    for condition_slug, summary in _summarize_seed_improvements(observations).items():
        rows.append(
            ImprovementRow(
                condition=conditions[condition_slug],
                **_summary_fields(summary),
            )
        )

    focus_order = {focus_slug: index for index, focus_slug in enumerate(FOCUS_BASE_COLORS)}
    rows.sort(
        key=lambda row: (
            focus_order.get(row.condition.focus_slug, len(focus_order)),
            -row.average_mean,
            -row.execution_mean,
            -row.decision_mean,
        )
    )
    return rows, selected_bucket


def bar_color(condition: CanonicalCondition) -> str:
    """Return the row color with focus hue and confidence shade."""

    if condition.is_default:
        return CHARCOAL

    base = FOCUS_BASE_COLORS.get(condition.focus_slug, "#B8B8B8")
    if condition.confidence_slug == "weak":
        return _blend(base, "#FFFFFF", 0.28)
    if condition.confidence_slug == "strong":
        return _blend(base, CHARCOAL, 0.24)
    return base


def bar_label(condition: CanonicalCondition) -> str:
    """Return the short label drawn on each bar."""

    if condition.is_default:
        return "DEF"
    bias_code = BIAS_CODES.get(condition.bias_slug, condition.bias_slug[:2].upper())
    confidence_code = CONFIDENCE_CODES.get(condition.confidence_slug, condition.confidence_slug[:1])
    return f"{bias_code}-{confidence_code}"


def _grouped_y_positions(rows: Sequence[ImprovementRow], group_gap: float = 0.85) -> tuple[np.ndarray, list[float]]:
    """Return y positions with extra space between focus-area groups."""

    return _shared_grouped_y_positions([row.condition.focus_slug for row in rows], group_gap)


def _bucket_title_phrase(bucket: int) -> str:
    """Return a compact figure-title phrase for one observation-count bucket."""

    if bucket == 5:
        return "the lowest-data bucket"
    return f"the {bucket}-observation bucket"


def _write_plot_data(output_csv: Path, rows: Sequence[ImprovementRow], lowest_bucket: int) -> None:
    """Write the plotted data so the figure can be audited."""

    fieldnames = [
        "condition_slug",
        "condition_label",
        "focus_slug",
        "focus_label",
        "bias_slug",
        "bias_label",
        "confidence_slug",
        "confidence_label",
        "is_default",
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
                    "condition_slug": row.condition.condition_slug,
                    "condition_label": row.condition.condition_label,
                    "focus_slug": row.condition.focus_slug,
                    "focus_label": row.condition.focus_label,
                    "bias_slug": row.condition.bias_slug,
                    "bias_label": row.condition.bias_label,
                    "confidence_slug": row.condition.confidence_slug,
                    "confidence_label": row.condition.confidence_label,
                    "is_default": row.condition.is_default,
                    "count_bucket": lowest_bucket,
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


def _symmetric_x_limits(rows: Sequence[ImprovementRow]) -> tuple[float, float]:
    """Return the earlier symmetric x-axis limits around zero."""

    max_extent = max(
        max(
            abs(row.execution_ci_lower),
            abs(row.execution_ci_upper),
            abs(row.decision_ci_lower),
            abs(row.decision_ci_upper),
        )
        for row in rows
    )
    axis_limit = float(int(math.ceil(max_extent / 5.0) * 5.0 + 5.0))
    return -axis_limit, axis_limit


def plot_improvement_bars(
    *,
    rows: Sequence[ImprovementRow],
    lowest_bucket: int,
    output_stem: Path,
    dpi: int,
    hide_negative_bars: bool = False,
    label_placement: str = "center",
    x_limits: tuple[float, float] | None = None,
    symmetric_x_axis: bool = False,
) -> None:
    """Render the mirrored robustness bar plot."""

    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    from matplotlib.patches import Patch

    if not rows:
        raise ValueError("No rows available to plot.")

    if x_limits is not None:
        x_min, x_max = x_limits
        if x_min >= 0.0 or x_max <= 0.0 or x_min >= x_max:
            raise ValueError(f"x-axis limits must bracket zero in increasing order, got {x_limits}")
    elif symmetric_x_axis:
        x_min, x_max = _symmetric_x_limits(rows)
    else:
        x_min, x_max = _shared_compact_x_limits(
            rows,
            hide_negative_bars,
            base_extent=2.5,
            negative_placeholder=6.0,
        )

    y_positions, group_boundaries = _grouped_y_positions(rows)
    colors = [bar_color(row.condition) for row in rows]
    labels = [bar_label(row.condition) for row in rows]

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

    figure_height = max(9.0, 0.138 * len(rows) + 2.75)
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
        label_x = x_min + 0.9
        for y_value, _row, label in zip(y_positions, rows, labels):
            axis.text(
                label_x,
                y_value,
                label,
                ha="left",
                va="center",
                fontsize=5.2,
                color=TEXT_COLOR,
                zorder=5,
            )

    for y_value, row, label in zip(y_positions, rows, labels):
        label_color = "#FFFFFF" if row.condition.is_default else TEXT_COLOR
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
                fontsize=4.9,
                color=label_color,
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
                fontsize=4.9,
                color=label_color,
                zorder=5,
            )

    axis.axvline(0.0, color=TEXT_COLOR, linewidth=0.9, zorder=6)
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(-0.7, float(y_positions[-1]) + 0.7)
    axis.invert_yaxis()
    axis.set_yticks([])
    tick_step = 5 if (x_max - x_min) <= 40 else 10
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
        f"H-JEEDS improvement in {_bucket_title_phrase(lowest_bucket)} ({lowest_bucket} observations per agent)",
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

    focus_handles = [
        Patch(facecolor=color, edgecolor="none", label=FOCUS_LEGEND_LABELS[slug])
        for slug, color in FOCUS_BASE_COLORS.items()
    ]
    focus_handles.append(Patch(facecolor=CHARCOAL, edgecolor="none", label="Default/baseline"))
    legend = figure.legend(
        handles=focus_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=5,
        frameon=False,
        columnspacing=1.1,
        handlelength=1.1,
        handleheight=0.9,
        borderaxespad=0.0,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    figure.text(
        0.5,
        0.012,
        "Labels: SR/MR/U/MA/SA = strong reverse, moderate reverse, unbiased, moderate adverse, strong adverse; "
        "w/d/s = weak, default, strong confidence. Shade darkens with confidence.",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=CHARCOAL,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.subplots_adjust(left=0.045, right=0.985, top=0.895, bottom=0.073)
    _save_figure_bundle(figure, output_stem, dpi)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""

    args = parse_args(argv)
    rows, lowest_bucket = compute_improvement_rows(
        agent_level_csv=args.agent_level_csv,
        conditions_csv=args.conditions_csv,
        lowest_bucket=args.lowest_bucket,
    )
    _write_plot_data(args.output_stem.with_suffix(".csv"), rows, lowest_bucket)
    plot_improvement_bars(
        rows=rows,
        lowest_bucket=lowest_bucket,
        output_stem=args.output_stem,
        dpi=args.dpi,
        hide_negative_bars=args.hide_negative_bars,
        label_placement=args.label_placement,
        x_limits=tuple(args.x_limits) if args.x_limits is not None else None,
        symmetric_x_axis=args.symmetric_x_axis,
    )
    print(
        f"Wrote {len(rows)} settings to {args.output_stem.with_suffix('.png')} "
        f"and {args.output_stem.with_suffix('.svg')}"
    )


if __name__ == "__main__":
    main()
