# This file was AI-generated and still requires human review. Remove this comment when done.
"""Create a default-only agents-per-bucket sensitivity figure."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from HJEEDS.plot_agents_per_bucket_robustness import (
    AGENTS_BASE_COLORS,
    DEFAULT_AGENT_LEVEL_CSV,
    DEFAULT_RESULTS_DIR,
    ImprovementRow,
    compute_improvement_rows,
    compute_overall_improvement_rows,
)
from HJEEDS.sensitivity_plot_common import (
    GRID_COLOR,
    NEGATIVE_TEXT_COLOR,
    TEXT_COLOR,
    compact_x_limits as _compact_x_limits,
    configure_matplotlib as _configure_matplotlib,
    format_axis_tick as _format_axis_tick,
    mirrored_bar_arrays as _mirrored_bar_arrays,
    missing_bar_label_latex as _missing_bar_label,
    save_figure_bundle as _save_figure_bundle,
    tick_step as _tick_step,
)


DEFAULT_OUTPUT_STEM = DEFAULT_RESULTS_DIR / "agents_per_bucket_default_lowest_bucket_improvement_bars"
DEFAULT_OVERALL_OUTPUT_STEM = DEFAULT_RESULTS_DIR / "agents_per_bucket_default_all_agents_improvement_bars"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-level-csv", type=Path, default=DEFAULT_AGENT_LEVEL_CSV)
    parser.add_argument("--output-stem", type=Path, default=None)
    parser.add_argument("--count-bucket", type=int, default=5)
    parser.add_argument(
        "--average-all-buckets",
        action="store_true",
        help="Average over all observation-count buckets instead of selecting one bucket.",
    )
    parser.add_argument("--dpi", type=int, default=450)
    return parser.parse_args(argv)


def _default_rows(agent_level_csv: Path, count_bucket: int, average_all_buckets: bool) -> list[ImprovementRow]:
    """Return default-condition rows for one observation-count bucket or all buckets."""

    if average_all_buckets:
        rows = compute_overall_improvement_rows(agent_level_csv=agent_level_csv)
        selected_bucket = "all"
    else:
        rows, selected_bucket = compute_improvement_rows(
            agent_level_csv=agent_level_csv,
            count_bucket=count_bucket,
        )
    default_rows = [row for row in rows if row.condition.condition_slug == "default"]
    if not default_rows:
        raise ValueError(f"No default rows found for count bucket {selected_bucket}.")
    return default_rows


def _write_plot_data(output_csv: Path, rows: Sequence[ImprovementRow], count_bucket: int | str) -> None:
    """Write plotted values so the figure can be audited."""

    fieldnames = [
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
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "agents_per_bucket": row.condition.agents_per_bucket,
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


def render(
    rows: Sequence[ImprovementRow],
    count_bucket: int | str,
    output_stem: Path,
    dpi: int,
) -> None:
    """Render the default-only agents-per-bucket figure."""

    _configure_matplotlib()

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    hide_negative_bars = True
    x_min, x_max = _compact_x_limits(rows, hide_negative_bars, base_extent=4.0)
    y_positions = list(range(len(rows)))
    colors = [AGENTS_BASE_COLORS[row.condition.agents_per_bucket] for row in rows]
    execution_values, decision_values, execution_xerr, decision_xerr = _mirrored_bar_arrays(rows, hide_negative_bars)

    figure, axis = plt.subplots(figsize=(5.95, 2.85))
    bar_height = 0.58

    axis.barh(
        y_positions,
        execution_values,
        height=bar_height,
        color=colors,
        edgecolor="#F7F5F8",
        linewidth=0.38,
        zorder=2,
    )
    axis.barh(
        y_positions,
        decision_values,
        height=bar_height,
        color=colors,
        edgecolor="#F7F5F8",
        linewidth=0.38,
        zorder=2,
    )
    axis.errorbar(
        execution_values,
        y_positions,
        xerr=execution_xerr,
        fmt="none",
        ecolor=TEXT_COLOR,
        elinewidth=0.62,
        capsize=1.7,
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
        capsize=1.7,
        capthick=0.62,
        alpha=0.74,
        zorder=4,
    )

    for y_value, row in zip(y_positions, rows):
        if hide_negative_bars and row.execution_mean < 0.0:
            axis.text(
                -0.5,
                y_value,
                _missing_bar_label(row.execution_mean, row.execution_ci_lower, row.execution_ci_upper),
                ha="right",
                va="center",
                fontsize=6.1,
                color=NEGATIVE_TEXT_COLOR,
                fontweight="bold",
                zorder=5,
            )
        if hide_negative_bars and row.decision_mean < 0.0:
            axis.text(
                0.5,
                y_value,
                _missing_bar_label(row.decision_mean, row.decision_ci_lower, row.decision_ci_upper),
                ha="left",
                va="center",
                fontsize=6.1,
                color=NEGATIVE_TEXT_COLOR,
                fontweight="bold",
                zorder=5,
            )

    axis.axvline(0.0, color=TEXT_COLOR, linewidth=0.9, zorder=6)
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(-0.55, len(rows) - 0.45)
    axis.invert_yaxis()
    axis.set_yticks(y_positions)
    axis.set_yticklabels([str(row.condition.agents_per_bucket) for row in rows], fontweight="bold")
    for tick_label in axis.get_yticklabels():
        tick_label.set_color(TEXT_COLOR)
    axis.set_ylabel("Agents per bucket", color=TEXT_COLOR, labelpad=8)

    axis.xaxis.set_major_locator(ticker.MultipleLocator(_tick_step(x_min, x_max)))
    axis.xaxis.set_major_formatter(ticker.FuncFormatter(_format_axis_tick))
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.45, alpha=0.75, zorder=1)
    axis.tick_params(axis="x", length=2.5, colors=TEXT_COLOR)
    axis.tick_params(axis="y", length=0, pad=6)

    for spine in ("left", "right", "top"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color("#AAA4B3")
    axis.spines["bottom"].set_linewidth(0.6)

    axis.set_xlabel("Percent improvement over JEEDS in absolute error")
    title = (
        "Agents-per-bucket sensitivity (all agents)"
        if count_bucket == "all"
        else f"Agents-per-bucket sensitivity ({count_bucket} observations/agent)"
    )
    figure.suptitle(
        title,
        y=0.985,
        fontsize=9.8,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    axis.text(
        (x_min + 0.0) / 2.0,
        1.025,
        "Execution skill",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=TEXT_COLOR,
        fontweight="bold",
    )
    axis.text(
        x_max / 2.0,
        1.025,
        "Decision-making skill",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=TEXT_COLOR,
        fontweight="bold",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.subplots_adjust(left=0.12, right=0.985, top=0.84, bottom=0.15)
    _save_figure_bundle(figure, output_stem, dpi)
    plt.close(figure)
    _write_plot_data(output_stem.with_suffix(".csv"), rows, count_bucket)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""

    args = parse_args(argv)
    output_stem = args.output_stem
    if output_stem is None:
        output_stem = DEFAULT_OVERALL_OUTPUT_STEM if args.average_all_buckets else DEFAULT_OUTPUT_STEM
    count_bucket: int | str = "all" if args.average_all_buckets else args.count_bucket
    rows = _default_rows(args.agent_level_csv, args.count_bucket, args.average_all_buckets)
    render(rows, count_bucket, output_stem, args.dpi)
    print(f"Wrote default-only agents-per-bucket figure to {output_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
