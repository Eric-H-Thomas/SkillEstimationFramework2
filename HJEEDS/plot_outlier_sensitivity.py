# This file was AI-generated and still requires human review. Remove this comment when done.
"""Plot H-JEEDS improvement under explicit outlier contamination."""

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
    NUMERIC_3_COLORS,
    TEXT_COLOR,
    compact_x_limits,
    configure_matplotlib,
    format_axis_tick,
    mirrored_bar_arrays,
    missing_bar_label_plain,
    save_figure_bundle,
    seed_observation_from_agent_row,
    summarize_seed_improvements,
)


DEFAULT_RESULTS_DIR = Path("HJEEDS/results/hjeeds_paper_500_seeds/outlier_sensitivity")
DEFAULT_AGENT_LEVEL_CSV = DEFAULT_RESULTS_DIR / "outlier_sensitivity_agent_level_results.csv"
CONTAMINATION_ORDER = (0, 1, 5)
SUBSET_ORDER = ("all", "inliers", "outliers")
SUBSET_LABELS = {
    "all": "All agents",
    "inliers": "Uncontaminated agents",
    "outliers": "Injected outliers",
}
CONTAMINATION_LABELS = {0: "No outliers", 1: "1 outlier", 5: "5 outliers"}
CONTAMINATION_COLORS = dict(zip(CONTAMINATION_ORDER, NUMERIC_3_COLORS))


@dataclass(frozen=True)
class OutlierRow:
    """One subset-by-contamination result shown in a plot."""

    subset_slug: str
    contamination_count: int
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-level-csv", type=Path, default=DEFAULT_AGENT_LEVEL_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--show-negative-bars", action="store_true")
    return parser.parse_args(argv)


def compute_rows(agent_level_csv: Path) -> dict[str, list[OutlierRow]]:
    """Calculate all-agent, inlier, and outlier improvements from seed means."""

    observations = []
    with agent_level_csv.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("jeeds_status") != "ok" or row.get("hierarchical_status") != "ok":
                continue
            contamination_count = int(row["contamination_count"])
            if contamination_count not in CONTAMINATION_ORDER:
                continue
            is_outlier = row["is_injected_outlier"] == "1"
            subset_slugs = ("all", "outliers" if is_outlier else "inliers")
            for subset_slug in subset_slugs:
                observation = seed_observation_from_agent_row(
                    row,
                    (subset_slug, contamination_count),
                )
                if observation is not None:
                    observations.append(observation)

    summaries = summarize_seed_improvements(observations)
    rows_by_subset: dict[str, list[OutlierRow]] = {subset: [] for subset in SUBSET_ORDER}
    for subset_slug in SUBSET_ORDER:
        for contamination_count in CONTAMINATION_ORDER:
            summary = summaries.get((subset_slug, contamination_count))
            if summary is None:
                continue
            rows_by_subset[subset_slug].append(
                OutlierRow(
                    subset_slug=subset_slug,
                    contamination_count=contamination_count,
                    execution_mean=summary.execution_mean,
                    execution_ci_lower=summary.execution_ci_lower,
                    execution_ci_upper=summary.execution_ci_upper,
                    decision_mean=summary.decision_mean,
                    decision_ci_lower=summary.decision_ci_lower,
                    decision_ci_upper=summary.decision_ci_upper,
                    average_mean=summary.average_mean,
                    num_seeds=summary.num_seeds,
                    num_agents_per_seed=summary.num_agents_per_seed,
                )
            )
    return rows_by_subset


def write_plot_data(path: Path, rows_by_subset: dict[str, list[OutlierRow]]) -> None:
    """Write every plotted estimate and interval for auditability."""

    fieldnames = [
        "population_subset",
        "contamination_count",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for subset_slug in SUBSET_ORDER:
            for row in rows_by_subset[subset_slug]:
                writer.writerow(
                    {
                        "population_subset": subset_slug,
                        "contamination_count": row.contamination_count,
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


def _draw_axis(axis, rows: Sequence[OutlierRow], x_limits: tuple[float, float], hide_negative_bars: bool) -> None:
    """Draw one mirrored bar panel."""

    import matplotlib.ticker as ticker

    y_positions = np.arange(len(rows), dtype=float)
    colors = [CONTAMINATION_COLORS[row.contamination_count] for row in rows]
    execution, decision, execution_xerr, decision_xerr = mirrored_bar_arrays(rows, hide_negative_bars)
    for values in (execution, decision):
        axis.barh(
            y_positions,
            values,
            height=0.62,
            color=colors,
            edgecolor="#F7F5F8",
            linewidth=0.45,
            zorder=2,
        )
    for values, errors in ((execution, execution_xerr), (decision, decision_xerr)):
        axis.errorbar(
            values,
            y_positions,
            xerr=errors,
            fmt="none",
            ecolor=TEXT_COLOR,
            elinewidth=0.65,
            capsize=1.7,
            capthick=0.65,
            alpha=0.76,
            zorder=4,
        )

    negative_label_offset = 0.04 * (x_limits[1] - x_limits[0])
    for y_value, row in zip(y_positions, rows):
        if hide_negative_bars and row.execution_mean < 0.0:
            axis.text(
                -negative_label_offset,
                y_value,
                missing_bar_label_plain(row.execution_mean, row.execution_ci_lower, row.execution_ci_upper),
                ha="right",
                va="center",
                fontsize=5.0,
                fontweight="bold",
                color=NEGATIVE_TEXT_COLOR,
                zorder=5,
            )
        if hide_negative_bars and row.decision_mean < 0.0:
            axis.text(
                negative_label_offset,
                y_value,
                missing_bar_label_plain(row.decision_mean, row.decision_ci_lower, row.decision_ci_upper),
                ha="left",
                va="center",
                fontsize=5.0,
                fontweight="bold",
                color=NEGATIVE_TEXT_COLOR,
                zorder=5,
            )

    x_min, x_max = x_limits
    axis.axvline(0.0, color=TEXT_COLOR, linewidth=0.85, zorder=6)
    axis.set_xlim(x_limits)
    axis.set_ylim(-0.65, len(rows) - 0.35)
    axis.invert_yaxis()
    axis.set_yticks(y_positions, [CONTAMINATION_LABELS[row.contamination_count] for row in rows])
    axis.tick_params(axis="y", length=0, labelsize=7.0, colors=TEXT_COLOR, pad=3.0)
    for label in axis.get_yticklabels():
        label.set_fontweight("bold")
    tick_step = 5 if (x_max - x_min) <= 50 else 10
    axis.xaxis.set_major_locator(ticker.MultipleLocator(tick_step))
    axis.xaxis.set_major_formatter(ticker.FuncFormatter(format_axis_tick))
    axis.tick_params(axis="x", length=2.2, labelsize=6.4, colors=TEXT_COLOR)
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.42, alpha=0.74, zorder=1)
    for spine in ("left", "right", "top"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color("#AAA4B3")
    axis.spines["bottom"].set_linewidth(0.55)
    axis.text(
        (x_min + 0.0) / 2.0,
        1.03,
        "Execution",
        transform=axis.get_xaxis_transform(),
        ha="center",
        fontsize=7.0,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    axis.text(
        x_max / 2.0,
        1.03,
        "Decision",
        transform=axis.get_xaxis_transform(),
        ha="center",
        fontsize=7.0,
        fontweight="bold",
        color=TEXT_COLOR,
    )


def render(rows_by_subset: dict[str, list[OutlierRow]], output_dir: Path, dpi: int, hide_negative_bars: bool) -> None:
    """Render a main all-agent plot and an appendix subgroup figure."""

    configure_matplotlib()
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    all_rows = [row for subset in SUBSET_ORDER for row in rows_by_subset[subset]]
    x_limits = compact_x_limits(all_rows, hide_negative_bars, negative_placeholder=10.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_plot_data(output_dir / "outlier_sensitivity_plot_data.csv", rows_by_subset)

    left, right = 0.20, 0.985
    figure, axis = plt.subplots(figsize=(7.2, 3.15))
    _draw_axis(axis, rows_by_subset["all"], x_limits, hide_negative_bars)
    axis.set_xlabel("Percent improvement over JEEDS in absolute error", color=CHARCOAL, labelpad=7.0)
    figure.suptitle(
        "Outlier contamination sensitivity (all agents)",
        x=(left + right) / 2.0,
        y=0.94,
        fontsize=11.5,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    figure.subplots_adjust(left=left, right=right, top=0.77, bottom=0.2)
    save_figure_bundle(figure, output_dir / "outlier_sensitivity_all_agents", dpi)
    plt.close(figure)

    left, right = 0.09, 0.99
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 3.75))
    for axis, subset_slug in zip(axes, SUBSET_ORDER):
        _draw_axis(axis, rows_by_subset[subset_slug], x_limits, hide_negative_bars)
        axis.set_title(SUBSET_LABELS[subset_slug], fontsize=9.0, fontweight="bold", color=TEXT_COLOR, pad=15.0)
    figure.suptitle(
        "Outlier contamination sensitivity",
        x=(left + right) / 2.0,
        y=0.975,
        fontsize=13.0,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    figure.supxlabel(
        "Percent improvement over JEEDS in absolute error",
        x=(left + right) / 2.0,
        y=0.035,
        fontsize=7.4,
        color=CHARCOAL,
    )
    figure.subplots_adjust(left=left, right=right, top=0.79, bottom=0.19, wspace=0.31)
    save_figure_bundle(figure, output_dir / "outlier_sensitivity_subgroups", dpi)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows_by_subset = compute_rows(args.agent_level_csv)
    render(rows_by_subset, args.output_dir, args.dpi, not args.show_negative_bars)
    print(f"Wrote outlier-sensitivity plots to {args.output_dir}")


if __name__ == "__main__":
    main()
