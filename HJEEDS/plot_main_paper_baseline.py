# This file was AI-generated and still requires human review. Remove this comment when done.
"""Render the baseline H-JEEDS comparison for one-column paper layout.

This plotting-only entry point reads the existing across-seed summary table. It
does not rerun simulation, inference, or aggregation.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from HJEEDS.artifacts import (
    METHOD_ORDER,
    method_label,
)
from HJEEDS.sensitivity_plot_common import (
    CHARCOAL,
    GRID_COLOR,
    TEXT_COLOR,
    configure_matplotlib,
    save_figure_bundle,
)


DEFAULT_SUMMARY_CSV = Path(
    "HJEEDS/results/hjeeds_paper_500_seeds/true_correlation/"
    "true_correlation_r_neg_0_5/agents_per_bucket_005/summary_by_bucket.csv"
)
DEFAULT_OUTPUT_STEM = Path(
    "HJEEDS/results/hjeeds_paper_500_seeds/final_paper_plots/"
    "00_baseline_error_by_count_bucket"
)

EXECUTION_METRIC = "abs_sigma_error"
DECISION_METRIC = "abs_rationality_percent_error"
METRICS = (EXECUTION_METRIC, DECISION_METRIC)

SERIES_STYLES = {
    (EXECUTION_METRIC, "jeeds"): {
        "color": "#339CFF",
        "marker": "o",
        "linestyle": "-",
        "metric_label": "execution",
    },
    (EXECUTION_METRIC, "hierarchical"): {
        "color": "#F3883B",
        "marker": "s",
        "linestyle": "-",
        "metric_label": "execution",
    },
    (DECISION_METRIC, "jeeds"): {
        "color": "#5DC977",
        "marker": "^",
        "linestyle": "--",
        "metric_label": "decision",
    },
    (DECISION_METRIC, "hierarchical"): {
        "color": "#EB77B1",
        "marker": "D",
        "linestyle": "--",
        "metric_label": "decision",
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-by-bucket-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="Existing across-seed summary table to plot.",
    )
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--dpi", type=int, default=450)
    return parser.parse_args(argv)


def _optional_float(value: Any) -> float | None:
    """Parse one optional numeric field."""

    if value in (None, ""):
        return None
    return float(value)


def read_summary_rows(summary_csv: Path) -> list[dict[str, Any]]:
    """Read and validate the rows needed by the paper figure."""

    required_metrics = set(METRICS)
    rows: list[dict[str, Any]] = []
    with summary_csv.open("r", newline="") as handle:
        for source_row in csv.DictReader(handle):
            metric = str(source_row.get("metric", ""))
            if metric not in required_metrics:
                continue

            mean = _optional_float(source_row.get("mean"))
            ci_lower = _optional_float(source_row.get("ci_lower"))
            ci_upper = _optional_float(source_row.get("ci_upper"))
            if mean is None or ci_lower is None or ci_upper is None:
                continue
            if ci_lower > mean or mean > ci_upper:
                raise ValueError(
                    "Expected ci_lower <= mean <= ci_upper, received "
                    f"{ci_lower}, {mean}, {ci_upper}."
                )

            rows.append(
                {
                    "method": str(source_row.get("method", "")),
                    "metric": metric,
                    "count_bucket": int(source_row["count_bucket"]),
                    "mean": mean,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
            )

    if not rows:
        raise ValueError(f"No baseline paper metrics found in {summary_csv}.")
    return rows


def _method_order(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Return methods in the paper's standard order."""

    return sorted(
        {str(row["method"]) for row in rows},
        key=lambda method: (METHOD_ORDER.get(method, len(METHOD_ORDER)), method),
    )


def _draw_metric(
    axis,
    rows: Sequence[dict[str, Any]],
    metric_name: str,
    bucket_positions: dict[int, int],
) -> dict[str, Any]:
    """Draw one metric on its assigned y-axis and return its legend handles."""

    metric_rows = [row for row in rows if row["metric"] == metric_name]
    handles: dict[str, Any] = {}
    for method in _method_order(metric_rows):
        style = SERIES_STYLES.get((metric_name, method))
        if style is None:
            raise ValueError(
                f"No paper style is defined for method={method!r}, metric={metric_name!r}."
            )
        method_rows = sorted(
            [row for row in metric_rows if row["method"] == method],
            key=lambda row: bucket_positions[int(row["count_bucket"])],
        )
        x_values = [bucket_positions[int(row["count_bucket"])] for row in method_rows]
        means = np.asarray([float(row["mean"]) for row in method_rows], dtype=float)
        lower = np.asarray([float(row["ci_lower"]) for row in method_rows], dtype=float)
        upper = np.asarray([float(row["ci_upper"]) for row in method_rows], dtype=float)
        y_error = np.vstack((means - lower, upper - means))

        errorbar = axis.errorbar(
            x_values,
            means,
            yerr=y_error,
            color=style["color"],
            ecolor=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.65,
            linewidth=1.4,
            elinewidth=1.0,
            capsize=2.7,
            capthick=1.0,
            label=f"{method_label(method)} · {style['metric_label']}",
            zorder=4,
        )
        handles[method] = errorbar

    return handles


def _style_axis(axis, *, side: str) -> None:
    """Apply compact paper styling to one side of the dual-axis plot."""

    axis.set_axisbelow(True)
    axis.tick_params(
        axis="both",
        colors=TEXT_COLOR,
        labelsize=9.0,
        length=3.0,
        width=0.7,
        pad=2.5,
    )

    axis.spines["top"].set_visible(False)
    if side == "left":
        axis.spines["right"].set_visible(False)
        visible_spines = ("left", "bottom")
    else:
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_visible(False)
        visible_spines = ("right",)
    for spine_name in visible_spines:
        axis.spines[spine_name].set_color(CHARCOAL)
        axis.spines[spine_name].set_linewidth(0.75)


def render(rows: Sequence[dict[str, Any]], output_stem: Path, dpi: int) -> None:
    """Render and save the native one-column baseline figure."""

    configure_matplotlib()
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )

    bucket_values = sorted({int(row["count_bucket"]) for row in rows})
    bucket_positions = {bucket: index for index, bucket in enumerate(bucket_values)}
    figure, execution_axis = plt.subplots(figsize=(3.35, 2.65))
    decision_axis = execution_axis.twinx()
    figure.patch.set_facecolor("white")

    execution_handles = _draw_metric(
        execution_axis,
        rows,
        EXECUTION_METRIC,
        bucket_positions,
    )
    decision_handles = _draw_metric(
        decision_axis,
        rows,
        DECISION_METRIC,
        bucket_positions,
    )

    execution_axis.set_xticks(
        list(bucket_positions.values()),
        [str(bucket) for bucket in bucket_values],
    )
    execution_axis.set_xlabel("Observations per agent", color=TEXT_COLOR, labelpad=4.0)
    execution_axis.set_ylabel(
        "Execution error " + r"($|\hat{\sigma}-\sigma|$)",
        color=TEXT_COLOR,
        labelpad=5.0,
    )
    decision_axis.set_ylabel(
        "Decision error\n(percentage points)",
        color=TEXT_COLOR,
        labelpad=5.0,
    )

    execution_max = max(
        float(row["ci_upper"]) for row in rows if row["metric"] == EXECUTION_METRIC
    )
    decision_max = max(
        float(row["ci_upper"]) for row in rows if row["metric"] == DECISION_METRIC
    )
    execution_limit = max(0.2, np.ceil(execution_max / 0.2) * 0.2)
    decision_limit = max(5.0, np.ceil(decision_max / 5.0) * 5.0)
    execution_axis.set_ylim(0.0, execution_limit)
    decision_axis.set_ylim(0.0, decision_limit)
    execution_axis.set_yticks(np.arange(0.0, execution_limit + 0.1, 0.2))
    decision_axis.set_yticks(np.arange(0.0, decision_limit + 2.5, 5.0))

    execution_axis.grid(
        axis="y",
        color=GRID_COLOR,
        linewidth=0.55,
        alpha=0.8,
        zorder=1,
    )
    _style_axis(execution_axis, side="left")
    _style_axis(decision_axis, side="right")

    legend_entries = (
        execution_handles["jeeds"],
        decision_handles["jeeds"],
        execution_handles["hierarchical"],
        decision_handles["hierarchical"],
    )
    legend_labels = tuple(handle.get_label() for handle in legend_entries)
    legend = figure.legend(
        legend_entries,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        columnspacing=1.1,
        handlelength=1.8,
        handletextpad=0.5,
        labelspacing=0.55,
        borderaxespad=0.0,
    )
    for label in legend.get_texts():
        label.set_color(TEXT_COLOR)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.subplots_adjust(left=0.195, right=0.805, top=0.80, bottom=0.18)
    save_figure_bundle(figure, output_stem, dpi)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""

    args = parse_args(argv)
    rows = read_summary_rows(args.summary_by_bucket_csv)
    render(rows, args.output_stem, args.dpi)
    print(
        f"Wrote baseline paper figure to {args.output_stem.with_suffix('.png')} "
        f"from {args.summary_by_bucket_csv}"
    )


if __name__ == "__main__":
    main()
