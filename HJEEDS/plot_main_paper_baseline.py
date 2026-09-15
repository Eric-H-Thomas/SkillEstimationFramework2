# Paper correspondence: Main Figure `fig:baseline_results_plot`.
"""Render the baseline H-JEEDS comparison for one-column paper layout.

This plotting-only entry point reads the existing across-seed summary table. It
does not rerun simulation, inference, or aggregation.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
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
from scripts.validate_publication_results import validate_agent_csv


DEFAULT_SUMMARY_CSV = Path(
    "HJEEDS/results/hjeeds_paper_500_seeds/baseline/summary_by_bucket.csv"
)
DEFAULT_OUTPUT_STEM = Path(
    "HJEEDS/results/hjeeds_paper_500_seeds/final_paper_plots/"
    "00_baseline_error_by_count_bucket"
)

EXECUTION_METRIC = "abs_sigma_error"
DECISION_METRIC = "abs_rationality_percent_error"
METRICS = (EXECUTION_METRIC, DECISION_METRIC)
RAW_DECISION_METRIC = "abs_log_lambda_error"
CANONICAL_SUMMARY_METRICS = frozenset((*METRICS, RAW_DECISION_METRIC))
CANONICAL_METHODS = frozenset(("jeeds", "hierarchical"))
CANONICAL_BUCKETS = (5, 10, 25, 100, 1000)
PAPER_FIRST_SEED = 12345
PAPER_NUM_SEEDS = 500
PAPER_AGENTS_PER_BUCKET = 5
REQUIRED_SUMMARY_COLUMNS = frozenset(
    ("method", "metric", "count_bucket", "num_agents", "mean", "ci_lower", "ci_upper")
)

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
    parser.add_argument(
        "--expected-first-seed",
        type=int,
        default=PAPER_FIRST_SEED,
        help="First required seed in the linked agent-level paper run (default: 12345).",
    )
    parser.add_argument(
        "--expected-num-seeds",
        type=int,
        default=PAPER_NUM_SEEDS,
        help="Exact required seed count in the linked agent-level paper run (default: 500).",
    )
    parser.add_argument(
        "--agent-level-csv",
        type=Path,
        help=(
            "Agent-level results used to verify that the summary is current and complete. "
            "Defaults to agent_level_results.csv beside the summary."
        ),
    )
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--dpi", type=int, default=450)
    return parser.parse_args(argv)


def _required_finite_float(value: Any, *, field: str, context: str) -> float:
    """Parse one required finite numeric field with an actionable error."""

    if value in (None, ""):
        raise ValueError(f"Missing {field} in {context}.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field}={value!r} in {context}.") from error
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite {field}={value!r} in {context}.")
    return parsed


def _required_int(value: Any, *, field: str, context: str) -> int:
    """Parse an integer without silently truncating fractional values."""

    numeric = _required_finite_float(value, field=field, context=context)
    parsed = int(numeric)
    if float(parsed) != numeric:
        raise ValueError(f"Expected integer {field}, received {value!r} in {context}.")
    return parsed


def read_summary_rows(summary_csv: Path) -> list[dict[str, Any]]:
    """Read a complete canonical baseline summary and return its plotted rows."""

    all_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, int]] = set()
    with summary_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing_columns = REQUIRED_SUMMARY_COLUMNS - columns
        if missing_columns:
            raise ValueError(
                f"{summary_csv} is missing required columns: {sorted(missing_columns)}."
            )
        for row_number, source_row in enumerate(reader, start=2):
            context = f"{summary_csv}:{row_number}"
            method = str(source_row.get("method", ""))
            metric = str(source_row.get("metric", ""))
            bucket = _required_int(
                source_row.get("count_bucket"), field="count_bucket", context=context
            )
            num_agents = _required_int(
                source_row.get("num_agents"), field="num_agents", context=context
            )
            mean = _required_finite_float(source_row.get("mean"), field="mean", context=context)
            ci_lower = _required_finite_float(
                source_row.get("ci_lower"), field="ci_lower", context=context
            )
            ci_upper = _required_finite_float(
                source_row.get("ci_upper"), field="ci_upper", context=context
            )
            if num_agents <= 0:
                raise ValueError(f"Expected positive num_agents in {context}.")
            if mean < 0 or ci_lower < 0 or ci_upper < 0:
                raise ValueError(f"Absolute-error summary contains a negative value in {context}.")
            if ci_lower > mean or mean > ci_upper:
                raise ValueError(
                    f"Expected ci_lower <= mean <= ci_upper in {context}; received "
                    f"{ci_lower}, {mean}, {ci_upper}."
                )
            key = (method, metric, bucket)
            if key in seen_keys:
                raise ValueError(f"Duplicate baseline summary row for {key!r} in {summary_csv}.")
            seen_keys.add(key)
            all_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "count_bucket": bucket,
                    "num_agents": num_agents,
                    "mean": mean,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
            )

    observed_methods = {str(row["method"]) for row in all_rows}
    observed_metrics = {str(row["metric"]) for row in all_rows}
    observed_buckets = {int(row["count_bucket"]) for row in all_rows}
    if observed_methods != CANONICAL_METHODS:
        raise ValueError(
            f"Baseline methods are {sorted(observed_methods)}; expected "
            f"{sorted(CANONICAL_METHODS)}."
        )
    if observed_metrics != CANONICAL_SUMMARY_METRICS:
        raise ValueError(
            f"Baseline metrics are {sorted(observed_metrics)}; expected "
            f"{sorted(CANONICAL_SUMMARY_METRICS)}."
        )
    if observed_buckets != set(CANONICAL_BUCKETS):
        raise ValueError(
            f"Baseline observation buckets are {sorted(observed_buckets)}; expected "
            f"{list(CANONICAL_BUCKETS)}."
        )
    expected_keys = {
        (method, metric, bucket)
        for method in CANONICAL_METHODS
        for metric in CANONICAL_SUMMARY_METRICS
        for bucket in CANONICAL_BUCKETS
    }
    if seen_keys != expected_keys:
        missing = sorted(expected_keys - seen_keys)
        unexpected = sorted(seen_keys - expected_keys)
        raise ValueError(
            "Baseline summary is partial or noncanonical: "
            f"missing={missing}, unexpected={unexpected}."
        )
    expected_num_agents = {int(row["num_agents"]) for row in all_rows}
    if len(expected_num_agents) != 1:
        raise ValueError(
            "Baseline summary has inconsistent num_agents across canonical cells: "
            f"{sorted(expected_num_agents)}."
        )
    plotted_rows = [row for row in all_rows if str(row["metric"]) in METRICS]
    _require_complete_series(plotted_rows, summary_csv)
    return plotted_rows


def _mean_and_ci(seed_means: Sequence[float]) -> tuple[float, float, float]:
    """Reproduce the across-seed aggregation used by the experiment pipeline."""

    values = np.asarray(seed_means, dtype=float)
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, mean, mean
    half_width = 1.96 * float(np.std(values, ddof=1)) / math.sqrt(values.size)
    return mean, max(0.0, mean - half_width), mean + half_width


def validate_summary_against_agent_results(
    rows: Sequence[dict[str, Any]],
    agent_level_csv: Path,
    *,
    expected_first_seed: int = PAPER_FIRST_SEED,
    expected_num_seeds: int = PAPER_NUM_SEEDS,
    expected_agents_per_bucket: int = PAPER_AGENTS_PER_BUCKET,
) -> None:
    """Reject a stale summary by recomputing every plotted cell from agent rows."""

    if expected_num_seeds <= 0 or expected_agents_per_bucket <= 0:
        raise ValueError("Expected seed and per-bucket agent counts must be positive.")
    expected_seeds = set(
        range(expected_first_seed, expected_first_seed + expected_num_seeds)
    )
    try:
        agent_validation = validate_agent_csv(
            agent_level_csv,
            expected_seeds,
            expected_environment="1d",
        )
    except ValueError as error:
        if "seed mismatch" in str(error):
            raise ValueError(
                "Baseline agent-level seed coverage is partial or noncanonical; "
                f"expected {expected_first_seed}.."
                f"{expected_first_seed + expected_num_seeds - 1}."
            ) from error
        raise
    expected_agents_per_seed = expected_agents_per_bucket * len(CANONICAL_BUCKETS)
    if agent_validation.agents_per_seed != expected_agents_per_seed:
        raise ValueError(
            f"{agent_level_csv}: expected {expected_agents_per_seed} agents per seed, "
            f"found {agent_validation.agents_per_seed}."
        )

    summary_by_key = {
        (str(row["method"]), str(row["metric"]), int(row["count_bucket"])): row
        for row in rows
    }
    values_by_key_seed: dict[tuple[str, str, int], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen_agents: set[tuple[int, int]] = set()
    agents_by_seed_bucket: dict[tuple[int, int], int] = defaultdict(int)
    required_columns = {
        "seed",
        "environment",
        "agent_id",
        "count_bucket",
        "sigma_true",
        "rationality_percent_true",
        *(f"{method}_posterior_mean_sigma" for method in CANONICAL_METHODS),
        *(f"{method}_rationality_percent" for method in CANONICAL_METHODS),
        *(f"{method}_status" for method in CANONICAL_METHODS),
    }
    with agent_level_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = required_columns - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                f"{agent_level_csv} is missing required columns: {sorted(missing_columns)}."
            )
        for row_number, source_row in enumerate(reader, start=2):
            context = f"{agent_level_csv}:{row_number}"
            if source_row.get("environment") != "1d":
                raise ValueError(f"Expected environment='1d' in {context}.")
            seed = _required_int(source_row.get("seed"), field="seed", context=context)
            agent_id = _required_int(
                source_row.get("agent_id"), field="agent_id", context=context
            )
            agent_key = (seed, agent_id)
            if agent_key in seen_agents:
                raise ValueError(f"Duplicate agent row {agent_key!r} in {agent_level_csv}.")
            seen_agents.add(agent_key)
            bucket = _required_int(
                source_row.get("count_bucket"), field="count_bucket", context=context
            )
            if bucket not in CANONICAL_BUCKETS:
                raise ValueError(f"Noncanonical baseline bucket {bucket} in {context}.")
            agents_by_seed_bucket[(seed, bucket)] += 1
            sigma_true = _required_finite_float(
                source_row.get("sigma_true"), field="sigma_true", context=context
            )
            rationality_true = _required_finite_float(
                source_row.get("rationality_percent_true"),
                field="rationality_percent_true",
                context=context,
            )
            for method in CANONICAL_METHODS:
                if source_row.get(f"{method}_status") != "ok":
                    raise ValueError(f"Non-ok {method} estimate in {context}.")
                sigma_estimate = _required_finite_float(
                    source_row.get(f"{method}_posterior_mean_sigma"),
                    field=f"{method}_posterior_mean_sigma",
                    context=context,
                )
                rationality_estimate = _required_finite_float(
                    source_row.get(f"{method}_rationality_percent"),
                    field=f"{method}_rationality_percent",
                    context=context,
                )
                values_by_key_seed[(method, EXECUTION_METRIC, bucket)][seed].append(
                    abs(sigma_estimate - sigma_true)
                )
                values_by_key_seed[(method, DECISION_METRIC, bucket)][seed].append(
                    abs(rationality_estimate - rationality_true)
                )

    if not seen_agents:
        raise ValueError(f"No agent rows found in {agent_level_csv}.")
    observed_seeds = {seed for seed, _agent_id in seen_agents}
    if observed_seeds != expected_seeds:
        raise ValueError(
            "Baseline agent-level seed coverage is partial or noncanonical: "
            f"expected {expected_first_seed}..{expected_first_seed + expected_num_seeds - 1}, "
            f"found {len(observed_seeds)} seed(s)."
        )
    incorrect_cells = {
        (seed, bucket): agents_by_seed_bucket.get((seed, bucket), 0)
        for seed in sorted(expected_seeds)
        for bucket in CANONICAL_BUCKETS
        if agents_by_seed_bucket.get((seed, bucket), 0) != expected_agents_per_bucket
    }
    if incorrect_cells:
        preview = list(incorrect_cells.items())[:10]
        raise ValueError(
            "Baseline agent-level seed/bucket coverage is incomplete; expected "
            f"{expected_agents_per_bucket} agents in every cell, examples={preview}."
        )
    if set(values_by_key_seed) != set(summary_by_key):
        raise ValueError(
            "Agent-level results do not cover the exact plotted method/metric/bucket cells."
        )
    for key, per_seed in values_by_key_seed.items():
        summary = summary_by_key[key]
        num_agents = sum(len(values) for values in per_seed.values())
        seed_means = [float(np.mean(per_seed[seed])) for seed in sorted(per_seed)]
        expected_mean, expected_lower, expected_upper = _mean_and_ci(seed_means)
        comparisons = (
            ("num_agents", int(summary["num_agents"]), num_agents),
            ("mean", float(summary["mean"]), expected_mean),
            ("ci_lower", float(summary["ci_lower"]), expected_lower),
            ("ci_upper", float(summary["ci_upper"]), expected_upper),
        )
        for field, actual, expected in comparisons:
            if field == "num_agents":
                matches = actual == expected
            else:
                matches = bool(np.isclose(actual, expected, rtol=1e-12, atol=1e-12))
            if not matches:
                raise ValueError(
                    f"Stale baseline summary for {key!r}: {field}={actual!r}, "
                    f"recomputed value={expected!r} from {agent_level_csv}."
                )


def _require_complete_series(rows: Sequence[dict[str, Any]], summary_csv: Path) -> None:
    """Fail loudly when any method/metric/bucket cell the figure needs is absent.

    Rows with unusable numeric fields are skipped while reading, so this check is
    what stops a partially populated table from rendering a figure whose traces
    silently cover different observation counts.
    """

    buckets = sorted({int(row["count_bucket"]) for row in rows})
    present = {
        (str(row["method"]), str(row["metric"]), int(row["count_bucket"]))
        for row in rows
    }
    missing = [
        f"{method}/{metric}/N={bucket}"
        for method, metric, bucket in (
            (method, metric, bucket)
            for method in ("jeeds", "hierarchical")
            for metric in METRICS
            for bucket in buckets
        )
        if (method, metric, bucket) not in present
    ]
    if missing:
        raise ValueError(
            f"Incomplete baseline paper metrics in {summary_csv}. "
            f"Missing rows: {', '.join(missing)}."
        )


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
    # Keep the native AAAI column width while trimming vertical space that the
    # shared legend and five-point traces do not need.
    figure, execution_axis = plt.subplots(figsize=(3.35, 2.38))
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
    figure.subplots_adjust(left=0.195, right=0.805, top=0.79, bottom=0.20)
    save_figure_bundle(figure, output_stem, dpi)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""

    args = parse_args(argv)
    rows = read_summary_rows(args.summary_by_bucket_csv)
    agent_level_csv = (
        args.agent_level_csv
        if args.agent_level_csv is not None
        else args.summary_by_bucket_csv.with_name("agent_level_results.csv")
    )
    validate_summary_against_agent_results(
        rows,
        agent_level_csv,
        expected_first_seed=args.expected_first_seed,
        expected_num_seeds=args.expected_num_seeds,
    )
    render(rows, args.output_stem, args.dpi)
    print(
        f"Wrote baseline paper figure to {args.output_stem.with_suffix('.png')} "
        f"from validated inputs {args.summary_by_bucket_csv} and {agent_level_csv}"
    )


if __name__ == "__main__":
    main()
