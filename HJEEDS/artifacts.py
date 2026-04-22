# This file still requires human verification. Delete this comment when done.
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import (
    AGENT_LEVEL_CSV_HEADER,
    SUMMARY_BY_BUCKET_CSV_HEADER,
    SUMMARY_OVERALL_CSV_HEADER,
    planned_output_paths,
)
from .models import AgentResult


# This module takes already-computed results and turns them into the artifacts a
# human reviewer or paper workflow will inspect: flat CSVs and a compact error
# figure.  There is no modeling here, only formatting and presentation.


def _value_or_blank(value: Any) -> Any:
    """Return CSV-friendly blanks for missing estimate values."""

    return "" if value is None else value


def _agent_result_to_row(result: AgentResult) -> dict[str, Any]:
    """Flatten an ``AgentResult`` into the agent-level CSV schema."""

    # Flatten the nested dataclass structure into the exact CSV column layout.
    # Using one helper avoids duplicating this mapping in multiple writers.
    return {
        "seed": result.seed,
        "agent_id": result.agent_id,
        "count_bucket": result.count_bucket,
        "num_observations": result.num_observations,
        "sigma_true": result.sigma_true,
        "lambda_true": result.lambda_true,
        "jeeds_posterior_mean_sigma": _value_or_blank(result.jeeds.posterior_mean_sigma),
        "jeeds_posterior_mean_lambda": _value_or_blank(result.jeeds.posterior_mean_lambda),
        "jeeds_map_sigma": _value_or_blank(result.jeeds.map_sigma),
        "jeeds_map_lambda": _value_or_blank(result.jeeds.map_lambda),
        "jeeds_status": result.jeeds.status,
        "hierarchical_posterior_mean_sigma": _value_or_blank(result.hierarchical.posterior_mean_sigma),
        "hierarchical_posterior_mean_lambda": _value_or_blank(result.hierarchical.posterior_mean_lambda),
        "hierarchical_map_sigma": _value_or_blank(result.hierarchical.map_sigma),
        "hierarchical_map_lambda": _value_or_blank(result.hierarchical.map_lambda),
        "hierarchical_status": result.hierarchical.status,
        "notes": result.notes,
    }


def write_agent_level_csv(output_path: Path, agent_results: Sequence[AgentResult]) -> None:
    """Write the agent-level CSV schema."""

    # Writers create parent directories on demand so the caller only needs to
    # decide the output root once.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGENT_LEVEL_CSV_HEADER)
        writer.writeheader()
        for result in agent_results:
            writer.writerow(_agent_result_to_row(result))


def _write_dict_rows(output_path: Path, header: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    """Internal helper for writing summary CSVs with a fixed schema."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in header})


def write_summary_csvs(
    output_dir: Path,
    summary_by_bucket_rows: Sequence[dict[str, Any]],
    summary_overall_rows: Sequence[dict[str, Any]],
) -> None:
    """Write the bucketed and overall summary CSV schemas."""

    paths = planned_output_paths(output_dir)
    _write_dict_rows(paths["summary_by_bucket_csv"], SUMMARY_BY_BUCKET_CSV_HEADER, summary_by_bucket_rows)
    _write_dict_rows(paths["summary_overall_csv"], SUMMARY_OVERALL_CSV_HEADER, summary_overall_rows)


def plot_error_by_bucket(output_path: Path, summary_by_bucket_rows: Sequence[dict[str, Any]]) -> None:
    """Create the two-panel figure for sigma and log-lambda error.

    The input rows are the across-seed summaries, so the figure plots the
    reported mean and 95% CI for each method at each observation-count bucket.
    """

    # Import matplotlib lazily so dry-run and non-plotting helper tests do not
    # need to import the plotting stack. Force a non-interactive backend so the
    # same script works on headless SLURM nodes.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    def as_float(value: Any) -> float | None:
        """Convert CSV-ish values to floats, treating blanks as missing."""

        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def as_bucket(value: Any) -> int | None:
        """Convert a summary row's bucket value to an integer label."""

        numeric_value = as_float(value)
        if numeric_value is None:
            return None
        return int(numeric_value)

    metric_panels = [
        (
            "abs_sigma_error",
            "Execution Skill Error by Count Bucket",
            r"|$\hat{\sigma} - \sigma$|",
        ),
        (
            "abs_log10_lambda_error",
            "Decision Skill Error by Count Bucket",
            r"|$\log_{10}\hat{\lambda} - \log_{10}\lambda$|",
        ),
    ]
    method_order = {
        "jeeds": 0,
        "hierarchical": 1,
    }

    # Parse rows once into a consistent internal representation before plotting.
    parsed_rows: list[dict[str, Any]] = []
    for row in summary_by_bucket_rows:
        bucket = as_bucket(row.get("count_bucket"))
        mean = as_float(row.get("mean"))
        if bucket is None or mean is None:
            continue
        parsed_rows.append(
            {
                "method": str(row.get("method", "")),
                "metric": str(row.get("metric", "")),
                "count_bucket": bucket,
                "mean": mean,
                "ci_lower": as_float(row.get("ci_lower")),
                "ci_upper": as_float(row.get("ci_upper")),
            }
        )

    bucket_values = sorted({row["count_bucket"] for row in parsed_rows})
    bucket_positions = {bucket: index for index, bucket in enumerate(bucket_values)}

    # The two panels mirror the two headline error metrics used in the paper.
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for axis, (metric_name, title, ylabel) in zip(axes, metric_panels):
        axis.set_title(title)
        axis.set_xlabel("Observation count bucket")
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        metric_rows = [row for row in parsed_rows if row["metric"] == metric_name]
        methods = sorted(
            {row["method"] for row in metric_rows},
            key=lambda method: (method_order.get(method, len(method_order)), method),
        )

        if metric_rows and bucket_values:
            for method in methods:
                method_rows = sorted(
                    [row for row in metric_rows if row["method"] == method],
                    key=lambda row: bucket_positions[row["count_bucket"]],
                )

                x_values: list[int] = []
                y_values: list[float] = []
                lower_errors: list[float] = []
                upper_errors: list[float] = []

                for row in method_rows:
                    mean = row["mean"]
                    ci_lower = row["ci_lower"]
                    ci_upper = row["ci_upper"]

                    x_values.append(bucket_positions[row["count_bucket"]])
                    y_values.append(mean)
                    lower_errors.append(max(0.0, mean - ci_lower) if ci_lower is not None else 0.0)
                    upper_errors.append(max(0.0, ci_upper - mean) if ci_upper is not None else 0.0)

                # Error bars visualize the across-seed uncertainty that was
                # computed during aggregation.
                axis.errorbar(
                    x_values,
                    y_values,
                    yerr=np.array([lower_errors, upper_errors], dtype=float),
                    marker="o",
                    capsize=4,
                    linewidth=2,
                    label=method,
                )

            axis.set_xticks(list(bucket_positions.values()))
            axis.set_xticklabels([str(bucket) for bucket in bucket_values])
            axis.legend(title="Method")
        else:
            axis.text(
                0.5,
                0.5,
                f"No rows for {metric_name}",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

    # Save directly to disk rather than showing interactively because this code
    # is also used in headless environments such as Slurm jobs.
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
