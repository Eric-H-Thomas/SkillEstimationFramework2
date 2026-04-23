# This file still requires human verification. Delete this comment when done.
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .models import SeedResult


# Aggregation happens in two layers:
# 1. summarize one seed into compact bucket-level and overall rows, and
# 2. aggregate those seed-level rows across random seeds to get final tables.
# This separation matters because the code treats seeds, not agents, as the
# unit of replication when computing confidence intervals.


def summarize_seed_results(seed_result: SeedResult) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize one seed's agent-level outputs into bucketed and overall rows.

    We summarize posterior-mean estimates for every method that completed with
    ``status == "ok"``. MAP estimates remain in the agent-level CSV as
    diagnostics, but posterior means are the primary estimates for the paper.
    """

    def add_metric(
        container: dict[tuple[str, str, int], list[float]],
        *,
        method: str,
        metric: str,
        count_bucket: int,
        value: float,
    ) -> None:
        """Accumulate one metric value under a method/metric/bucket key."""

        key = (method, metric, count_bucket)
        if key not in container:
            container[key] = []
        container[key].append(value)

    # Separate dictionaries keep the bucketed and overall summaries distinct.
    # The bucketed tables answer "how do methods compare at each data volume?"
    # while the overall tables answer "how do methods compare across the whole
    # experiment regardless of bucket?"
    bucket_metrics: dict[tuple[str, str, int], list[float]] = {}
    overall_metrics: dict[tuple[str, str], list[float]] = {}

    for result in seed_result.agent_results:
        method_info = {
            "jeeds": result.jeeds,
            "hierarchical": result.hierarchical,
        }

        for method_name, estimate in method_info.items():
            # Only summarize methods that produced real posterior-mean
            # estimates. Failed methods keep their status in the agent-level
            # CSV, but they should not contribute numeric error rows.
            if estimate.status != "ok":
                continue
            if estimate.posterior_mean_sigma is None or estimate.posterior_mean_log_lambda is None:
                continue

            # This experiment reports execution skill error on the original
            # sigma scale. Decision skill is summarized canonically in
            # natural-log lambda space, so the error metric compares those log
            # values directly instead of converting to base-10.
            abs_sigma_error = abs(estimate.posterior_mean_sigma - result.sigma_true)
            abs_log_lambda_error = abs(estimate.posterior_mean_log_lambda - result.log_lambda_true)

            add_metric(
                bucket_metrics,
                method=method_name,
                metric="abs_sigma_error",
                count_bucket=result.count_bucket,
                value=abs_sigma_error,
            )
            add_metric(
                bucket_metrics,
                method=method_name,
                metric="abs_log_lambda_error",
                count_bucket=result.count_bucket,
                value=abs_log_lambda_error,
            )

            overall_sigma_key = (method_name, "abs_sigma_error")
            overall_lambda_key = (method_name, "abs_log_lambda_error")
            overall_metrics.setdefault(overall_sigma_key, []).append(abs_sigma_error)
            overall_metrics.setdefault(overall_lambda_key, []).append(abs_log_lambda_error)

    summary_by_bucket_rows: list[dict[str, Any]] = []
    for (method_name, metric_name, count_bucket), values in sorted(bucket_metrics.items()):
        summary_by_bucket_rows.append(
            {
                "method": method_name,
                "metric": metric_name,
                "count_bucket": count_bucket,
                "num_agents": len(values),
                "mean": float(np.mean(values)),
                "ci_lower": "",
                "ci_upper": "",
                "notes": (
                    "Seed-level mean over agents with valid posterior-mean estimates. "
                    "Confidence intervals are added during across-seed aggregation."
                ),
            }
        )

    summary_overall_rows: list[dict[str, Any]] = []
    for (method_name, metric_name), values in sorted(overall_metrics.items()):
        summary_overall_rows.append(
            {
                "method": method_name,
                "metric": metric_name,
                "num_agents": len(values),
                "mean": float(np.mean(values)),
                "ci_lower": "",
                "ci_upper": "",
                "notes": (
                    "Seed-level overall mean over agents with valid posterior-mean estimates. "
                    "Confidence intervals are added during across-seed aggregation."
                ),
            }
        )

    return summary_by_bucket_rows, summary_overall_rows


def aggregate_results_across_seeds(
    seed_results: Sequence[SeedResult],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate seed-level summaries into final experiment tables.

    Combine per-seed summary rows by averaging the seed-level means, then add
    normal-approximation 95% confidence intervals across seeds.
    """

    def mean_confidence_interval(
        values: Sequence[float],
    ) -> tuple[float, float]:
        """Return a closed-form 95% CI for the mean of ``values``.

        We treat seeds as the unit of replication and compute the CI from the
        standard error of the seed-level means. With the number of seeds we
        expect to run, the normal approximation should be reasonable.
        """

        values_array = np.asarray(values, dtype=float)
        if values_array.size == 0:
            raise ValueError("Cannot compute a confidence interval from an empty set of values.")
        if values_array.size == 1:
            scalar_value = float(values_array[0])
            return scalar_value, scalar_value

        mean_value = float(np.mean(values_array))
        sample_std = float(np.std(values_array, ddof=1))
        standard_error = sample_std / math.sqrt(values_array.size)
        half_width = 1.96 * standard_error
        return mean_value - half_width, mean_value + half_width

    def clamp_ci_to_metric_support(metric_name: str, ci_lower: float, ci_upper: float) -> tuple[float, float]:
        """Keep reported CIs inside the known support of the metric."""

        if metric_name.startswith("abs_"):
            ci_lower = max(0.0, ci_lower)
        return ci_lower, ci_upper

    if not seed_results:
        return [], []

    # These groups collect one mean per seed.  We intentionally aggregate
    # seed-level means rather than pooling all agents together, because the
    # experiment is replicated across seeds.
    bucket_groups: dict[tuple[str, str, int], dict[str, Any]] = {}
    overall_groups: dict[tuple[str, str], dict[str, Any]] = {}

    for seed_result in seed_results:
        for row in seed_result.summary_by_bucket_rows:
            key = (str(row["method"]), str(row["metric"]), int(row["count_bucket"]))
            if key not in bucket_groups:
                bucket_groups[key] = {
                    "means": [],
                    "num_agents": 0,
                    "num_seeds": 0,
                }
            bucket_groups[key]["means"].append(float(row["mean"]))
            bucket_groups[key]["num_agents"] += int(row["num_agents"])
            bucket_groups[key]["num_seeds"] += 1

        for row in seed_result.summary_overall_rows:
            key = (str(row["method"]), str(row["metric"]))
            if key not in overall_groups:
                overall_groups[key] = {
                    "means": [],
                    "num_agents": 0,
                    "num_seeds": 0,
                }
            overall_groups[key]["means"].append(float(row["mean"]))
            overall_groups[key]["num_agents"] += int(row["num_agents"])
            overall_groups[key]["num_seeds"] += 1

    summary_by_bucket_rows: list[dict[str, Any]] = []
    for (method_name, metric_name, count_bucket), info in sorted(bucket_groups.items()):
        mean_values = info["means"]
        ci_lower, ci_upper = mean_confidence_interval(mean_values)
        ci_lower, ci_upper = clamp_ci_to_metric_support(metric_name, ci_lower, ci_upper)
        summary_by_bucket_rows.append(
            {
                "method": method_name,
                "metric": metric_name,
                "count_bucket": count_bucket,
                "num_agents": info["num_agents"],
                "mean": float(np.mean(mean_values)),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "notes": (
                    "Across-seed mean of seed-level summary means with a normal-approximation "
                    f"95% CI over {info['num_seeds']} seeds."
                ),
            }
        )

    summary_overall_rows: list[dict[str, Any]] = []
    for (method_name, metric_name), info in sorted(overall_groups.items()):
        mean_values = info["means"]
        ci_lower, ci_upper = mean_confidence_interval(mean_values)
        ci_lower, ci_upper = clamp_ci_to_metric_support(metric_name, ci_lower, ci_upper)
        summary_overall_rows.append(
            {
                "method": method_name,
                "metric": metric_name,
                "num_agents": info["num_agents"],
                "mean": float(np.mean(mean_values)),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "notes": (
                    "Across-seed mean of seed-level summary means with a normal-approximation "
                    f"95% CI over {info['num_seeds']} seeds."
                ),
            }
        )

    return summary_by_bucket_rows, summary_overall_rows
