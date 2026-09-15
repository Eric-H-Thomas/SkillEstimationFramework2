# Paper correspondence: Main `subsec:baseball`; Supplement `app:baseball_sigma_gap`.
"""CLI entry point for Statcast baseball convergence (Phase 2 / paper BBIP).

Public CLI over ``baseball_convergence``: argparse, CSV/plot writers, and
Slurm mode dispatch (``--prepare-roster`` / ``--agent-index`` /
``--aggregate-results``). Estimation lives in ``baseball_convergence``;
this module does not change likelihoods or roster selection.

Paper BBIP: ``submit_hjeeds_baseball_convergence_paper_bbip.sh`` →
``submit_hjeeds_baseball_convergence_array.sh`` → this module with
``--bbip-extremes 10``, ``--season-year 2021``, ``--pitch-types FF``,
``min_pitches_per_agent=100``, ``convergence_ns=5,10,25,50,100``,
``max_reference_pitches=100``,
``--hyperprior-preset baseball-literature-informed``.

Drift is self-reference (JEEDS→JEEDS@N_max, H-JEEDS→H-JEEDS@N_max), not
ground truth. Statcast estimates are deterministic in seed. When roster
metadata includes ``bbip_selection``, aggregate / ``--plot-only`` also write
BBIP separability plots beside the drift figures.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HJEEDS.artifacts import (
    _abs_difference_or_blank,
    _value_or_blank,
    _write_dict_rows,
    error_metric_figure_size,
)
from HJEEDS.baseball_config import (
    DEFAULT_LAMBDA_MAX,
    DEFAULT_LAMBDA_MIN,
    DEFAULT_NUM_LAMBDA_GRID,
    DEFAULT_NUM_SIGMA_GRID,
    DEFAULT_PITCH_TYPES,
)
from HJEEDS.baseball_convergence import (
    CONVERGENCE_ROSTER_METADATA_FILENAME,
    DEFAULT_CONVERGENCE_NS,
    DEFAULT_OUTPUT_DIR_CONVERGENCE,
    DEFAULT_PITCHER_IDS,
    aggregate_convergence_across_seeds,
    aggregate_convergence_results,
    begin_convergence_run_metadata,
    build_baseball_convergence_config_from_args,
    build_convergence_run_provenance,
    build_drift_summary_tables,
    finalize_convergence_run_metadata,
    load_convergence_roster,
    load_convergence_roster_metadata,
    planned_convergence_output_paths,
    prepare_convergence_roster,
    print_baseball_convergence_dry_run_summary,
    required_min_pitches_for_convergence,
    restamp_convergence_run_incomplete,
    run_convergence_agent_index,
    run_single_baseball_convergence_seed,
    validate_complete_convergence_run_metadata,
)
from HJEEDS.baseball_provenance import (
    build_baseball_run_provenance,
    require_matching_fingerprint,
    roster_fingerprint,
)
from HJEEDS.baseball_pitch import DEFAULT_DELTA, DEFAULT_EXECUTION_SKILL_MAX, DEFAULT_EXECUTION_SKILL_MIN
from HJEEDS.baseball_plot_style import BASEBALL_METHOD_STYLES
from HJEEDS.baseball_roster import (
    add_common_roster_arguments,
    add_hyperprior_arguments,
    parse_pitch_types,
    print_eligible_agents,
)
from HJEEDS.config import (
    DEFAULT_SEED,
    SUMMARY_BY_BUCKET_CSV_HEADER,
    SUMMARY_OVERALL_CSV_HEADER,
    _parse_count_buckets,
    add_require_paper_config_argument,
    parse_seed_argument,
)
from HJEEDS.models import StatcastConvergenceAgentResult

CONVERGENCE_AGENT_LEVEL_HEADER = [
    "seed",
    "environment",
    "agent_id",
    "pitcher_id",
    "pitch_type",
    "convergence_n",
    "num_observations",
    "num_reference_observations",
    "reference_posterior_mean_sigma",
    "reference_posterior_mean_log_lambda",
    "reference_status",
    "jeeds_posterior_mean_sigma",
    "jeeds_posterior_mean_log_lambda",
    "jeeds_abs_sigma_drift_vs_full",
    "jeeds_abs_log_lambda_drift_vs_full",
    "jeeds_status",
    "hierarchical_posterior_mean_sigma",
    "hierarchical_posterior_mean_log_lambda",
    "hierarchical_abs_sigma_drift_vs_full",
    "hierarchical_abs_log_lambda_drift_vs_full",
    "hierarchical_closer_sigma",
    "hierarchical_closer_log_lambda",
    "hierarchical_status",
    "notes",
]

DRIFT_METRIC_PANELS = (
    (
        "abs_sigma_drift_vs_full",
        r"Execution skill ($\hat{\sigma}$)",
        r"$|\hat{\sigma}_c - \hat{\sigma}_{c_{\max}}|$",
        "No rows for execution skill drift",
    ),
    (
        "abs_log_lambda_drift_vs_full",
        r"Decision skill ($\widehat{\log\lambda}$)",
        r"$|\widehat{\log\lambda}_c - \widehat{\log\lambda}_{c_{\max}}|$",
        "No rows for log decision skill drift",
    ),
)


def parse_convergence_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convergence study on Statcast baseball: compare JEEDS and HJEEDS at "
            "reduced pitch counts, each against that method's own estimate at max N."
        )
    )
    parser.add_argument(
        "--seed",
        type=parse_seed_argument,
        required=False,
        default=None,
        help=(
            "Base seed (or 'default' for 12345). Kept for CLI compatibility with darts; "
            "Statcast likelihoods are deterministic in seed, so varying it does not change estimates."
        ),
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help=(
            "Number of consecutive seeds to run (default: 1). For baseball, keep at 1: "
            "additional seeds repeat the same numerics and do not produce uncertainty bands."
        ),
    )
    parser.add_argument(
        "--convergence-ns",
        type=str,
        default=",".join(str(value) for value in DEFAULT_CONVERGENCE_NS),
        help="Comma-separated cumulative pitch-count checkpoints (newest-first per agent).",
    )
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--num-sigma-grid", type=int, default=DEFAULT_NUM_SIGMA_GRID)
    parser.add_argument("--num-lambda-grid", type=int, default=DEFAULT_NUM_LAMBDA_GRID)
    parser.add_argument("--sigma-min", type=float, default=DEFAULT_EXECUTION_SKILL_MIN)
    parser.add_argument("--sigma-max", type=float, default=DEFAULT_EXECUTION_SKILL_MAX)
    parser.add_argument("--lambda-min", type=float, default=DEFAULT_LAMBDA_MIN)
    parser.add_argument("--lambda-max", type=float, default=DEFAULT_LAMBDA_MAX)
    parser.add_argument("--pitcher-ids", type=str, default=",".join(str(pid) for pid in DEFAULT_PITCHER_IDS))
    parser.add_argument("--pitch-types", type=str, default=",".join(DEFAULT_PITCH_TYPES))
    add_common_roster_arguments(parser)
    add_hyperprior_arguments(parser)
    parser.add_argument(
        "--max-reference-pitches",
        type=int,
        default=None,
        help="Cap full-data reference pitches per agent (smoke tests). Default: all available.",
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR_CONVERGENCE))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help=(
            "Regenerate summary CSVs and drift plots from an existing agent-level CSV. "
            "Also regenerates walk/IP-proxy separability when bbip_selection is present."
        ),
    )
    parser.add_argument(
        "--plot-agent-estimates",
        action="store_true",
        help=(
            "Write per-agent intermediate-estimate PNGs under output_dir/agents/ "
            "from an existing convergence_agent_level_results.csv."
        ),
    )
    parser.add_argument(
        "--prepare-roster",
        action="store_true",
        help="Write convergence_roster.json and exit (no RNN inference).",
    )
    parser.add_argument(
        "--agent-index",
        type=int,
        default=None,
        help="Build per-agent convergence cache for one roster agent (0-based Slurm array index).",
    )
    parser.add_argument(
        "--aggregate-results",
        action="store_true",
        help=(
            "Combine per-agent caches and write convergence CSVs and drift_by_N.png. "
            "When roster metadata includes bbip_selection, also write separability plots."
        ),
    )
    parser.add_argument(
        "--use-prepared-roster",
        action="store_true",
        help="Load convergence_roster.json from --output-dir instead of re-resolving roster selectors.",
    )
    parser.add_argument(
        "--allow-partial-caches",
        action="store_true",
        help=(
            "NON-PAPER ONLY: aggregate when prepared-roster agent caches are missing. "
            "The publication path fails on an incomplete cohort."
        ),
    )
    parser.add_argument(
        "--allow-separability-failure",
        action="store_true",
        help=(
            "NON-PAPER ONLY: keep convergence outputs if walk/IP-proxy separability fails. "
            "The publication path treats separability as required."
        ),
    )
    add_require_paper_config_argument(parser)
    return parser.parse_args(argv)


def _convergence_result_to_row(result: StatcastConvergenceAgentResult) -> dict[str, Any]:
    return {
        "seed": result.seed,
        "environment": "baseball",
        "agent_id": result.agent_id,
        "pitcher_id": result.pitcher_id,
        "pitch_type": result.pitch_type,
        "convergence_n": result.convergence_n,
        "num_observations": result.num_observations,
        "num_reference_observations": result.num_reference_observations,
        "reference_posterior_mean_sigma": _value_or_blank(result.reference.posterior_mean_sigma),
        "reference_posterior_mean_log_lambda": _value_or_blank(result.reference.posterior_mean_log_lambda),
        "reference_status": result.reference.status,
        "jeeds_posterior_mean_sigma": _value_or_blank(result.jeeds.posterior_mean_sigma),
        "jeeds_posterior_mean_log_lambda": _value_or_blank(result.jeeds.posterior_mean_log_lambda),
        "jeeds_abs_sigma_drift_vs_full": _value_or_blank(result.abs_sigma_drift_vs_full_jeeds),
        "jeeds_abs_log_lambda_drift_vs_full": _value_or_blank(result.abs_log_lambda_drift_vs_full_jeeds),
        "jeeds_status": result.jeeds.status,
        "hierarchical_posterior_mean_sigma": _value_or_blank(result.hierarchical.posterior_mean_sigma),
        "hierarchical_posterior_mean_log_lambda": _value_or_blank(
            result.hierarchical.posterior_mean_log_lambda
        ),
        "hierarchical_abs_sigma_drift_vs_full": _value_or_blank(
            result.abs_sigma_drift_vs_full_hierarchical
        ),
        "hierarchical_abs_log_lambda_drift_vs_full": _value_or_blank(
            result.abs_log_lambda_drift_vs_full_hierarchical
        ),
        "hierarchical_closer_sigma": _value_or_blank(result.hierarchical_closer_sigma),
        "hierarchical_closer_log_lambda": _value_or_blank(result.hierarchical_closer_log_lambda),
        "hierarchical_status": result.hierarchical.status,
        "notes": result.notes,
    }


def write_convergence_agent_level_csv(
    output_path: Path,
    agent_results: Sequence[StatcastConvergenceAgentResult],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONVERGENCE_AGENT_LEVEL_HEADER)
        writer.writeheader()
        for result in agent_results:
            writer.writerow(_convergence_result_to_row(result))


def write_convergence_summary_csvs(
    output_dir: Path,
    summary_by_n_rows: Sequence[dict[str, Any]],
    summary_overall_rows: Sequence[dict[str, Any]],
) -> None:
    paths = planned_convergence_output_paths(output_dir)
    _write_dict_rows(paths["summary_by_n_csv"], SUMMARY_BY_BUCKET_CSV_HEADER, summary_by_n_rows)
    _write_dict_rows(paths["summary_overall_csv"], SUMMARY_OVERALL_CSV_HEADER, summary_overall_rows)


def _roster_has_bbip_selection(output_dir: Path) -> bool:
    """True when prepare-roster wrote a walk/IP-proxy extremes manifest."""

    import json

    metadata_path = output_dir / CONVERGENCE_ROSTER_METADATA_FILENAME
    if not metadata_path.is_file():
        return False
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("bbip_selection"))


def maybe_write_separability_outputs(
    output_dir: Path,
    *,
    allow_failure: bool = False,
) -> dict[str, Any] | None:
    """Run proxy-group separability when ``bbip_selection`` is present.

    Other rosters (top-pitchers / pitcher-ids / all-eligible) skip silently so
    aggregate jobs stay unchanged. Walk/IP-proxy failures are fatal by default
    because the separability result is a paper endpoint; ``allow_failure`` is
    non-paper only.
    """

    if not _roster_has_bbip_selection(output_dir):
        return None

    from HJEEDS.baseball_separability import run_separability_analysis

    try:
        summary = run_separability_analysis(output_dir)
    except Exception as exc:
        if not allow_failure:
            raise RuntimeError(
                f"Required walk/IP-proxy separability failed under "
                f"{output_dir.resolve()}: {exc}"
            ) from exc
        print(
            f"[baseball-convergence] NON-PAPER WARNING: walk/IP-proxy separability failed under "
            f"{output_dir.resolve()}: {exc}",
            flush=True,
        )
        return None

    print(
        f"[baseball-convergence] Wrote walk/IP-proxy separability artifacts under "
        f"{output_dir.resolve()}",
        flush=True,
    )
    return summary


def write_convergence_outputs(
    output_dir: Path,
    agent_results: Sequence[StatcastConvergenceAgentResult],
    summary_by_n_rows: Sequence[dict[str, Any]],
    summary_overall_rows: Sequence[dict[str, Any]],
    *,
    allow_separability_failure: bool = False,
) -> dict[str, Path]:
    """Write agent/summary CSVs, drift plots, and proxy separability artifacts."""

    roster_metadata_path = output_dir / CONVERGENCE_ROSTER_METADATA_FILENAME
    if roster_metadata_path.is_file():
        frozen_metadata = load_convergence_roster_metadata(output_dir)
        frozen_roster = load_convergence_roster(output_dir)
        validate_convergence_agent_results(
            agent_results,
            expected_agent_ids={spec.agent_id for spec in frozen_roster},
            expected_checkpoints={
                int(value) for value in frozen_metadata.get("convergence_ns", ())
            },
        )
    else:
        validate_convergence_agent_results(agent_results)
    paths = planned_convergence_output_paths(output_dir)
    write_convergence_agent_level_csv(paths["agent_level_csv"], agent_results)
    write_convergence_summary_csvs(output_dir, summary_by_n_rows, summary_overall_rows)
    write_drift_plots(output_dir, summary_by_n_rows)
    maybe_write_separability_outputs(
        output_dir,
        allow_failure=allow_separability_failure,
    )
    finalize_convergence_run_metadata(output_dir)
    print(
        f"[baseball-convergence] Regenerated dual-reference summaries and plots under "
        f"{output_dir.resolve()}",
        flush=True,
    )
    return paths


def validate_convergence_agent_results(
    agent_results: Sequence[StatcastConvergenceAgentResult],
    *,
    expected_agent_ids: set[int] | None = None,
    expected_checkpoints: set[int] | None = None,
) -> None:
    """Fail before publication output if any agent/checkpoint estimate is incomplete."""

    if not agent_results:
        raise ValueError("No baseball convergence agent results were produced.")
    seen: set[tuple[int, int, int]] = set()
    checkpoints_by_seed_agent: dict[tuple[int, int], set[int]] = defaultdict(set)
    agents_by_seed: dict[int, set[int]] = defaultdict(set)
    for result in agent_results:
        identity = (int(result.seed), int(result.agent_id), int(result.convergence_n))
        if identity in seen:
            raise ValueError(f"Duplicate baseball convergence result {identity}.")
        seen.add(identity)
        checkpoints_by_seed_agent[(result.seed, result.agent_id)].add(result.convergence_n)
        agents_by_seed[result.seed].add(result.agent_id)

        for label, estimate in (
            ("reference", result.reference),
            ("JEEDS", result.jeeds),
            ("H-JEEDS", result.hierarchical),
        ):
            if estimate.status != "ok":
                raise ValueError(f"{label} status is {estimate.status!r} for result {identity}.")
            for field_name, value in (
                ("posterior_mean_sigma", estimate.posterior_mean_sigma),
                ("posterior_mean_log_lambda", estimate.posterior_mean_log_lambda),
            ):
                if value is None or not math.isfinite(float(value)):
                    raise ValueError(f"{label} {field_name} is invalid for result {identity}: {value}.")
                if field_name == "posterior_mean_sigma" and float(value) <= 0.0:
                    raise ValueError(
                        f"{label} posterior_mean_sigma must be positive for result "
                        f"{identity}: {value}."
                    )

        for field_name, value in (
            ("jeeds_sigma_drift", result.abs_sigma_drift_vs_full_jeeds),
            ("jeeds_log_lambda_drift", result.abs_log_lambda_drift_vs_full_jeeds),
            ("hierarchical_sigma_drift", result.abs_sigma_drift_vs_full_hierarchical),
            ("hierarchical_log_lambda_drift", result.abs_log_lambda_drift_vs_full_hierarchical),
        ):
            if value is None or not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{field_name} is invalid for result {identity}: {value}.")

    if expected_checkpoints is None:
        expected_checkpoints = set().union(*checkpoints_by_seed_agent.values())
    if not expected_checkpoints:
        raise ValueError("Expected baseball convergence checkpoints cannot be empty.")
    for seed_agent, checkpoints in checkpoints_by_seed_agent.items():
        if checkpoints != expected_checkpoints:
            raise ValueError(
                f"Seed/agent {seed_agent} has checkpoints {sorted(checkpoints)}; "
                f"expected {sorted(expected_checkpoints)}."
            )
    expected_agents = (
        set(expected_agent_ids)
        if expected_agent_ids is not None
        else set().union(*agents_by_seed.values())
    )
    if not expected_agents:
        raise ValueError("Expected baseball convergence agent roster cannot be empty.")
    for seed, agents in agents_by_seed.items():
        if agents != expected_agents:
            raise ValueError(
                f"Seed {seed} has agents {sorted(agents)}; expected {sorted(expected_agents)}."
            )


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def recompute_self_reference_drifts_in_agent_rows(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rewrite drift columns so each method references its own estimate at max N."""

    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("seed"), row.get("agent_id"))].append(dict(row))

    rewritten: list[dict[str, Any]] = []
    for group_rows in grouped.values():
        ns = [int(row["convergence_n"]) for row in group_rows]
        max_n = max(ns)
        final_row = next(row for row in group_rows if int(row["convergence_n"]) == max_n)
        jeeds_ref_sigma = _float_or_none(final_row.get("jeeds_posterior_mean_sigma"))
        jeeds_ref_log_lambda = _float_or_none(final_row.get("jeeds_posterior_mean_log_lambda"))
        hier_ref_sigma = _float_or_none(final_row.get("hierarchical_posterior_mean_sigma"))
        hier_ref_log_lambda = _float_or_none(final_row.get("hierarchical_posterior_mean_log_lambda"))

        for row in sorted(group_rows, key=lambda item: int(item["convergence_n"])):
            jeeds_sigma = _float_or_none(row.get("jeeds_posterior_mean_sigma"))
            jeeds_log_lambda = _float_or_none(row.get("jeeds_posterior_mean_log_lambda"))
            hier_sigma = _float_or_none(row.get("hierarchical_posterior_mean_sigma"))
            hier_log_lambda = _float_or_none(row.get("hierarchical_posterior_mean_log_lambda"))

            row["jeeds_abs_sigma_drift_vs_full"] = _abs_difference_or_blank(
                jeeds_sigma, jeeds_ref_sigma
            )
            row["jeeds_abs_log_lambda_drift_vs_full"] = _abs_difference_or_blank(
                jeeds_log_lambda, jeeds_ref_log_lambda
            )
            row["hierarchical_abs_sigma_drift_vs_full"] = _abs_difference_or_blank(
                hier_sigma, hier_ref_sigma
            )
            row["hierarchical_abs_log_lambda_drift_vs_full"] = _abs_difference_or_blank(
                hier_log_lambda, hier_ref_log_lambda
            )
            jeeds_sigma_drift = _float_or_none(row["jeeds_abs_sigma_drift_vs_full"])
            hier_sigma_drift = _float_or_none(row["hierarchical_abs_sigma_drift_vs_full"])
            jeeds_lambda_drift = _float_or_none(row["jeeds_abs_log_lambda_drift_vs_full"])
            hier_lambda_drift = _float_or_none(row["hierarchical_abs_log_lambda_drift_vs_full"])
            if jeeds_sigma_drift is not None and hier_sigma_drift is not None:
                row["hierarchical_closer_sigma"] = hier_sigma_drift < jeeds_sigma_drift
            else:
                row["hierarchical_closer_sigma"] = ""
            if jeeds_lambda_drift is not None and hier_lambda_drift is not None:
                row["hierarchical_closer_log_lambda"] = hier_lambda_drift < jeeds_lambda_drift
            else:
                row["hierarchical_closer_log_lambda"] = ""
            rewritten.append(row)

    rewritten.sort(key=lambda row: (int(row["agent_id"]), int(row["convergence_n"])))
    return rewritten


def summarize_drift_rows_from_agent_csv_rows(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build summary_by_N / summary_overall tables from agent-level drift columns."""

    bucket_metrics: dict[tuple[str, str, int], list[float]] = {}
    overall_metrics: dict[tuple[str, str], list[float]] = {}

    method_columns = {
        "jeeds": (
            "jeeds_abs_sigma_drift_vs_full",
            "jeeds_abs_log_lambda_drift_vs_full",
        ),
        "hierarchical": (
            "hierarchical_abs_sigma_drift_vs_full",
            "hierarchical_abs_log_lambda_drift_vs_full",
        ),
    }

    for row in rows:
        convergence_n = int(row["convergence_n"])
        for method_name, (sigma_col, lambda_col) in method_columns.items():
            sigma_drift = _float_or_none(row.get(sigma_col))
            lambda_drift = _float_or_none(row.get(lambda_col))
            if sigma_drift is not None:
                bucket_metrics.setdefault(
                    (method_name, "abs_sigma_drift_vs_full", convergence_n), []
                ).append(sigma_drift)
                overall_metrics.setdefault(
                    (method_name, "abs_sigma_drift_vs_full"), []
                ).append(sigma_drift)
            if lambda_drift is not None:
                bucket_metrics.setdefault(
                    (method_name, "abs_log_lambda_drift_vs_full", convergence_n), []
                ).append(lambda_drift)
                overall_metrics.setdefault(
                    (method_name, "abs_log_lambda_drift_vs_full"), []
                ).append(lambda_drift)

    return build_drift_summary_tables(
        bucket_metrics,
        overall_metrics,
        by_n_notes=(
            "Mean absolute self-reference drift over agents. "
            "Baseball Statcast estimates are deterministic in seed; no CI."
        ),
        overall_notes=(
            "Mean absolute self-reference drift over agents x N. "
            "Baseball Statcast estimates are deterministic in seed; no CI."
        ),
    )


def plot_drift_by_n(
    output_path: Path,
    summary_by_n_rows: Sequence[dict[str, Any]],
    *,
    x_scale: str = "categorical",
) -> None:
    """Plot drift of each method toward its own max-N estimate.

    ``x_scale`` is ``"categorical"`` (equal spacing) or ``"proportional"``
    (true pitch-count spacing on a linear axis).
    """

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    from HJEEDS.artifacts import METHOD_ORDER, _parse_bucket_summary_rows

    if x_scale not in {"categorical", "proportional"}:
        raise ValueError(f"Unsupported x_scale={x_scale!r}")

    parsed_rows = _parse_bucket_summary_rows(summary_by_n_rows)
    bucket_values = sorted({row["count_bucket"] for row in parsed_rows})
    figure_size = error_metric_figure_size(len(DRIFT_METRIC_PANELS))
    figure, axes = plt.subplots(1, len(DRIFT_METRIC_PANELS), figsize=figure_size, sharex=True)
    axes_array = np.atleast_1d(axes)
    handles = []
    labels = []

    for axis_index, (axis, (metric_name, title, ylabel, missing_message)) in enumerate(
        zip(axes_array, DRIFT_METRIC_PANELS)
    ):
        axis.set_title(title)
        axis.set_xlabel(r"Pitch-count checkpoint $c$")
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.45)

        metric_rows = [row for row in parsed_rows if row["metric"] == metric_name]
        methods = sorted(
            {row["method"] for row in metric_rows},
            key=lambda method: (METHOD_ORDER.get(method, len(METHOD_ORDER)), method),
        )

        if metric_rows and bucket_values:
            for method in methods:
                style = BASEBALL_METHOD_STYLES.get(
                    method,
                    {"label": method, "color": "#555555", "marker": "o"},
                )
                method_rows = sorted(
                    [row for row in metric_rows if row["method"] == method],
                    key=lambda row: row["count_bucket"],
                )
                if x_scale == "categorical":
                    x_values = [bucket_values.index(row["count_bucket"]) for row in method_rows]
                else:
                    x_values = [float(row["count_bucket"]) for row in method_rows]
                y_values = [row["mean"] for row in method_rows]
                if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in y_values):
                    raise ValueError(
                        f"Drift plot values must be finite and nonnegative for "
                        f"{method}/{metric_name}: {y_values}."
                    )

                (line,) = axis.plot(
                    x_values,
                    y_values,
                    color=style["color"],
                    marker=style["marker"],
                    markersize=6,
                    linestyle="-",
                    linewidth=2.0,
                    label=style["label"],
                )
                if axis_index == 0:
                    handles.append(line)
                    labels.append(style["label"])

            if x_scale == "categorical":
                axis.set_xticks(list(range(len(bucket_values))))
                axis.set_xticklabels([str(bucket) for bucket in bucket_values])
            else:
                axis.set_xticks(bucket_values)
                axis.set_xticklabels([str(bucket) for bucket in bucket_values])
                axis.set_xlim(0, max(bucket_values) * 1.05)
        else:
            axis.text(
                0.5,
                0.5,
                missing_message,
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(handles),
            frameon=False,
            fontsize=9,
            bbox_to_anchor=(0.5, 1.02),
        )

    ref_n = max(bucket_values) if bucket_values else r"N_{\max}"
    figure.suptitle(
        rf"Distance to each method's own estimate at $N={ref_n}$ (not ground truth)",
        fontsize=10,
        y=1.08,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_drift_plots(output_dir: Path, summary_by_n_rows: Sequence[dict[str, Any]]) -> None:
    paths = planned_convergence_output_paths(output_dir)
    plot_drift_by_n(paths["drift_plot"], summary_by_n_rows, x_scale="categorical")
    plot_drift_by_n(
        paths["drift_plot_proportional"],
        summary_by_n_rows,
        x_scale="proportional",
    )


def regenerate_plot_from_existing_results(
    output_dir: Path,
    *,
    allow_separability_failure: bool = False,
) -> None:
    """Recompute dual self-reference drifts from agent CSV, then rewrite summaries/plots."""

    paths = planned_convergence_output_paths(output_dir)
    agent_path = paths["agent_level_csv"]
    if not agent_path.exists():
        raise FileNotFoundError(
            f"Cannot regenerate plots because agent-level CSV is missing: {agent_path}"
        )

    with agent_path.open("r", newline="") as handle:
        agent_rows = list(csv.DictReader(handle))
    if not agent_rows:
        raise ValueError(f"No rows in {agent_path}")

    validate_plot_only_agent_rows(output_dir, agent_rows)
    restamp_convergence_run_incomplete(output_dir)
    updated_rows = recompute_self_reference_drifts_in_agent_rows(agent_rows)
    with agent_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONVERGENCE_AGENT_LEVEL_HEADER)
        writer.writeheader()
        writer.writerows(updated_rows)

    summary_by_n_rows, summary_overall_rows = summarize_drift_rows_from_agent_csv_rows(updated_rows)
    write_convergence_summary_csvs(output_dir, summary_by_n_rows, summary_overall_rows)
    write_drift_plots(output_dir, summary_by_n_rows)
    maybe_write_separability_outputs(
        output_dir,
        allow_failure=allow_separability_failure,
    )
    finalize_convergence_run_metadata(output_dir)


def validate_plot_only_agent_rows(
    output_dir: Path,
    agent_rows: Sequence[dict[str, Any]],
) -> None:
    """Require the frozen deterministic roster×checkpoint product before plotting."""

    metadata = load_convergence_roster_metadata(output_dir)
    roster = load_convergence_roster(output_dir)
    expected_ns = tuple(int(value) for value in metadata.get("convergence_ns", ()))
    if not expected_ns:
        raise ValueError("Frozen convergence metadata has no checkpoints.")
    if int(metadata.get("num_agents", -1)) != len(roster):
        raise ValueError("Frozen convergence roster size does not match its metadata.")

    frozen_provenance = metadata.get("run_provenance")
    if not isinstance(frozen_provenance, dict):
        raise ValueError("Frozen convergence metadata lacks run_provenance.")
    expected_provenance = build_baseball_run_provenance(
        season_year=metadata.get("season_year"),
        pitch_types=metadata.get("pitch_types") or (),
        sigma_grid=frozen_provenance.get("sigma_grid") or (),
        log_lambda_grid=frozen_provenance.get("log_lambda_grid") or (),
        configuration=frozen_provenance.get("configuration") or {},
    )
    require_matching_fingerprint(
        frozen_provenance,
        expected_provenance,
        label="plot-only convergence metadata",
    )
    frozen_configuration = frozen_provenance.get("configuration") or {}
    for key, expected_value in (
        ("convergence_ns", list(expected_ns)),
        ("max_reference_pitches", metadata.get("max_reference_pitches")),
        ("min_pitches_per_agent", metadata.get("min_pitches_per_agent")),
        ("num_agents", len(roster)),
    ):
        if frozen_configuration.get(key) != expected_value:
            raise ValueError(
                f"Frozen convergence provenance {key}={frozen_configuration.get(key)!r}; "
                f"metadata expects {expected_value!r}."
            )
    expected_roster_hash = roster_fingerprint(
        [(spec.agent_id, spec.pitcher_id, spec.pitch_type) for spec in roster]
    )
    if (frozen_provenance.get("configuration") or {}).get(
        "roster_fingerprint"
    ) != expected_roster_hash:
        raise ValueError("Frozen convergence provenance does not match convergence_roster.json.")

    seeds = {int(row["seed"]) for row in agent_rows}
    if len(seeds) != 1:
        raise ValueError(
            f"Paper baseball plotting requires one deterministic seed; found {sorted(seeds)}."
        )
    seed = next(iter(seeds))
    expected_identity = {
        (seed, spec.agent_id, checkpoint)
        for spec in roster
        for checkpoint in expected_ns
    }
    spec_by_agent = {spec.agent_id: spec for spec in roster}
    available_counts = {
        (int(row["pitcher_id"]), str(row["pitch_type"])): int(row["num_available_pitches"])
        for row in metadata.get("agent_pitch_counts", [])
    }
    if len(available_counts) != len(roster):
        raise ValueError("Frozen convergence metadata has incomplete agent pitch counts.")
    actual_identity: set[tuple[int, int, int]] = set()
    for row in agent_rows:
        identity = (int(row["seed"]), int(row["agent_id"]), int(row["convergence_n"]))
        if identity in actual_identity:
            raise ValueError(f"Duplicate convergence agent CSV row {identity}.")
        actual_identity.add(identity)
        spec = spec_by_agent.get(identity[1])
        if spec is None or (int(row["pitcher_id"]), str(row["pitch_type"])) != (
            spec.pitcher_id,
            spec.pitch_type,
        ):
            raise ValueError(f"Convergence agent CSV identity mismatch for {identity}.")
        available = available_counts[(spec.pitcher_id, spec.pitch_type)]
        expected_observations = min(identity[2], available)
        max_reference = metadata.get("max_reference_pitches")
        expected_reference = (
            available if max_reference is None else min(available, int(max_reference))
        )
        if int(row.get("num_observations", -1)) != expected_observations:
            raise ValueError(f"Incorrect num_observations for {identity}.")
        if int(row.get("num_reference_observations", -1)) != expected_reference:
            raise ValueError(f"Incorrect num_reference_observations for {identity}.")
        for status_field in ("reference_status", "jeeds_status", "hierarchical_status"):
            if str(row.get(status_field, "")).strip().lower() != "ok":
                raise ValueError(f"Non-successful {status_field} for {identity}.")
        for value_field in (
            "reference_posterior_mean_sigma",
            "reference_posterior_mean_log_lambda",
            "jeeds_posterior_mean_sigma",
            "jeeds_posterior_mean_log_lambda",
            "hierarchical_posterior_mean_sigma",
            "hierarchical_posterior_mean_log_lambda",
        ):
            value = _float_or_none(row.get(value_field))
            if value is None or not math.isfinite(value):
                raise ValueError(f"Invalid {value_field} for {identity}: {row.get(value_field)!r}.")
            if value_field.endswith("posterior_mean_sigma") and value <= 0.0:
                raise ValueError(
                    f"Nonpositive {value_field} for {identity}: {row.get(value_field)!r}."
                )
    missing = expected_identity - actual_identity
    extra = actual_identity - expected_identity
    if missing or extra:
        raise ValueError(
            "Convergence agent CSV is not the frozen roster×checkpoint product: "
            f"missing={sorted(missing)}, extra={sorted(extra)}."
        )


def _load_pitcher_name_lookup(output_dir: Path) -> dict[int, str]:
    import json

    metadata_path = output_dir / "convergence_roster_metadata.json"
    if not metadata_path.is_file():
        return {}
    payload = json.loads(metadata_path.read_text())
    names: dict[int, str] = {}
    for row in payload.get("bbip_selection", []):
        try:
            names[int(row["pitcher_id"])] = str(row["player_name"])
        except (KeyError, TypeError, ValueError):
            continue
    return names


def plot_agent_intermediate_estimates(output_dir: Path) -> list[Path]:
    """Plot JEEDS/H-JEEDS checkpoint estimates per agent into ``agents/*.png``."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import pandas as pd

    validate_complete_convergence_run_metadata(output_dir)
    paths = planned_convergence_output_paths(output_dir)
    agent_csv = paths["agent_level_csv"]
    if not agent_csv.is_file():
        raise FileNotFoundError(f"Missing agent-level CSV: {agent_csv}")

    frame = pd.read_csv(agent_csv)
    if frame.empty:
        raise ValueError(f"No rows in {agent_csv}")

    agents_dir = output_dir / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    name_lookup = _load_pitcher_name_lookup(output_dir)
    written: list[Path] = []

    metric_panels = (
        ("jeeds_posterior_mean_sigma", "hierarchical_posterior_mean_sigma", r"$\hat{\sigma}$", "sigma"),
        (
            "jeeds_posterior_mean_log_lambda",
            "hierarchical_posterior_mean_log_lambda",
            r"$\widehat{\log\lambda}$",
            "log_lambda",
        ),
    )

    for agent_id, agent_frame in frame.groupby("agent_id", sort=True):
        agent_frame = agent_frame.sort_values("convergence_n").copy()
        ns = agent_frame["convergence_n"].astype(int).tolist()
        pitcher_id = int(agent_frame["pitcher_id"].iloc[0])
        pitch_type = str(agent_frame["pitch_type"].iloc[0])
        player_name = name_lookup.get(pitcher_id, f"pitcher {pitcher_id}")
        final_n = max(ns)

        for jeeds_col, hier_col, _ylabel, _key in metric_panels:
            agent_frame[jeeds_col] = pd.to_numeric(agent_frame[jeeds_col], errors="coerce")
            agent_frame[hier_col] = pd.to_numeric(agent_frame[hier_col], errors="coerce")

        usable = False
        for jeeds_col, hier_col, _ylabel, _key in metric_panels:
            if agent_frame[jeeds_col].notna().any() or agent_frame[hier_col].notna().any():
                usable = True
                break
        if not usable:
            print(
                f"[baseball-convergence] Skipping agent {agent_id}: no numeric posterior means",
                flush=True,
            )
            continue

        figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        for axis, (jeeds_col, hier_col, ylabel, _key) in zip(axes, metric_panels):
            jeeds_y = agent_frame[jeeds_col].tolist()
            hier_y = agent_frame[hier_col].tolist()
            jeeds_style = BASEBALL_METHOD_STYLES["jeeds"]
            hier_style = BASEBALL_METHOD_STYLES["hierarchical"]

            axis.plot(
                ns,
                jeeds_y,
                color=jeeds_style["color"],
                marker=jeeds_style["marker"],
                linestyle="-",
                linewidth=2.0,
                markersize=6,
                label=jeeds_style["label"],
            )
            axis.plot(
                ns,
                hier_y,
                color=hier_style["color"],
                marker=hier_style["marker"],
                linestyle="-",
                linewidth=2.0,
                markersize=6,
                label=hier_style["label"],
            )

            final_row = agent_frame.loc[agent_frame["convergence_n"] == final_n].iloc[0]
            jeeds_ref = final_row[jeeds_col]
            hier_ref = final_row[hier_col]
            if pd.notna(jeeds_ref):
                axis.axhline(
                    float(jeeds_ref),
                    color=jeeds_style["color"],
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.85,
                    label=f"JEEDS @ {final_n}",
                )
            if pd.notna(hier_ref):
                axis.axhline(
                    float(hier_ref),
                    color=hier_style["color"],
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.85,
                    label=f"H-JEEDS @ {final_n}",
                )
            axis.set_xlabel(r"Pitch-count checkpoint $c$")
            axis.set_ylabel(ylabel)
            axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
            axis.set_xticks(ns)

        axes[0].legend(loc="best", fontsize=8)
        figure.suptitle(f"{player_name} ({pitcher_id}, {pitch_type})", fontsize=11)
        figure.tight_layout()
        output_path = agents_dir / f"agent_{int(agent_id):04d}.png"
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        written.append(output_path)

    return written


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_convergence_args(argv)

    if args.list_eligible_pitchers:
        convergence_ns = _parse_count_buckets(args.convergence_ns)
        min_pitches = (
            args.min_pitches_per_agent
            if args.min_pitches_per_agent is not None
            else required_min_pitches_for_convergence(convergence_ns, args.max_reference_pitches)
        )
        print_eligible_agents(
            season_year=args.season_year,
            pitch_types=parse_pitch_types(args.pitch_types),
            min_pitches=min_pitches,
            limit=args.list_eligible_limit,
        )
        return 0

    if args.seed is None:
        if (
            args.dry_run
            or args.plot_only
            or args.plot_agent_estimates
            or args.prepare_roster
            or args.aggregate_results
            or args.agent_index is not None
        ):
            args.seed = DEFAULT_SEED
        else:
            raise SystemExit(
                "error: --seed is required unless using --list-eligible-pitchers, --dry-run, "
                "--plot-only, --plot-agent-estimates, --prepare-roster, --agent-index, or --aggregate-results"
            )

    if args.prepare_roster:
        prepare_convergence_roster(args)
        return 0

    if args.plot_agent_estimates:
        written = plot_agent_intermediate_estimates(Path(args.output_dir))
        print(
            f"[baseball-convergence] Wrote {len(written)} agent estimate plots under "
            f"{(Path(args.output_dir) / 'agents').resolve()}",
            flush=True,
        )
        return 0

    config = build_baseball_convergence_config_from_args(args)

    if config.base.dry_run:
        print_baseball_convergence_dry_run_summary(config)
        return 0

    if args.plot_only:
        regenerate_plot_from_existing_results(
            config.base.output_dir,
            allow_separability_failure=args.allow_separability_failure,
        )
        return 0

    if args.aggregate_results:
        all_agent_results, summary_by_n_rows, summary_overall_rows = aggregate_convergence_results(args)
        write_convergence_outputs(
            config.base.output_dir,
            all_agent_results,
            summary_by_n_rows,
            summary_overall_rows,
            allow_separability_failure=args.allow_separability_failure,
        )
        print(f"[baseball-convergence] Wrote aggregated results to {config.base.output_dir.resolve()}", flush=True)
        return 0

    if args.agent_index is not None:
        run_convergence_agent_index(args, args.agent_index)
        return 0

    begin_convergence_run_metadata(
        config.base.output_dir,
        args=args,
        config=config,
        run_provenance=build_convergence_run_provenance(config),
    )
    seed_results = []
    for seed_index, seed in enumerate(config.seed_values, start=1):
        print(
            f"[baseball-convergence] Running seed {seed_index}/{config.base.num_seeds}: {seed}",
            flush=True,
        )
        seed_results.append(run_single_baseball_convergence_seed(config, seed))

    all_agent_results = [result for seed_result in seed_results for result in seed_result.agent_results]
    summary_by_n_rows, summary_overall_rows = aggregate_convergence_across_seeds(seed_results)
    write_convergence_outputs(
        config.base.output_dir,
        all_agent_results,
        summary_by_n_rows,
        summary_overall_rows,
        allow_separability_failure=args.allow_separability_failure,
    )
    print(f"[baseball-convergence] Wrote results to {config.base.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
