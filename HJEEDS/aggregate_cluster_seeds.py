# Paper correspondence: Main `subsec:two_d_darts`.
"""Aggregate per-part H-JEEDS results into one group directory."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from pathlib import Path
from typing import Iterable

from HJEEDS.aggregation import aggregate_results_across_seeds, summarize_seed_results
from HJEEDS.artifacts import plot_error_by_bucket, write_agent_level_csv, write_summary_csvs
from HJEEDS.config import AGENT_LEVEL_CSV_HEADER, planned_output_paths
from HJEEDS.models import AgentResult, MethodEstimate, SeedResult
from HJEEDS.two_d_completion import (
    begin_two_d_aggregation_metadata,
    finalize_two_d_aggregation_metadata,
    validate_complete_two_d_part_metadata,
)


def _parse_optional_float(value: str | None, field_name: str) -> float | None:
    if value is None:
        return None
    raw = value.strip()
    if raw == "":
        return None
    try:
        parsed = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {field_name}: {value}") from exc
    if not math.isfinite(parsed):
        return None
    return parsed


def _parse_required_float(value: str | None, field_name: str) -> float:
    parsed = _parse_optional_float(value, field_name)
    if parsed is None:
        raise ValueError(f"Missing required float for {field_name}.")
    return parsed


def _parse_required_int(value: str | None, field_name: str) -> int:
    if value is None:
        raise ValueError(f"Missing required int for {field_name}.")
    raw = value.strip()
    if raw == "":
        raise ValueError(f"Missing required int for {field_name}.")
    try:
        return int(raw)
    except ValueError:
        try:
            return int(float(raw))
        except ValueError as exc:
            raise ValueError(f"Invalid int for {field_name}: {value}") from exc


def _parse_environment(value: str | None) -> str | None:
    if value is None:
        return None
    raw = value.strip()
    return raw if raw else None


def _method_from_row(row: dict[str, str], prefix: str) -> MethodEstimate:
    return MethodEstimate(
        method_name=prefix,
        posterior_mean_sigma=_parse_optional_float(row.get(f"{prefix}_posterior_mean_sigma"), f"{prefix}_posterior_mean_sigma"),
        posterior_mean_log_lambda=_parse_optional_float(
            row.get(f"{prefix}_posterior_mean_log_lambda"),
            f"{prefix}_posterior_mean_log_lambda",
        ),
        map_sigma=_parse_optional_float(row.get(f"{prefix}_map_sigma"), f"{prefix}_map_sigma"),
        map_log_lambda=_parse_optional_float(row.get(f"{prefix}_map_log_lambda"), f"{prefix}_map_log_lambda"),
        rationality_percent=_parse_optional_float(
            row.get(f"{prefix}_rationality_percent"),
            f"{prefix}_rationality_percent",
        ),
        status=(row.get(f"{prefix}_status") or "").strip() or "unknown",
    )


def _read_agent_results(agent_csv: Path) -> tuple[list[AgentResult], str]:
    if not agent_csv.exists():
        raise FileNotFoundError(f"Missing agent-level CSV: {agent_csv}")

    agent_results: list[AgentResult] = []
    environment: str | None = None

    with agent_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in {agent_csv}")
        if list(reader.fieldnames) != AGENT_LEVEL_CSV_HEADER:
            raise ValueError(
                "Unexpected agent-level CSV header in "
                f"{agent_csv}. Expected {AGENT_LEVEL_CSV_HEADER} but found {reader.fieldnames}."
            )

        for row in reader:
            row_env = _parse_environment(row.get("environment"))
            if row_env is not None:
                if environment is None:
                    environment = row_env
                elif environment != row_env:
                    raise ValueError(
                        f"Mixed environments in {agent_csv}: {environment} vs {row_env}."
                    )

            seed = _parse_required_int(row.get("seed"), "seed")
            agent_id = _parse_required_int(row.get("agent_id"), "agent_id")
            count_bucket = _parse_required_int(row.get("count_bucket"), "count_bucket")
            num_observations = _parse_required_int(row.get("num_observations"), "num_observations")
            sigma_true = _parse_required_float(row.get("sigma_true"), "sigma_true")
            log_lambda_true = _parse_required_float(row.get("log_lambda_true"), "log_lambda_true")
            rationality_percent_true = _parse_optional_float(
                row.get("rationality_percent_true"),
                "rationality_percent_true",
            )

            agent_results.append(
                AgentResult(
                    seed=seed,
                    agent_id=agent_id,
                    count_bucket=count_bucket,
                    num_observations=num_observations,
                    sigma_true=sigma_true,
                    log_lambda_true=log_lambda_true,
                    rationality_percent_true=rationality_percent_true,
                    jeeds=_method_from_row(row, "jeeds"),
                    hierarchical=_method_from_row(row, "hierarchical"),
                    notes=(row.get("notes") or ""),
                )
            )

    if environment is None:
        raise ValueError(f"No environment field found in {agent_csv}.")

    return agent_results, environment


def _group_by_seed(agent_results: Iterable[AgentResult]) -> dict[int, list[AgentResult]]:
    grouped: dict[int, list[AgentResult]] = {}
    for result in agent_results:
        grouped.setdefault(result.seed, []).append(result)
    return grouped


def _validate_seed_agent_coverage(
    agent_results: Iterable[AgentResult],
    *,
    expected_seed_start: int,
    expected_num_seeds: int,
    expected_agents_per_seed: int,
) -> None:
    """Reject partial, duplicated, or out-of-design cluster result rows."""

    if expected_seed_start < 0:
        raise ValueError("expected_seed_start must be nonnegative.")
    if expected_num_seeds <= 0:
        raise ValueError("expected_num_seeds must be positive.")
    if expected_agents_per_seed <= 0:
        raise ValueError("expected_agents_per_seed must be positive.")

    results = list(agent_results)
    expected_seeds = set(
        range(expected_seed_start, expected_seed_start + expected_num_seeds)
    )
    actual_seeds = {result.seed for result in results}
    missing_seeds = sorted(expected_seeds - actual_seeds)
    unexpected_seeds = sorted(actual_seeds - expected_seeds)
    if missing_seeds or unexpected_seeds:
        raise ValueError(
            "Seed coverage mismatch: "
            f"missing={missing_seeds or 'none'}, "
            f"unexpected={unexpected_seeds or 'none'}."
        )

    seen_keys: set[tuple[int, int]] = set()
    duplicate_keys: set[tuple[int, int]] = set()
    by_seed: dict[int, set[int]] = {}
    failed_statuses: list[tuple[int, int, str, str]] = []
    incomplete_estimates: list[tuple[int, int, str]] = []
    missing_truth_metrics: list[tuple[int, int]] = []
    invalid_population_fits: list[tuple[int, int, str]] = []
    for result in results:
        key = (result.seed, result.agent_id)
        if key in seen_keys:
            duplicate_keys.add(key)
        seen_keys.add(key)
        by_seed.setdefault(result.seed, set()).add(result.agent_id)
        if result.rationality_percent_true is None:
            missing_truth_metrics.append(key)
        notes = result.notes or ""
        if "population_fit: converged=True; selected=optimizer;" not in notes:
            invalid_population_fits.append((result.seed, result.agent_id, notes))
        if result.jeeds.status != "ok" or result.hierarchical.status != "ok":
            failed_statuses.append(
                (result.seed, result.agent_id, result.jeeds.status, result.hierarchical.status)
            )
        for method_name, estimate in (
            ("jeeds", result.jeeds),
            ("hierarchical", result.hierarchical),
        ):
            if estimate.status == "ok" and any(
                value is None
                for value in (
                    estimate.posterior_mean_sigma,
                    estimate.posterior_mean_log_lambda,
                    estimate.rationality_percent,
                )
            ):
                incomplete_estimates.append((result.seed, result.agent_id, method_name))

    if duplicate_keys:
        raise ValueError(
            "Duplicate seed-agent rows found: "
            f"{sorted(duplicate_keys)}."
        )

    expected_agent_ids = set(range(expected_agents_per_seed))
    coverage_errors: list[str] = []
    for seed in sorted(expected_seeds):
        actual_agent_ids = by_seed.get(seed, set())
        missing_agent_ids = sorted(expected_agent_ids - actual_agent_ids)
        unexpected_agent_ids = sorted(actual_agent_ids - expected_agent_ids)
        if missing_agent_ids or unexpected_agent_ids:
            coverage_errors.append(
                f"seed {seed}: missing agent IDs {missing_agent_ids or 'none'}, "
                f"unexpected agent IDs {unexpected_agent_ids or 'none'}"
            )
    if coverage_errors:
        raise ValueError("Agent coverage mismatch: " + "; ".join(coverage_errors))

    if failed_statuses:
        preview = failed_statuses[:10]
        suffix = (
            ""
            if len(failed_statuses) <= len(preview)
            else f" (+{len(failed_statuses) - len(preview)} more)"
        )
        raise ValueError(
            "Estimator failures found; refusing to aggregate incomplete results: "
            f"{preview}{suffix}."
        )
    if invalid_population_fits:
        preview = invalid_population_fits[:10]
        suffix = (
            ""
            if len(invalid_population_fits) <= len(preview)
            else f" (+{len(invalid_population_fits) - len(preview)} more)"
        )
        raise ValueError(
            "Missing a converged optimizer-selected population-fit diagnostic; "
            "refusing to aggregate 2D results: "
            f"{preview}{suffix}."
        )
    if missing_truth_metrics:
        preview = missing_truth_metrics[:10]
        suffix = (
            ""
            if len(missing_truth_metrics) <= len(preview)
            else f" (+{len(missing_truth_metrics) - len(preview)} more)"
        )
        raise ValueError(
            "Missing true decision-skill percentage for seed-agent rows: "
            f"{preview}{suffix}."
        )
    if incomplete_estimates:
        preview = incomplete_estimates[:10]
        suffix = (
            ""
            if len(incomplete_estimates) <= len(preview)
            else f" (+{len(incomplete_estimates) - len(preview)} more)"
        )
        raise ValueError(
            "Successful estimators have missing publication metrics: "
            f"{preview}{suffix}."
        )


def aggregate_group(
    group_dir: Path,
    parts_per_group: int,
    cleanup: bool,
    *,
    expected_seed_start: int,
    expected_num_seeds: int,
    expected_agents_per_seed: int,
    include_raw_rationality_error: bool = False,
) -> None:
    if parts_per_group <= 0:
        raise ValueError("parts_per_group must be positive.")

    # Invalidate any prior aggregate before inspecting or rewriting part
    # outputs. If validation, plotting, cleanup, or any later write fails, this
    # directory remains explicitly incomplete and the paper plotter refuses it.
    begin_two_d_aggregation_metadata(
        group_dir,
        expected_seed_start=expected_seed_start,
        expected_num_seeds=expected_num_seeds,
        expected_agents_per_seed=expected_agents_per_seed,
        parts_per_group=parts_per_group,
    )
    if expected_num_seeds % parts_per_group != 0:
        raise ValueError(
            "expected_num_seeds must be divisible by parts_per_group so every "
            "worker partition has an exact provenance-bound seed range."
        )
    seeds_per_part = expected_num_seeds // parts_per_group

    part_dirs = [group_dir / f"part_{index}" for index in range(parts_per_group)]
    for part_index, part_dir in enumerate(part_dirs):
        if not part_dir.exists():
            raise FileNotFoundError(f"Missing part directory: {part_dir}")
        validate_complete_two_d_part_metadata(
            part_dir,
            expected_seed_start=expected_seed_start + part_index * seeds_per_part,
            expected_num_seeds=seeds_per_part,
            expected_agents_per_seed=expected_agents_per_seed,
        )

    all_agent_results: list[AgentResult] = []
    environment: str | None = None

    for part_dir in part_dirs:
        part_paths = planned_output_paths(part_dir)
        agent_results, part_env = _read_agent_results(part_paths["agent_level_csv"])
        if environment is None:
            environment = part_env
        elif environment != part_env:
            raise ValueError(
                f"Mixed environments across parts in {group_dir}: {environment} vs {part_env}."
            )
        all_agent_results.extend(agent_results)

    if environment is None:
        raise ValueError(f"Unable to determine environment for {group_dir}.")
    if not all_agent_results:
        raise ValueError(f"No agent results found under {group_dir}.")
    if environment != "2d":
        raise ValueError(
            f"The 2D cluster aggregator expected environment='2d', found {environment!r}."
        )

    _validate_seed_agent_coverage(
        all_agent_results,
        expected_seed_start=expected_seed_start,
        expected_num_seeds=expected_num_seeds,
        expected_agents_per_seed=expected_agents_per_seed,
    )

    seed_results: list[SeedResult] = []
    for seed, agent_rows in sorted(_group_by_seed(all_agent_results).items()):
        seed_result = SeedResult(seed=seed, reward_surface=(), agent_results=agent_rows)
        seed_result.summary_by_bucket_rows, seed_result.summary_overall_rows = summarize_seed_results(seed_result)
        seed_results.append(seed_result)

    summary_by_bucket_rows, summary_overall_rows = aggregate_results_across_seeds(seed_results)

    output_paths = planned_output_paths(group_dir)
    write_agent_level_csv(output_paths["agent_level_csv"], all_agent_results, environment=environment)
    write_summary_csvs(group_dir, summary_by_bucket_rows, summary_overall_rows)
    plot_error_by_bucket(
        output_paths["error_plot"],
        summary_by_bucket_rows,
        include_raw_rationality_error=include_raw_rationality_error,
    )

    # This atomic completion record is the final publication-output write. It
    # binds the seed/agent design and current 2D computational sources to hashes
    # and byte sizes for every required CSV/plot artifact.
    finalize_two_d_aggregation_metadata(group_dir)

    # Part cleanup is optional post-publication housekeeping and is deliberately
    # outside the completed artifact contract. Sealing first avoids deleting the
    # only complete worker inputs if completion validation itself ever fails.
    if cleanup:
        for part_dir in part_dirs:
            try:
                shutil.rmtree(part_dir)
            except OSError as exc:
                print(
                    f"[aggregate] Warning: completed outputs are sealed, but optional "
                    f"part cleanup failed for {part_dir}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    print(f"[aggregate] Wrote combined results to {group_dir.resolve()}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine part_* H-JEEDS outputs into one aggregated directory.")
    parser.add_argument("--group-dir", type=Path, required=True, help="Group directory containing part_* folders.")
    parser.add_argument(
        "--parts-per-group",
        type=int,
        default=10,
        help="Number of part_* subdirectories to aggregate (default: 10).",
    )
    parser.add_argument(
        "--expected-seed-start",
        type=int,
        required=True,
        help="First seed that must appear in this group.",
    )
    parser.add_argument(
        "--expected-num-seeds",
        type=int,
        required=True,
        help="Exact number of consecutive seeds required for this group.",
    )
    parser.add_argument(
        "--expected-agents-per-seed",
        type=int,
        default=25,
        help="Exact agent count and ID range 0..N-1 required per seed (default: 25).",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete part_* directories after successful aggregation.",
    )
    parser.add_argument(
        "--include-raw-rationality-error",
        "--include-log-decision-error",
        dest="include_raw_rationality_error",
        action="store_true",
        help=(
            "Include the raw log-decision-skill error panel in addition to "
            "execution error and decision-skill percentage-point error."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    aggregate_group(
        args.group_dir,
        args.parts_per_group,
        args.cleanup,
        expected_seed_start=args.expected_seed_start,
        expected_num_seeds=args.expected_num_seeds,
        expected_agents_per_seed=args.expected_agents_per_seed,
        include_raw_rationality_error=args.include_raw_rationality_error,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
