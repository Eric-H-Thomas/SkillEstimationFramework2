"""Aggregate per-part H-JEEDS results into one group directory."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Iterable

from HJEEDS.aggregation import aggregate_results_across_seeds, summarize_seed_results
from HJEEDS.artifacts import plot_error_by_bucket, write_agent_level_csv, write_summary_csvs
from HJEEDS.config import AGENT_LEVEL_CSV_HEADER, planned_output_paths
from HJEEDS.models import AgentResult, MethodEstimate, SeedResult


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


def aggregate_group(
    group_dir: Path,
    parts_per_group: int,
    cleanup: bool,
    *,
    include_raw_rationality_error: bool = False,
) -> None:
    if parts_per_group <= 0:
        raise ValueError("parts_per_group must be positive.")

    part_dirs = [group_dir / f"part_{index}" for index in range(parts_per_group)]
    for part_dir in part_dirs:
        if not part_dir.exists():
            raise FileNotFoundError(f"Missing part directory: {part_dir}")

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

    if cleanup:
        for part_dir in part_dirs:
            shutil.rmtree(part_dir)

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
            "execution error and rationality percentage-point error."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    aggregate_group(
        args.group_dir,
        args.parts_per_group,
        args.cleanup,
        include_raw_rationality_error=args.include_raw_rationality_error,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
