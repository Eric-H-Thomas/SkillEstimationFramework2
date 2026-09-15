#!/usr/bin/env python3
# Paper correspondence: Main `sec:experiments`; validates the full 1D paper design.
"""Fail-fast validation for a completed 1D publication result tree.

The unified paper runner creates one ``agent_level_results.csv`` per concrete
scenario. This validator checks the exact canonical scenario paths and
population sizes, seed/agent completeness, estimator status, finite metrics,
and population-fit convergence before plots or archives can be treated as
publication artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_EXPECTED_SCENARIOS = 93
# Unified-suite population sizes summed across all 93 concrete scenarios:
# baseline 25; hyperpriors 60*25; APB 5*(1+2+5+10+25); shapes 3*25;
# outliers 3*25; anchors 6*50; decision 4*25; correlation 5*25;
# grid 3*25; compound 3*25.
DEFAULT_EXPECTED_ROWS_PER_SEED = 2565


def _canonical_scenario_agent_counts() -> dict[str, int]:
    """Return the exact relative CSV path and population size for all 93 scenarios."""

    scenarios: dict[str, int] = {}

    def add(relative_path: str, agents_per_seed: int) -> None:
        if relative_path in scenarios:
            raise RuntimeError(f"Duplicate canonical publication scenario: {relative_path}")
        scenarios[relative_path] = agents_per_seed

    add("baseline/agent_level_results.csv", 25)

    for focus in ("average_skill", "population_spread", "correlation", "combined"):
        for bias in (
            "strong_reverse_misspecification",
            "moderate_reverse_misspecification",
            "unbiased",
            "moderate_adverse_misspecification",
            "strong_adverse_misspecification",
        ):
            for confidence in ("weak", "default", "strong"):
                add(
                    "hyperprior_robustness/"
                    f"{focus}__{bias}__{confidence}/agent_level_results.csv",
                    25,
                )

    for agents_per_bucket in (1, 2, 5, 10, 25):
        add(
            "agents_per_bucket/"
            f"agents_per_bucket_{agents_per_bucket:03d}/default/agent_level_results.csv",
            5 * agents_per_bucket,
        )

    for shape in ("default", "uniform", "bimodal"):
        add(
            "population_shape/"
            f"population_shape_{shape}/agents_per_bucket_005/agent_level_results.csv",
            25,
        )

    for outlier_count in (0, 1, 5):
        add(
            f"outlier_sensitivity/outliers_{outlier_count:03d}/agent_level_results.csv",
            25,
        )

    for anchor_count in (0, 1, 2, 5, 10, 25):
        add(
            f"anchor_availability/anchor_agents_{anchor_count:03d}/agent_level_results.csv",
            50,
        )

    for decision_model in ("softmax", "rational", "flip", "deceptive"):
        add(
            "decision_model/"
            f"decision_model_{decision_model}/agents_per_bucket_005/agent_level_results.csv",
            25,
        )

    for correlation_slug in ("r_neg_0_9", "r_neg_0_5", "r_0_0", "r_pos_0_5", "r_pos_0_9"):
        add(
            "true_correlation/"
            f"true_correlation_{correlation_slug}/agents_per_bucket_005/agent_level_results.csv",
            25,
        )

    for grid_size in (11, 21, 41):
        add(
            f"grid_resolution/grid_{grid_size:03d}x{grid_size:03d}/agent_level_results.csv",
            25,
        )

    for stress in ("default", "moderate_compound_stress", "strong_compound_stress"):
        add(
            "compound_stress/"
            f"compound_stress_{stress}/agents_per_bucket_005/agent_level_results.csv",
            25,
        )

    if len(scenarios) != DEFAULT_EXPECTED_SCENARIOS:
        raise RuntimeError(
            f"Canonical scenario map has {len(scenarios)} entries; "
            f"expected {DEFAULT_EXPECTED_SCENARIOS}."
        )
    rows_per_seed = sum(scenarios.values())
    if rows_per_seed != DEFAULT_EXPECTED_ROWS_PER_SEED:
        raise RuntimeError(
            f"Canonical scenario map has {rows_per_seed} rows per seed; "
            f"expected {DEFAULT_EXPECTED_ROWS_PER_SEED}."
        )
    return scenarios


CANONICAL_SCENARIO_AGENT_COUNTS = _canonical_scenario_agent_counts()

REQUIRED_COLUMNS = {
    "seed",
    "agent_id",
    "count_bucket",
    "num_observations",
    "sigma_true",
    "log_lambda_true",
    "rationality_percent_true",
    "jeeds_posterior_mean_sigma",
    "jeeds_posterior_mean_log_lambda",
    "jeeds_rationality_percent",
    "jeeds_status",
    "hierarchical_posterior_mean_sigma",
    "hierarchical_posterior_mean_log_lambda",
    "hierarchical_rationality_percent",
    "hierarchical_status",
    "notes",
}

FINITE_COLUMNS = (
    "sigma_true",
    "log_lambda_true",
    "rationality_percent_true",
    "jeeds_posterior_mean_sigma",
    "jeeds_posterior_mean_log_lambda",
    "jeeds_rationality_percent",
    "hierarchical_posterior_mean_sigma",
    "hierarchical_posterior_mean_log_lambda",
    "hierarchical_rationality_percent",
)

RATIONALITY_BOUND_TOLERANCE = 1e-9


@dataclass(frozen=True)
class FileValidation:
    path: str
    num_rows: int
    num_seeds: int
    agents_per_seed: int


def _parse_int(row: dict[str, str], column: str, path: Path, row_number: int) -> int:
    try:
        return int(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{row_number}: invalid integer {column}={row.get(column)!r}") from exc


def _require_finite(row: dict[str, str], column: str, path: Path, row_number: int) -> None:
    raw = row.get(column, "")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{row_number}: invalid numeric {column}={raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{path}:{row_number}: non-finite {column}={raw!r}")


def validate_agent_csv(
    path: Path,
    expected_seeds: set[int],
    *,
    expected_environment: str | None = None,
) -> FileValidation:
    """Validate one scenario CSV and return its dimensions."""

    identities: set[tuple[int, int]] = set()
    agents_by_seed: dict[int, set[int]] = {}
    num_rows = 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing_columns = sorted(REQUIRED_COLUMNS - columns)
        if missing_columns:
            raise ValueError(f"{path}: missing required columns {missing_columns}")
        if expected_environment is not None and "environment" not in columns:
            raise ValueError(f"{path}: missing required environment column")

        for row_number, row in enumerate(reader, start=2):
            seed = _parse_int(row, "seed", path, row_number)
            agent_id = _parse_int(row, "agent_id", path, row_number)
            if (
                expected_environment is not None
                and row.get("environment") != expected_environment
            ):
                raise ValueError(
                    f"{path}:{row_number}: environment={row.get('environment')!r}; "
                    f"expected {expected_environment!r}"
                )
            count_bucket = _parse_int(row, "count_bucket", path, row_number)
            num_observations = _parse_int(row, "num_observations", path, row_number)
            if count_bucket != num_observations:
                raise ValueError(
                    f"{path}:{row_number}: count_bucket={count_bucket} differs from "
                    f"num_observations={num_observations}"
                )
            identity = (seed, agent_id)
            if identity in identities:
                raise ValueError(f"{path}:{row_number}: duplicate seed/agent identity {identity}")
            identities.add(identity)
            agents_by_seed.setdefault(seed, set()).add(agent_id)

            for status_column in ("jeeds_status", "hierarchical_status"):
                if row.get(status_column) != "ok":
                    raise ValueError(
                        f"{path}:{row_number}: {status_column}={row.get(status_column)!r}; expected 'ok'"
                    )
            for numeric_column in FINITE_COLUMNS:
                _require_finite(row, numeric_column, path, row_number)

            for rationality_column in (
                "rationality_percent_true",
                "jeeds_rationality_percent",
                "hierarchical_rationality_percent",
            ):
                rationality_value = float(row[rationality_column])
                if not (
                    -RATIONALITY_BOUND_TOLERANCE
                    <= rationality_value
                    <= 100.0 + RATIONALITY_BOUND_TOLERANCE
                ):
                    raise ValueError(
                        f"{path}:{row_number}: {rationality_column}={rationality_value} "
                        "is outside [0, 100]"
                    )

            notes = row.get("notes", "")
            if "population_fit: converged=True; selected=optimizer;" not in notes:
                raise ValueError(
                    f"{path}:{row_number}: missing a converged optimizer-selected "
                    "post-fix population-fit diagnostic in notes"
                )

            if "decision_model_rational" in path.parts:
                rationality_truth = float(row["rationality_percent_true"])
                if not math.isclose(rationality_truth, 100.0, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(
                        f"{path}:{row_number}: rational-policy behavioral truth must be "
                        f"100%, found {rationality_truth}"
                    )
            num_rows += 1

    actual_seeds = set(agents_by_seed)
    if actual_seeds != expected_seeds:
        missing = sorted(expected_seeds - actual_seeds)
        unexpected = sorted(actual_seeds - expected_seeds)
        raise ValueError(f"{path}: seed mismatch; missing={missing}, unexpected={unexpected}")
    if not agents_by_seed:
        raise ValueError(f"{path}: no data rows")

    reference_seed = min(expected_seeds)
    reference_agents = agents_by_seed[reference_seed]
    if reference_agents != set(range(len(reference_agents))):
        raise ValueError(f"{path}: agent IDs are not contiguous from zero for seed {reference_seed}")
    for seed, agent_ids in agents_by_seed.items():
        if agent_ids != reference_agents:
            raise ValueError(f"{path}: agent-ID set for seed {seed} differs from seed {reference_seed}")

    return FileValidation(
        path=str(path),
        num_rows=num_rows,
        num_seeds=len(actual_seeds),
        agents_per_seed=len(reference_agents),
    )


def validate_publication_root(
    root: Path,
    *,
    first_seed: int,
    num_seeds: int,
    expected_scenarios: int = DEFAULT_EXPECTED_SCENARIOS,
    expected_rows_per_seed: int = DEFAULT_EXPECTED_ROWS_PER_SEED,
    expected_scenario_agents: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Validate all concrete scenario CSVs beneath ``root``."""

    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive")
    root = root.resolve()
    scenario_files = sorted(root.rglob("agent_level_results.csv"))
    if (
        expected_scenario_agents is None
        and expected_scenarios == DEFAULT_EXPECTED_SCENARIOS
        and expected_rows_per_seed == DEFAULT_EXPECTED_ROWS_PER_SEED
    ):
        expected_scenario_agents = CANONICAL_SCENARIO_AGENT_COUNTS

    actual_relative_paths = {
        path.relative_to(root).as_posix(): path
        for path in scenario_files
    }
    if expected_scenario_agents is not None:
        expected_relative_paths = set(expected_scenario_agents)
        actual_path_set = set(actual_relative_paths)
        missing_paths = sorted(expected_relative_paths - actual_path_set)
        unexpected_paths = sorted(actual_path_set - expected_relative_paths)
        if missing_paths or unexpected_paths:
            raise ValueError(
                f"{root}: scenario path mismatch; "
                f"missing={missing_paths or 'none'}, "
                f"unexpected={unexpected_paths or 'none'}"
            )

    if len(scenario_files) != expected_scenarios:
        raise ValueError(
            f"{root}: expected {expected_scenarios} scenario agent CSVs, found {len(scenario_files)}"
        )
    expected_seeds = set(range(first_seed, first_seed + num_seeds))
    validations = [
        validate_agent_csv(
            path,
            expected_seeds,
            expected_environment=("1d" if expected_scenario_agents is not None else None),
        )
        for path in scenario_files
    ]
    if expected_scenario_agents is not None:
        validation_by_path = {
            Path(item.path).relative_to(root).as_posix(): item
            for item in validations
        }
        population_mismatches = {
            relative_path: {
                "actual": validation_by_path[relative_path].agents_per_seed,
                "expected": expected_agents,
            }
            for relative_path, expected_agents in expected_scenario_agents.items()
            if validation_by_path[relative_path].agents_per_seed != expected_agents
        }
        if population_mismatches:
            raise ValueError(
                f"{root}: per-scenario agent-count mismatch: {population_mismatches}"
            )
    total_rows = sum(item.num_rows for item in validations)
    expected_total_rows = expected_rows_per_seed * num_seeds
    if total_rows != expected_total_rows:
        raise ValueError(
            f"{root}: expected {expected_total_rows} total scenario rows, found {total_rows}"
        )
    return {
        "root": str(root.resolve()),
        "first_seed": first_seed,
        "num_seeds": num_seeds,
        "num_scenarios": len(validations),
        "total_rows": total_rows,
        "status": "ok",
        "files": [asdict(item) for item in validations],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, default=12345)
    parser.add_argument("--num-seeds", type=int, required=True)
    parser.add_argument("--expected-scenarios", type=int, default=DEFAULT_EXPECTED_SCENARIOS)
    parser.add_argument("--expected-rows-per-seed", type=int, default=DEFAULT_EXPECTED_ROWS_PER_SEED)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = validate_publication_root(
        args.results_root,
        first_seed=args.first_seed,
        num_seeds=args.num_seeds,
        expected_scenarios=args.expected_scenarios,
        expected_rows_per_seed=args.expected_rows_per_seed,
    )
    report_path = args.report or args.results_root / "publication_result_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"[validate-publication-results] OK: {report['num_scenarios']} scenarios, "
        f"{report['total_rows']} rows; wrote {report_path.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
