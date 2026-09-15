# Paper correspondence: Supplement `app:baseball_hyperpriors`.
"""Reproduce the MLB decision-skill prior-predictive interpretability check.

The check uses 25 chronologically spaced 2021 four-seam-fastball contexts from
the canonical, hash-validated processed Statcast artifact.  For each context it
constructs the execution-adjusted expected-utility surface at the execution
grid point nearest 0.5 ft and computes the softmax policy's normalized expected
utility for several round candidate lambda values.  Zero denotes uniform target
selection and one denotes optimal target selection.  The selected center is the
candidate whose median normalized expected utility is closest to 0.80.

Run from the repository root:

    python -m HJEEDS.baseball_prior_predictive_check
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .baseball_pitch import (
    DEFAULT_DELTA,
    baseball_execution_skill_key,
    build_baseball_runtime,
    build_execution_skill_grid,
    build_pitch_observation,
    filter_statcast_by_season,
    load_processed_statcast,
)
from .baseball_provenance import load_processed_artifact_reference


DEFAULT_CANDIDATE_LAMBDAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
DEFAULT_OUTPUT = Path("HJEEDS/data/baseball_prior_predictive_check.json")
CONTEXT_ID_COLUMNS = (
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
)


def _parse_candidate_lambdas(raw: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("Candidate lambdas must be finite positive values.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Candidate lambdas must be unique.")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season-year", type=int, default=2021)
    parser.add_argument("--pitch-type", type=str, default="FF")
    parser.add_argument("--num-contexts", type=int, default=25)
    parser.add_argument("--sigma-center-feet", type=float, default=0.5)
    parser.add_argument(
        "--candidate-lambdas",
        type=_parse_candidate_lambdas,
        default=DEFAULT_CANDIDATE_LAMBDAS,
        help="Comma-separated positive values (default: 1,3,10,30,100,300,1000).",
    )
    parser.add_argument("--target-median-normalized-utility", type=float, default=0.80)
    parser.add_argument("--expected-selected-lambda", type=float, default=100.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _json_value(value: Any) -> Any:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def select_contexts(
    all_data: pd.DataFrame,
    *,
    season_year: int,
    pitch_type: str,
    num_contexts: int,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Select evenly spaced rows after deterministic chronological ordering."""

    if num_contexts < 2:
        raise ValueError("num_contexts must be at least two.")
    season_data = filter_statcast_by_season(all_data, season_year)
    eligible = season_data.loc[season_data["pitch_type"] == pitch_type]
    if len(eligible) < num_contexts:
        raise ValueError(
            f"Only {len(eligible)} rows match season={season_year}, pitch_type={pitch_type}; "
            f"cannot select {num_contexts} contexts."
        )
    ordering_columns = [
        column
        for column in ("game_date", "game_pk", "at_bat_number", "pitch_number")
        if column in eligible.columns
    ]
    if "game_date" not in ordering_columns:
        raise ValueError("Processed Statcast data is missing game_date.")
    eligible = eligible.sort_values(
        by=ordering_columns,
        ascending=[True] * len(ordering_columns),
        kind="mergesort",
    )
    positions = np.rint(np.linspace(0, len(eligible) - 1, num_contexts)).astype(int)
    if len(np.unique(positions)) != num_contexts:
        raise RuntimeError("Evenly spaced context positions were not unique.")
    return eligible.iloc[positions].copy(), positions, len(eligible)


def normalized_expected_utility(expected_utilities: np.ndarray, decision_skill: float) -> float:
    """Return softmax EV scaled from uniform selection (zero) to optimal (one)."""

    evs = np.asarray(expected_utilities, dtype=float).reshape(-1)
    if evs.size == 0 or np.any(~np.isfinite(evs)):
        raise ValueError("Expected-utility surface must be finite and nonempty.")
    scaled = decision_skill * evs
    weights = np.exp(scaled - float(np.max(scaled)))
    probabilities = weights / float(np.sum(weights))
    uniform_ev = float(np.mean(evs))
    optimal_ev = float(np.max(evs))
    denominator = optimal_ev - uniform_ev
    if denominator <= 0.0:
        raise ValueError("Expected-utility surface has no improvement over uniform target selection.")
    policy_ev = float(np.sum(probabilities * evs))
    return float((policy_ev - uniform_ev) / denominator)


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 < args.target_median_normalized_utility < 1.0:
        raise ValueError("target median normalized utility must lie strictly between zero and one.")
    if args.sigma_center_feet <= 0.0:
        raise ValueError("sigma center must be positive.")

    all_data = load_processed_statcast()
    contexts, positions, eligible_count = select_contexts(
        all_data,
        season_year=args.season_year,
        pitch_type=args.pitch_type,
        num_contexts=args.num_contexts,
    )
    sigma_grid = build_execution_skill_grid(DEFAULT_DELTA)
    sigma = float(sigma_grid[int(np.argmin(np.abs(sigma_grid - args.sigma_center_feet)))])
    runtime = build_baseball_runtime(np.random.default_rng(12345), (sigma,), delta=DEFAULT_DELTA)

    expected_utility_surfaces: list[np.ndarray] = []
    context_records: list[dict[str, Any]] = []
    for selected_index, ((artifact_index, row), filtered_position) in enumerate(
        zip(contexts.iterrows(), positions, strict=True)
    ):
        observation = build_pitch_observation(row, runtime, (sigma,))
        expected_utilities = observation.evs_per_execution_skill[baseball_execution_skill_key(sigma)]
        expected_utility_surfaces.append(expected_utilities)
        identifiers = {
            column: _json_value(row[column])
            for column in CONTEXT_ID_COLUMNS
            if column in row.index
        }
        context_records.append(
            {
                "selected_context_index": selected_index,
                "filtered_artifact_position": int(filtered_position),
                "artifact_dataframe_index": _json_value(artifact_index),
                **identifiers,
            }
        )

    candidate_results: list[dict[str, Any]] = []
    for decision_skill in args.candidate_lambdas:
        normalized_utilities = [
            normalized_expected_utility(expected_utilities, decision_skill)
            for expected_utilities in expected_utility_surfaces
        ]
        candidate_results.append(
            {
                "lambda": float(decision_skill),
                "normalized_expected_utilities": normalized_utilities,
                **_summary(normalized_utilities),
            }
        )

    selected = min(
        candidate_results,
        key=lambda result: (
            abs(result["median"] - args.target_median_normalized_utility),
            result["lambda"],
        ),
    )
    reference = load_processed_artifact_reference()
    payload = {
        "purpose": "MLB decision-skill prior-predictive interpretability check",
        "relationship_to_frozen_hyperprior_artifact": (
            "This reproducible audit corrects the older provenance label in the hash-bound "
            "baseball_hyperpriors_literature_informed.json: the approximately 80 percent "
            "quantity is normalized expected utility, not the literal probability assigned "
            "to one optimal grid target. The numerical hyperprior is unchanged."
        ),
        "selection_rule": (
            "Choose the round candidate lambda whose median normalized expected utility "
            f"is closest to {args.target_median_normalized_utility:.2f}; zero is uniform "
            "target selection and one is optimal target selection."
        ),
        "selected_lambda": selected["lambda"],
        "season_year": args.season_year,
        "pitch_type": args.pitch_type,
        "num_contexts": args.num_contexts,
        "eligible_context_count": eligible_count,
        "context_selection": (
            "Evenly spaced positions after deterministic ascending ordering by "
            "game_date, game_pk, at_bat_number, and pitch_number."
        ),
        "sigma_requested_feet": args.sigma_center_feet,
        "sigma_grid_value_feet": sigma,
        "target_median_normalized_utility": args.target_median_normalized_utility,
        "processed_pickle_sha256": reference["pickle_sha256"],
        "model_weights_sha256": reference["model_weights_sha256"],
        "contexts": context_records,
        "candidate_results": candidate_results,
    }
    if not math.isclose(selected["lambda"], args.expected_selected_lambda, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"Check selected lambda={selected['lambda']}, not expected "
            f"lambda={args.expected_selected_lambda}. Review the candidate set and paper claim."
        )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_check(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    selected = next(
        result for result in payload["candidate_results"] if result["lambda"] == payload["selected_lambda"]
    )
    print(f"Selected lambda: {payload['selected_lambda']:g}")
    print(
        "Normalized expected utility at selected lambda: "
        f"median={100.0 * selected['median']:.2f}%, "
        f"IQR={100.0 * selected['q1']:.2f}%--{100.0 * selected['q3']:.2f}%"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
