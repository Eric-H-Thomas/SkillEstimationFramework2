#!/usr/bin/env python3
# Paper correspondence: Main `subsec:baseball`; validates the publication runtime optimization.
"""Benchmark and validate optimized MLB inference on a few real pitches.

This script intentionally runs both the legacy per-target RNN loop and the
batched implementation, then compares utility surfaces, likelihood grids, and
JEEDS posterior means. It never writes publication results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HJEEDS.baseball_likelihood import (
    compute_baseball_log_likelihood_grid,
    compute_baseball_log_likelihood_grids_by_prefix,
)
from HJEEDS.baseball_pitch import (
    DEFAULT_DELTA,
    DEFAULT_RNN_INFERENCE_BATCH_SIZE,
    build_baseball_runtime,
    build_execution_skill_grid,
    build_log_lambda_grid,
    build_pitch_observation,
    get_agent_pitch_rows,
)
from HJEEDS.baseball_roster import load_statcast_for_roster
from HJEEDS.estimation import run_independent_jeeds_baseline


def _timed(function):
    start = time.perf_counter()
    value = function()
    return value, time.perf_counter() - start


def _estimate_payload(estimate) -> dict[str, Any]:
    return {
        "status": estimate.status,
        "posterior_mean_sigma": estimate.posterior_mean_sigma,
        "posterior_mean_log_lambda": estimate.posterior_mean_log_lambda,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pitcher-id", type=int, default=521230)
    parser.add_argument("--pitch-type", default="FF")
    parser.add_argument("--season-year", type=int, default=2021)
    parser.add_argument("--num-pitches", type=int, default=2)
    parser.add_argument(
        "--rnn-batch-size",
        type=int,
        default=DEFAULT_RNN_INFERENCE_BATCH_SIZE,
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_pitches < 1:
        raise ValueError("--num-pitches must be positive.")

    sigma_grid = build_execution_skill_grid(DEFAULT_DELTA)
    log_lambda_grid = build_log_lambda_grid()
    execution_skills = tuple(float(value) for value in sigma_grid)

    all_data, data_seconds = _timed(lambda: load_statcast_for_roster(args.season_year))
    rows = get_agent_pitch_rows(
        all_data,
        args.pitcher_id,
        args.pitch_type,
        max_rows=args.num_pitches,
    )
    if len(rows) != args.num_pitches:
        raise ValueError(
            f"Requested {args.num_pitches} pitches but found {len(rows)} for "
            f"pitcher={args.pitcher_id}, pitch_type={args.pitch_type}."
        )
    runtime, runtime_seconds = _timed(
        lambda: build_baseball_runtime(
            np.random.default_rng(12345),
            execution_skills,
            delta=DEFAULT_DELTA,
        )
    )

    legacy_observations, legacy_rnn_seconds = _timed(
        lambda: [
            build_pitch_observation(
                row,
                runtime,
                execution_skills,
                inference_batch_size=None,
            )
            for _, row in rows.iterrows()
        ]
    )
    optimized_observations, optimized_rnn_seconds = _timed(
        lambda: [
            build_pitch_observation(
                row,
                runtime,
                execution_skills,
                inference_batch_size=args.rnn_batch_size,
            )
            for _, row in rows.iterrows()
        ]
    )

    max_ev_abs_difference = 0.0
    for legacy, optimized in zip(legacy_observations, optimized_observations):
        if legacy.evs_per_execution_skill.keys() != optimized.evs_per_execution_skill.keys():
            raise AssertionError("Legacy and optimized execution-skill keys differ.")
        for key in legacy.evs_per_execution_skill:
            left = legacy.evs_per_execution_skill[key]
            right = optimized.evs_per_execution_skill[key]
            max_ev_abs_difference = max(
                max_ev_abs_difference,
                float(np.max(np.abs(left - right))),
            )
            np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)

    likelihood_kwargs = {
        "possible_targets_feet": runtime.grids.possible_targets_feet,
        "all_covs": runtime.all_covs,
        "sigma_grid": sigma_grid,
        "log_lambda_grid": log_lambda_grid,
        "delta": DEFAULT_DELTA,
    }
    prefixes = tuple(range(1, args.num_pitches + 1))
    repeated_grids, repeated_likelihood_seconds = _timed(
        lambda: {
            prefix: compute_baseball_log_likelihood_grid(
                pitch_observations=optimized_observations[:prefix],
                **likelihood_kwargs,
            )
            for prefix in prefixes
        }
    )
    cumulative_grids, cumulative_likelihood_seconds = _timed(
        lambda: compute_baseball_log_likelihood_grids_by_prefix(
            pitch_observations=optimized_observations,
            prefix_lengths=prefixes,
            **likelihood_kwargs,
        )
    )
    for prefix in prefixes:
        np.testing.assert_array_equal(cumulative_grids[prefix], repeated_grids[prefix])

    estimate_comparisons: dict[str, Any] = {}
    for prefix in prefixes:
        legacy_grid = compute_baseball_log_likelihood_grid(
            pitch_observations=legacy_observations[:prefix],
            **likelihood_kwargs,
        )
        legacy_estimate = run_independent_jeeds_baseline(
            log_likelihood_grid=legacy_grid,
            sigma_grid=sigma_grid,
            log_lambda_grid=log_lambda_grid,
        )
        optimized_estimate = run_independent_jeeds_baseline(
            log_likelihood_grid=cumulative_grids[prefix],
            sigma_grid=sigma_grid,
            log_lambda_grid=log_lambda_grid,
        )
        if legacy_estimate.status != optimized_estimate.status:
            raise AssertionError(
                f"Estimator status differs at prefix {prefix}: "
                f"{legacy_estimate.status!r} vs {optimized_estimate.status!r}."
            )
        sigma_difference = abs(
            float(legacy_estimate.posterior_mean_sigma)
            - float(optimized_estimate.posterior_mean_sigma)
        )
        log_lambda_difference = abs(
            float(legacy_estimate.posterior_mean_log_lambda)
            - float(optimized_estimate.posterior_mean_log_lambda)
        )
        if sigma_difference > 1e-6 or log_lambda_difference > 1e-6:
            raise AssertionError(
                f"Posterior means differ beyond tolerance at prefix {prefix}: "
                f"sigma={sigma_difference}, log_lambda={log_lambda_difference}."
            )
        estimate_comparisons[str(prefix)] = {
            "legacy": _estimate_payload(legacy_estimate),
            "optimized": _estimate_payload(optimized_estimate),
            "absolute_sigma_difference": sigma_difference,
            "absolute_log_lambda_difference": log_lambda_difference,
        }

    payload = {
        "status": "validated",
        "pitcher_id": args.pitcher_id,
        "pitch_type": args.pitch_type,
        "season_year": args.season_year,
        "num_pitches": args.num_pitches,
        "rnn_batch_size": args.rnn_batch_size,
        "max_ev_absolute_difference": max_ev_abs_difference,
        "timing_seconds": {
            "load_data": data_seconds,
            "build_runtime": runtime_seconds,
            "legacy_rnn_observations": legacy_rnn_seconds,
            "batched_rnn_observations": optimized_rnn_seconds,
            "rnn_speedup": legacy_rnn_seconds / optimized_rnn_seconds,
            "repeated_checkpoint_likelihoods": repeated_likelihood_seconds,
            "cumulative_checkpoint_likelihoods": cumulative_likelihood_seconds,
            "likelihood_speedup": repeated_likelihood_seconds / cumulative_likelihood_seconds,
        },
        "estimate_comparisons": estimate_comparisons,
    }
    rendered = json.dumps(payload, indent=2, allow_nan=False)
    print(rendered, flush=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
