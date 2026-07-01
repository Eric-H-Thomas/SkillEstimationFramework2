# This file was written or edited by AI and still requires human review. Delete this comment when done.
"""Calibrate baseball hyperpriors from independent JEEDS posterior means."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HJEEDS.baseball_config import DEFAULT_MIN_PITCHES_PER_AGENT, build_baseball_skill_grids
from HJEEDS.baseball_hyperpriors import (
    build_hyperpriors_from_jeeds_estimates,
    hyperprior_config_to_dict,
    write_hyperprior_config,
)
from HJEEDS.baseball_likelihood import compute_baseball_log_likelihood_grid
from HJEEDS.baseball_pitch import build_baseball_runtime, build_pitch_observation, get_agent_pitch_rows
from HJEEDS.baseball_roster import (
    add_common_roster_arguments,
    load_statcast_for_roster,
    parse_pitch_types,
    print_eligible_agents,
    resolve_baseball_roster,
)
from HJEEDS.config import parse_seed_argument
from HJEEDS.estimation import run_independent_jeeds_baseline
from HJEEDS.models import MethodEstimate

DEFAULT_OUTPUT_DIR = Path("HJEEDS/results/baseball_hyperprior_calibration")

JEEDS_CALIBRATION_HEADER = [
    "agent_id",
    "pitcher_id",
    "pitch_type",
    "num_observations",
    "posterior_mean_sigma",
    "posterior_mean_log_lambda",
    "map_sigma",
    "map_log_lambda",
    "status",
    "notes",
]


def parse_calibration_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run independent JEEDS on a Statcast roster and write suggested baseball hyperpriors."
        )
    )
    parser.add_argument("--seed", type=parse_seed_argument, default=12345)
    parser.add_argument("--pitch-types", type=str, default="FF")
    parser.add_argument(
        "--pitcher-ids",
        type=str,
        default="543037,453286",
        help="Used when neither --all-eligible-agents nor --top-pitchers is set.",
    )
    parser.add_argument(
        "--max-pitches-per-agent",
        type=int,
        default=None,
        help="Cap pitches per agent when fitting independent JEEDS (newest first).",
    )
    parser.add_argument(
        "--confidence",
        choices=("low", "darts"),
        default="low",
        help="How tight the suggested hyperprior should be around the sample moments.",
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    add_common_roster_arguments(parser)
    return parser.parse_args(argv)


def _estimate_to_row(
    agent_id: int,
    pitcher_id: int,
    pitch_type: str,
    num_observations: int,
    estimate: MethodEstimate,
) -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "pitcher_id": pitcher_id,
        "pitch_type": pitch_type,
        "num_observations": num_observations,
        "posterior_mean_sigma": estimate.posterior_mean_sigma,
        "posterior_mean_log_lambda": estimate.posterior_mean_log_lambda,
        "map_sigma": estimate.map_sigma,
        "map_log_lambda": estimate.map_log_lambda,
        "status": estimate.status,
        "notes": estimate.notes,
    }


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    pitch_types = parse_pitch_types(args.pitch_types)
    min_pitches = (
        args.min_pitches_per_agent
        if args.min_pitches_per_agent is not None
        else DEFAULT_MIN_PITCHES_PER_AGENT
    )
    all_data = load_statcast_for_roster(args.season_year)

    if args.all_eligible_agents:
        roster_selector = dict(all_eligible_agents=True, pitcher_ids=None, top_pitchers=None)
    elif args.top_pitchers is not None:
        roster_selector = dict(all_eligible_agents=False, pitcher_ids=None, top_pitchers=args.top_pitchers)
    else:
        pitcher_ids = tuple(int(piece.strip()) for piece in args.pitcher_ids.split(",") if piece.strip())
        roster_selector = dict(all_eligible_agents=False, pitcher_ids=pitcher_ids, top_pitchers=None)

    roster = resolve_baseball_roster(
        all_data=all_data,
        season_year=args.season_year,
        pitch_types=pitch_types,
        min_pitches_per_agent=min_pitches,
        max_agents=args.max_agents,
        **roster_selector,
    )

    from HJEEDS.baseball_config import BaseballExperimentConfig
    from HJEEDS.baseball_hyperpriors import resolve_baseball_hyperpriors
    from HJEEDS.config import ExperimentConfig, TruePopulationConfig
    import math

    hyperpriors = resolve_baseball_hyperpriors(preset="low-confidence", calibrated_path=None)
    base = ExperimentConfig(
        environment="baseball",
        seed=args.seed,
        num_seeds=1,
        num_agents=len(roster.agent_specs),
        count_buckets=(0,),
        agents_per_bucket=len(roster.agent_specs),
        delta=0.0417,
        num_sigma_grid=21,
        num_lambda_grid=21,
        sigma_min=0.17,
        sigma_max=2.81,
        lambda_min=1e-3,
        lambda_max=10**3.6,
        output_dir=Path(args.output_dir),
        environment_grids={},
        dry_run=False,
        min_success_regions=2,
        max_success_regions=6,
        min_region_width=0.25,
        hyperpriors=hyperpriors,
        true_population=TruePopulationConfig(
            mean_log_sigma=hyperpriors.mean_vector[0],
            mean_log_lambda=hyperpriors.mean_vector[1],
            tau_eta=math.exp(hyperpriors.log_tau_eta_mean),
            tau_rho=math.exp(hyperpriors.log_tau_rho_mean),
            correlation=math.tanh(hyperpriors.m_r),
        ),
    )
    config = BaseballExperimentConfig(
        base=base,
        season_year=args.season_year,
        pitcher_ids=roster.pitcher_ids,
        pitch_types=roster.pitch_types,
        max_pitches_per_agent=args.max_pitches_per_agent,
        use_natural_pitch_counts=True,
        agent_specs=roster.agent_specs,
        agents=tuple((spec.pitcher_id, spec.pitch_type) for spec in roster.agent_specs),
        agent_pitch_counts=roster.agent_pitch_counts,
        excluded_agents=roster.excluded_agents,
    )

    rng = np.random.default_rng(args.seed)
    sigma_grid, log_lambda_grid = build_baseball_skill_grids(config)
    execution_skills = tuple(float(value) for value in sigma_grid)
    runtime = build_baseball_runtime(rng, execution_skills, delta=config.base.delta)

    rows: list[dict[str, Any]] = []
    estimates: list[MethodEstimate] = []
    for agent_spec in roster.agent_specs:
        agent_rows = get_agent_pitch_rows(
            all_data,
            agent_spec.pitcher_id,
            agent_spec.pitch_type,
            max_rows=args.max_pitches_per_agent,
        )
        pitch_observations = [
            build_pitch_observation(row, runtime, execution_skills) for _, row in agent_rows.iterrows()
        ]
        if not pitch_observations:
            estimate = MethodEstimate(method_name="jeeds", status="no_data", notes="No pitches for agent.")
        else:
            log_likelihood_grid = compute_baseball_log_likelihood_grid(
                pitch_observations=pitch_observations,
                possible_targets_feet=runtime.grids.possible_targets_feet,
                all_covs=runtime.all_covs,
                sigma_grid=sigma_grid,
                log_lambda_grid=log_lambda_grid,
                delta=config.base.delta,
            )
            estimate = run_independent_jeeds_baseline(
                log_likelihood_grid=log_likelihood_grid,
                sigma_grid=sigma_grid,
                log_lambda_grid=log_lambda_grid,
            )
        rows.append(
            _estimate_to_row(
                agent_spec.agent_id,
                agent_spec.pitcher_id,
                agent_spec.pitch_type,
                len(pitch_observations),
                estimate,
            )
        )
        estimates.append(estimate)

    hyperpriors = build_hyperpriors_from_jeeds_estimates(estimates, confidence=args.confidence)
    ok_estimates = [
        estimate
        for estimate in estimates
        if estimate.status == "ok"
        and estimate.posterior_mean_sigma is not None
        and estimate.posterior_mean_log_lambda is not None
    ]
    log_sigmas = [float(np.log(estimate.posterior_mean_sigma)) for estimate in ok_estimates]
    log_lambdas = [float(estimate.posterior_mean_log_lambda) for estimate in ok_estimates]

    summary = {
        "season_year": args.season_year,
        "pitch_types": list(pitch_types),
        "num_agents": len(rows),
        "num_successful_estimates": len(ok_estimates),
        "sample_mean_log_sigma": float(np.mean(log_sigmas)) if log_sigmas else None,
        "sample_mean_log_lambda": float(np.mean(log_lambdas)) if log_lambdas else None,
        "sample_std_log_sigma": float(np.std(log_sigmas, ddof=1)) if len(log_sigmas) > 1 else None,
        "sample_std_log_lambda": float(np.std(log_lambdas, ddof=1)) if len(log_lambdas) > 1 else None,
        "confidence": args.confidence,
        "suggested_hyperpriors": hyperprior_config_to_dict(hyperpriors),
    }
    return {"rows": rows, "summary": summary, "hyperpriors": hyperpriors}


def write_calibration_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "jeeds_calibration_agent_estimates.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=JEEDS_CALIBRATION_HEADER)
        writer.writeheader()
        writer.writerows(payload["rows"])

    summary_path = output_dir / "calibration_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload["summary"], handle, indent=2)
        handle.write("\n")

    write_hyperprior_config(output_dir / "suggested_hyperpriors.json", payload["hyperpriors"])


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_calibration_args(argv)

    if args.list_eligible_pitchers:
        min_pitches = (
            args.min_pitches_per_agent
            if args.min_pitches_per_agent is not None
            else DEFAULT_MIN_PITCHES_PER_AGENT
        )
        print_eligible_agents(
            season_year=args.season_year,
            pitch_types=parse_pitch_types(args.pitch_types),
            min_pitches=min_pitches,
            limit=args.list_eligible_limit,
        )
        return 0

    if args.dry_run:
        print("=== DRY RUN: Baseball Hyperprior Calibration ===")
        print(f"Season year: {args.season_year or 'all seasons in pickle'}")
        print(f"Pitch types: {args.pitch_types}")
        print(f"Max agents: {args.max_agents or 'no cap'}")
        print(f"Max pitches per agent: {args.max_pitches_per_agent or 'all available'}")
        print(f"Output directory: {Path(args.output_dir).resolve()}")
        print("Artifacts:")
        print("  - jeeds_calibration_agent_estimates.csv")
        print("  - calibration_summary.json")
        print("  - suggested_hyperpriors.json")
        return 0

    payload = run_calibration(args)
    output_dir = Path(args.output_dir)
    write_calibration_outputs(output_dir, payload)
    isummary = payload["summary"]
    print(f"[baseball-calibrate] Wrote calibration outputs to {output_dir.resolve()}")
    print(
        "[baseball-calibrate] Suggested centers "
        f"(log sigma, log lambda): ({isummary['sample_mean_log_sigma']:.4f}, "
        f"{isummary['sample_mean_log_lambda']:.4f}) from {isummary['num_successful_estimates']} agents"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
