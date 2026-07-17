# This file has been fully reviewed by a human researcher as of 07/17/26 at 2:25 PM MDT.
"""Calibrate baseball hyperpriors from independent JEEDS posterior means.

Offline tool that fits independent JEEDS on a Statcast roster, then builds a
``HyperpriorConfig`` from the sample of posterior means
(``build_hyperpriors_from_jeeds_estimates``). Paper BBIP convergence does **not**
call this module; it loads the committed artifact via
``--hyperprior-preset baseball-2021-ff`` →
``HJEEDS/data/baseball_hyperpriors_2021_ff.json``. Re-running calibration writes
under ``HJEEDS/results/`` (gitignored); copy into ``HJEEDS/data/`` only if you
intend to update the paper prior.

Modes (``submit_hjeeds_baseball_hyperprior_calibration.sh``):
  ``--prepare-roster`` → per-agent ``--agent-index`` array → ``--aggregate-results``.
Local sequential: omit those flags to run all agents in one process.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HJEEDS.baseball_config import (
    DEFAULT_LAMBDA_MAX,
    DEFAULT_LAMBDA_MIN,
    DEFAULT_MIN_PITCHES_PER_AGENT,
    DEFAULT_NUM_LAMBDA_GRID,
    DEFAULT_NUM_SIGMA_GRID,
    DEFAULT_PITCHER_IDS,
    BaseballExperimentConfig,
    build_baseball_skill_grids,
)
from HJEEDS.baseball_hyperpriors import (
    build_hyperpriors_from_jeeds_estimates,
    hyperprior_config_to_dict,
    resolve_baseball_hyperpriors,
    true_population_from_hyperpriors,
    write_hyperprior_config,
)
from HJEEDS.baseball_likelihood import compute_baseball_log_likelihood_grid
from HJEEDS.baseball_pitch import (
    DEFAULT_DELTA,
    DEFAULT_EXECUTION_SKILL_MAX,
    DEFAULT_EXECUTION_SKILL_MIN,
    BaseballRuntime,
    StatcastAgentSpec,
    build_baseball_runtime,
    build_pitch_observations_for_rows,
    get_agent_pitch_rows,
)
from HJEEDS.baseball_roster import (
    BaseballRosterSelection,
    add_common_roster_arguments,
    load_statcast_for_roster,
    parse_pitch_types,
    print_eligible_agents,
    resolve_baseball_roster,
    roster_selector_kwargs_from_args,
)
from HJEEDS.config import ExperimentConfig, parse_seed_argument
from HJEEDS.estimation import run_independent_jeeds_baseline
from HJEEDS.models import HyperpriorConfig, MethodEstimate

DEFAULT_OUTPUT_DIR = Path("HJEEDS/results/baseball_hyperprior_calibration")
ROSTER_FILENAME = "calibration_roster.json"
ROSTER_METADATA_FILENAME = "calibration_roster_metadata.json"
AGENT_RESULTS_SUBDIR = "agents"

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


@dataclass(frozen=True)
class CalibrationContext:
    """Shared skill grids + baseball runtime for one calibration workload."""

    args: argparse.Namespace
    config: BaseballExperimentConfig
    all_data: pd.DataFrame
    sigma_grid: np.ndarray
    log_lambda_grid: np.ndarray
    runtime: BaseballRuntime


def parse_calibration_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run independent JEEDS on a Statcast roster and write suggested baseball hyperpriors. "
            "Supports local sequential runs, per-agent cluster tasks, and aggregation."
        )
    )
    parser.add_argument(
        "--seed",
        type=parse_seed_argument,
        default=12345,
        help=(
            "Base seed kept for ExperimentConfig / runtime API compatibility. Statcast "
            "JEEDS likelihoods are deterministic in seed (PDF via multivariate_normal.pdf); "
            "prefer a single fixed value."
        ),
    )
    parser.add_argument("--pitch-types", type=str, default="FF")
    parser.add_argument(
        "--pitcher-ids",
        type=str,
        default=",".join(str(pid) for pid in DEFAULT_PITCHER_IDS),
        help=(
            "Used when none of --all-eligible-agents / --top-pitchers / --bbip-extremes is set."
        ),
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
    parser.add_argument(
        "--prepare-roster",
        action="store_true",
        help="Write calibration_roster.json and exit (no JEEDS inference).",
    )
    parser.add_argument(
        "--agent-index",
        type=int,
        default=None,
        help="Run independent JEEDS for one roster agent (0-based index for Slurm arrays).",
    )
    parser.add_argument(
        "--aggregate-results",
        action="store_true",
        help="Combine per-agent outputs and write final hyperprior artifacts.",
    )
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


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _row_from_estimate_row(row: dict[str, Any]) -> MethodEstimate:
    return MethodEstimate(
        method_name="jeeds",
        posterior_mean_sigma=_optional_float(row.get("posterior_mean_sigma")),
        posterior_mean_log_lambda=_optional_float(row.get("posterior_mean_log_lambda")),
        map_sigma=_optional_float(row.get("map_sigma")),
        map_log_lambda=_optional_float(row.get("map_log_lambda")),
        status=str(row.get("status", "")),
        notes=str(row.get("notes", "")),
    )


def _min_pitches_from_args(args: argparse.Namespace) -> int:
    if args.min_pitches_per_agent is not None:
        return int(args.min_pitches_per_agent)
    return DEFAULT_MIN_PITCHES_PER_AGENT


def _resolve_roster_from_args(
    args: argparse.Namespace,
) -> tuple[BaseballRosterSelection, tuple[str, ...], int, pd.DataFrame]:
    pitch_types = parse_pitch_types(args.pitch_types)
    min_pitches = _min_pitches_from_args(args)
    all_data = load_statcast_for_roster(args.season_year)
    roster = resolve_baseball_roster(
        all_data=all_data,
        season_year=args.season_year,
        pitch_types=pitch_types,
        min_pitches_per_agent=min_pitches,
        max_agents=args.max_agents,
        output_dir=Path(args.output_dir),
        **roster_selector_kwargs_from_args(args),
    )
    return roster, pitch_types, min_pitches, all_data


def _build_calibration_config(
    args: argparse.Namespace,
    roster: BaseballRosterSelection,
    output_dir: Path,
) -> BaseballExperimentConfig:
    """Build a natural-count baseball config used only for skill grids / JEEDS."""

    hyperpriors = resolve_baseball_hyperpriors(preset="low-confidence", calibrated_path=None)
    base = ExperimentConfig(
        environment="baseball",
        seed=args.seed,
        num_seeds=1,
        num_agents=len(roster.agent_specs),
        count_buckets=(0,),
        agents_per_bucket=len(roster.agent_specs),
        delta=DEFAULT_DELTA,
        num_sigma_grid=DEFAULT_NUM_SIGMA_GRID,
        num_lambda_grid=DEFAULT_NUM_LAMBDA_GRID,
        sigma_min=DEFAULT_EXECUTION_SKILL_MIN,
        sigma_max=DEFAULT_EXECUTION_SKILL_MAX,
        lambda_min=DEFAULT_LAMBDA_MIN,
        lambda_max=DEFAULT_LAMBDA_MAX,
        output_dir=output_dir,
        environment_grids={},
        dry_run=False,
        min_success_regions=2,
        max_success_regions=6,
        min_region_width=0.25,
        hyperpriors=hyperpriors,
        true_population=true_population_from_hyperpriors(hyperpriors),
    )
    return BaseballExperimentConfig(
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


def _make_calibration_context(
    args: argparse.Namespace,
    *,
    roster: BaseballRosterSelection,
    all_data: pd.DataFrame,
    output_dir: Path,
) -> CalibrationContext:
    """Build skill grids and one shared runtime (same pattern as baseball_pipeline)."""

    config = _build_calibration_config(args, roster, output_dir)
    sigma_grid, log_lambda_grid = build_baseball_skill_grids(config)
    execution_skills = tuple(float(value) for value in sigma_grid)
    # Seed is API-only: getNormalDistribution uses N.pdf (mean/cov), never N.rvs().
    rng = np.random.default_rng(args.seed)
    runtime = build_baseball_runtime(rng, execution_skills, delta=config.base.delta)
    return CalibrationContext(
        args=args,
        config=config,
        all_data=all_data,
        sigma_grid=sigma_grid,
        log_lambda_grid=log_lambda_grid,
        runtime=runtime,
    )


def roster_path_for(output_dir: Path) -> Path:
    return output_dir / ROSTER_FILENAME


def agent_result_path_for(output_dir: Path, agent_id: int) -> Path:
    return output_dir / AGENT_RESULTS_SUBDIR / f"agent_{agent_id:04d}.json"


def write_calibration_roster(
    output_dir: Path,
    roster: BaseballRosterSelection,
    *,
    season_year: int | None,
    pitch_types: Sequence[str],
    min_pitches_per_agent: int,
    max_pitches_per_agent: int | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = roster_path_for(output_dir)
    payload = [
        {
            "agent_id": spec.agent_id,
            "pitcher_id": spec.pitcher_id,
            "pitch_type": spec.pitch_type,
        }
        for spec in roster.agent_specs
    ]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    metadata = {
        "season_year": season_year,
        "pitch_types": list(pitch_types),
        "min_pitches_per_agent": min_pitches_per_agent,
        "max_pitches_per_agent": max_pitches_per_agent,
        "num_agents": len(payload),
    }
    with (output_dir / ROSTER_METADATA_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return path


def load_calibration_roster(output_dir: Path) -> tuple[StatcastAgentSpec, ...]:
    path = roster_path_for(output_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"Calibration roster not found: {path}. Run with --prepare-roster first."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return tuple(
        StatcastAgentSpec(
            agent_id=int(row["agent_id"]),
            pitcher_id=int(row["pitcher_id"]),
            pitch_type=str(row["pitch_type"]),
        )
        for row in payload
    )


def _roster_selection_from_specs(
    args: argparse.Namespace,
    agent_specs: Sequence[StatcastAgentSpec],
) -> BaseballRosterSelection:
    pitch_types = parse_pitch_types(args.pitch_types)
    return BaseballRosterSelection(
        season_year=args.season_year,
        pitch_types=pitch_types,
        pitcher_ids=tuple(dict.fromkeys(spec.pitcher_id for spec in agent_specs)),
        agent_specs=tuple(agent_specs),
        agent_pitch_counts=(),
        excluded_agents=(),
    )


def run_single_agent_calibration(
    context: CalibrationContext,
    agent_spec: StatcastAgentSpec,
) -> dict[str, Any]:
    """Independent JEEDS for one (pitcher, pitchType); returns a CSV-shaped row."""

    agent_rows = get_agent_pitch_rows(
        context.all_data,
        agent_spec.pitcher_id,
        agent_spec.pitch_type,
        max_rows=context.args.max_pitches_per_agent,
    )
    pitch_observations = build_pitch_observations_for_rows(
        agent_rows,
        context.runtime,
        tuple(float(value) for value in context.sigma_grid),
    )
    if not pitch_observations:
        estimate = MethodEstimate(
            method_name="jeeds",
            status="no_data",
            notes="No pitches for agent.",
        )
    else:
        log_likelihood_grid = compute_baseball_log_likelihood_grid(
            pitch_observations=pitch_observations,
            possible_targets_feet=context.runtime.grids.possible_targets_feet,
            all_covs=context.runtime.all_covs,
            sigma_grid=context.sigma_grid,
            log_lambda_grid=context.log_lambda_grid,
            delta=context.config.base.delta,
        )
        estimate = run_independent_jeeds_baseline(
            log_likelihood_grid=log_likelihood_grid,
            sigma_grid=context.sigma_grid,
            log_lambda_grid=context.log_lambda_grid,
        )
    return _estimate_to_row(
        agent_spec.agent_id,
        agent_spec.pitcher_id,
        agent_spec.pitch_type,
        len(pitch_observations),
        estimate,
    )


def write_agent_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2)
        handle.write("\n")


def load_agent_results(output_dir: Path) -> tuple[list[dict[str, Any]], list[int]]:
    agents_dir = output_dir / AGENT_RESULTS_SUBDIR
    if not agents_dir.is_dir():
        raise FileNotFoundError(f"Missing per-agent results directory: {agents_dir}")

    roster = load_calibration_roster(output_dir)
    rows: list[dict[str, Any]] = []
    missing_agent_ids: list[int] = []
    for agent_spec in roster:
        path = agent_result_path_for(output_dir, agent_spec.agent_id)
        if not path.is_file():
            missing_agent_ids.append(agent_spec.agent_id)
            continue
        with path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    rows.sort(key=lambda row: int(row["agent_id"]))
    return rows, missing_agent_ids


def build_calibration_summary(
    rows: Sequence[dict[str, Any]],
    estimates: Sequence[MethodEstimate],
    *,
    confidence: str,
    season_year: int | None,
    pitch_types: Sequence[str],
    missing_agent_ids: Sequence[int],
    hyperpriors: HyperpriorConfig | None = None,
) -> dict[str, Any]:
    if hyperpriors is None:
        hyperpriors = build_hyperpriors_from_jeeds_estimates(estimates, confidence=confidence)
    ok_estimates = [
        estimate
        for estimate in estimates
        if estimate.status == "ok"
        and estimate.posterior_mean_sigma is not None
        and estimate.posterior_mean_log_lambda is not None
    ]
    log_sigmas = [float(np.log(estimate.posterior_mean_sigma)) for estimate in ok_estimates]
    log_lambdas = [float(estimate.posterior_mean_log_lambda) for estimate in ok_estimates]

    return {
        "season_year": season_year,
        "pitch_types": list(pitch_types),
        "num_agents": len(rows) + len(missing_agent_ids),
        "num_completed_agent_results": len(rows),
        "num_missing_agent_results": len(missing_agent_ids),
        "missing_agent_ids": list(missing_agent_ids),
        "num_successful_estimates": len(ok_estimates),
        "sample_mean_log_sigma": float(np.mean(log_sigmas)) if log_sigmas else None,
        "sample_mean_log_lambda": float(np.mean(log_lambdas)) if log_lambdas else None,
        "sample_std_log_sigma": float(np.std(log_sigmas, ddof=1)) if len(log_sigmas) > 1 else None,
        "sample_std_log_lambda": float(np.std(log_lambdas, ddof=1)) if len(log_lambdas) > 1 else None,
        "confidence": confidence,
        "suggested_hyperpriors": hyperprior_config_to_dict(hyperpriors),
    }


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


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    """Local sequential path: resolve roster, JEEDS every agent, write hyperpriors."""

    roster, pitch_types, _, all_data = _resolve_roster_from_args(args)
    context = _make_calibration_context(
        args,
        roster=roster,
        all_data=all_data,
        output_dir=Path(args.output_dir),
    )

    rows: list[dict[str, Any]] = []
    estimates: list[MethodEstimate] = []
    for agent_spec in roster.agent_specs:
        row = run_single_agent_calibration(context, agent_spec)
        rows.append(row)
        estimates.append(_row_from_estimate_row(row))

    hyperpriors = build_hyperpriors_from_jeeds_estimates(estimates, confidence=args.confidence)
    summary = build_calibration_summary(
        rows,
        estimates,
        confidence=args.confidence,
        season_year=args.season_year,
        pitch_types=pitch_types,
        missing_agent_ids=(),
        hyperpriors=hyperpriors,
    )
    return {"rows": rows, "summary": summary, "hyperpriors": hyperpriors}


def prepare_roster(args: argparse.Namespace) -> Path:
    """Write ``calibration_roster.json`` (+ metadata) for Slurm array indexing."""

    roster, pitch_types, min_pitches, _all_data = _resolve_roster_from_args(args)
    output_dir = Path(args.output_dir)
    path = write_calibration_roster(
        output_dir,
        roster,
        season_year=args.season_year,
        pitch_types=pitch_types,
        min_pitches_per_agent=min_pitches,
        max_pitches_per_agent=args.max_pitches_per_agent,
    )
    print(f"[baseball-calibrate] Wrote roster with {len(roster.agent_specs)} agents to {path.resolve()}")
    return path


def run_agent_index(args: argparse.Namespace) -> Path:
    """Cluster worker: JEEDS for ``calibration_roster.json[agent_index]`` only."""

    output_dir = Path(args.output_dir)
    agent_specs = load_calibration_roster(output_dir)
    if args.agent_index is None:
        raise ValueError("--agent-index is required for per-agent calibration tasks.")
    if args.agent_index < 0 or args.agent_index >= len(agent_specs):
        raise IndexError(
            f"agent_index={args.agent_index} is out of range for roster size {len(agent_specs)}."
        )

    agent_spec = agent_specs[args.agent_index]
    roster = _roster_selection_from_specs(args, agent_specs)
    all_data = load_statcast_for_roster(args.season_year)
    context = _make_calibration_context(
        args,
        roster=roster,
        all_data=all_data,
        output_dir=output_dir,
    )
    row = run_single_agent_calibration(context, agent_spec)
    out_path = agent_result_path_for(output_dir, agent_spec.agent_id)
    write_agent_result(out_path, row)
    print(
        f"[baseball-calibrate] agent_index={args.agent_index} "
        f"pitcher={agent_spec.pitcher_id} pitch_type={agent_spec.pitch_type} "
        f"status={row['status']} -> {out_path.resolve()}",
        flush=True,
    )
    return out_path


def aggregate_results(args: argparse.Namespace) -> dict[str, Any]:
    """Combine ``agents/agent_*.json`` into CSV + ``suggested_hyperpriors.json``."""

    output_dir = Path(args.output_dir)
    rows, missing_agent_ids = load_agent_results(output_dir)
    if missing_agent_ids:
        print(
            f"[baseball-calibrate] Warning: missing {len(missing_agent_ids)} agent result files.",
            flush=True,
        )

    estimates = [_row_from_estimate_row(row) for row in rows]
    metadata_path = output_dir / ROSTER_METADATA_FILENAME
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        pitch_types = metadata.get("pitch_types", parse_pitch_types(args.pitch_types))
        season_year = metadata.get("season_year", args.season_year)
    else:
        pitch_types = parse_pitch_types(args.pitch_types)
        season_year = args.season_year

    hyperpriors = build_hyperpriors_from_jeeds_estimates(estimates, confidence=args.confidence)
    summary = build_calibration_summary(
        rows,
        estimates,
        confidence=args.confidence,
        season_year=season_year,
        pitch_types=pitch_types,
        missing_agent_ids=missing_agent_ids,
        hyperpriors=hyperpriors,
    )
    payload = {"rows": rows, "summary": summary, "hyperpriors": hyperpriors}
    write_calibration_outputs(output_dir, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_calibration_args(argv)

    if args.list_eligible_pitchers:
        print_eligible_agents(
            season_year=args.season_year,
            pitch_types=parse_pitch_types(args.pitch_types),
            min_pitches=_min_pitches_from_args(args),
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
        print("Modes:")
        print("  --prepare-roster            write calibration_roster.json")
        print("  --agent-index N             one agent per Slurm array task")
        print("  --aggregate-results         combine agents/ and write hyperpriors")
        print("Artifacts:")
        print(f"  - {ROSTER_FILENAME}")
        print(f"  - {AGENT_RESULTS_SUBDIR}/agent_XXXX.json")
        print("  - jeeds_calibration_agent_estimates.csv")
        print("  - calibration_summary.json")
        print("  - suggested_hyperpriors.json")
        print(
            "Paper BBIP loads HJEEDS/data/baseball_hyperpriors_2021_ff.json "
            "via --hyperprior-preset baseball-2021-ff (does not re-run this tool)."
        )
        return 0

    if args.prepare_roster:
        prepare_roster(args)
        return 0

    if args.aggregate_results:
        payload = aggregate_results(args)
        isummary = payload["summary"]
        print(f"[baseball-calibrate] Wrote aggregated outputs to {Path(args.output_dir).resolve()}")
        print(
            "[baseball-calibrate] Suggested centers "
            f"(log sigma, log lambda): ({isummary['sample_mean_log_sigma']:.4f}, "
            f"{isummary['sample_mean_log_lambda']:.4f}) from {isummary['num_successful_estimates']} agents"
        )
        return 0

    if args.agent_index is not None:
        run_agent_index(args)
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
