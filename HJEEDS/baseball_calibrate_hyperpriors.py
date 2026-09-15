"""Calibrate baseball hyperpriors from independent JEEDS posterior means.

Offline tool that fits independent JEEDS on a Statcast roster, then builds a
``HyperpriorConfig`` from the sample of posterior means
(``build_hyperpriors_from_jeeds_estimates``). Paper BBIP convergence does **not**
call this module; it loads the committed artifact via
``--hyperprior-preset baseball-2021-ff`` →
``HJEEDS/data/baseball_hyperpriors_2021_ff.json``. Re-running calibration writes
under ``HJEEDS/results/`` (gitignored). Use ``--copy-validated-hyperpriors-to``
to install a replacement paper prior only after the canonical 528/528 run and
all output hashes validate.

Modes (``submit_hjeeds_baseball_hyperprior_calibration.sh``):
  ``--prepare-roster`` → per-agent ``--agent-index`` array → ``--aggregate-results``.
Local sequential: omit those flags to run all agents in one process.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HJEEDS.artifacts import _optional_float
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
    load_hyperprior_config,
    resolve_baseball_hyperpriors,
    true_population_from_hyperpriors,
    write_hyperprior_config,
)
from HJEEDS.baseball_likelihood import (
    add_independent_log_likelihood_grids,
    compute_baseball_log_likelihood_grid,
)
from HJEEDS.baseball_provenance import (
    PAPER_CALIBRATION_ROSTER_SHA256,
    build_baseball_run_provenance,
    canonical_json_sha256,
    roster_fingerprint,
)
from HJEEDS.baseball_pitch import (
    DEFAULT_DELTA,
    DEFAULT_EXECUTION_SKILL_MAX,
    DEFAULT_EXECUTION_SKILL_MIN,
    BaseballRuntime,
    PROCESSED_ARTIFACT_REFERENCE,
    StatcastAgentSpec,
    build_baseball_runtime,
    build_pitch_observation,
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
COMPLETION_FILENAME = "calibration_completion.json"
CALIBRATION_OUTPUT_FILENAMES = (
    ROSTER_FILENAME,
    ROSTER_METADATA_FILENAME,
    "jeeds_calibration_agent_estimates.csv",
    "calibration_summary.json",
    "suggested_hyperpriors.json",
)

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
    "provenance_fingerprint",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Replace ``path`` only after a complete JSON file is durable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write_calibration_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=JEEDS_CALIBRATION_HEADER)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _run_fingerprint_is_valid(run_provenance: dict[str, Any]) -> bool:
    fingerprint_payload = dict(run_provenance)
    recorded = fingerprint_payload.pop("fingerprint_sha256", None)
    return recorded == canonical_json_sha256(fingerprint_payload)


def _paper_calibration_fields_match(summary: dict[str, Any]) -> bool:
    run_provenance = summary.get("run_provenance") or {}
    configuration = run_provenance.get("configuration") or {}
    selector = summary.get("roster_selector") or {}
    return (
        summary.get("season_year") == 2021
        and summary.get("pitch_types") == ["FF"]
        and summary.get("num_agents") == 528
        and summary.get("num_completed_agent_results") == 528
        and summary.get("num_successful_estimates") == 528
        and summary.get("num_missing_agent_results") == 0
        and summary.get("confidence") == "low"
        and summary.get("min_pitches_per_agent") == 100
        and summary.get("max_pitches_per_agent") is None
        and summary.get("max_agents") is None
        and selector.get("all_eligible_agents") is True
        and all(
            selector.get(key) in (None, [])
            for key in ("pitcher_ids", "top_pitchers", "bbip_extremes")
        )
        and configuration.get("roster_fingerprint") == PAPER_CALIBRATION_ROSTER_SHA256
    )


def _validate_paper_roster_if_requested(
    agent_specs: Sequence[StatcastAgentSpec],
    *,
    season_year: int | None,
    pitch_types: Sequence[str],
    min_pitches_per_agent: int,
    max_pitches_per_agent: int | None,
    max_agents: int | None,
    confidence: str,
    roster_selector: dict[str, Any],
) -> None:
    """Fail before inference if the canonical paper design resolves differently."""

    is_paper_design = (
        season_year == 2021
        and list(pitch_types) == ["FF"]
        and min_pitches_per_agent == 100
        and max_pitches_per_agent is None
        and max_agents is None
        and confidence == "low"
        and roster_selector.get("all_eligible_agents") is True
        and all(
            roster_selector.get(key) in (None, [])
            for key in ("pitcher_ids", "top_pitchers", "bbip_extremes")
        )
    )
    if not is_paper_design:
        return
    actual_hash = _spec_roster_fingerprint(agent_specs)
    if len(agent_specs) != 528 or actual_hash != PAPER_CALIBRATION_ROSTER_SHA256:
        raise RuntimeError(
            "Canonical paper calibration roster mismatch: expected the pinned 528-agent "
            f"cohort ({PAPER_CALIBRATION_ROSTER_SHA256}), received {len(agent_specs)} agents "
            f"({actual_hash}). Do not run the publication calibration with this cohort."
        )


def _write_incomplete_completion(
    output_dir: Path,
    *,
    stage: str,
    run_provenance: dict[str, Any] | None = None,
) -> Path:
    """Invalidate any older successful calibration before new work starts."""

    path = output_dir / COMPLETION_FILENAME
    payload: dict[str, Any] = {
        "schema_version": 1,
        "workflow": "baseball-hyperprior-calibration",
        "status": "incomplete",
        "stage": stage,
        "run_fingerprint_sha256": (run_provenance or {}).get("fingerprint_sha256"),
        "output_sha256": {},
    }
    payload["completion_fingerprint_sha256"] = canonical_json_sha256(payload)
    _atomic_write_json(path, payload)
    return path


def _write_complete_completion(
    output_dir: Path,
    summary: dict[str, Any],
    rows: Sequence[dict[str, Any]],
) -> Path:
    """Write the completion sentinel last, after every hashed output exists."""

    run_provenance = summary.get("run_provenance")
    if not isinstance(run_provenance, dict) or not _run_fingerprint_is_valid(run_provenance):
        raise ValueError("Cannot complete calibration with missing or invalid run provenance.")
    configuration = run_provenance.get("configuration") or {}
    output_hashes = {
        filename: _sha256_file(output_dir / filename)
        for filename in CALIBRATION_OUTPUT_FILENAMES
    }
    agent_result_hashes = {
        str(agent_result_path_for(output_dir, int(row["agent_id"])).relative_to(output_dir)): (
            _sha256_file(agent_result_path_for(output_dir, int(row["agent_id"])))
        )
        for row in rows
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "workflow": "baseball-hyperprior-calibration",
        "status": "complete",
        "stage": "outputs-committed",
        "run_fingerprint_sha256": run_provenance["fingerprint_sha256"],
        "roster_fingerprint_sha256": configuration.get("roster_fingerprint"),
        "num_expected_agents": int(summary["num_agents"]),
        "num_completed_agent_results": int(summary["num_completed_agent_results"]),
        "num_successful_estimates": int(summary["num_successful_estimates"]),
        "publication_ready_2021_ff": _paper_calibration_fields_match(summary),
        "output_sha256": output_hashes,
        "agent_result_sha256": agent_result_hashes,
    }
    payload["completion_fingerprint_sha256"] = canonical_json_sha256(payload)
    path = output_dir / COMPLETION_FILENAME
    _atomic_write_json(path, payload)
    return path


@dataclass(frozen=True)
class CalibrationContext:
    """Shared skill grids + baseball runtime for one calibration workload."""

    args: argparse.Namespace
    config: BaseballExperimentConfig
    all_data: pd.DataFrame
    sigma_grid: np.ndarray
    log_lambda_grid: np.ndarray
    runtime: BaseballRuntime
    run_provenance: dict[str, Any]


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
    parser.add_argument(
        "--allow-partial-results",
        action="store_true",
        help=(
            "NON-PAPER ONLY: calibrate from missing or unsuccessful agent results. "
            "The publication calibration requires the complete prepared roster."
        ),
    )
    parser.add_argument(
        "--validate-results",
        action="store_true",
        help=(
            "Validate a completed calibration, including output hashes. The canonical "
            "paper calibration additionally requires the pinned 528-agent roster."
        ),
    )
    parser.add_argument(
        "--copy-validated-hyperpriors-to",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "MANUAL ONLY: validate the canonical 528/528 paper calibration, then "
            "atomically copy suggested_hyperpriors.json to PATH."
        ),
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
    frozen_metadata: dict[str, Any] | None = None,
) -> CalibrationContext:
    """Build skill grids and one shared runtime (same pattern as baseball_pipeline)."""

    config = _build_calibration_config(args, roster, output_dir)
    sigma_grid, log_lambda_grid = build_baseball_skill_grids(config)
    execution_skills = tuple(float(value) for value in sigma_grid)
    # Seed is API-only: getNormalDistribution uses N.pdf (mean/cov), never N.rvs().
    rng = np.random.default_rng(args.seed)
    runtime = build_baseball_runtime(rng, execution_skills, delta=config.base.delta)
    configuration = _calibration_provenance_configuration(
        args,
        roster.agent_specs,
        frozen_metadata=frozen_metadata,
    )
    run_provenance = build_baseball_run_provenance(
        season_year=config.season_year,
        pitch_types=config.pitch_types,
        sigma_grid=sigma_grid,
        log_lambda_grid=log_lambda_grid,
        configuration=configuration,
    )
    return CalibrationContext(
        args=args,
        config=config,
        all_data=all_data,
        sigma_grid=sigma_grid,
        log_lambda_grid=log_lambda_grid,
        runtime=runtime,
        run_provenance=run_provenance,
    )


def roster_path_for(output_dir: Path) -> Path:
    return output_dir / ROSTER_FILENAME


def agent_result_path_for(output_dir: Path, agent_id: int) -> Path:
    return output_dir / AGENT_RESULTS_SUBDIR / f"agent_{agent_id:04d}.json"


def _spec_roster_fingerprint(agent_specs: Sequence[StatcastAgentSpec]) -> str:
    return roster_fingerprint(
        [(spec.agent_id, spec.pitcher_id, spec.pitch_type) for spec in agent_specs]
    )


def _calibration_provenance_configuration(
    args: argparse.Namespace,
    agent_specs: Sequence[StatcastAgentSpec],
    *,
    frozen_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = frozen_metadata or {}
    return {
        "workflow": "baseball-hyperprior-calibration",
        "min_pitches_per_agent": int(
            metadata.get("min_pitches_per_agent", _min_pitches_from_args(args))
        ),
        "max_pitches_per_agent": metadata.get(
            "max_pitches_per_agent", args.max_pitches_per_agent
        ),
        "max_agents": metadata.get("max_agents", args.max_agents),
        "confidence": str(metadata.get("confidence", args.confidence)),
        "roster_selector": dict(
            metadata.get("roster_selector") or roster_selector_kwargs_from_args(args)
        ),
        "roster_fingerprint": _spec_roster_fingerprint(agent_specs),
        "num_agents": len(agent_specs),
    }


def load_calibration_roster_metadata(output_dir: Path) -> dict[str, Any]:
    path = output_dir / ROSTER_METADATA_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing calibration roster metadata: {path}. Run --prepare-roster again."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed calibration roster metadata: {path}")
    return payload


def _validate_worker_args_against_calibration_metadata(
    args: argparse.Namespace,
    metadata: dict[str, Any],
) -> None:
    expected = {
        "season_year": metadata.get("season_year"),
        "pitch_types": tuple(metadata.get("pitch_types") or ()),
        "max_pitches_per_agent": metadata.get("max_pitches_per_agent"),
        "confidence": metadata.get("confidence"),
    }
    actual = {
        "season_year": args.season_year,
        "pitch_types": tuple(parse_pitch_types(args.pitch_types)),
        "max_pitches_per_agent": args.max_pitches_per_agent,
        "confidence": args.confidence,
    }
    mismatches = {
        key: (actual[key], expected[key])
        for key in expected
        if actual[key] != expected[key]
    }
    if mismatches:
        raise ValueError(
            "Calibration worker/aggregation arguments do not match the frozen roster metadata: "
            f"{mismatches}. Re-submit the workload with the original arguments."
        )


def write_calibration_roster(
    output_dir: Path,
    roster: BaseballRosterSelection,
    *,
    season_year: int | None,
    pitch_types: Sequence[str],
    min_pitches_per_agent: int,
    max_pitches_per_agent: int | None,
    max_agents: int | None,
    roster_selector: dict[str, Any],
    confidence: str,
    sigma_grid: Sequence[float],
    log_lambda_grid: Sequence[float],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_paper_roster_if_requested(
        roster.agent_specs,
        season_year=season_year,
        pitch_types=pitch_types,
        min_pitches_per_agent=min_pitches_per_agent,
        max_pitches_per_agent=max_pitches_per_agent,
        max_agents=max_agents,
        confidence=confidence,
        roster_selector=roster_selector,
    )
    path = roster_path_for(output_dir)
    payload = [
        {
            "agent_id": spec.agent_id,
            "pitcher_id": spec.pitcher_id,
            "pitch_type": spec.pitch_type,
        }
        for spec in roster.agent_specs
    ]
    _atomic_write_json(path, payload)

    metadata = {
        "season_year": season_year,
        "pitch_types": list(pitch_types),
        "min_pitches_per_agent": min_pitches_per_agent,
        "max_pitches_per_agent": max_pitches_per_agent,
        "max_agents": max_agents,
        "num_agents": len(payload),
        "roster_selector": roster_selector,
        "confidence": confidence,
        "agent_pitch_counts": [
            {
                "pitcher_id": int(pitcher_id),
                "pitch_type": str(pitch_type),
                "num_available_pitches": int(pitch_count),
            }
            for pitcher_id, pitch_type, pitch_count in roster.agent_pitch_counts
        ],
    }
    metadata["run_provenance"] = build_baseball_run_provenance(
        season_year=season_year,
        pitch_types=pitch_types,
        sigma_grid=sigma_grid,
        log_lambda_grid=log_lambda_grid,
        configuration={
            "workflow": "baseball-hyperprior-calibration",
            "min_pitches_per_agent": int(min_pitches_per_agent),
            "max_pitches_per_agent": max_pitches_per_agent,
            "max_agents": max_agents,
            "confidence": confidence,
            "roster_selector": roster_selector,
            "roster_fingerprint": _spec_roster_fingerprint(roster.agent_specs),
            "num_agents": len(roster.agent_specs),
        },
    )
    _atomic_write_json(output_dir / ROSTER_METADATA_FILENAME, metadata)
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
    if agent_rows.empty:
        estimate = MethodEstimate(
            method_name="jeeds",
            status="no_data",
            notes="No pitches for agent.",
        )
    else:
        execution_skills = tuple(float(value) for value in context.sigma_grid)
        log_likelihood_grid: np.ndarray | None = None
        # Each pitch observation holds one EV grid per execution hypothesis. Stream
        # and discard them so high-volume pitchers do not require tens of GB.
        for _, pitch_row in agent_rows.iterrows():
            observation = build_pitch_observation(
                pitch_row,
                context.runtime,
                execution_skills,
            )
            contribution = compute_baseball_log_likelihood_grid(
                pitch_observations=[observation],
                possible_targets_feet=context.runtime.grids.possible_targets_feet,
                all_covs=context.runtime.all_covs,
                sigma_grid=context.sigma_grid,
                log_lambda_grid=context.log_lambda_grid,
                delta=context.config.base.delta,
            )
            log_likelihood_grid = add_independent_log_likelihood_grids(
                log_likelihood_grid,
                contribution,
            )
        if log_likelihood_grid is None:
            raise RuntimeError("Nonempty agent rows produced no likelihood contributions.")
        estimate = run_independent_jeeds_baseline(
            log_likelihood_grid=log_likelihood_grid,
            sigma_grid=context.sigma_grid,
            log_lambda_grid=context.log_lambda_grid,
        )
    row = _estimate_to_row(
        agent_spec.agent_id,
        agent_spec.pitcher_id,
        agent_spec.pitch_type,
        len(agent_rows),
        estimate,
    )
    row["provenance_fingerprint"] = context.run_provenance["fingerprint_sha256"]
    return row


def write_agent_result(path: Path, row: dict[str, Any]) -> None:
    _atomic_write_json(path, row)


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
            row = json.load(handle)
        identity = (
            int(row.get("agent_id", -1)),
            int(row.get("pitcher_id", -1)),
            str(row.get("pitch_type", "")),
        )
        expected_identity = (
            agent_spec.agent_id,
            agent_spec.pitcher_id,
            agent_spec.pitch_type,
        )
        if identity != expected_identity:
            raise ValueError(
                f"Calibration result {path} identifies {identity}; expected {expected_identity}."
            )
        rows.append(row)
    rows.sort(key=lambda row: int(row["agent_id"]))
    return rows, missing_agent_ids


def validate_calibration_completion(
    output_dir: Path,
    *,
    require_paper_calibration: bool = False,
) -> dict[str, Any]:
    """Fail closed unless the completion sentinel and every final output agree."""

    completion_path = output_dir / COMPLETION_FILENAME
    if not completion_path.is_file():
        raise FileNotFoundError(
            f"Missing calibration completion sentinel: {completion_path}. "
            "Re-run aggregation before using these results."
        )
    with completion_path.open("r", encoding="utf-8") as handle:
        completion = json.load(handle)
    if (
        completion.get("schema_version") != 1
        or completion.get("workflow") != "baseball-hyperprior-calibration"
    ):
        raise RuntimeError(f"Unrecognized calibration completion schema: {completion_path}")
    if completion.get("status") != "complete":
        raise RuntimeError(
            f"Calibration at {output_dir} is not complete "
            f"(stage={completion.get('stage')!r})."
        )
    fingerprint_payload = dict(completion)
    recorded_completion_fingerprint = fingerprint_payload.pop(
        "completion_fingerprint_sha256", None
    )
    if recorded_completion_fingerprint != canonical_json_sha256(fingerprint_payload):
        raise RuntimeError(f"Calibration completion sentinel is malformed: {completion_path}")

    recorded_hashes = completion.get("output_sha256")
    if not isinstance(recorded_hashes, dict) or set(recorded_hashes) != set(
        CALIBRATION_OUTPUT_FILENAMES
    ):
        raise RuntimeError(
            "Calibration completion sentinel does not enumerate the exact final output set."
        )
    for filename in CALIBRATION_OUTPUT_FILENAMES:
        path = output_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Completed calibration is missing output: {path}")
        actual_hash = _sha256_file(path)
        if actual_hash != recorded_hashes[filename]:
            raise RuntimeError(
                f"Calibration output hash mismatch for {path}: "
                f"recorded={recorded_hashes[filename]}, actual={actual_hash}."
            )

    summary_path = output_dir / "calibration_summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    run_provenance = summary.get("run_provenance")
    if not isinstance(run_provenance, dict) or not _run_fingerprint_is_valid(run_provenance):
        raise RuntimeError(f"Calibration summary has invalid run provenance: {summary_path}")
    run_fingerprint = run_provenance["fingerprint_sha256"]
    if completion.get("run_fingerprint_sha256") != run_fingerprint:
        raise RuntimeError("Completion and summary run fingerprints do not match.")
    roster_hash = (run_provenance.get("configuration") or {}).get("roster_fingerprint")
    if completion.get("roster_fingerprint_sha256") != roster_hash:
        raise RuntimeError("Completion and summary roster fingerprints do not match.")

    roster_specs = load_calibration_roster(output_dir)
    actual_roster_hash = _spec_roster_fingerprint(roster_specs)
    if actual_roster_hash != roster_hash:
        raise RuntimeError("Frozen calibration roster does not match the recorded roster hash.")
    metadata = load_calibration_roster_metadata(output_dir)
    metadata_run = metadata.get("run_provenance")
    if not isinstance(metadata_run, dict) or metadata_run != run_provenance:
        raise RuntimeError("Frozen roster metadata does not match the completed calibration run.")

    expected = int(summary.get("num_agents", -1))
    completed = int(summary.get("num_completed_agent_results", -1))
    successful = int(summary.get("num_successful_estimates", -1))
    missing = int(summary.get("num_missing_agent_results", -1))
    if expected < 1 or completed < 0 or successful < 0 or missing < 0:
        raise RuntimeError("Calibration summary has invalid cohort counts.")
    if completed + missing != expected or successful > completed:
        raise RuntimeError("Calibration summary cohort counts are internally inconsistent.")
    completion_counts = (
        int(completion.get("num_expected_agents", -1)),
        int(completion.get("num_completed_agent_results", -1)),
        int(completion.get("num_successful_estimates", -1)),
    )
    if completion_counts != (expected, completed, successful):
        raise RuntimeError("Completion and summary cohort counts do not match.")

    csv_path = output_dir / "jeeds_calibration_agent_estimates.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != completed:
        raise RuntimeError(
            f"Calibration CSV has {len(rows)} rows, but summary reports {completed}."
        )
    recorded_agent_hashes = completion.get("agent_result_sha256")
    expected_agent_paths = {
        str(
            agent_result_path_for(output_dir, int(row["agent_id"])).relative_to(output_dir)
        )
        for row in rows
    }
    if not isinstance(recorded_agent_hashes, dict) or set(recorded_agent_hashes) != (
        expected_agent_paths
    ):
        raise RuntimeError("Completion does not enumerate the exact calibration agent outputs.")
    for row in rows:
        relative_path = str(
            agent_result_path_for(output_dir, int(row["agent_id"])).relative_to(output_dir)
        )
        agent_path = output_dir / relative_path
        if not agent_path.is_file():
            raise FileNotFoundError(f"Completed calibration is missing agent output: {agent_path}")
        actual_hash = _sha256_file(agent_path)
        if actual_hash != recorded_agent_hashes[relative_path]:
            raise RuntimeError(f"Calibration agent output hash mismatch for {agent_path}.")
        with agent_path.open("r", encoding="utf-8") as handle:
            agent_payload = json.load(handle)
        for key in (
            "agent_id",
            "pitcher_id",
            "pitch_type",
            "status",
            "num_observations",
            "provenance_fingerprint",
        ):
            if str(agent_payload.get(key)) != str(row.get(key)):
                raise RuntimeError(
                    f"Calibration agent output and aggregate CSV disagree for {relative_path} "
                    f"field {key}."
                )
        if row.get("provenance_fingerprint") != run_fingerprint:
            raise RuntimeError(
                f"Calibration agent output has stale run provenance: {relative_path}."
            )
    successful_rows = 0
    for row in rows:
        if row.get("status") != "ok":
            continue
        try:
            sigma = float(row["posterior_mean_sigma"])
            log_lambda = float(row["posterior_mean_log_lambda"])
        except (KeyError, TypeError, ValueError):
            continue
        if sigma > 0.0 and np.isfinite(sigma) and np.isfinite(log_lambda):
            successful_rows += 1
    if successful_rows != successful:
        raise RuntimeError(
            f"Calibration CSV has {successful_rows} successful finite rows, "
            f"but summary reports {successful}."
        )
    row_ids = [int(row["agent_id"]) for row in rows]
    missing_ids = [int(agent_id) for agent_id in summary.get("missing_agent_ids", [])]
    if len(set(row_ids)) != len(row_ids) or set(row_ids).intersection(missing_ids):
        raise RuntimeError("Calibration rows or missing-agent IDs are duplicated/overlapping.")
    if set(row_ids).union(missing_ids) != set(range(expected)):
        raise RuntimeError("Calibration rows and missing-agent IDs do not cover the frozen cohort.")
    if len(missing_ids) != missing:
        raise RuntimeError("Calibration missing-agent list does not match its reported count.")
    roster_by_id = {
        spec.agent_id: (spec.pitcher_id, spec.pitch_type) for spec in roster_specs
    }
    if len(roster_by_id) != expected or set(roster_by_id) != set(range(expected)):
        raise RuntimeError("Frozen calibration roster does not cover the expected agent IDs.")
    for row in rows:
        agent_id = int(row["agent_id"])
        if (int(row["pitcher_id"]), str(row["pitch_type"])) != roster_by_id[agent_id]:
            raise RuntimeError(f"Calibration CSV identity mismatch for agent {agent_id}.")

    suggested_path = output_dir / "suggested_hyperpriors.json"
    with suggested_path.open("r", encoding="utf-8") as handle:
        suggested = json.load(handle)
    suggested_run = (suggested.get("_provenance") or {}).get("run_provenance")
    if (
        not isinstance(suggested_run, dict)
        or not _run_fingerprint_is_valid(suggested_run)
        or suggested_run != run_provenance
    ):
        raise RuntimeError("Suggested hyperpriors do not match the completed calibration run.")

    paper_ready = _paper_calibration_fields_match(summary)
    if bool(completion.get("publication_ready_2021_ff")) != paper_ready:
        raise RuntimeError(
            "Completion paper-readiness flag does not match the calibration summary."
        )
    if require_paper_calibration:
        if not paper_ready:
            raise RuntimeError(
                "This is not the canonical complete 528/528 all-eligible 2021 FF calibration."
            )
        if expected != 528 or completed != 528 or successful != 528:
            raise RuntimeError("Paper calibration must contain 528/528 successful estimates.")
        if roster_hash != PAPER_CALIBRATION_ROSTER_SHA256:
            raise RuntimeError(
                "Paper calibration roster does not match the pinned 528-agent cohort."
            )
        load_hyperprior_config(
            suggested_path,
            require_paper_2021_ff_calibration=True,
        )
    return completion


def copy_validated_paper_hyperpriors(output_dir: Path, destination: Path) -> str:
    """Atomically copy only a hash-validated canonical paper calibration."""

    completion = validate_calibration_completion(
        output_dir,
        require_paper_calibration=True,
    )
    source = output_dir / "suggested_hyperpriors.json"
    expected_hash = completion["output_sha256"][source.name]
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with source.open("rb") as source_handle, os.fdopen(descriptor, "wb") as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        copied_hash = _sha256_file(temporary_path)
        if copied_hash != expected_hash:
            raise RuntimeError(
                "Temporary hyperprior copy does not match the validated source hash."
            )
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return expected_hash


def build_calibration_summary(
    rows: Sequence[dict[str, Any]],
    estimates: Sequence[MethodEstimate],
    *,
    confidence: str,
    season_year: int | None,
    pitch_types: Sequence[str],
    missing_agent_ids: Sequence[int],
    roster_selector: dict[str, Any],
    min_pitches_per_agent: int,
    max_pitches_per_agent: int | None,
    max_agents: int | None,
    run_provenance: dict[str, Any] | None = None,
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
        "roster_selector": roster_selector,
        "min_pitches_per_agent": min_pitches_per_agent,
        "max_pitches_per_agent": max_pitches_per_agent,
        "max_agents": max_agents,
        "run_provenance": run_provenance,
        "skill_grid": {
            "delta": DEFAULT_DELTA,
            "requested_num_sigma_grid": DEFAULT_NUM_SIGMA_GRID,
            "actual_num_sigma_hypotheses": (
                len(run_provenance["sigma_grid"]) if run_provenance is not None else 66
            ),
            "sigma_min": DEFAULT_EXECUTION_SKILL_MIN,
            "sigma_max": DEFAULT_EXECUTION_SKILL_MAX,
            "requested_num_lambda_grid": DEFAULT_NUM_LAMBDA_GRID,
            "actual_num_log_lambda_hypotheses": (
                len(run_provenance["log_lambda_grid"])
                if run_provenance is not None
                else DEFAULT_NUM_LAMBDA_GRID
            ),
            "lambda_min": DEFAULT_LAMBDA_MIN,
            "lambda_max": DEFAULT_LAMBDA_MAX,
        },
        "suggested_hyperpriors": hyperprior_config_to_dict(hyperpriors),
    }


def write_calibration_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    _write_incomplete_completion(
        output_dir,
        stage="committing-outputs",
        run_provenance=summary.get("run_provenance"),
    )
    csv_path = output_dir / "jeeds_calibration_agent_estimates.csv"
    _atomic_write_calibration_csv(csv_path, payload["rows"])

    summary_path = output_dir / "calibration_summary.json"
    _atomic_write_json(summary_path, summary)

    artifact_reference = json.loads(PROCESSED_ARTIFACT_REFERENCE.read_text(encoding="utf-8"))
    write_hyperprior_config(
        output_dir / "suggested_hyperpriors.json",
        payload["hyperpriors"],
        provenance={
            "processed_pickle_sha256": artifact_reference["pickle_sha256"],
            "model_weights_sha256": artifact_reference["model_weights_sha256"],
            "season_year": summary["season_year"],
            "pitch_types": summary["pitch_types"],
            "num_agents": summary["num_agents"],
            "num_completed_agent_results": summary["num_completed_agent_results"],
            "num_missing_agent_results": summary["num_missing_agent_results"],
            "num_successful_estimates": summary["num_successful_estimates"],
            "confidence": summary["confidence"],
            "roster_selector": summary["roster_selector"],
            "min_pitches_per_agent": summary["min_pitches_per_agent"],
            "max_pitches_per_agent": summary["max_pitches_per_agent"],
            "max_agents": summary["max_agents"],
            "skill_grid": summary["skill_grid"],
            "run_provenance": summary.get("run_provenance"),
            "roster_fingerprint_sha256": (
                (summary.get("run_provenance") or {}).get("configuration") or {}
            ).get("roster_fingerprint"),
        },
    )
    _write_complete_completion(output_dir, summary, payload["rows"])
    try:
        validate_calibration_completion(output_dir)
    except BaseException:
        _write_incomplete_completion(
            output_dir,
            stage="output-validation-failed",
            run_provenance=summary.get("run_provenance"),
        )
        raise


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    """Local sequential path: resolve roster, JEEDS every agent, write hyperpriors."""

    output_dir = Path(args.output_dir)
    _write_incomplete_completion(output_dir, stage="local-calibration-started")
    roster, pitch_types, min_pitches, all_data = _resolve_roster_from_args(args)
    _validate_paper_roster_if_requested(
        roster.agent_specs,
        season_year=args.season_year,
        pitch_types=pitch_types,
        min_pitches_per_agent=min_pitches,
        max_pitches_per_agent=args.max_pitches_per_agent,
        max_agents=args.max_agents,
        confidence=args.confidence,
        roster_selector=roster_selector_kwargs_from_args(args),
    )
    context = _make_calibration_context(
        args,
        roster=roster,
        all_data=all_data,
        output_dir=output_dir,
    )
    write_calibration_roster(
        output_dir,
        roster,
        season_year=args.season_year,
        pitch_types=pitch_types,
        min_pitches_per_agent=min_pitches,
        max_pitches_per_agent=args.max_pitches_per_agent,
        max_agents=args.max_agents,
        roster_selector=roster_selector_kwargs_from_args(args),
        confidence=args.confidence,
        sigma_grid=context.sigma_grid,
        log_lambda_grid=context.log_lambda_grid,
    )
    _write_incomplete_completion(
        output_dir,
        stage="local-agent-inference",
        run_provenance=context.run_provenance,
    )

    rows: list[dict[str, Any]] = []
    estimates: list[MethodEstimate] = []
    for agent_spec in roster.agent_specs:
        row = run_single_agent_calibration(context, agent_spec)
        write_agent_result(
            agent_result_path_for(output_dir, agent_spec.agent_id),
            row,
        )
        rows.append(row)
        estimates.append(_row_from_estimate_row(row))

    unsuccessful = [
        row["agent_id"]
        for row, estimate in zip(rows, estimates)
        if estimate.status != "ok"
        or estimate.posterior_mean_sigma is None
        or estimate.posterior_mean_log_lambda is None
        or estimate.posterior_mean_sigma <= 0.0
        or not np.isfinite(estimate.posterior_mean_sigma)
        or not np.isfinite(estimate.posterior_mean_log_lambda)
    ]
    if unsuccessful and not args.allow_partial_results:
        raise RuntimeError(f"Unsuccessful calibration estimates for agent IDs {unsuccessful}.")
    hyperpriors = build_hyperpriors_from_jeeds_estimates(estimates, confidence=args.confidence)
    summary = build_calibration_summary(
        rows,
        estimates,
        confidence=args.confidence,
        season_year=args.season_year,
        pitch_types=pitch_types,
        missing_agent_ids=(),
        roster_selector=roster_selector_kwargs_from_args(args),
        min_pitches_per_agent=_min_pitches_from_args(args),
        max_pitches_per_agent=args.max_pitches_per_agent,
        max_agents=args.max_agents,
        run_provenance=context.run_provenance,
        hyperpriors=hyperpriors,
    )
    return {"rows": rows, "summary": summary, "hyperpriors": hyperpriors}


def prepare_roster(args: argparse.Namespace) -> Path:
    """Write ``calibration_roster.json`` (+ metadata) for Slurm array indexing."""

    output_dir = Path(args.output_dir)
    _write_incomplete_completion(output_dir, stage="preparing-roster")
    roster, pitch_types, min_pitches, _all_data = _resolve_roster_from_args(args)
    config = _build_calibration_config(args, roster, output_dir)
    sigma_grid, log_lambda_grid = build_baseball_skill_grids(config)
    path = write_calibration_roster(
        output_dir,
        roster,
        season_year=args.season_year,
        pitch_types=pitch_types,
        min_pitches_per_agent=min_pitches,
        max_pitches_per_agent=args.max_pitches_per_agent,
        max_agents=args.max_agents,
        roster_selector=roster_selector_kwargs_from_args(args),
        confidence=args.confidence,
        sigma_grid=sigma_grid,
        log_lambda_grid=log_lambda_grid,
    )
    metadata = load_calibration_roster_metadata(output_dir)
    _write_incomplete_completion(
        output_dir,
        stage="awaiting-agent-results",
        run_provenance=metadata["run_provenance"],
    )
    print(
        f"[baseball-calibrate] Wrote roster with {len(roster.agent_specs)} agents "
        f"to {path.resolve()}"
    )
    return path


def run_agent_index(args: argparse.Namespace) -> Path:
    """Cluster worker: JEEDS for ``calibration_roster.json[agent_index]`` only."""

    output_dir = Path(args.output_dir)
    agent_specs = load_calibration_roster(output_dir)
    metadata = load_calibration_roster_metadata(output_dir)
    _validate_worker_args_against_calibration_metadata(args, metadata)
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
        frozen_metadata=metadata,
    )
    expected_provenance = metadata.get("run_provenance")
    if not isinstance(expected_provenance, dict):
        raise ValueError("Frozen calibration metadata lacks run_provenance; prepare the roster again.")
    if context.run_provenance["fingerprint_sha256"] != expected_provenance.get(
        "fingerprint_sha256"
    ):
        raise ValueError(
            "Frozen calibration metadata is stale for the current artifact/kernel/grid. "
            "Prepare the roster and rerun every worker."
        )
    _write_incomplete_completion(
        output_dir,
        stage="agent-inference",
        run_provenance=expected_provenance,
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
    _write_incomplete_completion(output_dir, stage="aggregation-started")
    rows, missing_agent_ids = load_agent_results(output_dir)
    if missing_agent_ids:
        message = f"Missing calibration agent results for IDs {missing_agent_ids}."
        if not args.allow_partial_results:
            raise FileNotFoundError(message)
        print(f"[baseball-calibrate] NON-PAPER WARNING: {message}", flush=True)

    estimates = [_row_from_estimate_row(row) for row in rows]
    unsuccessful = [
        int(row["agent_id"])
        for row, estimate in zip(rows, estimates)
        if estimate.status != "ok"
        or estimate.posterior_mean_sigma is None
        or estimate.posterior_mean_log_lambda is None
        or estimate.posterior_mean_sigma <= 0.0
        or not np.isfinite(estimate.posterior_mean_sigma)
        or not np.isfinite(estimate.posterior_mean_log_lambda)
    ]
    if unsuccessful and not args.allow_partial_results:
        raise RuntimeError(f"Unsuccessful calibration estimates for agent IDs {unsuccessful}.")
    metadata_path = output_dir / ROSTER_METADATA_FILENAME
    if metadata_path.is_file():
        metadata = load_calibration_roster_metadata(output_dir)
        _validate_worker_args_against_calibration_metadata(args, metadata)
        pitch_types = metadata.get("pitch_types", parse_pitch_types(args.pitch_types))
        season_year = metadata.get("season_year", args.season_year)
    else:
        if not args.allow_partial_results:
            raise FileNotFoundError(
                f"Missing calibration roster metadata: {metadata_path}. "
                "Publication calibration requires a frozen cohort selector."
            )
        pitch_types = parse_pitch_types(args.pitch_types)
        season_year = args.season_year
        metadata = {"roster_selector": roster_selector_kwargs_from_args(args)}

    roster_specs = load_calibration_roster(output_dir)
    _validate_paper_roster_if_requested(
        roster_specs,
        season_year=season_year,
        pitch_types=pitch_types,
        min_pitches_per_agent=int(
            metadata.get("min_pitches_per_agent", _min_pitches_from_args(args))
        ),
        max_pitches_per_agent=metadata.get("max_pitches_per_agent"),
        max_agents=metadata.get("max_agents"),
        confidence=str(metadata.get("confidence", args.confidence)),
        roster_selector=dict(metadata.get("roster_selector") or {}),
    )
    provenance_roster = _roster_selection_from_specs(args, roster_specs)
    provenance_config = _build_calibration_config(args, provenance_roster, output_dir)
    provenance_sigma_grid, provenance_log_lambda_grid = build_baseball_skill_grids(
        provenance_config
    )
    expected_provenance = build_baseball_run_provenance(
        season_year=season_year,
        pitch_types=pitch_types,
        sigma_grid=provenance_sigma_grid,
        log_lambda_grid=provenance_log_lambda_grid,
        configuration=_calibration_provenance_configuration(
            args,
            roster_specs,
            frozen_metadata=metadata,
        ),
    )
    frozen_provenance = metadata.get("run_provenance")
    if not isinstance(frozen_provenance, dict) or frozen_provenance.get(
        "fingerprint_sha256"
    ) != expected_provenance["fingerprint_sha256"]:
        raise ValueError(
            "Calibration roster metadata is stale for the current artifact/kernel/grid/config. "
            "Prepare the roster and rerun every worker."
        )
    stale_agents = [
        int(row["agent_id"])
        for row in rows
        if row.get("provenance_fingerprint") != expected_provenance["fingerprint_sha256"]
    ]
    if stale_agents:
        raise ValueError(
            "Calibration agent results were produced with a stale artifact/kernel/grid for "
            f"agent IDs {stale_agents}. Re-run those workers."
        )

    available_counts = {
        (int(row["pitcher_id"]), str(row["pitch_type"])): int(row["num_available_pitches"])
        for row in metadata.get("agent_pitch_counts", [])
    }
    if len(available_counts) != len(roster_specs):
        raise ValueError(
            "Calibration roster metadata has incomplete agent pitch counts; prepare the roster again."
        )
    max_pitches = metadata.get("max_pitches_per_agent")
    observation_mismatches: dict[int, tuple[int, int]] = {}
    for row in rows:
        key = (int(row["pitcher_id"]), str(row["pitch_type"]))
        available = available_counts.get(key)
        if available is None:
            observation_mismatches[int(row["agent_id"])] = (
                int(row.get("num_observations", -1)),
                -1,
            )
            continue
        expected_observations = available if max_pitches is None else min(available, int(max_pitches))
        actual_observations = int(row.get("num_observations", -1))
        if actual_observations != expected_observations:
            observation_mismatches[int(row["agent_id"])] = (
                actual_observations,
                expected_observations,
            )
    if observation_mismatches:
        raise ValueError(
            "Calibration rows have stale/incomplete observation counts "
            f"(actual, expected): {observation_mismatches}. Re-run those workers."
        )

    hyperpriors = build_hyperpriors_from_jeeds_estimates(estimates, confidence=args.confidence)
    summary = build_calibration_summary(
        rows,
        estimates,
        confidence=args.confidence,
        season_year=season_year,
        pitch_types=pitch_types,
        missing_agent_ids=missing_agent_ids,
        roster_selector=dict(metadata.get("roster_selector") or {}),
        min_pitches_per_agent=int(metadata.get("min_pitches_per_agent", _min_pitches_from_args(args))),
        max_pitches_per_agent=metadata.get("max_pitches_per_agent"),
        max_agents=metadata.get("max_agents"),
        run_provenance=expected_provenance,
        hyperpriors=hyperpriors,
    )
    payload = {"rows": rows, "summary": summary, "hyperpriors": hyperpriors}
    write_calibration_outputs(output_dir, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_calibration_args(argv)

    if args.copy_validated_hyperpriors_to is not None:
        destination = Path(args.copy_validated_hyperpriors_to)
        copied_hash = copy_validated_paper_hyperpriors(
            Path(args.output_dir),
            destination,
        )
        print(
            f"[baseball-calibrate] Atomically copied validated paper hyperpriors to "
            f"{destination.resolve()} (sha256={copied_hash})",
            flush=True,
        )
        return 0

    if args.validate_results:
        completion = validate_calibration_completion(Path(args.output_dir))
        print(
            f"[baseball-calibrate] Validated completed calibration at "
            f"{Path(args.output_dir).resolve()} "
            f"(agents={completion['num_completed_agent_results']}/"
            f"{completion['num_expected_agents']}, "
            f"successful={completion['num_successful_estimates']}, "
            f"paper_ready={completion['publication_ready_2021_ff']})",
            flush=True,
        )
        return 0

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
        print("  --validate-results          verify completion and output hashes")
        print("  --copy-validated-hyperpriors-to PATH  manual canonical install")
        print("Artifacts:")
        print(f"  - {ROSTER_FILENAME}")
        print(f"  - {ROSTER_METADATA_FILENAME}")
        print(f"  - {AGENT_RESULTS_SUBDIR}/agent_XXXX.json")
        print("  - jeeds_calibration_agent_estimates.csv")
        print("  - calibration_summary.json")
        print("  - suggested_hyperpriors.json")
        print(f"  - {COMPLETION_FILENAME} (written complete only after hashed outputs)")
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
