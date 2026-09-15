# Paper correspondence: Main `subsec:two_d_darts`; Supplement `app:two_d_hyperpriors`.
"""Fail-closed completion records for publication 2D-Darts result bundles.

The distributed 2D workflow writes many ``part_*`` directories before one
aggregation job publishes the CSV summaries and diagnostic plot used by the
paper.  This module makes that publication boundary explicit: aggregation
first writes an atomic ``incomplete`` record, and only writes ``complete``
after the exact seed/agent design, current computational sources, and every
required output artifact have been validated and hashed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from . import config as experiment_config


REPO_ROOT = Path(__file__).resolve().parent.parent
TWO_D_COMPLETION_FILENAME = "two_d_run_metadata.json"
TWO_D_PART_METADATA_FILENAME = "two_d_part_metadata.json"
TWO_D_COMPLETION_SCHEMA_VERSION = 1
TWO_D_WORKFLOW = "hjeeds-2d-cluster-aggregation"
TWO_D_PART_WORKFLOW = "hjeeds-2d-cluster-part"

PAPER_TWO_D_SEED_START = 1000
PAPER_TWO_D_NUM_SEEDS = 500
PAPER_TWO_D_AGENTS_PER_SEED = 25
PAPER_TWO_D_COUNT_BUCKETS = (5, 10, 25, 100, 1000)
PAPER_TWO_D_AGENTS_PER_BUCKET = 5

REQUIRED_TWO_D_ARTIFACT_FILENAMES = (
    experiment_config.AGENT_LEVEL_FILENAME,
    experiment_config.SUMMARY_BY_BUCKET_FILENAME,
    experiment_config.SUMMARY_OVERALL_FILENAME,
    experiment_config.ERROR_PLOT_FILENAME,
)

# These are the source and environment declarations that can change simulated
# 2D actions, estimator outputs, error summaries, or the exact cluster command.
# Hashing the explicit list keeps the provenance auditable and makes a checkout
# change fail closed instead of silently reusing a stale summary CSV.
TWO_D_COMPUTATION_SOURCE_PATHS = (
    "Environments/Darts/RandomDarts/two_d_darts.py",
    "HJEEDS/aggregate_cluster_seeds.py",
    "HJEEDS/aggregation.py",
    "HJEEDS/artifacts.py",
    "HJEEDS/config.py",
    "HJEEDS/darts_hierarchical_vs_jeeds.py",
    "HJEEDS/darts_environment.py",
    "HJEEDS/decision_models.py",
    "HJEEDS/environment_adapters.py",
    "HJEEDS/estimation.py",
    "HJEEDS/likelihood.py",
    "HJEEDS/models.py",
    "HJEEDS/pipeline.py",
    "HJEEDS/population_shapes.py",
    "HJEEDS/rationality.py",
    "HJEEDS/sampling.py",
    "HJEEDS/two_d_completion.py",
    "run_hjeeds_2d_cluster_tests.sbatch",
    "submit_hjeeds_2d_cluster_tests.sh",
    "environment-hjeeds.yml",
    "requirements-hjeeds.txt",
)


def _canonical_json_sha256(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_runtime_provenance() -> dict[str, str]:
    """Return the numerical runtime versions active for this workflow stage."""

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write one JSON record atomically within its destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def _with_metadata_fingerprint(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("metadata_fingerprint_sha256", None)
    result["metadata_fingerprint_sha256"] = _canonical_json_sha256(result)
    return result


def _validate_metadata_fingerprint(payload: dict[str, Any], path: Path) -> None:
    fingerprint_payload = dict(payload)
    recorded = fingerprint_payload.pop("metadata_fingerprint_sha256", None)
    if recorded != _canonical_json_sha256(fingerprint_payload):
        raise ValueError(f"2D completion metadata fingerprint is invalid: {path}")


def paper_two_d_configuration() -> dict[str, Any]:
    """Return the fixed model/design configuration required by the paper."""

    mean_log_sigma = math.log(math.sqrt(8.0 * 60.0))
    return {
        "environment": "2d",
        "num_agents": 25,
        "count_buckets": list(PAPER_TWO_D_COUNT_BUCKETS),
        "agents_per_bucket": PAPER_TWO_D_AGENTS_PER_BUCKET,
        "delta": 5.0,
        "num_sigma_grid": 21,
        "num_lambda_grid": 21,
        "sigma_min": 8.0,
        "sigma_max": 60.0,
        "lambda_min": 1e-3,
        "lambda_max": 1e2,
        "true_decision_model_slug": "softmax",
        "true_population": {
            "mean_log_sigma": mean_log_sigma,
            "mean_log_lambda": 0.0,
            "tau_eta": 0.35,
            "tau_rho": 1.0,
            "correlation": -0.5,
            "population_shape_slug": "default",
        },
        "hyperpriors": {
            "mean_vector": [mean_log_sigma, 0.0],
            "covariance_diagonal": [0.6**2, 3.0**2],
            "log_tau_eta_mean": math.log(0.35),
            "log_tau_eta_sd": 0.5,
            "log_tau_rho_mean": 0.0,
            "log_tau_rho_sd": 0.5,
            "m_r": math.atanh(-0.5),
            "s_r": 0.75,
        },
        "reward_surface": {
            "family": "standard 2D darts board",
            "state_distribution": "normal",
            "num_reward_surfaces_per_seed": 1,
        },
        "execution_noise": "independent advancing-RNG isotropic Gaussian draw per action",
    }


def _current_default_two_d_configuration() -> dict[str, Any]:
    """Translate the live configuration constants into the paper schema."""

    true_population = experiment_config.DEFAULT_TRUE_POPULATION_2D
    hyperpriors = experiment_config.DEFAULT_HYPERPRIORS_2D
    return {
        "environment": "2d",
        "num_agents": experiment_config.DEFAULT_NUM_AGENTS,
        "count_buckets": list(experiment_config.DEFAULT_COUNT_BUCKETS),
        "agents_per_bucket": experiment_config.DEFAULT_AGENTS_PER_BUCKET,
        "delta": experiment_config.DEFAULT_DELTA_2D,
        "num_sigma_grid": experiment_config.DEFAULT_NUM_SIGMA_GRID,
        "num_lambda_grid": experiment_config.DEFAULT_NUM_LAMBDA_GRID,
        "sigma_min": experiment_config.DEFAULT_SIGMA_MIN_2D,
        "sigma_max": experiment_config.DEFAULT_SIGMA_MAX_2D,
        "lambda_min": experiment_config.DEFAULT_LAMBDA_MIN,
        "lambda_max": experiment_config.DEFAULT_LAMBDA_MAX,
        "true_decision_model_slug": experiment_config.DEFAULT_TRUE_DECISION_MODEL_SLUG,
        "true_population": {
            "mean_log_sigma": true_population.mean_log_sigma,
            "mean_log_lambda": true_population.mean_log_lambda,
            "tau_eta": true_population.tau_eta,
            "tau_rho": true_population.tau_rho,
            "correlation": true_population.correlation,
            "population_shape_slug": true_population.population_shape_slug,
        },
        "hyperpriors": {
            "mean_vector": list(hyperpriors.mean_vector),
            "covariance_diagonal": list(hyperpriors.covariance_diagonal),
            "log_tau_eta_mean": hyperpriors.log_tau_eta_mean,
            "log_tau_eta_sd": hyperpriors.log_tau_eta_sd,
            "log_tau_rho_mean": hyperpriors.log_tau_rho_mean,
            "log_tau_rho_sd": hyperpriors.log_tau_rho_sd,
            "m_r": hyperpriors.m_r,
            "s_r": hyperpriors.s_r,
        },
        "reward_surface": {
            "family": "standard 2D darts board",
            "state_distribution": "normal",
            "num_reward_surfaces_per_seed": 1,
        },
        "execution_noise": "independent advancing-RNG isotropic Gaussian draw per action",
    }


def paper_two_d_effective_launch(
    part_dir: Path,
    *,
    seed_start: int,
    num_seeds: int,
) -> dict[str, Any]:
    """Return the exact resolved worker launch contract for one paper part."""

    return {
        "model_configuration": paper_two_d_configuration(),
        "seed_start": int(seed_start),
        "num_seeds": int(num_seeds),
        "output_dir": str(Path(part_dir).resolve()),
        "dry_run": False,
        "plot_only": False,
        "include_raw_rationality_error": False,
    }


def _configuration_schema_from_resolved(config) -> dict[str, Any]:
    """Translate a resolved ``ExperimentConfig`` into the paper schema."""

    true_population = config.true_population
    hyperpriors = config.hyperpriors
    return {
        "environment": config.environment,
        "num_agents": config.num_agents,
        "count_buckets": list(config.count_buckets),
        "agents_per_bucket": config.agents_per_bucket,
        "delta": config.delta,
        "num_sigma_grid": config.num_sigma_grid,
        "num_lambda_grid": config.num_lambda_grid,
        "sigma_min": config.sigma_min,
        "sigma_max": config.sigma_max,
        "lambda_min": config.lambda_min,
        "lambda_max": config.lambda_max,
        "true_decision_model_slug": config.true_decision_model_slug,
        "true_population": {
            "mean_log_sigma": true_population.mean_log_sigma,
            "mean_log_lambda": true_population.mean_log_lambda,
            "tau_eta": true_population.tau_eta,
            "tau_rho": true_population.tau_rho,
            "correlation": true_population.correlation,
            "population_shape_slug": true_population.population_shape_slug,
        },
        "hyperpriors": {
            "mean_vector": list(hyperpriors.mean_vector),
            "covariance_diagonal": list(hyperpriors.covariance_diagonal),
            "log_tau_eta_mean": hyperpriors.log_tau_eta_mean,
            "log_tau_eta_sd": hyperpriors.log_tau_eta_sd,
            "log_tau_rho_mean": hyperpriors.log_tau_rho_mean,
            "log_tau_rho_sd": hyperpriors.log_tau_rho_sd,
            "m_r": hyperpriors.m_r,
            "s_r": hyperpriors.s_r,
        },
        "reward_surface": {
            "family": "standard 2D darts board",
            "state_distribution": "normal",
            "num_reward_surfaces_per_seed": 1,
        },
        "execution_noise": "independent advancing-RNG isotropic Gaussian draw per action",
    }


def resolve_two_d_effective_launch(experiment_argv: list[str]) -> dict[str, Any]:
    """Resolve the exact argv later passed to the worker experiment."""

    parsed = experiment_config.parse_args(experiment_argv)
    resolved = experiment_config.build_config_from_args(parsed)
    return {
        "model_configuration": _configuration_schema_from_resolved(resolved),
        "seed_start": resolved.seed,
        "num_seeds": resolved.num_seeds,
        "output_dir": str(resolved.output_dir.resolve()),
        "dry_run": resolved.dry_run,
        "plot_only": bool(parsed.plot_only),
        "include_raw_rationality_error": bool(parsed.include_raw_rationality_error),
    }


def paper_two_d_experiment_argv(
    part_dir: Path,
    *,
    seed_start: int,
    num_seeds: int,
) -> list[str]:
    """Construct the one authoritative argv for a paper 2D worker part."""

    configuration = paper_two_d_configuration()
    return [
        "--num-seeds",
        str(num_seeds),
        "--environment",
        "2d",
        "--output-dir",
        str(part_dir),
        "--seed",
        str(seed_start),
        "--num-agents",
        str(configuration["num_agents"]),
        "--count-buckets",
        ",".join(str(value) for value in configuration["count_buckets"]),
        "--agents-per-bucket",
        str(configuration["agents_per_bucket"]),
        "--delta",
        str(configuration["delta"]),
        "--num-sigma-grid",
        str(configuration["num_sigma_grid"]),
        "--num-lambda-grid",
        str(configuration["num_lambda_grid"]),
        "--sigma-min",
        str(configuration["sigma_min"]),
        "--sigma-max",
        str(configuration["sigma_max"]),
        "--lambda-min",
        str(configuration["lambda_min"]),
        "--lambda-max",
        str(configuration["lambda_max"]),
    ]


def run_paper_two_d_part(
    part_dir: Path,
    *,
    seed_start: int,
    num_seeds: int,
    expected_agents_per_seed: int = PAPER_TWO_D_AGENTS_PER_SEED,
) -> Path:
    """Run one canonical part under the same argv that its seal records."""

    experiment_argv = paper_two_d_experiment_argv(
        part_dir,
        seed_start=seed_start,
        num_seeds=num_seeds,
    )
    effective_launch = resolve_two_d_effective_launch(experiment_argv)
    begin_two_d_part_metadata(
        part_dir,
        expected_seed_start=seed_start,
        expected_num_seeds=num_seeds,
        expected_agents_per_seed=expected_agents_per_seed,
        effective_launch_configuration=effective_launch,
    )
    from .darts_hierarchical_vs_jeeds import main as experiment_main

    return_code = experiment_main(experiment_argv)
    if return_code != 0:
        raise RuntimeError(f"Canonical 2D worker returned nonzero status {return_code}.")
    return finalize_two_d_part_metadata(part_dir)


def require_current_paper_two_d_defaults() -> None:
    """Fail if live defaults drift from the fixed publication configuration."""

    actual = _current_default_two_d_configuration()
    expected = paper_two_d_configuration()
    if actual != expected:
        raise ValueError(
            "The current 2D defaults do not match the fixed paper configuration. "
            f"actual={actual}, expected={expected}"
        )


def build_current_two_d_source_provenance() -> dict[str, Any]:
    """Hash the exact source/config files required to reproduce 2D results."""

    files: dict[str, dict[str, Any]] = {}
    for relative_name in TWO_D_COMPUTATION_SOURCE_PATHS:
        path = REPO_ROOT / relative_name
        if not path.is_file():
            raise FileNotFoundError(f"Missing required 2D provenance source: {path}")
        files[relative_name] = {
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    result: dict[str, Any] = {
        "provenance_version": "2d-computational-sources-v1",
        "files": files,
    }
    result["fingerprint_sha256"] = _canonical_json_sha256(result)
    return result


def _expected_agent_assignments() -> list[dict[str, int]]:
    assignments: list[dict[str, int]] = []
    agent_id = 0
    for count_bucket in PAPER_TWO_D_COUNT_BUCKETS:
        for _ in range(PAPER_TWO_D_AGENTS_PER_BUCKET):
            assignments.append(
                {
                    "agent_id": agent_id,
                    "count_bucket": count_bucket,
                    "num_observations": count_bucket,
                }
            )
            agent_id += 1
    return assignments


def _validate_agent_csv_contract(
    path: Path,
    *,
    expected_seeds: list[int],
    expected_agent_ids: list[int],
) -> dict[str, Any]:
    """Validate the exact row grid and return its seed-invariant agent design."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing completed 2D agent CSV: {path}")
    expected_seed_set = set(expected_seeds)
    expected_agent_set = set(expected_agent_ids)
    rows_by_seed: dict[int, dict[int, tuple[int, int]]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "seed",
            "environment",
            "agent_id",
            "count_bucket",
            "num_observations",
        }
        missing_columns = required_columns - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                f"2D agent CSV is missing completion fields {sorted(missing_columns)}: {path}"
            )
        for row in reader:
            if row["environment"].strip() != "2d":
                raise ValueError(f"Non-2D row found in completed 2D bundle: {path}")
            seed = int(row["seed"])
            agent_id = int(row["agent_id"])
            if seed not in expected_seed_set or agent_id not in expected_agent_set:
                raise ValueError(
                    f"Unexpected seed-agent row ({seed}, {agent_id}) in completed 2D bundle."
                )
            design = (int(row["count_bucket"]), int(row["num_observations"]))
            by_agent = rows_by_seed.setdefault(seed, {})
            if agent_id in by_agent:
                raise ValueError(
                    f"Duplicate seed-agent row ({seed}, {agent_id}) in completed 2D bundle."
                )
            by_agent[agent_id] = design

    if set(rows_by_seed) != expected_seed_set:
        raise ValueError("Completed 2D bundle does not contain the exact recorded seed set.")
    reference_design: dict[int, tuple[int, int]] | None = None
    for seed in expected_seeds:
        by_agent = rows_by_seed[seed]
        if set(by_agent) != expected_agent_set:
            raise ValueError(
                f"Completed 2D bundle does not contain the exact agent set for seed {seed}."
            )
        if reference_design is None:
            reference_design = by_agent
        elif by_agent != reference_design:
            raise ValueError(
                "The 2D observation-count assignment changed across seeds; refusing completion."
            )
    if reference_design is None:
        raise ValueError("Completed 2D bundle contains no agent rows.")

    assignments = [
        {
            "agent_id": agent_id,
            "count_bucket": reference_design[agent_id][0],
            "num_observations": reference_design[agent_id][1],
        }
        for agent_id in expected_agent_ids
    ]
    result: dict[str, Any] = {"agent_assignments": assignments}
    result["fingerprint_sha256"] = _canonical_json_sha256(result)
    return result


def begin_two_d_part_metadata(
    part_dir: Path,
    *,
    expected_seed_start: int,
    expected_num_seeds: int,
    expected_agents_per_seed: int,
    effective_launch_configuration: dict[str, Any],
) -> Path:
    """Invalidate one worker partition before its simulation begins."""

    if expected_seed_start < 0 or expected_num_seeds <= 0:
        raise ValueError("A 2D part requires a nonnegative start and positive seed count.")
    if expected_agents_per_seed <= 0:
        raise ValueError("A 2D part requires a positive agent count.")
    require_current_paper_two_d_defaults()
    expected_launch = paper_two_d_effective_launch(
        part_dir,
        seed_start=expected_seed_start,
        num_seeds=expected_num_seeds,
    )
    if effective_launch_configuration != expected_launch:
        raise ValueError(
            "The effective 2D worker configuration is not the canonical paper launch. "
            f"actual={effective_launch_configuration}, expected={expected_launch}"
        )
    if expected_agents_per_seed != effective_launch_configuration["model_configuration"]["num_agents"]:
        raise ValueError("The 2D worker metadata agent count disagrees with its resolved argv.")
    expected_seeds = list(
        range(expected_seed_start, expected_seed_start + expected_num_seeds)
    )
    payload = {
        "schema_version": TWO_D_COMPLETION_SCHEMA_VERSION,
        "workflow": TWO_D_PART_WORKFLOW,
        "completion_status": "incomplete",
        "completion_stage": "worker-started",
        "expected_seed_start": expected_seed_start,
        "expected_seed_end_inclusive": expected_seeds[-1],
        "expected_num_seeds": expected_num_seeds,
        "expected_seeds": expected_seeds,
        "expected_agents_per_seed": expected_agents_per_seed,
        "expected_agent_ids": list(range(expected_agents_per_seed)),
        "paper_configuration": paper_two_d_configuration(),
        "effective_launch_configuration": effective_launch_configuration,
        "runtime_provenance": current_runtime_provenance(),
        "source_provenance": build_current_two_d_source_provenance(),
        "observed_agent_design": None,
        "artifact_sha256": {},
        "artifact_sizes_bytes": {},
    }
    path = Path(part_dir) / TWO_D_PART_METADATA_FILENAME
    return _atomic_write_json(path, _with_metadata_fingerprint(payload))


def _load_part_metadata(part_dir: Path) -> tuple[Path, dict[str, Any]]:
    path = Path(part_dir) / TWO_D_PART_METADATA_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing 2D part provenance: {path}. Rerun this partition with the corrected worker."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed 2D part provenance: {path}")
    if (
        payload.get("schema_version") != TWO_D_COMPLETION_SCHEMA_VERSION
        or payload.get("workflow") != TWO_D_PART_WORKFLOW
    ):
        raise ValueError(f"Unrecognized 2D part provenance schema: {path}")
    _validate_metadata_fingerprint(payload, path)
    return path, payload


def finalize_two_d_part_metadata(part_dir: Path) -> Path:
    """Seal one completed worker partition under its source/config fingerprint."""

    path, payload = _load_part_metadata(part_dir)
    if payload.get("completion_status") != "incomplete":
        raise ValueError("2D part metadata must be incomplete before finalization.")
    if payload.get("paper_configuration") != paper_two_d_configuration():
        raise ValueError("2D part configuration changed before finalization.")
    expected_launch = paper_two_d_effective_launch(
        part_dir,
        seed_start=int(payload.get("expected_seed_start", -1)),
        num_seeds=int(payload.get("expected_num_seeds", -1)),
    )
    if payload.get("effective_launch_configuration") != expected_launch:
        raise ValueError("2D part effective worker configuration is not the paper launch.")
    if payload.get("runtime_provenance") != current_runtime_provenance():
        raise ValueError("2D numerical runtime changed during the worker run.")
    if payload.get("source_provenance") != build_current_two_d_source_provenance():
        raise ValueError("2D computational sources changed during the worker run; rerun the part.")
    expected_seeds, expected_agent_ids = _validate_recorded_design(payload)
    agent_csv = Path(part_dir) / experiment_config.AGENT_LEVEL_FILENAME
    payload["observed_agent_design"] = _validate_agent_csv_contract(
        agent_csv,
        expected_seeds=expected_seeds,
        expected_agent_ids=expected_agent_ids,
    )
    payload["artifact_sha256"] = {agent_csv.name: _file_sha256(agent_csv)}
    payload["artifact_sizes_bytes"] = {agent_csv.name: agent_csv.stat().st_size}
    payload["completion_stage"] = "worker-complete"
    payload["completion_status"] = "complete"
    return _atomic_write_json(path, _with_metadata_fingerprint(payload))


def validate_complete_two_d_part_metadata(
    part_dir: Path,
    *,
    expected_seed_start: int,
    expected_num_seeds: int,
    expected_agents_per_seed: int,
) -> dict[str, Any]:
    """Verify one partition was produced by the current corrected worker."""

    _path, payload = _load_part_metadata(part_dir)
    if payload.get("completion_status") != "complete":
        raise ValueError(f"2D worker partition is incomplete: {part_dir}")
    expected_fields = {
        "expected_seed_start": expected_seed_start,
        "expected_num_seeds": expected_num_seeds,
        "expected_agents_per_seed": expected_agents_per_seed,
    }
    mismatches = {
        field: (payload.get(field), expected)
        for field, expected in expected_fields.items()
        if payload.get(field) != expected
    }
    if mismatches:
        raise ValueError(f"2D part design does not match its cluster partition: {mismatches}")
    expected_seeds, expected_agent_ids = _validate_recorded_design(payload)
    if payload.get("paper_configuration") != paper_two_d_configuration():
        raise ValueError("2D part does not use the fixed paper model configuration.")
    expected_launch = paper_two_d_effective_launch(
        part_dir,
        seed_start=expected_seed_start,
        num_seeds=expected_num_seeds,
    )
    if payload.get("effective_launch_configuration") != expected_launch:
        raise ValueError("2D part effective worker configuration does not match its partition.")
    if payload.get("runtime_provenance") != current_runtime_provenance():
        raise ValueError("2D part runtime differs from the aggregation runtime.")
    if payload.get("source_provenance") != build_current_two_d_source_provenance():
        raise ValueError(
            "2D part source provenance is stale relative to the corrected checkout."
        )
    agent_csv = Path(part_dir) / experiment_config.AGENT_LEVEL_FILENAME
    expected_hashes = payload.get("artifact_sha256") or {}
    expected_sizes = payload.get("artifact_sizes_bytes") or {}
    if set(expected_hashes) != {agent_csv.name} or set(expected_sizes) != {agent_csv.name}:
        raise ValueError("2D part metadata does not enumerate exactly its agent CSV.")
    if not agent_csv.is_file():
        raise FileNotFoundError(f"Completed 2D part agent CSV is missing: {agent_csv}")
    if agent_csv.stat().st_size != expected_sizes[agent_csv.name]:
        raise ValueError(f"Completed 2D part agent CSV size changed: {agent_csv}")
    if _file_sha256(agent_csv) != expected_hashes[agent_csv.name]:
        raise ValueError(f"Completed 2D part agent CSV hash changed: {agent_csv}")
    observed_design = _validate_agent_csv_contract(
        agent_csv,
        expected_seeds=expected_seeds,
        expected_agent_ids=expected_agent_ids,
    )
    if payload.get("observed_agent_design") != observed_design:
        raise ValueError("2D part metadata and agent-level observation design disagree.")
    return payload


def begin_two_d_aggregation_metadata(
    group_dir: Path,
    *,
    expected_seed_start: int,
    expected_num_seeds: int,
    expected_agents_per_seed: int,
    parts_per_group: int,
) -> Path:
    """Atomically invalidate any prior aggregate before validation or writes."""

    if expected_seed_start < 0 or expected_num_seeds <= 0:
        raise ValueError("The 2D completion record requires a nonnegative start and positive seed count.")
    if expected_agents_per_seed <= 0 or parts_per_group <= 0:
        raise ValueError("The 2D completion record requires positive agent and part counts.")
    require_current_paper_two_d_defaults()
    expected_seeds = list(
        range(expected_seed_start, expected_seed_start + expected_num_seeds)
    )
    payload = {
        "schema_version": TWO_D_COMPLETION_SCHEMA_VERSION,
        "workflow": TWO_D_WORKFLOW,
        "completion_status": "incomplete",
        "completion_stage": "aggregation-started",
        "expected_seed_start": expected_seed_start,
        "expected_seed_end_inclusive": expected_seeds[-1],
        "expected_num_seeds": expected_num_seeds,
        "expected_seeds": expected_seeds,
        "expected_agents_per_seed": expected_agents_per_seed,
        "expected_agent_ids": list(range(expected_agents_per_seed)),
        "parts_per_group": parts_per_group,
        "paper_configuration": paper_two_d_configuration(),
        "runtime_provenance": current_runtime_provenance(),
        "source_provenance": build_current_two_d_source_provenance(),
        "observed_agent_design": None,
        "required_artifacts": list(REQUIRED_TWO_D_ARTIFACT_FILENAMES),
        "artifact_sha256": {},
        "artifact_sizes_bytes": {},
    }
    path = Path(group_dir) / TWO_D_COMPLETION_FILENAME
    return _atomic_write_json(path, _with_metadata_fingerprint(payload))


def _load_metadata(group_dir: Path) -> tuple[Path, dict[str, Any]]:
    path = Path(group_dir) / TWO_D_COMPLETION_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing 2D completion metadata: {path}. Re-run the corrected 2D aggregation."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed 2D completion metadata: {path}")
    if (
        payload.get("schema_version") != TWO_D_COMPLETION_SCHEMA_VERSION
        or payload.get("workflow") != TWO_D_WORKFLOW
    ):
        raise ValueError(f"Unrecognized 2D completion metadata schema: {path}")
    _validate_metadata_fingerprint(payload, path)
    return path, payload


def _validate_recorded_design(payload: dict[str, Any]) -> tuple[list[int], list[int]]:
    seed_start = int(payload.get("expected_seed_start", -1))
    seed_count = int(payload.get("expected_num_seeds", -1))
    if seed_start < 0 or seed_count <= 0:
        raise ValueError("2D completion metadata contains an invalid seed design.")
    expected_seeds = list(range(seed_start, seed_start + seed_count))
    if payload.get("expected_seeds") != expected_seeds:
        raise ValueError("2D completion metadata does not bind the exact consecutive seed set.")
    if payload.get("expected_seed_end_inclusive") != expected_seeds[-1]:
        raise ValueError("2D completion metadata has an inconsistent final seed.")
    agent_count = int(payload.get("expected_agents_per_seed", -1))
    if agent_count <= 0:
        raise ValueError("2D completion metadata contains an invalid agent count.")
    expected_agent_ids = list(range(agent_count))
    if payload.get("expected_agent_ids") != expected_agent_ids:
        raise ValueError("2D completion metadata does not bind the exact agent ID set.")
    return expected_seeds, expected_agent_ids


def finalize_two_d_aggregation_metadata(group_dir: Path) -> Path:
    """Hash the complete output bundle and atomically write ``complete`` last."""

    path, payload = _load_metadata(group_dir)
    if payload.get("completion_status") != "incomplete":
        raise ValueError("2D aggregation metadata must be incomplete before finalization.")
    if payload.get("runtime_provenance") != current_runtime_provenance():
        raise ValueError("2D numerical runtime changed during aggregation.")
    if payload.get("source_provenance") != build_current_two_d_source_provenance():
        raise ValueError("2D computational sources changed during aggregation; rerun all parts.")
    expected_seeds, expected_agent_ids = _validate_recorded_design(payload)
    observed_design = _validate_agent_csv_contract(
        Path(group_dir) / experiment_config.AGENT_LEVEL_FILENAME,
        expected_seeds=expected_seeds,
        expected_agent_ids=expected_agent_ids,
    )
    required_names = tuple(payload.get("required_artifacts") or ())
    if required_names != REQUIRED_TWO_D_ARTIFACT_FILENAMES:
        raise ValueError("2D completion metadata does not enumerate the exact artifact contract.")
    artifacts = [Path(group_dir) / name for name in required_names]
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Cannot complete 2D aggregation; required artifacts are missing: {missing}"
        )

    payload["observed_agent_design"] = observed_design
    payload["artifact_sha256"] = {
        artifact.name: _file_sha256(artifact) for artifact in artifacts
    }
    payload["artifact_sizes_bytes"] = {
        artifact.name: artifact.stat().st_size for artifact in artifacts
    }
    payload["completion_stage"] = "aggregation-complete"
    payload["completion_status"] = "complete"
    return _atomic_write_json(path, _with_metadata_fingerprint(payload))


def validate_complete_two_d_run_metadata(
    group_dir: Path,
    *,
    require_paper_configuration: bool = False,
) -> dict[str, Any]:
    """Reject missing, incomplete, tampered, stale, or non-paper 2D bundles."""

    _path, payload = _load_metadata(group_dir)
    if payload.get("completion_status") != "complete":
        raise ValueError(
            "2D result bundle is incomplete; rerun the corrected cluster aggregation."
        )
    expected_seeds, expected_agent_ids = _validate_recorded_design(payload)
    if payload.get("paper_configuration") != paper_two_d_configuration():
        raise ValueError("2D result metadata does not match the fixed paper model configuration.")
    if payload.get("source_provenance") != build_current_two_d_source_provenance():
        raise ValueError(
            "2D result source provenance is stale relative to the current corrected checkout."
        )

    required_names = tuple(payload.get("required_artifacts") or ())
    recorded_hashes = payload.get("artifact_sha256") or {}
    recorded_sizes = payload.get("artifact_sizes_bytes") or {}
    if (
        required_names != REQUIRED_TWO_D_ARTIFACT_FILENAMES
        or set(recorded_hashes) != set(required_names)
        or set(recorded_sizes) != set(required_names)
    ):
        raise ValueError("2D completion metadata does not enumerate the exact final artifact set.")
    for name in required_names:
        if Path(name).name != name:
            raise ValueError(f"Unsafe 2D artifact name in completion metadata: {name!r}")
        artifact = Path(group_dir) / name
        if not artifact.is_file():
            raise FileNotFoundError(f"Completed 2D artifact is missing: {artifact}")
        actual_size = artifact.stat().st_size
        if actual_size != recorded_sizes[name]:
            raise ValueError(
                f"Completed 2D artifact size changed for {name}: "
                f"recorded={recorded_sizes[name]}, actual={actual_size}."
            )
        actual_hash = _file_sha256(artifact)
        if actual_hash != recorded_hashes[name]:
            raise ValueError(
                f"Completed 2D artifact hash changed for {name}: "
                f"recorded={recorded_hashes[name]}, actual={actual_hash}."
            )

    observed_design = _validate_agent_csv_contract(
        Path(group_dir) / experiment_config.AGENT_LEVEL_FILENAME,
        expected_seeds=expected_seeds,
        expected_agent_ids=expected_agent_ids,
    )
    if payload.get("observed_agent_design") != observed_design:
        raise ValueError("2D completion metadata and agent-level observation design disagree.")

    if require_paper_configuration:
        required = {
            "expected_seed_start": PAPER_TWO_D_SEED_START,
            "expected_num_seeds": PAPER_TWO_D_NUM_SEEDS,
            "expected_agents_per_seed": PAPER_TWO_D_AGENTS_PER_SEED,
        }
        mismatches = {
            field: (payload.get(field), expected)
            for field, expected in required.items()
            if payload.get(field) != expected
        }
        if mismatches:
            raise ValueError(f"2D results are not the canonical paper run: {mismatches}")
        if observed_design.get("agent_assignments") != _expected_agent_assignments():
            raise ValueError(
                "2D results do not use the canonical five-agents-per-observation-bucket design."
            )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or finalize fail-closed provenance for one 2D Slurm worker part."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser(
        "begin-run",
        help="Atomically invalidate one cluster aggregate before workers are submitted.",
    )
    run_parser.add_argument("--group-dir", type=Path, required=True)
    run_parser.add_argument("--seed-start", type=int, required=True)
    run_parser.add_argument("--num-seeds", type=int, required=True)
    run_parser.add_argument("--expected-agents", type=int, default=25)
    run_parser.add_argument("--parts-per-group", type=int, required=True)

    part_parser = subparsers.add_parser(
        "run-paper-part",
        help="Run and seal one partition from the authoritative paper configuration.",
    )
    part_parser.add_argument("--part-dir", type=Path, required=True)
    part_parser.add_argument("--seed-start", type=int, required=True)
    part_parser.add_argument("--num-seeds", type=int, required=True)
    part_parser.add_argument("--expected-agents", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "begin-run":
        path = begin_two_d_aggregation_metadata(
            args.group_dir,
            expected_seed_start=args.seed_start,
            expected_num_seeds=args.num_seeds,
            expected_agents_per_seed=args.expected_agents,
            parts_per_group=args.parts_per_group,
        )
        print(f"[2d-provenance] Marked cluster aggregate incomplete: {path.resolve()}")
        return 0
    if args.command == "run-paper-part":
        path = run_paper_two_d_part(
            args.part_dir,
            seed_start=args.seed_start,
            num_seeds=args.num_seeds,
            expected_agents_per_seed=args.expected_agents,
        )
        print(f"[2d-provenance] Completed and sealed paper worker part: {path.resolve()}")
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
