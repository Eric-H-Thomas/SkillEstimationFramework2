"""Run one player subsample-stability job from a config JSON.

Each job re-runs an existing estimator on a pre-drawn subset of the player's cached
shots and writes a single result JSON. Estimator outputs are redirected into a
per-job scratch directory so that concurrent array tasks -- which all share the same
aggregate season tag, and therefore the same default CSV filename -- cannot overwrite
each other, and so nothing lands in the per-season log tree the cross-season
stability scripts glob over.

Offline only: shots and reward surfaces come from cached parquet/npz.

Examples
--------
Dry-run the expanded job list:

    python -m BlackhawksSkillEstimation.run_player_subsample_config \
      --config Data/Hockey/jobs/player_subsample_950160.json --dry-run

Run one job by index:

    python -m BlackhawksSkillEstimation.run_player_subsample_config \
      --config Data/Hockey/jobs/player_subsample_950160.json --job-index 0

Run one job from a Slurm array index:

    SLURM_ARRAY_TASK_ID=1 python -m BlackhawksSkillEstimation.run_player_subsample_config \
      --config Data/Hockey/jobs/player_subsample_950160.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from BlackhawksSkillEstimation.BlackhawksJEEDS import load_player_data
from BlackhawksSkillEstimation.BlackhawksJEEDS import (
    estimate_player_skill as estimate_player_skill_jeeds,
)
from BlackhawksSkillEstimation.BlackhawksMCSE import (
    estimate_player_skill as estimate_player_skill_mcse,
)
from BlackhawksSkillEstimation.player_subsample_stability import (
    EXECUTION_SKILL_MAX,
    EXECUTION_SKILL_MIN,
    filter_shot_group,
    job_tag,
    logs_dir,
    result_path,
    results_dir,
    season_counts,
    season_fractions,
    sort_chronologically,
    subset_pool,
    trace_csv_path,
    unified_execution_skill,
    unified_map_execution_skill,
    work_dir,
)

# Keys copied verbatim from each estimator's result dict into the result JSON.
_JEEDS_ESTIMATE_KEYS = (
    "execution_skill",
    "rationality",
    "log10_rationality",
    "ees",
    "eps",
    "log10_eps",
)
_MCSE_ESTIMATE_KEYS = (
    "execution_skill_y",
    "execution_skill_z",
    "ees_y",
    "ees_z",
    "rho_map",
    "rho_ees",
    "rationality",
    "log10_rationality",
    "eps",
    "log10_eps",
    "num_particles",
)


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expand_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(job) for job in config.get("cluster_plan", {}).get("jobs", [])]


def _run_jeeds(
    *,
    player_id: int,
    seasons: list[int],
    offline_data: tuple[Any, Any],
    shot_group: str,
    settings: dict[str, Any],
    scratch_root: Path,
    scratch_name: str,
) -> dict[str, Any]:
    num_execution_skills = int(settings.get("num_execution_skills", 250))
    candidate_skills = np.linspace(
        EXECUTION_SKILL_MIN, EXECUTION_SKILL_MAX, num_execution_skills
    ).tolist()

    return estimate_player_skill_jeeds(
        player_id=player_id,
        seasons=seasons,
        per_season=False,
        candidate_skills=candidate_skills,
        num_planning_skills=int(settings.get("num_planning_skills", 250)),
        rng_seed=int(settings.get("rng_seed", 0)),
        save_intermediate_csv=bool(settings.get("save_intermediate_csv", True)),
        confirm=False,
        offline_data=offline_data,
        shot_group=shot_group,
        data_dir=scratch_root,
        player_dir_name=scratch_name,
    )


def _run_mcse(
    *,
    player_id: int,
    seasons: list[int],
    offline_data: tuple[Any, Any],
    shot_group: str,
    settings: dict[str, Any],
    scratch_root: Path,
    scratch_name: str,
) -> dict[str, Any]:
    return estimate_player_skill_mcse(
        player_id=player_id,
        seasons=seasons,
        per_season=False,
        num_particles=int(settings.get("num_particles", 500)),
        noise=settings.get("noise"),
        resample_percent=float(settings.get("resample_percent", 0.9)),
        resample_neff=bool(settings.get("resample_neff", True)),
        resampling_method=str(settings.get("resampling_method", "systematic")),
        ranges=settings.get("ranges"),
        lambda_mode=str(settings.get("lambda_mode", "estimated")),
        rng_seed=int(settings.get("rng_seed", 0)),
        save_intermediate_csv=bool(settings.get("save_intermediate_csv", True)),
        confirm=False,
        offline_data=offline_data,
        shot_group=shot_group,
        data_dir=scratch_root,
        player_dir_name=scratch_name,
        compute_maxg=False,
    )


def run_single_job(
    job: dict[str, Any],
    *,
    config: dict[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    estimator = str(job["estimator"])
    sample_key = str(job["sample_key"])

    player_id = int(config["player_id"])
    shot_group = str(config["shot_group"])
    seasons = [int(s) for s in config["seasons"]]
    data_root = Path(config["data_root"])
    run_dir = Path(config["output_root"])

    sample = config["samples"][sample_key]
    event_ids = [int(x) for x in sample["event_ids"]]

    df, shot_maps = load_player_data(player_id=player_id, seasons=seasons, data_dir=data_root)
    pool_df = sort_chronologically(filter_shot_group(df, shot_group))
    sub_df = subset_pool(pool_df, event_ids)

    missing = len(event_ids) - len(sub_df)
    if missing:
        print(f"  WARNING: {missing} sampled event_id(s) absent from the cached pool.")

    counts = season_counts(sub_df)
    record: dict[str, Any] = {
        "experiment": config["experiment"],
        "run_name": config["run_name"],
        "config_path": str(config_path),
        "player_id": player_id,
        "estimator": estimator,
        "sample_key": sample_key,
        "is_baseline": bool(job.get("is_baseline", False)),
        "n_requested": job.get("n_requested"),
        "seed": job.get("seed"),
        "shot_group": shot_group,
        "seasons": seasons,
        "sampled_event_ids": len(event_ids),
        "season_counts": counts,
        "season_fractions": season_fractions(counts),
        "smoke": bool(config.get("smoke", False)),
    }

    if sub_df.empty:
        record.update({"status": "no_data", "num_shots": 0})
        return record

    scratch_root = work_dir(run_dir)
    scratch_name = job_tag(estimator, sample_key)
    settings = config["estimators"][estimator]
    offline_data = (sub_df, shot_maps)

    started = time.perf_counter()
    if estimator == "jeeds":
        result = _run_jeeds(
            player_id=player_id,
            seasons=seasons,
            offline_data=offline_data,
            shot_group=shot_group,
            settings=settings,
            scratch_root=scratch_root,
            scratch_name=scratch_name,
        )
        estimate_keys = _JEEDS_ESTIMATE_KEYS
    elif estimator == "mcse":
        result = _run_mcse(
            player_id=player_id,
            seasons=seasons,
            offline_data=offline_data,
            shot_group=shot_group,
            settings=settings,
            scratch_root=scratch_root,
            scratch_name=scratch_name,
        )
        estimate_keys = _MCSE_ESTIMATE_KEYS
    else:
        raise ValueError(f"Unknown estimator: {estimator!r}")
    elapsed = time.perf_counter() - started

    estimates = {key: result.get(key) for key in estimate_keys}
    record.update(
        {
            "status": str(result.get("status", "unknown")),
            "num_shots": int(result.get("num_shots", 0) or 0),
            "estimates": estimates,
            # Flattened, estimator-agnostic metrics so plotting does not have to
            # branch on which estimator produced the row.
            "exec_skill": unified_execution_skill(estimator, estimates),
            "map_exec_skill": unified_map_execution_skill(estimator, estimates),
            "log10_eps": _as_optional_float(estimates.get("log10_eps")),
            "log10_map_rationality": _as_optional_float(estimates.get("log10_rationality")),
            "estimator_settings": {k: v for k, v in settings.items() if k != "ranges"},
            "runtime_seconds": round(elapsed, 3),
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    if result.get("warning"):
        record["warning"] = str(result["warning"])

    trace_source = result.get("csv_path")
    if isinstance(trace_source, str) and trace_source and Path(trace_source).exists():
        destination = trace_csv_path(run_dir, estimator, sample_key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(trace_source, destination)
        record["trace_csv"] = str(destination)

    # The scratch tree only holds estimator bookkeeping (timing logs and the CSV we
    # just moved out); dropping it keeps a 302-job array from leaving 302 stub dirs.
    shutil.rmtree(scratch_root / "players" / scratch_name, ignore_errors=True)

    return record


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _describe(job: dict[str, Any]) -> str:
    return (
        f"estimator={job['estimator']} sample={job['sample_key']} "
        f"n_shots={job.get('n_shots')} seed={job.get('seed')}"
    )


def existing_result_is_complete(path: Path) -> bool:
    """True only for a readable result JSON whose status is ``success``.

    Failed, unreadable, or incomplete files must be retried on resume. Writing
    ``status=error`` into the same path as a success would otherwise make a
    crashed array task look finished.
    """
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(record, dict) and record.get("status") == "success"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run player subsample-stability jobs from a config JSON.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job-index", type=int, default=None, help="Zero-based job index to run.")
    parser.add_argument("--dry-run", action="store_true", help="Print expanded jobs and exit.")
    parser.add_argument(
        "--array-base",
        type=int,
        choices=[0, 1],
        default=1,
        help="Interpret SLURM_ARRAY_TASK_ID as 0-based or 1-based (default: 1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run jobs even when a successful result JSON already exists. "
        "Failed or unreadable results are retried without this flag.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    jobs = [job for job in _expand_jobs(config) if job.get("eligible", True)]

    if args.dry_run:
        print(f"Config: {args.config}")
        print(f"Player: {config['player_id']}  seasons={config['seasons']}")
        print(f"Pool:   {config['sampling']['pool_size']} shots")
        print(f"Jobs:   {len(jobs)}")
        for idx, job in enumerate(jobs):
            print(f"[{idx}] {_describe(job)}")
        return

    index = args.job_index
    if index is None:
        env_index = os.getenv("SLURM_ARRAY_TASK_ID")
        if env_index is not None:
            index = int(env_index) - int(args.array_base)

    if index is None:
        selected = jobs
    else:
        if index < 0 or index >= len(jobs):
            raise IndexError(f"job-index {index} is out of range for {len(jobs)} jobs")
        selected = [jobs[index]]

    run_dir = Path(config["output_root"])
    results_dir(run_dir).mkdir(parents=True, exist_ok=True)
    logs_dir(run_dir).mkdir(parents=True, exist_ok=True)

    print(f"Running {len(selected)} job(s) from {args.config}...")
    for local_idx, job in enumerate(selected, start=1):
        out_path = result_path(run_dir, job["estimator"], job["sample_key"])
        if not args.overwrite and existing_result_is_complete(out_path):
            print(f"[{local_idx}/{len(selected)}] skip (exists): {_describe(job)}")
            continue

        print(f"[{local_idx}/{len(selected)}] {_describe(job)}")
        try:
            record = run_single_job(job, config=config, config_path=args.config)
        except Exception as exc:  # noqa: BLE001 - one bad job must not kill the array task
            record = {
                "experiment": config["experiment"],
                "player_id": int(config["player_id"]),
                "estimator": job["estimator"],
                "sample_key": job["sample_key"],
                "status": "error",
                "error": str(exc),
                "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            print(f"  FAILED: {exc}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"  status={record.get('status')} -> {out_path}")


if __name__ == "__main__":
    main()
