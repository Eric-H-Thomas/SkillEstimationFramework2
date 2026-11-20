"""Run one MCSE parameter-sweep job from a config JSON.

Each array task estimates one (setting, player, seed) on the shared shot
prefix and writes one result JSON plus a per-shot trace. Estimator scratch
goes to a private work directory so concurrent tasks cannot overwrite each
other, and so nothing lands in the per-player log tree.

Offline only: shots and reward surfaces come from cached parquet/npz.

Examples
--------
Dry-run::

    python -m BlackhawksSkillEstimation.run_mcse_param_sweep_config \\
      --config Data/Hockey/jobs/mcse_param_sweep_stage1.json --dry-run

One job::

    python -m BlackhawksSkillEstimation.run_mcse_param_sweep_config \\
      --config Data/Hockey/jobs/mcse_param_sweep_stage1.json --job-index 0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from BlackhawksSkillEstimation.BlackhawksJEEDS import load_player_data
from BlackhawksSkillEstimation.BlackhawksMCSE import (
    estimate_player_skill as estimate_player_skill_mcse,
)
from BlackhawksSkillEstimation.maxg_evaluator import compute_maxg_for_mcse_profile
from BlackhawksSkillEstimation.mcse_param_sweep import (
    TRACE_COLUMNS,
    event_ids_for_job,
    result_path,
    results_dir,
    summarize_trace,
    trace_csv_path,
    work_dir,
)
from BlackhawksSkillEstimation.player_subsample_stability import (
    filter_shot_group,
    sort_chronologically,
    subset_pool,
)

_ESTIMATE_KEYS = (
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
)


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expand_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(job) for job in config.get("cluster_plan", {}).get("jobs", [])]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def write_trace_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TRACE_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in TRACE_COLUMNS})


def _maybe_maxg(config: dict[str, Any], result: dict[str, Any]) -> float | None:
    """MAXG of the final EES profile when the tagged benchmark is on disk."""
    maxg = config.get("maxg") or {}
    tag = maxg.get("benchmark_tag")
    if not tag or result.get("ees_y") is None:
        return None
    directory = Path(str(maxg.get("benchmark_dir", "")))
    shots = directory / f"benchmark_shots_{tag}.parquet"
    maps = directory / f"benchmark_shot_maps_{tag}.npz"
    if not shots.exists() or not maps.exists():
        return None
    return float(
        compute_maxg_for_mcse_profile(
            benchmark_dir=directory,
            benchmark_tag=str(tag),
            x_y=float(result["ees_y"]),
            x_z=float(result["ees_z"]),
            rho=float(result.get("rho_ees") or 0.0),
        )
    )


def run_single_job(
    job: dict[str, Any],
    *,
    config: dict[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    player_id = int(job["player_id"])
    season = int(job["season"])
    shot_group = str(job["shot_group"])
    data_root = Path(config["data_root"])
    run_dir = Path(config["output_root"])
    event_ids = event_ids_for_job(config, job)
    # The filter writes proposal bookkeeping onto the ranges dict it is given.
    # Keep the saved copy to the start/end endpoints the score divides by.
    ranges = {
        "start": [float(value) for value in job["ranges"]["start"]],
        "end": [float(value) for value in job["ranges"]["end"]],
    }

    record: dict[str, Any] = {
        "experiment": config["experiment"],
        "run_name": config["run_name"],
        "stage": config["stage"],
        "config_path": str(config_path),
        "job_id": job["job_id"],
        "setting_key": job["setting_key"],
        "player_id": player_id,
        "seed": int(job["seed"]),
        "season": season,
        "shot_group": shot_group,
        "role": job.get("role"),
        "num_particles": int(job["num_particles"]),
        "noise": [float(value) for value in job["noise"]],
        "resample_percent": float(job["resample_percent"]),
        "resample_neff": bool(job["resample_neff"]),
        "resampling_method": job["resampling_method"],
        "lambda_mode": job["lambda_mode"],
        "ranges_name": job["ranges_name"],
        "ranges": ranges,
        "sampled_event_ids": len(event_ids),
        "smoke": bool(config.get("smoke", False)),
    }

    df, shot_maps = load_player_data(player_id=player_id, seasons=[season], data_dir=data_root)
    pool = sort_chronologically(filter_shot_group(df, shot_group))
    if "season" in pool.columns:
        pool = pool[pool["season"].astype(int) == season]
    sub_df = subset_pool(pool, event_ids)
    missing = len(event_ids) - len(sub_df)
    if missing:
        print(f"  WARNING: {missing} prefix event_id(s) absent from the cached pool.")
    if sub_df.empty:
        record.update({"status": "no_data", "num_shots": 0})
        return record

    scratch_root = work_dir(run_dir)
    scratch_name = str(job["job_id"])
    # Initial particles, rejuvenation draws, and filterpy's systematic resample
    # all use the global NumPy RNG; only the process noise uses ``rng_seed``.
    np.random.seed(int(job["seed"]))
    started = time.perf_counter()
    result = estimate_player_skill_mcse(
        player_id=player_id,
        seasons=[season],
        per_season=False,
        num_particles=int(job["num_particles"]),
        noise=job["noise"],
        resample_percent=float(job["resample_percent"]),
        resample_neff=bool(job["resample_neff"]),
        resampling_method=str(job["resampling_method"]),
        ranges={
            "start": list(ranges["start"]),
            "end": list(ranges["end"]),
        },
        lambda_mode=str(job["lambda_mode"]),
        rng_seed=int(job["seed"]),
        return_intermediate_estimates=True,
        record_filter_health=True,
        save_intermediate_csv=False,
        confirm=False,
        offline_data=(sub_df, shot_maps),
        shot_group=shot_group,
        data_dir=scratch_root,
        player_dir_name=scratch_name,
        compute_maxg=False,
    )
    elapsed = time.perf_counter() - started

    skill_log = result.get("skill_log") or []
    if not isinstance(skill_log, list):
        skill_log = []
    health = summarize_trace(skill_log, ranges, int(job["num_particles"]))
    estimates = {key: _as_optional_float(result.get(key)) for key in _ESTIMATE_KEYS}
    record.update(estimates)
    record.update(
        {
            "status": str(result.get("status", "unknown")),
            "num_shots": int(result.get("num_shots", 0) or 0),
            "skipped_invalid_map": int(result.get("skipped_invalid_map", 0) or 0),
            "skipped_proximity": int(result.get("skipped_proximity", 0) or 0),
            "runtime_seconds": round(elapsed, 3),
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **health,
        }
    )
    if result.get("warning"):
        record["warning"] = str(result["warning"])

    try:
        maxg_ees = _maybe_maxg(config, result)
    except Exception as exc:  # noqa: BLE001 - MAXG is a side column, not the sweep
        record["maxg_error"] = str(exc)
        maxg_ees = None
    record["maxg_ees"] = maxg_ees

    if skill_log and record["status"] == "success":
        destination = trace_csv_path(run_dir, str(job["job_id"]))
        write_trace_csv(destination, skill_log)
        record["trace_csv"] = str(destination)

    shutil.rmtree(scratch_root / "players" / scratch_name, ignore_errors=True)
    return _json_ready(record)


def existing_result_is_complete(path: Path) -> bool:
    """True only for a readable result JSON whose status is ``success``."""
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(record, dict) and record.get("status") == "success"


def _describe(job: dict[str, Any]) -> str:
    noise = job.get("noise")
    return (
        f"{job['job_id']} player={job['player_id']} seed={job['seed']} "
        f"N={job['num_particles']} noise={noise} resample={job['resample_percent']} "
        f"neff={job['resample_neff']} ranges={job['ranges_name']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCSE parameter-sweep jobs from a config JSON.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job-index", type=int, default=None, help="Zero-based index into eligible jobs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--array-base", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run a job even when a successful result JSON already exists.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    jobs = [job for job in _expand_jobs(config) if job.get("eligible", True)]

    if args.dry_run:
        print(f"Config: {args.config}")
        print(f"Stage:  {config.get('stage')}  season={config.get('season')} players={config.get('player_ids')}")
        print(f"Jobs:   {len(jobs)}")
        for index, job in enumerate(jobs):
            print(f"[{index}] {_describe(job)}")
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

    print(f"Running {len(selected)} job(s) from {args.config}...")
    for local_idx, job in enumerate(selected, start=1):
        out_path = result_path(run_dir, str(job["job_id"]))
        if not args.overwrite and existing_result_is_complete(out_path):
            print(f"[{local_idx}/{len(selected)}] skip (exists): {_describe(job)}")
            continue
        print(f"[{local_idx}/{len(selected)}] {_describe(job)}")
        try:
            record = run_single_job(job, config=config, config_path=args.config)
        except Exception as exc:  # noqa: BLE001 - one bad job must not kill the array task
            record = {
                "experiment": config.get("experiment"),
                "job_id": job.get("job_id"),
                "setting_key": job.get("setting_key"),
                "player_id": job.get("player_id"),
                "seed": job.get("seed"),
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
