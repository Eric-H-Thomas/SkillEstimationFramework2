"""Run JEEDS on legacy data with xG maps downsampled to new-grid resolution."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from BlackhawksSkillEstimation.BlackhawksJEEDS import (
    DEFAULT_NUM_EXECUTION_SKILLS,
    DEFAULT_NUM_PLANNING_SKILLS,
    estimate_player_skill,
    load_player_data,
)


DEFAULT_CONFIG_PATH = Path("Data/Hockey/jobs/forwards21-25.json")
DEFAULT_DATA_ROOT = Path("Data/Hockey")
DEFAULT_OUTPUT_ROOT = Path("Data/Hockey_resolution_testing")
DEFAULT_TARGET_NZ = 31
DEFAULT_TARGET_NY = 51

_SHOT_GROUP_ALIASES = {
    "ws": "wristshot_snapshot",
    "bh": "backhand",
    "ss": "slapshot",
    "dk": "deke",
}


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expand_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    cluster_plan = config.get("cluster_plan", {})
    jobs = cluster_plan.get("jobs")
    if isinstance(jobs, list) and jobs:
        return [dict(job) for job in jobs]

    filters = config.get("data_filters", {})
    player_ids = [int(x) for x in filters.get("player_ids", [])]
    seasons = [int(x) for x in filters.get("seasons", [])]
    shot_groups = [str(x) for x in filters.get("shot_groups", [])]
    split_mode = str(cluster_plan.get("split_mode", "per_season"))

    out: list[dict[str, Any]] = []
    if split_mode == "all_selected_seasons_together":
        for player_id in player_ids:
            for shot_group in shot_groups:
                out.append(
                    {
                        "player_id": player_id,
                        "season": -1,
                        "shot_group": shot_group,
                        "count": None,
                        "missing_local_data": False,
                        "eligible": True,
                    }
                )
        return out

    for player_id in player_ids:
        for season in seasons:
            for shot_group in shot_groups:
                out.append(
                    {
                        "player_id": player_id,
                        "season": int(season),
                        "shot_group": shot_group,
                        "count": None,
                        "missing_local_data": False,
                        "eligible": True,
                    }
                )
    return out


def _normalize_shot_group(shot_group: str) -> str:
    return _SHOT_GROUP_ALIASES.get(shot_group, shot_group)


def _build_weights(n_src: int, n_dst: int) -> np.ndarray:
    if n_src <= 0 or n_dst <= 0:
        raise ValueError("Source/target sizes must be positive.")

    src_edges = np.linspace(0.0, float(n_src), n_src + 1)
    dst_edges = np.linspace(0.0, float(n_src), n_dst + 1)
    weights = np.zeros((n_dst, n_src), dtype=float)

    for j in range(n_dst):
        left = float(dst_edges[j])
        right = float(dst_edges[j + 1])
        width = right - left
        if width <= 0:
            continue
        start_idx = int(np.floor(left))
        end_idx = int(np.ceil(right))
        for i in range(start_idx, end_idx):
            if i < 0 or i >= n_src:
                continue
            overlap = min(right, src_edges[i + 1]) - max(left, src_edges[i])
            if overlap > 0:
                weights[j, i] = overlap / width

    return weights


def _downsample_value_map(
    value_map: np.ndarray,
    target_nz: int,
    target_ny: int,
    weight_cache: dict[tuple[str, int, int], np.ndarray],
) -> np.ndarray:
    src_nz, src_ny = int(value_map.shape[0]), int(value_map.shape[1])
    if src_nz == target_nz and src_ny == target_ny:
        return value_map

    key_z = ("z", src_nz, target_nz)
    key_y = ("y", src_ny, target_ny)
    if key_z not in weight_cache:
        weight_cache[key_z] = _build_weights(src_nz, target_nz)
    if key_y not in weight_cache:
        weight_cache[key_y] = _build_weights(src_ny, target_ny)

    w_z = weight_cache[key_z]
    w_y = weight_cache[key_y]

    tmp = w_z @ value_map
    return tmp @ w_y.T


def _downsample_shot_maps(
    shot_maps: dict[int, dict[str, object]],
    event_ids: set[int],
    target_nz: int,
    target_ny: int,
    weight_cache: dict[tuple[str, int, int], np.ndarray],
) -> tuple[dict[int, dict[str, object]], dict[str, Any]]:
    downsampled: dict[int, dict[str, object]] = {}
    stats: dict[str, Any] = {
        "processed": 0,
        "missing": 0,
        "skipped_invalid": 0,
        "shape_counts": {},
        "target_shape": [int(target_nz), int(target_ny)],
    }

    grid_y = np.linspace(-5.0, 5.0, int(target_ny))
    grid_z = np.linspace(0.0, 6.0, int(target_nz))

    for eid in event_ids:
        entry = shot_maps.get(int(eid))
        if entry is None:
            stats["missing"] += 1
            continue

        value_map = np.asarray(entry.get("value_map"))
        if value_map.ndim != 2:
            stats["skipped_invalid"] += 1
            continue

        shape_key = f"{value_map.shape[0]}x{value_map.shape[1]}"
        stats["shape_counts"][shape_key] = int(stats["shape_counts"].get(shape_key, 0)) + 1

        downsampled_map = _downsample_value_map(
            value_map=value_map,
            target_nz=target_nz,
            target_ny=target_ny,
            weight_cache=weight_cache,
        )

        new_entry = dict(entry)
        new_entry["value_map"] = downsampled_map
        new_entry["grid_y"] = grid_y
        new_entry["grid_z"] = grid_z
        downsampled[int(eid)] = new_entry
        stats["processed"] += 1

    return downsampled, stats


def _select_job(
    jobs: list[dict[str, Any]],
    job_index: int,
) -> dict[str, Any]:
    if job_index < 0 or job_index >= len(jobs):
        raise IndexError(f"Job index {job_index} out of range for {len(jobs)} jobs.")
    return jobs[job_index]


def _build_candidate_skills(num_execution_skills: int) -> list[float]:
    return np.linspace(0.004, 0.25, int(num_execution_skills)).tolist()


def _season_tag(seasons: list[int]) -> str:
    if not seasons:
        return "none"
    if len(seasons) == 1:
        return str(seasons[0])
    return "multi"


def _write_partial(output_root: Path, name: str, payload: dict[str, Any]) -> Path:
    partials_dir = output_root / "reports" / "partials"
    partials_dir.mkdir(parents=True, exist_ok=True)
    output_path = partials_dir / name
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return output_path


def _run_job(
    *,
    job: dict[str, Any],
    job_index: int,
    config: dict[str, Any],
    data_root: Path,
    output_root: Path,
    target_nz: int,
    target_ny: int,
    include_ineligible: bool,
) -> Path:
    player_id = int(job.get("player_id"))
    season = int(job.get("season", -1))
    shot_group = _normalize_shot_group(str(job.get("shot_group", "")))

    filters = config.get("data_filters", {})
    fallback_seasons = [int(x) for x in filters.get("seasons", [])]
    seasons_for_job = fallback_seasons if season < 0 else [season]

    estimator_cfg = config.get("estimator", {})
    num_execution_skills = int(estimator_cfg.get("num_execution_skills", DEFAULT_NUM_EXECUTION_SKILLS))
    num_planning_skills = int(estimator_cfg.get("num_planning_skills", DEFAULT_NUM_PLANNING_SKILLS))
    rng_seed = int(estimator_cfg.get("rng_seed", 0))
    per_season = bool(estimator_cfg.get("per_season", True))
    save_intermediate_csv = bool(estimator_cfg.get("save_intermediate_csv", True))

    run_meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "job_index": int(job_index),
        "player_id": player_id,
        "season": season,
        "seasons": seasons_for_job,
        "shot_group": shot_group,
        "eligible": bool(job.get("eligible", True)),
        "missing_local_data": bool(job.get("missing_local_data", False)),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "target_shape": [int(target_nz), int(target_ny)],
        "num_execution_skills": num_execution_skills,
        "num_planning_skills": num_planning_skills,
        "rng_seed": rng_seed,
    }

    if not include_ineligible and not bool(job.get("eligible", True)):
        payload = {
            "run_metadata": run_meta,
            "status": "skipped_ineligible",
        }
        filename = f"job_{job_index}_player_{player_id}_season_{season}_skipped.json"
        return _write_partial(output_root, filename, payload)

    if bool(job.get("missing_local_data", False)):
        payload = {
            "run_metadata": run_meta,
            "status": "missing_local_data",
        }
        filename = f"job_{job_index}_player_{player_id}_season_{season}_missing.json"
        return _write_partial(output_root, filename, payload)

    try:
        df, shot_maps = load_player_data(
            player_id=player_id,
            seasons=seasons_for_job,
            data_dir=data_root,
        )
    except FileNotFoundError as exc:
        payload = {
            "run_metadata": run_meta,
            "status": "missing_files",
            "error": str(exc),
        }
        filename = f"job_{job_index}_player_{player_id}_season_{season}_missing_files.json"
        return _write_partial(output_root, filename, payload)

    df_lc = df.rename(columns=str.lower)
    if "event_id" in df_lc.columns:
        event_ids = set(
            pd.to_numeric(df_lc["event_id"], errors="coerce").dropna().astype(int).tolist()
        )
    else:
        event_ids = set()

    weight_cache: dict[tuple[str, int, int], np.ndarray] = {}
    start = time.perf_counter()
    downsampled_maps, downsample_stats = _downsample_shot_maps(
        shot_maps=shot_maps,
        event_ids=event_ids,
        target_nz=target_nz,
        target_ny=target_ny,
        weight_cache=weight_cache,
    )
    elapsed = time.perf_counter() - start
    downsample_stats["seconds"] = round(float(elapsed), 3)
    downsample_stats["event_ids"] = int(len(event_ids))

    if not downsampled_maps:
        payload = {
            "run_metadata": run_meta,
            "status": "no_shot_maps",
            "downsample": downsample_stats,
        }
        filename = f"job_{job_index}_player_{player_id}_season_{season}_no_maps.json"
        return _write_partial(output_root, filename, payload)

    result = estimate_player_skill(
        player_id=player_id,
        seasons=seasons_for_job,
        per_season=per_season,
        candidate_skills=_build_candidate_skills(num_execution_skills),
        num_planning_skills=num_planning_skills,
        rng_seed=rng_seed,
        return_intermediate_estimates=False,
        save_intermediate_csv=save_intermediate_csv,
        data_dir=output_root,
        confirm=False,
        offline_data=(df, downsampled_maps),
        shot_group=shot_group,
    )

    payload = {
        "run_metadata": run_meta,
        "status": "success",
        "downsample": downsample_stats,
        "result": result,
    }
    season_tag = _season_tag(seasons_for_job)
    filename = f"job_{job_index}_player_{player_id}_season_{season_tag}.json"
    return _write_partial(output_root, filename, payload)


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Run JEEDS with legacy xG maps downsampled to new-grid resolution",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-nz", type=int, default=DEFAULT_TARGET_NZ)
    parser.add_argument("--target-ny", type=int, default=DEFAULT_TARGET_NY)
    parser.add_argument("--job-index", type=int, default=None)
    parser.add_argument(
        "--array-base",
        type=int,
        choices=[0, 1],
        default=1,
        help="Interpret SLURM_ARRAY_TASK_ID as 0-based or 1-based (default: 1).",
    )
    parser.add_argument(
        "--include-ineligible",
        action="store_true",
        help="Include jobs marked ineligible in the config.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    jobs = _expand_jobs(config)

    if not args.include_ineligible:
        jobs = [j for j in jobs if bool(j.get("eligible", True))]

    if args.job_index is None:
        if "SLURM_ARRAY_TASK_ID" not in os.environ:
            raise RuntimeError("Missing SLURM_ARRAY_TASK_ID; provide --job-index for manual runs.")
        array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        job_index = array_id - args.array_base
    else:
        job_index = int(args.job_index)

    job = _select_job(jobs, job_index)
    output_path = _run_job(
        job=job,
        job_index=job_index,
        config=config,
        data_root=args.data_root,
        output_root=args.output_root,
        target_nz=int(args.target_nz),
        target_ny=int(args.target_ny),
        include_ineligible=bool(args.include_ineligible),
    )

    print(f"Wrote partial JSON: {output_path}")


if __name__ == "__main__":
    _cli()
