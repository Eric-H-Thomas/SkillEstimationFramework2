"""Run Blackhawks JEEDS jobs from a JSON config.

This runner is intended for local dry-runs and cluster job-array execution.
It consumes config files exported by the Streamlit Config Builder.

Cluster usage
-------------
Submit via the sbatch wrapper script:
    sbatch run_blackhawks_config.sbatch Data/Hockey/jobs/<config>.json

Examples
--------
Dry-run all jobs:
    python -m BlackhawksSkillEstimation.run_blackhawks_config --config Data/Hockey/jobs/<config>.json --dry-run

Run one job by index:
    python -m BlackhawksSkillEstimation.run_blackhawks_config --config Data/Hockey/jobs/<config>.json --job-index 12

Run one job from SLURM array index:
    SLURM_ARRAY_TASK_ID=12 python -m BlackhawksSkillEstimation.run_blackhawks_config --config Data/Hockey/jobs/<config>.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from BlackhawksSkillEstimation.BlackhawksJEEDS import estimate_player_skill, load_player_data
from BlackhawksSkillEstimation.plot_intermediate_estimates import plot_intermediate_estimates


_SHOT_GROUP_ALIASES = {
    "ws": "wristshot_snapshot",
    "bh": "backhand",
    "ss": "slapshot",
    "dk": "deke",
}


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_candidate_skills(num_execution_skills: int) -> list[float]:
    return np.linspace(0.004, np.pi / 4.0, int(num_execution_skills)).tolist()


def _normalize_shot_group(shot_group: str) -> str:
    return _SHOT_GROUP_ALIASES.get(shot_group, shot_group)


def _slug(value: object) -> str:
    """Convert value to lowercase alphanumeric slug (dashes for separators)."""
    raw = str(value).strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", raw)
    return cleaned.strip("-") or "value"


def _partition_column_slug(partition_column: str) -> str:
    """Return compact filename token for a partition column."""
    if partition_column == "partition_geometry":
        return "geometry"
    return _slug(partition_column)


def _partition_suffix_from_values(
    partition_column: str | None,
    partition_values: list[str],
) -> str:
    """Generate filename suffix encoding partition filters to prevent collisions.
    
    Examples:
        - Single value: '__partition-slot'
        - Multiple values: '__partition-multi-left-shallow-or-right-shallow'
        - Geometry: '__geometry-left-shallow'
    """
    if not partition_column or not partition_values:
        return ""

    unique_values = sorted({str(v) for v in partition_values})
    if len(unique_values) == 1:
        value_slug = _slug(unique_values[0])
    else:
        value_slug = "multi-" + "-or-".join(_slug(v) for v in unique_values)
    return f"__{_partition_column_slug(partition_column)}-{value_slug}"


def _apply_partition_filter(
    df: pd.DataFrame,
    partition_column: str | None,
    partition_values: list[str] | None,
) -> pd.DataFrame:
    """Filter dataframe by partition column and values. Returns all rows if filter not specified."""
    if not partition_column:
        return df
    if partition_column not in df.columns:
        return df.iloc[0:0].copy()
    if not partition_values:
        return df

    values = set(str(v) for v in partition_values)
    series = df[partition_column].fillna("").astype(str)
    return df[series.isin(values)].copy()


def _log_partition_filter_stats(
    *,
    player_id: int,
    shot_group: str,
    partition_column: str | None,
    partition_values: list[str],
    pre_partition_count: int,
    df: pd.DataFrame,
) -> None:
    """Log partition filter telemetry: shots kept/dropped."""
    if not partition_column:
        return

    kept = len(df)
    dropped = pre_partition_count - kept
    selected_values = ", ".join(str(v) for v in partition_values) if partition_values else "ALL"
    print(f"  Partition [{partition_column}={selected_values}]: {kept}/{pre_partition_count} kept, {dropped} dropped.")
    if kept == 0:
        print(f"  WARNING: partition filter removed all shots (player_id={player_id}, shot_group={shot_group}).")


def _finalize_csv_and_png(
    result: dict[str, Any],
    *,
    partition_column: str | None,
    partition_values: list[str],
    generate_convergence_png: bool,
    include_map_estimates: bool,
) -> None:
    """Rename intermediate outputs to include partition filters, then render PNGs."""
    output_suffix = _partition_suffix_from_values(partition_column, partition_values)

    csv_path = result.get("csv_path")
    if isinstance(csv_path, str) and csv_path:
        csv_file = Path(csv_path)
        if output_suffix and not csv_file.stem.endswith(output_suffix):
            renamed_csv = csv_file.with_name(f"{csv_file.stem}{output_suffix}{csv_file.suffix}")
            csv_file.replace(renamed_csv)
            csv_file = renamed_csv
            result["csv_path"] = str(renamed_csv)

        if generate_convergence_png:
            plot_intermediate_estimates(
                csv_file,
                show=False,
                include_map_estimates=include_map_estimates,
            )
            result["convergence_png"] = str(csv_file.with_suffix(".png"))


def _expand_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    cluster_plan = config.get("cluster_plan", {})
    jobs = cluster_plan.get("jobs")
    if jobs:
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


def _run_single_job(
    job: dict[str, Any],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    data_root = Path(config.get("data_root", "Data/Hockey"))
    filters = config.get("data_filters", {})
    estimator = config.get("estimator", {})

    player_id = int(job["player_id"])
    shot_group = _normalize_shot_group(str(job["shot_group"]))

    selected_seasons = [int(x) for x in filters.get("seasons", [])]
    if int(job.get("season", -1)) >= 0:
        seasons_for_job = [int(job["season"])]
    else:
        seasons_for_job = selected_seasons

    if not seasons_for_job:
        return {
            "status": "invalid_job",
            "player_id": player_id,
            "shot_group": shot_group,
            "error": "No seasons available for this job.",
        }

    df, shot_maps = load_player_data(
        player_id=player_id,
        seasons=seasons_for_job,
        data_dir=data_root,
    )

    pre_partition_count = len(df)
    partition_column = filters.get("partition_column")
    partition_values = filters.get("partition_values") or []
    job_partition_value = job.get("partition_value")
    if partition_column and job_partition_value is not None:
        partition_values = [str(job_partition_value)]

    df = _apply_partition_filter(
        df,
        partition_column=partition_column,
        partition_values=partition_values,
    )

    _log_partition_filter_stats(
        player_id=player_id,
        shot_group=shot_group,
        partition_column=partition_column,
        partition_values=partition_values,
        pre_partition_count=pre_partition_count,
        df=df,
    )

    if df.empty:
        return {
            "status": "no_data",
            "player_id": player_id,
            "shot_group": shot_group,
            "seasons": seasons_for_job,
            "num_shots": 0,
        }

    candidate_skills = _build_candidate_skills(
        int(estimator.get("num_execution_skills", 50))
    )

    result = estimate_player_skill(
        player_id=player_id,
        seasons=seasons_for_job,
        per_season=bool(estimator.get("per_season", True)),
        candidate_skills=candidate_skills,
        num_planning_skills=int(estimator.get("num_planning_skills", 100)),
        rng_seed=int(estimator.get("rng_seed", 0)),
        save_intermediate_csv=bool(estimator.get("save_intermediate_csv", True)),
        confirm=False,
        offline_data=(df, shot_maps),
        shot_group=shot_group,
        data_dir=data_root,
    )

    generate_convergence_png = bool(estimator.get("generate_convergence_png", True))
    include_map_estimates = bool(estimator.get("convergence_png_include_map", True))
    if isinstance(result.get("per_season_results"), dict):
        for season_result in result["per_season_results"].values():
            if isinstance(season_result, dict):
                _finalize_csv_and_png(
                    season_result,
                    partition_column=partition_column,
                    partition_values=partition_values,
                    generate_convergence_png=generate_convergence_png,
                    include_map_estimates=include_map_estimates,
                )
    else:
        _finalize_csv_and_png(
            result,
            partition_column=partition_column,
            partition_values=partition_values,
            generate_convergence_png=generate_convergence_png,
            include_map_estimates=include_map_estimates,
        )

    result["player_id"] = player_id
    result["shot_group"] = shot_group
    result["seasons"] = seasons_for_job
    if job_partition_value is not None:
        result["partition_value"] = str(job_partition_value)
    return result


def _default_summary_path(config_path: Path) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return config_path.parent / f"run_summary_{ts}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Blackhawks JEEDS jobs from config JSON")
    parser.add_argument("--config", type=Path, required=True, help="Path to exported JSON config")
    parser.add_argument("--job-index", type=int, default=None, help="Zero-based job index to run")
    parser.add_argument("--dry-run", action="store_true", help="Print expanded jobs and exit")
    parser.add_argument(
        "--include-ineligible",
        action="store_true",
        help="Run jobs even when config marks them as ineligible",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional JSON output path for results summary",
    )
    parser.add_argument(
        "--array-base",
        type=int,
        choices=[0, 1],
        default=1,
        help="Interpret SLURM_ARRAY_TASK_ID as 0-based or 1-based (default: 1).",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    all_jobs = _expand_jobs(config)

    if not args.include_ineligible:
        jobs = [j for j in all_jobs if bool(j.get("eligible", True))]
    else:
        jobs = all_jobs

    if args.dry_run:
        print(f"Config: {args.config}")
        print(f"Jobs in config: {len(all_jobs)}")
        print(f"Jobs selected for execution: {len(jobs)}")
        for idx, job in enumerate(jobs):
            season = job.get("season", -1)
            season_txt = "ALL_SELECTED" if int(season) < 0 else str(season)
            partition_txt = f" partition_value={job.get('partition_value')}" if "partition_value" in job else ""
            print(
                f"[{idx}] player_id={job['player_id']} season={season_txt} "
                f"shot_group={job['shot_group']}{partition_txt} count={job.get('count')} eligible={job.get('eligible')}"
            )
        return

    requested_index = args.job_index
    if requested_index is None:
        env_index = os.getenv("SLURM_ARRAY_TASK_ID")
        if env_index is not None:
            requested_index = int(env_index) - int(args.array_base)

    selected_jobs: list[dict[str, Any]]
    if requested_index is None:
        selected_jobs = jobs
    else:
        if requested_index < 0 or requested_index >= len(jobs):
            raise IndexError(
                f"job-index {requested_index} is out of range for {len(jobs)} selected jobs"
            )
        selected_jobs = [jobs[requested_index]]

    print(f"Running {len(selected_jobs)} job(s) from config {args.config}...")

    results: list[dict[str, Any]] = []
    for local_idx, job in enumerate(selected_jobs):
        try:
            result = _run_single_job(job, config=config)
            result["job"] = job
            result["run_index"] = local_idx
            results.append(result)
            status = result.get("status", "unknown")
            print(
                f"Completed job {local_idx + 1}/{len(selected_jobs)}: "
                f"player_id={job['player_id']} shot_group={job['shot_group']} "
                f"partition_value={job.get('partition_value')} status={status}"
            )
        except Exception as exc:
            error_payload = {
                "status": "error",
                "error": str(exc),
                "job": job,
                "run_index": local_idx,
            }
            results.append(error_payload)
            print(
                f"Failed job {local_idx + 1}/{len(selected_jobs)}: "
                f"player_id={job['player_id']} shot_group={job['shot_group']} "
                f"partition_value={job.get('partition_value')} error={exc}"
            )

    write_summary = bool(config.get("output", {}).get("write_run_summary", False))
    if args.summary_out is not None:
        write_summary = True

    if write_summary:
        summary = {
            "config": str(args.config),
            "job_count": len(selected_jobs),
            "results": results,
            "completed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }

        out_path = args.summary_out or _default_summary_path(args.config)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote summary: {out_path}")
    else:
        print("Skipping run summary write (output.write_run_summary is false).")


if __name__ == "__main__":
    main()
