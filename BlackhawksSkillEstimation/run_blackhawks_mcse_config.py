"""Run Blackhawks MCSE jobs from a JSON config."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from BlackhawksSkillEstimation.BlackhawksJEEDS import load_player_data
from BlackhawksSkillEstimation.BlackhawksMCSE import (
    DEFAULT_MCSE_RANGES,
    DEFAULT_MCSE_NOISE,
    DEFAULT_NUM_PARTICLES,
    DEFAULT_RESAMPLE_PERCENT,
    DEFAULT_RESAMPLING_METHOD,
    estimate_player_skill,
)
from BlackhawksSkillEstimation.plot_intermediate_estimates_mcse import plot_intermediate_estimates_mcse
from BlackhawksSkillEstimation.run_blackhawks_config import (
    _apply_partition_filter,
    _default_summary_path,
    _expand_jobs,
    _load_config,
    _log_partition_filter_stats,
    _normalize_shot_group,
    _partition_suffix_from_values,
)


def _finalize_mcse_csv_and_png(
    result: dict[str, Any],
    *,
    partition_column: str | None,
    partition_values: list[str],
    generate_convergence_png: bool,
    include_map_estimates: bool,
) -> None:
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
            png = plot_intermediate_estimates_mcse(
                csv_file,
                show=False,
                include_map_estimates=include_map_estimates,
            )
            if png is not None:
                result["convergence_png"] = str(png)


def _run_single_job(job: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
    data_root = Path(config.get("data_root", "Data/Hockey"))
    filters = config.get("data_filters", {})
    estimator = config.get("estimator", {})
    maxg_cfg = config.get("maxg", {})

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

    df = _apply_partition_filter(df, partition_column=partition_column, partition_values=partition_values)
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

    ranges = estimator.get("ranges", DEFAULT_MCSE_RANGES)
    result = estimate_player_skill(
        player_id=player_id,
        seasons=seasons_for_job,
        per_season=bool(estimator.get("per_season", True)),
        num_particles=int(estimator.get("num_particles", DEFAULT_NUM_PARTICLES)),
        noise=estimator.get("noise", DEFAULT_MCSE_NOISE),
        resample_percent=float(estimator.get("resample_percent", DEFAULT_RESAMPLE_PERCENT)),
        resample_neff=bool(estimator.get("resample_neff", True)),
        resampling_method=str(estimator.get("resampling_method", DEFAULT_RESAMPLING_METHOD)),
        ranges=ranges,
        rng_seed=int(estimator.get("rng_seed", 0)),
        save_intermediate_csv=bool(estimator.get("save_intermediate_csv", True)),
        confirm=False,
        offline_data=(df, shot_maps),
        shot_group=shot_group,
        data_dir=data_root,
        compute_maxg=bool(maxg_cfg.get("compute", estimator.get("compute_maxg", False))),
        benchmark_tag=maxg_cfg.get("benchmark_tag") or estimator.get("benchmark_tag"),
        benchmark_dir=Path(maxg_cfg.get("benchmark_dir", "Data/Hockey/benchmarks")),
    )

    generate_convergence_png = bool(estimator.get("generate_convergence_png", True))
    include_map_estimates = bool(estimator.get("convergence_png_include_map", True))
    if isinstance(result.get("per_season_results"), dict):
        for season_result in result["per_season_results"].values():
            if isinstance(season_result, dict):
                _finalize_mcse_csv_and_png(
                    season_result,
                    partition_column=partition_column,
                    partition_values=partition_values,
                    generate_convergence_png=generate_convergence_png,
                    include_map_estimates=include_map_estimates,
                )
    else:
        _finalize_mcse_csv_and_png(
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Blackhawks MCSE jobs from config JSON")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job-index", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-ineligible", action="store_true")
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--array-base", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()

    config = _load_config(args.config)
    all_jobs = _expand_jobs(config)
    jobs = all_jobs if args.include_ineligible else [j for j in all_jobs if bool(j.get("eligible", True))]

    if args.dry_run:
        print(f"Config: {args.config}")
        print(f"Jobs selected: {len(jobs)}")
        for idx, job in enumerate(jobs):
            print(f"[{idx}] player_id={job['player_id']} season={job.get('season')} shot_group={job['shot_group']}")
        return

    requested_index = args.job_index
    if requested_index is None:
        env_index = os.getenv("SLURM_ARRAY_TASK_ID")
        if env_index is not None:
            requested_index = int(env_index) - int(args.array_base)

    selected_jobs = jobs if requested_index is None else [jobs[requested_index]]
    results: list[dict[str, Any]] = []
    for local_idx, job in enumerate(selected_jobs):
        try:
            result = _run_single_job(job, config=config)
            result["job"] = job
            results.append(result)
            print(f"Completed job: player_id={job['player_id']} status={result.get('status')}")
        except Exception as exc:
            results.append({"status": "error", "error": str(exc), "job": job})
            print(f"Failed job: player_id={job['player_id']} error={exc}")

    if args.summary_out or config.get("output", {}).get("write_run_summary", False):
        out_path = args.summary_out or _default_summary_path(args.config)
        out_path.write_text(
            json.dumps({
                "config": str(args.config),
                "results": results,
                "completed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote summary: {out_path}")


if __name__ == "__main__":
    main()
