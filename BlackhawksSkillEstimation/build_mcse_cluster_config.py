"""Build a JSON config for league-wide MCSE cluster runs.

Scans local cached shot data (no estimator runs) and writes a config compatible
with ``run_blackhawks_mcse_config.py`` / ``run_blackhawks_mcse_config.sbatch``.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from BlackhawksApp import data_io
from BlackhawksSkillEstimation.BlackhawksMCSE import (
    DEFAULT_MCSE_RANGES,
    DEFAULT_MCSE_NOISE,
    DEFAULT_NUM_PARTICLES,
    DEFAULT_RESAMPLE_PERCENT,
    DEFAULT_RESAMPLING_METHOD,
)

LEGACY_DATA_ROOT = Path("Data/Hockey")
NEW_XG_DATA_ROOT = Path("Data/Hockey_xg_new")
LEGACY_BENCHMARK_TAG = "wristshot-snapshot_v1"
NEW_XG_BENCHMARK_TAG = "wristshot_snapshot_xgnew_v1"


def benchmark_settings_for_data_root(data_root: Path | str) -> tuple[str, str]:
    """Return (benchmark_tag, benchmark_dir) for a MCSE data root."""
    root = Path(data_root)
    normalized = str(root).replace("\\", "/")
    if normalized.endswith("Hockey_xg_new") or "Hockey_xg_new" in normalized:
        return NEW_XG_BENCHMARK_TAG, str(root / "benchmarks")
    return LEGACY_BENCHMARK_TAG, str(root / "benchmarks")


def refresh_cluster_jobs(
    config: dict[str, Any],
    *,
    data_root: Path,
    min_shots_per_job: int | None = None,
) -> dict[str, Any]:
    """Rebuild cluster_plan.jobs eligibility from cached data at ``data_root``."""
    filters = config.get("data_filters", {})
    player_ids = [int(x) for x in filters.get("player_ids", [])]
    seasons = [int(x) for x in filters.get("seasons", [])]
    shot_groups = [str(x) for x in filters.get("shot_groups", ["wristshot_snapshot"])]
    split_mode = str(config.get("cluster_plan", {}).get("split_mode", "per_season"))
    min_shots = int(
        min_shots_per_job
        if min_shots_per_job is not None
        else config.get("validation", {}).get("min_shots_per_job", 50)
    )

    summary_df = data_io.build_observation_summary(
        player_ids,
        seasons,
        shot_groups,
        data_dir=data_root,
    )
    if split_mode == "all_selected_seasons_together":
        jobs = _aggregate_jobs(summary_df, min_shots_per_job=min_shots, shot_groups=shot_groups)
    else:
        jobs = data_io.build_job_rows(summary_df, min_shots_per_job=min_shots)

    cluster = dict(config.get("cluster_plan", {}))
    cluster["jobs"] = jobs
    cluster["total_jobs"] = len(jobs)
    cluster["eligible_jobs"] = sum(1 for job in jobs if job.get("eligible"))
    out = dict(config)
    out["cluster_plan"] = cluster
    return out


def derive_mcse_config_for_data_root(
    config: dict[str, Any],
    data_root: Path | str,
    *,
    min_shots_per_job: int | None = None,
) -> dict[str, Any]:
    """Clone a MCSE config for another data root and refresh job eligibility."""
    root = Path(data_root)
    benchmark_tag, benchmark_dir = benchmark_settings_for_data_root(root)
    out = dict(config)
    out["data_root"] = str(root).replace("\\", "/")
    out["notes"] = (
        f"MCSE cluster config: data_root={root} "
        f"({'new xG' if 'xg_new' in str(root) else 'legacy xG'})."
    )
    maxg = dict(out.get("maxg", {}))
    maxg["benchmark_tag"] = benchmark_tag
    maxg["benchmark_dir"] = benchmark_dir
    out["maxg"] = maxg
    return refresh_cluster_jobs(out, data_root=root, min_shots_per_job=min_shots_per_job)


def derived_config_path(config_path: Path, *, suffix: str = "xgnew") -> Path:
    return config_path.with_name(f"{config_path.stem}.{suffix}{config_path.suffix}")


def _json_safe_ranges(ranges: dict[str, list]) -> dict[str, list[float]]:
    """Ensure estimator ranges are JSON-serializable plain floats."""
    return {
        "start": [float(x) for x in ranges["start"]],
        "end": [float(x) for x in ranges["end"]],
    }


def _aggregate_jobs(
    summary_df,
    *,
    min_shots_per_job: int,
    shot_groups: list[str],
) -> list[dict[str, Any]]:
    """One job per (player_id, shot_group) with season=-1 (all seasons together)."""
    if summary_df.empty:
        return []

    jobs: list[dict[str, Any]] = []
    grouped = summary_df.groupby(["player_id", "shot_group"], dropna=False)
    for (player_id, shot_group), group in grouped:
        missing = bool(group["missing_local_data"].all())
        count = int(group["count"].sum()) if not missing else 0
        jobs.append(
            {
                "player_id": int(player_id),
                "season": -1,
                "shot_group": str(shot_group),
                "count": count,
                "missing_local_data": missing,
                "eligible": bool(count >= min_shots_per_job and not missing),
            }
        )
    return jobs


def build_mcse_cluster_config(
    *,
    data_root: Path,
    player_ids: list[int],
    seasons: list[int],
    shot_groups: list[str],
    split_mode: str,
    min_shots_per_job: int,
    num_particles: int,
    generate_convergence_png: bool,
    sbatch_time: str,
    sbatch_mem: str,
    max_concurrent: int,
    rng_seed: int,
) -> dict[str, Any]:
    summary_df = data_io.build_observation_summary(
        player_ids,
        seasons,
        shot_groups,
        data_dir=data_root,
    )

    if split_mode == "all_selected_seasons_together":
        jobs = _aggregate_jobs(
            summary_df,
            min_shots_per_job=min_shots_per_job,
            shot_groups=shot_groups,
        )
    else:
        jobs = data_io.build_job_rows(summary_df, min_shots_per_job=min_shots_per_job)

    eligible_jobs = sum(1 for job in jobs if job.get("eligible"))
    benchmark_tag, benchmark_dir = benchmark_settings_for_data_root(data_root)

    return {
        "config_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": (
            f"MCSE cluster config: data_root={data_root} "
            f"({'new xG' if 'xg_new' in str(data_root) else 'legacy xG'})."
        ),
        "data_root": str(data_root).replace("\\", "/"),
        "data_filters": {
            "player_ids": player_ids,
            "seasons": seasons,
            "shot_groups": shot_groups,
        },
        "estimator": {
            "method": "mcse",
            "per_season": split_mode != "all_selected_seasons_together",
            "num_particles": int(num_particles),
            "noise": list(DEFAULT_MCSE_NOISE),
            "resample_percent": float(DEFAULT_RESAMPLE_PERCENT),
            "resample_neff": True,
            "resampling_method": DEFAULT_RESAMPLING_METHOD,
            "rng_seed": int(rng_seed),
            "ranges": _json_safe_ranges(DEFAULT_MCSE_RANGES),
            "save_intermediate_csv": True,
            "generate_convergence_png": bool(generate_convergence_png),
            "convergence_png_include_map": False,
            "compute_maxg": False,
        },
        "maxg": {
            "compute": False,
            "benchmark_tag": benchmark_tag,
            "benchmark_dir": benchmark_dir,
        },
        "output": {
            "write_run_summary": False,
        },
        "validation": {
            "min_shots_per_job": int(min_shots_per_job),
            "fail_policy": "skip",
        },
        "cluster_plan": {
            "split_mode": split_mode,
            "total_jobs": len(jobs),
            "eligible_jobs": eligible_jobs,
            "jobs": jobs,
            "sbatch_recommendation": {
                "time": sbatch_time,
                "mem": sbatch_mem,
                "max_concurrent": int(max_concurrent),
            },
        },
    }


def _resolve_player_ids(args: argparse.Namespace, data_root: Path) -> list[int]:
    if args.all_cached_players:
        return data_io.get_players(data_dir=data_root)
    if args.player_ids:
        return sorted(set(int(pid) for pid in args.player_ids))
    if args.player_file:
        return data_io.load_player_ids_from_text_file(args.player_file)
    raise SystemExit("Provide --player-ids, --player-file, or --all-cached-players.")


def _resolve_seasons(args: argparse.Namespace, player_ids: list[int], data_root: Path) -> list[int]:
    if args.seasons:
        return sorted(set(int(s) for s in args.seasons))
    if args.all_seasons:
        return data_io.get_all_available_seasons(player_ids, data_dir=data_root)
    raise SystemExit("Provide --seasons or --all-seasons.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MCSE cluster job config JSON")
    parser.add_argument("--data-root", type=Path, default=Path("Data/Hockey"))
    parser.add_argument(
        "--derive-from",
        type=Path,
        default=None,
        help="Derive a config from an existing JSON for --data-root (refresh job eligibility).",
    )
    player_src = parser.add_mutually_exclusive_group(required=False)
    player_src.add_argument("--player-ids", type=int, nargs="+")
    player_src.add_argument("--player-file", type=Path)
    player_src.add_argument("--all-cached-players", action="store_true")
    season_src = parser.add_mutually_exclusive_group(required=False)
    season_src.add_argument("--seasons", type=int, nargs="+")
    season_src.add_argument("--all-seasons", action="store_true")
    parser.add_argument(
        "--shot-groups",
        nargs="+",
        default=["wristshot_snapshot"],
    )
    parser.add_argument(
        "--split-mode",
        choices=["per_season", "all_selected_seasons_together"],
        default="per_season",
    )
    parser.add_argument("--min-shots-per-job", type=int, default=100)
    parser.add_argument("--num-particles", type=int, default=500)
    parser.add_argument("--generate-convergence-png", action="store_true")
    parser.add_argument("--sbatch-time", default="48:00:00")
    parser.add_argument("--sbatch-mem", default="32G")
    parser.add_argument("--max-concurrent", type=int, default=100)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--also-write-xgnew",
        action="store_true",
        help="Also write a derived config for Data/Hockey_xg_new with refreshed eligibility.",
    )
    parser.add_argument(
        "--xgnew-data-root",
        type=Path,
        default=NEW_XG_DATA_ROOT,
        help="New-xG data root used with --also-write-xgnew.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print summary only; do not write JSON")
    args = parser.parse_args()

    if args.derive_from is not None:
        if not args.output:
            raise SystemExit("--output is required with --derive-from")
        base_config = json.loads(args.derive_from.read_text(encoding="utf-8"))
        derived = derive_mcse_config_for_data_root(
            base_config,
            args.data_root,
            min_shots_per_job=args.min_shots_per_job,
        )
        cluster = derived["cluster_plan"]
        print(f"Derived from: {args.derive_from}")
        print(f"data_root: {derived['data_root']}")
        print(f"Eligible jobs: {cluster['eligible_jobs']} / {cluster['total_jobs']}")
        if args.dry_run:
            return
        if int(cluster.get("eligible_jobs", 0)) <= 0:
            raise SystemExit("Derived config has no eligible jobs.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(derived, indent=2), encoding="utf-8")
        print(f"Wrote config: {args.output}")
        return

    if not (args.player_ids or args.player_file or args.all_cached_players):
        raise SystemExit("Provide --player-ids, --player-file, or --all-cached-players.")
    if not (args.seasons or args.all_seasons):
        raise SystemExit("Provide --seasons or --all-seasons.")

    data_root = Path(args.data_root)
    player_ids = _resolve_player_ids(args, data_root)
    if not player_ids:
        raise SystemExit(f"No player IDs resolved under {data_root}")

    seasons = _resolve_seasons(args, player_ids, data_root)
    if not seasons:
        raise SystemExit("No seasons resolved.")

    config = build_mcse_cluster_config(
        data_root=data_root,
        player_ids=player_ids,
        seasons=seasons,
        shot_groups=[str(g) for g in args.shot_groups],
        split_mode=args.split_mode,
        min_shots_per_job=args.min_shots_per_job,
        num_particles=args.num_particles,
        generate_convergence_png=args.generate_convergence_png,
        sbatch_time=args.sbatch_time,
        sbatch_mem=args.sbatch_mem,
        max_concurrent=args.max_concurrent,
        rng_seed=args.rng_seed,
    )

    cluster = config["cluster_plan"]
    print(f"Players: {len(player_ids)}")
    print(f"Seasons: {seasons}")
    print(f"Split mode: {args.split_mode}")
    print(f"Total jobs: {cluster['total_jobs']}")
    print(f"Eligible jobs: {cluster['eligible_jobs']}")
    print(f"Min shots per job: {args.min_shots_per_job}")
    print(f"Particles: {args.num_particles}")

    if args.dry_run:
        if args.also_write_xgnew:
            xg_config = derive_mcse_config_for_data_root(
                config,
                args.xgnew_data_root,
                min_shots_per_job=args.min_shots_per_job,
            )
            print(
                f"New-xG eligible jobs: {xg_config['cluster_plan']['eligible_jobs']} "
                f"(data_root={args.xgnew_data_root})"
            )
        return

    if int(cluster.get("eligible_jobs", 0)) <= 0:
        raise SystemExit("No eligible jobs in legacy config; not writing output.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Wrote config: {args.output}")

    if args.also_write_xgnew:
        xg_path = derived_config_path(args.output)
        xg_config = derive_mcse_config_for_data_root(
            config,
            args.xgnew_data_root,
            min_shots_per_job=args.min_shots_per_job,
        )
        if int(xg_config["cluster_plan"].get("eligible_jobs", 0)) <= 0:
            raise SystemExit("No eligible jobs in new-xG derived config; legacy config was written.")
        xg_path.write_text(json.dumps(xg_config, indent=2), encoding="utf-8")
        print(f"Wrote new-xG config: {xg_path}")
        print(f"New-xG eligible jobs: {xg_config['cluster_plan']['eligible_jobs']}")


if __name__ == "__main__":
    main()
