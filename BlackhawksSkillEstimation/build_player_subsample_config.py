"""Build the job config for a player subsample-stability run.

Scans the cached parquet for one player, draws every ``(N, seed)`` subset once, and
writes a JSON config holding the drawn event ids plus one job per
``(estimator, sample)`` pair. Drawing at build time (rather than inside each worker)
is what lets JEEDS and MCSE run on byte-identical shot sets.

Metadata scan only -- no estimator runs, no Snowflake.

Examples
--------
Preview job counts for the default player across every cached season:

    python -m BlackhawksSkillEstimation.build_player_subsample_config \
      --player-id 950160 --all-seasons \
      --n-shots 100 200 400 --num-seeds 50 \
      --output Data/Hockey/jobs/player_subsample_950160.json --dry-run

Write the config:

    python -m BlackhawksSkillEstimation.build_player_subsample_config \
      --player-id 950160 --all-seasons \
      --output Data/Hockey/jobs/player_subsample_950160.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from BlackhawksSkillEstimation.BlackhawksMCSE import (
    DEFAULT_LAMBDA_MODE,
    DEFAULT_MCSE_NOISE,
    DEFAULT_MCSE_RANGES,
    DEFAULT_RESAMPLE_PERCENT,
    DEFAULT_RESAMPLING_METHOD,
)
from BlackhawksSkillEstimation.player_subsample_stability import (
    BASELINE_SAMPLE_KEY,
    DEFAULT_BASE_SEED,
    DEFAULT_DATA_ROOT,
    DEFAULT_EXPERIMENT_ROOT,
    DEFAULT_N_SHOTS,
    DEFAULT_NUM_SEEDS,
    DEFAULT_PLAYER_ID,
    DEFAULT_SHOT_GROUP,
    ESTIMATORS,
    EXPERIMENT_NAME,
    PRODUCTION_NUM_EXECUTION_SKILLS,
    PRODUCTION_NUM_PARTICLES,
    PRODUCTION_NUM_PLANNING_SKILLS,
    SMOKE_NUM_EXECUTION_SKILLS,
    SMOKE_NUM_PARTICLES,
    SMOKE_NUM_PLANNING_SKILLS,
    SMOKE_NUM_SEEDS,
    build_samples,
    discover_cached_seasons,
    load_pool_metadata,
    parse_sample_key,
    season_counts,
)

DEFAULT_SBATCH_TIME = "24:00:00"
DEFAULT_SBATCH_MEM = "16G"
DEFAULT_MAX_CONCURRENT = 100


def seasons_from_cli(
    seasons: Sequence[int] | None,
    all_seasons: bool,
) -> Sequence[int] | None:
    """Resolve the mutually exclusive ``--seasons`` / ``--all-seasons`` flags.

    ``None`` means discover every season cached on disk. That is the default
    when neither flag is passed, and also what ``--all-seasons`` requests.
    """
    if seasons is not None and all_seasons:
        raise ValueError("Pass --seasons or --all-seasons, not both.")
    if all_seasons:
        return None
    return seasons


def build_estimator_settings(
    estimators: Sequence[str],
    *,
    smoke: bool,
    num_execution_skills: int | None,
    num_planning_skills: int | None,
    num_particles: int | None,
    rng_seed: int,
    save_intermediate_csv: bool,
) -> dict[str, dict[str, Any]]:
    """Resolve per-estimator settings, defaulting to the production grids.

    The production grids are what the existing per-season runs used. Anything smaller
    produces estimates that cannot be plotted against those season dots.
    """
    jeeds_exec = num_execution_skills or (
        SMOKE_NUM_EXECUTION_SKILLS if smoke else PRODUCTION_NUM_EXECUTION_SKILLS
    )
    jeeds_planning = num_planning_skills or (
        SMOKE_NUM_PLANNING_SKILLS if smoke else PRODUCTION_NUM_PLANNING_SKILLS
    )
    mcse_particles = num_particles or (
        SMOKE_NUM_PARTICLES if smoke else PRODUCTION_NUM_PARTICLES
    )

    settings: dict[str, dict[str, Any]] = {}
    if "jeeds" in estimators:
        settings["jeeds"] = {
            "num_execution_skills": int(jeeds_exec),
            "num_planning_skills": int(jeeds_planning),
            "rng_seed": int(rng_seed),
            "save_intermediate_csv": bool(save_intermediate_csv),
        }
    if "mcse" in estimators:
        settings["mcse"] = {
            "num_particles": int(mcse_particles),
            "noise": list(DEFAULT_MCSE_NOISE),
            "resample_percent": float(DEFAULT_RESAMPLE_PERCENT),
            "resample_neff": True,
            "resampling_method": DEFAULT_RESAMPLING_METHOD,
            "lambda_mode": DEFAULT_LAMBDA_MODE,
            # Current EV-normalized bounds. Estimates made under the older end[3]=4.0
            # lambda bound are not comparable to these.
            "ranges": {key: list(values) for key, values in DEFAULT_MCSE_RANGES.items()},
            "rng_seed": int(rng_seed),
            "save_intermediate_csv": bool(save_intermediate_csv),
        }
    return settings


def build_jobs(
    samples: dict[str, dict[str, Any]],
    estimators: Sequence[str],
) -> list[dict[str, Any]]:
    """One job per (estimator, sample). Baseline first, then N ascending."""

    def sort_key(sample_key: str) -> tuple[int, int, int]:
        if sample_key == BASELINE_SAMPLE_KEY:
            return (0, 0, 0)
        n_requested, seed = parse_sample_key(sample_key)
        return (1, int(n_requested or 0), int(seed or 0))

    jobs: list[dict[str, Any]] = []
    for sample_key in sorted(samples, key=sort_key):
        sample = samples[sample_key]
        for estimator in estimators:
            jobs.append(
                {
                    "estimator": estimator,
                    "sample_key": sample_key,
                    "is_baseline": sample_key == BASELINE_SAMPLE_KEY,
                    "n_requested": sample["n_requested"],
                    "seed": sample["seed"],
                    "n_shots": sample["n_shots"],
                    "eligible": True,
                }
            )
    return jobs


def build_config(
    *,
    player_id: int = DEFAULT_PLAYER_ID,
    seasons: Sequence[int] | None = None,
    data_root: Path | str = DEFAULT_DATA_ROOT,
    experiment_root: Path | str = DEFAULT_EXPERIMENT_ROOT,
    run_name: str | None = None,
    shot_group: str = DEFAULT_SHOT_GROUP,
    n_shots: Sequence[int] = DEFAULT_N_SHOTS,
    num_seeds: int | None = None,
    base_seed: int = DEFAULT_BASE_SEED,
    estimators: Sequence[str] = ESTIMATORS,
    smoke: bool = False,
    num_execution_skills: int | None = None,
    num_planning_skills: int | None = None,
    num_particles: int | None = None,
    rng_seed: int = 0,
    save_intermediate_csv: bool = True,
    sbatch_time: str = DEFAULT_SBATCH_TIME,
    sbatch_mem: str = DEFAULT_SBATCH_MEM,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
) -> dict[str, Any]:
    data_root = Path(data_root)

    resolved_seasons = sorted({int(s) for s in seasons}) if seasons else discover_cached_seasons(
        player_id, data_root
    )
    if not resolved_seasons:
        raise SystemExit(
            f"No cached seasons found for player {player_id} under {data_root}. "
            "Download shots parquet + shot_maps npz first."
        )

    pool_df = load_pool_metadata(
        player_id,
        resolved_seasons,
        data_root=data_root,
        shot_group=shot_group,
    )
    if pool_df.empty:
        raise SystemExit(
            f"Shot pool for player {player_id} is empty after the '{shot_group}' filter."
        )

    unknown = sorted(set(estimators) - set(ESTIMATORS))
    if unknown:
        raise SystemExit(f"Unknown estimator(s): {', '.join(unknown)}")

    # --smoke only supplies defaults, so an explicit --num-seeds still wins.
    if num_seeds is not None:
        effective_num_seeds = int(num_seeds)
    else:
        effective_num_seeds = SMOKE_NUM_SEEDS if smoke else DEFAULT_NUM_SEEDS
    samples, skipped_n = build_samples(
        pool_df,
        n_shots=n_shots,
        num_seeds=effective_num_seeds,
        base_seed=base_seed,
    )
    for n in skipped_n:
        print(f"  Skipping N={n}: larger than the {len(pool_df)}-shot pool.")

    jobs = build_jobs(samples, estimators)
    run_name = run_name or f"player_{player_id}"

    return {
        "config_version": 1,
        "experiment": EXPERIMENT_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_name": run_name,
        "data_root": str(data_root),
        "output_root": str(Path(experiment_root) / run_name),
        "player_id": int(player_id),
        "shot_group": shot_group,
        "seasons": resolved_seasons,
        "sampling": {
            "n_shots": sorted({int(x) for x in n_shots}),
            "skipped_n_shots": skipped_n,
            "num_seeds": effective_num_seeds,
            "base_seed": int(base_seed),
            "pool_size": int(len(pool_df)),
            "pool_season_counts": season_counts(pool_df),
        },
        "estimators": build_estimator_settings(
            estimators,
            smoke=smoke,
            num_execution_skills=num_execution_skills,
            num_planning_skills=num_planning_skills,
            num_particles=num_particles,
            rng_seed=rng_seed,
            save_intermediate_csv=save_intermediate_csv,
        ),
        "smoke": bool(smoke),
        "samples": samples,
        "cluster_plan": {
            "jobs": jobs,
            "total_jobs": len(jobs),
            "eligible_jobs": sum(1 for job in jobs if job["eligible"]),
            "sbatch_recommendation": {
                "time": sbatch_time,
                "mem": sbatch_mem,
                "max_concurrent": int(max_concurrent),
            },
        },
    }


def _print_summary(config: dict[str, Any]) -> None:
    sampling = config["sampling"]
    cluster = config["cluster_plan"]

    print(f"Experiment:   {config['experiment']} ({config['run_name']})")
    print(f"Player:       {config['player_id']}")
    print(f"Shot group:   {config['shot_group']}")
    print(f"Seasons:      {config['seasons']}")
    print(f"Pool size:    {sampling['pool_size']} shots")
    for season, count in sampling["pool_season_counts"].items():
        print(f"  {season}: {count}")
    print(f"N values:     {sampling['n_shots']} x {sampling['num_seeds']} seeds")
    if sampling["skipped_n_shots"]:
        print(f"  skipped (larger than pool): {sampling['skipped_n_shots']}")
    print(f"Estimators:   {', '.join(sorted(config['estimators']))}")
    for name, settings in sorted(config["estimators"].items()):
        detail = ", ".join(f"{k}={v}" for k, v in settings.items() if k != "ranges")
        print(f"  {name}: {detail}")
    print(f"Samples:      {len(config['samples'])}")
    print(f"Jobs:         {cluster['total_jobs']} ({cluster['eligible_jobs']} eligible)")
    print(f"Output root:  {config['output_root']}")
    print(f"sbatch:       {cluster['sbatch_recommendation']}")
    if config["smoke"]:
        print("SMOKE CONFIG: grids are reduced; do not compare these to season estimates.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a player subsample-stability job config from cached shot data.",
    )
    parser.add_argument("--player-id", type=int, default=DEFAULT_PLAYER_ID)
    season_src = parser.add_mutually_exclusive_group()
    season_src.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        help="Explicit seasons, e.g. 20212022 20222023 20232024 20242025 20252026",
    )
    season_src.add_argument(
        "--all-seasons",
        action="store_true",
        help="Use every season cached on disk. This is also the default when "
        "--seasons is omitted.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output subdirectory name (default: player_<id>).",
    )
    parser.add_argument("--shot-group", default=DEFAULT_SHOT_GROUP)
    parser.add_argument("--n-shots", type=int, nargs="+", default=list(DEFAULT_N_SHOTS))
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help=f"Draws per N (default: {DEFAULT_NUM_SEEDS}, or {SMOKE_NUM_SEEDS} with --smoke).",
    )
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--estimators", nargs="+", default=list(ESTIMATORS), choices=list(ESTIMATORS))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Reduced grids and 3 seeds for a local check. Not comparable to season runs.",
    )
    parser.add_argument("--num-execution-skills", type=int, default=None)
    parser.add_argument("--num-planning-skills", type=int, default=None)
    parser.add_argument("--num-particles", type=int, default=None)
    parser.add_argument("--rng-seed", type=int, default=0, help="Estimator RNG seed (not the sampling seed).")
    parser.add_argument(
        "--no-intermediate-csv",
        action="store_true",
        help="Skip per-shot convergence traces.",
    )
    parser.add_argument("--sbatch-time", default=DEFAULT_SBATCH_TIME)
    parser.add_argument("--sbatch-mem", default=DEFAULT_SBATCH_MEM)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--output", type=Path, default=None, help="Config JSON output path.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without writing.")

    args = parser.parse_args()

    config = build_config(
        player_id=args.player_id,
        seasons=seasons_from_cli(args.seasons, args.all_seasons),
        data_root=args.data_root,
        experiment_root=args.experiment_root,
        run_name=args.run_name,
        shot_group=args.shot_group,
        n_shots=args.n_shots,
        num_seeds=args.num_seeds,
        base_seed=args.base_seed,
        estimators=args.estimators,
        smoke=args.smoke,
        num_execution_skills=args.num_execution_skills,
        num_planning_skills=args.num_planning_skills,
        num_particles=args.num_particles,
        rng_seed=args.rng_seed,
        save_intermediate_csv=not args.no_intermediate_csv,
        sbatch_time=args.sbatch_time,
        sbatch_mem=args.sbatch_mem,
        max_concurrent=args.max_concurrent,
    )

    _print_summary(config)

    if args.dry_run:
        print("\nDry run: no config written.")
        return

    output = args.output or (
        Path(args.data_root) / "jobs" / f"player_subsample_{args.player_id}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote config: {output}")


if __name__ == "__main__":
    main()
