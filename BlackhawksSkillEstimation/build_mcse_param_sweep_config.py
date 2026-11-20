"""Build the job config for a Blackhawks MCSE parameter sweep.

Stage 1 scans cached shots, picks one season and its highest-volume players,
and writes one job per (noise, resample fraction, player, seed). Stage 2
reads the stage-1 ranking and writes the confirmation jobs. Neither stage
runs the filter.

Examples
--------
Preview the stage-1 grid::

    python -m BlackhawksSkillEstimation.build_mcse_param_sweep_config --dry-run

Write it::

    python -m BlackhawksSkillEstimation.build_mcse_param_sweep_config \\
      --output Data/Hockey/jobs/mcse_param_sweep_stage1.json

A local wiring check (1 player, 1 seed, 2 settings, 5 shots)::

    python -m BlackhawksSkillEstimation.build_mcse_param_sweep_config --smoke --dry-run

Stage 2, after the ranking exists::

    python -m BlackhawksSkillEstimation.build_mcse_param_sweep_config \\
      --stage 2 \\
      --from-config Data/Hockey/jobs/mcse_param_sweep_stage1.json \\
      --from-ranking Data/Hockey/experiments/mcse_param_sweep/stage1/summaries/ranking.csv \\
      --output Data/Hockey/jobs/mcse_param_sweep_stage2.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from BlackhawksApp import data_io
from BlackhawksSkillEstimation.build_mcse_cluster_config import benchmark_settings_for_data_root
from BlackhawksSkillEstimation.mcse_param_sweep import (
    DEFAULT_DATA_ROOT,
    DEFAULT_EXPERIMENT_ROOT,
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_SHOT_GROUP,
    SMOKE_MAX_SHOTS,
    SMOKE_NUM_PLAYERS,
    SMOKE_SEEDS,
    STAGE1_MAX_SHOTS,
    STAGE1_NUM_PLAYERS,
    STAGE1_SEEDS,
    STAGE2_TOP_K,
    assemble_config,
    chronological_prefix,
    select_cohort,
    smoke_settings,
    stage1_settings,
    stage2_settings_from_ranking,
)
from BlackhawksSkillEstimation.player_subsample_stability import (
    discover_cached_seasons,
    load_pool_metadata,
)


def cached_shot_counts(
    data_root: Path | str,
    shot_group: str,
) -> pd.DataFrame:
    """Eligible-shot counts for every cached (player, season), metadata only."""
    rows: list[dict[str, int]] = []
    for player_id in data_io.get_players(data_root):
        for season in discover_cached_seasons(player_id, data_root):
            pool = load_pool_metadata(
                player_id,
                [season],
                data_root=data_root,
                shot_group=shot_group,
            )
            rows.append(
                {
                    "player_id": int(player_id),
                    "season": int(season),
                    "count": int(len(pool)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["player_id", "season", "count"])
    return pd.DataFrame(rows)


def _stage1_shape(
    *,
    smoke: bool,
    num_players: int | None,
    max_shots: int | None,
    seeds: Sequence[int] | None,
) -> tuple[int, int, tuple[int, ...], list[dict[str, Any]]]:
    if smoke:
        resolved_players = SMOKE_NUM_PLAYERS if num_players is None else int(num_players)
        resolved_shots = SMOKE_MAX_SHOTS if max_shots is None else int(max_shots)
        resolved_seeds = tuple(SMOKE_SEEDS if seeds is None else (int(seed) for seed in seeds))
        settings = smoke_settings()
    else:
        resolved_players = STAGE1_NUM_PLAYERS if num_players is None else int(num_players)
        resolved_shots = STAGE1_MAX_SHOTS if max_shots is None else int(max_shots)
        resolved_seeds = tuple(STAGE1_SEEDS if seeds is None else (int(seed) for seed in seeds))
        settings = stage1_settings()
    if resolved_players <= 0:
        raise ValueError("--num-players must be positive.")
    if resolved_shots <= 0:
        raise ValueError("--max-shots must be positive.")
    if not resolved_seeds:
        raise ValueError("At least one seed is required.")
    return resolved_players, resolved_shots, resolved_seeds, settings


def build_stage1_config(
    *,
    data_root: Path | str = DEFAULT_DATA_ROOT,
    experiment_root: Path | str = DEFAULT_EXPERIMENT_ROOT,
    shot_group: str = DEFAULT_SHOT_GROUP,
    season: int | None = None,
    player_ids: Sequence[int] | None = None,
    num_players: int | None = None,
    max_shots: int | None = None,
    seeds: Sequence[int] | None = None,
    smoke: bool = False,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    counts: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Stage-1 config. Pass ``counts`` to skip the parquet scan (tests)."""
    resolved_players, resolved_shots, resolved_seeds, settings = _stage1_shape(
        smoke=smoke,
        num_players=num_players,
        max_shots=max_shots,
        seeds=seeds,
    )
    data_root = Path(data_root)
    table = counts if counts is not None else cached_shot_counts(data_root, shot_group)
    chosen_season, chosen_players = select_cohort(
        table,
        max_shots=resolved_shots,
        num_players=resolved_players,
        shot_group=shot_group,
        season=season,
        player_ids=player_ids,
    )

    prefixes: dict[int, dict[str, Any]] = {}
    for player_id in chosen_players:
        pool = load_pool_metadata(
            player_id,
            [chosen_season],
            data_root=data_root,
            shot_group=shot_group,
        )
        prefixes[player_id] = {
            "event_ids": chronological_prefix(pool, resolved_shots),
            "pool_size": int(len(pool)),
        }

    benchmark_tag, benchmark_dir = benchmark_settings_for_data_root(data_root)
    run_name = "stage1_smoke" if smoke else "stage1"
    notes = (
        "MCSE parameter sweep stage 1. "
        f"{len(settings)} settings x {len(chosen_players)} players x {len(resolved_seeds)} seeds. "
        "Seeds share one chronological shot prefix per player; only the particle RNG changes."
    )
    config = assemble_config(
        stage=1,
        season=chosen_season,
        player_ids=chosen_players,
        seeds=resolved_seeds,
        shot_group=shot_group,
        max_shots=resolved_shots,
        settings=settings,
        shot_prefixes=prefixes,
        data_root=data_root,
        experiment_root=experiment_root,
        run_name=run_name,
        smoke=smoke,
        benchmark_tag=benchmark_tag,
        benchmark_dir=benchmark_dir,
        max_concurrent=max_concurrent,
        notes=notes,
    )
    config["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return config


def build_stage2_config(
    stage1_config: dict[str, Any],
    ranking_rows: Sequence[dict[str, Any]],
    *,
    top_k: int = STAGE2_TOP_K,
    max_concurrent: int | None = None,
) -> dict[str, Any]:
    """Confirmation jobs for the top stage-1 settings, on the same shot prefixes."""
    settings = stage2_settings_from_ranking(ranking_rows, top_k=top_k)
    data_root = Path(stage1_config["data_root"])
    output_root = Path(stage1_config["output_root"])
    experiment_root = output_root.parent
    maxg = stage1_config.get("maxg") or {}
    concurrent = (
        int(max_concurrent)
        if max_concurrent is not None
        else int(stage1_config["cluster_plan"]["sbatch_recommendation"]["max_concurrent"])
    )
    notes = (
        "MCSE parameter sweep stage 2. "
        "Top stage-1 settings at 250 and 1000 particles, the best setting with "
        "the NEFF gate off, and the numeric defaults on the legacy bounds. "
        "Same players, seeds, and shot prefixes as stage 1."
    )
    config = assemble_config(
        stage=2,
        season=int(stage1_config["season"]),
        player_ids=stage1_config["player_ids"],
        seeds=stage1_config["seeds"],
        shot_group=str(stage1_config["shot_group"]),
        max_shots=int(stage1_config["max_shots"]),
        settings=settings,
        shot_prefixes=stage1_config["shot_prefixes"],
        data_root=data_root,
        experiment_root=experiment_root,
        run_name="stage2",
        smoke=False,
        thresholds=stage1_config.get("thresholds"),
        benchmark_tag=maxg.get("benchmark_tag"),
        benchmark_dir=maxg.get("benchmark_dir"),
        max_concurrent=concurrent,
        notes=notes,
    )
    config["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config["source_stage1_output"] = str(output_root)
    return config


def _print_summary(config: dict[str, Any]) -> None:
    cluster = config["cluster_plan"]
    jobs = cluster["jobs"]
    groups: dict[str, int] = {}
    for job in jobs:
        groups[str(job["submit_group"])] = groups.get(str(job["submit_group"]), 0) + 1
    print(f"Experiment:  {config['experiment']} {config['run_name']} (stage {config['stage']})")
    print(f"Season:      {config['season']}")
    print(f"Players:     {config['player_ids']}")
    print(f"Shot group:  {config['shot_group']}")
    print(f"Seeds:       {config['seeds']}")
    print(f"Max shots:   {config['max_shots']}")
    print(f"Jobs:        {cluster['total_jobs']}")
    print(f"Groups:      {groups}")
    print(f"Output root: {config['output_root']}")
    print(f"sbatch:      {cluster['sbatch_recommendation']}")
    if config.get("smoke"):
        print("SMOKE CONFIG: two settings, not a result to rank against the full grid.")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an MCSE parameter-sweep job config.")
    parser.add_argument("--stage", type=int, choices=(1, 2), default=1)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--shot-group", default=DEFAULT_SHOT_GROUP)
    parser.add_argument("--season", type=int, default=None, help="Cached season. Default: the season with the most eligible players.")
    parser.add_argument("--player-ids", type=int, nargs="+", default=None)
    parser.add_argument("--num-players", type=int, default=None, help=f"Default {STAGE1_NUM_PLAYERS}, or {SMOKE_NUM_PLAYERS} with --smoke.")
    parser.add_argument("--max-shots", type=int, default=None, help=f"Chronological prefix length. Default {STAGE1_MAX_SHOTS}, or {SMOKE_MAX_SHOTS} with --smoke.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Particle RNG seeds. Default 0 1 2.")
    parser.add_argument("--smoke", action="store_true", help="1 player, 1 seed, 2 settings, 5 shots.")
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--from-config", type=Path, default=None, help="Stage-1 config. Required for --stage 2.")
    parser.add_argument("--from-ranking", type=Path, default=None, help="Stage-1 ranking CSV. Required for --stage 2.")
    parser.add_argument("--top-k", type=int, default=STAGE2_TOP_K, help="Passing settings to carry into stage 2.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.stage == 1:
        config = build_stage1_config(
            data_root=args.data_root,
            experiment_root=args.experiment_root,
            shot_group=args.shot_group,
            season=args.season,
            player_ids=args.player_ids,
            num_players=args.num_players,
            max_shots=args.max_shots,
            seeds=args.seeds,
            smoke=args.smoke,
            max_concurrent=args.max_concurrent,
        )
        default_name = "mcse_param_sweep_stage1_smoke.json" if args.smoke else "mcse_param_sweep_stage1.json"
    else:
        if args.from_config is None or args.from_ranking is None:
            raise SystemExit("--stage 2 requires --from-config and --from-ranking.")
        stage1 = _load_json(args.from_config)
        ranking = pd.read_csv(args.from_ranking)
        config = build_stage2_config(
            stage1,
            ranking.to_dict(orient="records"),
            top_k=args.top_k,
            max_concurrent=args.max_concurrent,
        )
        default_name = "mcse_param_sweep_stage2.json"

    _print_summary(config)
    if args.dry_run:
        print("\nDry run: no config written.")
        return

    output = args.output or (Path(config["data_root"]) / "jobs" / default_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote config: {output}")


if __name__ == "__main__":
    main()
