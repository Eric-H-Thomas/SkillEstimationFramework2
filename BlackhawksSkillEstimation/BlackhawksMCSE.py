"""Run MCSE (particle-filter QRE) on Blackhawks shot data.

MCSE outputs a 2D execution-skill profile (direction y, elevation z) plus rho.
Intermediate CSV/plots are for convergence visualization; MAXG is the primary
scalar for reporting and comparison with JEEDS (see ``maxg_evaluator``).
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from Environments.Hockey import getAngularHeatmapsPerPlayer as angular_heatmaps
from Estimators.joint_pfe import QREMethod_Multi_Particles

from .BlackhawksJEEDS import (
    DEFAULT_SHOT_GROUPS,
    MIN_DISTANCE_FROM_NET_FT,
    SHOT_TYPE_GROUPS,
    _NET_CENTER,
    _compute_aggregate_season_tag,
    _get_game_shot_maps_cached,
    _infer_grid_axes_from_value_map,
    ensure_player_directories,
    load_player_data,
    query_player_game_info,
    query_player_season_shots,
)
from .BlackhawksPFESpaces import BlackhawksPFESpaces
from .plot_intermediate_estimates_mcse import plot_intermediate_estimates_mcse


DEFAULT_NUM_PARTICLES = 1000
DEFAULT_MCSE_NOISE = [200, 200]
DEFAULT_RESAMPLE_PERCENT = 0.90
DEFAULT_RESAMPLING_METHOD = "systematic"
# Match JEEDS Blackhawks grids: execution skill in [0.004, 0.25] rad per axis;
# rationality lambda on log10 scale in [0, 4] (same as JEEDS logspace(0, 4)).
#
# Historical MCSE defaults (pre-2026-07 bounds alignment):
#   execution skill per axis: [0.004, pi/4]  (~0.785 rad upper bound)
#   rationality log10 lambda: [-3, 1.6]      (~0.001 to ~40 on the particle grid)
# The smoke run on 950160/950184 (200 particles, legacy Data/Hockey) used those
# older bounds. joint_pfe.py previously hardcoded the lambda grid as linspace(-3, 1.6).
DEFAULT_EXECUTION_SKILL_MAX = 0.25
DEFAULT_RATIONALITY_LOG10_MIN = 0.0
DEFAULT_RATIONALITY_LOG10_MAX = 4.0
# Prior execution upper bound (radians); kept for reference / sensitivity reruns.
LEGACY_EXECUTION_SKILL_MAX = float(np.pi / 4)
# Prior rationality log10 endpoints; kept for reference / sensitivity reruns.
LEGACY_RATIONALITY_LOG10_MIN = -3.0
LEGACY_RATIONALITY_LOG10_MAX = 1.6
LEGACY_MCSE_RANGES = {
    "start": [0.004, 0.004, -0.75, LEGACY_RATIONALITY_LOG10_MIN],
    "end": [LEGACY_EXECUTION_SKILL_MAX, LEGACY_EXECUTION_SKILL_MAX, 0.75, LEGACY_RATIONALITY_LOG10_MAX],
}
DEFAULT_MCSE_RANGES = {
    "start": [0.004, 0.004, -0.75, DEFAULT_RATIONALITY_LOG10_MIN],
    "end": [DEFAULT_EXECUTION_SKILL_MAX, DEFAULT_EXECUTION_SKILL_MAX, 0.75, DEFAULT_RATIONALITY_LOG10_MAX],
}


@dataclass
class _MCSEEnv:
    domain_name: str = "hockey-multi"
    resultsFolder: str = ""
    seedNum: int = 0


@dataclass
class MCSEInputs:
    spaces_per_shot: list[BlackhawksPFESpaces]
    actions: list[list[float]]
    info_rows: list[dict[str, object]]
    skipped_proximity: int = 0


def ensure_mcse_directories(player_dir: Path) -> None:
    ensure_player_directories(player_dir)
    (player_dir / "logs" / "mcse").mkdir(parents=True, exist_ok=True)
    (player_dir / "times" / "mcse" / "estimators").mkdir(parents=True, exist_ok=True)


def transform_shots_for_mcse(
    df: pd.DataFrame,
    shot_maps: dict[int, dict[str, object]],
) -> MCSEInputs:
    """Convert shot rows into per-shot PFE spaces and angular utility grids."""
    df = df.rename(columns=str.lower)
    skipped_proximity = 0
    info_rows: list[dict[str, object]] = []
    actions: list[list[float]] = []
    spaces_per_shot: list[BlackhawksPFESpaces] = []

    for _, row in df.iterrows():
        event_id = int(row["event_id"])
        if event_id not in shot_maps:
            continue

        player_location = np.array([float(row["start_x"]), float(row["start_y"])])
        executed_action = np.array([float(row["location_y"]), float(row["location_z"])])

        dist_to_net = np.linalg.norm(player_location - _NET_CENTER)
        if dist_to_net < MIN_DISTANCE_FROM_NET_FT:
            skipped_proximity += 1
            continue

        shot_map_data = shot_maps[event_id]
        base_ev = shot_map_data["value_map"]
        grid_y, grid_z = _infer_grid_axes_from_value_map(base_ev)

        angular_out = angular_heatmaps.getAngularHeatmap(
            base_ev,
            player_location,
            executed_action,
            grid_y=grid_y,
            grid_z=grid_z,
        )
        dirs = np.array(angular_out[0])
        elevations = np.array(angular_out[1])
        grid_targets_angular = angular_out[3]
        grid_utilities_computed = angular_out[7]
        executed_action_angular = angular_out[8]
        skip = bool(angular_out[9])
        if skip:
            continue

        spaces = BlackhawksPFESpaces(dirs, elevations, grid_targets_angular)
        info_rows.append({"Zs": grid_utilities_computed})
        actions.append([float(executed_action_angular[0]), float(executed_action_angular[1])])
        spaces_per_shot.append(spaces)

    if skipped_proximity:
        print(f"  Filtered {skipped_proximity} shot(s) within {MIN_DISTANCE_FROM_NET_FT}ft of the net.")

    return MCSEInputs(
        spaces_per_shot=spaces_per_shot,
        actions=actions,
        info_rows=info_rows,
        skipped_proximity=skipped_proximity,
    )


def save_intermediate_estimates_csv_mcse(
    skill_log: list[dict[str, object]],
    player_id: int,
    output_dir: Path | str,
    tag: str = "",
    shot_group: str = "",
) -> Path:
    output_dir = Path(output_dir)
    logs_dir = output_dir / "logs" / "mcse"
    if shot_group:
        logs_dir = logs_dir / shot_group
    logs_dir.mkdir(parents=True, exist_ok=True)

    tag_str = f"_{tag}" if tag else ""
    csv_path = logs_dir / f"intermediate_estimates{tag_str}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "shot_count",
            "ees_y",
            "ees_z",
            "map_execution_skill_y",
            "map_execution_skill_z",
            "rho_ees",
            "map_rho",
            "expected_rationality",
            "map_rationality",
            "log10_expected_rationality",
            "log10_map_rationality",
        ])
        for row in skill_log:
            eps_val = row["eps"]
            map_rat_val = row["map_rationality"]
            writer.writerow([
                row["shot_count"],
                row["ees_y"],
                row["ees_z"],
                row["map_execution_skill_y"],
                row["map_execution_skill_z"],
                row["rho_ees"],
                row["map_rho"],
                eps_val,
                map_rat_val,
                np.log10(eps_val) if eps_val and eps_val > 0 else None,
                np.log10(map_rat_val) if map_rat_val and map_rat_val > 0 else None,
            ])
    return csv_path


def _filter_df_by_shot_group(df: pd.DataFrame, shot_group: str) -> pd.DataFrame:
    if shot_group:
        if shot_group not in SHOT_TYPE_GROUPS:
            raise ValueError(
                f"Unknown shot_group '{shot_group}'. "
                f"Valid groups: {', '.join(DEFAULT_SHOT_GROUPS)}"
            )
        _, allowed_types, include_null = SHOT_TYPE_GROUPS[shot_group]
    else:
        _, allowed_types, include_null = "Wristshot/Snapshot", {"wristshot", "snapshot"}, True

    df_lc = df.rename(columns=str.lower)
    if "shot_type" not in df_lc.columns:
        return df_lc

    shot_series = df_lc["shot_type"].where(pd.notna(df_lc["shot_type"]))
    shot_lower = shot_series.astype(str).str.lower()
    if include_null:
        mask = shot_series.isna() | shot_lower.isin(allowed_types)
    else:
        mask = shot_lower.isin(allowed_types)
    return df_lc[mask]


def _build_mcse_estimator(
    player_dir: Path,
    *,
    num_particles: int,
    noise: Sequence[float],
    resample_percent: float,
    resample_neff: bool,
    resampling_method: str,
    ranges: dict[str, list[float]],
    rng_seed: int | None,
) -> QREMethod_Multi_Particles:
    env = _MCSEEnv(
        domain_name="hockey-multi",
        resultsFolder=str(player_dir),
        seedNum=int(rng_seed or 0),
    )
    return QREMethod_Multi_Particles(
        env,
        int(num_particles),
        list(noise),
        float(resample_percent),
        bool(resample_neff),
        resampling_method,
        ranges,
        otherArgs={"verbose": False, "retain_history": False},
    )


def _extract_skill_vectors_from_estimator(estimator: QREMethod_Multi_Particles) -> dict[str, object]:
    """Read latest MAP/EES estimates without materializing particle history."""
    map_name = estimator.names[0]
    ees_name = estimator.names[1]
    map_skills = estimator.estimatesXskills.get(map_name, [])
    ees_skills = estimator.estimatesXskills.get(ees_name, [])
    map_rhos = estimator.estimatesRhos.get(map_name, [])
    ees_rhos = estimator.estimatesRhos.get(ees_name, [])
    map_pskills = estimator.estimatesPskills.get(map_name, [])
    ees_pskills = estimator.estimatesPskills.get(ees_name, [])

    if not map_skills or not ees_skills:
        return {}

    map_vec = map_skills[-1]
    ees_vec = ees_skills[-1]
    return {
        "execution_skill_y": float(map_vec[0]),
        "execution_skill_z": float(map_vec[1]),
        "ees_y": float(ees_vec[0]),
        "ees_z": float(ees_vec[1]),
        "rho_map": float(map_rhos[-1]) if map_rhos else None,
        "rho_ees": float(ees_rhos[-1]) if ees_rhos else None,
        "rationality": float(map_pskills[-1]) if map_pskills else None,
        "eps": float(ees_pskills[-1]) if ees_pskills else None,
        "estimator_name": map_name,
    }


def _extract_skill_vectors(results: dict, estimator: QREMethod_Multi_Particles) -> dict[str, object]:
    """Backward-compatible wrapper; prefer ``_extract_skill_vectors_from_estimator``."""
    del results  # unused; kept for call-site compatibility
    return _extract_skill_vectors_from_estimator(estimator)


def _run_mcse_estimation(
    df: pd.DataFrame,
    game_ids: Sequence[int],
    player_id: int,
    player_dir: Path,
    *,
    num_particles: int = DEFAULT_NUM_PARTICLES,
    noise: Sequence[float] | None = None,
    resample_percent: float = DEFAULT_RESAMPLE_PERCENT,
    resample_neff: bool = True,
    resampling_method: str = DEFAULT_RESAMPLING_METHOD,
    ranges: dict[str, list[float]] | None = None,
    rng_seed: int | None = 0,
    return_intermediate_estimates: bool = False,
    preloaded_shot_maps: dict[int, dict[str, object]] | None = None,
    save_intermediate_csv: bool = False,
    csv_tag: str = "",
    shot_group: str = "",
    compute_maxg: bool = False,
    benchmark_tag: str | None = None,
    benchmark_dir: Path | str = Path("Data/Hockey/benchmarks"),
) -> dict[str, object]:
    if df.empty:
        return {
            "status": "no_data",
            "num_shots": 0,
            "warning": "No shot data for this estimation.",
        }

    if preloaded_shot_maps is not None:
        shot_maps = preloaded_shot_maps
    else:
        shot_maps = {}
        for game_id in game_ids:
            try:
                shot_maps.update(_get_game_shot_maps_cached(int(game_id)))
            except Exception as exc:
                print(f"Warning: Could not fetch shot maps for game {game_id}: {exc}")

    filtered_df = _filter_df_by_shot_group(df, shot_group)
    mcse_inputs = transform_shots_for_mcse(filtered_df, shot_maps=shot_maps)
    if not mcse_inputs.actions:
        return {
            "status": "no_usable_shots",
            "num_shots": 0,
            "warning": "No usable shots after angular conversion.",
            "skipped_proximity": mcse_inputs.skipped_proximity,
        }

    noise = list(noise or DEFAULT_MCSE_NOISE)
    ranges = ranges or {k: list(v) for k, v in DEFAULT_MCSE_RANGES.items()}
    ensure_mcse_directories(player_dir)

    estimator = _build_mcse_estimator(
        player_dir,
        num_particles=num_particles,
        noise=noise,
        resample_percent=resample_percent,
        resample_neff=resample_neff,
        resampling_method=resampling_method,
        ranges=ranges,
        rng_seed=rng_seed,
    )

    rng = np.random.default_rng(rng_seed)
    tag = f"Player_{player_id}"
    skill_log: list[dict[str, object]] = []

    for idx, (spaces, info, action) in enumerate(
        zip(mcse_inputs.spaces_per_shot, mcse_inputs.info_rows, mcse_inputs.actions)
    ):
        estimator.add_observation(
            rng,
            spaces,
            state=None,
            action=action,
            resultsFolder=str(player_dir),
            tag=tag,
            Zs=info["Zs"],
            s=str(idx),
            i=idx,
        )
        # Drop per-shot PDF/EV caches; they are not reused across shots.
        spaces.clear_particle_caches()

        if return_intermediate_estimates or save_intermediate_csv:
            partial = _extract_skill_vectors_from_estimator(estimator)
            if partial:
                skill_log.append({
                    "shot_count": idx + 1,
                    "ees_y": partial["ees_y"],
                    "ees_z": partial["ees_z"],
                    "map_execution_skill_y": partial["execution_skill_y"],
                    "map_execution_skill_z": partial["execution_skill_z"],
                    "rho_ees": partial["rho_ees"],
                    "map_rho": partial["rho_map"],
                    "eps": partial["eps"],
                    "map_rationality": partial["rationality"],
                    "log10_eps": (
                        np.log10(partial["eps"]) if partial.get("eps") and partial["eps"] > 0 else None
                    ),
                    "log10_map_rationality": (
                        np.log10(partial["rationality"])
                        if partial.get("rationality") and partial["rationality"] > 0
                        else None
                    ),
                })

    final = _extract_skill_vectors_from_estimator(estimator)
    if not final:
        return {
            "status": "estimation_failed",
            "num_shots": len(mcse_inputs.actions),
            "warning": "MCSE returned no estimates.",
        }

    result: dict[str, object] = {
        **final,
        "log10_rationality": (
            np.log10(final["rationality"]) if final.get("rationality") and final["rationality"] > 0 else None
        ),
        "log10_eps": np.log10(final["eps"]) if final.get("eps") and final["eps"] > 0 else None,
        "num_shots": len(mcse_inputs.actions),
        "num_particles": int(num_particles),
        "status": "success",
        "maxg_ees": None,
        "maxg_map": None,
    }

    if return_intermediate_estimates or save_intermediate_csv:
        result["skill_log"] = skill_log

    if save_intermediate_csv and skill_log:
        csv_path = save_intermediate_estimates_csv_mcse(
            skill_log=skill_log,
            player_id=player_id,
            output_dir=player_dir,
            tag=csv_tag,
            shot_group=shot_group,
        )
        result["csv_path"] = str(csv_path)

    if shot_group:
        result["shot_group"] = shot_group

    if compute_maxg and benchmark_tag:
        from .maxg_evaluator import compute_maxg_for_mcse_profile

        maxg_ees = compute_maxg_for_mcse_profile(
            benchmark_dir=Path(benchmark_dir),
            benchmark_tag=benchmark_tag,
            x_y=float(final["ees_y"]),
            x_z=float(final["ees_z"]),
            rho=float(final["rho_ees"] or 0.0),
        )
        result["maxg_ees"] = maxg_ees
        if final.get("execution_skill_y") is not None:
            result["maxg_map"] = compute_maxg_for_mcse_profile(
                benchmark_dir=Path(benchmark_dir),
                benchmark_tag=benchmark_tag,
                x_y=float(final["execution_skill_y"]),
                x_z=float(final["execution_skill_z"]),
                rho=float(final["rho_map"] or 0.0),
            )

    return result


def estimate_player_skill(
    player_id: int,
    game_ids: Sequence[int] | None = None,
    *,
    seasons: Sequence[int] | None = None,
    per_season: bool = True,
    num_particles: int = DEFAULT_NUM_PARTICLES,
    noise: Sequence[float] | None = None,
    resample_percent: float = DEFAULT_RESAMPLE_PERCENT,
    resample_neff: bool = True,
    resampling_method: str = DEFAULT_RESAMPLING_METHOD,
    ranges: dict[str, list[float]] | None = None,
    rng_seed: int | None = 0,
    return_intermediate_estimates: bool = False,
    save_intermediate_csv: bool = False,
    data_dir: Path | str = Path("Data/Hockey"),
    player_dir_name: str | None = None,
    confirm: bool = True,
    offline_data: tuple[pd.DataFrame, dict[int, dict[str, object]]] | None = None,
    shot_group: str = "",
    shot_groups: Sequence[str] | None = None,
    compute_maxg: bool = False,
    benchmark_tag: str | None = None,
    benchmark_dir: Path | str = Path("Data/Hockey/benchmarks"),
) -> dict[str, object]:
    """Estimate MCSE skills for a player (offline, game, or season modes)."""
    if shot_groups is not None:
        per_group: dict[str, dict[str, object]] = {}
        for grp in shot_groups:
            per_group[grp] = estimate_player_skill(
                player_id=player_id,
                game_ids=game_ids,
                seasons=seasons,
                per_season=per_season,
                num_particles=num_particles,
                noise=noise,
                resample_percent=resample_percent,
                resample_neff=resample_neff,
                resampling_method=resampling_method,
                ranges=ranges,
                rng_seed=rng_seed,
                return_intermediate_estimates=return_intermediate_estimates,
                save_intermediate_csv=save_intermediate_csv,
                data_dir=data_dir,
                player_dir_name=player_dir_name,
                confirm=False,
                offline_data=offline_data,
                shot_group=grp,
                compute_maxg=compute_maxg,
                benchmark_tag=benchmark_tag,
                benchmark_dir=benchmark_dir,
            )
        return {"player_id": player_id, "shot_groups": list(shot_groups), "per_group_results": per_group}

    if game_ids is None and seasons is None and offline_data is None:
        raise ValueError("Must provide 'game_ids', 'seasons', or 'offline_data'.")

    data_dir = Path(data_dir)
    player_dir = data_dir / "players" / (player_dir_name or f"player_{player_id}")

    mcse_kwargs = dict(
        player_id=player_id,
        player_dir=player_dir,
        num_particles=num_particles,
        noise=noise,
        resample_percent=resample_percent,
        resample_neff=resample_neff,
        resampling_method=resampling_method,
        ranges=ranges,
        rng_seed=rng_seed,
        return_intermediate_estimates=return_intermediate_estimates,
        save_intermediate_csv=save_intermediate_csv,
        shot_group=shot_group,
        compute_maxg=compute_maxg,
        benchmark_tag=benchmark_tag,
        benchmark_dir=benchmark_dir,
    )

    if offline_data is not None:
        df, shot_maps = offline_data
        if confirm and not df.empty:
            print(f"\n{len(df)} shots loaded from offline data for player {player_id}.")
            if input("Proceed with MCSE estimation? [y/n]: ").strip().lower() != "y":
                return {"status": "cancelled", "num_shots": len(df)}

        if per_season and "season" in df.columns:
            per_season_results: dict[int, dict[str, object]] = {}
            for season in sorted(df["season"].unique()):
                season_df = df[df["season"] == season]
                season_result = _run_mcse_estimation(
                    df=season_df,
                    game_ids=season_df["game_id"].unique().tolist(),
                    preloaded_shot_maps=shot_maps,
                    csv_tag=str(season),
                    **mcse_kwargs,
                )
                season_result["season"] = int(season)
                per_season_results[int(season)] = season_result
            return {
                "player_id": player_id,
                "seasons": sorted(df["season"].unique().tolist()),
                "per_season_results": per_season_results,
            }

        seasons_in_data = sorted(df["season"].unique().tolist()) if "season" in df.columns else []
        aggregate_tag = _compute_aggregate_season_tag(seasons_in_data) if seasons_in_data else "aggregate"
        return _run_mcse_estimation(
            df=df,
            game_ids=df["game_id"].unique().tolist(),
            preloaded_shot_maps=shot_maps,
            csv_tag=aggregate_tag,
            **mcse_kwargs,
        )

    if game_ids is not None:
        df = query_player_game_info(player_id=player_id, game_ids=list(game_ids))
        return _run_mcse_estimation(df=df, game_ids=game_ids, csv_tag="games", **mcse_kwargs)

    seasons_list = list(seasons or [])
    df = query_player_season_shots(player_id=player_id, seasons=seasons_list)
    all_game_ids = df["game_id"].unique().tolist()

    if not per_season:
        aggregate_tag = _compute_aggregate_season_tag(sorted(df["season"].unique().tolist()))
        return _run_mcse_estimation(
            df=df,
            game_ids=all_game_ids,
            csv_tag=aggregate_tag,
            **mcse_kwargs,
        )

    per_season_results = {}
    for season in sorted(df["season"].unique()):
        season_df = df[df["season"] == season]
        season_result = _run_mcse_estimation(
            df=season_df,
            game_ids=season_df["game_id"].unique().tolist(),
            csv_tag=str(season),
            **mcse_kwargs,
        )
        season_result["season"] = int(season)
        per_season_results[int(season)] = season_result

    return {"player_id": player_id, "seasons": seasons_list, "per_season_results": per_season_results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCSE on Blackhawks player shot data")
    parser.add_argument("player_id", type=int)
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    parser.add_argument("--shot-group", default="wristshot_snapshot")
    parser.add_argument("--data-dir", type=Path, default=Path("Data/Hockey"))
    parser.add_argument("--num-particles", type=int, default=DEFAULT_NUM_PARTICLES)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--save-intermediate-csv", action="store_true")
    parser.add_argument("--compute-maxg", action="store_true")
    parser.add_argument("--benchmark-tag", default=None)
    args = parser.parse_args()

    if args.seasons:
        df, shot_maps = load_player_data(args.player_id, args.seasons, data_dir=args.data_dir)
        result = estimate_player_skill(
            args.player_id,
            seasons=args.seasons,
            offline_data=(df, shot_maps),
            shot_group=args.shot_group,
            confirm=False,
            save_intermediate_csv=args.save_intermediate_csv,
            num_particles=args.num_particles,
            rng_seed=args.rng_seed,
            compute_maxg=args.compute_maxg,
            benchmark_tag=args.benchmark_tag,
            data_dir=args.data_dir,
        )
    else:
        raise SystemExit("Provide --seasons with offline cached data for CLI usage.")

    print(result)


if __name__ == "__main__":
    main()
