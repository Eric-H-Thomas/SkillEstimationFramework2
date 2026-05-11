"""Phase 3 (xG migration): run JEEDS A/B estimation for overlap-heavy players.

A/B definition
--------------
A (legacy): hawks_analytics.post_shot_xg_value_maps
B (new):    hawks_analytics.expected_goal_values_post_shot_net_grid (EXPECTED_GOALS)

This script:
- selects 10 overlap-heavy forwards (shots present in BOTH map tables)
- downloads offline data for each source into separate roots
- runs identical JEEDS estimation offline for each source
- writes a single summary CSV + JSON under Data/Hockey/migration_reports/

Scope: estimation only (no app/streamlit changes).

Usage
-----
python -m BlackhawksSkillEstimation.run_phase3_ab_estimation
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from BlackhawksSkillEstimation.BlackhawksJEEDS import (
    DEFAULT_NUM_EXECUTION_SKILLS,
    DEFAULT_NUM_PLANNING_SKILLS,
    load_player_data,
    save_player_data,
    estimate_player_skill,
)
from BlackhawksSkillEstimation.phase3_select_ab_players import select_top_overlap_players


DEFAULT_SEASONS: list[int] = [20212022, 20222023, 20232024, 20242025]
DEFAULT_LEGACY_ROOT = Path("Data/Hockey_xg_legacy")
DEFAULT_NEW_ROOT = Path("Data/Hockey_xg_new")
DEFAULT_REPORT_DIR = Path("Data/Hockey/migration_reports")


def _available_seasons_for_player(player_id: int, seasons: list[int], root: Path) -> list[int]:
    data_dir = root / "players" / f"player_{player_id}" / "data"
    available: list[int] = []
    for season in seasons:
        if (data_dir / f"shots_{season}.parquet").exists() and (data_dir / f"shot_maps_{season}.npz").exists():
            available.append(int(season))
    return available


def _filter_to_wristshot_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df_lc = df.rename(columns=str.lower)
    if "shot_type" not in df_lc.columns:
        return df_lc
    shot = df_lc["shot_type"].where(pd.notna(df_lc["shot_type"]))
    shot_lower = shot.astype(str).str.lower()
    mask = shot_lower.isin({"wristshot", "snapshot"})
    return df_lc[mask].reset_index(drop=True)


def _prune_shot_maps(shot_maps: dict[int, dict[str, object]], event_ids: pd.Series) -> dict[int, dict[str, object]]:
    keep = {int(eid) for eid in event_ids.dropna().astype(int).tolist()}
    return {int(eid): payload for eid, payload in shot_maps.items() if int(eid) in keep}


def _flatten_per_season_results(
    *,
    player_id: int,
    source: str,
    result: dict[str, Any],
    seasons_requested: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if "per_season_results" in result and isinstance(result["per_season_results"], dict):
        for season, season_result in result["per_season_results"].items():
            rows.append(
                {
                    "player_id": int(player_id),
                    "source": source,
                    "season": int(season),
                    "execution_skill": season_result.get("execution_skill"),
                    "rationality": season_result.get("rationality"),
                    "ees": season_result.get("ees"),
                    "eps": season_result.get("eps"),
                    "num_shots": season_result.get("num_shots"),
                    "status": season_result.get("status"),
                    "warning": season_result.get("warning"),
                    "error": season_result.get("error"),
                    "skipped_proximity": season_result.get("skipped_proximity"),
                }
            )
        return rows

    # Aggregate / error shape
    rows.append(
        {
            "player_id": int(player_id),
            "source": source,
            "season": None,
            "execution_skill": result.get("execution_skill"),
            "rationality": result.get("rationality"),
            "ees": result.get("ees"),
            "eps": result.get("eps"),
            "num_shots": result.get("num_shots"),
            "status": result.get("status"),
            "warning": result.get("warning"),
            "error": result.get("error"),
            "skipped_proximity": result.get("skipped_proximity"),
        }
    )
    return rows


def run_phase3_ab(
    *,
    limit_players: int,
    seasons: list[int],
    legacy_root: Path,
    new_root: Path,
    report_dir: Path,
    rng_seed: int,
    num_planning_skills: int,
    candidate_skills: list[float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    report_dir.mkdir(parents=True, exist_ok=True)

    player_ids, overlap_df = select_top_overlap_players(limit=limit_players, seasons=seasons)

    raw_results: dict[str, Any] = {
        "run_metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "seasons": seasons,
            "player_ids": player_ids,
            "rng_seed": rng_seed,
            "num_planning_skills": num_planning_skills,
            "candidate_skills": candidate_skills,
            "shot_type_filter": ["wristshot", "snapshot"],
            "legacy_root": str(legacy_root),
            "new_root": str(new_root),
        },
        "overlap_counts": overlap_df.to_dict(orient="records") if not overlap_df.empty else [],
        "per_player": {},
    }

    summary_rows: list[dict[str, Any]] = []

    for player_id in player_ids:
        raw_results["per_player"][str(player_id)] = {}

        for source, root, maps_source in (
            ("legacy", legacy_root, "legacy"),
            ("new", new_root, "new"),
        ):
            try:
                # 1) Persist offline data unless already present.
                save_player_data(
                    player_id=int(player_id),
                    seasons=seasons,
                    output_dir=root,
                    overwrite=False,
                    maps_source=maps_source,
                    value_column="expected_goals",
                )

                # 2) Load only seasons that are present on disk.
                available = _available_seasons_for_player(int(player_id), seasons, root)
                if not available:
                    per_source_result = {
                        "status": "no_offline_data",
                        "warning": "No seasons had both shots_*.parquet and shot_maps_*.npz present.",
                        "num_shots": 0,
                    }
                    raw_results["per_player"][str(player_id)][source] = per_source_result
                    summary_rows.extend(
                        _flatten_per_season_results(
                            player_id=int(player_id),
                            source=source,
                            result=per_source_result,
                            seasons_requested=seasons,
                        )
                    )
                    continue

                df, shot_maps = load_player_data(
                    player_id=int(player_id),
                    seasons=available,
                    data_dir=root,
                )

                # 3) Restrict to wristshot/snapshot only (strict).
                df = _filter_to_wristshot_snapshot(df)
                if df.empty:
                    per_source_result = {
                        "status": "no_wristshot_snapshot_shots",
                        "warning": "No wristshot/snapshot shots remained after filtering.",
                        "num_shots": 0,
                    }
                    raw_results["per_player"][str(player_id)][source] = per_source_result
                    summary_rows.extend(
                        _flatten_per_season_results(
                            player_id=int(player_id),
                            source=source,
                            result=per_source_result,
                            seasons_requested=seasons,
                        )
                    )
                    continue

                shot_maps = _prune_shot_maps(shot_maps, df["event_id"])

                # 4) Run JEEDS offline with identical hyperparameters.
                result = estimate_player_skill(
                    player_id=int(player_id),
                    seasons=available,
                    per_season=True,
                    candidate_skills=candidate_skills,
                    num_planning_skills=num_planning_skills,
                    rng_seed=rng_seed,
                    return_intermediate_estimates=False,
                    save_intermediate_csv=True,
                    data_dir=root,
                    confirm=False,
                    offline_data=(df, shot_maps),
                    shot_group="wristshot_snapshot",
                )

                raw_results["per_player"][str(player_id)][source] = result
                summary_rows.extend(
                    _flatten_per_season_results(
                        player_id=int(player_id),
                        source=source,
                        result=result,
                        seasons_requested=seasons,
                    )
                )

            except Exception as e:
                per_source_result = {
                    "status": "error",
                    "error": str(e),
                    "num_shots": 0,
                }
                raw_results["per_player"][str(player_id)][source] = per_source_result
                summary_rows.extend(
                    _flatten_per_season_results(
                        player_id=int(player_id),
                        source=source,
                        result=per_source_result,
                        seasons_requested=seasons,
                    )
                )

    summary_df = pd.DataFrame(summary_rows)
    # Stable sort for easy diffs.
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["player_id", "source", "season"], na_position="last")

    return summary_df, raw_results


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 xG migration: JEEDS A/B run")

    parser.add_argument("--limit", type=int, default=10, help="Number of players to estimate")
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=DEFAULT_SEASONS,
        help="Seasons to estimate (default: 20212022 20222023 20232024 20242025)",
    )
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--num-planning-skills", type=int, default=DEFAULT_NUM_PLANNING_SKILLS)
    parser.add_argument(
        "--num-execution-skills",
        type=int,
        default=DEFAULT_NUM_EXECUTION_SKILLS,
        help="Number of candidate execution-skill hypotheses (shared across A/B)",
    )
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=DEFAULT_LEGACY_ROOT,
        help="Offline persistence root for legacy maps",
    )
    parser.add_argument(
        "--new-root",
        type=Path,
        default=DEFAULT_NEW_ROOT,
        help="Offline persistence root for new-table maps",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for phase3 summary artifacts",
    )

    args = parser.parse_args()

    candidate_skills = np.linspace(0.004, 0.25, int(args.num_execution_skills)).tolist()

    summary_df, raw_results = run_phase3_ab(
        limit_players=int(args.limit),
        seasons=list(args.seasons),
        legacy_root=args.legacy_root,
        new_root=args.new_root,
        report_dir=args.report_dir,
        rng_seed=int(args.rng_seed),
        num_planning_skills=int(args.num_planning_skills),
        candidate_skills=candidate_skills,
    )

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = args.report_dir / f"xg_migration_phase3_ab_{ts}_summary.csv"
    json_path = args.report_dir / f"xg_migration_phase3_ab_{ts}_summary.json"

    summary_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(raw_results, indent=2, sort_keys=True))

    print(f"Wrote summary CSV:  {csv_path}")
    print(f"Wrote summary JSON: {json_path}")


if __name__ == "__main__":
    _cli()
