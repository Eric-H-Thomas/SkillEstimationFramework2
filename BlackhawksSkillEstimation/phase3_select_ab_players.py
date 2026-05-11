"""Phase 3 (xG migration): select overlap-heavy forwards for A/B JEEDS.

Selection goal
--------------
Pick players where A/B differences are driven by the xG surface change, not by
missing maps. We therefore count only shots (events) that have *both*:
- legacy map row(s) in hawks_analytics.post_shot_xg_value_maps
- new-table grid row(s) in hawks_analytics.expected_goal_values_post_shot_net_grid

We also restrict to shot_type in {'wristshot','snapshot'} and to the forwards
listed in Data/Hockey/forwards23-25.txt.

This module is intentionally small and scriptable.

Usage
-----
python -m BlackhawksSkillEstimation.phase3_select_ab_players --limit 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from BlackhawksAPI import db


DEFAULT_SEASONS: list[int] = [20212022, 20222023, 20232024, 20242025]
DEFAULT_SHOT_TYPES: tuple[str, ...] = ("wristshot", "snapshot")
DEFAULT_FORWARDS_FILE = Path("Data/Hockey/forwards23-25.txt")
DEFAULT_PHASE2_REPORT_DIR = Path("Data/Hockey/migration_reports")

def read_player_ids(path: Path | str) -> list[int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing player list file: {path}")
    ids: list[int] = []
    for raw in path.read_text().splitlines():
        text = raw.strip()
        if not text:
            continue
        ids.append(int(text))
    return sorted(set(ids))


def _latest_phase2_cluster_players_csv(report_dir: Path) -> Path | None:
    if not report_dir.exists():
        return None
    candidates = sorted(report_dir.glob("xg_map_migration_phase2_*_cluster_players_top25.csv"))
    return candidates[-1] if candidates else None


def _fallback_players_from_phase2_csv(
    *,
    report_dir: Path,
    seasons: list[int],
    forwards_ids: set[int],
    limit: int,
) -> list[int]:
    csv_path = _latest_phase2_cluster_players_csv(report_dir)
    if csv_path is None:
        raise FileNotFoundError(
            f"No Phase 2 cluster_players CSV found under {report_dir}. "
            "Expected a file like xg_map_migration_phase2_*_cluster_players_top25.csv"
        )

    df = pd.read_csv(csv_path)
    df = df.rename(columns=str.lower)

    required = {"season", "player_id_hawks", "overlap_count"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Phase 2 CSV is missing required columns: {sorted(missing)}")

    df = df[df["season"].isin(seasons)]
    df = df[df["player_id_hawks"].isin(sorted(forwards_ids))]

    totals = (
        df.groupby("player_id_hawks", as_index=False)["overlap_count"].sum()
        .rename(columns={"overlap_count": "total_overlap"})
        .sort_values("total_overlap", ascending=False)
    )

    return totals["player_id_hawks"].head(limit).astype(int).tolist()


def query_overlap_counts(
    *,
    seasons: list[int],
    player_ids: list[int],
    shot_types: tuple[str, ...] = DEFAULT_SHOT_TYPES,
) -> pd.DataFrame:
    """Query overlap shot counts per player per season.

    Returns a dataframe with columns:
    - player_id_hawks
    - season
    - overlap_shots
    """

    if not seasons or not player_ids:
        return pd.DataFrame(columns=["player_id_hawks", "season", "overlap_shots"])

    seasons_sql = ",".join(str(s) for s in seasons)
    players_sql = ",".join(str(pid) for pid in player_ids)
    shot_types_sql = ",".join(f"'{t}'" for t in shot_types)

    query = f"""
        SELECT
            e.PLAYER_ID_HAWKS AS player_id_hawks,
            g.SEASON          AS season,
            COUNT(*)          AS overlap_shots
        FROM HAWKS_HOCKEY.PUBLIC.EVENT AS e
        JOIN HAWKS_HOCKEY.PUBLIC.GAME AS g
          ON g.GAME_ID_HAWKS = e.GAME_ID_HAWKS
        WHERE g.SEASON IN ({seasons_sql})
          AND e.EVENT_NAME = 'shot'
          AND e.PLAYER_ID_HAWKS IN ({players_sql})
          AND LOWER(e.SHOT_TYPE) IN ({shot_types_sql})
          AND EXISTS (
              SELECT 1
              FROM HAWKS_HOCKEY.HAWKS_ANALYTICS.SHOT_TRAJECTORIES AS st
              WHERE st.EVENT_ID_HAWKS = e.EVENT_ID_HAWKS
          )
          AND EXISTS (
              SELECT 1
              FROM hawks_analytics.post_shot_xg_value_maps AS p
              WHERE p.EVENT_ID_HAWKS = e.EVENT_ID_HAWKS
          )
          AND EXISTS (
              SELECT 1
              FROM hawks_analytics.expected_goal_values_post_shot_net_grid AS n
              WHERE n.EVENT_ID_HAWKS = e.EVENT_ID_HAWKS
          )
        GROUP BY e.PLAYER_ID_HAWKS, g.SEASON
        ORDER BY overlap_shots DESC;
    """

    df = db.get_df(query).rename(columns=str.lower)
    if df.empty:
        return df

    df["player_id_hawks"] = df["player_id_hawks"].astype(int)
    df["season"] = df["season"].astype(int)
    df["overlap_shots"] = df["overlap_shots"].astype(int)
    return df


def select_top_overlap_players(
    *,
    limit: int = 10,
    seasons: list[int] | None = None,
    forwards_file: Path | str = DEFAULT_FORWARDS_FILE,
    use_phase2_fallback: bool = True,
    phase2_report_dir: Path = DEFAULT_PHASE2_REPORT_DIR,
    refine_fallback_with_sql: bool = True,
) -> tuple[list[int], pd.DataFrame]:
    """Return (player_ids, overlap_df) for Phase 3 A/B estimation."""

    seasons = list(seasons or DEFAULT_SEASONS)
    forwards_ids = set(read_player_ids(forwards_file))

    # Primary path: run overlap SQL across all candidate forwards.
    try:
        overlap_df = query_overlap_counts(
            seasons=seasons,
            player_ids=sorted(forwards_ids),
        )
        if overlap_df.empty:
            raise RuntimeError("Overlap SQL returned no rows.")

        totals = (
            overlap_df.groupby("player_id_hawks", as_index=False)["overlap_shots"].sum()
            .rename(columns={"overlap_shots": "total_overlap_shots"})
            .sort_values("total_overlap_shots", ascending=False)
        )

        top_ids = totals["player_id_hawks"].head(limit).astype(int).tolist()
        return top_ids, overlap_df

    except Exception as e:
        if not use_phase2_fallback:
            raise

        # Fallback: use Phase 2 clustering outputs to propose players,
        # optionally refining them via a smaller SQL query.
        candidate_ids = _fallback_players_from_phase2_csv(
            report_dir=phase2_report_dir,
            seasons=seasons,
            forwards_ids=forwards_ids,
            limit=max(limit, 25),
        )

        if refine_fallback_with_sql:
            try:
                overlap_df = query_overlap_counts(
                    seasons=seasons,
                    player_ids=candidate_ids,
                )
                totals = (
                    overlap_df.groupby("player_id_hawks", as_index=False)["overlap_shots"].sum()
                    .rename(columns={"overlap_shots": "total_overlap_shots"})
                    .sort_values("total_overlap_shots", ascending=False)
                )
                top_ids = totals["player_id_hawks"].head(limit).astype(int).tolist()
                return top_ids, overlap_df
            except Exception:
                # If even the refined query fails, fall back to Phase 2 CSV ordering.
                pass

        # Last-resort selection: Phase 2 CSV list.
        # Note: this overlap_count is not shot-type-filtered.
        return candidate_ids[:limit], pd.DataFrame()


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 selector: top overlap-heavy forwards for A/B JEEDS",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=DEFAULT_SEASONS,
        help="Seasons to consider (default: 20212022 20222023 20232024 20242025)",
    )
    parser.add_argument(
        "--forwards-file",
        type=Path,
        default=DEFAULT_FORWARDS_FILE,
        help="Text file with one PLAYER_ID_HAWKS per line.",
    )
    parser.add_argument(
        "--no-phase2-fallback",
        action="store_true",
        help="Disable fallback to Phase 2 CSVs.",
    )

    args = parser.parse_args()

    ids, overlap_df = select_top_overlap_players(
        limit=args.limit,
        seasons=list(args.seasons),
        forwards_file=args.forwards_file,
        use_phase2_fallback=not args.no_phase2_fallback,
    )

    print("Selected player IDs:")
    for pid in ids:
        print(pid)

    if not overlap_df.empty:
        totals = (
            overlap_df.groupby("player_id_hawks", as_index=False)["overlap_shots"].sum()
            .rename(columns={"overlap_shots": "total_overlap_shots"})
            .sort_values("total_overlap_shots", ascending=False)
        )
        print("\nTop overlap totals:")
        print(totals.head(len(ids)).to_string(index=False))


if __name__ == "__main__":
    _cli()
