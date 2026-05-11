"""Smoke test for the new post-shot xG net-grid table.

Phase 1 goal: fail fast if Snowflake permissions/schema names are wrong.

Runs a few tiny queries:
- total row count
- min/max LAST_UPDATED
- distinct EVENT_ID_HAWKS count for a given season (via EVENT→GAME join)

Usage
-----
python -m BlackhawksSkillEstimation.smoke_test_new_xg_table --season 20242025
"""

from __future__ import annotations

import argparse
from datetime import datetime

from BlackhawksAPI import db

NEW_TABLE = "hawks_analytics.expected_goal_values_post_shot_net_grid"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Smoke test Snowflake access for the new xG net-grid table."
    )
    p.add_argument(
        "--season",
        type=int,
        default=20242025,
        help="Season ID to use for the DISTINCT EVENT_ID_HAWKS sanity check.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    print("=" * 80)
    print("New xG Net-Grid Table Smoke Test")
    print("=" * 80)
    print(f"Table:   {NEW_TABLE}")
    print(f"Season:  {args.season}")
    print(f"Time:    {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 80)

    count_df = db.get_df(f"SELECT COUNT(*) AS row_count FROM {NEW_TABLE};").rename(
        columns=str.lower
    )
    row_count = int(count_df.iloc[0]["row_count"]) if not count_df.empty else None
    print(f"Row count: {row_count}")

    updated_df = db.get_df(
        f"""
        SELECT
            MIN(LAST_UPDATED) AS min_last_updated,
            MAX(LAST_UPDATED) AS max_last_updated
        FROM {NEW_TABLE};
        """
    ).rename(columns=str.lower)

    if updated_df.empty:
        print("LAST_UPDATED range: <no rows>")
    else:
        print(
            "LAST_UPDATED range: "
            f"{updated_df.iloc[0].get('min_last_updated')} → {updated_df.iloc[0].get('max_last_updated')}"
        )

    season_df = db.get_df(
        f"""
        SELECT COUNT(DISTINCT e.EVENT_ID_HAWKS) AS distinct_event_ids
                FROM public.event AS e
                JOIN public.game AS g
          ON g.GAME_ID_HAWKS = e.GAME_ID_HAWKS
        WHERE g.SEASON = %(season)s
          AND e.EVENT_NAME = 'shot'
          AND EXISTS (
              SELECT 1
              FROM {NEW_TABLE} AS n
              WHERE n.EVENT_ID_HAWKS = e.EVENT_ID_HAWKS
          );
        """,
        query_params={"season": args.season},
    ).rename(columns=str.lower)

    distinct_event_ids = (
        int(season_df.iloc[0]["distinct_event_ids"]) if not season_df.empty else None
    )
    print(f"Distinct EVENT_ID_HAWKS with new grid in season {args.season}: {distinct_event_ids}")

    print("=" * 80)
    print("OK")


if __name__ == "__main__":
    main()
