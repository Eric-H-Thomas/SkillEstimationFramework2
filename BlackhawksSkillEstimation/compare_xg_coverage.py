"""Coverage comparison: legacy xG maps vs new xG net-grid table.

Phase 1 goal: compare coverage (event_id overlap) per season and confirm the
new-table grid is complete and reshapeable.

This utility prefers set-based SQL (DISTINCT + EXISTS) for coverage and only
pulls per-cell rows for a small sample for grid integrity checks.

Note: the Phase 1 migration uses ``EXPECTED_GOALS`` from the new table.

Usage
-----
python -m BlackhawksSkillEstimation.compare_xg_coverage
python -m BlackhawksSkillEstimation.compare_xg_coverage --seasons 20232024 20242025
python -m BlackhawksSkillEstimation.compare_xg_coverage --player-id 12345 --seasons 20242025
python -m BlackhawksSkillEstimation.compare_xg_coverage --game-ids 123456 123457
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from BlackhawksAPI import db
from BlackhawksAPI import queries as api_queries

LEGACY_TABLE = "hawks_analytics.post_shot_xg_value_maps"
NEW_TABLE = "hawks_analytics.expected_goal_values_post_shot_net_grid"

# Legacy map grid dimensions for reference (post_shot_xg_value_maps).
# The new net-grid table uses a different resolution (observed as 51×31 in practice),
# so Phase 1 integrity checks treat "complete" as "rectangular and non-sparse":
# n_rows == n_unique_y * n_unique_z.
LEGACY_Y_UNIQUE = 120
LEGACY_Z_UNIQUE = 72

DEFAULT_SEASONS = [20212022, 20222023, 20232024, 20242025, 20252026]


@dataclass
class SeasonCoverage:
    season: int
    legacy_distinct_event_ids: int
    new_distinct_event_ids: int
    overlap_event_ids: int
    legacy_only_event_ids: int
    new_only_event_ids: int


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare legacy post-shot xG map coverage vs new xG net-grid table."
    )
    p.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=DEFAULT_SEASONS,
        help="Season IDs to include (default includes overlap seasons plus 20252026 new-only).",
    )
    p.add_argument(
        "--player-id",
        type=int,
        default=None,
        help="Optional PLAYER_ID_HAWKS filter (faster, smaller sets).",
    )
    p.add_argument(
        "--game-ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional GAME_ID_HAWKS filter (faster, smaller sets).",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="How many event_ids to sample from each set for integrity checks.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible sampling.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Data/Hockey/migration_reports"),
        help="Output directory for CSV/JSON reports.",
    )
    return p


def _sql_game_filter(game_ids: list[int] | None) -> str:
    if not game_ids:
        return ""
    ids_str = ",".join(str(int(g)) for g in game_ids)
    return f"AND e.GAME_ID_HAWKS IN ({ids_str})"


def _sql_player_filter(player_id: int | None) -> str:
    if not player_id:
        return ""
    return "AND e.PLAYER_ID_HAWKS = %(player_id)s"


def _fetch_event_ids_for_season(
    *,
    season: int,
    exists_table: str,
    player_id: int | None,
    game_ids: list[int] | None,
) -> set[int]:
    query = f"""
        SELECT DISTINCT e.EVENT_ID_HAWKS AS event_id_hawks
                FROM public.event AS e
                JOIN public.game AS g
          ON g.GAME_ID_HAWKS = e.GAME_ID_HAWKS
        WHERE g.SEASON = %(season)s
          AND e.EVENT_NAME = 'shot'
          {_sql_game_filter(game_ids)}
          {_sql_player_filter(player_id)}
          AND EXISTS (
              SELECT 1
              FROM {exists_table} AS t
              WHERE t.EVENT_ID_HAWKS = e.EVENT_ID_HAWKS
          )
        ;
    """

    params: dict[str, object] = {"season": season}
    if player_id is not None:
        params["player_id"] = player_id

    df = db.get_df(query, query_params=params).rename(columns=str.lower)
    if df.empty:
        return set()
    return set(int(x) for x in df["event_id_hawks"].tolist())


def _sample_ids(rng: random.Random, ids: set[int], k: int) -> list[int]:
    if k <= 0 or not ids:
        return []
    k = min(k, len(ids))
    return rng.sample(sorted(ids), k=k)


def _grid_integrity_new_table(event_ids: list[int]) -> pd.DataFrame:
    if not event_ids:
        return pd.DataFrame()

    ids_str = ",".join(str(int(eid)) for eid in event_ids)
    query = f"""
        SELECT
            EVENT_ID_HAWKS AS event_id_hawks,
            COUNT(*) AS n_rows,
            COUNT(DISTINCT GOALLINE_Y_MODEL) AS n_unique_y,
            COUNT(DISTINCT GOALLINE_Z_MODEL) AS n_unique_z,
            MIN(GOALLINE_Y_MODEL) AS min_y,
            MAX(GOALLINE_Y_MODEL) AS max_y,
            MIN(GOALLINE_Z_MODEL) AS min_z,
            MAX(GOALLINE_Z_MODEL) AS max_z
        FROM {NEW_TABLE}
        WHERE EVENT_ID_HAWKS IN ({ids_str})
        GROUP BY EVENT_ID_HAWKS
        ORDER BY EVENT_ID_HAWKS
        ;
    """

    df = db.get_df(query).rename(columns=str.lower)
    if df.empty:
        return df

    df["expected_rows_rectangular"] = df["n_unique_y"] * df["n_unique_z"]
    df["grid_ok"] = (df["n_rows"] == df["expected_rows_rectangular"]) & (df["n_rows"] > 0)
    return df


def main() -> None:
    args = _build_parser().parse_args()

    if not args.seasons:
        raise ValueError("--seasons must contain at least one season")
    if args.sample_size < 0:
        raise ValueError("--sample-size must be >= 0")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    started_at = datetime.now()

    per_season: list[SeasonCoverage] = []
    samples: dict[int, dict[str, list[int]]] = {}
    grid_checks: list[dict[str, object]] = []
    warnings: list[str] = []
    reshape_validation: list[dict[str, object]] = []
    integrity_by_event: dict[int, dict[str, object]] = {}

    print("=" * 80)
    print("xG Coverage Comparison (Legacy vs New)")
    print("=" * 80)
    print(f"Legacy table: {LEGACY_TABLE}")
    print(f"New table:    {NEW_TABLE}")
    print(f"Seasons:      {args.seasons}")
    print(f"Player ID:    {args.player_id}")
    print(f"Game IDs:     {args.game_ids}")
    print(f"Sample size:  {args.sample_size}")
    print(f"Seed:         {args.seed}")
    print(f"Out dir:      {args.out_dir}")
    print("=" * 80)

    for season in args.seasons:
        legacy_ids = _fetch_event_ids_for_season(
            season=season,
            exists_table=LEGACY_TABLE,
            player_id=args.player_id,
            game_ids=args.game_ids,
        )
        new_ids = _fetch_event_ids_for_season(
            season=season,
            exists_table=NEW_TABLE,
            player_id=args.player_id,
            game_ids=args.game_ids,
        )

        overlap = legacy_ids & new_ids
        legacy_only = legacy_ids - new_ids
        new_only = new_ids - legacy_ids

        per_season.append(
            SeasonCoverage(
                season=season,
                legacy_distinct_event_ids=len(legacy_ids),
                new_distinct_event_ids=len(new_ids),
                overlap_event_ids=len(overlap),
                legacy_only_event_ids=len(legacy_only),
                new_only_event_ids=len(new_only),
            )
        )

        samples[season] = {
            "overlap": _sample_ids(rng, overlap, args.sample_size),
            "legacy_only": _sample_ids(rng, legacy_only, args.sample_size),
            "new_only": _sample_ids(rng, new_only, args.sample_size),
        }

        # Grid integrity: check only sampled IDs that actually exist in the new table.
        integrity_ids = sorted(set(samples[season]["overlap"] + samples[season]["new_only"]))
        integrity_df = _grid_integrity_new_table(integrity_ids)
        if not integrity_df.empty:
            for row in integrity_df.to_dict(orient="records"):
                row["season"] = season
                grid_checks.append(row)
                integrity_by_event[int(row["event_id_hawks"])] = row

            bad = integrity_df[~integrity_df["grid_ok"]]
            if not bad.empty:
                warnings.append(
                    f"Season {season}: {len(bad)} sampled new-table events failed grid integrity checks."
                )

            # If dimensions vary within the sample, call it out.
            dims = set(zip(integrity_df["n_unique_y"], integrity_df["n_unique_z"]))
            if len(dims) > 1:
                warnings.append(
                    f"Season {season}: sampled new-table grid dimensions vary: {sorted(dims)}"
                )

    # End-to-end reshape validation (very small sample): actually build maps.
    # This confirms the new fetcher can produce a (72, 120) value_map.
    candidate_event_ids: list[int] = []
    for season in args.seasons:
        candidate_event_ids.extend(samples.get(season, {}).get("overlap", [])[:1])
        candidate_event_ids.extend(samples.get(season, {}).get("new_only", [])[:1])
    candidate_event_ids = [eid for eid in candidate_event_ids if eid is not None]
    candidate_event_ids = list(dict.fromkeys(candidate_event_ids))[:3]

    for event_id in candidate_event_ids:
        try:
            maps = api_queries.get_shot_maps_by_event_ids_new_xg([event_id])
            m = maps.get(event_id)
            if not m:
                reshape_validation.append(
                    {
                        "event_id_hawks": event_id,
                        "ok": False,
                        "error": "No map returned (empty result)",
                    }
                )
                continue
            value_map = m.get("value_map")
            shape = tuple(getattr(value_map, "shape", ()))

            integrity = integrity_by_event.get(int(event_id))
            if integrity:
                expected_shape = (int(integrity["n_unique_z"]), int(integrity["n_unique_y"]))
                ok = shape == expected_shape
            else:
                expected_shape = None
                ok = bool(shape) and len(shape) == 2

            reshape_validation.append(
                {
                    "event_id_hawks": event_id,
                    "ok": bool(ok),
                    "value_map_shape": shape,
                    "expected_shape": expected_shape,
                }
            )
            if not ok:
                warnings.append(
                    f"Event {event_id}: new fetcher produced value_map shape {shape}, expected {expected_shape}."
                )
        except Exception as exc:  # noqa: BLE001
            reshape_validation.append(
                {
                    "event_id_hawks": event_id,
                    "ok": False,
                    "error": repr(exc),
                }
            )
            warnings.append(
                f"Event {event_id}: new fetcher raised during reshape validation: {exc!r}"
            )

    # Write outputs
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    stem = f"xg_coverage_{timestamp}"

    summary_df = pd.DataFrame([asdict(x) for x in per_season]).sort_values("season")
    summary_csv = args.out_dir / f"{stem}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    grid_df = pd.DataFrame(grid_checks)
    grid_csv = args.out_dir / f"{stem}_grid_checks.csv"
    if not grid_df.empty:
        grid_df.to_csv(grid_csv, index=False)

    payload = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "seasons": args.seasons,
        "player_id": args.player_id,
        "game_ids": args.game_ids,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "legacy_grid_reference": {
            "n_unique_y": LEGACY_Y_UNIQUE,
            "n_unique_z": LEGACY_Z_UNIQUE,
        },
        "tables": {"legacy": LEGACY_TABLE, "new": NEW_TABLE},
        "per_season": [asdict(x) for x in per_season],
        "samples": {str(k): v for k, v in samples.items()},
        "grid_checks": grid_checks,
        "reshape_validation": reshape_validation,
        "warnings": warnings,
    }

    report_json = args.out_dir / f"{stem}_report.json"
    report_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    # Console summary
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"- {w}")

    print("\nWrote:")
    print(f"- {summary_csv}")
    print(f"- {report_json}")
    if not grid_df.empty:
        print(f"- {grid_csv}")


if __name__ == "__main__":
    main()
