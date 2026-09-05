"""Derive a JEEDS cluster config for a chosen set of seasons.

jeeds_forwards_all_seasons.json is the canonical base: it holds the 218 forwards and the
250x250 estimator block every JEEDS cluster run uses. This rewrites
``data_filters.seasons`` and ``cluster_plan.jobs`` for the requested seasons, recomputing
eligibility with the same shot-type filter and 100-shot threshold, and copies the roster,
estimator settings, and sbatch recommendation from the base unchanged.

Usage:
    python BlackhawksSkillEstimation/build_jeeds_seasons_config.py \
        --seasons 20252026 \
        --out Data/Hockey/jobs/jeeds_forwards_20252026.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from BlackhawksSkillEstimation.BlackhawksJEEDS import SHOT_TYPE_GROUPS  # noqa: E402

MIN_SHOTS = 100


def count_group_shots(parquet_path: Path, shot_group: str) -> int:
    """Shots surviving the shot-type filter, matching BlackhawksJEEDS.estimate_player_skill."""
    _, allowed_types, include_null = SHOT_TYPE_GROUPS[shot_group]
    df = pd.read_parquet(parquet_path).rename(columns=str.lower)
    if "shot_type" not in df.columns:
        return len(df)
    shot_series = df["shot_type"].where(pd.notna(df["shot_type"]))
    shot_lower = shot_series.astype(str).str.lower()
    mask = shot_lower.isin(allowed_types)
    if include_null:
        mask = shot_series.isna() | mask
    return int(mask.sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="Data/Hockey/jobs/jeeds_forwards_all_seasons.json")
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-shots", type=int, default=MIN_SHOTS)
    args = parser.parse_args()

    base_path = REPO_ROOT / args.base
    cfg = json.loads(base_path.read_text())
    data_root = REPO_ROOT / cfg["data_root"]
    player_ids = [int(p) for p in cfg["data_filters"]["player_ids"]]
    shot_groups = [str(g) for g in cfg["data_filters"]["shot_groups"]]

    jobs = []
    for player_id in player_ids:
        data_dir = data_root / "players" / f"player_{player_id}" / "data"
        for season in args.seasons:
            parquet = data_dir / f"shots_{season}.parquet"
            npz = data_dir / f"shot_maps_{season}.npz"
            missing = not (parquet.exists() and npz.exists())
            for shot_group in shot_groups:
                count = None if missing else count_group_shots(parquet, shot_group)
                jobs.append(
                    {
                        "count": count,
                        "eligible": bool(not missing and count >= args.min_shots),
                        "missing_local_data": missing,
                        "player_id": player_id,
                        "season": int(season),
                        "shot_group": shot_group,
                    }
                )

    eligible = [j for j in jobs if j["eligible"]]
    cfg["data_filters"]["seasons"] = [int(s) for s in args.seasons]
    cfg["cluster_plan"]["jobs"] = jobs
    cfg["cluster_plan"]["eligible_jobs"] = len(eligible)
    cfg["cluster_plan"]["total_jobs"] = len(jobs)
    cfg["notes"] = (
        f"Derived from {args.base} for seasons {args.seasons}; eligibility recomputed "
        f"with the {shot_groups} shot-type filter at >={args.min_shots} shots."
    )

    out_path = REPO_ROOT / args.out
    out_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")

    print(f"Wrote {args.out}")
    print(f"  total jobs   : {len(jobs)}")
    print(f"  eligible     : {len(eligible)}")
    print(f"  missing cache: {sum(1 for j in jobs if j['missing_local_data'])}")
    for season in args.seasons:
        n = sum(1 for j in eligible if j["season"] == season)
        print(f"  eligible {season}: {n}")
    print(f"  array size for legacy submission : {len(jobs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
