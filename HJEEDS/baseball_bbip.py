# This file was written or edited by AI and still requires human review. Delete this comment when done.
"""BB/IP-based pitcher selection for baseball HJEEDS rosters."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import unicodedata
from pathlib import Path
from typing import Sequence

import pandas as pd

BBIP_INNINGS_CACHE_FILENAME = "bbip_innings_cache.json"
BUNDLED_INNINGS_FILENAME_TEMPLATE = "baseball_innings_pitched_{season_year}.json"


def _ascii_fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(character for character in normalized if not unicodedata.combining(character))


def canonical_player_name(name: str) -> str:
    """Map Statcast 'Last, First' and BRef 'First Last' to one lookup key."""

    name = _ascii_fold(str(name).strip().lower())
    if "," in name:
        last, first = (piece.strip() for piece in name.split(",", maxsplit=1))
        return f"{last} {first}"
    parts = name.split()
    if len(parts) >= 2:
        last = parts[-1]
        first = " ".join(parts[:-1])
        return f"{last} {first}"
    return name


def _normalize_player_name(name: str) -> str:
    return canonical_player_name(name)


def parse_innings_pitched(raw_value: object) -> float:
    """Parse baseball IP notation (e.g. 130.1 = 130 and 1/3 innings)."""

    text = str(raw_value).strip()
    if "." in text:
        whole_str, frac_str = text.split(".", 1)
        whole = int(whole_str) if whole_str else 0
        outs = int(frac_str)
        if outs in (0, 1, 2):
            return whole + outs / 3.0
    return float(text)


def bundled_innings_path_for(season_year: int) -> Path:
    return Path(__file__).resolve().parent / "data" / BUNDLED_INNINGS_FILENAME_TEMPLATE.format(
        season_year=season_year
    )


def bbip_cache_path_for(output_dir: Path) -> Path:
    return Path(output_dir) / BBIP_INNINGS_CACHE_FILENAME


def resolve_bbip_cache_path(*, output_dir: Path | None = None) -> Path | None:
    """Return a BB/IP innings cache path from env or the experiment output directory."""

    env_path = os.environ.get("BBIP_CACHE_PATH")
    if env_path:
        path = Path(env_path)
        if path.is_file():
            return path
    if output_dir is not None:
        default_path = bbip_cache_path_for(output_dir)
        if default_path.is_file():
            return default_path
    return None


def _innings_dict_from_stats_frame(stats: pd.DataFrame, *, source: str) -> dict[str, float]:
    if stats is None or len(stats) == 0:
        raise ValueError(f"No pitching stats returned from {source}.")

    innings_by_name: dict[str, float] = {}
    for _, row in stats.iterrows():
        name = canonical_player_name(str(row["Name"]))
        innings = parse_innings_pitched(row["IP"])
        if innings > 0:
            innings_by_name[name] = innings
    if not innings_by_name:
        raise ValueError(f"No usable innings-pitched rows from {source}.")
    return innings_by_name


def _fetch_innings_from_bref(season_year: int) -> tuple[dict[str, float], str]:
    from pybaseball import pitching_stats_bref

    stats = pitching_stats_bref(season_year)
    return _innings_dict_from_stats_frame(stats, source="pybaseball.pitching_stats_bref"), (
        "pybaseball.pitching_stats_bref"
    )


def _fetch_innings_from_fangraphs(season_year: int) -> tuple[dict[str, float], str]:
    from pybaseball import pitching_stats

    stats = pitching_stats(season_year, qual=0)
    return _innings_dict_from_stats_frame(stats, source="pybaseball.pitching_stats"), "pybaseball.pitching_stats"


def _load_bundled_innings_pitched(season_year: int) -> tuple[dict[str, float], str]:
    path = bundled_innings_path_for(season_year)
    if not path.is_file():
        raise FileNotFoundError(f"No bundled innings cache for season_year={season_year}: {path}")
    cached_season_year, innings_by_name = load_bbip_innings_cache(path)
    if cached_season_year != season_year:
        raise ValueError(
            f"Bundled innings cache season_year={cached_season_year} does not match {season_year}."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    source = str(payload.get("source", f"bundled:{path.name}"))
    return innings_by_name, source


def fetch_innings_pitched_by_name(season_year: int) -> tuple[dict[str, float], str]:
    """Fetch season IP by normalized player name, with network fallbacks then bundled data."""

    errors: list[str] = []
    for fetcher in (_fetch_innings_from_bref, _fetch_innings_from_fangraphs):
        try:
            return fetcher(season_year)
        except Exception as exc:
            errors.append(f"{fetcher.__name__}: {exc}")
    try:
        return _load_bundled_innings_pitched(season_year)
    except Exception as exc:
        errors.append(f"bundled: {exc}")
    raise RuntimeError(
        "Could not load innings pitched for season_year="
        f"{season_year}. Tried Baseball Reference, FanGraphs, and bundled data.\n"
        + "\n".join(errors)
    )


def load_bbip_innings_cache(path: Path) -> tuple[int, dict[str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    season_year = int(payload["season_year"])
    innings_by_name = {
        str(name): float(innings)
        for name, innings in payload["innings_by_name"].items()
    }
    return season_year, innings_by_name


def write_bbip_innings_cache(output_dir: Path, *, season_year: int) -> Path:
    """Write innings cache for offline compute nodes (network fetch or bundled fallback)."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = bbip_cache_path_for(output_dir)

    try:
        innings_by_name, source = fetch_innings_pitched_by_name(season_year)
    except Exception:
        bundled_path = bundled_innings_path_for(season_year)
        if not bundled_path.is_file():
            raise
        shutil.copyfile(bundled_path, path)
        os.environ["BBIP_CACHE_PATH"] = str(path.resolve())
        print(
            f"[baseball-bbip] Network fetch unavailable; copied bundled innings cache from {bundled_path}",
            flush=True,
        )
        return path

    payload = {
        "season_year": season_year,
        "source": source,
        "innings_by_name": innings_by_name,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.environ["BBIP_CACHE_PATH"] = str(path.resolve())
    return path


def _innings_pitched_by_name(
    season_year: int,
    *,
    output_dir: Path | None = None,
) -> dict[str, float]:
    cache_path = resolve_bbip_cache_path(output_dir=output_dir)
    if cache_path is not None:
        cached_season_year, innings_by_name = load_bbip_innings_cache(cache_path)
        if cached_season_year != season_year:
            raise ValueError(
                f"BB/IP cache season_year={cached_season_year} does not match requested "
                f"season_year={season_year} ({cache_path})."
            )
        return innings_by_name

    try:
        innings_by_name, _source = fetch_innings_pitched_by_name(season_year)
        return innings_by_name
    except Exception as exc:
        raise FileNotFoundError(
            "BB/IP innings cache not found and live innings fetch failed. "
            "On a network-enabled host, run:\n"
            f"  python -m HJEEDS.baseball_bbip --write-cache --season-year {season_year} "
            f"--output-dir <experiment_output_dir>\n"
            "Submit scripts copy bundled season data automatically when remote fetch is blocked."
        ) from exc


def _walk_count_by_pitcher(all_data: pd.DataFrame) -> pd.Series:
    walks = all_data.loc[all_data["events"] == "walk", "pitcher"]
    return walks.groupby(walks).size()


def _pitcher_name_map(all_data: pd.DataFrame) -> dict[int, str]:
    names: dict[int, str] = {}
    for pitcher_id in all_data["pitcher"].unique():
        rows = all_data.loc[all_data["pitcher"] == pitcher_id, "player_name"]
        if len(rows) == 0:
            continue
        names[int(pitcher_id)] = str(rows.iloc[0])
    return names


def compute_bbip_by_pitcher(
    all_data: pd.DataFrame,
    *,
    season_year: int,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Return pitcher-level BB/IP for one season using Statcast walks and season IP."""

    walk_counts = _walk_count_by_pitcher(all_data)
    innings_by_name = _innings_pitched_by_name(season_year, output_dir=output_dir)
    name_map = _pitcher_name_map(all_data)

    rows: list[dict[str, object]] = []
    for pitcher_id, player_name in name_map.items():
        normalized = _normalize_player_name(player_name)
        innings = innings_by_name.get(normalized)
        if innings is None or innings <= 0:
            continue
        walks = int(walk_counts.get(pitcher_id, 0))
        if walks == 0:
            continue
        rows.append(
            {
                "pitcher_id": pitcher_id,
                "player_name": player_name,
                "walks": walks,
                "innings_pitched": innings,
                "bbip": walks / innings,
            }
        )

    if not rows:
        raise ValueError(
            f"Could not compute BB/IP for any pitchers in season_year={season_year}."
        )
    return pd.DataFrame(rows).sort_values("bbip")


def build_bbip_manifest(
    all_data: pd.DataFrame,
    *,
    season_year: int,
    pitcher_ids: Sequence[int],
    extremes_count: int,
    output_dir: Path | None = None,
) -> list[dict[str, object]]:
    """Return BB/IP metadata for selected pitchers (bottom/top tiers)."""

    table = compute_bbip_by_pitcher(all_data, season_year=season_year, output_dir=output_dir)
    bottom_ids = set(table.head(extremes_count)["pitcher_id"].astype(int).tolist())
    top_ids = set(table.tail(extremes_count)["pitcher_id"].astype(int).tolist())
    selected = table.loc[table["pitcher_id"].isin(pitcher_ids)].copy()
    manifest: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        pitcher_id = int(row["pitcher_id"])
        if pitcher_id in bottom_ids:
            tier = "bottom"
        elif pitcher_id in top_ids:
            tier = "top"
        else:
            tier = "middle"
        manifest.append(
            {
                "pitcher_id": pitcher_id,
                "player_name": row["player_name"],
                "walks": int(row["walks"]),
                "innings_pitched": float(row["innings_pitched"]),
                "bbip": float(row["bbip"]),
                "tier": tier,
            }
        )
    manifest.sort(key=lambda row: (str(row["tier"]), float(row["bbip"])))
    return manifest


def select_bbip_extreme_pitcher_ids(
    all_data: pd.DataFrame,
    *,
    season_year: int,
    count: int,
    output_dir: Path | None = None,
    eligible_pitcher_ids: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Return top-N and bottom-N pitcher IDs by season BB/IP (deduplicated)."""

    if count <= 0:
        raise ValueError(f"count must be positive. Received {count}.")

    table = compute_bbip_by_pitcher(all_data, season_year=season_year, output_dir=output_dir)
    if eligible_pitcher_ids is not None:
        eligible = {int(pitcher_id) for pitcher_id in eligible_pitcher_ids}
        table = table.loc[table["pitcher_id"].isin(eligible)].copy()
    if len(table) < count * 2:
        raise ValueError(
            f"Need at least {count * 2} eligible pitchers with BB/IP data. Received {len(table)}."
        )

    bottom = table.head(count)["pitcher_id"].astype(int).tolist()
    top = table.tail(count)["pitcher_id"].astype(int).tolist()
    ordered: list[int] = []
    for pitcher_id in bottom + top:
        if pitcher_id not in ordered:
            ordered.append(pitcher_id)
    return tuple(ordered)


def parse_bbip_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare offline BB/IP artifacts for baseball HJEEDS.")
    parser.add_argument(
        "--write-cache",
        action="store_true",
        help="Write bbip_innings_cache.json (network fetch with bundled fallback).",
    )
    parser.add_argument("--season-year", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_bbip_args(argv)
    if not args.write_cache:
        raise SystemExit("Specify --write-cache.")
    path = write_bbip_innings_cache(Path(args.output_dir), season_year=args.season_year)
    print(f"[baseball-bbip] Wrote innings cache to {path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
