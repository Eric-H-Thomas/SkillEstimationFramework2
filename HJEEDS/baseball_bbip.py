# This file was written or edited by AI and still requires human review. Delete this comment when done.
"""BB/IP-based pitcher selection for baseball HJEEDS rosters.

Paper run ``submit_hjeeds_baseball_convergence_paper_bbip.sh`` selects
``--bbip-extremes 10`` (top 10 + bottom 10) among FF-eligible 2021 pitchers
with ``min_pitches_per_agent=100``. Login-node submit scripts write
``bbip_innings_cache.json`` via ``--write-cache``; compute nodes read that
cache (or ``BBIP_CACHE_PATH``) without network access.

BB/IP here is Statcast walk events in the loaded season slice divided by
season innings pitched from Baseball Reference / FanGraphs / bundled JSON.
Pitchers with zero Statcast walks or unmatched names are dropped from the
ranking table (not assigned BB/IP = 0).
"""

from __future__ import annotations

import argparse
import json
import os
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


def load_bbip_innings_cache(path: Path) -> tuple[int, dict[str, float], str]:
    """Load ``(season_year, innings_by_name, source)`` from a cache JSON file."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    season_year = int(payload["season_year"])
    innings_by_name = {
        str(name): float(innings)
        for name, innings in payload["innings_by_name"].items()
    }
    source = str(payload.get("source", f"cache:{path.name}"))
    return season_year, innings_by_name, source


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
    cached_season_year, innings_by_name, source = load_bbip_innings_cache(path)
    if cached_season_year != season_year:
        raise ValueError(
            f"Bundled innings cache season_year={cached_season_year} does not match {season_year}."
        )
    return innings_by_name, source


def fetch_innings_pitched_by_name(season_year: int) -> tuple[dict[str, float], str]:
    """Fetch season IP by normalized player name: BRef, then FanGraphs, then bundled data."""

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


def write_bbip_innings_cache(output_dir: Path, *, season_year: int) -> Path:
    """Write innings cache for offline compute nodes (network fetch with bundled fallback)."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = bbip_cache_path_for(output_dir)
    innings_by_name, source = fetch_innings_pitched_by_name(season_year)
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
        cached_season_year, innings_by_name, _source = load_bbip_innings_cache(cache_path)
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
            "Submit scripts write this cache (with bundled fallback) before Slurm compute."
        ) from exc


def _walk_count_by_pitcher(all_data: pd.DataFrame) -> pd.Series:
    walks = all_data.loc[all_data["events"] == "walk", "pitcher"]
    return walks.groupby(walks).size()


def _pitcher_name_map(all_data: pd.DataFrame) -> dict[int, str]:
    subset = all_data.drop_duplicates(subset=["pitcher"], keep="first")
    return {
        int(pitcher_id): str(player_name)
        for pitcher_id, player_name in zip(subset["pitcher"], subset["player_name"])
    }


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
        normalized = canonical_player_name(player_name)
        innings = innings_by_name.get(normalized)
        if innings is None or innings <= 0:
            continue
        walks = int(walk_counts.get(pitcher_id, 0))
        # Exclude zero-walk pitchers rather than ranking them at BB/IP = 0.
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


def _filter_bbip_table(
    table: pd.DataFrame,
    eligible_pitcher_ids: Sequence[int] | None,
) -> pd.DataFrame:
    if eligible_pitcher_ids is None:
        return table
    eligible = {int(pitcher_id) for pitcher_id in eligible_pitcher_ids}
    return table.loc[table["pitcher_id"].isin(eligible)].copy()


def _require_extremes_capacity(table: pd.DataFrame, extremes_count: int, *, label: str) -> None:
    if extremes_count <= 0:
        raise ValueError(f"extremes_count must be positive. Received {extremes_count}.")
    if len(table) < extremes_count * 2:
        raise ValueError(
            f"Need at least {extremes_count * 2} {label} with BB/IP data. "
            f"Received {len(table)}."
        )


def _extreme_pitcher_id_lists(
    table: pd.DataFrame,
    extremes_count: int,
    *,
    label: str = "pitchers",
) -> tuple[list[int], list[int]]:
    """Return (lowest-BB/IP ids, highest-BB/IP ids) from a BB/IP table.

    Always sorts by ``bbip`` with a stable algorithm: callers may pass an
    already-sorted compute table (selection) or a tier-ordered manifest table
    (``baseball_separability.resolve_bbip_tiers``). Stable sort preserves tie
    order from ``compute_bbip_by_pitcher``, matching historical ``head``/``tail``
    on that sorted frame.
    """

    ordered = table.sort_values("bbip", kind="mergesort")
    _require_extremes_capacity(ordered, extremes_count, label=label)
    bottom = ordered.head(extremes_count)["pitcher_id"].astype(int).tolist()
    top = ordered.tail(extremes_count)["pitcher_id"].astype(int).tolist()
    return bottom, top


def label_bbip_tiers_for_pitcher_ids(
    table: pd.DataFrame,
    pitcher_ids: Sequence[int],
    *,
    extremes_count: int,
) -> dict[int, str]:
    """Label selected pitchers as top/bottom by BB/IP rank within the selected set.

    Ranking within ``pitcher_ids`` (not the global league table) matches
    ``--bbip-extremes`` selection among an eligible subset.
    """

    selected = table.loc[table["pitcher_id"].isin({int(pid) for pid in pitcher_ids})].copy()
    bottom_ids, top_ids = _extreme_pitcher_id_lists(
        selected,
        extremes_count,
        label="selected pitchers",
    )
    bottom_set = set(bottom_ids)
    top_set = set(top_ids)
    labels: dict[int, str] = {}
    for pitcher_id in pitcher_ids:
        pid = int(pitcher_id)
        if pid in bottom_set:
            labels[pid] = "bottom"
        elif pid in top_set:
            labels[pid] = "top"
        else:
            labels[pid] = "middle"
    return labels


def build_bbip_manifest(
    all_data: pd.DataFrame,
    *,
    season_year: int,
    pitcher_ids: Sequence[int],
    extremes_count: int,
    output_dir: Path | None = None,
    eligible_pitcher_ids: Sequence[int] | None = None,
) -> list[dict[str, object]]:
    """Return BB/IP metadata for selected pitchers (bottom/top/middle tiers).

    ``eligible_pitcher_ids`` only restricts which rows enter the BB/IP table
    (same filter as ``select_bbip_extreme_pitcher_ids``). Tier labels are always
    assigned by rank among ``pitcher_ids`` present in that table.
    """

    table = _filter_bbip_table(
        compute_bbip_by_pitcher(all_data, season_year=season_year, output_dir=output_dir),
        eligible_pitcher_ids,
    )
    tier_by_id = label_bbip_tiers_for_pitcher_ids(
        table,
        pitcher_ids,
        extremes_count=extremes_count,
    )
    selected = table.loc[table["pitcher_id"].isin({int(pid) for pid in pitcher_ids})].copy()
    manifest: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        pitcher_id = int(row["pitcher_id"])
        manifest.append(
            {
                "pitcher_id": pitcher_id,
                "player_name": row["player_name"],
                "walks": int(row["walks"]),
                "innings_pitched": float(row["innings_pitched"]),
                "bbip": float(row["bbip"]),
                "tier": tier_by_id.get(pitcher_id, "middle"),
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

    table = _filter_bbip_table(
        compute_bbip_by_pitcher(all_data, season_year=season_year, output_dir=output_dir),
        eligible_pitcher_ids,
    )
    label = "eligible pitchers" if eligible_pitcher_ids is not None else "pitchers"
    bottom, top = _extreme_pitcher_id_lists(table, count, label=label)
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
