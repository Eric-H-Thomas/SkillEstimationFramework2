"""Centralized data loading for the Blackhawks Streamlit app.

This module is the app-facing boundary for data storage layout. If storage
paths or file formats change later, update this file and keep UI code intact.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from BlackhawksSkillEstimation.BlackhawksJEEDS import (
    SHOT_TYPE_GROUPS,
    load_player_data,
    load_player_data_by_games,
)
from BlackhawksSkillEstimation.plot_intermediate_estimates import (
    load_intermediate_estimates,
)

_DEFAULT_DATA_DIR = Path("Data/Hockey")
_PLAYER_DIR_RE = re.compile(r"player_(\d+)$")
_SEASON_RE = re.compile(r"^shots_(\d{8})\.parquet$")


def _resolve_data_dir(data_dir: Path | str | None = None) -> Path:
    return Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR


def get_players(data_dir: Path | str | None = None) -> list[int]:
    """Return available player IDs from the local data directory."""
    root = _resolve_data_dir(data_dir)
    players_dir = root / "players"
    if not players_dir.exists():
        return []

    player_ids: list[int] = []
    for child in sorted(players_dir.iterdir()):
        if not child.is_dir():
            continue
        m = _PLAYER_DIR_RE.match(child.name)
        if m:
            player_ids.append(int(m.group(1)))
    return player_ids


def get_seasons(player_id: int, data_dir: Path | str | None = None) -> list[int]:
    """Return available season tags for a player based on shots parquet files."""
    root = _resolve_data_dir(data_dir)
    data_subdir = root / "players" / f"player_{player_id}" / "data"
    if not data_subdir.exists():
        return []

    seasons: list[int] = []
    for p in sorted(data_subdir.glob("shots_*.parquet")):
        m = _SEASON_RE.match(p.name)
        if m:
            seasons.append(int(m.group(1)))
    return seasons


def get_intermediate_csvs(player_id: int, data_dir: Path | str | None = None) -> list[Path]:
    """Return intermediate-estimate CSVs for a player."""
    root = _resolve_data_dir(data_dir)
    logs_dir = root / "players" / f"player_{player_id}" / "logs"
    if not logs_dir.exists():
        return []
    return sorted(logs_dir.glob("intermediate_estimates*.csv"))


def load_estimates(csv_path: Path | str) -> dict[str, list[float]]:
    """Load one intermediate-estimates CSV into arrays keyed by metric name."""
    return load_intermediate_estimates(csv_path)


def load_heatmaps(
    player_id: int,
    *,
    seasons: list[int] | None = None,
    tag: str | None = None,
    data_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, dict[int, dict[str, object]]]:
    """Load shot metadata and shot maps for a player.

    Use exactly one mode:
    - `seasons=[...]` for season files (`shots_<season>.parquet`, `shot_maps_<season>.npz`)
    - `tag="..."` for game-tag files (`shots_<tag>.parquet`, `shot_maps_<tag>.npz`)
    """
    root = _resolve_data_dir(data_dir)

    if tag is not None and seasons is not None:
        raise ValueError("Provide either `tag` or `seasons`, not both.")

    if tag is not None:
        return load_player_data_by_games(player_id=player_id, tag=tag, data_dir=root)

    if seasons is None:
        seasons = get_seasons(player_id=player_id, data_dir=root)
        if not seasons:
            raise FileNotFoundError(
                f"No season parquet files found for player {player_id} under {root}."
            )

    return load_player_data(player_id=player_id, seasons=seasons, data_dir=root)


def load_shots_metadata(
    player_id: int,
    season: int,
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Load one season's shot metadata parquet for a player."""
    root = _resolve_data_dir(data_dir)
    parquet_path = root / "players" / f"player_{player_id}" / "data" / f"shots_{season}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing shots parquet: {parquet_path}")
    return pd.read_parquet(parquet_path)


def load_shot_maps_for_season(
    player_id: int,
    season: int,
    data_dir: Path | str | None = None,
) -> dict[int, dict[str, object]]:
    """Load one season's shot-map npz into event_id-keyed payloads."""
    root = _resolve_data_dir(data_dir)
    npz_path = root / "players" / f"player_{player_id}" / "data" / f"shot_maps_{season}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing shot-map npz: {npz_path}")

    data = np.load(npz_path)
    out: dict[int, dict[str, object]] = {}
    event_ids = data["event_ids"]
    for i, eid in enumerate(event_ids):
        out[int(eid)] = {
            "value_map": data["value_maps"][i].astype(np.float64),
            "net_cov": data["net_covs"][i],
            "net_coords": data["net_coords"][i],
        }
    return out


def get_heatmap_for_shot(
    shot_maps: dict[int, dict[str, object]],
    event_id: int,
) -> dict[str, object] | None:
    """Return shot-map payload for one event id (or None if missing)."""
    return shot_maps.get(int(event_id))


def get_shot_row(df: pd.DataFrame, event_id: int) -> pd.Series | None:
    """Return one shot row for event_id (or None if missing)."""
    matches = df[df["event_id"] == int(event_id)]
    if matches.empty:
        return None
    return matches.iloc[0]


def get_shot_types(df: pd.DataFrame) -> list[str]:
    """Return sorted raw shot_type values present in a dataframe."""
    if "shot_type" not in df.columns:
        return []
    vals = df["shot_type"].dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    return sorted(vals.unique().tolist())


def get_shot_group_tags() -> list[str]:
    """Return estimator shot-group tags in canonical order."""
    return list(SHOT_TYPE_GROUPS.keys())


def get_shot_group_display(shot_group: str) -> str:
    """Return user-facing display name for a shot-group tag."""
    return SHOT_TYPE_GROUPS.get(shot_group, (shot_group, set(), False))[0]


def filter_by_shot_type(df: pd.DataFrame, shot_type: str | None) -> pd.DataFrame:
    """Filter rows by one raw shot_type value (or return all when None/all)."""
    if shot_type is None or shot_type == "All":
        return df.copy()
    if "shot_type" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["shot_type"].fillna("") == shot_type].copy()


def filter_by_shot_group(df: pd.DataFrame, shot_group: str) -> pd.DataFrame:
    """Filter a shots dataframe by configured estimator shot group."""
    group_info = SHOT_TYPE_GROUPS.get(shot_group)
    if group_info is None:
        return df.iloc[0:0].copy()

    _display, allowed_types, include_null = group_info
    if "shot_type" not in df.columns:
        return df.iloc[0:0].copy()

    stype = df["shot_type"].fillna("").astype(str).str.strip().str.lower()
    mask = stype.isin(allowed_types)
    if include_null:
        mask = mask | (stype == "")
    return df[mask].copy()


def with_shot_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with 1-based shot_index preserving current row order."""
    out = df.reset_index(drop=True).copy()
    out["shot_index"] = out.index + 1
    return out


def get_shot_row_by_index(df: pd.DataFrame, shot_index: int) -> pd.Series | None:
    """Return row for a 1-based shot_index (None when out of range)."""
    if shot_index < 1 or shot_index > len(df):
        return None
    return df.iloc[int(shot_index) - 1]


def _raw_shot_to_group(raw_shot_type: str | None) -> str | None:
    """Map one raw shot_type to its configured group tag (if any)."""
    if raw_shot_type is None:
        return None
    needle = str(raw_shot_type).strip().lower()
    if needle == "":
        return None
    for group_tag, (_display, allowed, _include_null) in SHOT_TYPE_GROUPS.items():
        if needle in allowed:
            return group_tag
    return None


def get_convergence_artifact(
    player_id: int,
    season: int,
    *,
    shot_group: str | None = None,
    shot_type: str | None = None,
    data_dir: Path | str | None = None,
    suffix: str = ".png",
) -> Path | None:
    """Find a convergence artifact path for season and optional raw shot_type.

    Search order:
    1) Group-specific logs path (if raw shot_type maps to a group)
    2) Flat logs path
    """
    root = _resolve_data_dir(data_dir)
    logs_dir = root / "players" / f"player_{player_id}" / "logs"
    if not logs_dir.exists():
        return None

    season_tag = str(season)
    group_tag = shot_group if shot_group is not None else _raw_shot_to_group(shot_type)
    if group_tag is not None:
        grouped = logs_dir / group_tag / f"intermediate_estimates_{season_tag}{suffix}"
        if grouped.exists():
            return grouped

    flat = logs_dir / f"intermediate_estimates_{season_tag}{suffix}"
    if flat.exists():
        return flat
    return None
