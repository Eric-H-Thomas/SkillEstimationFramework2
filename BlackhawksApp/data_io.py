"""Centralized data loading for the Blackhawks Streamlit app.

This module is the app-facing boundary for data storage layout. If storage
paths or file formats change later, update this file and keep UI code intact.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from BlackhawksSkillEstimation.BlackhawksJEEDS import (
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
    if not root.exists():
        return []

    player_ids: list[int] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        m = _PLAYER_DIR_RE.match(child.name)
        if m:
            player_ids.append(int(m.group(1)))
    return player_ids


def get_seasons(player_id: int, data_dir: Path | str | None = None) -> list[int]:
    """Return available season tags for a player based on shots parquet files."""
    root = _resolve_data_dir(data_dir)
    data_subdir = root / f"player_{player_id}" / "data"
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
    logs_dir = root / f"player_{player_id}" / "logs"
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
