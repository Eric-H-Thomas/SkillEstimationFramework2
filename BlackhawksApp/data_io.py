"""Centralized data loading for the Blackhawks Streamlit app.

This module is the app-facing boundary for data storage layout. If storage
paths or file formats change later, update this file and keep UI code intact.

This module also supports config export helpers used by the cluster workflow:
    sbatch run_blackhawks_config.sbatch Data/Hockey/jobs/<config>.json
"""
from __future__ import annotations

import json
import re
from datetime import datetime
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

# Blackhawks value_map coordinate extents.
_BH_Y_MIN, _BH_Y_MAX, _BH_Y_LEN = -5.0, 5.0, 120
_BH_Z_MIN, _BH_Z_MAX, _BH_Z_LEN = 0.0, 6.0, 72


def _resolve_data_dir(data_dir: Path | str | None = None) -> Path:
    return Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR


def get_player_id_text_files(data_dir: Path | str | None = None) -> list[Path]:
    """Return root-level text files that may contain player ID lists."""
    root = _resolve_data_dir(data_dir)
    if not root.exists():
        return []
    return sorted(root.glob("*.txt"))


def parse_player_ids_text(raw_text: str) -> list[int]:
    """Parse player IDs from loose text (comma/newline/space separated)."""
    if not raw_text or not raw_text.strip():
        return []
    matches = re.findall(r"\d+", raw_text)
    seen: set[int] = set()
    pids: list[int] = []
    for match in matches:
        pid = int(match)
        if pid in seen:
            continue
        seen.add(pid)
        pids.append(pid)
    return pids


def load_player_ids_from_text_file(path: Path | str) -> list[int]:
    """Load player IDs from a text file."""
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    return parse_player_ids_text(text)


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


def resolve_player_ids(
    *,
    selected_players: list[int] | None = None,
    player_file: Path | str | None = None,
    pasted_player_ids: str = "",
) -> list[int]:
    """Resolve a merged, deduplicated list of player IDs from multiple sources."""
    merged: list[int] = []
    seen: set[int] = set()

    def _add_many(values: list[int]) -> None:
        for value in values:
            pid = int(value)
            if pid in seen:
                continue
            seen.add(pid)
            merged.append(pid)

    if selected_players:
        _add_many([int(pid) for pid in selected_players])

    if player_file:
        try:
            _add_many(load_player_ids_from_text_file(player_file))
        except FileNotFoundError:
            pass

    if pasted_player_ids:
        _add_many(parse_player_ids_text(pasted_player_ids))

    return merged


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


def get_all_available_seasons(
    player_ids: list[int],
    data_dir: Path | str | None = None,
) -> list[int]:
    """Return union of all known seasons across player IDs."""
    season_set: set[int] = set()
    for player_id in player_ids:
        season_set.update(get_seasons(player_id=player_id, data_dir=data_dir))
    return sorted(season_set)


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


def filter_by_partition(
    df: pd.DataFrame,
    *,
    partition_column: str | None = None,
    partition_values: list[str] | None = None,
) -> pd.DataFrame:
    """Filter a dataframe by partition column and selected values."""
    if partition_column is None or partition_column == "":
        return df.copy()
    if partition_column not in df.columns:
        return df.iloc[0:0].copy()
    if not partition_values:
        return df.copy()

    needle = set(str(v) for v in partition_values)
    series = df[partition_column].fillna("").astype(str)
    return df[series.isin(needle)].copy()


def discover_partition_values(
    player_ids: list[int],
    seasons: list[int],
    *,
    data_dir: Path | str | None = None,
) -> dict[str, list[str]]:
    """Discover categorical partition candidates from local season files.

    This avoids hard-coding partition names while still supporting columns
    like side/partition buckets when present in local metadata.
    """
    candidates = {
        "partition",
        "partition_tag",
        "lane_partition",
        "shallow_partition",
        "shot_partition",
        "start_side",
        "side",
    }
    collected: dict[str, set[str]] = {}

    for player_id in player_ids:
        for season in seasons:
            try:
                df = load_shots_metadata(player_id=player_id, season=season, data_dir=data_dir)
            except FileNotFoundError:
                continue

            matching_cols = [c for c in df.columns if c.lower() in candidates]
            for col in matching_cols:
                values = (
                    df[col]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )
                values = values[values != ""]
                if values.empty:
                    continue
                collected.setdefault(col, set()).update(values.unique().tolist())

    return {col: sorted(vals) for col, vals in sorted(collected.items())}


def build_observation_summary(
    player_ids: list[int],
    seasons: list[int],
    shot_groups: list[str],
    *,
    data_dir: Path | str | None = None,
    partition_column: str | None = None,
    partition_values: list[str] | None = None,
) -> pd.DataFrame:
    """Build counts of observations matching filter combinations."""
    rows: list[dict[str, object]] = []
    for player_id in player_ids:
        for season in seasons:
            try:
                season_df = load_shots_metadata(player_id=player_id, season=season, data_dir=data_dir)
            except FileNotFoundError:
                for shot_group in shot_groups:
                    rows.append(
                        {
                            "player_id": player_id,
                            "season": season,
                            "shot_group": shot_group,
                            "count": 0,
                            "missing_local_data": True,
                        }
                    )
                continue

            for shot_group in shot_groups:
                filtered = filter_by_shot_group(season_df, shot_group=shot_group)
                filtered = filter_by_partition(
                    filtered,
                    partition_column=partition_column,
                    partition_values=partition_values,
                )
                rows.append(
                    {
                        "player_id": player_id,
                        "season": season,
                        "shot_group": shot_group,
                        "count": int(len(filtered)),
                        "missing_local_data": False,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=["player_id", "season", "shot_group", "count", "missing_local_data"]
        )
    return pd.DataFrame(rows)


def build_job_rows(
    summary_df: pd.DataFrame,
    *,
    min_shots_per_job: int,
) -> list[dict[str, object]]:
    """Build flattened per-job rows for cluster arrays."""
    jobs: list[dict[str, object]] = []
    if summary_df.empty:
        return jobs

    for _, row in summary_df.iterrows():
        count = int(row["count"])
        jobs.append(
            {
                "player_id": int(row["player_id"]),
                "season": int(row["season"]),
                "shot_group": str(row["shot_group"]),
                "count": count,
                "missing_local_data": bool(row.get("missing_local_data", False)),
                "eligible": bool(count >= min_shots_per_job and not bool(row.get("missing_local_data", False))),
            }
        )
    return jobs


def save_job_config(
    config: dict[str, object],
    *,
    data_dir: Path | str | None = None,
    output_subdir: str = "jobs",
    custom_filename: str | None = None,
) -> Path:
    """Persist a JSON run config under Data/Hockey/jobs by default.
    
    Parameters
    ----------
    config : dict[str, object]
        The configuration dictionary to save.
    data_dir : Path | str | None
        The root data directory (defaults to Data/Hockey).
    output_subdir : str
        The subdirectory to save to (defaults to "jobs").
    custom_filename : str | None
        Optional custom filename (without .json extension). If not provided,
        a timestamp-based filename will be generated.
    
    Returns
    -------
    Path
        The path where the config was saved.
    """
    root = _resolve_data_dir(data_dir)
    out_dir = root / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if custom_filename:
        # Ensure filename doesn't already have .json extension
        if custom_filename.endswith(".json"):
            custom_filename = custom_filename[:-5]
        path = out_dir / f"{custom_filename}.json"
    else:
        # Generate timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"bhjeeds_job_config_{timestamp}.json"
    
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return path


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


def add_post_shot_xg_column(
    df: pd.DataFrame,
    shot_maps: dict[int, dict[str, object]],
    *,
    out_column: str = "post_shot_xg",
) -> pd.DataFrame:
    """Return copy of df with sampled post-shot xG from value-map grids.

    xG is sampled at each shot's observed goal-line coordinates
    (location_y, location_z) using nearest-grid-point lookup on the
    72x120 post-shot xG map.
    """
    out = df.copy()
    out[out_column] = np.nan

    required = {"event_id", "location_y", "location_z"}
    if shot_maps is None or out.empty or not required.issubset(out.columns):
        return out

    y_vals = pd.to_numeric(out["location_y"], errors="coerce").to_numpy(dtype=np.float64)
    z_vals = pd.to_numeric(out["location_z"], errors="coerce").to_numpy(dtype=np.float64)
    eids = pd.to_numeric(out["event_id"], errors="coerce").to_numpy(dtype=np.float64)

    y_idx = np.rint((y_vals - _BH_Y_MIN) / (_BH_Y_MAX - _BH_Y_MIN) * (_BH_Y_LEN - 1)).astype(np.int64)
    z_idx = np.rint((z_vals - _BH_Z_MIN) / (_BH_Z_MAX - _BH_Z_MIN) * (_BH_Z_LEN - 1)).astype(np.int64)
    y_idx = np.clip(y_idx, 0, _BH_Y_LEN - 1)
    z_idx = np.clip(z_idx, 0, _BH_Z_LEN - 1)

    sampled: list[float] = []
    for i in range(len(out)):
        eid = eids[i]
        if np.isnan(eid):
            sampled.append(np.nan)
            continue

        payload = shot_maps.get(int(eid))
        if not payload:
            sampled.append(np.nan)
            continue

        value_map = payload.get("value_map")
        if value_map is None:
            sampled.append(np.nan)
            continue

        try:
            sampled.append(float(value_map[z_idx[i], y_idx[i]]))
        except (IndexError, TypeError, ValueError):
            sampled.append(np.nan)

    out[out_column] = sampled
    return out


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
