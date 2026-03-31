"""Run JEEDS on Blackhawks shot data to estimate player execution skill.

This module stitches together the Snowflake-backed Blackhawks API with the
framework's JEEDS implementation (``JointMethodQRE``).  The workflow:

1. Pull per-shot metadata for a given player across selected games via
   :func:`BlackhawksAPI.query_player_game_info`.
2. Fetch precomputed reward surfaces (post-shot xG probability grids) for each
   shot via :func:`BlackhawksAPI.get_game_shot_maps`.
3. Convert those raw SQL rows and reward surfaces into angular heatmaps and
   covariances expected by the JEEDS estimator for the hockey domain.
4. Feed each observed shot into JEEDS and return the final MAP execution-skill
   and rationality estimates.

This integration uses the Blackhawks analytics' precomputed shot maps, which
incorporate detailed models of shooting position, angle, goalie positioning,
and other factors to estimate expected value for each possible aim direction.

Example
-------
from BlackhawksSkillEstimation.BlackhawksJEEDS import estimate_player_skill
estimates = estimate_player_skill(player_id=950160, game_ids=[44604, 270247])
print(f"Execution skill: {estimates['execution_skill']:.4f} rad (lower=better)")
print(f"Rationality: {estimates['rationality']:.2f} (higher=better, experimental)")
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from BlackhawksAPI import (
    get_game_shot_maps,
    get_games_shot_maps_batch,
    query_player_game_info,
    query_player_season_shots,
)
from Environments.Hockey import getAngularHeatmapsPerPlayer as angular_heatmaps
from Estimators.joint import JointMethodQRE


# Net center position used for proximity filtering (NHL standard).
_NET_CENTER = np.array([89.0, 0.0])

# Blackhawks xG grid: Y: [-5, 5] (120 pts), Z: [0, 6] (72 pts)
# Passed to getAngularHeatmap as grid_y and grid_z to use full native extents.
_BH_Y = np.linspace(-5.0, 5.0, 120)
_BH_Z = np.linspace(0.0, 6.0, 72)

# Minimum distance (in feet) from the net center below which a shot is
# discarded.  The Blackhawks analytics team cannot accurately model shots
# taken within this radius, so including them would add noise.
MIN_DISTANCE_FROM_NET_FT = 10.0

# Default number of execution skill hypotheses for JEEDS estimation.
# Trade-off between resolution and computational cost:
#   10 hypotheses: ~0.08 rad spacing, fast, good for testing
#   20 hypotheses: ~0.04 rad spacing, balanced for most uses
#   30 hypotheses: ~0.025 rad spacing, matches production experiments
#   50 hypotheses: ~0.015 rad spacing, high precision for research
DEFAULT_NUM_EXECUTION_SKILLS = 50

# Default number of planning/rationality skill hypotheses for JEEDS estimation.
# Higher values give finer rationality resolution but increase compute cost.
# These are currently between the range of 0-3.5 in logspace.
DEFAULT_NUM_PLANNING_SKILLS = 100

# ---------------------------------------------------------------------------
# Shot-type grouping definitions
# ---------------------------------------------------------------------------
# Each entry maps tag -> (display_name, allowed_shot_types, include_null).
# ``include_null`` controls whether shots with NULL/missing shot_type are included.
SHOT_TYPE_GROUPS: dict[str, tuple[str, set[str], bool]] = {
    "wristshot_snapshot": ("Wristshot/Snapshot", {"wristshot", "snapshot"}, True),
    "backhand": ("Backhand", {"backhand"}, False),
    "slapshot": ("Slapshot", {"slapshot"}, False),
    "deke": ("Deke", {"forehandbackhand", "backhandforehand"}, False),
}

# Convenience tuple of all defined group tags, in canonical order.
DEFAULT_SHOT_GROUPS: tuple[str, ...] = ("wristshot_snapshot", "backhand", "slapshot", "deke")


def _compute_aggregate_season_tag(seasons: Sequence[int]) -> str:
    """Generate a canonical season tag for multi-season aggregated estimates.
    
    Returns a deterministic, filename-safe tag that identifies which seasons
    were included in the aggregate. Adjacent seasons use a range format
    (e.g. 's20222023to20242025'), non-adjacent use explicit list format
    (e.g. 's20222023__20242025').
    
    Parameters
    ----------
    seasons : Sequence[int]
        List of season codes (e.g., [20232024, 20242025, 20222023]).
    
    Returns
    -------
    str
        Canonical aggregate season tag (e.g., 's20222023to20242025').
    """
    # Normalize: sort and dedupe
    unique_sorted = sorted(set(seasons))
    
    if not unique_sorted:
        raise ValueError("At least one season must be provided")
    
    if len(unique_sorted) == 1:
        # Single season in aggregate mode is unusual but supported
        return f"s{unique_sorted[0]}"
    
    # Check adjacency based on season boundaries:
    # 20232024 is adjacent to 20242025 because 2024 == 2024.
    def _is_adjacent_pair(current: int, nxt: int) -> bool:
        cur_str = str(current)
        nxt_str = str(nxt)
        if len(cur_str) != 8 or len(nxt_str) != 8 or not cur_str.isdigit() or not nxt_str.isdigit():
            return False
        return int(cur_str[4:]) == int(nxt_str[:4])

    is_adjacent = all(
        _is_adjacent_pair(unique_sorted[i], unique_sorted[i + 1])
        for i in range(len(unique_sorted) - 1)
    )
    
    if is_adjacent:
        # Range form: s20222023to20242025
        return f"s{unique_sorted[0]}to{unique_sorted[-1]}"
    else:
        # Explicit list form: s20222023__20242025__20262027...
        return "s" + "__".join(str(season) for season in unique_sorted)


def save_intermediate_estimates_csv(
    skill_log: list[dict[str, object]],
    player_id: int,
    output_dir: Path | str,
    tag: str = "",
    shot_group: str = "",
) -> Path:
    """Save intermediate estimates to a CSV file.
    
    Parameters
    ----------
    skill_log : list[dict]
        List of dicts with keys: shot_count, ees (expected execution skill),
        map_execution_skill, eps (expected rationality), map_rationality
    player_id : int
        Player ID for filename.
    output_dir : Path | str
        Base directory (e.g., "Data/Hockey/players/player_950160").
    tag : str
        Optional tag for filename (e.g., "20242025" for season or "2games_test").
    shot_group : str
        Shot-type group tag (e.g., ``"wristshot_snapshot"``, ``"backhand"``).  When
        non-empty the CSV is written into a subdirectory ``logs/<shot_group>/`` instead
        of ``logs/``.
    
    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    output_dir = Path(output_dir)
    logs_dir = output_dir / "logs"
    if shot_group:
        logs_dir = logs_dir / shot_group
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    tag_str = f"_{tag}" if tag else ""
    csv_path = logs_dir / f"intermediate_estimates{tag_str}.csv"
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "shot_count",
            "expected_execution_skill",
            "map_execution_skill", 
            "expected_rationality",
            "map_rationality",
            "log10_expected_rationality",
            "log10_map_rationality",
        ])
        for row in skill_log:
            eps_val = row["eps"]
            map_rat_val = row["map_rationality"]
            writer.writerow([
                row["shot_count"],
                row["ees"],
                row["map_execution_skill"],
                eps_val,
                map_rat_val,
                np.log10(eps_val) if eps_val and eps_val > 0 else None,
                np.log10(map_rat_val) if map_rat_val and map_rat_val > 0 else None,
            ])
    
    return csv_path


@dataclass
class SimpleHockeySpaces:
    """Minimal space object exposing the hooks JEEDS expects for hockey.

    The production framework builds a rich :class:`SpacesHockey` structure that
    caches precomputed EV surfaces, covariances, and target grids.  For this
    integration we only need a subset of those attributes:

    ``possibleTargets``
        Flattened list of (y, z) candidates used when evaluating shot PDFs.
    ``delta``
        Grid spacing (dy, dz); maintained for parity with the framework API.
    ``allCovs``
        Covariance matrices keyed by ``get_key([x, x], 0.0)``.
    """

    y_grid: np.ndarray
    z_grid: np.ndarray
    candidate_execution_skills: Sequence[float]
    possible_targets: np.ndarray | None = None
    possibleTargets: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        dy = np.diff(self.y_grid).mean() if len(self.y_grid) > 1 else 1.0
        dz = np.diff(self.z_grid).mean() if len(self.z_grid) > 1 else 1.0
        self.delta = (float(dy), float(dz))

        if self.possible_targets is None:
            self.possibleTargets = np.array(
                [(y, z) for y in self.y_grid for z in self.z_grid], dtype=float
            )
        else:
            self.possibleTargets = np.array(self.possible_targets, dtype=float)

        self.all_covs: dict[str, np.ndarray] = {}
        for skill in self.candidate_execution_skills:
            key = self.get_key([skill, skill], r=0.0)
            self.all_covs[key] = self._covariance_from_skill(skill)

    # Compare to getCovMatrix in hockey.py. Assume rho=0, so we simplify to np.diag
    @staticmethod
    def _covariance_from_skill(skill: float) -> np.ndarray:
        """Return a 2D diagonal covariance matrix for execution error in angular space.
        
        The input skill is the standard deviation (in radians) of execution error
        in both direction and elevation. Variance = skill², so:
        - Higher skill → larger variance → worse execution (looser spread)
        - Lower skill → smaller variance → better execution (tighter cluster)
        """
        variance = max(skill, 1e-6) ** 2
        return np.diag([variance, variance])

    '''
    Examples of how skill (in radians) translates to variance:
    skill: 0.004; variance: 0.000016 (elite execution, tight cluster)
    skill: 0.02;  variance: 0.0004   (strong execution, tight spread)
    skill: 0.1;   variance: 0.01     (moderate execution)
    skill: 0.3;   variance: 0.09     (loose execution, wide spread)
    skill: 0.6;   variance: 0.36     (poor execution, very wide spread)
    skill: 0.785 (π/4); variance: 0.616 (terrible execution, nearly random)
    '''

    @staticmethod
    def get_key(info: Sequence[float], r: float) -> str:
        return "|".join(map(str, info)) + f"|{r}"


@dataclass
class JEEDSInputs:
    spaces_per_shot: list[SimpleHockeySpaces]
    actions: list[list[float]]
    info_rows: list[dict[str, object]]
    skipped_proximity: int = 0


# ============================================================================
# ARCHITECTURAL NOTE: Spaces as Reward Maps/Reward Surfaces
# ============================================================================
#
# A "Space" in this framework is a REWARD MAP—a lookup table that tells the
# estimator: "For this shot location, here are the possible aims and how
# valuable each one is."
#
# **SpacesHockey (production framework):**
#   - Pre-computes reward surfaces for EVERY POSSIBLE SHOT LOCATION
#   - Uses expensive scipy.signal.convolve2d operations with skill-based kernels
#   - Stores all results in memory or on disk
#   - Reuses the same Space object across many shots in an experiment
#   - Used by: Estimators when processing simulation data
#
# **SimpleHockeySpaces (this module):**
#   - Uses Blackhawks precomputed reward surfaces (value_map) for each shot
#   - Retrieved from hawks_analytics.post_shot_xg_value_maps via get_game_shot_maps()
#   - Each shot's value_map is a 120x72 grid of post-shot xG probabilities
#   - Creates a new Space instance per shot, discards it after use
#   - No expensive local precomputation needed; data comes from Snowflake
#   - Used by: BlackhawksJEEDS when processing real player data from Snowflake
#
# Both expose the same interface (possibleTargets, all_covs, delta, get_key())
# so JointMethodQRE treats them identically during skill estimation.


def transform_shots_for_jeeds(
    df,
    shot_maps: dict[int, dict[str, object]],
    candidate_skills: Sequence[float],
) -> JEEDSInputs:
    """Convert SQL rows into the structures JEEDS expects.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`query_player_game_info`.
    shot_maps:
        Dictionary of shot metadata keyed by event_id_hawks, returned by
        :func:`get_game_shot_maps`. Contains 'value_map', 'net_cov', and
        'net_coords' for each shot.
    candidate_skills:
        Execution-skill hypotheses passed to JEEDS.
    """

    df = df.rename(columns=str.lower)

    skipped_proximity = 0

    info_rows: list[dict[str, object]] = []
    actions: list[list[float]] = []
    spaces_per_shot: list[SimpleHockeySpaces] = []

    for _, row in df.iterrows():
        event_id = int(row["event_id"])
        
        # Skip shots without precomputed Blackhawks reward surfaces
        if event_id not in shot_maps:
            continue

        player_location = np.array([float(row["start_x"]), float(row["start_y"])])
        executed_action = np.array([float(row["location_y"]), float(row["location_z"])])

        # ---- 10-ft net proximity filter ----
        # The Blackhawks analytics team cannot accurately gather data for
        # shots taken within a 10-ft radius of the net center.  Discard
        # them before they enter the estimation pipeline.
        dist_to_net = np.linalg.norm(player_location - _NET_CENTER)
        if dist_to_net < MIN_DISTANCE_FROM_NET_FT:
            skipped_proximity += 1
            continue

        shot_map_data = shot_maps[event_id]
        base_ev = shot_map_data["value_map"]  # native (72×120) over Y∈[-5,5] Z∈[0,6]

        # Convert Blackhawks Cartesian reward surface to angular coordinates.
        (
            dirs,
            elevations,
            _,
            grid_targets_angular,
            _,
            _,
            _,
            grid_utilities_computed,
            executed_action_angular,
            skip,
            _,
        ) = angular_heatmaps.getAngularHeatmap(
            base_ev,
            player_location,
            executed_action,
            grid_y=_BH_Y,
            grid_z=_BH_Z,
        )

        if skip:
            continue

        spaces = SimpleHockeySpaces(
            np.array(dirs),
            np.array(elevations),
            candidate_skills,
            possible_targets=grid_targets_angular.reshape(-1, 2),
        )

        entry = {"evsPerXskill": {}, "maxEVPerXskill": {}, "focalActions": []}

        # Compute skill-dependent EV surface smoothing.
        # Lower xskill (tight execution) → sharp EV surface (player hits what they aim at)
        # Higher xskill (loose execution) → blurred EV surface (expected value smears across potential hit locations)
        # NOTE: We generate a blurred version of the xG surface for each xskill hypothesis.
        dir_bin_size = (dirs[-1] - dirs[0]) / (len(dirs) - 1) if len(dirs) > 1 else 0.01
        elev_bin_size = (elevations[-1] - elevations[0]) / (len(elevations) - 1) if len(elevations) > 1 else 0.01
        avg_bin_size = (dir_bin_size + elev_bin_size) / 2

        for skill in candidate_skills:
            key = spaces.get_key([skill, skill], r=0.0)
            # sigma in grid bins: skill (rad) / avg_bin_size (rad/bin) = bins
            # Clamp to avoid extreme smoothing or no smoothing.
            sigma = max(min(skill / avg_bin_size, 1.0), 1e-3)
            evs = gaussian_filter(grid_utilities_computed, sigma=sigma)

            entry["evsPerXskill"][key] = evs
            entry["maxEVPerXskill"][key] = float(np.max(evs))

            # Find most rational target location for this xskill's blurred xG surface
            best_idx = int(np.argmax(evs))
            iy, iz = np.unravel_index(best_idx, (len(dirs), len(elevations)))
            entry["focalActions"].append(
                [float(grid_targets_angular[iy, iz, 0]), float(grid_targets_angular[iy, iz, 1])]
            )

        info_rows.append(entry)
        actions.append([float(executed_action_angular[0]), float(executed_action_angular[1])])
        spaces_per_shot.append(spaces)

    if skipped_proximity:
        print(f"  Filtered {skipped_proximity} shot(s) within {MIN_DISTANCE_FROM_NET_FT}ft of the net.")

    return JEEDSInputs(
        spaces_per_shot=spaces_per_shot,
        actions=actions,
        info_rows=info_rows,
        skipped_proximity=skipped_proximity,
    )


def ensure_player_directories(player_dir: Path) -> None:
    """Create the standard per-player subdirectory layout.

    Creates ``data/``, ``logs/``, ``plots/``, and ``times/estimators/``
    beneath *player_dir*.
    """
    (player_dir / "data").mkdir(parents=True, exist_ok=True)
    (player_dir / "logs").mkdir(parents=True, exist_ok=True)
    (player_dir / "plots").mkdir(parents=True, exist_ok=True)
    (player_dir / "times" / "estimators").mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=256)
def _get_game_shot_maps_cached(game_id: int) -> dict[int, dict[str, object]]:
    """LRU-cached wrapper for get_game_shot_maps to avoid redundant DB queries."""
    return get_game_shot_maps(game_id)


def clear_shot_maps_cache() -> None:
    """Clear the shot maps cache (useful between estimation runs)."""
    _get_game_shot_maps_cached.cache_clear()


def _prune_shot_maps_to_event_ids(
    shot_maps: dict[int, dict[str, object]],
    event_ids: Sequence[object],
) -> tuple[dict[int, dict[str, object]], int]:
    """Keep only shot-map entries whose event_id appears in ``event_ids``."""
    if not shot_maps:
        return {}, 0

    keep_ids = {int(eid) for eid in event_ids}
    pruned = {int(eid): payload for eid, payload in shot_maps.items() if int(eid) in keep_ids}
    removed = len(shot_maps) - len(pruned)
    return pruned, removed


def _filter_estimable_shots_for_persistence(
    df: pd.DataFrame,
    shot_maps: dict[int, dict[str, object]],
) -> tuple[pd.DataFrame, dict[int, dict[str, object]], dict[str, int]]:
    """Keep only shots that would be usable by JEEDS in practice.

    Rejection rules mirror estimator-time exclusions:
    - missing shot map for event_id
    - within 10ft net radius
    - angular transform skip
    """
    df_lc = df.rename(columns=str.lower)
    stats = {
        "total": len(df_lc),
        "kept": 0,
        "rejected_missing_map": 0,
        "rejected_proximity": 0,
        "rejected_angular_skip": 0,
        "maps_pruned": 0,
    }

    if df_lc.empty:
        return df_lc, {}, stats

    keep_mask: list[bool] = []
    for _, row in df_lc.iterrows():
        event_id = int(row["event_id"])
        if event_id not in shot_maps:
            stats["rejected_missing_map"] += 1
            keep_mask.append(False)
            continue

        player_location = np.array([float(row["start_x"]), float(row["start_y"])])
        executed_action = np.array([float(row["location_y"]), float(row["location_z"])])

        dist_to_net = np.linalg.norm(player_location - _NET_CENTER)
        if dist_to_net < MIN_DISTANCE_FROM_NET_FT:
            stats["rejected_proximity"] += 1
            keep_mask.append(False)
            continue

        base_ev = shot_maps[event_id]["value_map"]
        try:
            angular_out = angular_heatmaps.getAngularHeatmap(
                base_ev,
                player_location,
                executed_action,
                grid_y=_BH_Y,
                grid_z=_BH_Z,
            )
            skip = bool(angular_out[9])
        except Exception:
            skip = True

        if skip:
            stats["rejected_angular_skip"] += 1
            keep_mask.append(False)
            continue

        keep_mask.append(True)

    filtered_df = df_lc.loc[keep_mask].reset_index(drop=True)
    filtered_maps, maps_pruned = _prune_shot_maps_to_event_ids(shot_maps, filtered_df["event_id"].tolist())
    stats["maps_pruned"] = maps_pruned
    stats["kept"] = len(filtered_df)
    return filtered_df, filtered_maps, stats


def _log_persistence_filter_stats(stats: dict[str, int]) -> None:
    """Print a compact summary of save-time usable-shot filtering."""
    rejected_total = (
        stats["rejected_missing_map"]
        + stats["rejected_proximity"]
        + stats["rejected_angular_skip"]
    )
    print(
        "  Pre-save usable-shot filter: "
        f"kept {stats['kept']}/{stats['total']} shots; "
        f"rejected missing_map={stats['rejected_missing_map']}, "
        f"proximity<{MIN_DISTANCE_FROM_NET_FT}ft={stats['rejected_proximity']}, "
        f"angular_skip={stats['rejected_angular_skip']}"
    )
    if stats["maps_pruned"]:
        print(f"  Pruned {stats['maps_pruned']} shot-map entries not present in persisted shots.")
    if rejected_total == 0:
        print("  No additional save-time rejections beyond queried shots.")


# =============================================================================
# OFFLINE DATA PERSISTENCE: SHOT MAPS SERIALIZATION
# =============================================================================


def _save_shot_maps_npz(
    shot_maps: dict[int, dict[str, object]], 
    path: Path | str,
) -> None:
    """Serialize shot_maps dict to compressed NPZ format.
    
    Stacks multiple shot_map dicts (each with value_map, net_cov, net_coords)
    into 3D numpy arrays, converts value_map to float32 for space efficiency,
    and saves using savez_compressed (zlib).
    
    Parameters
    ----------
    shot_maps : dict[int, dict[str, object]]
        Mapping of event_id_hawks -> shot data containing 
        'value_map' (72x120 array), 'net_cov' (2x2), 'net_coords' (2,).
    path : Path | str
        Output .npz file path.
    """
    if not shot_maps:
        np.savez_compressed(str(path), event_ids=np.array([], dtype=np.int64))
        return
    
    event_ids = sorted(shot_maps.keys())
    value_maps = np.array(
        [shot_maps[eid]["value_map"].astype(np.float32) for eid in event_ids],
        dtype=np.float32
    )
    net_covs = np.array(
        [shot_maps[eid]["net_cov"] for eid in event_ids],
        dtype=np.float64
    )
    net_coords = np.array(
        [shot_maps[eid]["net_coords"] for eid in event_ids],
        dtype=np.float64
    )
    
    np.savez_compressed(
        str(path),
        event_ids=np.array(event_ids, dtype=np.int64),
        value_maps=value_maps,
        net_covs=net_covs,
        net_coords=net_coords,
    )


def _load_shot_maps_npz(
    path: Path | str,
) -> dict[int, dict[str, object]]:
    """Deserialize shot_maps dict from NPZ file.
    
    Reconstructs the nested dict structure from stacked numpy arrays,
    converting event_ids to int and value_maps back to float64.
    
    Parameters
    ----------
    path : Path | str
        Input .npz file path.
    
    Returns
    -------
    dict[int, dict[str, object]]
        Mapping of event_id_hawks -> {value_map, net_cov, net_coords}.
    """
    data = np.load(str(path))
    event_ids = data["event_ids"]
    
    if len(event_ids) == 0:
        return {}
    
    shot_maps = {}
    for i, eid in enumerate(event_ids):
        shot_maps[int(eid)] = {
            "value_map": data["value_maps"][i].astype(np.float64),
            "net_cov": data["net_covs"][i],
            "net_coords": data["net_coords"][i],
        }
    return shot_maps


# =============================================================================
# OFFLINE DATA PERSISTENCE
# =============================================================================


def save_player_data_by_games(
    player_id: int,
    game_ids: list[int],
    output_dir: Path | str = Path("Data/Hockey"),
    overwrite: bool = False,
    tag: str = "games",
) -> dict[str, Path]:
    """Fetch and save player shot data + shot maps for specific games.

    Creates files in ``output_dir/players/player_{player_id}/data/`` with:
    - ``shots_{tag}.parquet``: DataFrame of shots (Apache Parquet format)
    - ``shot_maps_{tag}.npz``: Compressed shot maps (Numpy savez format)

    Shot maps are stored space-efficiently with value_map arrays cast to float32.
    This is a lighter-weight alternative to save_player_data() when you only
    need a few games (e.g., for testing the pipeline on limited hardware).

    Parameters
    ----------
    player_id : int
        The player identifier.
    game_ids : list[int]
        List of game identifiers to fetch data for.
    output_dir : Path | str
        Base directory for saved data (default: ``Data/Hockey``).
    overwrite : bool
        If False (default), skip if files already exist.
        If True, overwrite existing files.
    tag : str
        Tag to use in the filename (default: "games"). Use something
        descriptive like "2games_test" or a hash of game_ids.

    Returns
    -------
    dict[str, Path]
        {"shots": path, "shot_maps": path} for saved files.
    """
    output_dir = Path(output_dir)
    player_dir = output_dir / "players" / f"player_{player_id}"
    data_dir_path = player_dir / "data"
    data_dir_path.mkdir(parents=True, exist_ok=True)

    shots_path = data_dir_path / f"shots_{tag}.parquet"
    maps_path = data_dir_path / f"shot_maps_{tag}.npz"

    # Check for existing files
    if not overwrite and shots_path.exists() and maps_path.exists():
        print(f"Files for {tag} already exist. Skipping. Use overwrite=True to replace.")
        return {"shots": shots_path, "shot_maps": maps_path}

    print(f"Fetching data for player {player_id}, games {game_ids}...")

    # Fetch shots for these games
    df = query_player_game_info(player_id=player_id, game_ids=game_ids)
    df = df.rename(columns=str.lower)

    if df.empty:
        print(f"  No shots found for games {game_ids}.")
        return {}

    filtered_game_ids = df["game_id"].dropna().astype(int).unique().tolist()
    print(
        f"  Found {len(df)} queried shots across {len(filtered_game_ids)} games. "
        "Fetching shot maps..."
    )
    try:
        shot_maps = get_games_shot_maps_batch(filtered_game_ids, player_id=player_id)
    except Exception as e:
        print(f"  Warning: Could not fetch shot maps: {e}")
        shot_maps = {}

    df, shot_maps, stats = _filter_estimable_shots_for_persistence(df, shot_maps)
    _log_persistence_filter_stats(stats)
    if df.empty:
        print("  All shots were rejected by pre-save usable-shot filtering.")
        return {}
    print(f"  Persisting {len(df)} shots and {len(shot_maps)} shot maps.")

    # Save to parquet and npz
    df.to_parquet(shots_path, engine="pyarrow", compression="snappy")
    _save_shot_maps_npz(shot_maps, maps_path)

    print(f"  Saved: {shots_path.name}, {maps_path.name}")
    return {"shots": shots_path, "shot_maps": maps_path}


def load_player_data_by_games(
    player_id: int,
    tag: str = "games",
    data_dir: Path | str = Path("Data/Hockey"),
) -> tuple[pd.DataFrame, dict[int, dict[str, object]]]:
    """Load previously saved player data from disk (saved by game IDs).

    Loads data from parquet and npz files created by save_player_data_by_games().

    Parameters
    ----------
    player_id : int
        The player identifier.
    tag : str
        The tag used when saving (default: "games").
    data_dir : Path | str
        Base directory containing saved data (default: ``Data/Hockey``).

    Returns
    -------
    tuple[pd.DataFrame, dict[int, dict[str, object]]]
        (shots_df, shot_maps_dict) ready for estimation.

    Raises
    ------
    FileNotFoundError
        If parquet/npz files are missing.
    """
    data_dir = Path(data_dir)
    player_dir = data_dir / "players" / f"player_{player_id}"
    data_subdir = player_dir / "data"

    shots_path = data_subdir / f"shots_{tag}.parquet"
    maps_path = data_subdir / f"shot_maps_{tag}.npz"

    if not shots_path.exists():
        raise FileNotFoundError(f"Missing shots file: {shots_path}")
    if not maps_path.exists():
        raise FileNotFoundError(f"Missing shot maps file: {maps_path}")

    df = pd.read_parquet(shots_path)
    shot_maps = _load_shot_maps_npz(maps_path)

    return df, shot_maps


def save_player_data(
    player_id: int,
    seasons: list[int],
    output_dir: Path | str = Path("Data/Hockey"),
    overwrite: bool = False,
) -> dict[int, dict[str, Path]]:
    """Fetch and save player shot data + shot maps to disk for offline use.

    Creates files in ``output_dir/players/player_{player_id}/data/`` with:
    - ``shots_{season}.parquet``: DataFrame of shots (Apache Parquet format)
    - ``shot_maps_{season}.npz``: Compressed shot maps (Numpy savez format)

    Shot maps are stored space-efficiently with value_map arrays cast to float32.

    Parameters
    ----------
    player_id : int
        The player identifier.
    seasons : list[int]
        List of season identifiers (e.g., ``[20232024, 20242025]``).
    output_dir : Path | str
        Base directory for saved data (default: ``Data/Hockey``).
    overwrite : bool
        If False (default), skip seasons where files already exist.
        If True, overwrite existing files.

    Returns
    -------
    dict[int, dict[str, Path]]
        Mapping of season -> {"shots": path, "shot_maps": path} for saved files.
    """
    output_dir = Path(output_dir)
    player_dir = output_dir / "players" / f"player_{player_id}"
    data_dir_path = player_dir / "data"
    data_dir_path.mkdir(parents=True, exist_ok=True)

    saved_files: dict[int, dict[str, Path]] = {}

    for season in seasons:
        shots_path = data_dir_path / f"shots_{season}.parquet"
        maps_path = data_dir_path / f"shot_maps_{season}.npz"

        # Check for existing files
        if not overwrite and shots_path.exists() and maps_path.exists():
            print(f"Warning: Files for season {season} already exist. Skipping. Use overwrite=True to replace.")
            saved_files[season] = {"shots": shots_path, "shot_maps": maps_path}
            continue

        print(f"Fetching data for player {player_id}, season {season}...")

        # Fetch shots for this season
        df = query_player_season_shots(player_id=player_id, seasons=[season])
        df = df.rename(columns=str.lower)

        if df.empty:
            print(f"  No shots found for season {season}. Skipping.")
            continue

        # Fetch shot maps for all games in this season (single batched query)
        game_ids = df["game_id"].dropna().astype(int).unique().tolist()

        print(f"  Found {len(df)} shots across {len(game_ids)} games. Fetching shot maps...")
        try:
            shot_maps = get_games_shot_maps_batch(game_ids, player_id=player_id)
        except Exception as e:
            print(f"  Warning: Could not fetch shot maps: {e}")
            shot_maps = {}

        df, shot_maps, stats = _filter_estimable_shots_for_persistence(df, shot_maps)
        _log_persistence_filter_stats(stats)
        if df.empty:
            print(f"  All shots were rejected by pre-save usable-shot filtering for season {season}.")
            continue
        print(f"  Persisting {len(df)} shots and {len(shot_maps)} shot maps.")

        # Save to parquet and npz
        df.to_parquet(shots_path, engine="pyarrow", compression="snappy")
        _save_shot_maps_npz(shot_maps, maps_path)

        print(f"  Saved: {shots_path.name}, {maps_path.name}")
        saved_files[season] = {"shots": shots_path, "shot_maps": maps_path}

    return saved_files


def load_player_data(
    player_id: int,
    seasons: list[int],
    data_dir: Path | str = Path("Data/Hockey"),
) -> tuple[pd.DataFrame, dict[int, dict[str, object]]]:
    """Load previously saved player data from disk.

    Loads data from parquet and npz files created by save_player_data().

    Parameters
    ----------
    player_id : int
        The player identifier.
    seasons : list[int]
        List of season identifiers to load.
    data_dir : Path | str
        Base directory containing saved data (default: ``Data/Hockey``).

    Returns
    -------
    tuple[pd.DataFrame, dict[int, dict[str, object]]]
        (shots_df, shot_maps_dict) ready for estimation.
        The DataFrame includes all shots across requested seasons.
        The shot_maps dict maps event_id -> shot_map_data.

    Raises
    ------
    FileNotFoundError
        If parquet/npz files for any requested season are missing.
    """
    data_dir = Path(data_dir)
    player_dir = data_dir / "players" / f"player_{player_id}"
    data_subdir = player_dir / "data"

    all_dfs: list[pd.DataFrame] = []
    all_shot_maps: dict[int, dict[str, object]] = {}

    for season in seasons:
        shots_path = data_subdir / f"shots_{season}.parquet"
        maps_path = data_subdir / f"shot_maps_{season}.npz"

        if not shots_path.exists():
            raise FileNotFoundError(f"Missing shots file: {shots_path}")
        if not maps_path.exists():
            raise FileNotFoundError(f"Missing shot maps file: {maps_path}")

        df = pd.read_parquet(shots_path)
        shot_maps = _load_shot_maps_npz(maps_path)

        all_dfs.append(df)
        all_shot_maps.update(shot_maps)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df, all_shot_maps


def _run_jeeds_estimation(
    df,
    game_ids: Sequence[int],
    player_id: int,
    candidate_skills: list[float],
    num_planning_skills: int,
    player_dir: Path,
    rng_seed: int | None,
    return_intermediate_estimates: bool,
    tag_suffix: str = "",
    preloaded_shot_maps: dict[int, dict[str, object]] | None = None,
    save_intermediate_csv: bool = False,
    csv_tag: str = "",
    shot_group: str = "",
) -> dict[str, object]:
    """Internal helper to run JEEDS on a DataFrame of shots.
    
    Returns a result dict with execution_skill, rationality, num_shots, and optionally skill_log.
    
    Timing logs are written to ``player_dir/times/estimators/``.
    Intermediate CSVs are written to ``player_dir/logs/`` (or ``logs/<shot_group>/`` when set).
    
    If preloaded_shot_maps is provided, uses those instead of fetching from DB.
    If save_intermediate_csv is True, saves the skill_log to a CSV file.
    
    Parameters
    ----------
    shot_group : str
        Shot-type group tag (e.g., ``"wristshot_snapshot"``, ``"backhand"``).  When
        non-empty, the shot-type filter is driven by :data:`SHOT_TYPE_GROUPS` and output
        CSVs are written into a ``logs/<shot_group>/`` subdirectory.
    """
    if df.empty:
        return {
            "execution_skill": None,
            "rationality": None,
            "num_shots": 0,
            "status": "no_data",
            "warning": "No shot data for this estimation.",
        }

    # Use preloaded shot maps if provided, otherwise fetch from DB
    if preloaded_shot_maps is not None:
        shot_maps = preloaded_shot_maps
    else:
        # Fetch Blackhawks precomputed reward surfaces for each game (with caching)
        shot_maps = {}
        for game_id in game_ids:
            try:
                game_shot_maps = _get_game_shot_maps_cached(game_id)
                shot_maps.update(game_shot_maps)
            except Exception as e:
                print(f"Warning: Could not fetch shot maps for game {game_id}: {e}")

    # ----------------- Shot-type filtering (estimator-only) -----------------
    # Determine the allowed shot types and whether NULLs are included.
    # When *shot_group* is provided, look up SHOT_TYPE_GROUPS; otherwise fall
    # back to the legacy default (wristshot + snapshot + NULL).
    if shot_group:
        if shot_group not in SHOT_TYPE_GROUPS:
            raise ValueError(
                f"Unknown shot_group '{shot_group}'. "
                f"Valid groups: {', '.join(DEFAULT_SHOT_GROUPS)}"
            )
        group_display, allowed_types, include_null = SHOT_TYPE_GROUPS[shot_group]
    else:
        group_display, allowed_types, include_null = "Wristshot/Snapshot", {"wristshot", "snapshot"}, True

    # Work with lower-cased column names for robustness
    df_lc = df.rename(columns=str.lower)

    if "shot_type" in df_lc.columns:
        # Normalize to lowercase strings (preserve NaNs)
        shot_series = df_lc["shot_type"].where(pd.notna(df_lc["shot_type"]))
        shot_lower = shot_series.astype(str).str.lower()
        if include_null:
            mask = shot_series.isna() | shot_lower.isin(allowed_types)
        else:
            mask = shot_lower.isin(allowed_types)
        filtered_df = df_lc[mask]
        num_before = len(df_lc)
        num_after = len(filtered_df)
        allowed_str = ",".join(sorted(allowed_types))
        null_str = ",NULL" if include_null else ""
        if num_after != num_before:
            print(f"  Shot-type filter [{group_display}]: kept {num_after}/{num_before} shots (allowed: {allowed_str}{null_str})")
    else:
        # `shot_type` not present in the shots DataFrame
        print(
            "  WARNING: 'shot_type' column not found in shots DataFrame; proceeding with all shots."
        )
        filtered_df = df_lc

    # Use the filtered DataFrame going forward
    jeeds_inputs = transform_shots_for_jeeds(
        filtered_df,
        shot_maps=shot_maps,
        candidate_skills=candidate_skills,
    )
    if not jeeds_inputs.actions:
        return {
            "execution_skill": None,
            "rationality": None,
            "num_shots": 0,
            "status": "no_usable_shots",
            "warning": "No usable shot data remained after angular conversion.",
            "skipped_proximity": jeeds_inputs.skipped_proximity,
        }

    estimator = JointMethodQRE(
        list(candidate_skills), num_planning_skills, "hockey-multi",
        times_base_dir=str(player_dir),
    )
    ensure_player_directories(player_dir)

    rng = np.random.default_rng(rng_seed)
    tag = f"Player_{player_id}{tag_suffix}"

    skill_log: list[dict[str, object]] = []
    # Keys for retrieving estimates from JEEDS
    # JEEDS uses "xSkills" for execution skill and "pSkills" for planning skill (rationality)
    base_key = f"{estimator.method_type}-{{}}-{estimator.num_execution_skills}-{estimator.num_rationality_levels}"
    map_xskill_key = f"{base_key.format('MAP')}-xSkills"
    map_rationality_key = f"{base_key.format('MAP')}-pSkills"
    ees_key = f"{base_key.format('EES')}-xSkills"
    eps_key = f"{base_key.format('EES')}-pSkills"

    for idx, (spaces, info, action) in enumerate(
        zip(jeeds_inputs.spaces_per_shot, jeeds_inputs.info_rows, jeeds_inputs.actions)
    ):
        estimator.add_observation(
            rng,
            spaces,
            state=None,
            action=action,
            resultsFolder=str(player_dir),
            tag=tag,
            infoPerRow=info,
            s=str(idx),
        )

        if return_intermediate_estimates or save_intermediate_csv:
            results = estimator.get_results()
            map_xskill = results.get(map_xskill_key, [])
            map_rationality = results.get(map_rationality_key, [])
            ees = results.get(ees_key, [])
            eps = results.get(eps_key, [])
            
            if map_xskill and map_rationality and ees and eps:
                eps_val = float(eps[-1])
                map_rat_val = float(map_rationality[-1])
                skill_log.append({
                    "shot_count": idx + 1,  # 1-indexed
                    "ees": float(ees[-1]),  # Expected Execution Skill
                    "map_execution_skill": float(map_xskill[-1]),
                    "eps": eps_val,  # Expected Rationality
                    "map_rationality": map_rat_val,
                    "log10_eps": np.log10(eps_val) if eps_val > 0 else None,
                    "log10_map_rationality": np.log10(map_rat_val) if map_rat_val > 0 else None,
                })

    results = estimator.get_results()
    map_xskill_estimates = results.get(map_xskill_key, [])
    map_rationality_estimates = results.get(map_rationality_key, [])
    ees_estimates = results.get(ees_key, [])
    eps_estimates = results.get(eps_key, [])
    
    if not map_xskill_estimates or not map_rationality_estimates:
        return {
            "execution_skill": None,
            "rationality": None,
            "num_shots": len(jeeds_inputs.actions),
            "status": "estimation_failed",
            "warning": "JEEDS returned no MAP estimates.",
        }

    final_rationality = float(map_rationality_estimates[-1])
    final_eps = float(eps_estimates[-1]) if eps_estimates else None

    result: dict[str, object] = {
        # MAP estimates (primary)
        "execution_skill": float(map_xskill_estimates[-1]),
        "rationality": final_rationality,
        "log10_rationality": np.log10(final_rationality) if final_rationality > 0 else None,
        # EES/EPS estimates (expected values under posterior)
        "ees": float(ees_estimates[-1]) if ees_estimates else None,
        "eps": final_eps,
        "log10_eps": np.log10(final_eps) if final_eps and final_eps > 0 else None,
        "num_shots": len(jeeds_inputs.actions),
        "status": "success",
    }
    
    if return_intermediate_estimates or save_intermediate_csv:
        result["skill_log"] = skill_log
    
    # Save CSV if requested
    if save_intermediate_csv and skill_log:
        csv_path = save_intermediate_estimates_csv(
            skill_log=skill_log,
            player_id=player_id,
            output_dir=player_dir,
            tag=csv_tag,
            shot_group=shot_group,
        )
        result["csv_path"] = str(csv_path)

    if shot_group:
        result["shot_group"] = shot_group

    return result


def estimate_player_skill(
    player_id: int,
    game_ids: Sequence[int] | None = None,
    *,
    seasons: Sequence[int] | None = None,
    per_season: bool = True,
    candidate_skills: Sequence[float] | None = None,
    num_planning_skills: int = DEFAULT_NUM_PLANNING_SKILLS,
    rng_seed: int | None = 0,
    return_intermediate_estimates: bool = False,
    save_intermediate_csv: bool = False,
    data_dir: Path | str = Path("Data/Hockey"),
    confirm: bool = True,
    offline_data: tuple[pd.DataFrame, dict[int, dict[str, object]]] | None = None,
    shot_group: str = "",
    shot_groups: Sequence[str] | None = None,
) -> dict[str, object]:
    """Return JEEDS MAP and EES estimates of execution skill and rationality for a player.

    Supports three modes of operation:
    1. **Game-based**: Pass ``game_ids`` to estimate across specific games.
    2. **Season-based**: Pass ``seasons`` to auto-discover games and estimate.
       When ``per_season=True`` (default), returns separate estimates per season.
    3. **Offline**: Pass ``offline_data`` from :func:`load_player_data` to skip DB queries.

    Parameters
    ----------
    player_id : int
        The player identifier.
    game_ids : Sequence[int] | None
        Explicit list of game IDs to include. If provided, ``seasons`` is ignored.
    seasons : Sequence[int] | None
        List of season identifiers (e.g., ``[20232024, 20242025]``).
        Used to auto-discover games when ``game_ids`` is not provided.
    per_season : bool
        If True (default), return separate estimates for each season.
        If False, aggregate all shots across seasons into one estimate.
    candidate_skills : Sequence[float] | None
        Execution-skill hypotheses for JEEDS (defaults to 50 values in [0.004, π/4]).
    num_planning_skills : int
        Number of planning-skill hypotheses passed to JEEDS.
    rng_seed : int | None
        Seed for the numpy random generator used during estimation.
    return_intermediate_estimates : bool
        If True, return a 'skill_log' key with all 4 estimates after each shot.
    save_intermediate_csv : bool
        If True, save intermediate estimates to a CSV file under data_dir/players/player_{id}/logs/.
    data_dir : Path | str
        Base directory for player data. Default is "Data/Hockey".
        Each player's outputs (timing logs, CSVs, plots) live under
        ``data_dir/players/player_{id}/{data,logs,plots,times}/``.
    confirm : bool
        If True (default), prompt for confirmation before running estimation.
        Set to False for batch/automated runs.
    offline_data : tuple[pd.DataFrame, dict] | None
        Pre-loaded data from :func:`load_player_data`. When provided, skips all
        DB queries and uses the loaded data directly. Format: (shots_df, shot_maps).
    shot_group : str
        Single shot-type group tag (e.g., ``"wristshot_snapshot"``, ``"backhand"``).  Drives the
        shot-type filter and routes CSVs into ``logs/<shot_group>/``.
        When empty (default), uses the legacy wristshot/snapshot/NULL filter.
    shot_groups : Sequence[str] | None
        Convenience parameter to run estimation for *multiple* shot-type groups
        in a single call.  When provided, ``shot_group`` is ignored and the
        function loops over each group, returning results keyed by group tag
        under ``'per_group_results'``.

    Returns
    -------
    dict
        When ``per_season=False`` or using ``game_ids``:
            - 'execution_skill': MAP estimate of xskill in radians (lower is better).
            - 'rationality': MAP estimate of decision-making optimality (higher is better).
            - 'ees_execution_skill': Expected value estimate of xskill.
            - 'ees_rationality': Expected value estimate of rationality.
            - 'num_shots': Number of shots used.
            - 'skill_log' (optional): Intermediate estimates per shot.
            - 'csv_path' (optional): Path to saved CSV if save_intermediate_csv=True.
        
        When ``per_season=True`` and using ``seasons``:
            - 'per_season_results': Dict mapping season -> result dict.
            - Each season result contains all skill estimates plus num_shots.

        When ``shot_groups`` is provided:
            - 'per_group_results': Dict mapping group tag -> single-group result.
            - Each single-group result has the structure described above.
    """
    # ------------------------------------------------------------------
    # Multi-group convenience: loop over groups, collect results
    # ------------------------------------------------------------------
    if shot_groups is not None:
        per_group_results: dict[str, dict[str, object]] = {}
        for grp in shot_groups:
            display_name = SHOT_TYPE_GROUPS[grp][0] if grp in SHOT_TYPE_GROUPS else grp
            print(f"\n--- Shot group: {display_name} ({grp}) ---")
            per_group_results[grp] = estimate_player_skill(
                player_id=player_id,
                game_ids=game_ids,
                seasons=seasons,
                per_season=per_season,
                candidate_skills=candidate_skills,
                num_planning_skills=num_planning_skills,
                rng_seed=rng_seed,
                return_intermediate_estimates=return_intermediate_estimates,
                save_intermediate_csv=save_intermediate_csv,
                data_dir=data_dir,
                confirm=False,  # already confirmed or batch
                offline_data=offline_data,
                shot_group=grp,
            )
        return {
            "player_id": player_id,
            "shot_groups": list(shot_groups),
            "per_group_results": per_group_results,
        }

    # Validate inputs
    if game_ids is None and seasons is None and offline_data is None:
        raise ValueError("Must provide 'game_ids', 'seasons', or 'offline_data'.")

    candidate_skills = list(
        candidate_skills or np.linspace(0.004, np.pi / 4, DEFAULT_NUM_EXECUTION_SKILLS)
    )
    data_dir = Path(data_dir)
    player_data_dir = data_dir / "players" / f"player_{player_id}"

    # Mode 0: Offline data (pre-loaded from disk)
    if offline_data is not None:
        df, preloaded_shot_maps = offline_data
        df = df.rename(columns=str.lower) if not df.empty else df
        
        if df.empty:
            raise RuntimeError("Offline data contains no shots.")
        
        if confirm:
            num_shots = len(df)
            print(f"\n{num_shots} shots loaded from offline data for player {player_id}.")
            response = input("Proceed with estimation? [y/n]: ").strip().lower()
            if response != "y":
                print("Estimation cancelled.")
                return {"status": "cancelled", "num_shots": num_shots}

        # Check if we should do per-season (only if seasons column exists)
        if per_season and "season" in df.columns:
            per_season_results: dict[int, dict[str, object]] = {}
            
            for season in sorted(df["season"].unique()):
                season_df = df[df["season"] == season]
                season_game_ids = season_df["game_id"].unique().tolist()
                
                season_result = _run_jeeds_estimation(
                    df=season_df,
                    game_ids=season_game_ids,
                    player_id=player_id,
                    candidate_skills=candidate_skills,
                    num_planning_skills=num_planning_skills,
                    player_dir=player_data_dir,
                    rng_seed=rng_seed,
                    return_intermediate_estimates=return_intermediate_estimates,
                    tag_suffix=f"_S{season}",
                    preloaded_shot_maps=preloaded_shot_maps,
                    save_intermediate_csv=save_intermediate_csv,
                    csv_tag=str(season),
                    shot_group=shot_group,
                )
                season_result["season"] = int(season)
                per_season_results[int(season)] = season_result

            return {
                "player_id": player_id,
                "seasons": sorted(df["season"].unique().tolist()),
                "per_season_results": per_season_results,
            }
        else:
            # Aggregate mode
            all_game_ids = df["game_id"].unique().tolist()
            seasons_in_data = sorted(df["season"].unique().tolist()) if "season" in df.columns else []
            aggregate_tag = _compute_aggregate_season_tag(seasons_in_data) if seasons_in_data else "aggregate"
            return _run_jeeds_estimation(
                df=df,
                game_ids=all_game_ids,
                player_id=player_id,
                candidate_skills=candidate_skills,
                num_planning_skills=num_planning_skills,
                player_dir=player_data_dir,
                rng_seed=rng_seed,
                return_intermediate_estimates=return_intermediate_estimates,
                tag_suffix=f"_agg_{aggregate_tag}",
                preloaded_shot_maps=preloaded_shot_maps,
                save_intermediate_csv=save_intermediate_csv,
                csv_tag=aggregate_tag,
                shot_group=shot_group,
            )

    # Mode 1: Explicit game_ids (original behavior)
    if game_ids is not None:
        df = query_player_game_info(player_id=player_id, game_ids=list(game_ids))
        if df.empty:
            raise RuntimeError("No shot data returned for the requested player/games.")
        
        if confirm:
            num_shots = len(df)
            print(f"\n{num_shots} shots found for player {player_id}.")
            response = input("Proceed with estimation? [y/n]: ").strip().lower()
            if response != "y":
                print("Estimation cancelled.")
                return {"status": "cancelled", "num_shots": num_shots}
        
        return _run_jeeds_estimation(
            df=df,
            game_ids=game_ids,
            player_id=player_id,
            candidate_skills=candidate_skills,
            num_planning_skills=num_planning_skills,
            player_dir=player_data_dir,
            rng_seed=rng_seed,
            return_intermediate_estimates=return_intermediate_estimates,
            save_intermediate_csv=save_intermediate_csv,
            csv_tag="games",
            shot_group=shot_group,
        )

    # Mode 2: Season-based discovery
    seasons_list = list(seasons)
    df = query_player_season_shots(player_id=player_id, seasons=seasons_list)
    df = df.rename(columns=str.lower)
    
    if df.empty:
        raise RuntimeError(
            f"No shot data returned for player {player_id} in seasons {seasons_list}."
        )

    # Get all game IDs we'll need for shot maps
    all_game_ids = df["game_id"].unique().tolist()

    if confirm:
        num_shots = len(df)
        seasons_str = ", ".join(str(s) for s in seasons_list)
        print(f"\n{num_shots} shots found for player {player_id} across seasons [{seasons_str}].")
        response = input("Proceed with estimation? [y/n]: ").strip().lower()
        if response != "y":
            print("Estimation cancelled.")
            return {"status": "cancelled", "num_shots": num_shots, "seasons": seasons_list}

    if not per_season:
        # Aggregate mode: single estimate across all seasons
        seasons_in_data = sorted(df["season"].unique().tolist())
        aggregate_tag = _compute_aggregate_season_tag(seasons_in_data)
        return _run_jeeds_estimation(
            df=df,
            game_ids=all_game_ids,
            player_id=player_id,
            candidate_skills=candidate_skills,
            num_planning_skills=num_planning_skills,
            player_dir=player_data_dir,
            rng_seed=rng_seed,
            return_intermediate_estimates=return_intermediate_estimates,
            tag_suffix=f"_agg_{aggregate_tag}",
            save_intermediate_csv=save_intermediate_csv,
            csv_tag=aggregate_tag,
            shot_group=shot_group,
        )

    # Per-season mode: separate estimate for each season
    per_season_results: dict[int, dict[str, object]] = {}
    
    for season in sorted(df["season"].unique()):
        season_df = df[df["season"] == season]
        season_game_ids = season_df["game_id"].unique().tolist()
        
        season_result = _run_jeeds_estimation(
            df=season_df,
            game_ids=season_game_ids,
            player_id=player_id,
            candidate_skills=candidate_skills,
            num_planning_skills=num_planning_skills,
            player_dir=player_data_dir,
            rng_seed=rng_seed,
            return_intermediate_estimates=return_intermediate_estimates,
            tag_suffix=f"_S{season}",
            save_intermediate_csv=save_intermediate_csv,
            csv_tag=str(season),
            shot_group=shot_group,
        )
        season_result["season"] = int(season)
        per_season_results[int(season)] = season_result

    return {
        "player_id": player_id,
        "seasons": seasons_list,
        "per_season_results": per_season_results,
    }

# TODO: add per-season support for multiple players
# Currently only works with game ids. Should be simple fix
def estimate_multiple_players(
    player_ids: Sequence[int],
    game_ids: Sequence[int],
    *,
    candidate_skills: Sequence[float] | None = None,
    num_planning_skills: int = DEFAULT_NUM_PLANNING_SKILLS,
    rng_seed: int | None = 0,
    capture_skill_logs: bool = False,
    data_dir: Path | str = Path("Data/Hockey"),
    shot_group: str = "",
    shot_groups: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Estimate execution skill and rationality for multiple players.

    Parameters
    ----------
    player_ids : Sequence[int]
        List of player identifiers to estimate.
    game_ids : Sequence[int]
        List of game identifiers to include in estimation for all players.
    capture_skill_logs : bool
        If True, capture intermediate MAP estimates after each shot for each player.
        Default is False.
    
    Other parameters are passed to :func:`estimate_player_skill`.

    Returns
    -------
    list[dict]
        List of result dictionaries, one per player, each containing:
        - 'player_id': The player identifier.
        - 'execution_skill': MAP estimate of xskill in radians.
        - 'rationality': MAP estimate of decision-making optimality.
        - 'num_shots': Number of shots used in estimation.
        - 'skill_log' (if capture_skill_logs=True): Intermediate estimates per shot.
        - 'status': 'success' or 'error'.
        - 'error' (if status='error'): Error message.
    """
    results: list[dict[str, object]] = []

    for player_id in player_ids:
        try:
            player_result = estimate_player_skill(
                player_id=player_id,
                game_ids=game_ids,
                candidate_skills=candidate_skills,
                num_planning_skills=num_planning_skills,
                rng_seed=rng_seed,
                return_intermediate_estimates=capture_skill_logs,
                data_dir=data_dir,
                confirm=False,  # Disable confirmation for batch runs
                shot_group=shot_group,
                shot_groups=shot_groups,
            )
            player_result["player_id"] = player_id
            player_result["status"] = "success"
            results.append(player_result)
        except Exception as e:
            results.append({
                "player_id": player_id,
                "status": "error",
                "error": str(e),
            })

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JEEDS on Blackhawks shots.")
    parser.add_argument(
        "player_ids",
        type=int,
        nargs="+",
        help="One or more player identifiers to estimate.",
    )
    parser.add_argument(
        "game_ids",
        type=int,
        nargs="+",
        help="One or more game identifiers to include in the estimation run.",
    )
    parser.add_argument(
        "--candidate-skills",
        type=float,
        nargs="+",
        default=None,
        help=f"Optional execution-skill grid for JEEDS in radians (defaults to {DEFAULT_NUM_EXECUTION_SKILLS} values between 0.004 and π/4).",
    )
    parser.add_argument(
        "--num-planning-skills",
        type=int,
        default=DEFAULT_NUM_PLANNING_SKILLS,
        help=f"Number of planning-skill hypotheses passed to JEEDS (defaults to {DEFAULT_NUM_PLANNING_SKILLS}).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Data/Hockey"),
        help="Base directory for player data (default: Data/Hockey).",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=0,
        help="Seed for the numpy random generator used during estimation.",
    )
    parser.add_argument(
        "--capture-logs",
        action="store_true",
        help="If set, capture intermediate MAP estimates after each shot.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    
    # Support both single and multiple players
    player_ids = args.player_ids if isinstance(args.player_ids, list) else [args.player_ids]
    
    results = estimate_multiple_players(
        player_ids=player_ids,
        game_ids=args.game_ids,
        candidate_skills=args.candidate_skills,
        num_planning_skills=args.num_planning_skills,
        rng_seed=args.rng_seed,
        capture_skill_logs=args.capture_logs,
        data_dir=args.data_dir,
    )
    
    # Print results summary
    print("\n" + "=" * 80)
    print("JEEDS Multi-Player Skill Estimation Results")
    print("=" * 80)
    
    for result in results:
        player_id = result.get("player_id")
        status = result.get("status")
        
        if status == "success":
            print(f"\nPlayer {player_id}:")
            print(f"  Execution Skill: {result['execution_skill']:.4f} rad (lower is better)")
            print(f"  Rationality:     {result['rationality']:.2f} (higher is better, EXPERIMENTAL)")
            print(f"  Shots Used:      {result['num_shots']}")
            
            if "skill_log" in result and result["skill_log"]:
                print(f"  Tracked {len(result['skill_log'])} intermediate estimates")
        else:
            print(f"\nPlayer {player_id}: ERROR")
            print(f"  {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 80)
    print(f"Completed: {sum(1 for r in results if r['status'] == 'success')}/{len(results)} players")
    print("=" * 80)


if __name__ == "__main__":
    main()
