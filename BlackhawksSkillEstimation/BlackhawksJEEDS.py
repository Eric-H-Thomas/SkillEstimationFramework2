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
import pickle
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, zoom

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
# Default used to be 25, so consider that for research computation.
DEFAULT_NUM_PLANNING_SKILLS = 100


def save_intermediate_estimates_csv(
    skill_log: list[dict[str, object]],
    player_id: int,
    output_dir: Path | str,
    tag: str = "",
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
        Base directory (e.g., "Data/Hockey/player_950160").
    tag : str
        Optional tag for filename (e.g., "20242025" for season or "2games_test").
    
    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    output_dir = Path(output_dir)
    logs_dir = output_dir / "logs"
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
        ])
        for row in skill_log:
            writer.writerow([
                row["shot_count"],
                row["ees"],
                row["map_execution_skill"],
                row["eps"],
                row["map_rationality"],
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
        base_ev = shot_map_data["value_map"]

        # Resize xG map from (72, 120) = (Z, Y) to (40, 60) = (len(Z), len(Y))
        # queries.py produces axis-0=Z(72), axis-1=Y(120) after .T and flip.
        # getAngularHeatmap expects shape (len(Z), len(Y)) from meshgrid(Y, Z).
        base_ev = zoom(base_ev, (40 / base_ev.shape[0], 60 / base_ev.shape[1]), order=1)

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

    Creates pickle files in ``output_dir/player_{player_id}/`` with:
    - ``shots_{tag}.pkl``: DataFrame of shots for the specified games
    - ``shot_maps_{tag}.pkl``: Dict mapping event_id -> shot_map_data

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
    player_dir = output_dir / f"player_{player_id}"
    data_dir_path = player_dir / "data"
    data_dir_path.mkdir(parents=True, exist_ok=True)

    shots_path = data_dir_path / f"shots_{tag}.pkl"
    maps_path = data_dir_path / f"shot_maps_{tag}.pkl"

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

    print(f"  Found {len(df)} shots across {len(game_ids)} games. Fetching shot maps...")
    try:
        shot_maps = get_games_shot_maps_batch(game_ids, player_id=player_id)
    except Exception as e:
        print(f"  Warning: Could not fetch shot maps: {e}")
        shot_maps = {}

    # Save to pickle
    with open(shots_path, "wb") as f:
        pickle.dump(df, f)
    with open(maps_path, "wb") as f:
        pickle.dump(shot_maps, f)

    print(f"  Saved: {shots_path.name}, {maps_path.name}")
    return {"shots": shots_path, "shot_maps": maps_path}


def load_player_data_by_games(
    player_id: int,
    tag: str = "games",
    data_dir: Path | str = Path("Data/Hockey"),
) -> tuple[pd.DataFrame, dict[int, dict[str, object]]]:
    """Load previously saved player data from disk (saved by game IDs).

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
        If pickle files are missing.
    """
    data_dir = Path(data_dir)
    player_dir = data_dir / f"player_{player_id}"
    data_subdir = player_dir / "data"

    shots_path = data_subdir / f"shots_{tag}.pkl"
    maps_path = data_subdir / f"shot_maps_{tag}.pkl"

    # Fall back to legacy layout (files directly in player_dir)
    if not shots_path.exists():
        legacy_shots = player_dir / f"shots_{tag}.pkl"
        if legacy_shots.exists():
            shots_path = legacy_shots
        else:
            raise FileNotFoundError(f"Missing shots file: {shots_path}")
    if not maps_path.exists():
        legacy_maps = player_dir / f"shot_maps_{tag}.pkl"
        if legacy_maps.exists():
            maps_path = legacy_maps
        else:
            raise FileNotFoundError(f"Missing shot maps file: {maps_path}")

    with open(shots_path, "rb") as f:
        df = pickle.load(f)
    with open(maps_path, "rb") as f:
        shot_maps = pickle.load(f)

    return df, shot_maps


def save_player_data(
    player_id: int,
    seasons: list[int],
    output_dir: Path | str = Path("Data/Hockey"),
    overwrite: bool = False,
) -> dict[int, dict[str, Path]]:
    """Fetch and save player shot data + shot maps to disk for offline use.

    Creates pickle files in ``output_dir/player_{player_id}/data/`` with:
    - ``shots_{season}.pkl``: DataFrame of shots for that season
    - ``shot_maps_{season}.pkl``: Dict mapping event_id -> shot_map_data

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
    player_dir = output_dir / f"player_{player_id}"
    data_dir_path = player_dir / "data"
    data_dir_path.mkdir(parents=True, exist_ok=True)

    saved_files: dict[int, dict[str, Path]] = {}

    for season in seasons:
        shots_path = data_dir_path / f"shots_{season}.pkl"
        maps_path = data_dir_path / f"shot_maps_{season}.pkl"

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
        game_ids = df["game_id"].unique().tolist()

        print(f"  Found {len(df)} shots across {len(game_ids)} games. Fetching shot maps...")
        try:
            shot_maps = get_games_shot_maps_batch(game_ids, player_id=player_id)
        except Exception as e:
            print(f"  Warning: Could not fetch shot maps: {e}")
            shot_maps = {}

        # Save to pickle
        with open(shots_path, "wb") as f:
            pickle.dump(df, f)
        with open(maps_path, "wb") as f:
            pickle.dump(shot_maps, f)

        print(f"  Saved: {shots_path.name}, {maps_path.name}")
        saved_files[season] = {"shots": shots_path, "shot_maps": maps_path}

    return saved_files


def load_player_data(
    player_id: int,
    seasons: list[int],
    data_dir: Path | str = Path("Data/Hockey"),
) -> tuple[pd.DataFrame, dict[int, dict[str, object]]]:
    """Load previously saved player data from disk.

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
        If pickle files for any requested season are missing.
    """
    data_dir = Path(data_dir)
    player_dir = data_dir / f"player_{player_id}"
    data_subdir = player_dir / "data"

    all_dfs: list[pd.DataFrame] = []
    all_shot_maps: dict[int, dict[str, object]] = {}

    for season in seasons:
        shots_path = data_subdir / f"shots_{season}.pkl"
        maps_path = data_subdir / f"shot_maps_{season}.pkl"

        # Fall back to legacy layout (files directly in player_dir)
        if not shots_path.exists():
            legacy_shots = player_dir / f"shots_{season}.pkl"
            if legacy_shots.exists():
                shots_path = legacy_shots
            else:
                raise FileNotFoundError(f"Missing shots file: {shots_path}")
        if not maps_path.exists():
            legacy_maps = player_dir / f"shot_maps_{season}.pkl"
            if legacy_maps.exists():
                maps_path = legacy_maps
            else:
                raise FileNotFoundError(f"Missing shot maps file: {maps_path}")

        with open(shots_path, "rb") as f:
            df = pickle.load(f)
        with open(maps_path, "rb") as f:
            shot_maps = pickle.load(f)

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
) -> dict[str, object]:
    """Internal helper to run JEEDS on a DataFrame of shots.
    
    Returns a result dict with execution_skill, rationality, num_shots, and optionally skill_log.
    
    Timing logs are written to ``player_dir/times/estimators/``.
    Intermediate CSVs are written to ``player_dir/logs/``.
    
    If preloaded_shot_maps is provided, uses those instead of fetching from DB.
    If save_intermediate_csv is True, saves the skill_log to a CSV file.
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

    jeeds_inputs = transform_shots_for_jeeds(
        df,
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
                skill_log.append({
                    "shot_count": idx + 1,  # 1-indexed
                    "ees": float(ees[-1]),  # Expected Execution Skill
                    "map_execution_skill": float(map_xskill[-1]),
                    "eps": float(eps[-1]),  # Expected Rationality
                    "map_rationality": float(map_rationality[-1]),
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

    result: dict[str, object] = {
        # MAP estimates (primary)
        "execution_skill": float(map_xskill_estimates[-1]),
        "rationality": float(map_rationality_estimates[-1]),
        # EES/EPS estimates (expected values under posterior)
        "ees": float(ees_estimates[-1]) if ees_estimates else None,
        "eps": float(eps_estimates[-1]) if eps_estimates else None,
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
        )
        result["csv_path"] = str(csv_path)

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
        If True, save intermediate estimates to a CSV file under data_dir/player_{id}/logs/.
    data_dir : Path | str
        Base directory for player data. Default is "Data/Hockey".
        Each player's outputs (timing logs, CSVs, plots) live under
        ``data_dir/player_{id}/{data,logs,plots,times}/``.
    confirm : bool
        If True (default), prompt for confirmation before running estimation.
        Set to False for batch/automated runs.
    offline_data : tuple[pd.DataFrame, dict] | None
        Pre-loaded data from :func:`load_player_data`. When provided, skips all
        DB queries and uses the loaded data directly. Format: (shots_df, shot_maps).

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
    """
    # Validate inputs
    if game_ids is None and seasons is None and offline_data is None:
        raise ValueError("Must provide 'game_ids', 'seasons', or 'offline_data'.")

    candidate_skills = list(
        candidate_skills or np.linspace(0.004, np.pi / 4, DEFAULT_NUM_EXECUTION_SKILLS)
    )
    data_dir = Path(data_dir)
    player_data_dir = data_dir / f"player_{player_id}"

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
            return _run_jeeds_estimation(
                df=df,
                game_ids=all_game_ids,
                player_id=player_id,
                candidate_skills=candidate_skills,
                num_planning_skills=num_planning_skills,
                player_dir=player_data_dir,
                rng_seed=rng_seed,
                return_intermediate_estimates=return_intermediate_estimates,
                preloaded_shot_maps=preloaded_shot_maps,
                save_intermediate_csv=save_intermediate_csv,
                csv_tag="aggregate",
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
        return _run_jeeds_estimation(
            df=df,
            game_ids=all_game_ids,
            player_id=player_id,
            candidate_skills=candidate_skills,
            num_planning_skills=num_planning_skills,
            player_dir=player_data_dir,
            rng_seed=rng_seed,
            return_intermediate_estimates=return_intermediate_estimates,
            save_intermediate_csv=save_intermediate_csv,
            csv_tag="aggregate",
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
    # TODO: last exited recursive code check here

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
