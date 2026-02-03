"""Public querying surface mirroring the Blackhawks analytics helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from matplotlib import transforms

from . import db

# Shot map grid dimensions (from post_shot_xg_value_maps reshaping)
# Used (along with covariance extraction helper) in get_game_shot_maps()
SHOT_MAP_HEIGHT = 120
SHOT_MAP_WIDTH = 72


def _extract_covariance_matrix(row: pd.Series) -> np.ndarray:
    """Extract 2x2 covariance matrix from database row."""
    cov_cols = ["cov_00", "cov_01", "cov_10", "cov_11"]
    return row[cov_cols].values.reshape(2, 2)


def plot_ellipse(ax, mean, cov, n_std: float = 2.0, **kwargs):
    """Plot an ellipse representing covariance of a 2D normal distribution."""

    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor="none",
        **kwargs,
    )
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean[0], mean[1])
    )
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


def shot_games(season: int) -> pd.Series:
    """Return game identifiers for which shot maps exist.

    Parameters
    ----------
    season : int
        The season identifier (e.g., ``20242025``).
    """

    query = """
        SELECT DISTINCT e.game_id_hawks
        FROM hawks_analytics.post_shot_xg_value_maps p
        JOIN public.event e ON e.event_id_hawks = p.event_id_hawks
        JOIN public.game g ON g.game_id_hawks = e.game_id_hawks
        WHERE g.season = %(season)s
    """

    return db.get_df(query, query_params={"season": season}).rename(columns=str.lower)["game_id_hawks"]


def get_games_for_seasons(seasons: list[int]) -> pd.DataFrame:
    """Return game IDs and their seasons for the given season list.

    Parameters
    ----------
    seasons : list[int]
        List of season identifiers (e.g., ``[20232024, 20242025]``).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'game_id_hawks' and 'season'.
    """
    seasons_str = ",".join(str(s) for s in seasons)
    query = f"""
        SELECT DISTINCT g.game_id_hawks, g.season
        FROM public.game g
        JOIN hawks_analytics.post_shot_xg_value_maps p
          ON p.event_id_hawks IN (
              SELECT e.event_id_hawks
              FROM public.event e
              WHERE e.game_id_hawks = g.game_id_hawks
          )
        WHERE g.season IN ({seasons_str})
        ORDER BY g.season, g.game_id_hawks
    """
    return db.get_df(query).rename(columns=str.lower)


def query_player_season_shots(
    player_id: int,
    seasons: list[int],
) -> pd.DataFrame:
    """Fetch shot-level metadata for a player across specified seasons.

    This query joins EVENT → GAME → SHOT_TRAJECTORIES to filter by season
    without requiring explicit game IDs.

    Parameters
    ----------
    player_id : int
        The player identifier.
    seasons : list[int]
        List of season identifiers (e.g., ``[20232024, 20242025]``).

    Returns
    -------
    pd.DataFrame
        One row per shot with event metadata, player position, shot location,
        and the season the shot occurred in.
    """
    seasons_str = ",".join(str(s) for s in seasons)
    query = f"""
        SELECT
            e.PLAYER_ID_HAWKS        AS player_id,
            e.GAME_ID_HAWKS          AS game_id,
            e.EVENT_ID_HAWKS         AS event_id,
            g.SEASON                 AS season,
            e.SHOT_IS_BLOCKED,
            e.SHOT_IS_CONTACT_PRESSURE,
            e.SHOT_IS_FANNED_SHOT,
            e.SHOT_IS_GOAL,
            e.SHOT_IS_HIGH_DANGER_MISSED_SHOT_RECOVERY,
            e.SHOT_IS_LOW_HIGH,
            e.SHOT_IS_ONE_TIMER,
            e.SHOT_IS_OUTSIDE,
            e.SHOT_IS_PENALTY_SHOT,
            e.SHOT_IS_QUICK_RELEASE,
            e.SHOT_IS_SCREENED,
            e.SHOT_IS_SEAM,
            e.SHOT_IS_SLOT,
            e.SHOT_IS_SPACE_PRESSURE,
            e.SHOT_IS_TIME_PRESSURE,
            e.SHOT_IS_WITH_PRESSURE,
            e.SHOT_IS_WITH_REBOUND,
            e.X_ADJ_COORD             AS start_x,
            e.Y_ADJ_COORD             AS start_y,
            st.GOALLINE_Y_MODEL       AS location_y,
            st.GOALLINE_Z_MODEL       AS location_z
        FROM HAWKS_HOCKEY.PUBLIC.EVENT AS e
        JOIN HAWKS_HOCKEY.PUBLIC.GAME AS g
          ON g.GAME_ID_HAWKS = e.GAME_ID_HAWKS
        JOIN HAWKS_HOCKEY.HAWKS_ANALYTICS.SHOT_TRAJECTORIES AS st
          ON st.EVENT_ID_HAWKS = e.EVENT_ID_HAWKS
        WHERE e.PLAYER_ID_HAWKS = %(player_id)s
          AND g.SEASON IN ({seasons_str})
          AND e.EVENT_NAME = 'shot';
    """
    return db.get_df(query, query_params={"player_id": player_id})


def get_game_shot_maps(game_id_hawks: int) -> dict[int, dict[str, object]]:
    """Return shot metadata for a given game keyed by ``event_id_hawks``."""

    query = """
            SELECT p.*
                , st.goalline_y_model
                , st.goalline_z_model
                , st.cov_00
                , st.cov_01
                , st.cov_10
                , st.cov_11
                , st.percent_on_net
            FROM hawks_analytics.post_shot_xg_value_maps p
            JOIN public.event e ON e.event_id_hawks = p.event_id_hawks
            JOIN hawks_analytics.shot_trajectories st ON st.event_id_hawks = p.event_id_hawks
            WHERE e.game_id_hawks = %(game_id_hawks)s
            ORDER BY p.event_id_hawks ASC, location_y DESC, location_z ASC
            ;
            """
    df = db.get_df(query, query_params={"game_id_hawks": game_id_hawks}).rename(
        columns=str.lower
    )

    shot_maps: dict[int, dict[str, object]] = {}
    for event_id_hawks in df["event_id_hawks"].unique():
        shot_df = df[df["event_id_hawks"] == event_id_hawks]

        shot_data: dict[str, object] = {}
        shot_data["df"] = shot_df
        shot_data["value_map"] = np.flip(
            shot_df["post_shot_xg"].values.reshape(SHOT_MAP_HEIGHT, SHOT_MAP_WIDTH).T, axis=1
        )
        shot_data["net_cov"] = _extract_covariance_matrix(shot_df.iloc[0])
        shot_data["net_coords"] = shot_df.iloc[0][
            ["goalline_y_model", "goalline_z_model"]
        ].values

        shot_maps[event_id_hawks] = shot_data
    return shot_maps


def query_player_game_info(player_id: int, game_ids: list[int]) -> pd.DataFrame:
    """Fetch shot-level metadata for a player across specific games.
    
    Note: The actual shot location (where the puck crossed the goal line) comes
    from SHOT_TRAJECTORIES via get_game_shot_maps(), not from this query.
    This query returns one row per shot with event metadata and player position.
    """

    game_ids_str = ",".join(str(g) for g in game_ids)
    query = f"""
        SELECT
            e.PLAYER_ID_HAWKS        AS player_id,
            e.GAME_ID_HAWKS          AS game_id,
            e.EVENT_ID_HAWKS         AS event_id,
            e.SHOT_IS_BLOCKED,
            e.SHOT_IS_CONTACT_PRESSURE,
            e.SHOT_IS_FANNED_SHOT,
            e.SHOT_IS_GOAL,
            e.SHOT_IS_HIGH_DANGER_MISSED_SHOT_RECOVERY,
            e.SHOT_IS_LOW_HIGH,
            e.SHOT_IS_ONE_TIMER,
            e.SHOT_IS_OUTSIDE,
            e.SHOT_IS_PENALTY_SHOT,
            e.SHOT_IS_QUICK_RELEASE,
            e.SHOT_IS_SCREENED,
            e.SHOT_IS_SEAM,
            e.SHOT_IS_SLOT,
            e.SHOT_IS_SPACE_PRESSURE,
            e.SHOT_IS_TIME_PRESSURE,
            e.SHOT_IS_WITH_PRESSURE,
            e.SHOT_IS_WITH_REBOUND,
            e.X_ADJ_COORD             AS start_x,
            e.Y_ADJ_COORD             AS start_y,
            st.GOALLINE_Y_MODEL       AS location_y,
            st.GOALLINE_Z_MODEL       AS location_z
        FROM HAWKS_HOCKEY.PUBLIC.EVENT AS e
        JOIN HAWKS_HOCKEY.HAWKS_ANALYTICS.SHOT_TRAJECTORIES AS st
          ON st.EVENT_ID_HAWKS = e.EVENT_ID_HAWKS
        WHERE e.PLAYER_ID_HAWKS = %(player_id)s
          AND e.GAME_ID_HAWKS IN ({game_ids_str})
          AND e.EVENT_NAME = 'shot';
    """
    return db.get_df(query, query_params={"player_id": player_id})
