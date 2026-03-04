"""Blackhawks-specific plotting adapter.

Bridges Blackhawks offline data (pickled shot maps, JEEDS CSVs) with the
framework's existing hockey visualization functions so that angular heatmaps,
rink diagrams, and convergence plots can be generated without a live
experiment folder.

Delegates to
------------
- ``Environments.Hockey.getAngularHeatmapsPerPlayer.getAngularHeatmap``
  for Cartesian-to-angular coordinate transforms.
- ``Environments.Hockey.makePlotsAngularHeatmaps.drawRink``
  for rink diagram rendering.
- ``BlackhawksSkillEstimation.plot_intermediate_estimates``
  for convergence plots.

Typical usage
-------------
::

    from BlackhawksSkillEstimation.blackhawks_plots import (
        plot_shot_angular_heatmap,
        plot_shot_rink,
        plot_player_convergence,
    )

    # One shot's angular heatmap (needs its value_map and player location)
    plot_shot_angular_heatmap(value_map, player_location, executed_action,
                              save_path="Data/Hockey/player_950160/plots/angular/shot_42.png")

    # Rink diagram with scattered shots
    plot_shot_rink(player_locations, executed_actions,
                    save_path="Data/Hockey/player_950160/plots/rink/all_shots.png")

    # Convergence (wraps plot_intermediate_estimates)
    plot_player_convergence(csv_path="Data/Hockey/player_950160/logs/intermediate_estimates_20242025.csv")
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from scipy.ndimage import zoom


from BlackhawksSkillEstimation.player_cache import lookup_player

# ---------------------------------------------------------------------------
# Framework imports
# ---------------------------------------------------------------------------
from Environments.Hockey.getAngularHeatmapsPerPlayer import getAngularHeatmap
from BlackhawksSkillEstimation.plot_intermediate_estimates import (
    plot_intermediate_estimates,
    plot_all_intermediate_for_player,
    plot_comparison,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Rink geometry (feet, NHL standard)
_GOAL_LINE_X = 89
_LEFT_POST = np.array([_GOAL_LINE_X, -3])
_RIGHT_POST = np.array([_GOAL_LINE_X,  3])
_NET_CENTER = np.array([89.0, 0.0])

# Net proximity filter (shots within 10ft of net are excluded)
_MIN_DISTANCE_FROM_NET_FT = 10.0

# Default base directory for player data
_DEFAULT_DATA_DIR = Path("Data/Hockey")

# Blackhawks value-map native shape (from Snowflake queries.py)
_BH_VALUE_MAP_Z = 72
_BH_VALUE_MAP_Y = 120

# SpacesHockey grid dimensions (what getAngularHeatmap expects by default)
_SPACES_NUM_Y = 60
_SPACES_NUM_Z = 40


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_custom_cmap(base: str = "gist_rainbow", brightness: float = 0.4):
    """Build a brighter custom colormap for heatmap scatter plots."""
    n = plt.cm.jet.N
    cmap = (1.0 - brightness) * plt.get_cmap(base)(np.linspace(0.0, 1.0, n)) + brightness * np.ones((n, 4))
    return ListedColormap(cmap)


def _resize_value_map(value_map: np.ndarray) -> np.ndarray:
    """Downsample a Blackhawks value map to SpacesHockey grid dimensions.

    The raw maps from Snowflake are (72, 120) = (Z, Y).  The angular
    transform in ``getAngularHeatmap`` expects (40, 60) matching the
    SpacesHockey grid.
    """
    return zoom(
        value_map,
        (_SPACES_NUM_Z / value_map.shape[0], _SPACES_NUM_Y / value_map.shape[1]),
        order=1,
    )


def _format_rink_title(
    player_id: int | None = None,
    player_name: str | None = None,
    shot_count: int | None = None,
    seasons: list[int] | None = None,
) -> str:
    """Build a compact, rink-specific title string.

    Examples:
        "Player 950160 (First Last) | Seasons: 20232024, 20242025 — 12 shots"
        "Player 950160 — 3 shots"
        "Player 950160 (First Last)"
    """
    if player_id is None and player_name is None:
        parts = []
        if seasons:
            parts.append("Seasons: " + ", ".join(str(s) for s in seasons))
        base = " | ".join(parts) if parts else "Shot locations"
        if shot_count is not None:
            return f"{base} — {shot_count} shots"
        return base

    if player_name:
        base = f"Player {player_id} ({player_name})" if player_id is not None else str(player_name)
    else:
        base = f"Player {player_id}"

    if seasons:
        base = base + " | Seasons: " + ", ".join(str(s) for s in seasons)

    if shot_count is not None:
        return f"{base} — {shot_count} shots"
    return base


# ---------------------------------------------------------------------------
# Rink diagram
# ---------------------------------------------------------------------------

# NHL offensive zone face-off circle positions (feet)
# Two circles: one above center (y=+22) and one below (y=-22),
# both at x=69 (20 ft from the goal line at x=89).
_FACEOFF_X = 69.0
_FACEOFF_Y_UPPER = 22.0
_FACEOFF_Y_LOWER = -22.0
_FACEOFF_RADIUS = 15.0


def _draw_rink() -> tuple[plt.Figure, plt.Axes]:
    """Create a stylized offensive half-rink diagram.

    Draws the goal line, crease, two NHL-standard offensive zone face-off
    circles, and the net using standard NHL dimensions.  Returns the figure
    and axes so callers can overlay additional annotations.

    The axis spines are hidden but x/y tick marks and labels are kept so
    that readers can read off precise rink coordinates (feet from center
    ice along each axis).

    Returns
    -------
    (plt.Figure, plt.Axes)
    """
    fig, ax = plt.subplots()
    ax.set_aspect(1)

    wall_y = 42.5
    rink_boundary = plt.Rectangle((0, -wall_y), _GOAL_LINE_X, 2 * wall_y,
                                   ec="blue", fc="none", lw=2)
    ax.add_patch(rink_boundary)

    ax.plot([_GOAL_LINE_X, _GOAL_LINE_X], [-wall_y, wall_y], color="red", lw=2)

    crease = plt.Circle((_GOAL_LINE_X, 0), 6, color="red", fill=False, lw=2)
    ax.add_patch(crease)

    # NHL offensive zone: two face-off circles at (69, ±22) — not one at (69, 0)
    for fo_y in (_FACEOFF_Y_UPPER, _FACEOFF_Y_LOWER):
        faceoff = plt.Circle((_FACEOFF_X, fo_y), _FACEOFF_RADIUS,
                              color="red", fill=False, lw=2)
        ax.add_patch(faceoff)

    net = plt.Rectangle((_GOAL_LINE_X - 4, -3), 4, 6, ec="black", fc="none", lw=2)
    ax.add_patch(net)

    ax.set_xlim(0, _GOAL_LINE_X + 5)
    ax.set_ylim(wall_y, -wall_y)

    # Hide box spines but keep tick marks for coordinate readability
    # (follows the convention in Environments/Hockey/makePlotsFilteredShots.py)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    x_ticks = np.linspace(0, _GOAL_LINE_X, 10, dtype=int, endpoint=True)
    ax.set_xticks(x_ticks)
    ax.tick_params(axis="y", which="both", left=True, right=False)
    ax.tick_params(axis="x", which="both", bottom=True, top=False)
    ax.set_xlabel("X (ft from center ice)", fontsize=9)
    ax.set_ylabel("Y (ft)", fontsize=9)

    return fig, ax


# ---------------------------------------------------------------------------
# Angular heatmap
# ---------------------------------------------------------------------------

def plot_shot_angular_heatmap(
    value_map: np.ndarray,
    player_location: Sequence[float],
    executed_action: Sequence[float],
    *,
    save_path: Path | str | None = None,
    show: bool = False,
    title: str | None = None,
    event_id: int | None = None,
    is_goal: bool = False,
) -> plt.Figure | None:
    """Side-by-side Cartesian vs angular heatmap for one shot.

    Downsamples the value_map from native Snowflake resolution to the
    SpacesHockey grid (40×60), runs the angular transform, and produces
    a two-panel figure.

    Parameters
    ----------
    value_map : np.ndarray
        Raw value map from ``queries.py`` with shape ``(Z, Y)``
        (e.g. ``(72, 120)`` at native Blackhawks resolution).
    player_location : sequence of float
        ``[x, y]`` rink coordinates of the shooter.
    executed_action : sequence of float
        ``[y, z]`` goal-face coordinates where the shot ended up.
    save_path : Path | str | None
        Where to save the figure.  If *None*, auto-generates under
        ``Data/Hockey/plots/angular/``.
    show : bool
        If True, call ``plt.show()`` instead of closing.
    title : str | None
        Optional super-title for the figure.
    event_id : int | None
        Event identifier shown in the title annotation.
    is_goal : bool
        Whether the shot resulted in a goal.  Goals are rendered as a
        green star; misses as a black X.

    Returns
    -------
    Path | None
        Path of the saved figure, or None if only ``show=True``.
    """
    # Downsample to the 40×60 grid expected by getAngularHeatmap.
    heatmap = _resize_value_map(value_map)

    (
        dirs, elevations,
        _listedTargetsAngular,
        _gridTargetsAngular,
        _listedTargetsAngular2YZ,
        _gridTargetsAngular2YZ,
        listedUtilitiesComputed,
        _gridUtilitiesComputed,
        executedActionAngular,
        skip,
        _,
    ) = getAngularHeatmap(
        heatmap, np.asarray(player_location), np.asarray(executed_action),
    )

    if skip:
        print("Warning: angular heatmap skipped (player location too close to goal line).")
        return None

    # Build Cartesian target grid matching the downsampled heatmap.
    Y = np.linspace(-3.0, 3.0, _SPACES_NUM_Y)
    Z = np.linspace( 0.0, 4.0, _SPACES_NUM_Z)
    tY, tZ = np.meshgrid(Y, Z)
    cartesian_targets = np.stack((tY, tZ), axis=-1).reshape(-1, 2)

    cmap = _make_custom_cmap()
    flat_utils = heatmap.reshape(-1, 1)
    norm = plt.Normalize(0.0, float(np.nanmax(flat_utils)))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    listedTargetsAngular = np.asarray(_listedTargetsAngular)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Marker style depends on whether the shot was a goal
    _marker = "*" if is_goal else None #"x"
    _color = "#FFD700" if is_goal else "steelblue" #"black"
    _edgecolors = 'k' #if is_goal else None
    _label = "Goal" if is_goal else "Shot"
    _size = 160 if is_goal else 80

    # Left: Cartesian
    ax1.scatter(cartesian_targets[:, 0], cartesian_targets[:, 1],
                c=cmap(norm(flat_utils)))
    ax1.scatter(executed_action[0], executed_action[1],
                c=_color, edgecolors=_edgecolors, marker=_marker, s=_size, zorder=5, label=_label)
    ax1.set_title("Cartesian (Y, Z)")
    ax1.set_xlabel("Y")
    ax1.set_ylabel("Z")
    ax1.legend(loc="upper right", fontsize=8)
    fig.colorbar(sm, ax=ax1)

    # Right: Angular
    ax2.scatter(listedTargetsAngular[:, 0], listedTargetsAngular[:, 1],
                c=cmap(norm(listedUtilitiesComputed)))
    ax2.scatter(executedActionAngular[0], executedActionAngular[1],
                c=_color, edgecolors=_edgecolors, marker=_marker, s=_size, zorder=5, label=_label)
    ax2.set_title("Angular (direction, elevation)")
    ax2.set_xlabel("Direction (rad)")
    ax2.set_ylabel("Elevation (rad)")
    ax2.legend(loc="upper right", fontsize=8)
    fig.colorbar(sm, ax=ax2)

    # Build informative super-title with event ID and player XY
    suptitle_parts: list[str] = []
    if title:
        suptitle_parts.append(title)
    if event_id is not None:
        suptitle_parts.append(f"Event {event_id}")
    suptitle_parts.append(
        f"Player XY=({player_location[0]:.1f}, {player_location[1]:.1f})"
    )
    outcome_tag = "GOAL" if is_goal else "Shot"
    suptitle_parts.append(outcome_tag)
    plt.suptitle(" | ".join(suptitle_parts), fontsize=10)
    plt.tight_layout()

    out_path = None
    if save_path is not None:
        out_path = Path(save_path)
        _ensure_dir(out_path.parent)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    elif not show:
        out_path = _DEFAULT_DATA_DIR / "plots" / "angular" / "shot.png"
        _ensure_dir(out_path.parent)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)

    if show:
        fig.show()
    else:
        plt.close(fig)

    # Return the Figure for programmatic use (Streamlit can call st.pyplot(fig)).
    return fig


# ---------------------------------------------------------------------------
# Rink shot scatter
# ---------------------------------------------------------------------------

def plot_shot_rink(
    player_locations: np.ndarray | list,
    executed_actions: np.ndarray | list | None = None,
    *,
    is_goal: np.ndarray | list | None = None,
    save_path: Path | str | None = None,
    show: bool = False,
    title: str = "Shot locations",
) -> plt.Figure | None:
    """Plot one or more shots on a rink diagram.

    Shots and goals are rendered with distinct colours so they can be
    distinguished at a glance:

    * **Shots** (``is_goal=False``)  — blue circles
    * **Goals** (``is_goal=True``)   — red stars

    A legend is added automatically when ``is_goal`` is supplied.

    Parameters
    ----------
    player_locations : array-like, shape (N, 2)
        ``[x, y]`` rink positions for each shot.
    executed_actions : array-like, shape (N, 2), optional
        Unused — kept for backward compatibility.
    is_goal : array-like of bool, shape (N,), optional
        Whether each entry in *player_locations* resulted in a goal.
        When *None*, all markers are rendered as blue circles (original
        behaviour).
    save_path : Path | str | None
        Output file.  Defaults to ``Data/Hockey/plots/rink/shots.png``.
    show : bool
        Whether to ``plt.show()`` instead of saving.
    title : str
        Plot title.

    Returns
    -------
    plt.Figure | None
    """
    player_locations = np.atleast_2d(player_locations)

    fig, ax = _draw_rink()

    if is_goal is not None:
        is_goal_arr = np.asarray(is_goal, dtype=bool)
        shots_mask = ~is_goal_arr
        goals_mask = is_goal_arr

        if np.any(shots_mask):
            ax.scatter(
                player_locations[shots_mask, 0],
                player_locations[shots_mask, 1],
                c="steelblue", s=30, zorder=5, label="Shot",
            )
        if np.any(goals_mask):
            ax.scatter(
                player_locations[goals_mask, 0],
                player_locations[goals_mask, 1],
                c="#FFD700", edgecolors='k', marker="*", s=90, zorder=6, label="Goal",
            )
        ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    else:
        # No goal info — render everything uniformly
        ax.scatter(
            player_locations[:, 0], player_locations[:, 1],
            c="steelblue", s=30, zorder=5, label="Shot location",
        )

    ax.set_title(title)
    plt.tight_layout()

    out_path = None
    if save_path is not None:
        out_path = Path(save_path)
        _ensure_dir(out_path.parent)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    elif not show:
        out_path = _DEFAULT_DATA_DIR / "plots" / "rink" / "shots.png"
        _ensure_dir(out_path.parent)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)

    if show:
        fig.show()
    else:
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Convergence (delegates to plot_intermediate_estimates)
# ---------------------------------------------------------------------------

def plot_player_convergence(
    csv_path: Path | str,
    *,
    save_path: Path | str | None = None,
    show: bool = False,
    title: str | None = None,
 ) -> plt.Figure | None:
    """Plot JEEDS convergence for one estimation run.

    Thin wrapper around
    :func:`plot_intermediate_estimates.plot_intermediate_estimates`.

    Parameters
    ----------
    csv_path : Path | str
        Path to the intermediate estimates CSV.
    save_path : Path | str | None
        Output figure path; defaults to ``logs/<csv_name>.png`` (next to the CSV).
    show : bool
        Show interactively instead of saving.
    title : str | None
        Figure title; auto-derived from filename if *None*.

    Returns
    -------
    Path | None
    """
    # plot_intermediate_estimates now returns a Figure.
    return plot_intermediate_estimates(
        csv_path,
        output_path=save_path,
        title=title,
        show=show,
    )


def plot_all_player_convergence(
    player_id: int,
    data_dir: Path | str = Path("Data/Hockey"),
) -> list[plt.Figure]:
    """Generate convergence plots for every CSV under a player's logs/.

    Delegates to
    :func:`plot_intermediate_estimates.plot_all_intermediate_for_player`.

    Returns
    -------
    list[Path]
        Paths of generated figures.
    """
    return plot_all_intermediate_for_player(player_id, data_dir=data_dir)


# ---------------------------------------------------------------------------
# Helper: load player pickle data
# ---------------------------------------------------------------------------

def _load_player_pickles(
    player_id: int,
    data_dir: Path | str = Path("Data/Hockey"),
) -> tuple[pd.DataFrame | None, dict]:
    """Load shots DataFrame and shot_maps dict from player's pickle files.

    Parameters
    ----------
    player_id : int
        Blackhawks player ID.
    data_dir : Path | str
        Root data directory.

    Returns
    -------
    (shots_df, shot_maps)
        shots_df is None if no pickle files found.
    """
    import pickle
    import glob

    data_dir = Path(data_dir)
    player_dir = data_dir / f"player_{player_id}"
    data_subdir = player_dir / "data"

    shots_df = None
    shot_maps = {}

    for pkl_file in sorted(glob.glob(str(data_subdir / "shots_*.pkl"))):
        with open(pkl_file, "rb") as f:
            df = pickle.load(f)
            shots_df = df if shots_df is None else pd.concat([shots_df, df], ignore_index=True)

    for pkl_file in sorted(glob.glob(str(data_subdir / "shot_maps_*.pkl"))):
        with open(pkl_file, "rb") as f:
            maps = pickle.load(f)
            shot_maps.update(maps)

    return shots_df, shot_maps


# ---------------------------------------------------------------------------
# Batch: all shots for a player from offline data
# ---------------------------------------------------------------------------

def plot_player_shots_from_offline(
    player_id: int,
    data_dir: Path | str = Path("Data/Hockey"),
    *,
    tag: str | None = None,
    seasons: list[int] | None = None,
    max_shots: int = 10,
    output_dir: Path | str | None = None,
) -> dict[str, list[plt.Figure]]:
    """Generate rink + angular heatmap plots from offline pickle data.

    Loads shot data (via ``load_player_data`` or ``load_player_data_by_games``)
    and generates per-shot angular heatmap comparisons and a combined rink
    diagram.

    Parameters
    ----------
    player_id : int
        Blackhawks player ID.
    data_dir : Path | str
        Root directory containing ``player_{id}/`` folders.
    tag : str | None
        If provided, load game-tagged data via ``load_player_data_by_games``.
    seasons : list[int] | None
        If provided (and tag is None), load per-season data via
        ``load_player_data``.
    max_shots : int
        Maximum number of shots to render angular heatmaps for.
    output_dir : Path | str | None
        Where to write figures.  Defaults to
        ``Data/Hockey/player_{id}/plots/``.

    Returns
    -------
    dict with keys ``"angular"`` and ``"rink"``, each a list of Paths.
    """
    from BlackhawksSkillEstimation.BlackhawksJEEDS import (
        load_player_data,
        load_player_data_by_games,
    )

    data_dir = Path(data_dir)
    if output_dir is None:
        output_dir = data_dir / f"player_{player_id}" / "plots"
    output_dir = Path(output_dir)

    # Load data
    if tag is not None:
        df, shot_maps = load_player_data_by_games(
            player_id=player_id, tag=tag, data_dir=data_dir,
        )
    elif seasons is not None:
        df, shot_maps = load_player_data(
            player_id=player_id, seasons=seasons, data_dir=data_dir,
        )
    else:
        raise ValueError("Provide either 'tag' or 'seasons' to load data.")

    player_name = lookup_player(player_id)

    angular_paths: list[plt.Figure] = []
    player_locs: list[list[float]] = []
    executed_acts: list[list[float]] = []
    is_goals: list[bool] = []

    shot_count = 0
    for _, row in df.iterrows():
        event_id = int(row["event_id"])
        if event_id not in shot_maps:
            continue

        player_loc = [float(row["start_x"]), float(row["start_y"])]
        exec_act = [float(row["location_y"]), float(row["location_z"])]
        is_goal = bool(row.get("shot_is_goal", False))

        # ---- 10-ft net proximity filter ----
        # Skip shots within 10ft radius of net center (same as estimator filter)
        dist_to_net = np.linalg.norm(np.array(player_loc) - _NET_CENTER)
        if dist_to_net < _MIN_DISTANCE_FROM_NET_FT:
            continue

        player_locs.append(player_loc)
        executed_acts.append(exec_act)
        is_goals.append(is_goal)

        if shot_count < max_shots:
            value_map = shot_maps[event_id]["value_map"]
            _player_label = (
                f"Player {player_id} ({player_name})"
                if player_name is not None
                else f"Player {player_id}"
            )
            fig = plot_shot_angular_heatmap(
                value_map, player_loc, exec_act,
                save_path=output_dir / "angular" / f"shot_{event_id}.png",
                title=_player_label,
                event_id=event_id,
                is_goal=is_goal,
            )
            if fig is not None:
                angular_paths.append(fig)

        shot_count += 1

    # Combined rink diagram
    rink_path = None
    if player_locs:
        # Build a title consistent with angular plots (player name, seasons, shot count)
        title_str = _format_rink_title(
            player_id=player_id,
            player_name=player_name,
            shot_count=len(player_locs),
            seasons=seasons,
        )
        rink_fig = plot_shot_rink(
            player_locs, executed_acts,
            is_goal=is_goals,
            save_path=output_dir / "rink" / f"player_{player_id}_all_shots.png",
            title=title_str,
        )
        rink_path = rink_fig
    return {
        "angular": angular_paths,
        "rink": [rink_path] if rink_path else [],
    }


# ---------------------------------------------------------------------------
# CLI: Single shot heatmap
# ---------------------------------------------------------------------------

def plot_single_shot_cli(
    player_id: int,
    event_id: int,
    data_dir: Path | str = Path("Data/Hockey"),
    output_dir: Path | str | None = None,
) -> Path | None:
    """Generate heatmap for a single shot event given player_id and event_id.

    Searches all pickle files in player's data directory for the event_id,
    extracts shot data, and saves angular heatmap to output_dir.

    Parameters
    ----------
    player_id : int
        Blackhawks player ID.
    event_id : int
        Shot event ID to plot.
    data_dir : Path | str
        Root data directory (default: "Data/Hockey").
    output_dir : Path | str | None
        Output directory (default: Data/Hockey/player_{id}/plots/).

    Returns
    -------
    Path | None
        Path to saved figure, or None if event not found.
    """
    data_dir = Path(data_dir)
    player_dir = data_dir / f"player_{player_id}"

    if output_dir is None:
        output_dir = player_dir / "plots"
    output_dir = Path(output_dir)

    shots_df, shot_maps = _load_player_pickles(player_id, data_dir)

    if shots_df is None or event_id not in shot_maps:
        print(f"Event {event_id} not found for player {player_id}")
        return None

    # Find shot row
    shot_row = shots_df[shots_df["event_id"] == event_id]
    if shot_row.empty:
        print(f"Event {event_id} metadata not found")
        return None

    shot_row = shot_row.iloc[0]
    value_map = shot_maps[event_id]["value_map"]
    player_loc = [float(shot_row.get("start_x", 0)), float(shot_row.get("start_y", 0))]
    exec_act = [float(shot_row.get("location_y", 0)), float(shot_row.get("location_z", 0))]
    is_goal = bool(shot_row.get("shot_is_goal", False))

    player_name = lookup_player(player_id)
    title = f"Player {player_id} ({player_name})" if player_name else f"Player {player_id}"

    fig = plot_shot_angular_heatmap(
        value_map, player_loc, exec_act,
        save_path=output_dir / "angular" / f"shot_{event_id}.png",
        title=title,
        event_id=event_id,
        is_goal=is_goal,
    )

    if fig is not None:
        return output_dir / "angular" / f"shot_{event_id}.png"
    return None


# ---------------------------------------------------------------------------
# Rink: Single shot by event ID
# ---------------------------------------------------------------------------

def plot_single_shot_rink(
    player_id: int,
    event_id: int,
    data_dir: Path | str = Path("Data/Hockey"),
    output_dir: Path | str | None = None,
    *,
    seasons: list[int] | None = None,
    show: bool = False,
) -> Path | None:
    """Plot a single shot by event ID on a rink diagram.

    Loads the shot's rink-space origin from offline pickle data, looks
    up the player name from the player cache, and saves a rink diagram
    with that shot highlighted.  Goals appear as red stars; shots as
    blue circles.

    Parameters
    ----------
    player_id : int
        Blackhawks player ID.  The player's pickle files must exist
        under ``data_dir/player_{player_id}/data/``.
    event_id : int
        Shot event ID to plot.  Must be present in one of the
        ``shots_*.pkl`` files for the player.
    data_dir : Path | str
        Root data directory (default: ``"Data/Hockey"``).
    output_dir : Path | str | None
        Where to save the figure.  Defaults to
        ``data_dir/player_{player_id}/plots/``.
    seasons : list[int] | None
        Season(s) to annotate on the rink (e.g. ``[20242025]``).
    show : bool
        Show interactively with ``plt.show()`` instead of saving.

    Returns
    -------
    Path | None
        Path of the saved PNG, or *None* if the event was not found or
        ``show=True``.
    """
    data_dir = Path(data_dir)
    player_dir = data_dir / f"player_{player_id}"

    if output_dir is None:
        output_dir = player_dir / "plots"
    output_dir = Path(output_dir)

    shots_df, _ = _load_player_pickles(player_id, data_dir)

    if shots_df is None:
        print(f"No shot data found for player {player_id}")
        return None

    shot_row = shots_df[shots_df["event_id"] == event_id]
    if shot_row.empty:
        print(f"Event {event_id} not found for player {player_id}")
        return None

    shot_row = shot_row.iloc[0]
    player_loc = [float(shot_row.get("start_x", 0)), float(shot_row.get("start_y", 0))]
    is_goal_flag = bool(shot_row.get("shot_is_goal", False))

    player_name = lookup_player(player_id)
    outcome = "Goal" if is_goal_flag else "Shot"
    name_part = f"Player {player_id} ({player_name})" if player_name else f"Player {player_id}"
    title = f"{name_part} — Event {event_id} [{outcome}]"

    save_path: Path | None = None
    if not show:
        save_path = output_dir / "rink" / f"shot_{event_id}_rink.png"

    plot_shot_rink(
        [player_loc],
        is_goal=[is_goal_flag],
        save_path=save_path,
        show=show,
        title=title,
    )

    if save_path is not None:
        print(f"Saved rink plot: {save_path}")

    return save_path


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Generate heatmap(s) and/or rink plot(s) for shot(s).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Angular heatmap for a single event:
  python blackhawks_plots.py -player_id 950160 -event_id 12345

  # Rink plot for a single event (shows position highlighted on rink):
  python blackhawks_plots.py -player_id 950160 -rink_event_id 12345

  # Rink + angular heatmaps for all shots in one season:
  python blackhawks_plots.py -player_id 950160 -seasons 20242025

  # Rink plot only (no individual angular heatmaps) for one season:
  python blackhawks_plots.py -player_id 950160 -seasons 20242025 -limit 0

  # Multiple seasons at once:
  python blackhawks_plots.py -player_id 950160 -seasons 20232024 20242025
""",
    )
    parser.add_argument("-player_id", type=int, required=True, help="Player ID")
    parser.add_argument("-event_id", type=int, default=None,
                        help="Single event ID — generates an angular heatmap for that shot")
    parser.add_argument("-rink_event_id", type=int, default=None,
                        help="Single event ID — generates a rink diagram for that shot")
    parser.add_argument("-seasons", type=int, nargs="+", default=None,
                        help="One or more seasons (e.g. -seasons 20242025  or  -seasons 20232024 20242025)")
    parser.add_argument("-limit", type=int, default=10,
                        help="Max angular heatmaps to generate when running over a full season (default: 10). Use -limit 0 to generate only the rink plot without individual angular heatmaps.")
    parser.add_argument("-data_dir", type=str, default="Data/Hockey", help="Data directory")
    parser.add_argument("-output_dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    # Resolve effective seasons list
    _seasons: list[int] | None = args.seasons
    if _seasons is None and args.season is not None:
        _seasons = [args.season]

    if args.event_id is not None:
        # Angular heatmap for a single shot event
        result = plot_single_shot_cli(
            player_id=args.player_id,
            event_id=args.event_id,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
        )
        if result:
            print(f"Saved: {result}")
        else:
            print("Failed to generate heatmap")
            sys.exit(1)

    elif args.rink_event_id is not None:
        # Rink dot-plot for a single shot event
        result_path = plot_single_shot_rink(
            player_id=args.player_id,
            event_id=args.rink_event_id,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            seasons=_seasons,
        )
        if result_path is None:
            print("Failed to generate rink plot")
            sys.exit(1)

    elif _seasons is not None:
        # Full-season batch: angular heatmaps + combined rink
        result = plot_player_shots_from_offline(
            player_id=args.player_id,
            data_dir=args.data_dir,
            seasons=_seasons,
            max_shots=args.limit,
            output_dir=args.output_dir,
        )
        print(f"Generated {len(result.get('angular', []))} angular heatmaps")
        print(f"Generated {len(result.get('rink', []))} rink diagram(s)")

    else:
        parser.print_help()
        sys.exit(1)
