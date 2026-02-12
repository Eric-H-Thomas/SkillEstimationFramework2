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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from scipy.ndimage import zoom

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
# Value-map dimensions produced by BlackhawksAPI.queries
_BH_VALUE_MAP_Z = 72   # axis-0 = Z (vertical, 72 pixels)
_BH_VALUE_MAP_Y = 120  # axis-1 = Y (lateral, 120 pixels)

# SpacesHockey target grid sizes (hardcoded in setupSpaces.py)
_SPACES_NUM_Z = 40
_SPACES_NUM_Y = 60

# Rink geometry (feet, NHL standard)
_GOAL_LINE_X = 89
_LEFT_POST = np.array([_GOAL_LINE_X, -3])
_RIGHT_POST = np.array([_GOAL_LINE_X,  3])

# Default base directory for player data
_DEFAULT_DATA_DIR = Path("Data/Hockey")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resize_value_map(value_map: np.ndarray) -> np.ndarray:
    """Resize a Blackhawks value_map to framework grid dimensions.

    The Blackhawks value_map from ``queries.py`` has shape (72, 120) with
    axis-0 = Z and axis-1 = Y.  ``getAngularHeatmap`` expects shape
    ``(len(Z), len(Y))`` = ``(40, 60)``, matching the ``SpacesHockey``
    target grids.

    Parameters
    ----------
    value_map : np.ndarray
        Raw value map with shape ``(72, 120)``.

    Returns
    -------
    np.ndarray
        Resized array with shape ``(40, 60)``.
    """
    return zoom(
        value_map,
        (_SPACES_NUM_Z / value_map.shape[0], _SPACES_NUM_Y / value_map.shape[1]),
        order=1,
    )


def _make_custom_cmap(base: str = "gist_rainbow", brightness: float = 0.4):
    """Build a brighter custom colormap for heatmap scatter plots."""
    n = plt.cm.jet.N
    cmap = (1.0 - brightness) * plt.get_cmap(base)(np.linspace(0.0, 1.0, n)) + brightness * np.ones((n, 4))
    return ListedColormap(cmap)


# ---------------------------------------------------------------------------
# Rink diagram
# ---------------------------------------------------------------------------

def draw_rink() -> plt.Axes:
    """Create a stylized offensive half-rink diagram.

    Draws the goal line, crease, face-off circle, and net using
    standard NHL dimensions.  Returns the axes object so callers can
    overlay additional annotations.

    Returns
    -------
    plt.Axes
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

    faceoff = plt.Circle((69, 0), 15, color="red", fill=False, lw=2)
    ax.add_patch(faceoff)

    net = plt.Rectangle((_GOAL_LINE_X - 4, -3), 4, 6, ec="black", fc="none", lw=2)
    ax.add_patch(net)

    ax.set_xlim(0, _GOAL_LINE_X + 5)
    ax.set_ylim(wall_y, -wall_y)
    ax.axis("off")

    return ax


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
) -> Path | None:
    """Side-by-side Cartesian vs angular heatmap for one shot.

    Resizes the Blackhawks value_map to the framework grid, runs the
    angular transform, and produces a two-panel figure.

    Parameters
    ----------
    value_map : np.ndarray
        Raw value map from ``queries.py`` with shape ``(72, 120)``.
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
    ) = getAngularHeatmap(heatmap, np.asarray(player_location), np.asarray(executed_action))

    if skip:
        print("Warning: angular heatmap skipped (player location too close to goal line).")
        return None

    # Build Cartesian target grid (matches SpacesHockey)
    Y = np.linspace(-3.0, 3.0, _SPACES_NUM_Y)
    Z = np.linspace(0.0, 4.0, _SPACES_NUM_Z)
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
    _marker = "*" if is_goal else "x"
    _color = "green" if is_goal else "black"
    _label = "Goal" if is_goal else "Shot"
    _size = 160 if is_goal else 80

    # Left: Cartesian
    ax1.scatter(cartesian_targets[:, 0], cartesian_targets[:, 1],
                c=cmap(norm(flat_utils)))
    ax1.scatter(executed_action[0], executed_action[1],
                c=_color, marker=_marker, s=_size, zorder=5, label=_label)
    ax1.set_title("Cartesian (Y, Z)")
    ax1.set_xlabel("Y")
    ax1.set_ylabel("Z")
    ax1.legend(loc="upper right", fontsize=8)
    fig.colorbar(sm, ax=ax1)

    # Right: Angular
    ax2.scatter(listedTargetsAngular[:, 0], listedTargetsAngular[:, 1],
                c=cmap(norm(listedUtilitiesComputed)))
    ax2.scatter(executedActionAngular[0], executedActionAngular[1],
                c=_color, marker=_marker, s=_size, zorder=5, label=_label)
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
        plt.show()
    else:
        plt.close(fig)

    return out_path


# ---------------------------------------------------------------------------
# Rink shot scatter
# ---------------------------------------------------------------------------

def plot_shot_rink(
    player_locations: np.ndarray | list,
    executed_actions: np.ndarray | list | None = None,
    *,
    save_path: Path | str | None = None,
    show: bool = False,
    title: str = "Shot locations",
) -> Path | None:
    """Plot one or more shots on a rink diagram.

    Parameters
    ----------
    player_locations : array-like, shape (N, 2)
        ``[x, y]`` rink positions for each shot.
    executed_actions : array-like, shape (N, 2), optional
        If provided, lines are draw from each player location toward
        the goal posts to indicate the shooting angle.
    save_path : Path | str | None
        Output file.  Defaults to ``Data/Hockey/plots/rink/shots.png``.
    show : bool
        Whether to ``plt.show()`` instead of saving.
    title : str
        Plot title.

    Returns
    -------
    Path | None
    """
    player_locations = np.atleast_2d(player_locations)

    ax = draw_rink()
    ax.scatter(player_locations[:, 0], player_locations[:, 1],
               c="blue", s=30, zorder=5, label="Shot location")

    if executed_actions is not None:
        executed_actions = np.atleast_2d(executed_actions)

    # Draw shooting-angle lines for each shot
    for i, loc in enumerate(player_locations):
        ax.plot([loc[0], _LEFT_POST[0]], [loc[1], _LEFT_POST[1]],
                color="cornflowerblue", alpha=0.3, lw=0.8)
        ax.plot([loc[0], _RIGHT_POST[0]], [loc[1], _RIGHT_POST[1]],
                color="cornflowerblue", alpha=0.3, lw=0.8)

    ax.set_title(title)
    plt.tight_layout()

    out_path = None
    if save_path is not None:
        out_path = Path(save_path)
        _ensure_dir(out_path.parent)
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
    elif not show:
        out_path = _DEFAULT_DATA_DIR / "plots" / "rink" / "shots.png"
        _ensure_dir(out_path.parent)
        plt.savefig(out_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    else:
        plt.close()

    return out_path


# ---------------------------------------------------------------------------
# Convergence (delegates to plot_intermediate_estimates)
# ---------------------------------------------------------------------------

def plot_player_convergence(
    csv_path: Path | str,
    *,
    save_path: Path | str | None = None,
    show: bool = False,
    title: str | None = None,
) -> Path | None:
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
    return plot_intermediate_estimates(
        csv_path,
        output_path=save_path,
        title=title,
        show=show,
    )


def plot_all_player_convergence(
    player_id: int,
    data_dir: Path | str = Path("Data/Hockey"),
) -> list[Path]:
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
) -> dict[str, list[Path]]:
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

    angular_paths: list[Path] = []
    player_locs: list[list[float]] = []
    executed_acts: list[list[float]] = []

    shot_count = 0
    for _, row in df.iterrows():
        event_id = int(row["event_id"])
        if event_id not in shot_maps:
            continue

        player_loc = [float(row["start_x"]), float(row["start_y"])]
        exec_act = [float(row["location_y"]), float(row["location_z"])]
        is_goal = bool(row.get("shot_is_goal", False))
        player_locs.append(player_loc)
        executed_acts.append(exec_act)

        if shot_count < max_shots:
            value_map = shot_maps[event_id]["value_map"]
            out = plot_shot_angular_heatmap(
                value_map, player_loc, exec_act,
                save_path=output_dir / "angular" / f"shot_{event_id}.png",
                title=f"Player {player_id}",
                event_id=event_id,
                is_goal=is_goal,
            )
            if out is not None:
                angular_paths.append(out)

        shot_count += 1

    # Combined rink diagram
    rink_path = None
    if player_locs:
        rink_path = plot_shot_rink(
            player_locs, executed_acts,
            save_path=output_dir / "rink" / f"player_{player_id}_all_shots.png",
            title=f"Player {player_id} – {len(player_locs)} shots",
        )

    return {
        "angular": angular_paths,
        "rink": [rink_path] if rink_path else [],
    }
