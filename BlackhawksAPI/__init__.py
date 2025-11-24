"""Convenience exports for the Blackhawks data API package."""

from .queries import (
    get_game_shot_maps,
    plot_ellipse,
    query_player_game_info,
    shot_games,
)

__all__ = [
    "get_game_shot_maps",
    "plot_ellipse",
    "query_player_game_info",
    "shot_games",
]
