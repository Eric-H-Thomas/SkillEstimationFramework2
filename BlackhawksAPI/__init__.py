"""Convenience exports for the Blackhawks data API package."""

from .queries import (
    get_game_shot_maps,
    get_games_for_player,
    get_games_shot_maps_batch,
    get_games_for_seasons,
    get_top_shooters_in_games,
    plot_ellipse,
    query_player_game_info,
    query_player_season_shots,
    shot_games,
)

__all__ = [
    "get_game_shot_maps",
    "get_games_for_player",
    "get_games_shot_maps_batch",
    "get_games_for_seasons",
    "get_top_shooters_in_games",
    "plot_ellipse",
    "query_player_game_info",
    "query_player_season_shots",
    "shot_games",
]
