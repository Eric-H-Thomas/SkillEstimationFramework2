# Blackhawks API

This package provides a minimal wrapper around the Blackhawks Snowflake warehouse so
future modules can query data without duplicating connection logic.

## Configuration

Set the following environment variables before running code that imports this package:

- `BLACKHAWKS_SNOWFLAKE_USER`
- `BLACKHAWKS_SNOWFLAKE_PASSWORD`
- `BLACKHAWKS_SNOWFLAKE_ACCOUNT`
- `BLACKHAWKS_SNOWFLAKE_DATABASE`
- `BLACKHAWKS_SNOWFLAKE_ROLE` (optional)
- `BLACKHAWKS_SNOWFLAKE_WAREHOUSE` (optional)

## Usage

```python
from BlackhawksAPI import query_player_game_info, shot_games, get_game_shot_maps

player_rows = query_player_game_info(player_id=950160, game_ids=[44604, 270247])
print(player_rows.head())

for game_id in shot_games(season=20242025):
    shot_maps = get_game_shot_maps(game_id)
    # process shot_maps here
```

The helper `plot_ellipse` matches the reference script's visualization utility and
can be used when plotting covariance ellipses for shot trajectories.
