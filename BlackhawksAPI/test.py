"""Example script to call `query_player_game_info`.

Set the required Snowflake environment variables before running:
    export BLACKHAWKS_SNOWFLAKE_USER="your_username"
    export BLACKHAWKS_SNOWFLAKE_PASSWORD="your_password"
    export BLACKHAWKS_SNOWFLAKE_ACCOUNT="your_account"
    export BLACKHAWKS_SNOWFLAKE_DATABASE="your_database"
    export BLACKHAWKS_SNOWFLAKE_ROLE="your_role"  # optional
    export BLACKHAWKS_SNOWFLAKE_WAREHOUSE="your_warehouse"  # optional
"""

from BlackhawksAPI import query_player_game_info

# Update these placeholders with valid identifiers for your environment
PLAYER_ID = 950160
GAME_IDS = [44604, 270247]


def main() -> None:
    df = query_player_game_info(player_id=PLAYER_ID, game_ids=GAME_IDS)
    print(df.head())


if __name__ == "__main__":
    main()
