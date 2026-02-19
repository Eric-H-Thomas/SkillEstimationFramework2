import csv
from pathlib import Path
from BlackhawksAPI.queries import get_player_name

CACHE_FILE = Path("Data/Hockey/player_cache.csv")

def lookup_player(player_id: int | None = None, player_name: str | None = None) -> str | int | None:
    """
    Resolve a player_id to a player_name or vice versa using a local cache.
    If the mapping is not found in the cache, attempt to query the database
    and update the cache.

    Parameters
    ----------
    player_id : int | None
        The player ID to resolve to a name.
    player_name : str | None
        The player name to resolve to an ID.

    Returns
    -------
    str | int | None
        The resolved player_name or player_id, or None if not found.
    """
    if not CACHE_FILE.exists():
        # Create the cache file if it doesn't exist
        with open(CACHE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["player_id", "player_name"])

    # Read the cache
    cache = {}
    with open(CACHE_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cache[int(row["player_id"])] = row["player_name"]

    # Resolve from cache
    if player_id is not None:
        if player_id in cache:
            return cache[player_id]
    elif player_name is not None:
        for pid, name in cache.items():
            if name == player_name:
                return pid

    # If not found in cache, query the database
    if player_id is not None:
        player_name = get_player_name(player_id)
        if player_name:
            # Update the cache
            with open(CACHE_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([player_id, player_name])
            return player_name
    elif player_name is not None:
        # TODO: Add logic to query the database for player_id if needed
        pass  # Currently, the database query for player_id by name is not implemented
        # I added this in case we want to run the program by player name instead of id eventually.

    return None