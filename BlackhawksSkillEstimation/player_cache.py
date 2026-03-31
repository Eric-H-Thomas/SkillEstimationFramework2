import csv
from pathlib import Path

CACHE_FILE = Path("Data/Hockey/player_cache.csv")


def _ensure_cache_file() -> None:
    if CACHE_FILE.exists():
        return
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["player_id", "player_name"])


def _read_cache() -> dict[int, str]:
    _ensure_cache_file()
    cache: dict[int, str] = {}
    with open(CACHE_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cache[int(row["player_id"])] = row["player_name"]
            except Exception:
                continue
    return cache


def _normalize_name(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _safe_get_player_name(player_id: int) -> str | None:
    """Best-effort DB lookup, safe for offline/cluster environments.

    Importing Blackhawks API modules can fail on clusters without local
    credentials/config (e.g., missing hawks.ini). Keep this lazy and guarded
    so importing this module never requires DB connectivity.
    """
    try:
        from BlackhawksAPI.queries import get_player_name
    except Exception:
        return None

    try:
        return get_player_name(player_id)
    except Exception:
        return None


def lookup_player_ids_by_name(player_name: str) -> list[int]:
    """Return all player IDs in cache that match a name (case-insensitive)."""
    normalized_target = _normalize_name(player_name)
    if not normalized_target:
        return []

    cache = _read_cache()
    matches: list[int] = []
    for pid, cached_name in cache.items():
        if _normalize_name(cached_name) == normalized_target:
            matches.append(pid)
    return matches


def get_cached_player_map() -> dict[int, str]:
    """Return cached mapping of player_id -> player_name."""
    return _read_cache()

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
    cache = _read_cache()

    # Resolve from cache
    if player_id is not None:
        if player_id in cache:
            return cache[player_id]
    elif player_name is not None:
        matches = lookup_player_ids_by_name(player_name)
        return matches[0] if matches else None

    # If not found in cache, query the database
    if player_id is not None:
        player_name = _safe_get_player_name(player_id)
        if player_name:
            # Re-read the cache right before writing to catch concurrent updates
            updated_cache = _read_cache()
            if player_id not in updated_cache:
                # safe to append...
                # Update the cache
                with open(CACHE_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([player_id, player_name])
            return player_name

    return None