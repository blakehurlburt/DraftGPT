"""MiLB Stats API client with local JSON cache.

Fetches minor-league stats from the MLB Stats API (statsapi.mlb.com)
and caches responses to data/milb_cache/ for offline use.  No API key
required — the endpoint is public.
"""

import json
import time
from pathlib import Path

import httpx

BASE_URL = "https://statsapi.mlb.com/api/v1"
CACHE_DIR = Path(__file__).parent.parent / "data" / "milb_cache"

# MiLB level sport IDs used by the MLB Stats API
SPORT_IDS = {
    "AAA": 11,
    "AA": 12,
    "High-A": 13,
    "Single-A": 14,
    "Rookie": 16,
    "MLB": 1,
}

# Numeric ordering for level comparisons (higher = more advanced)
LEVEL_RANK = {
    "Rookie": 1,
    "Single-A": 2,
    "High-A": 3,
    "AA": 4,
    "AAA": 5,
    "MLB": 6,
}

# Reverse map: sport_id -> level name
SPORT_ID_TO_LEVEL = {v: k for k, v in SPORT_IDS.items()}

# Delay between API requests (seconds)
_REQUEST_DELAY = 0.1


def _ensure_cache():
    """Create cache directories if they don't exist."""
    for subdir in ("players", "stats", "draft"):
        (CACHE_DIR / subdir).mkdir(parents=True, exist_ok=True)


def _get(url: str, params: dict | None = None, timeout: float = 15) -> dict:
    """Make a GET request to the MLB Stats API with a polite delay."""
    time.sleep(_REQUEST_DELAY)
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Players at a MiLB level
# ---------------------------------------------------------------------------

def fetch_players_at_level(sport_id: int, season: int) -> list[dict]:
    """Fetch all players at a MiLB level for a given season.

    Returns list of dicts with keys: id, fullName, birthDate, primaryPosition, etc.
    Caches to data/milb_cache/players/{sport_id}_{season}.json.
    """
    _ensure_cache()
    cache_path = CACHE_DIR / "players" / f"{sport_id}_{season}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    data = _get(f"{BASE_URL}/sports/{sport_id}/players", params={"season": season})
    players = data.get("people", [])

    cache_path.write_text(json.dumps(players, indent=1))
    return players


# ---------------------------------------------------------------------------
# Player year-by-year stats
# ---------------------------------------------------------------------------

def fetch_player_stats(
    mlb_api_id: int,
    group: str = "hitting",
    levels: list[str] | None = None,
) -> list[dict]:
    """Fetch year-by-year stats for a player across specified levels.

    Args:
        mlb_api_id: MLB Stats API person ID.
        group: "hitting" or "pitching".
        levels: Which levels to fetch (e.g. ["AAA", "AA"]).
                Defaults to all levels.

    Returns:
        List of stat split dicts, each with keys like:
        season, stat (dict of stat values), team, league, sport, etc.
    Caches to data/milb_cache/stats/{mlb_api_id}_{group}.json.
    """
    _ensure_cache()
    cache_path = CACHE_DIR / "stats" / f"{mlb_api_id}_{group}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    fetch_levels = levels or list(SPORT_IDS.keys())
    all_splits = []
    for level_name in fetch_levels:
        sport_id = SPORT_IDS.get(level_name)
        if sport_id is None:
            continue
        try:
            data = _get(
                f"{BASE_URL}/people/{mlb_api_id}/stats",
                params={"stats": "yearByYear", "group": group, "sportId": sport_id},
            )
        except httpx.HTTPStatusError:
            continue

        for stat_group in data.get("stats", []):
            for split in stat_group.get("splits", []):
                split["_level"] = level_name
                split["_sport_id"] = sport_id
                all_splits.append(split)

    cache_path.write_text(json.dumps(all_splits, indent=1))
    return all_splits


# ---------------------------------------------------------------------------
# Draft data
# ---------------------------------------------------------------------------

def fetch_draft(year: int) -> list[dict]:
    """Fetch MLB draft picks for a given year.

    Returns list of pick dicts with keys: pickRound, pickNumber, pickValue,
    person (dict with id, fullName, birthDate, etc.), school, etc.
    Caches to data/milb_cache/draft/{year}.json.
    """
    _ensure_cache()
    cache_path = CACHE_DIR / "draft" / f"{year}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    data = _get(f"{BASE_URL}/draft/{year}")

    picks = []
    for rnd in data.get("drafts", {}).get("rounds", []):
        for pick in rnd.get("picks", []):
            picks.append(pick)

    cache_path.write_text(json.dumps(picks, indent=1))
    return picks


# ---------------------------------------------------------------------------
# Player biographical info
# ---------------------------------------------------------------------------

def fetch_player_bio(mlb_api_id: int) -> dict | None:
    """Fetch a single player's biographical info (name, birth date, etc.)."""
    try:
        data = _get(f"{BASE_URL}/people/{mlb_api_id}")
        people = data.get("people", [])
        return people[0] if people else None
    except httpx.HTTPStatusError:
        return None


def clear_stats_cache(mlb_api_id: int, group: str = "hitting"):
    """Remove a cached stats file so it gets re-fetched next time."""
    cache_path = CACHE_DIR / "stats" / f"{mlb_api_id}_{group}.json"
    cache_path.unlink(missing_ok=True)
