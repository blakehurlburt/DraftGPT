"""Async Sleeper API client with player cache."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

BASE_URL = "https://api.sleeper.app"
# CR opus: Relative path ".cache" resolves from the CWD, not from the module directory.
# If the server is started from a different working directory, the cache will be
# written to an unexpected location and won't be found on subsequent runs.
CACHE_DIR = Path(".cache")
PLAYER_CACHE = CACHE_DIR / "sleeper_players.json"
CACHE_TTL = 86400  # 24 hours


async def fetch_draft_meta(client: httpx.AsyncClient, draft_id: str) -> dict:
    """Fetch draft metadata: type, status, settings, draft_order, slot mapping."""
    resp = await client.get(f"{BASE_URL}/v1/draft/{draft_id}")
    resp.raise_for_status()
    return resp.json()


async def fetch_draft_picks(client: httpx.AsyncClient, draft_id: str) -> list[dict]:
    """Fetch all picks made so far in a draft."""
    resp = await client.get(f"{BASE_URL}/v1/draft/{draft_id}/picks")
    resp.raise_for_status()
    return resp.json()


async def fetch_all_players(client: httpx.AsyncClient) -> dict[str, dict]:
    """Fetch full NFL player database from Sleeper (~5MB). Cached for 24h."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # CR opus: File I/O (open/json.load/json.dump) is synchronous and blocks the
    # async event loop. For the ~5MB players file, this can block for noticeable time.
    # Consider using aiofiles or running in an executor.
    if PLAYER_CACHE.exists():
        age = time.time() - PLAYER_CACHE.stat().st_mtime
        if age < CACHE_TTL:
            with open(PLAYER_CACHE) as f:
                return json.load(f)

    # CR opus: Hardcoded to "nfl" — won't work if Sleeper adds other sports.
    resp = await client.get(f"{BASE_URL}/v1/players/nfl")
    resp.raise_for_status()
    data = resp.json()

    # CR opus: Race condition — if two concurrent requests both miss the cache,
    # they'll both fetch from the API and write to the same file simultaneously.
    # Not critical (last writer wins with valid data), but worth noting.
    with open(PLAYER_CACHE, "w") as f:
        json.dump(data, f)

    return data


async def fetch_projections(
    client: httpx.AsyncClient, season: int = 2026,
) -> dict[str, dict]:
    """Fetch season-long player projections from Sleeper. Cached 24h.

    Returns dict mapping player_id -> stat projection dict
    (keys like pass_yd, rush_yd, rec, rec_yd, rush_td, etc.).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"sleeper_projections_{season}.json"

    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < CACHE_TTL:
            with open(cache_path) as f:
                return json.load(f)

    resp = await client.get(
        f"{BASE_URL}/v1/projections/nfl/regular/{season}",
    )
    resp.raise_for_status()
    raw = resp.json()

    # Normalize: response is list of dicts with player_id + stats,
    # or a dict keyed by player_id. Handle both formats.
    if isinstance(raw, list):
        data = {}
        for entry in raw:
            pid = str(entry.get("player_id", ""))
            stats = entry.get("stats") or entry
            if pid:
                data[pid] = stats
    elif isinstance(raw, dict):
        data = {str(k): v for k, v in raw.items()}
    else:
        data = {}

    with open(cache_path, "w") as f:
        json.dump(data, f)

    return data
