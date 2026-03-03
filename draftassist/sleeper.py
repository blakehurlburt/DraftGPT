"""Async Sleeper API client with player cache."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

BASE_URL = "https://api.sleeper.app"
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

    if PLAYER_CACHE.exists():
        age = time.time() - PLAYER_CACHE.stat().st_mtime
        if age < CACHE_TTL:
            with open(PLAYER_CACHE) as f:
                return json.load(f)

    resp = await client.get(f"{BASE_URL}/v1/players/nfl")
    resp.raise_for_status()
    data = resp.json()

    with open(PLAYER_CACHE, "w") as f:
        json.dump(data, f)

    return data
