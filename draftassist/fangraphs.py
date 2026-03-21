"""Async FanGraphs projection client with caching.

Fetches Steamer (or other) projections for MLB batters and pitchers.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import httpx

log = logging.getLogger("draftassist.fangraphs")

BASE_URL = "https://www.fangraphs.com/api/projections"
CACHE_DIR = Path(".cache")
CACHE_TTL = 86400  # 24 hours


async def fetch_fangraphs_projections(
    client: httpx.AsyncClient,
    system: str = "steamer",
    season: int = 2026,
) -> list[dict]:
    """Fetch MLB projections from FanGraphs (batters + pitchers combined).

    Args:
        client: httpx async client.
        system: Projection system — steamer, zips, atc, thebatx, fangraphsdc.
        season: MLB season year.

    Returns:
        List of player projection dicts with keys like PlayerName, Team,
        minpos, HR, R, RBI, SB, AVG, W, SV, SO, ERA, WHIP, IP, G, etc.
        Each dict also has ``_fg_type`` set to ``"bat"`` or ``"pit"``.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"fangraphs_{system}_{season}.json"

    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < CACHE_TTL:
            with open(cache_path) as f:
                return json.load(f)

    # CR opus: The `system` parameter is user-supplied (via query param) and is passed
    # directly into the FanGraphs URL without validation. Should validate against
    # known systems (steamer, zips, atc, thebatx, fangraphsdc) to prevent unexpected
    # API calls or cache-path injection (e.g., system="../../etc/passwd").
    params = {
        "type": system,
        "pos": "all",
        "team": "0",
        "players": "0",
    }

    # CR opus: These two requests are sequential but independent — they could be
    # fetched concurrently with asyncio.gather for ~2x speedup.
    bat_resp = await client.get(BASE_URL, params={**params, "stats": "bat"})
    bat_resp.raise_for_status()
    batters = bat_resp.json()

    pit_resp = await client.get(BASE_URL, params={**params, "stats": "pit"})
    pit_resp.raise_for_status()
    pitchers = pit_resp.json()

    for entry in batters:
        entry["_fg_type"] = "bat"
    for entry in pitchers:
        entry["_fg_type"] = "pit"

    combined = batters + pitchers

    # CR opus: If the first request succeeds but the second fails (raise_for_status),
    # no data is cached and no partial result is returned. The caller gets an unhandled
    # httpx.HTTPStatusError. Consider caching batters/pitchers separately or handling
    # partial failures gracefully.
    with open(cache_path, "w") as f:
        json.dump(combined, f)

    log.info(
        "Fetched FanGraphs %s projections: %d batters, %d pitchers",
        system, len(batters), len(pitchers),
    )
    return combined
