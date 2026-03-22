"""
College football production features for rookie projections.

Fetches career college stats from the CollegeFootballData.com API and caches
them locally. Provides per-game production metrics (rushing yards/game,
receiving yards/game, etc.) that help the model distinguish high-production
college players from low-production ones.

Requires a free API key from https://collegefootballdata.com/key
Set via environment variable CFBD_API_KEY or pass directly.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import polars as pl
import requests

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / ".cache"
COLLEGE_CACHE = CACHE_DIR / "cfbd_player_stats.json"
CACHE_TTL = 7 * 24 * 60 * 60  # 7 days

CFBD_BASE = "https://api.collegefootballdata.com"

# Position groups we care about
_SKILL_POSITIONS = {"QB", "RB", "WR", "TE", "HB", "FB", "ATH"}

# Rate limiting: minimum seconds between CFBD API requests
_MIN_REQUEST_INTERVAL = 2.5  # CFBD free tier; pad generously to avoid 429s
_last_request_time = 0.0


def _get_api_key() -> Optional[str]:
    """Get CFBD API key from environment."""
    return os.environ.get("CFBD_API_KEY")


def _load_cache() -> dict:
    """Load cached college stats if fresh."""
    if COLLEGE_CACHE.exists():
        age = time.time() - COLLEGE_CACHE.stat().st_mtime
        if age < CACHE_TTL:
            with open(COLLEGE_CACHE) as f:
                return json.load(f)
    return {}


def _save_cache(data: dict):
    """Save college stats cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(COLLEGE_CACHE, "w") as f:
        json.dump(data, f)


def _fetch_player_stats(
    player_name: str,
    team: str,
    year: int,
    category: str,
    api_key: str,
    max_retries: int = 5,
) -> list[dict]:
    """Fetch a single player's season stats from CFBD API."""
    global _last_request_time
    for attempt in range(max_retries + 1):
        # Enforce minimum interval between requests to avoid rate limiting
        elapsed = time.time() - _last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
        _last_request_time = time.time()
        resp = requests.get(
            f"{CFBD_BASE}/stats/player/season",
            params={
                "year": year,
                "team": team,
                "category": category,
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "accept": "application/json",
            },
            timeout=15,
        )
        if resp.status_code == 429:
            if attempt < max_retries:
                wait = 5 * (2 ** attempt)  # 5s, 10s, 20s, 40s, 80s
                log.warning("CFBD rate limited (429), retrying in %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            else:
                log.error("CFBD rate limit exceeded after %d retries", max_retries)
                resp.raise_for_status()
        resp.raise_for_status()
        return resp.json()
    return []  # unreachable, but satisfies type checker


def fetch_college_stats_for_team(
    team: str,
    years: list[int],
    api_key: str,
) -> dict[str, dict]:
    """Fetch all skill-position player stats for a college team across years.

    Returns: {player_name: {stat_name: value, ...}}
    """
    categories = ["passing", "rushing", "receiving"]
    player_stats: dict[str, dict] = {}
    # Track distinct seasons each player appeared in to estimate games
    player_seasons: dict[str, set[int]] = {}

    for year in years:
        for cat in categories:
            try:
                results = _fetch_player_stats("", team, year, cat, api_key)
            except Exception:
                log.warning("Failed to fetch %s stats for %s (%d)", cat, team, year, exc_info=True)
                continue

            for entry in results:
                name = entry.get("player", "")
                if not name:
                    continue

                if name not in player_stats:
                    player_stats[name] = {
                        "team": team,
                        "college_games": 0,
                    }
                    player_seasons[name] = set()

                # Track that this player was active in this year
                player_seasons.setdefault(name, set()).add(year)

                stat_type = entry.get("statType", "")
                value = entry.get("stat", "")

                try:
                    val = float(value)
                except (ValueError, TypeError):
                    continue

                key = f"college_{cat}_{stat_type}"
                # Accumulate across years
                player_stats[name][key] = player_stats[name].get(key, 0) + val

    # Estimate games played: ~13 games per college season
    GAMES_PER_SEASON = 13
    for name, seasons in player_seasons.items():
        if name in player_stats:
            player_stats[name]["college_games"] = len(seasons) * GAMES_PER_SEASON

    return player_stats


def fetch_all_college_stats(
    players: list[dict],
    api_key: Optional[str] = None,
    cache: bool = True,
) -> dict[str, dict]:
    """Fetch college stats for a list of players.

    Args:
        players: List of dicts with 'name', 'school', and optionally 'seasons'
                 (list of college years to fetch, defaults to last 3 years)
        api_key: CFBD API key. If None, reads from CFBD_API_KEY env var.
        cache: Whether to use/update cache.

    Returns:
        {player_name: {college_stat_name: value, ...}}
    """
    if api_key is None:
        api_key = _get_api_key()
    if not api_key:
        log.warning("CFBD_API_KEY not set — skipping college stats fetch. "
                     "Get a free key at https://collegefootballdata.com/key")
        return {}

    # CR opus: When the cache is valid, this returns the ENTIRE cached dict
    # regardless of what players were requested. If a second call requests
    # different players, they get stale/irrelevant cached data. The cache
    # key should incorporate the player list or be invalidated accordingly.
    if cache:
        cached = _load_cache()
        if cached:
            return cached

    # Group players by school to minimize API calls
    schools: dict[str, list[dict]] = {}
    for p in players:
        school = p.get("school", "")
        if school:
            schools.setdefault(school, []).append(p)

    all_stats: dict[str, dict] = {}
    # Determine year range
    # CR opus: Hardcoded [2024] fallback — should use current year dynamically.
    # Also, the generator filter (if p.get("seasons")) means if NO players have
    # a "seasons" key, max() will get an empty iterable and raise ValueError.
    current_year = max(p.get("seasons", [2024])[-1] for p in players if p.get("seasons"))
    years = list(range(current_year - 3, current_year + 1))

    total_schools = len(schools)
    for idx, (school, school_players) in enumerate(schools.items(), 1):
        log.info("Fetching %s (%d/%d)...", school, idx, total_schools)
        try:
            team_stats = fetch_college_stats_for_team(school, years, api_key)
            # Match fetched stats to our player list
            for p in school_players:
                name = p["name"]
                # Try exact match first, then last-name match
                if name in team_stats:
                    all_stats[name] = team_stats[name]
                else:
                    # Fuzzy match: prefer first+last match, fall back to last-name-only
                    # if there's exactly one candidate (to avoid ambiguity).
                    parts = name.split() if name else []
                    first = parts[0].lower() if len(parts) > 0 else ""
                    last = parts[-1].lower() if len(parts) > 0 else ""
                    last_matches = []
                    for fetched_name, fstats in team_stats.items():
                        f_parts = fetched_name.split()
                        f_first = f_parts[0].lower() if f_parts else ""
                        f_last = f_parts[-1].lower() if f_parts else ""
                        if f_last == last:
                            if f_first == first:
                                # Full name match — use immediately
                                all_stats[name] = fstats
                                last_matches = []
                                break
                            last_matches.append(fstats)
                    else:
                        # No full-name match found; use last-name only if unambiguous
                        if len(last_matches) == 1:
                            all_stats[name] = last_matches[0]
        except Exception as e:
            print(f"  Warning: failed to fetch stats for {school}: {e}")
            continue

    if cache and all_stats:
        _save_cache(all_stats)

    return all_stats


def build_college_features(
    combine_df: pl.DataFrame,
    draft_picks_df: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """Build college production features for combine participants.

    Uses cached CFBD data if available, otherwise returns empty features.
    Features are per-game to normalize across players with different
    numbers of college seasons.

    Args:
        combine_df: DataFrame with pfr_id, player_name, school columns.
        draft_picks_df: Optional historical draft picks for bridging IDs.

    Returns:
        DataFrame with pfr_id and college feature columns.
    """
    cached = _load_cache()

    if not cached:
        # Return empty features — model handles nulls via XGBoost
        return pl.DataFrame({
            "pfr_id": combine_df["pfr_id"] if "pfr_id" in combine_df.columns else [],
        })

    # Build feature rows
    rows = []
    for row in combine_df.iter_rows(named=True):
        pfr_id = row.get("pfr_id", "")
        name = row.get("player_name", "")

        stats = cached.get(name, {})

        # Compute per-game features
        # CR opus: college_games is initialized to 0 in fetch_college_stats_for_team
        # and never incremented — it's always 0. So max(0, 1) always returns 1,
        # meaning per-game features are actually career totals. The games count
        # is never populated from the API response.
        games = max(stats.get("college_games", 0), 1)  # avoid div by zero

        rows.append({
            "pfr_id": pfr_id,
            "college_rush_ypg": stats.get("college_rushing_YDS", 0) / games,
            "college_rush_td_pg": stats.get("college_rushing_TD", 0) / games,
            "college_rec_ypg": stats.get("college_receiving_YDS", 0) / games,
            "college_rec_td_pg": stats.get("college_receiving_TD", 0) / games,
            "college_pass_ypg": stats.get("college_passing_YDS", 0) / games,
            "college_pass_td_pg": stats.get("college_passing_TD", 0) / games,
            "college_games": float(stats.get("college_games", 0)),
        })

    if not rows:
        return pl.DataFrame({"pfr_id": []})

    return pl.DataFrame(rows)
