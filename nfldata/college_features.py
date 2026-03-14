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
import os
import time
from pathlib import Path
from typing import Optional

import polars as pl
import requests

CACHE_DIR = Path(__file__).parent.parent / ".cache"
COLLEGE_CACHE = CACHE_DIR / "cfbd_player_stats.json"
CACHE_TTL = 7 * 24 * 60 * 60  # 7 days

CFBD_BASE = "https://api.collegefootballdata.com"

# Position groups we care about
_SKILL_POSITIONS = {"QB", "RB", "WR", "TE", "HB", "FB", "ATH"}


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
) -> list[dict]:
    """Fetch a single player's season stats from CFBD API."""
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
        time.sleep(2)
        return _fetch_player_stats(player_name, team, year, category, api_key)
    resp.raise_for_status()
    return resp.json()


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

    for year in years:
        for cat in categories:
            try:
                results = _fetch_player_stats("", team, year, cat, api_key)
            except Exception:
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

                stat_type = entry.get("statType", "")
                value = entry.get("stat", "")

                try:
                    val = float(value)
                except (ValueError, TypeError):
                    continue

                key = f"college_{cat}_{stat_type}"
                # Accumulate across years
                player_stats[name][key] = player_stats[name].get(key, 0) + val

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
        return {}

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
    current_year = max(p.get("seasons", [2024])[-1] for p in players if p.get("seasons"))
    years = list(range(current_year - 3, current_year + 1))

    for school, school_players in schools.items():
        try:
            team_stats = fetch_college_stats_for_team(school, years, api_key)
            # Match fetched stats to our player list
            for p in school_players:
                name = p["name"]
                # Try exact match first, then last-name match
                if name in team_stats:
                    all_stats[name] = team_stats[name]
                else:
                    # Try matching by last name
                    last = name.split()[-1] if name else ""
                    for fetched_name, stats in team_stats.items():
                        if fetched_name.split()[-1] == last:
                            all_stats[name] = stats
                            break
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
