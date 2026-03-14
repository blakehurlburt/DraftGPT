#!/usr/bin/env python3
"""Fetch and cache college football stats from the CFBD API.

Requires a free API key from https://collegefootballdata.com/key
Set via: export CFBD_API_KEY=your_key_here

This script fetches career college stats for all combine participants
across recent years and caches them locally for the projection pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import nflreadpy as nfl
from nfldata.college_features import fetch_all_college_stats, _load_cache


def main():
    # Load combine data for recent years to get player/school pairs
    years = list(range(2020, 2027))
    print(f"Loading combine data for {years[0]}-{years[-1]}...")
    combine = nfl.load_combine(years)

    skill_pos = {"QB", "RB", "WR", "TE", "HB", "FB"}
    combine = combine.filter(combine["pos"].is_in(skill_pos))
    combine = combine.filter(combine["pfr_id"].is_not_null())

    # Build player list for the fetch function
    players = []
    for row in combine.iter_rows(named=True):
        school = row.get("school", "")
        name = row.get("player_name", "")
        season = row.get("season", 2024)
        if school and name:
            players.append({
                "name": name,
                "school": school,
                "seasons": list(range(season - 4, season)),
            })

    print(f"Found {len(players)} skill-position combine participants")

    # Check existing cache
    cached = _load_cache()
    if cached:
        print(f"Cache exists with {len(cached)} players. Refreshing...")

    # Fetch (uses cache if fresh, otherwise hits API)
    stats = fetch_all_college_stats(players, cache=True)
    print(f"Done! Cached stats for {len(stats)} players")

    # Show sample
    for name, s in list(stats.items())[:3]:
        print(f"  {name}: {s}")


if __name__ == "__main__":
    main()
