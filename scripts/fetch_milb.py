"""Fetch MiLB stats from the MLB Stats API and cache locally.

Populates data/milb_cache/ with player rosters, year-by-year stats,
and draft data.  Only fetches stats for fantasy-relevant players
(AAA, AA, and MLB-level) to keep runtime reasonable (~30-45 min).

Usage:
    python scripts/fetch_milb.py [--start-year 2015] [--end-year 2025]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from mlbdata.milb_api import (
    SPORT_IDS,
    fetch_draft,
    fetch_player_stats,
    fetch_players_at_level,
)
from mlbdata.id_mapping import build_id_map, save_id_map

# Only fetch rosters from these levels — AAA and MLB are where
# fantasy-relevant players are.  AA is included to capture top
# prospects a year or two out.
ROSTER_LEVELS = ["AAA", "AA", "MLB"]

# For per-player stats, fetch these MiLB levels.
STAT_LEVELS = ["AAA", "AA", "High-A", "Single-A"]


def collect_player_ids(seasons: range, verbose: bool = True) -> dict[int, str]:
    """Collect unique MLB API player IDs from fantasy-relevant levels.

    Returns:
        Dict mapping player_id -> primary position type ("pitcher" or "hitter").
    """
    players_info: dict[int, str] = {}

    tasks = [(level, SPORT_IDS[level], season)
             for level in ROSTER_LEVELS
             for season in seasons]

    if verbose:
        print(f"\nStep 1: Collecting player rosters ({len(tasks)} level-season combos)...")

    pitcher_pos = {"P", "SP", "RP", "TWP"}
    for level, sport_id, season in tqdm(tasks, desc="Rosters", disable=not verbose):
        try:
            players = fetch_players_at_level(sport_id, season)
            for p in players:
                pid = p.get("id")
                if not pid or pid in players_info:
                    continue
                pos = p.get("primaryPosition", {}).get("abbreviation", "")
                ptype = "pitcher" if pos in pitcher_pos else "hitter"
                players_info[pid] = ptype
        except Exception as e:
            if verbose:
                tqdm.write(f"  Warning: {level} {season} failed: {e}")

    if verbose:
        hitters = sum(1 for v in players_info.values() if v == "hitter")
        pitchers = sum(1 for v in players_info.values() if v == "pitcher")
        print(f"  Unique players: {len(players_info)} ({hitters} hitters, {pitchers} pitchers)")

    return players_info


def fetch_all_stats(players_info: dict[int, str], verbose: bool = True):
    """Fetch year-by-year stats for all players.

    Uses player position to only fetch the relevant stat group
    (hitting OR pitching), cutting API calls in half.
    """
    if verbose:
        print(f"\nStep 2: Fetching stats for {len(players_info)} players...")
        print(f"  Levels: {STAT_LEVELS}")
        est_calls = len(players_info) * len(STAT_LEVELS)
        est_time = est_calls * 0.15 / 60
        print(f"  Estimated: ~{est_calls} API calls, ~{est_time:.0f} min")

    for pid in tqdm(sorted(players_info.keys()), desc="Stats", disable=not verbose):
        ptype = players_info[pid]
        group = "pitching" if ptype == "pitcher" else "hitting"
        try:
            fetch_player_stats(pid, group=group, levels=STAT_LEVELS)
        except Exception:
            pass


def fetch_drafts(seasons: range, verbose: bool = True):
    """Fetch draft data for each year in the range."""
    if verbose:
        print(f"\nStep 3: Fetching draft data ({len(list(seasons))} years)...")

    for year in tqdm(list(seasons), desc="Drafts", disable=not verbose):
        try:
            picks = fetch_draft(year)
            if verbose:
                tqdm.write(f"  {year}: {len(picks)} picks")
        except Exception as e:
            if verbose:
                tqdm.write(f"  Warning: {year} draft failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fetch MiLB stats from MLB Stats API")
    parser.add_argument("--start-year", type=int, default=2015,
                        help="First season to fetch (default: 2015)")
    parser.add_argument("--end-year", type=int, default=2025,
                        help="Last season to fetch (default: 2025)")
    parser.add_argument("--skip-stats", action="store_true",
                        help="Only fetch rosters and drafts, skip per-player stats")
    parser.add_argument("--skip-id-map", action="store_true",
                        help="Skip building the Lahman ID map")
    args = parser.parse_args()

    seasons = range(args.start_year, args.end_year + 1)
    print(f"Fetching MiLB data for {args.start_year}-{args.end_year}")

    # Step 1: Collect player IDs from AA+ rosters
    players_info = collect_player_ids(seasons)

    # Step 2: Fetch per-player stats
    if not args.skip_stats:
        fetch_all_stats(players_info)
    else:
        print("\nSkipping per-player stats (--skip-stats)")

    # Step 3: Fetch draft data
    fetch_drafts(seasons)

    # Step 4: Build ID mapping
    if not args.skip_id_map:
        print("\nStep 4: Building Lahman <-> API ID mapping...")
        id_map = build_id_map(seasons, verbose=True)
        save_id_map(id_map)
        print(f"  Saved {len(id_map)} mappings")
    else:
        print("\nSkipping ID mapping (--skip-id-map)")

    print("\nDone! Cache at data/milb_cache/")


if __name__ == "__main__":
    main()
