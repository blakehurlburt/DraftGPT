"""Fetch MiLB stats from the MLB Stats API and cache locally.

Populates data/milb_cache/ with player rosters, year-by-year stats,
and draft data.  Designed to be run once (takes ~30-60 min for a full
10-year window) and then incrementally for new seasons.

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


def collect_player_ids(seasons: range, verbose: bool = True) -> set[int]:
    """Collect all unique MLB API player IDs across MiLB levels and seasons."""
    all_ids: set[int] = set()

    levels = {k: v for k, v in SPORT_IDS.items() if k != "MLB"}
    tasks = [(level, sport_id, season)
             for level, sport_id in levels.items()
             for season in seasons]

    if verbose:
        print(f"\nStep 1: Collecting player rosters ({len(tasks)} level-season combos)...")

    for level, sport_id, season in tqdm(tasks, desc="Rosters", disable=not verbose):
        try:
            players = fetch_players_at_level(sport_id, season)
            for p in players:
                pid = p.get("id")
                if pid:
                    all_ids.add(pid)
        except Exception as e:
            if verbose:
                tqdm.write(f"  Warning: {level} {season} failed: {e}")

    if verbose:
        print(f"  Unique MiLB players: {len(all_ids)}")

    return all_ids


def fetch_all_stats(player_ids: set[int], verbose: bool = True):
    """Fetch year-by-year stats for all players (both hitting and pitching)."""
    if verbose:
        print(f"\nStep 2: Fetching stats for {len(player_ids)} players...")

    for pid in tqdm(sorted(player_ids), desc="Stats", disable=not verbose):
        try:
            fetch_player_stats(pid, group="hitting")
        except Exception:
            pass
        try:
            fetch_player_stats(pid, group="pitching")
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

    # Step 1: Collect all player IDs from rosters
    player_ids = collect_player_ids(seasons)

    # Step 2: Fetch per-player stats
    if not args.skip_stats:
        fetch_all_stats(player_ids)
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
