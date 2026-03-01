"""
Data loading functions for NFL analysis.

Thin wrappers are provided only where we add filtering logic beyond
what nflreadpy offers natively. For everything else, use nflreadpy
directly (re-exported via __init__.py).
"""

import nflreadpy as nfl
import polars as pl
from . import cache as _cache  # triggers auto-configure on first import


def get_player_stats(seasons=None, weeks=None, summary_level="week"):
    """Load player stats, optionally filtered by weeks.

    Args:
        seasons: Season(s) to load. None=current, True=all, int or list of ints.
        weeks: Optional list of weeks to filter to.
        summary_level: "week" (default), "reg", "post", or "reg+post".
    """
    df = nfl.load_player_stats(seasons, summary_level=summary_level)
    if weeks is not None:
        df = df.filter(pl.col("week").is_in(weeks))
    return df


def get_player_game_log(player_name, seasons=None):
    """Get all weekly game entries for a specific player.

    Matches on player_display_name first, falls back to player_name.
    """
    df = get_player_stats(seasons)
    name_lower = player_name.lower()
    result = df.filter(pl.col("player_display_name").str.to_lowercase() == name_lower)
    if result.height == 0:
        result = df.filter(pl.col("player_name").str.to_lowercase() == name_lower)
    return result.sort(["season", "week"])
