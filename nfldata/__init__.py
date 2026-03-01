"""
nfldata - NFL data loading and analysis library.

Configures filesystem caching on import, then re-exports nflreadpy
functions alongside our own helpers.
"""

# Our value-add wrappers
from .loader import get_player_stats, get_player_game_log
from .stats import (
    player_season_summary,
    compare_players,
    top_performers,
    PASSING_COLS,
    RUSHING_COLS,
    RECEIVING_COLS,
    FANTASY_COLS,
)
from .cache import configure_cache, clear_cache

# Re-export nflreadpy functions directly (no need for thin wrappers)
from nflreadpy import (
    load_player_stats,
    load_team_stats,
    load_pbp,
    load_rosters,
    load_rosters_weekly,
    load_schedules,
    load_teams,
    load_players,
    load_snap_counts,
    load_nextgen_stats,
    load_injuries,
    load_combine,
    load_depth_charts,
    load_draft_picks,
    load_contracts,
    load_trades,
    load_officials,
    load_pfr_advstats,
    load_ftn_charting,
    load_participation,
    load_ff_opportunity,
    load_ff_playerids,
    load_ff_rankings,
    get_current_season,
    get_current_week,
)
