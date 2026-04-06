"""Yahoo Fantasy Sports API client via yfpy.

Requires a Yahoo Developer app with consumer key/secret.
Set YAHOO_CONSUMER_KEY and YAHOO_CONSUMER_SECRET as environment variables
or provide them at connect time.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from draftsim.config import LeagueConfig
from draftsim.players import Player

log = logging.getLogger("yahoo")

_TOKEN_DIR = Path(".cache") / "yahoo"

# Yahoo game codes by sport
GAME_CODES = {
    "nfl": "nfl",
    "mlb": "mlb",
}


def _get_query(
    league_id: str,
    sport: str,
    consumer_key: str | None = None,
    consumer_secret: str | None = None,
    access_token: dict | None = None,
):
    """Create a YahooFantasySportsQuery instance.

    Args:
        league_id: Yahoo league ID (numeric).
        sport: "nfl" or "mlb".
        consumer_key: OAuth consumer key (falls back to YAHOO_CONSUMER_KEY env var).
        consumer_secret: OAuth consumer secret (falls back to YAHOO_CONSUMER_SECRET env var).
        access_token: Existing OAuth token dict to reuse (skips browser auth).
    """
    from yfpy.query import YahooFantasySportsQuery

    key = consumer_key or os.environ.get("YAHOO_CONSUMER_KEY", "")
    secret = consumer_secret or os.environ.get("YAHOO_CONSUMER_SECRET", "")

    if not key or not secret:
        raise ValueError(
            "Yahoo OAuth credentials required. Set YAHOO_CONSUMER_KEY and "
            "YAHOO_CONSUMER_SECRET environment variables or provide them."
        )

    game_code = GAME_CODES.get(sport, "nfl")

    _TOKEN_DIR.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        league_id=league_id,
        game_code=game_code,
        yahoo_consumer_key=key,
        yahoo_consumer_secret=secret,
        env_file_location=_TOKEN_DIR,
        save_token_data_to_env_file=True,
        browser_callback=True,
    )

    if access_token:
        kwargs["yahoo_access_token_json"] = access_token

    return YahooFantasySportsQuery(**kwargs)


def fetch_yahoo_league_info(
    league_id: str,
    sport: str,
    consumer_key: str | None = None,
    consumer_secret: str | None = None,
    access_token: dict | None = None,
) -> dict:
    """Fetch league metadata (name, settings, teams).

    Returns a dict with keys: name, num_teams, settings, roster_positions, teams.
    """
    query = _get_query(league_id, sport, consumer_key, consumer_secret, access_token)

    league_meta = query.get_league_metadata()
    settings = query.get_league_settings()
    teams = query.get_league_teams()

    # Parse roster positions from settings
    roster_positions = []
    if settings.roster_positions:
        for rp in settings.roster_positions:
            pos = rp.position if hasattr(rp, "position") else str(rp)
            count = int(rp.count) if hasattr(rp, "count") else 1
            for _ in range(count):
                roster_positions.append(pos)

    # Parse stat modifiers for scoring
    stat_modifiers = {}
    if settings.stat_modifiers:
        mods = settings.stat_modifiers
        if hasattr(mods, "stats"):
            for stat in mods.stats:
                stat_id = str(stat.stat_id) if hasattr(stat, "stat_id") else str(stat)
                value = float(stat.value) if hasattr(stat, "value") else 0
                stat_modifiers[stat_id] = value

    team_list = []
    for t in (teams or []):
        team_list.append({
            "team_id": str(t.team_id),
            "team_key": str(t.team_key) if t.team_key else "",
            "name": t.name or f"Team {t.team_id}",
            "manager": t.managers[0].nickname if t.managers else "",
        })

    return {
        "name": league_meta.name if league_meta else league_id,
        "num_teams": len(team_list),
        "roster_positions": roster_positions,
        "stat_modifiers": stat_modifiers,
        "teams": team_list,
    }


def fetch_yahoo_rosters(
    league_id: str,
    sport: str,
    team_keys: list[str],
    consumer_key: str | None = None,
    consumer_secret: str | None = None,
    access_token: dict | None = None,
) -> dict[str, list[dict]]:
    """Fetch rosters for each team.

    Returns dict mapping team_id -> list of player dicts.
    """
    query = _get_query(league_id, sport, consumer_key, consumer_secret, access_token)

    rosters: dict[str, list[dict]] = {}
    for team_key in team_keys:
        # Extract team_id from team_key (format: "game.l.league.t.team_id")
        team_id = team_key.rsplit(".", 1)[-1] if "." in team_key else team_key

        try:
            players = query.get_team_roster_player_info_by_week(team_id)
        except Exception as e:
            log.warning("Failed to fetch roster for team %s: %s", team_id, e)
            players = []

        roster = []
        for p in (players or []):
            roster.append({
                "player_id": str(p.player_id) if p.player_id else "",
                "player_key": str(p.player_key) if hasattr(p, "player_key") else "",
                "name": p.name.full if hasattr(p, "name") and p.name else (p.full_name or ""),
                "first_name": p.first_name or "",
                "last_name": p.last_name or "",
                "position": p.primary_position or p.display_position or "",
                "team": p.editorial_team_abbr or "",
                "status": p.status or "",
                "selected_position": (
                    p.selected_position_value
                    if hasattr(p, "selected_position_value") else ""
                ),
            })

        rosters[team_id] = roster

    return rosters


def yahoo_config_from_positions(
    roster_positions: list[str],
    num_teams: int,
    sport: str,
) -> LeagueConfig:
    """Build a LeagueConfig from Yahoo roster position list."""
    # Map Yahoo position names to our internal names
    yahoo_to_internal = {
        # NFL
        "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "K": "K",
        "DEF": "DST", "D": "DST",
        "W/R/T": "FLEX", "W/R": "FLEX", "W/T": "FLEX",
        "Q/W/R/T": "FLEX",  # superflex
        # MLB
        "C": "C", "1B": "1B", "2B": "2B", "3B": "3B", "SS": "SS",
        "LF": "OF", "CF": "OF", "RF": "OF", "OF": "OF",
        "Util": "FLEX", "SP": "SP", "RP": "RP", "P": "PFLEX",
        # Ignore
        "BN": None, "IR": None, "IR+": None, "IL": None, "IL+": None,
        "DL": None, "DL+": None, "NA": None,
    }

    lineup: dict[str, int] = {}
    roster_size = 0
    for pos in roster_positions:
        roster_size += 1
        mapped = yahoo_to_internal.get(pos)
        if mapped:
            lineup[mapped] = lineup.get(mapped, 0) + 1

    if not lineup:
        from .bridge import default_config_for_sport
        return default_config_for_sport(sport, num_teams=num_teams)

    return LeagueConfig(
        num_teams=num_teams,
        roster_size=roster_size,
        lineup=lineup,
    )


def resolve_yahoo_rosters(
    league_info: dict,
    rosters: dict[str, list[dict]],
    our_players: list[Player],
) -> list:
    """Convert Yahoo roster data into TradeTeam objects.

    Matches Yahoo players to our projection data by name/position.
    """
    from .bridge import _normalize
    from .trade import TradeTeam

    # Build lookup: (normalized_name, position) -> Player
    lookup: dict[tuple[str, str], Player] = {}
    for p in our_players:
        key = (_normalize(p.name), p.position)
        lookup[key] = p

    teams = []
    for team_info in league_info["teams"]:
        team_id = team_info["team_id"]
        name = team_info["name"]
        raw_roster = rosters.get(team_id, [])

        resolved: list[Player] = []
        for rp in raw_roster:
            yahoo_name = rp["name"]
            yahoo_pos = rp["position"]
            yahoo_team = rp["team"]

            # Try to match to our player data
            key = (_normalize(yahoo_name), yahoo_pos)
            matched = lookup.get(key)

            if matched:
                # Use a copy-like approach: use the matched player
                resolved.append(matched)
            else:
                # Create minimal player
                p = Player(
                    name=yahoo_name or f"Player {rp['player_id']}",
                    position=yahoo_pos,
                    team=yahoo_team,
                    projected_ppg=0.0,
                    projected_games=0.0,
                    projected_total=0.0,
                    pos_rank=999,
                    sleeper_id=f"yahoo_{rp['player_id']}",
                )
                resolved.append(p)

        starters = [
            f"yahoo_{rp['player_id']}" for rp in raw_roster
            if rp.get("selected_position", "BN") not in ("BN", "IR", "IL", "DL", "NA")
        ]

        teams.append(TradeTeam(
            team_id=team_id,
            name=name,
            roster=resolved,
            starters=starters,
        ))

    return teams
