"""Maps draft data to draftsim DraftState.

Handles both Sleeper API data and manual draft mode.
"""

from __future__ import annotations

import re
import unicodedata
from statistics import mean as _mean

from draftsim.config import LeagueConfig
from draftsim.draft import DraftState
from draftsim.players import Player

from mlbdata.fantasy_scoring import compute_batter_fpts, compute_pitcher_fpts
from .scoring import sleeper_stats_to_fantasy_points


def _normalize(name: str) -> str:
    """Lowercase, strip accents/suffixes, collapse whitespace."""
    # Decompose unicode and drop combining marks (accents)
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = name.lower().strip()
    name = re.sub(r"\b(jr\.?|sr\.?|ii|iii|iv|v)\b", "", name)
    name = re.sub(r"[^a-z ]", "", name)
    return " ".join(name.split())


def build_player_index(
    our_players: list[Player],
    sleeper_players: dict[str, dict],
) -> dict[str, Player]:
    """Match our Player objects to Sleeper player IDs.

    Uses normalized "first last" + position matching.
    Sets player.sleeper_id on matched players.

    Returns:
        dict mapping sleeper_id -> Player
    """
    # CR opus: If two players share the same normalized name AND position (e.g., two
    # "Mike Williams" WRs), the second one silently overwrites the first in the dict.
    # This causes one of them to never be matched from Sleeper data.
    # Build lookup: (normalized_name, position) -> Player
    lookup: dict[tuple[str, str], Player] = {}
    for p in our_players:
        key = (_normalize(p.name), p.position)
        lookup[key] = p

    id_to_player: dict[str, Player] = {}

    # Build DST lookup by team abbreviation
    dst_lookup: dict[str, Player] = {}
    for p in our_players:
        if p.position == "DST":
            # player_id is "DST_KC" -> team abbr is "KC"
            team_abbr = p.name.replace(" DST", "")
            dst_lookup[team_abbr] = p

    for sid, info in sleeper_players.items():
        if not isinstance(info, dict):
            continue
        first = info.get("first_name", "") or ""
        last = info.get("last_name", "") or ""
        pos = info.get("position", "") or ""

        # Match DST: Sleeper uses position "DEF" and team abbreviation
        if pos == "DEF":
            team = info.get("team", "") or ""
            if team in dst_lookup:
                player = dst_lookup[team]
                player.sleeper_id = sid
                id_to_player[sid] = player
            continue

        if not first or not last or pos not in ("QB", "RB", "WR", "TE", "K"):
            continue

        norm = _normalize(f"{first} {last}")
        key = (norm, pos)
        if key in lookup:
            player = lookup[key]
            player.sleeper_id = sid
            if info.get("years_exp", 99) == 0:
                player.is_rookie = True
            id_to_player[sid] = player

    return id_to_player


# Sleeper slot type -> our position mapping
_SLOT_MAP = {
    "QB": "QB",
    "RB": "RB",
    "WR": "WR",
    "TE": "TE",
    "FLEX": "FLEX",
    "SUPER_FLEX": "SUPER_FLEX",
    "REC_FLEX": "FLEX",
    "K": "K",
    "DEF": "DST",
}

# Slots we ignore (IDP, bench)
_IGNORED_SLOTS = {"BN", "DL", "LB", "DB", "IDP_FLEX"}


def config_from_sleeper_meta(meta: dict) -> LeagueConfig:
    """Build LeagueConfig from Sleeper draft metadata."""
    settings = meta.get("settings", {})
    num_teams = settings.get("teams", 12)
    rounds = settings.get("rounds", 15)

    # Count lineup slots from roster_positions
    roster_positions = meta.get("settings", {}).get("roster_positions", [])
    if not roster_positions:
        # Fallback: use slots dict
        slots = settings.get("slots", {})
        roster_positions = []
        for slot_type, count in slots.items():
            roster_positions.extend([slot_type] * count)

    lineup: dict[str, int] = {}
    for slot in roster_positions:
        mapped = _SLOT_MAP.get(slot)
        if mapped:
            lineup[mapped] = lineup.get(mapped, 0) + 1

    # Handle SUPER_FLEX as FLEX for our purposes
    if "SUPER_FLEX" in lineup:
        lineup["FLEX"] = lineup.get("FLEX", 0) + lineup.pop("SUPER_FLEX")

    # If no lineup parsed, use defaults
    if not lineup:
        lineup = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}

    # Calculate roster_size from rounds
    roster_size = rounds

    return LeagueConfig(
        num_teams=num_teams,
        roster_size=roster_size,
        lineup=lineup,
    )


def _make_placeholder(pick: dict) -> Player:
    """Create a zero-projection placeholder for unmatched picks (K/DST/IDP)."""
    meta = pick.get("metadata", {})
    first = meta.get("first_name", "Unknown")
    last = meta.get("last_name", "Player")
    # CR opus: Defaulting unknown positions to "RB" will inflate the team's RB count
    # in positional cap tracking, potentially causing legitimate RB picks to be blocked
    # by caps. Consider a dedicated "UNKNOWN" sentinel or "BN" (bench) position.
    pos = meta.get("position", "RB")  # Default to RB so it doesn't break caps
    team = meta.get("team", "")

    # Map Sleeper positions to our positions
    if pos == "DEF":
        pos = "DST"
    elif pos not in ("QB", "RB", "WR", "TE", "K", "DST"):
        pos = "RB"  # Placeholder position for draft state advancement

    return Player(
        name=f"{first} {last}",
        position=pos,
        team=team,
        projected_ppg=0.0,
        projected_games=0.0,
        projected_total=0.0,
        pos_rank=999,
        sleeper_id=str(pick.get("player_id", "")),
    )


def rebuild_draft_state(
    config: LeagueConfig,
    players: list[Player],
    picks: list[dict],
    id_to_player: dict[str, Player],
) -> DraftState:
    """Create fresh DraftState and replay Sleeper picks.

    Unmatched players (K/DST/IDP) get zero-projection placeholders
    that still advance the draft state correctly.
    """
    state = DraftState.create(config, list(players))

    # Sort picks by pick_no to replay in order
    sorted_picks = sorted(picks, key=lambda p: p.get("pick_no", 0))

    for pick in sorted_picks:
        if state.is_complete:
            break

        player_id = str(pick.get("player_id", ""))
        matched = id_to_player.get(player_id)

        if matched and matched in state.available:
            state.make_pick(matched)
        else:
            # Unmatched player — create placeholder and inject into available
            placeholder = _make_placeholder(pick)
            state.available.append(placeholder)
            state.make_pick(placeholder)

    return state


def default_config_for_sport(
    sport: str, num_teams: int = 12, roster_size: int = 15,
) -> LeagueConfig:
    """Build a LeagueConfig with sport-appropriate defaults."""
    if sport == "mlb":
        lineup = {
            "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1,
            "OF": 3, "FLEX": 1, "SP": 2, "RP": 2,
        }
        position_caps = {
            "C": 3, "1B": 3, "2B": 3, "3B": 3, "SS": 3,
            "OF": 6, "SP": 6, "RP": 4, "DH": 2,
        }
        # UTIL slot: all batters eligible
        flex_eligible = ["C", "1B", "2B", "3B", "SS", "OF", "DH"]
        return LeagueConfig(
            num_teams=num_teams,
            roster_size=roster_size,
            lineup=lineup,
            position_caps=position_caps,
            flex_eligible=flex_eligible,
        )
    # NFL defaults
    lineup = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1}
    return LeagueConfig(
        num_teams=num_teams,
        roster_size=roster_size,
        lineup=lineup,
    )


def attach_sleeper_projections(
    players: list[Player],
    sleeper_projections: dict[str, dict],
    scoring: dict[str, float] | None = None,
) -> int:
    """Populate Sleeper projection fields on players with a sleeper_id.

    Also saves model backup values (``_model_*`` fields) so they can be
    restored when toggling back to model projections.

    Returns the number of players that received Sleeper projections.
    """
    # Compute position-average floor/ceiling ratios from model data
    # so we can synthesize estimated floor/ceiling for Sleeper projections.
    _pos_ratios: dict[str, tuple[float, float]] = {}  # pos -> (floor_ratio, ceil_ratio)
    by_pos: dict[str, list[Player]] = {}
    for p in players:
        by_pos.setdefault(p.position, []).append(p)
    for pos, plist in by_pos.items():
        valid = [p for p in plist if p.projected_total > 0]
        if valid:
            floor_ratio = _mean(
                p.total_floor / p.projected_total
                for p in valid if p.total_floor > 0
            ) if any(p.total_floor > 0 for p in valid) else 0.7
            ceil_ratio = _mean(
                p.total_ceiling / p.projected_total
                for p in valid if p.total_ceiling > 0
            ) if any(p.total_ceiling > 0 for p in valid) else 1.3
            _pos_ratios[pos] = (floor_ratio, ceil_ratio)
        else:
            _pos_ratios[pos] = (0.7, 1.3)

    matched = 0
    for p in players:
        # Save model backup (always, even if no sleeper data)
        p._model_projected_total = p.projected_total
        p._model_projected_ppg = p.projected_ppg
        p._model_projected_games = p.projected_games
        p._model_total_floor = p.total_floor
        p._model_total_ceiling = p.total_ceiling

        if not p.sleeper_id:
            continue
        stats = sleeper_projections.get(p.sleeper_id)
        if not stats:
            continue

        pts = sleeper_stats_to_fantasy_points(stats, scoring)
        if pts <= 0:
            continue

        # CR opus: Hardcoded 17.0 games fallback is NFL-specific. If this is ever used
        # for MLB players, the fallback should be ~162 (or sport-appropriate).
        # Keep our model's projected games — Sleeper doesn't project GP
        games = p.projected_games if p.projected_games > 0 else 17.0

        p.sleeper_projected_total = pts
        p.sleeper_projected_games = games
        p.sleeper_projected_ppg = pts / games if games > 0 else 0.0
        matched += 1

    # CR opus: Storing mutable state as a function attribute is fragile — it's
    # essentially a hidden global. If attach_sleeper_projections is called multiple
    # times (e.g., across sessions), the ratios from the last call overwrite the
    # previous ones, affecting all sessions sharing this module.
    # Store position ratios on a module-level variable for swap_projection_source
    attach_sleeper_projections._pos_ratios = _pos_ratios
    return matched


def attach_fangraphs_projections(
    players: list[Player],
    fg_projections: list[dict],
) -> int:
    """Populate FanGraphs projection fields on MLB players matched by name.

    Also saves model backup values so they can be restored when toggling.

    Returns the number of players that received FanGraphs projections.
    """
    # FanGraphs position mapping: minpos contains comma-separated positions
    _FG_POS_MAP = {
        "C": "C", "1B": "1B", "2B": "2B", "3B": "3B", "SS": "SS",
        "LF": "OF", "CF": "OF", "RF": "OF", "OF": "OF", "DH": "DH",
    }
    _PITCHER_POSITIONS = {"SP", "RP"}

    # CR opus: Name-only matching (no position) means a batter and pitcher with the
    # same name (e.g., Shohei Ohtani) will collide — whichever appears last in the
    # fg_projections list wins. This could assign pitcher stats to a batter or vice versa.
    # Build lookup: normalized name -> FanGraphs entry
    # For batters, also store primary position for matching
    fg_lookup: dict[str, dict] = {}
    for entry in fg_projections:
        name = entry.get("PlayerName", "")
        if not name:
            continue
        key = _normalize(name)
        fg_lookup[key] = entry

    # Compute position-average floor/ceiling ratios from model data
    _pos_ratios: dict[str, tuple[float, float]] = {}
    by_pos: dict[str, list[Player]] = {}
    for p in players:
        by_pos.setdefault(p.position, []).append(p)
    for pos, plist in by_pos.items():
        valid = [p for p in plist if p.projected_total > 0]
        if valid:
            floor_ratio = _mean(
                p.total_floor / p.projected_total
                for p in valid if p.total_floor > 0
            ) if any(p.total_floor > 0 for p in valid) else 0.7
            ceil_ratio = _mean(
                p.total_ceiling / p.projected_total
                for p in valid if p.total_ceiling > 0
            ) if any(p.total_ceiling > 0 for p in valid) else 1.3
            _pos_ratios[pos] = (floor_ratio, ceil_ratio)
        else:
            _pos_ratios[pos] = (0.7, 1.3)

    matched = 0
    for p in players:
        # Save model backup (always, even if no FanGraphs data)
        p._model_projected_total = p.projected_total
        p._model_projected_ppg = p.projected_ppg
        p._model_projected_games = p.projected_games
        p._model_total_floor = p.total_floor
        p._model_total_ceiling = p.total_ceiling

        key = _normalize(p.name)
        entry = fg_lookup.get(key)
        if not entry:
            continue

        fg_type = entry.get("_fg_type", "bat")
        if fg_type == "pit":
            # Convert IP to outs for scoring
            ip = float(entry.get("IP", 0) or 0)
            stats = {
                "W": float(entry.get("W", 0) or 0),
                "L": float(entry.get("L", 0) or 0),
                "SV": float(entry.get("SV", 0) or 0),
                "SO": float(entry.get("SO", 0) or 0),
                "IPouts": ip * 3,
                "ER": float(entry.get("ER", 0) or 0),
                "H": float(entry.get("H", 0) or 0),
                "BB": float(entry.get("BB", 0) or 0),
                "HBP": float(entry.get("HBP", 0) or 0),
                "CG": float(entry.get("CG", 0) or 0),
                "SHO": float(entry.get("SHO", 0) or 0),
            }
            pts = compute_pitcher_fpts(stats)
        else:
            stats = {
                "R": float(entry.get("R", 0) or 0),
                "HR": float(entry.get("HR", 0) or 0),
                "RBI": float(entry.get("RBI", 0) or 0),
                "SB": float(entry.get("SB", 0) or 0),
                "CS": float(entry.get("CS", 0) or 0),
                "BB": float(entry.get("BB", 0) or 0),
                "HBP": float(entry.get("HBP", 0) or 0),
                "H": float(entry.get("H", 0) or 0),
                "2B": float(entry.get("2B", 0) or 0),
                "3B": float(entry.get("3B", 0) or 0),
                "SO": float(entry.get("SO", 0) or 0),
                "GIDP": float(entry.get("GDP", 0) or 0),
            }
            pts = compute_batter_fpts(stats)

        if pts <= 0:
            continue

        games = float(entry.get("G", 0) or 0)
        if fg_type == "pit":
            games = float(entry.get("GS", 0) or entry.get("G", 0) or 0)

        p.fangraphs_projected_total = round(pts, 1)
        p.fangraphs_projected_games = games
        p.fangraphs_projected_ppg = round(pts / games, 2) if games > 0 else 0.0
        matched += 1

    # Store position ratios for swap_projection_source
    attach_fangraphs_projections._pos_ratios = _pos_ratios
    return matched


def swap_projection_source(players: list[Player], source: str) -> None:
    """Swap the active projection values on all players.

    Args:
        source: ``"sleeper"``, ``"fangraphs"``, or ``"model"`` to restore.
    """
    pos_ratios = getattr(attach_sleeper_projections, "_pos_ratios", None)
    if pos_ratios is None:
        pos_ratios = getattr(attach_fangraphs_projections, "_pos_ratios", {})

    # CR opus: Players that DON'T have the selected projection source data (e.g.,
    # no Sleeper projection) retain whatever projection values they had before the swap.
    # This mixes projection sources within the same player pool, which could produce
    # misleading relative rankings.
    if source == "sleeper":
        for p in players:
            if p.sleeper_projected_total > 0:
                p.projected_total = p.sleeper_projected_total
                p.projected_ppg = p.sleeper_projected_ppg
                # Keep model's projected_games — Sleeper doesn't project GP
                fr, cr = pos_ratios.get(p.position, (0.7, 1.3))
                p.total_floor = round(p.projected_total * fr, 1)
                p.total_ceiling = round(p.projected_total * cr, 1)
    elif source == "fangraphs":
        fg_ratios = getattr(attach_fangraphs_projections, "_pos_ratios", pos_ratios)
        for p in players:
            if p.fangraphs_projected_total > 0:
                p.projected_total = p.fangraphs_projected_total
                p.projected_ppg = p.fangraphs_projected_ppg
                p.projected_games = p.fangraphs_projected_games
                fr, cr = fg_ratios.get(p.position, (0.7, 1.3))
                p.total_floor = round(p.projected_total * fr, 1)
                p.total_ceiling = round(p.projected_total * cr, 1)
    elif source == "model":
        for p in players:
            # CR opus: Accessing p._model_projected_total will raise AttributeError if
            # attach_sleeper_projections or attach_fangraphs_projections was never called
            # (the _model_* attributes are only set in those functions).
            if p._model_projected_total > 0:
                p.projected_total = p._model_projected_total
                p.projected_ppg = p._model_projected_ppg
                p.projected_games = p._model_projected_games
                p.total_floor = p._model_total_floor
                p.total_ceiling = p._model_total_ceiling

    # Re-sort and recompute pos_rank
    players.sort(key=lambda p: p.projected_total, reverse=True)
    pos_counts: dict[str, int] = {}
    for p in players:
        pos_counts[p.position] = pos_counts.get(p.position, 0) + 1
        p.pos_rank = pos_counts[p.position]


def rebuild_from_manual_picks(
    config: LeagueConfig,
    players: list[Player],
    picks: list[dict],
) -> DraftState:
    """Create DraftState and replay manual picks by player name.

    Each pick dict has metadata.first_name / metadata.last_name (or a
    combined player_name key) used to match against the player pool.
    """
    state = DraftState.create(config, list(players))

    # Build name lookup for fast matching
    name_lookup: dict[str, Player] = {}
    for p in players:
        name_lookup[p.name.lower()] = p

    sorted_picks = sorted(picks, key=lambda p: p.get("pick_no", 0))

    for pick in sorted_picks:
        if state.is_complete:
            break
        meta = pick.get("metadata", {})
        pname = f"{meta.get('first_name', '')} {meta.get('last_name', '')}".strip()
        matched = name_lookup.get(pname.lower())

        if matched and matched in state.available:
            state.make_pick(matched)
        else:
            # Create placeholder for unknown picks
            placeholder = _make_placeholder(pick)
            state.available.append(placeholder)
            state.make_pick(placeholder)

    return state
