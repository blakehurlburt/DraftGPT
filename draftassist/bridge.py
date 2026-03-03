"""Maps Sleeper draft data to draftsim DraftState."""

from __future__ import annotations

import re

from draftsim.config import LeagueConfig
from draftsim.draft import DraftState
from draftsim.players import Player


def _normalize(name: str) -> str:
    """Lowercase, strip suffixes (Jr., III, etc.), collapse whitespace."""
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
    # Build lookup: (normalized_name, position) -> Player
    lookup: dict[tuple[str, str], Player] = {}
    for p in our_players:
        key = (_normalize(p.name), p.position)
        lookup[key] = p

    id_to_player: dict[str, Player] = {}

    for sid, info in sleeper_players.items():
        if not isinstance(info, dict):
            continue
        first = info.get("first_name", "") or ""
        last = info.get("last_name", "") or ""
        pos = info.get("position", "") or ""
        if not first or not last or pos not in ("QB", "RB", "WR", "TE"):
            continue

        norm = _normalize(f"{first} {last}")
        key = (norm, pos)
        if key in lookup:
            player = lookup[key]
            player.sleeper_id = sid
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
}

# Slots we ignore (kicker, defense, IDP, bench)
_IGNORED_SLOTS = {"BN", "K", "DEF", "DL", "LB", "DB", "IDP_FLEX"}


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
    pos = meta.get("position", "RB")  # Default to RB so it doesn't break caps
    team = meta.get("team", "")

    # Map non-standard positions to something DraftState can handle
    if pos not in ("QB", "RB", "WR", "TE"):
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
