"""Trade calculator: data model, valuation, and evaluation engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from draftsim.config import LeagueConfig
from draftsim.players import Player
from draftsim.value import compute_replacement_levels, vbd

log = logging.getLogger("trade")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TradeTeam:
    """A team in the league with its current roster."""

    team_id: str
    name: str
    roster: list[Player] = field(default_factory=list)
    starters: list[str] = field(default_factory=list)  # player IDs of starters


@dataclass
class TradeSession:
    """Holds league state for trade analysis."""

    league_id: str
    platform: str  # "sleeper" | "yahoo"
    sport: str  # "nfl" | "mlb"
    config: LeagueConfig
    teams: list[TradeTeam] = field(default_factory=list)
    all_players: list[Player] = field(default_factory=list)
    replacement_levels: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.all_players and not self.replacement_levels:
            self.replacement_levels = compute_replacement_levels(
                self.all_players, self.config,
            )

    @property
    def rostered_ids(self) -> set[str]:
        """Set of sleeper_ids on any roster."""
        ids: set[str] = set()
        for team in self.teams:
            for p in team.roster:
                if p.sleeper_id:
                    ids.add(p.sleeper_id)
        return ids

    @property
    def free_agents(self) -> list[Player]:
        """Players not on any roster, sorted by VORP descending."""
        rostered = self.rostered_ids
        fa = [p for p in self.all_players if p.sleeper_id not in rostered]
        fa.sort(key=lambda p: vbd(p, self.replacement_levels), reverse=True)
        return fa


# ---------------------------------------------------------------------------
# Roster needs analysis
# ---------------------------------------------------------------------------

def roster_needs(team: TradeTeam, config: LeagueConfig) -> dict[str, dict]:
    """Analyse positional needs for a team.

    Returns dict[position -> {"have": int, "need": int, "surplus": int,
                              "severity": str}]
    severity: "critical" (below starters), "ok", "strong" (depth)
    """
    starters = config.starter_slots()
    all_positions = list(dict.fromkeys(
        list(starters.keys()) + list(config.position_caps.keys())
    ))

    counts: dict[str, int] = {}
    for p in team.roster:
        counts[p.position] = counts.get(p.position, 0) + 1

    result: dict[str, dict] = {}
    for pos in all_positions:
        have = counts.get(pos, 0)
        need = starters.get(pos, 0)
        surplus = have - need
        if surplus < 0:
            severity = "critical"
        elif surplus == 0:
            severity = "ok"
        else:
            severity = "strong"
        result[pos] = {
            "have": have,
            "need": need,
            "surplus": surplus,
            "severity": severity,
        }

    return result


def best_at_position(
    players: list[Player],
    replacement_levels: dict[str, float],
) -> dict[str, list[dict]]:
    """Best available players at each position with VORP.

    Returns dict[position -> list of {player, vorp}] sorted by vorp desc.
    """
    by_pos: dict[str, list[tuple[Player, float]]] = {}
    for p in players:
        v = vbd(p, replacement_levels)
        by_pos.setdefault(p.position, []).append((p, v))

    result: dict[str, list[dict]] = {}
    for pos, plist in by_pos.items():
        plist.sort(key=lambda t: t[1], reverse=True)
        result[pos] = [
            {"player": p, "vorp": round(v, 1)}
            for p, v in plist[:10]
        ]

    return result


# ---------------------------------------------------------------------------
# Trade evaluation
# ---------------------------------------------------------------------------

def player_trade_value(
    player: Player,
    replacement_levels: dict[str, float],
    ros_fraction: float = 1.0,
) -> float:
    """Compute a player's trade value (VORP scaled to rest-of-season).

    Args:
        player: The player to value.
        replacement_levels: Positional replacement baselines.
        ros_fraction: Fraction of season remaining (0.0 to 1.0).
    """
    return vbd(player, replacement_levels) * ros_fraction


def _starter_quality(
    roster: list[Player],
    config: LeagueConfig,
    replacement_levels: dict[str, float],
) -> dict[str, float]:
    """Total VORP of best starters at each position for a roster."""
    starters = config.starter_slots()
    by_pos: dict[str, list[Player]] = {}
    for p in roster:
        by_pos.setdefault(p.position, []).append(p)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda p: p.projected_total, reverse=True)

    quality: dict[str, float] = {}
    for pos, need in starters.items():
        pos_players = by_pos.get(pos, [])
        top = pos_players[:need]
        quality[pos] = sum(vbd(p, replacement_levels) for p in top)

    return quality


def evaluate_trade(
    session: TradeSession,
    team_a_id: str,
    team_b_id: str,
    a_gives: list[str],
    b_gives: list[str],
    ros_fraction: float = 1.0,
) -> dict:
    """Evaluate a proposed trade.

    Args:
        session: Current TradeSession with league data.
        team_a_id: Team A's identifier.
        team_b_id: Team B's identifier.
        a_gives: Player sleeper_ids that Team A sends away.
        b_gives: Player sleeper_ids that Team B sends away.
        ros_fraction: Fraction of season remaining.

    Returns:
        Dict with fairness score, per-team impact, and summary.
    """
    team_a = next((t for t in session.teams if t.team_id == team_a_id), None)
    team_b = next((t for t in session.teams if t.team_id == team_b_id), None)
    if not team_a or not team_b:
        raise ValueError("Team not found in session")

    repl = session.replacement_levels

    # Resolve players being traded
    a_players = [p for p in team_a.roster if p.sleeper_id in set(a_gives)]
    b_players = [p for p in team_b.roster if p.sleeper_id in set(b_gives)]

    # Total value exchanged
    a_gives_value = sum(player_trade_value(p, repl, ros_fraction) for p in a_players)
    b_gives_value = sum(player_trade_value(p, repl, ros_fraction) for p in b_players)

    # Fairness: positive = favours team A (receives more), negative = favours team B
    fairness_delta = b_gives_value - a_gives_value

    # Roster impact: before and after
    a_roster_before = list(team_a.roster)
    a_roster_after = [p for p in a_roster_before if p.sleeper_id not in set(a_gives)] + b_players
    b_roster_before = list(team_b.roster)
    b_roster_after = [p for p in b_roster_before if p.sleeper_id not in set(b_gives)] + a_players

    a_quality_before = _starter_quality(a_roster_before, session.config, repl)
    a_quality_after = _starter_quality(a_roster_after, session.config, repl)
    b_quality_before = _starter_quality(b_roster_before, session.config, repl)
    b_quality_after = _starter_quality(b_roster_after, session.config, repl)

    a_needs_before = roster_needs(
        TradeTeam(team_id=team_a_id, name=team_a.name, roster=a_roster_before),
        session.config,
    )
    a_needs_after = roster_needs(
        TradeTeam(team_id=team_a_id, name=team_a.name, roster=a_roster_after),
        session.config,
    )
    b_needs_before = roster_needs(
        TradeTeam(team_id=team_b_id, name=team_b.name, roster=b_roster_before),
        session.config,
    )
    b_needs_after = roster_needs(
        TradeTeam(team_id=team_b_id, name=team_b.name, roster=b_roster_after),
        session.config,
    )

    # Total starter VORP change per team
    a_total_before = sum(a_quality_before.values())
    a_total_after = sum(a_quality_after.values())
    b_total_before = sum(b_quality_before.values())
    b_total_after = sum(b_quality_after.values())

    # Build per-player value details
    def _player_detail(p: Player) -> dict:
        return {
            "name": p.name,
            "position": p.position,
            "team": p.team,
            "sleeper_id": p.sleeper_id,
            "projected_ppg": round(p.projected_ppg, 1),
            "projected_total": round(p.projected_total, 1),
            "total_floor": round(p.total_floor, 1),
            "total_ceiling": round(p.total_ceiling, 1),
            "pos_rank": p.pos_rank,
            "age": p.age,
            "trade_value": round(player_trade_value(p, repl, ros_fraction), 1),
        }

    # Determine winner
    a_net = a_total_after - a_total_before
    b_net = b_total_after - b_total_before
    if abs(a_net - b_net) < 5:
        winner = "even"
    elif a_net > b_net:
        winner = team_a_id
    else:
        winner = team_b_id

    # Build summary reasons
    reasons = []
    for pos in session.config.starter_slots():
        a_delta = a_quality_after.get(pos, 0) - a_quality_before.get(pos, 0)
        b_delta = b_quality_after.get(pos, 0) - b_quality_before.get(pos, 0)
        if abs(a_delta) > 5:
            direction = "upgrades" if a_delta > 0 else "downgrades"
            reasons.append(f"{team_a.name} {direction} at {pos}")
        if abs(b_delta) > 5:
            direction = "upgrades" if b_delta > 0 else "downgrades"
            reasons.append(f"{team_b.name} {direction} at {pos}")

    # Check needs filled/created
    for pos, need_after in a_needs_after.items():
        need_before = a_needs_before.get(pos, {})
        if need_before.get("severity") == "critical" and need_after.get("severity") != "critical":
            reasons.append(f"{team_a.name} fills {pos} need")
        elif need_before.get("severity") != "critical" and need_after.get("severity") == "critical":
            reasons.append(f"{team_a.name} creates {pos} need")
    for pos, need_after in b_needs_after.items():
        need_before = b_needs_before.get(pos, {})
        if need_before.get("severity") == "critical" and need_after.get("severity") != "critical":
            reasons.append(f"{team_b.name} fills {pos} need")
        elif need_before.get("severity") != "critical" and need_after.get("severity") == "critical":
            reasons.append(f"{team_b.name} creates {pos} need")

    return {
        "fairness_delta": round(fairness_delta, 1),
        "winner": winner,
        "reasons": reasons,
        "team_a": {
            "team_id": team_a_id,
            "name": team_a.name,
            "gives": [_player_detail(p) for p in a_players],
            "receives": [_player_detail(p) for p in b_players],
            "gives_total_value": round(a_gives_value, 1),
            "receives_total_value": round(b_gives_value, 1),
            "starter_vorp_before": round(a_total_before, 1),
            "starter_vorp_after": round(a_total_after, 1),
            "roster_impact": round(a_net, 1),
            "needs_before": a_needs_before,
            "needs_after": a_needs_after,
        },
        "team_b": {
            "team_id": team_b_id,
            "name": team_b.name,
            "gives": [_player_detail(p) for p in b_players],
            "receives": [_player_detail(p) for p in a_players],
            "gives_total_value": round(b_gives_value, 1),
            "receives_total_value": round(a_gives_value, 1),
            "starter_vorp_before": round(b_total_before, 1),
            "starter_vorp_after": round(b_total_after, 1),
            "roster_impact": round(b_net, 1),
            "needs_before": b_needs_before,
            "needs_after": b_needs_after,
        },
    }


# ---------------------------------------------------------------------------
# Sleeper roster resolution
# ---------------------------------------------------------------------------

def resolve_sleeper_rosters(
    rosters: list[dict],
    users: list[dict],
    all_sleeper_players: dict[str, dict],
    our_players: list[Player],
    id_to_player: dict[str, Player],
) -> list[TradeTeam]:
    """Convert Sleeper API roster data into TradeTeam objects.

    Args:
        rosters: Raw roster dicts from Sleeper /league/{id}/rosters.
        users: Raw user dicts from Sleeper /league/{id}/users.
        all_sleeper_players: Full Sleeper player DB (from fetch_all_players).
        our_players: Our Player objects with projections.
        id_to_player: Mapping of sleeper_id -> Player (from build_player_index).
    """
    # Build owner_id -> display_name mapping
    owner_names: dict[str, str] = {}
    for u in users:
        oid = u.get("user_id", "")
        name = u.get("display_name", "") or u.get("username", "") or oid
        if oid:
            owner_names[oid] = name

    teams: list[TradeTeam] = []
    for roster in rosters:
        roster_id = str(roster.get("roster_id", ""))
        owner_id = roster.get("owner_id", "") or ""
        name = owner_names.get(owner_id, f"Team {roster_id}")
        player_ids = roster.get("players") or []
        starters = roster.get("starters") or []

        # Resolve player IDs to Player objects
        resolved: list[Player] = []
        for pid in player_ids:
            pid = str(pid)
            if pid in id_to_player:
                resolved.append(id_to_player[pid])
            else:
                # Create minimal Player from Sleeper player DB
                info = all_sleeper_players.get(pid, {})
                if isinstance(info, dict):
                    first = info.get("first_name", "") or ""
                    last = info.get("last_name", "") or ""
                    pos = info.get("position", "") or "?"
                    team = info.get("team", "") or ""
                    if pos == "DEF":
                        pos = "DST"
                    p = Player(
                        name=f"{first} {last}".strip() or f"Player {pid}",
                        position=pos,
                        team=team,
                        projected_ppg=0.0,
                        projected_games=0.0,
                        projected_total=0.0,
                        pos_rank=999,
                        sleeper_id=pid,
                    )
                    resolved.append(p)

        teams.append(TradeTeam(
            team_id=roster_id,
            name=name,
            roster=resolved,
            starters=[str(s) for s in starters],
        ))

    return teams
