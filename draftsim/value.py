"""Replacement levels, VBD, and VONA valuation math."""

import math
from statistics import mean as _mean

from .config import LeagueConfig
from .draft import DraftState
from .players import Player


def compute_replacement_levels(
    players: list[Player],
    config: LeagueConfig,
) -> dict[str, float]:
    """Compute replacement-level points for each position.

    Replacement level = the projected_total of the Nth-best player at a position,
    where N is the number of starters league-wide plus FLEX adjustment.

    For 12-team: QB1*12=12 starters → replacement = QB13
                 RB2*12=24 starters + ~6 FLEX → replacement = RB30
                 WR2*12=24 starters + ~6 FLEX → replacement = WR30
                 TE1*12=12 starters → replacement = TE13
    """
    num_teams = config.num_teams
    starters = config.starter_slots()
    num_flex = config.num_flex()
    flex_positions = config.flex_positions()

    # Base starters per position
    all_positions = ["QB", "RB", "WR", "TE", "K", "DST"]
    starter_counts = {pos: starters.get(pos, 0) * num_teams for pos in all_positions}

    # Distribute FLEX slots roughly: ~50% RB, ~40% WR, ~10% TE
    # K and DST are NOT flex-eligible
    total_flex = num_flex * num_teams
    flex_split = {"RB": 0.5, "WR": 0.4, "TE": 0.1}
    for pos in flex_positions:
        starter_counts[pos] += int(total_flex * flex_split.get(pos, 0))

    # Group players by position
    by_pos: dict[str, list[Player]] = {pos: [] for pos in all_positions}
    for p in players:
        if p.position in by_pos:
            by_pos[p.position].append(p)

    # Sort each position by projected_total descending
    for pos in by_pos:
        by_pos[pos].sort(key=lambda p: p.projected_total, reverse=True)

    replacement = {}
    for pos, count in starter_counts.items():
        pos_players = by_pos.get(pos, [])
        # Replacement level is the player at the boundary (1-indexed)
        idx = min(count, len(pos_players) - 1)
        if idx >= 0 and pos_players:
            replacement[pos] = pos_players[idx].projected_total
        else:
            replacement[pos] = 0.0

    return replacement


def compute_dynamic_replacement_levels(
    available: list[Player],
    config: LeagueConfig,
    teams: list[list[Player]],
) -> dict[str, float]:
    """Compute replacement levels from *available* players and remaining need.

    Unlike ``compute_replacement_levels`` (which always indexes into the full
    pool), this version accounts for players already drafted and only counts
    unfilled starter+flex demand across all teams.
    """
    num_teams = config.num_teams
    starters = config.starter_slots()
    flex_positions = config.flex_positions()

    all_positions = ["QB", "RB", "WR", "TE", "K", "DST"]

    # Remaining starter need per position across all teams
    remaining_need: dict[str, int] = {pos: 0 for pos in all_positions}
    remaining_flex = 0
    for roster in teams:
        roster_counts: dict[str, int] = {}
        for p in roster:
            roster_counts[p.position] = roster_counts.get(p.position, 0) + 1
        for pos in all_positions:
            have = roster_counts.get(pos, 0)
            required = starters.get(pos, 0)
            remaining_need[pos] += max(0, required - have)
        # FLEX: count surplus flex-eligible beyond starters
        flex_surplus = 0
        for pos in flex_positions:
            have = roster_counts.get(pos, 0)
            required = starters.get(pos, 0)
            flex_surplus += max(0, have - required)
        flex_needed = config.num_flex()
        remaining_flex += max(0, flex_needed - flex_surplus)

    # Distribute remaining flex demand proportionally to available supply
    if remaining_flex > 0:
        flex_avail = {}
        total_flex_avail = 0
        for pos in flex_positions:
            cnt = sum(1 for p in available if p.position == pos)
            flex_avail[pos] = cnt
            total_flex_avail += cnt
        if total_flex_avail > 0:
            for pos in flex_positions:
                share = flex_avail[pos] / total_flex_avail
                remaining_need[pos] += round(remaining_flex * share)

    # Group available players by position, sorted descending
    by_pos: dict[str, list[Player]] = {pos: [] for pos in all_positions}
    for p in available:
        if p.position in by_pos:
            by_pos[p.position].append(p)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda p: p.projected_total, reverse=True)

    replacement: dict[str, float] = {}
    for pos in all_positions:
        count = remaining_need.get(pos, 0)
        pos_players = by_pos.get(pos, [])
        idx = min(count, len(pos_players) - 1) if pos_players else -1
        if idx >= 0:
            replacement[pos] = pos_players[idx].projected_total
        else:
            replacement[pos] = 0.0

    return replacement


def vbd(player: Player, replacement_levels: dict[str, float]) -> float:
    """Value Based Drafting: player value over replacement."""
    return player.projected_total - replacement_levels.get(player.position, 0.0)


def vona(
    state: DraftState,
    team_idx: int,
    position: str,
    adp_order: list[str],
    top_k: int = 3,
) -> float:
    """Value Over Next Available — estimated drop-off at a position.

    Computes the difference between the best available player at a position NOW
    vs. the best likely available at your NEXT pick, based on ADP-predicted removals.

    Args:
        state: Current draft state
        team_idx: Team computing VONA for
        position: Position to evaluate
        adp_order: Player names in ADP order (for estimating who gets taken)

    Returns:
        VONA value: higher means more urgent to draft this position now
    """
    available_at_pos = state.available_at_position(position)
    if not available_at_pos:
        return 0.0

    # Estimate who will be taken before our next pick
    gap = state.picks_until_next(team_idx)
    if gap <= 1:
        return 0.0  # Picking next, no urgency

    # Top-K averaging for a more stable signal (K=3)
    k = min(top_k, len(available_at_pos))
    avg_now = _mean([p.projected_total for p in available_at_pos[:k]])

    # Use ADP to predict which available players get taken
    available_names = {p.name for p in state.available}
    top_k_names = {p.name for p in available_at_pos[:k]}
    predicted_taken = set()
    taken_count = 0
    for name in adp_order:
        if taken_count >= gap - 1:  # gap-1 picks happen between now and our next
            break
        if name in available_names and name not in top_k_names:
            predicted_taken.add(name)
            taken_count += 1

    # Top-K at this position after predicted removals
    remaining = [p for p in available_at_pos if p.name not in predicted_taken]
    if not remaining:
        # All players at this position predicted to be taken — very high urgency
        return avg_now
    k_later = min(k, len(remaining))
    avg_later = _mean([p.projected_total for p in remaining[:k_later]])

    return avg_now - avg_later


# Risk profile names → multiplier on the variance bonus.
# Positive = prefer ceiling (aggressive), negative = prefer floor (safe).
RISK_PROFILES = {
    "safe": -1.0,
    "balanced": 0.0,
    "aggressive": 1.0,
}


def variance_bonus(
    player: Player,
    roster: list[Player],
    current_round: int,
    total_rounds: int,
    risk_profile: str = "balanced",
) -> float:
    """Score modifier based on projection variance and roster context.

    Three factors combine:
      1. Round-based risk tolerance — early rounds favour floor, late rounds
         favour ceiling.
      2. Portfolio diversity — if existing roster players at this position are
         all safe (narrow spread), nudge toward ceiling picks and vice-versa.
      3. User risk profile override — shifts the whole curve toward safe or
         aggressive.

    Returns a bonus (positive or negative) to add to a strategy score.
    The magnitude is scaled to be meaningful relative to VBD values (~5-15%
    of a typical top-player VBD).
    """
    if player.upside <= 0:
        return 0.0

    profile_mult = RISK_PROFILES.get(risk_profile, 0.0)

    # --- 1. Round-based risk curve ---
    # Maps round fraction [0, 1] to a preference in [-1, 1].
    # Early rounds → negative (prefer floor), late → positive (prefer ceiling).
    round_frac = (current_round - 1) / max(total_rounds - 1, 1)
    round_pref = (round_frac - 0.35) * 2  # ~-0.7 early, ~+1.3 late, 0 around round 6

    # --- 2. Portfolio diversity at this position ---
    pos_roster = [p for p in roster if p.position == player.position]
    if pos_roster:
        avg_upside = sum(p.upside for p in pos_roster) / len(pos_roster)
        player_upside = player.upside
        # If roster is safe (low avg_upside) → prefer ceiling (positive nudge)
        # If roster is volatile → prefer floor (negative nudge)
        if avg_upside > 0:
            diversity_pref = (player_upside - avg_upside) / avg_upside
            diversity_pref = max(-1.0, min(1.0, diversity_pref))  # clamp
        else:
            diversity_pref = 0.0
    else:
        diversity_pref = 0.0

    # --- Combine factors ---
    # Base weight: fraction of upside to apply (keeps bonus proportional)
    base = player.upside * 0.10  # 10% of spread as max swing

    combined_pref = round_pref * 0.4 + diversity_pref * 0.3 + profile_mult * 0.3
    return base * combined_pref


def positional_scarcity(
    state: DraftState,
    position: str,
    config: LeagueConfig,
) -> float:
    """Ratio of remaining startable players to remaining league-wide need.

    Lower values = more scarce = higher priority.
    """
    available = state.available_at_position(position)
    if not available:
        return float("inf")

    starters = config.starter_slots()
    num_teams = config.num_teams
    flex_split = {"RB": 0.5, "WR": 0.4, "TE": 0.1}

    total_need = starters.get(position, 0) * num_teams
    total_need += int(config.num_flex() * num_teams * flex_split.get(position, 0))

    # Subtract already-drafted starters across all teams
    drafted_at_pos = sum(
        state.team_position_count(i, position) for i in range(num_teams)
    )
    remaining_need = max(1, total_need - drafted_at_pos)

    # Count available players with positive VBD (above replacement)
    startable = len(available)  # simplified: all available are somewhat startable

    return startable / remaining_need


def vona_weight(current_round: int, total_rounds: int) -> float:
    """Round-adaptive VONA weight — higher early when tiers are steep."""
    frac = (current_round - 1) / max(total_rounds - 1, 1)
    return 0.7 - 0.3 * frac  # 0.7 early -> 0.4 late


def marginal_value_discount(
    player: Player,
    roster: list[Player],
    config: LeagueConfig,
) -> float:
    """Discount for picking a player whose position is already well-stocked.

    Returns a multiplier:
      1.0 — filling a starter slot
      0.6 — filling a FLEX slot
      0.3 — bench depth
    """
    starters = config.starter_slots()
    pos = player.position
    have = sum(1 for p in roster if p.position == pos)
    starter_need = starters.get(pos, 0)

    if have < starter_need:
        return 1.0

    # Check if this would fill a FLEX slot
    if pos in config.flex_positions():
        flex_surplus = 0
        for fpos in config.flex_positions():
            fhave = sum(1 for p in roster if p.position == fpos)
            freq = starters.get(fpos, 0)
            flex_surplus += max(0, fhave - freq)
        if flex_surplus < config.num_flex():
            return 0.6

    return 0.3
