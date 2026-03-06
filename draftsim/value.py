"""Replacement levels, VBD, and VONA valuation math."""

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
    starter_counts = {pos: starters.get(pos, 0) * num_teams for pos in ["QB", "RB", "WR", "TE"]}

    # Distribute FLEX slots roughly: ~50% RB, ~40% WR, ~10% TE
    total_flex = num_flex * num_teams
    flex_split = {"RB": 0.5, "WR": 0.4, "TE": 0.1}
    for pos in flex_positions:
        starter_counts[pos] += int(total_flex * flex_split.get(pos, 0))

    # Group players by position
    by_pos: dict[str, list[Player]] = {"QB": [], "RB": [], "WR": [], "TE": []}
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


def vbd(player: Player, replacement_levels: dict[str, float]) -> float:
    """Value Based Drafting: player value over replacement."""
    return player.projected_total - replacement_levels.get(player.position, 0.0)


def vona(
    state: DraftState,
    team_idx: int,
    position: str,
    adp_order: list[str],
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

    best_now = available_at_pos[0].projected_total

    # Estimate who will be taken before our next pick
    gap = state.picks_until_next(team_idx)
    if gap <= 1:
        return 0.0  # Picking next, no urgency

    # Use ADP to predict which available players get taken
    available_names = {p.name for p in state.available}
    predicted_taken = set()
    taken_count = 0
    for name in adp_order:
        if taken_count >= gap - 1:  # gap-1 picks happen between now and our next
            break
        if name in available_names and name != available_at_pos[0].name:
            predicted_taken.add(name)
            taken_count += 1

    # Best available at this position after predicted removals
    remaining = [p for p in available_at_pos if p.name not in predicted_taken]
    if not remaining:
        # All players at this position predicted to be taken — very high urgency
        return best_now
    best_later = remaining[0].projected_total

    return best_now - best_later


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
