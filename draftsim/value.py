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

    # Base starters per position (derive from config to support NFL + MLB)
    all_positions = list(dict.fromkeys(
        list(starters.keys()) + list(config.position_caps.keys())
    ))
    starter_counts = {pos: starters.get(pos, 0) * num_teams for pos in all_positions}

    # Distribute FLEX slots across eligible positions
    total_flex = num_flex * num_teams
    # CR opus: Hardcoded NFL-specific FLEX distribution (RB 50%, WR 40%, TE 10%).
    # For MLB drafts that use this function, these NFL splits are applied incorrectly
    # since MLB positions won't match these keys and will fall through to even_share.
    # Works by accident but the intent is unclear — consider making sport-aware.
    nfl_flex_split = {"RB": 0.5, "WR": 0.4, "TE": 0.1}
    # Use NFL weights when applicable, otherwise distribute evenly
    even_share = 1.0 / len(flex_positions) if flex_positions else 0
    for pos in flex_positions:
        share = nfl_flex_split.get(pos, even_share)
        # CR opus: int() truncates — int(12 * 0.1) = 1, so total_flex=12 distributes as
        # RB:6 + WR:4 + TE:1 = 11, losing 1 slot. Use round() instead of int().
        starter_counts[pos] = starter_counts.get(pos, 0) + int(total_flex * share)

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
        # CR opus: Off-by-one risk. If count=12 (12 starters), idx=12 gives the 13th
        # player (0-indexed), which is correct for replacement level. But the comment says
        # "1-indexed" which is misleading — this IS 0-indexed and the logic is actually
        # correct (replacement = first non-starter). The comment should be fixed.
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

    # Derive from config to support NFL + MLB
    all_positions = list(dict.fromkeys(
        list(starters.keys()) + list(config.position_caps.keys())
    ))

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


# VORP is the standard name for this metric in the fantasy community
vorp = vbd


def compute_last_starter_levels(
    available: list[Player],
    config: LeagueConfig,
    teams: list[list[Player]],
) -> dict[str, float]:
    """Projected points of the worst player who will end up as a starter.

    Unlike replacement levels (first waiver-wire player = N+1th), this returns
    the Nth player — the last starter slot being filled.  FLEX starters ARE
    counted since a player in a FLEX slot is still a starter.
    """
    starters = config.starter_slots()
    flex_positions = config.flex_positions()
    all_positions = list(dict.fromkeys(
        list(starters.keys()) + list(config.position_caps.keys())
    ))

    # Remaining starter need per position across all teams (including FLEX)
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

    last_starter: dict[str, float] = {}
    for pos in all_positions:
        count = remaining_need.get(pos, 0)
        pos_players = by_pos.get(pos, [])
        if count <= 0 or not pos_players:
            last_starter[pos] = 0.0
        else:
            # Last starter = the (count)th best available (0-indexed: count-1)
            idx = min(count - 1, len(pos_players) - 1)
            last_starter[pos] = pos_players[idx].projected_total

    return last_starter


def vols(player: Player, last_starter_levels: dict[str, float]) -> float:
    """Value Over Last Starter — how much better than the worst opponent starter.

    VOLS measures the difference between this player's projection and the
    worst player at this position who will end up as a starter for an opponent.
    Floored at 0 (a player below the last starter has no starter-tier value).
    """
    return max(0.0, player.projected_total - last_starter_levels.get(player.position, 0.0))


def vbd_score(
    vorp_val: float,
    vona_val: float,
    vols_val: float,
) -> float:
    """Composite VBD Score aggregating VORP, VONA, and VOLS.

    Simple sum of the three components (each already floored at 0).
    VORP provides the base value signal, VONA adds positional urgency,
    and VOLS adds starter-tier scarcity context.
    """
    return max(0.0, vorp_val) + max(0.0, vona_val) + max(0.0, vols_val)


def _pick_probability(
    adp_position: int,
    gap_size: int,
    current_round: int,
    total_rounds: int,
) -> float:
    """Probability that a player at *adp_position* is taken within *gap_size* picks.

    Uses a sigmoid centred on the gap boundary.  Uncertainty (spread) widens
    as the draft progresses because late-round ADP is noisier.

    Args:
        adp_position: 1-indexed position in the ADP-ordered available list
        gap_size: number of picks before our next turn
        current_round: 1-indexed current round
        total_rounds: total rounds in the draft
    """
    round_frac = (current_round - 1) / max(total_rounds - 1, 1)
    spread = 3.0 + 9.0 * round_frac          # 3 early → 12 late
    midpoint = gap_size - 1                   # picks that happen in the gap
    # CR opus: When gap_size == 1 (opponent picks once before our next turn), midpoint=0
    # and this returns 0.0, meaning we predict zero chance anyone is taken. But with
    # gap_size=1 there IS one pick happening. The check should be `if midpoint < 0`
    # or the sigmoid should handle midpoint=0 naturally (it would give ~0.5 probability
    # for adp_position=0, which is reasonable).
    if midpoint <= 0:
        return 0.0
    x = (adp_position - midpoint) / max(spread, 0.1)
    return 1.0 / (1.0 + math.exp(x))         # high when adp_position < midpoint


def _need_adjusted_adp(
    state: DraftState,
    team_idx: int,
    adp_order: list[str],
) -> list[str]:
    """Re-order *adp_order* by boosting players who match other teams' needs.

    For each team picking in the gap before *team_idx*'s next turn, check
    ``team_needs()`` and pull matching-position players earlier.
    """
    gap = state.picks_until_next(team_idx)
    if gap <= 1:
        return adp_order

    # Collect position demand from teams picking in the gap
    pos_demand: dict[str, float] = {}
    for i in range(state.current_pick + 1,
                   min(state.current_pick + gap, len(state.pick_order))):
        other = state.pick_order[i]
        if other == team_idx:
            break
        needs = state.team_needs(other)
        for pos, count in needs.items():
            if pos == "FLEX":
                continue
            pos_demand[pos] = pos_demand.get(pos, 0) + count

    if not pos_demand:
        return adp_order

    # Build a player-name → position lookup from available
    name_to_pos: dict[str, str] = {p.name: p.position for p in state.available}

    # Score each name: lower = earlier.  Original index is base;
    # subtract a bonus for positions in demand.
    max_demand = max(pos_demand.values()) if pos_demand else 1
    scored: list[tuple[float, str]] = []
    for idx, name in enumerate(adp_order):
        pos = name_to_pos.get(name, "")
        demand = pos_demand.get(pos, 0)
        # Boost proportional to demand: up to 30% of original index
        boost = (demand / max_demand) * 0.30 * (idx + 1) if demand else 0
        scored.append((idx - boost, name))

    scored.sort(key=lambda t: t[0])
    return [name for _, name in scored]


def vona(
    state: DraftState,
    team_idx: int,
    position: str,
    adp_order: list[str],
    top_k: int = 3,
    current_round: int | None = None,
    total_rounds: int | None = None,
) -> float:
    """Value Over Next Available — estimated drop-off at a position.

    Computes the difference between the best available player at a position NOW
    vs. the best likely available at your NEXT pick, using probability-weighted
    removal and need-aware ADP adjustment.

    Args:
        state: Current draft state
        team_idx: Team computing VONA for
        position: Position to evaluate
        adp_order: Player names in ADP order (for estimating who gets taken)
        top_k: Number of top players to average for stability
        current_round: 1-indexed current round (defaults to state.current_round)
        total_rounds: Total rounds in draft (defaults to state.config.num_rounds)

    Returns:
        VONA value: higher means more urgent to draft this position now
    """
    available_at_pos = state.available_at_position(position)
    if not available_at_pos:
        return 0.0

    gap = state.picks_until_next(team_idx)
    if gap <= 1:
        return 0.0

    if current_round is None:
        current_round = state.current_round
    if total_rounds is None:
        total_rounds = state.config.num_rounds

    # Top-K averaging for a more stable signal
    k = min(top_k, len(available_at_pos))
    # CR opus: available_at_position() docstring claims "sorted by projected_total desc"
    # but the implementation is just a list comprehension filter with no explicit sort.
    # This works only because state.available preserves the initial load order (sorted desc).
    # If anyone ever re-orders state.available (e.g. by ADP), this top-K will be wrong.
    avg_now = _mean([p.projected_total for p in available_at_pos[:k]])

    # Need-aware ADP adjustment
    adjusted_adp = _need_adjusted_adp(state, team_idx, adp_order)

    # Build ADP position lookup for available players (1-indexed)
    available_names = {p.name for p in state.available}
    adp_pos_map: dict[str, int] = {}
    adp_idx = 0
    for name in adjusted_adp:
        if name in available_names:
            adp_idx += 1
            adp_pos_map[name] = adp_idx

    # Probability-weighted expected value after the gap
    # Each player at this position has a probability of being taken
    remaining_values: list[tuple[float, float]] = []  # (projected, survival_prob)
    for p in available_at_pos:
        adp_position = adp_pos_map.get(p.name, len(available_names))
        prob_taken = _pick_probability(adp_position, gap, current_round, total_rounds)
        survival = 1.0 - prob_taken
        remaining_values.append((p.projected_total, survival))

    # Expected top-K value after removals using survival probabilities
    # Weight each player's contribution by their survival probability
    total_survival = sum(s for _, s in remaining_values)
    if total_survival < 0.01:
        return avg_now  # all likely taken — very high urgency

    # Compute expected average of top-K survivors
    # Sort by projected descending (already sorted), take weighted top-K
    weighted_sum = 0.0
    weight_sum = 0.0
    for val, surv in remaining_values:
        if weight_sum >= k:
            break
        contrib = min(surv, k - weight_sum)
        weighted_sum += val * contrib
        weight_sum += contrib

    if weight_sum < 0.01:
        return avg_now

    avg_later = weighted_sum / weight_sum
    return max(0.0, avg_now - avg_later)


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
    # CR opus: Hardcoded NFL flex_split again (same as compute_replacement_levels).
    # Should be a shared constant or derived from config to avoid drift.
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
