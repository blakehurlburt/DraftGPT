"""Draft strategies — each implements pick(state, team_idx) -> Player."""

from __future__ import annotations

from .config import LeagueConfig
from .draft import DraftState
from .players import Player
from .value import (
    compute_dynamic_replacement_levels,
    compute_replacement_levels,
    marginal_value_discount,
    vbd,
    variance_bonus,
    vona,
    vona_weight,
)


def _eligible(state: DraftState, team_idx: int) -> list[Player]:
    """Get eligible players respecting position caps."""
    return [
        p for p in state.available
        if state.can_draft_position(team_idx, p.position)
    ]


def _force_need_pick(state: DraftState, team_idx: int, eligible: list[Player]) -> Player | None:
    """If roster constraints demand a specific position, pick best there.

    Late-round safety valve: if remaining picks <= unfilled starter slots,
    must start filling needs.
    """
    needs = state.team_needs(team_idx)
    remaining = state.picks_remaining(team_idx)

    # Count mandatory needs (not FLEX since it's flexible)
    mandatory = {k: v for k, v in needs.items() if k != "FLEX"}
    mandatory_count = sum(mandatory.values())

    if mandatory_count >= remaining:
        # Must draft for need — find most urgent
        for pos in mandatory:
            pos_eligible = [p for p in eligible if p.position == pos]
            if pos_eligible:
                return max(pos_eligible, key=lambda p: p.projected_total)

    return None


def _var_bonus(player, state, team_idx, risk_profile):
    """Convenience wrapper for variance_bonus with state context."""
    roster = state.teams[team_idx] if team_idx < len(state.teams) else []
    return variance_bonus(
        player, roster, state.current_round, state.config.num_rounds, risk_profile
    )


def pick_bpa(state: DraftState, team_idx: int, **kwargs) -> Player:
    """Best Player Available — highest projected_total."""
    eligible = _eligible(state, team_idx)
    forced = _force_need_pick(state, team_idx, eligible)
    if forced:
        return forced
    risk = kwargs.get("risk_profile", "balanced")
    return max(eligible, key=lambda p: p.projected_total + _var_bonus(p, state, team_idx, risk))


def _dynamic_repl(state: DraftState, players: list[Player] | None = None) -> dict[str, float]:
    """Compute dynamic replacement levels from current draft state."""
    return compute_dynamic_replacement_levels(
        state.available, state.config, state.teams,
    )


def pick_vbd(state: DraftState, team_idx: int, players: list[Player] | None = None, **kwargs) -> Player:
    """Value Based Drafting — highest value over replacement."""
    eligible = _eligible(state, team_idx)
    forced = _force_need_pick(state, team_idx, eligible)
    if forced:
        return forced

    replacement = _dynamic_repl(state, players)
    risk = kwargs.get("risk_profile", "balanced")
    roster = state.teams[team_idx] if team_idx < len(state.teams) else []

    def _score(p):
        discount = marginal_value_discount(p, roster, state.config)
        return vbd(p, replacement) * discount + _var_bonus(p, state, team_idx, risk)

    return max(eligible, key=_score)


def pick_vona(
    state: DraftState,
    team_idx: int,
    adp_order: list[str] | None = None,
    players: list[Player] | None = None,
    **kwargs,
) -> Player:
    """VONA — scores ALL eligible players using cached per-position VONA."""
    eligible = _eligible(state, team_idx)
    forced = _force_need_pick(state, team_idx, eligible)
    if forced:
        return forced

    if adp_order is None:
        adp_order = [p.name for p in sorted(state.available, key=lambda p: p.projected_total, reverse=True)]

    replacement = _dynamic_repl(state, players)
    risk = kwargs.get("risk_profile", "balanced")
    roster = state.teams[team_idx] if team_idx < len(state.teams) else []
    weight = vona_weight(state.current_round, state.config.num_rounds)

    # Cache VONA per position once
    positions = {p.position for p in eligible}
    pos_vona_cache = {pos: vona(state, team_idx, pos, adp_order) for pos in positions}

    def _score(p):
        discount = marginal_value_discount(p, roster, state.config)
        return (vbd(p, replacement) * discount
                + pos_vona_cache.get(p.position, 0.0) * weight
                + _var_bonus(p, state, team_idx, risk))

    return max(eligible, key=_score)


def pick_zero_rb(
    state: DraftState,
    team_idx: int,
    players: list[Player] | None = None,
    **kwargs,
) -> Player:
    """Zero-RB — avoid RBs in rounds 1-4, load up on WR/TE/QB early."""
    eligible = _eligible(state, team_idx)
    forced = _force_need_pick(state, team_idx, eligible)
    if forced:
        return forced

    current_round = state.current_round

    risk = kwargs.get("risk_profile", "balanced")

    if current_round <= 4:
        # Avoid RBs in early rounds
        non_rb = [p for p in eligible if p.position != "RB"]
        if non_rb:
            replacement = _dynamic_repl(state, players)
            roster = state.teams[team_idx] if team_idx < len(state.teams) else []
            return max(non_rb, key=lambda p: (
                vbd(p, replacement) * marginal_value_discount(p, roster, state.config)
                + _var_bonus(p, state, team_idx, risk)
            ))

    # After round 4, use VBD for remaining picks
    return pick_vbd(state, team_idx, players=players, risk_profile=risk)


def pick_robust_rb(
    state: DraftState,
    team_idx: int,
    players: list[Player] | None = None,
    **kwargs,
) -> Player:
    """Robust-RB — prioritize RBs in rounds 1-3."""
    eligible = _eligible(state, team_idx)
    forced = _force_need_pick(state, team_idx, eligible)
    if forced:
        return forced

    current_round = state.current_round
    rb_count = state.team_position_count(team_idx, "RB")

    risk = kwargs.get("risk_profile", "balanced")

    if current_round <= 3 and rb_count < 2:
        # Prioritize RBs early (take top RB if decent value)
        rbs = [p for p in eligible if p.position == "RB"]
        if rbs:
            replacement = _dynamic_repl(state, players)
            roster = state.teams[team_idx] if team_idx < len(state.teams) else []
            score_fn = lambda p: (
                vbd(p, replacement) * marginal_value_discount(p, roster, state.config)
                + _var_bonus(p, state, team_idx, risk)
            )
            top_rb = max(rbs, key=score_fn)
            top_overall = max(eligible, key=score_fn)

            # Take the RB if within 80% of top overall value
            rb_val = score_fn(top_rb)
            top_val = score_fn(top_overall)
            if rb_val >= top_val * 0.8 or top_overall.position == "RB":
                return top_rb

    # Fall back to VBD
    return pick_vbd(state, team_idx, players=players, risk_profile=risk)


STRATEGIES = {
    "bpa": pick_bpa,
    "vbd": pick_vbd,
    "vona": pick_vona,
    "zero-rb": pick_zero_rb,
    "robust-rb": pick_robust_rb,
}


def get_strategy(name: str):
    """Get strategy function by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Choose from: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]
