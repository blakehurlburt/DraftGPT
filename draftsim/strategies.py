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
    pos_vona_cache = {
        pos: vona(state, team_idx, pos, adp_order,
                  current_round=state.current_round,
                  total_rounds=state.config.num_rounds)
        for pos in positions
    }

    def _score(p):
        discount = marginal_value_discount(p, roster, state.config)
        return (vbd(p, replacement) * discount
                + pos_vona_cache.get(p.position, 0.0) * weight
                + _var_bonus(p, state, team_idx, risk))

    return max(eligible, key=_score)


STRATEGIES = {
    "bpa": pick_bpa,
    "vbd": pick_vbd,
    "vona": pick_vona,
}


def get_strategy(name: str):
    """Get strategy function by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Choose from: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]
