"""Top-N pick recommendations using draftsim strategies."""

from __future__ import annotations

from dataclasses import dataclass

from draftsim.draft import DraftState
from draftsim.players import Player
from draftsim.strategies import _var_bonus, pick_vona
from draftsim.value import (
    compute_dynamic_replacement_levels,
    compute_last_starter_levels,
    compute_replacement_levels,
    marginal_value_discount,
    vbd,
    vbd_score,
    vols,
    vona,
    vona_weight,
)


@dataclass
class Recommendation:
    player: Player
    rank: int
    vbd_value: float  # VORP: value over replacement player
    strategy_score: float  # the score the strategy actually used to rank this pick
    vona_value: float = 0.0  # VONA: positional urgency
    vols_value: float = 0.0  # VOLS: value over last starter
    vbd_score_value: float = 0.0  # VBD Score: composite VORP + VONA + VOLS


def _compute_strategy_score(
    strategy_name: str,
    player: Player,
    replacement: dict[str, float],
    state: DraftState,
    team_idx: int,
    adp_order: list[str] | None,
    risk_profile: str = "balanced",
    pos_vona_cache: dict[str, float] | None = None,
) -> float:
    """Compute the score the strategy uses to rank this player."""
    var = _var_bonus(player, state, team_idx, risk_profile)
    roster = state.teams[team_idx] if team_idx < len(state.teams) else []
    discount = marginal_value_discount(player, roster, state.config)
    if strategy_name == "bpa":
        return player.projected_total + var
    elif strategy_name == "vona":
        player_vbd = vbd(player, replacement)
        weight = vona_weight(state.current_round, state.config.num_rounds)
        if pos_vona_cache is not None:
            pos_v = pos_vona_cache.get(player.position, 0.0)
        elif adp_order:
            pos_v = vona(state, team_idx, player.position, adp_order,
                         current_round=state.current_round,
                         total_rounds=state.config.num_rounds)
        else:
            pos_v = 0.0
        return player_vbd * discount + pos_v * weight + var
    else:
        # vbd uses VBD as its core metric
        return vbd(player, replacement) * discount + var


def top_n_picks(
    strategy_fn,
    state: DraftState,
    team_idx: int,
    n: int = 5,
    *,
    strategy_name: str = "vbd",
    adp_order: list[str] | None = None,
    risk_profile: str = "balanced",
    **kwargs,
) -> list[Recommendation]:
    """Get top N picks from a strategy by iteratively picking and restoring.

    Calls strategy -> temporarily removes pick from available -> repeats.
    Restores available list when done.
    """
    if state.is_complete:
        return []

    replacement = compute_dynamic_replacement_levels(
        state.available, state.config, state.teams,
    )
    last_starter = compute_last_starter_levels(
        state.available, state.config, state.teams,
    )

    # Pre-compute VONA cache (always, so we can populate vona_value on every rec)
    pos_vona_cache: dict[str, float] = {}
    if adp_order:
        positions = {p.position for p in state.available}
        pos_vona_cache = {
            pos: vona(state, team_idx, pos, adp_order,
                      current_round=state.current_round,
                      total_rounds=state.config.num_rounds)
            for pos in positions
        }

    # Include adp_order and risk_profile in kwargs for strategies
    strategy_kwargs = dict(kwargs)
    if adp_order is not None:
        strategy_kwargs["adp_order"] = adp_order
    strategy_kwargs["risk_profile"] = risk_profile

    # CR opus: This mutates state.available in-place during the loop and restores it
    # afterward. If an unhandled exception occurs between removal and restore (e.g., in
    # _compute_strategy_score or vbd_score), state.available will be left in a corrupted
    # state with some players missing. Consider using a try/finally block for the restore.
    # Save original available list
    original_available = list(state.available)
    picks: list[Recommendation] = []

    for i in range(n):
        if not state.available:
            break

        try:
            player = strategy_fn(state, team_idx, **strategy_kwargs)
        except (ValueError, IndexError):
            break

        score = _compute_strategy_score(
            strategy_name, player, replacement, state, team_idx, adp_order,
            risk_profile, pos_vona_cache,
        )

        vorp_val = vbd(player, replacement)
        vona_val = pos_vona_cache.get(player.position, 0.0)
        vols_val = vols(player, last_starter)

        picks.append(Recommendation(
            player=player,
            rank=i + 1,
            vbd_value=vorp_val,
            strategy_score=score,
            vona_value=vona_val,
            vols_value=vols_val,
            vbd_score_value=vbd_score(vorp_val, vona_val, vols_val),
        ))

        # Temporarily remove from available for next iteration
        if player in state.available:
            state.available.remove(player)

    # Restore original available list
    state.available = original_available

    return picks


def get_recommendations(
    state: DraftState,
    team_idx: int,
    players: list[Player],
    adp_order: list[str] | None = None,
    n: int = 5,
    risk_profile: str = "balanced",
) -> list[Recommendation]:
    """Get top-N recommendations using the VONA strategy.

    Returns a flat list of Recommendations with all value metrics populated
    (VORP, VONA, VOLS, VBD Score).
    """
    if adp_order is None:
        adp_order = [
            p.name for p in sorted(
                state.available, key=lambda p: p.projected_total, reverse=True
            )
        ]

    return top_n_picks(
        pick_vona, state, team_idx, n=n,
        strategy_name="vona", adp_order=adp_order, players=players,
        risk_profile=risk_profile,
    )
