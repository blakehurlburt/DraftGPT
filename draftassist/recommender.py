"""Top-N pick recommendations using draftsim strategies."""

from __future__ import annotations

from dataclasses import dataclass

from draftsim.draft import DraftState
from draftsim.players import Player
from draftsim.strategies import STRATEGIES
from draftsim.value import compute_replacement_levels, vbd, vona


@dataclass
class Recommendation:
    player: Player
    rank: int
    vbd_value: float
    strategy_score: float  # the score the strategy actually used to rank this pick


def _compute_strategy_score(
    strategy_name: str,
    player: Player,
    replacement: dict[str, float],
    state: DraftState,
    team_idx: int,
    adp_order: list[str] | None,
) -> float:
    """Compute the score the strategy uses to rank this player."""
    if strategy_name == "bpa":
        return player.projected_total
    elif strategy_name == "vona":
        player_vbd = vbd(player, replacement)
        if adp_order:
            pos_vona = vona(state, team_idx, player.position, adp_order)
        else:
            pos_vona = 0.0
        return player_vbd + pos_vona * 0.5
    else:
        # vbd, zero-rb, robust-rb all use VBD as their core metric
        return vbd(player, replacement)


def top_n_picks(
    strategy_fn,
    state: DraftState,
    team_idx: int,
    n: int = 5,
    *,
    strategy_name: str = "vbd",
    adp_order: list[str] | None = None,
    **kwargs,
) -> list[Recommendation]:
    """Get top N picks from a strategy by iteratively picking and restoring.

    Calls strategy -> temporarily removes pick from available -> repeats.
    Restores available list when done.
    """
    if state.is_complete:
        return []

    all_players = kwargs.get("players") or state.available
    replacement = compute_replacement_levels(all_players, state.config)

    # Include adp_order in kwargs for strategies that use it (e.g. VONA)
    strategy_kwargs = dict(kwargs)
    if adp_order is not None:
        strategy_kwargs["adp_order"] = adp_order

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
            strategy_name, player, replacement, state, team_idx, adp_order
        )

        picks.append(Recommendation(
            player=player,
            rank=i + 1,
            vbd_value=vbd(player, replacement),
            strategy_score=score,
        ))

        # Temporarily remove from available for next iteration
        if player in state.available:
            state.available.remove(player)

    # Restore original available list
    state.available = original_available

    return picks


def get_all_recommendations(
    state: DraftState,
    team_idx: int,
    players: list[Player],
    adp_order: list[str] | None = None,
    n: int = 5,
) -> dict[str, list[Recommendation]]:
    """Run top_n_picks for all 5 strategies.

    Returns dict mapping strategy_name -> list of Recommendations.
    """
    if adp_order is None:
        adp_order = [
            p.name for p in sorted(
                state.available, key=lambda p: p.projected_total, reverse=True
            )
        ]

    results: dict[str, list[Recommendation]] = {}

    for name, fn in STRATEGIES.items():
        results[name] = top_n_picks(
            fn, state, team_idx, n=n,
            strategy_name=name, adp_order=adp_order, players=players,
        )

    return results
