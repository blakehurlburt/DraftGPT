"""Top-N pick recommendations using draftsim strategies."""

from __future__ import annotations

from dataclasses import dataclass

from draftsim.draft import DraftState
from draftsim.players import Player
from draftsim.strategies import STRATEGIES
from draftsim.value import compute_replacement_levels, vbd


@dataclass
class Recommendation:
    player: Player
    rank: int
    vbd_value: float


def top_n_picks(
    strategy_fn,
    state: DraftState,
    team_idx: int,
    n: int = 5,
    **kwargs,
) -> list[Recommendation]:
    """Get top N picks from a strategy by iteratively picking and restoring.

    Calls strategy -> temporarily removes pick from available -> repeats.
    Restores available list when done.
    """
    if state.is_complete:
        return []

    replacement = compute_replacement_levels(state.available, state.config)

    # Save original available list
    original_available = list(state.available)
    picks: list[Recommendation] = []

    for i in range(n):
        if not state.available:
            break

        try:
            player = strategy_fn(state, team_idx, **kwargs)
        except (ValueError, IndexError):
            break

        picks.append(Recommendation(
            player=player,
            rank=i + 1,
            vbd_value=vbd(player, replacement),
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
        kwargs = {"players": players, "adp_order": adp_order}
        results[name] = top_n_picks(fn, state, team_idx, n=n, **kwargs)

    return results
