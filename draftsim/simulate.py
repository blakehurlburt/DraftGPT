"""Monte Carlo draft simulation runner."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .adp import generate_consensus_adp, generate_platform_adp, PLATFORMS
from .config import LeagueConfig
from .draft import DraftState
from .opponents import ADPOpponent
from .players import Player
from .results import compute_optimal_lineup
from .strategies import get_strategy


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo draft simulation."""

    league: LeagueConfig = field(default_factory=LeagueConfig)
    num_simulations: int = 5000
    user_slots: list[int] | None = None  # None = all slots (0-indexed)
    strategies: list[str] = field(default_factory=lambda: ["vbd"])
    opponent_platform: str = "consensus"  # sleeper, espn, yahoo, or consensus
    seed: int = 42


def _generate_adp_for_sim(
    players: list[Player],
    platform: str,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Generate fresh noisy ADP for a single simulation run."""
    if platform == "consensus":
        entries = generate_consensus_adp(players, rng)
    elif platform in PLATFORMS:
        entries = generate_platform_adp(players, platform, rng)
    else:
        raise ValueError(f"Unknown platform: {platform}")
    return {p.name: adp for p, adp in entries}


def _run_single_sim(
    state: DraftState,
    user_slot: int,
    first_pick: Player | None,
    strategy_fn,
    opponent: ADPOpponent,
    adp_order: list[str],
    players: list[Player],
    config: LeagueConfig,
) -> float:
    """Run one simulation from current state to completion.

    If *first_pick* is not None, it is forced as the user's immediate pick
    before the rest of the draft continues normally.

    Returns the user's optimal lineup total.
    """
    sim_state = state.copy()

    # Force the first pick if provided
    if first_pick is not None and not sim_state.is_complete:
        if sim_state.current_team_idx == user_slot and first_pick in sim_state.available:
            sim_state.make_pick(first_pick)

    # Run the remaining draft
    while not sim_state.is_complete:
        team_idx = sim_state.current_team_idx
        if team_idx == user_slot:
            player = strategy_fn(
                sim_state, team_idx, players=players, adp_order=adp_order,
            )
        else:
            player = opponent.pick(sim_state, team_idx)
        sim_state.make_pick(player)

    roster = sim_state.team_roster(user_slot)
    _, lineup_total = compute_optimal_lineup(roster, config)
    return lineup_total


def run_simulation(
    players: list[Player],
    sim_config: SimulationConfig,
) -> tuple[dict, dict]:
    """Run Monte Carlo draft simulations.

    For each (slot, strategy) combination, runs N simulated drafts.
    Each sim uses fresh ADP noise for opponents.

    Args:
        players: Full player pool
        sim_config: Simulation parameters

    Returns:
        (results, sample_rosters) where:
        - results: dict mapping (slot, strategy) -> list of lineup totals
        - sample_rosters: dict mapping (slot, strategy, sim_idx) -> roster list
          (only stores first 3 sims per combo for reporting)
    """
    config = sim_config.league
    slots = sim_config.user_slots
    if slots is None:
        slots = list(range(config.num_teams))

    results: dict[tuple[int, str], list[float]] = {}
    sample_rosters: dict[tuple[int, str, int], list[Player]] = {}

    master_rng = np.random.default_rng(sim_config.seed)
    total_combos = len(slots) * len(sim_config.strategies)
    combo_count = 0

    for strategy_name in sim_config.strategies:
        strategy_fn = get_strategy(strategy_name)

        for slot in slots:
            combo_count += 1
            key = (slot, strategy_name)
            results[key] = []

            print(
                f"  [{combo_count}/{total_combos}] "
                f"Slot {slot + 1}, {strategy_name}: ",
                end="",
                flush=True,
            )

            for sim_idx in range(sim_config.num_simulations):
                sim_rng = np.random.default_rng(
                    master_rng.integers(0, 2**31)
                )

                # Fresh ADP noise each simulation
                adp = _generate_adp_for_sim(
                    players, sim_config.opponent_platform, sim_rng
                )
                adp_order = sorted(adp.keys(), key=lambda n: adp[n])

                # Initialize draft
                state = DraftState.create(config, players)
                opponent = ADPOpponent(adp, sim_rng)

                lineup_total = _run_single_sim(
                    state, slot, None, strategy_fn,
                    opponent, adp_order, players, config,
                )
                results[key].append(lineup_total)

                # Store sample rosters for reporting
                if sim_idx < 3:
                    # Re-run to capture roster (cheap for first 3)
                    state2 = DraftState.create(config, players)
                    opponent2 = ADPOpponent(adp, np.random.default_rng(sim_rng.integers(0, 2**31)))
                    while not state2.is_complete:
                        tidx = state2.current_team_idx
                        if tidx == slot:
                            p = strategy_fn(state2, tidx, players=players, adp_order=adp_order)
                        else:
                            p = opponent2.pick(state2, tidx)
                        state2.make_pick(p)
                    sample_rosters[(slot, strategy_name, sim_idx)] = list(state2.team_roster(slot))

            mean = np.mean(results[key])
            std = np.std(results[key])
            print(f"mean={mean:.1f}, std={std:.1f}")

    return results, sample_rosters
