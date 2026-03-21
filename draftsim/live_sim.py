"""Live Monte Carlo simulation engine for in-draft use."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import numpy as np

from .adp import generate_consensus_adp, generate_platform_adp, PLATFORMS
from .config import LeagueConfig
from .draft import DraftState
from .opponents import ADPOpponent
from .players import Player
from .simulate import _run_single_sim
from .strategies import STRATEGIES, get_strategy, _eligible


@dataclass
class StrategyResult:
    """Aggregated sim results for one strategy."""

    mean_total: float = 0.0
    std_total: float = 0.0
    p10_total: float = 0.0
    p90_total: float = 0.0
    pick_values: dict[str, float] = field(default_factory=dict)


@dataclass
class SimSnapshot:
    """Progressive snapshot of simulation progress."""

    sims_completed: int = 0
    sims_target: int = 0
    strategy_results: dict[str, StrategyResult] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for JSON/SSE transport."""
        strategies = {}
        for name, sr in self.strategy_results.items():
            strategies[name] = {
                "mean": round(sr.mean_total, 1),
                "std": round(sr.std_total, 1),
                "p10": round(sr.p10_total, 1),
                "p90": round(sr.p90_total, 1),
                "pick_values": {k: round(v, 1) for k, v in sr.pick_values.items()},
            }
        return {
            "sims_completed": self.sims_completed,
            "sims_target": self.sims_target,
            "strategies": strategies,
        }


def _get_candidates(
    state: DraftState,
    user_slot: int,
    players: list[Player],
    adp_order: list[str] | None,
    top_n: int = 15,
) -> list[Player]:
    """Identify top candidate players to simulate as first pick."""
    eligible = _eligible(state, user_slot)
    if not eligible:
        return []

    # Use projected_total as a quick heuristic to pick top candidates
    eligible_sorted = sorted(eligible, key=lambda p: p.projected_total, reverse=True)
    return eligible_sorted[:top_n]


async def run_live_simulation(
    state: DraftState,
    user_slot: int,
    players: list[Player],
    adp_order: list[str] | None,
    strategies: list[str],
    cancel_event: asyncio.Event,
    on_snapshot: Callable[[SimSnapshot], Awaitable[None]],
    batch_size: int = 50,
    max_sims: int = 500,
    top_n_candidates: int = 15,
    opponent_platform: str = "consensus",
    seed: int | None = None,
) -> SimSnapshot:
    """Run progressive Monte Carlo sims, yielding snapshots between batches.

    Args:
        state: Current draft state (will be copied, not mutated).
        user_slot: 0-indexed user team slot.
        players: Full player pool.
        adp_order: ADP-ordered player names for VONA.
        strategies: List of strategy names to simulate.
        cancel_event: Set this to cancel the simulation.
        on_snapshot: Async callback invoked after each batch with running stats.
        batch_size: Sims per batch before yielding.
        max_sims: Total sims per (strategy, candidate) combination.
        top_n_candidates: Number of top candidates to simulate.
        opponent_platform: ADP platform for opponent modeling.
        seed: Optional RNG seed for reproducibility.

    Returns:
        Final SimSnapshot.
    """
    candidates = _get_candidates(state, user_slot, players, adp_order, top_n_candidates)
    if not candidates:
        return SimSnapshot()

    rng = np.random.default_rng(seed)

    # Accumulate results: strategy -> candidate_name -> list of lineup totals
    raw_results: dict[str, dict[str, list[float]]] = {
        s: {c.name: [] for c in candidates} for s in strategies
    }
    # Also track baseline (no forced pick) per strategy
    baseline: dict[str, list[float]] = {s: [] for s in strategies}

    # CR opus: sims_target is set to max_sims, but the actual total work is
    # max_sims * len(strategies) * (len(candidates) + 1). The progress reporting
    # (sims_completed / sims_target) will appear to complete at 100% but is actually
    # measuring only the outer sim loop, not the inner candidate * strategy loops.
    total_sims = max_sims
    snapshot = SimSnapshot(sims_completed=0, sims_target=total_sims)

    for batch_start in range(0, total_sims, batch_size):
        if cancel_event.is_set():
            break

        batch_end = min(batch_start + batch_size, total_sims)

        # CR opus: _run_batch is a closure that captures batch_start and batch_end by
        # reference. Since it's passed to asyncio.to_thread and awaited immediately,
        # this is fine. But if the code were ever changed to not await (e.g. fire-and-forget),
        # all batches would use the last loop's values. Also, raw_results and baseline
        # dicts are mutated from the thread without locks — safe only because the await
        # ensures sequential execution.
        def _run_batch():
            for sim_idx in range(batch_start, batch_end):
                if cancel_event.is_set():
                    return

                # CR opus: rng is captured from the outer scope (closure). Since _run_batch
                # runs in a thread via asyncio.to_thread, and rng.integers() mutates rng's
                # internal state, this is a race condition if run_live_simulation is called
                # concurrently. Each call should get its own independent RNG.
                sim_rng = np.random.default_rng(rng.integers(0, 2**63))

                # CR opus: sorted_players is recomputed on every sim iteration but
                # CR opus: players list never changes — this is an O(n log n) sort
                # CR opus: repeated max_sims times. Should sort once outside the loop.
                # Fresh ADP noise
                sorted_players = sorted(players, key=lambda p: p.projected_total, reverse=True)
                if opponent_platform == "consensus":
                    entries = generate_consensus_adp(sorted_players, sim_rng)
                elif opponent_platform in PLATFORMS:
                    entries = generate_platform_adp(sorted_players, opponent_platform, sim_rng)
                else:
                    entries = generate_consensus_adp(sorted_players, sim_rng)
                adp = {p.name: a for p, a in entries}
                sim_adp_order = sorted(adp.keys(), key=lambda n: adp[n])

                opponent = ADPOpponent(adp, sim_rng)

                for strat_name in strategies:
                    strategy_fn = get_strategy(strat_name)

                    # Sim each candidate as forced first pick
                    for cand in candidates:
                        # CR opus: Each candidate sim creates a new opponent RNG from
                        # sim_rng, but sim_rng is shared across all candidates in this
                        # loop iteration. The RNG state changes after each candidate,
                        # meaning later candidates get different opponent randomness than
                        # earlier ones within the same sim_idx. This biases comparisons.
                        opp2 = ADPOpponent(adp, np.random.default_rng(sim_rng.integers(0, 2**63)))
                        total = _run_single_sim(
                            state, user_slot, cand, strategy_fn,
                            opp2, sim_adp_order, players, state.config,
                        )
                        raw_results[strat_name][cand.name].append(total)

                    # Baseline (strategy picks naturally)
                    opp3 = ADPOpponent(adp, np.random.default_rng(sim_rng.integers(0, 2**63)))
                    base_total = _run_single_sim(
                        state, user_slot, None, strategy_fn,
                        opp3, sim_adp_order, players, state.config,
                    )
                    baseline[strat_name].append(base_total)

        await asyncio.to_thread(_run_batch)

        if cancel_event.is_set():
            break

        # Build snapshot from accumulated results
        sims_done = batch_end
        snapshot.sims_completed = sims_done
        snapshot.strategy_results = {}

        for strat_name in strategies:
            base_arr = np.array(baseline[strat_name]) if baseline[strat_name] else np.array([0.0])
            sr = StrategyResult(
                mean_total=float(np.mean(base_arr)),
                std_total=float(np.std(base_arr)),
                p10_total=float(np.percentile(base_arr, 10)) if len(base_arr) > 1 else float(base_arr[0]),
                p90_total=float(np.percentile(base_arr, 90)) if len(base_arr) > 1 else float(base_arr[0]),
            )
            for cand in candidates:
                vals = raw_results[strat_name][cand.name]
                if vals:
                    sr.pick_values[cand.name] = float(np.mean(vals))
            snapshot.strategy_results[strat_name] = sr

        await on_snapshot(snapshot)

    return snapshot
