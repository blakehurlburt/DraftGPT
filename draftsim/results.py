"""Scoring optimal lineups and reporting draft simulation results."""

from itertools import combinations

from .config import LeagueConfig
from .players import Player


def compute_optimal_lineup(roster: list[Player], config: LeagueConfig) -> tuple[list[Player], float]:
    """Find the highest-scoring legal lineup from a roster.

    Fills starters (QB, RB, WR, TE) then FLEX (best remaining RB/WR/TE).

    Returns:
        (lineup, total_points) — the optimal starters and their total projected points
    """
    starters = config.starter_slots()
    flex_count = config.num_flex()

    # Group roster by position (derive from config + roster, not hardcoded)
    all_positions = set(starters.keys()) | set(config.flex_positions())
    for p in roster:
        all_positions.add(p.position)
    by_pos: dict[str, list[Player]] = {pos: [] for pos in all_positions}
    for p in roster:
        if p.position in by_pos:
            by_pos[p.position].append(p)

    # Sort each position by projected_total descending
    for pos in by_pos:
        by_pos[pos].sort(key=lambda p: p.projected_total, reverse=True)

    lineup = []
    used = set()

    # Fill mandatory starter slots
    for pos, count in starters.items():
        for i in range(min(count, len(by_pos.get(pos, [])))):
            player = by_pos[pos][i]
            lineup.append(player)
            used.add(player.name)

    # Fill FLEX with best remaining RB/WR/TE
    # CR opus: The `used` set tracks by player name, but two different players could
    # share the same name (unlikely but possible). Using player identity (id(p) or the
    # Player object itself) would be safer.
    flex_candidates = []
    for pos in config.flex_positions():
        start_idx = starters.get(pos, 0)
        for p in by_pos.get(pos, [])[start_idx:]:
            if p.name not in used:
                flex_candidates.append(p)

    flex_candidates.sort(key=lambda p: p.projected_total, reverse=True)
    for i in range(min(flex_count, len(flex_candidates))):
        lineup.append(flex_candidates[i])

    total = sum(p.projected_total for p in lineup)
    return lineup, total


def format_strategy_comparison(results: dict) -> str:
    """Format strategy comparison results into a readable table.

    Args:
        results: Dict from simulate.run_simulation() with structure:
                 {(slot, strategy): [lineup_totals_per_sim]}
    """
    import numpy as np

    lines = []
    lines.append("\n" + "=" * 75)
    lines.append("DRAFT SIMULATION RESULTS — STRATEGY COMPARISON")
    lines.append("=" * 75)

    # Group by strategy
    strategies = sorted(set(s for _, s in results.keys()))
    slots = sorted(set(slot for slot, _ in results.keys()))

    # Overall strategy summary
    lines.append(f"\n{'Strategy':<12} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    lines.append("-" * 60)

    strategy_means = {}
    for strat in strategies:
        all_scores = []
        for slot in slots:
            key = (slot, strat)
            if key in results:
                all_scores.extend(results[key])
        if all_scores:
            arr = np.array(all_scores)
            mean = np.mean(arr)
            strategy_means[strat] = mean
            lines.append(
                f"{strat:<12} {mean:>8.1f} {np.median(arr):>8.1f} "
                f"{np.std(arr):>8.1f} {np.min(arr):>8.1f} {np.max(arr):>8.1f}"
            )

    # Best slot per strategy
    lines.append(f"\n{'Strategy':<12} {'Best Slot':>10} {'Slot Mean':>10} {'Worst Slot':>11} {'Slot Mean':>10}")
    lines.append("-" * 60)

    for strat in strategies:
        slot_means = {}
        for slot in slots:
            key = (slot, strat)
            if key in results and results[key]:
                slot_means[slot] = np.mean(results[key])
        if slot_means:
            best_slot = max(slot_means, key=slot_means.get)
            worst_slot = min(slot_means, key=slot_means.get)
            lines.append(
                f"{strat:<12} {best_slot + 1:>10} {slot_means[best_slot]:>10.1f} "
                f"{worst_slot + 1:>11} {slot_means[worst_slot]:>10.1f}"
            )

    # Per-slot breakdown for best strategy
    if strategy_means:
        best_strat = max(strategy_means, key=strategy_means.get)
        lines.append(f"\nPer-slot breakdown for best strategy ({best_strat}):")
        lines.append(f"{'Slot':>6} {'Mean':>8} {'Median':>8} {'Std':>8}")
        lines.append("-" * 35)
        for slot in slots:
            key = (slot, best_strat)
            if key in results and results[key]:
                arr = np.array(results[key])
                lines.append(
                    f"{slot + 1:>6} {np.mean(arr):>8.1f} {np.median(arr):>8.1f} {np.std(arr):>8.1f}"
                )

    return "\n".join(lines)


def format_draft_recap(roster: list[Player], config: LeagueConfig) -> str:
    """Format a round-by-round draft recap."""
    lines = []
    lines.append("\nDRAFT RECAP")
    lines.append("-" * 55)
    lines.append(f"{'Rnd':>3} {'Pick':>5}  {'Player':<28} {'Pos':<4} {'Proj':>6}")
    lines.append("-" * 55)

    for i, player in enumerate(roster):
        rnd = i + 1
        lines.append(
            f"{rnd:>3} {'-':>5}  {player.name:<28} {player.position:<4} {player.projected_total:>6.0f}"
        )

    lineup, total = compute_optimal_lineup(roster, config)
    lines.append(f"\nOptimal lineup total: {total:.0f} pts")
    lines.append("Starters:")
    starters = config.starter_slots()
    pos_counts: dict = {}
    _POS_ORDER = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "K": 4, "DST": 5}
    for p in sorted(lineup, key=lambda p: _POS_ORDER.get(p.position, 6)):
        pos_counts[p.position] = pos_counts.get(p.position, 0) + 1
        starter_slots = starters.get(p.position, 0)
        label = "FLEX" if pos_counts[p.position] > starter_slots else p.position
        lines.append(f"  {label:<4} {p.name:<28} {p.projected_total:>6.0f}")

    return "\n".join(lines)


def format_position_composition(results: dict, rosters: dict) -> str:
    """Format average roster composition by strategy.

    Args:
        results: simulation results dict
        rosters: dict mapping (slot, strategy, sim_idx) to list[Player]
    """
    import numpy as np

    lines = []
    lines.append("\nAVERAGE ROSTER COMPOSITION BY STRATEGY")
    lines.append("-" * 50)
    all_pos = ["QB", "RB", "WR", "TE", "K", "DST"]
    lines.append(f"{'Strategy':<12} {'QB':>5} {'RB':>5} {'WR':>5} {'TE':>5} {'K':>5} {'DST':>5}")
    lines.append("-" * 62)

    strategies = sorted(set(s for _, s in results.keys()))

    for strat in strategies:
        pos_counts = {pos: [] for pos in all_pos}
        # CR opus: Nested loop bug — the outer loop iterates over all (slot, strategy)
        # result keys matching this strat, and for EACH one, the inner loop iterates
        # over ALL rosters matching this strat. This means each roster is counted once
        # per matching slot, inflating the sample size by a factor of len(slots).
        # The outer loop over results.items() is unnecessary; just iterate rosters directly.
        for (slot, s), _ in results.items():
            if s != strat:
                continue
            for key, roster in rosters.items():
                if key[1] == strat:
                    counts = {pos: 0 for pos in all_pos}
                    for p in roster:
                        if p.position in counts:
                            counts[p.position] += 1
                    for pos in pos_counts:
                        pos_counts[pos].append(counts[pos])

        if pos_counts["QB"]:
            lines.append(
                f"{strat:<12} {np.mean(pos_counts['QB']):>5.1f} "
                f"{np.mean(pos_counts['RB']):>5.1f} "
                f"{np.mean(pos_counts['WR']):>5.1f} "
                f"{np.mean(pos_counts['TE']):>5.1f} "
                f"{np.mean(pos_counts['K']):>5.1f} "
                f"{np.mean(pos_counts['DST']):>5.1f}"
            )

    return "\n".join(lines)
