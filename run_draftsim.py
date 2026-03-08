#!/usr/bin/env python3
"""CLI entry point for draft simulation.

Usage:
    python run_draftsim.py                              # 12-team, all slots, VBD/BPA/VONA
    python run_draftsim.py --teams 10 --slot 3          # 10-team, 3rd pick
    python run_draftsim.py --strategy zero-rb --sims 10000
    python run_draftsim.py --compare                    # all 5 strategies head-to-head
    python run_draftsim.py --platform sleeper           # use Sleeper mock ADP for opponents
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from draftsim.adp import PLATFORMS
from draftsim.config import LeagueConfig
from draftsim.players import load_players
from draftsim.results import (
    compute_optimal_lineup,
    format_draft_recap,
    format_position_composition,
    format_strategy_comparison,
)
from draftsim.simulate import SimulationConfig, run_simulation


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo snake draft simulator"
    )
    parser.add_argument(
        "--teams", type=int, default=12, choices=[8, 10, 12],
        help="Number of teams in league (default: 12)"
    )
    parser.add_argument(
        "--slot", type=int, default=None,
        help="Draft slot to simulate (1-indexed). Default: all slots"
    )
    parser.add_argument(
        "--strategy", type=str, default=None,
        help="Strategy to use: bpa, vbd, vona, zero-rb, robust-rb"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare all 5 strategies head-to-head"
    )
    parser.add_argument(
        "--sims", type=int, default=5000,
        help="Number of simulations per slot/strategy combo (default: 5000)"
    )
    parser.add_argument(
        "--platform", type=str, default="consensus",
        choices=["sleeper", "espn", "yahoo", "consensus"],
        help="ADP platform for opponent modeling (default: consensus)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--projections", type=str, default="data/projections/all_projections.csv",
        help="Path to projections CSV"
    )
    parser.add_argument(
        "--recap", action="store_true",
        help="Show a sample draft recap"
    )
    args = parser.parse_args()

    # Load players
    print(f"Loading projections from {args.projections}...")
    players = load_players(args.projections)
    print(f"  Loaded {len(players)} players")

    # ADP data loaded from FantasyPros CSV (no generation needed)
    print("ADP data: FantasyPros consensus + per-platform rankings")

    # Configure league
    config = LeagueConfig(num_teams=args.teams)

    # Determine strategies
    if args.compare:
        strategies = ["bpa", "vbd", "vona", "zero-rb", "robust-rb"]
    elif args.strategy:
        strategies = [args.strategy]
    else:
        strategies = ["bpa", "vbd", "vona"]

    # Determine slots
    user_slots = None
    if args.slot is not None:
        if args.slot < 1 or args.slot > args.teams:
            print(f"Error: --slot must be between 1 and {args.teams}")
            sys.exit(1)
        user_slots = [args.slot - 1]  # convert to 0-indexed

    # Build simulation config
    sim_config = SimulationConfig(
        league=config,
        num_simulations=args.sims,
        user_slots=user_slots,
        strategies=strategies,
        opponent_platform=args.platform,
        seed=args.seed,
    )

    # Run simulation
    num_combos = len(strategies) * (len(user_slots) if user_slots else args.teams)
    total_sims = num_combos * args.sims
    print(
        f"\nRunning {total_sims:,} total simulations "
        f"({args.sims} sims x {num_combos} slot/strategy combos)..."
    )
    start = time.time()
    results, sample_rosters = run_simulation(players, sim_config)
    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f}s ({total_sims / elapsed:.0f} sims/sec)")

    # Report results
    print(format_strategy_comparison(results))

    # Position composition
    if sample_rosters:
        print(format_position_composition(results, sample_rosters))

    # Draft recap
    if args.recap and sample_rosters:
        # Show recap for first slot, first strategy, first sim
        first_key = next(
            (k for k in sample_rosters if k[2] == 0),
            None
        )
        if first_key:
            slot, strat, _ = first_key
            roster = sample_rosters[first_key]
            print(f"\nSample draft recap (Slot {slot + 1}, {strat}):")
            print(format_draft_recap(roster, config))


if __name__ == "__main__":
    main()
