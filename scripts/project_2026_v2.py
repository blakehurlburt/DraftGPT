"""
Project 2026 fantasy rankings using the season-level model.

Trains on all available season-level data, uses 2025 stats as the
"prior season" features, and predicts 2026 PPG and games for each player.
Combines with rosters.csv for team assignments.
"""

import polars as pl
import numpy as np
from pathlib import Path
from nfldata.season_features import build_season_features, get_season_feature_columns
from nfldata.season_model import train_final_model, project_season
from nfldata.kicker_features import build_kicker_features, build_kicker_projection_features
from nfldata.kicker_model import (
    train_final_model as train_kicker_model,
    project_season as project_kicker_season,
    walk_forward_eval as kicker_walk_forward_eval,
)
from nfldata.dst_features import build_dst_features, build_dst_projection_features
from nfldata.dst_model import (
    train_final_model as train_dst_model,
    project_season as project_dst_season,
    walk_forward_eval as dst_walk_forward_eval,
)

ROSTER_PATH = Path(__file__).parent.parent / "data" / "rosters.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "projections"


def get_current_rosters():
    """Load current rosters from rosters.csv (requires gsis_id column)."""
    if not ROSTER_PATH.exists():
        raise FileNotFoundError(
            "rosters.csv not found — run 'python scripts/update_rosters.py' first"
        )

    roster_df = pl.read_csv(ROSTER_PATH, comment_prefix="#")
    print(f"  Loaded {roster_df.shape[0]} players from rosters.csv")

    return roster_df.select([
        pl.col("gsis_id").alias("player_id"),
        pl.col("team").alias("current_team"),
        pl.col("position").alias("current_position"),
        pl.col("status").alias("current_status"),
    ])


def main():
    # Step 1: Get current rosters (needed for projection features)
    print("Loading current rosters...")
    rosters = get_current_rosters()

    # Load adjustment_ppg from rosters.csv if available
    if ROSTER_PATH.exists():
        roster_raw = pl.read_csv(ROSTER_PATH, comment_prefix="#")
        if "adjustment_ppg" in roster_raw.columns:
            adj_df = roster_raw.select([
                pl.col("gsis_id").alias("player_id"),
                "adjustment_ppg",
            ]).filter(pl.col("player_id").is_not_null())
            rosters = rosters.join(adj_df, on="player_id", how="left")

    # Step 2: Build unified feature matrix (training + projection in one pass)
    print("\nBuilding season features (2009-2025 + 2026 projection)...")
    full_df = build_season_features(range(2009, 2026), projection_year=2026, rosters=rosters)

    # Split into training and projection sets
    df = full_df.filter(pl.col("season") < 2026)
    proj_df = full_df.filter(pl.col("season") == 2026)

    # Step 3: Run walk-forward eval on training data
    print("\n--- Walk-Forward Evaluation (for reference) ---")
    from nfldata.season_model import walk_forward_eval
    walk_forward_eval(df)

    # Step 4: Train final model on all available training data
    print("\n--- Training Final Model ---")
    ppg_model, games_model, importance, quantile_models = train_final_model(df)

    # Step 6: Project (with floor/median/ceiling)
    results = project_season(ppg_model, games_model, proj_df, quantile_models=quantile_models)

    # Join with rosters
    results = results.join(rosters, on="player_id", how="left")

    # Fill missing current_team from the feature data's team column
    if "team" in results.columns and "current_team" in results.columns:
        results = results.with_columns(
            pl.col("current_team").fill_null(pl.col("team"))
        )

    # Filter to active players
    active_statuses = ["ACT", "RES", "PUP", "NFI"]
    results = results.filter(
        pl.col("current_status").is_in(active_statuses)
        | pl.col("current_status").is_null()
    )

    # --- Kicker Projections ---
    print("\n--- Kicker Model ---")
    k_df = build_kicker_features(range(2009, 2026))
    kicker_walk_forward_eval(k_df)
    k_ppg, k_games, k_imp, k_qmodels = train_kicker_model(k_df)
    k_proj_df = build_kicker_projection_features(range(2009, 2026), rosters=rosters)
    k_results = project_kicker_season(k_ppg, k_games, k_proj_df, quantile_models=k_qmodels)

    # --- DST Projections ---
    print("\n--- DST Model ---")
    dst_df = build_dst_features(range(2009, 2026))
    dst_walk_forward_eval(dst_df)
    dst_ppg, dst_games, dst_imp, dst_qmodels = train_dst_model(dst_df)
    dst_proj_df = build_dst_projection_features(range(2009, 2026))
    dst_results = project_dst_season(dst_ppg, dst_games, dst_proj_df, quantile_models=dst_qmodels)

    # Set current_team for DST from the team column
    if "team" in dst_results.columns:
        dst_results = dst_results.with_columns(
            pl.col("team").alias("current_team")
        )

    # Step 7: Write CSVs and print rankings
    _write_projection_csvs(results, k_results, dst_results)

    has_quantiles = "ppg_floor" in results.columns

    for pos in ["QB", "RB", "WR", "TE"]:
        n = 40 if pos in ("RB", "WR") else 20
        pos_df = (
            results.filter(pl.col("position_group") == pos)
            .sort("projected_total", descending=True)
            .head(n)
        )
        print(f"\n{'='*95}")
        print(f"  {pos} RANKINGS — 2026 Season-Level Projections (PPR)")
        print(f"{'='*95}")
        if has_quantiles:
            print(f"{'Rank':<6}{'Player':<26}{'Team':<6}{'PPG':>7}{'Median':>7}{'Floor':>7}{'Ceil':>7}{'Games':>7}{'Total':>7}")
            print(f"{'-'*6}{'-'*26}{'-'*6}{'-'*7}{'-'*7}{'-'*7}{'-'*7}{'-'*7}{'-'*7}")
        else:
            print(f"{'Rank':<6}{'Player':<26}{'Team':<6}{'PPG':>7}{'Games':>7}{'Total':>7}")
            print(f"{'-'*6}{'-'*26}{'-'*6}{'-'*7}{'-'*7}{'-'*7}")
        for i, row in enumerate(pos_df.iter_rows(named=True), 1):
            name = (row.get("player_display_name") or "???")[:24]
            team = (row.get("current_team") or "???")[:5]
            ppg = row["projected_ppg"]
            games = row["projected_games"]
            total = row["projected_total"]
            if has_quantiles:
                floor = row.get("ppg_floor", 0)
                median = row.get("ppg_median", 0)
                ceil = row.get("ppg_ceiling", 0)
                print(f"{i:<6}{name:<26}{team:<6}{ppg:>7.1f}{median:>7.1f}{floor:>7.1f}{ceil:>7.1f}{games:>7.1f}{total:>7}")
            else:
                print(f"{i:<6}{name:<26}{team:<6}{ppg:>7.1f}{games:>7.1f}{total:>7}")

    # K rankings
    k_top = k_results.sort("projected_total", descending=True).head(15)
    print(f"\n{'='*70}")
    print(f"  K RANKINGS — 2026 Season-Level Projections")
    print(f"{'='*70}")
    print(f"{'Rank':<6}{'Player':<26}{'Team':<6}{'PPG':>7}{'Games':>7}{'Total':>7}")
    print(f"{'-'*6}{'-'*26}{'-'*6}{'-'*7}{'-'*7}{'-'*7}")
    for i, row in enumerate(k_top.iter_rows(named=True), 1):
        name = (row.get("player_display_name") or "???")[:24]
        team = (row.get("current_team") or row.get("team") or "???")[:5]
        print(f"{i:<6}{name:<26}{team:<6}{row['projected_ppg']:>7.1f}{row['projected_games']:>7.1f}{row['projected_total']:>7}")

    # DST rankings
    dst_top = dst_results.sort("projected_total", descending=True).head(16)
    print(f"\n{'='*70}")
    print(f"  DST RANKINGS — 2026 Season-Level Projections")
    print(f"{'='*70}")
    print(f"{'Rank':<6}{'Team DST':<26}{'Team':<6}{'PPG':>7}{'Games':>7}{'Total':>7}")
    print(f"{'-'*6}{'-'*26}{'-'*6}{'-'*7}{'-'*7}{'-'*7}")
    for i, row in enumerate(dst_top.iter_rows(named=True), 1):
        name = (row.get("player_display_name") or "???")[:24]
        team = (row.get("current_team") or row.get("team") or "???")[:5]
        print(f"{i:<6}{name:<26}{team:<6}{row['projected_ppg']:>7.1f}{row['projected_games']:>7.1f}{row['projected_total']:>7}")

    # Overall draft board
    overall = results.sort("projected_total", descending=True).head(60)
    print(f"\n{'='*100}")
    print(f"  OVERALL DRAFT BOARD — Top 60 (PPR, Season-Level Model)")
    print(f"{'='*100}")
    if has_quantiles:
        print(f"{'#':<5}{'Pos':<7}{'Player':<26}{'Team':<6}{'PPG':>7}{'Median':>7}{'Floor':>7}{'Ceil':>7}{'Games':>7}{'Total':>7}")
        print(f"{'-'*5}{'-'*7}{'-'*26}{'-'*6}{'-'*7}{'-'*7}{'-'*7}{'-'*7}{'-'*7}{'-'*7}")
    else:
        print(f"{'#':<5}{'Pos':<7}{'Player':<26}{'Team':<6}{'PPG':>7}{'Games':>7}{'Total':>7}")
        print(f"{'-'*5}{'-'*7}{'-'*26}{'-'*6}{'-'*7}{'-'*7}{'-'*7}")
    for i, row in enumerate(overall.iter_rows(named=True), 1):
        name = (row.get("player_display_name") or "???")[:24]
        team = (row.get("current_team") or "???")[:5]
        pos = row.get("rank_label", "?")
        ppg = row["projected_ppg"]
        games = row["projected_games"]
        total = row["projected_total"]
        if has_quantiles:
            floor = row.get("ppg_floor", 0)
            median = row.get("ppg_median", 0)
            ceil = row.get("ppg_ceiling", 0)
            print(f"{i:<5}{pos:<7}{name:<26}{team:<6}{ppg:>7.1f}{median:>7.1f}{floor:>7.1f}{ceil:>7.1f}{games:>7.1f}{total:>7}")
        else:
            print(f"{i:<5}{pos:<7}{name:<26}{team:<6}{ppg:>7.1f}{games:>7.1f}{total:>7}")


def _write_projection_csvs(results, k_results=None, dst_results=None):
    """Write per-position and overall projection CSVs to projections/ dir."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Columns for the CSV output
    csv_cols = ["player_display_name", "position_group", "current_team",
                "projected_ppg", "ppg_median", "ppg_floor", "ppg_ceiling",
                "projected_games", "projected_total", "total_floor", "total_ceiling",
                "pos_rank"]
    available = [c for c in csv_cols if c in results.columns]

    # Per-position files for offensive players
    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = (
            results.filter(pl.col("position_group") == pos)
            .sort("projected_total", descending=True)
            .select(available)
        )
        path = OUTPUT_DIR / f"{pos.lower()}_projections.csv"
        pos_df.write_csv(path)
        print(f"  Wrote {pos_df.shape[0]} {pos}s to {path}")

    # K/DST files
    for label, extra in [("k", k_results), ("dst", dst_results)]:
        if extra is None:
            continue
        # Ensure current_team exists
        if "current_team" not in extra.columns and "team" in extra.columns:
            extra = extra.with_columns(pl.col("team").alias("current_team"))
        extra_avail = [c for c in csv_cols if c in extra.columns]
        extra_df = extra.sort("projected_total", descending=True).select(extra_avail)
        path = OUTPUT_DIR / f"{label}_projections.csv"
        extra_df.write_csv(path)
        print(f"  Wrote {extra_df.shape[0]} {label.upper()}s to {path}")

    # Overall file (all positions combined, including K/DST)
    all_parts = [results.sort("projected_total", descending=True).select(available)]
    for extra in [k_results, dst_results]:
        if extra is None:
            continue
        if "current_team" not in extra.columns and "team" in extra.columns:
            extra = extra.with_columns(pl.col("team").alias("current_team"))
        extra_avail = [c for c in available if c in extra.columns]
        all_parts.append(extra.sort("projected_total", descending=True).select(extra_avail))

    all_df = pl.concat(all_parts, how="diagonal").sort("projected_total", descending=True)
    path = OUTPUT_DIR / "all_projections.csv"
    all_df.write_csv(path)
    print(f"  Wrote {all_df.shape[0]} total players to {path}")


if __name__ == "__main__":
    main()
