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

ROSTER_PATH = Path(__file__).parent / "rosters.csv"
OUTPUT_DIR = Path(__file__).parent / "projections"


def get_current_rosters():
    """Load current rosters from rosters.csv."""
    import nflreadpy as nfl

    if not ROSTER_PATH.exists():
        print("  rosters.csv not found — run 'python update_rosters.py' first")
        print("  Falling back to nflverse weekly rosters...")
        rw = nfl.load_rosters_weekly([2025])
        latest = (
            rw.sort("week", descending=True)
            .group_by("gsis_id")
            .first()
            .select(["gsis_id", "team", "position", "status"])
            .rename({"gsis_id": "player_id", "team": "current_team",
                     "position": "current_position", "status": "current_status"})
        )
        return latest

    roster_df = pl.read_csv(ROSTER_PATH, comment_prefix="#")
    print(f"  Loaded {roster_df.shape[0]} players from rosters.csv")

    name_map = (
        nfl.load_players()
        .select(["gsis_id", "display_name"])
        .drop_nulls()
        .unique(subset=["display_name"])
        .rename({"gsis_id": "player_id", "display_name": "player_name"})
    )

    roster_df = roster_df.join(name_map, on="player_name", how="left")
    unmatched = roster_df.filter(pl.col("player_id").is_null()).shape[0]
    if unmatched > 0:
        print(f"  Warning: {unmatched} players could not be matched to nflverse IDs")

    roster_df = (
        roster_df.filter(pl.col("player_id").is_not_null())
        .select([
            pl.col("player_id"),
            pl.col("team").alias("current_team"),
            pl.col("position").alias("current_position"),
            pl.col("status").alias("current_status"),
        ])
    )

    return roster_df


def main():
    # Step 1: Build season features including 2025 (which becomes prior-season data for 2026)
    # We include 2026 in the range so that players with 2025 data get a "2026 row"
    # with prior-season features from 2025. But 2026 actuals won't exist yet.
    print("Building season features (2018-2025)...")
    df = build_season_features(range(2018, 2026))

    # Step 2: Run walk-forward eval first to see model quality
    print("\n--- Walk-Forward Evaluation (for reference) ---")
    from nfldata.season_model import walk_forward_eval
    walk_forward_eval(df)

    # Step 3: Train final model on all available data
    print("\n--- Training Final Model ---")
    ppg_model, games_model, importance = train_final_model(df)

    # Step 4: Build projection features for 2026
    # Players who played in 2025 will have prior-season features.
    # We need to create "2026 rows" with 2025 as the prior season.
    # The easiest way: rebuild features including a synthetic 2026 entry.
    # But since build_season_features only looks at actual data, we need
    # the 2025 season rows with their prior-season features already computed.
    # The "2025 target rows" in df already have prior1_* features from 2024.
    # We need features where 2025 IS the prior season.

    # Rebuild with 2025 as the most recent season to project from
    print("\nBuilding projection features...")
    # Get the 2025 season-level stats (these become "prior" for 2026 projection)
    proj_df = _build_projection_features(range(2018, 2026))

    # Step 5: Get current rosters
    print("\nLoading current rosters...")
    rosters = get_current_rosters()

    # Step 6: Project
    results = project_season(ppg_model, games_model, proj_df)

    # Join with rosters
    results = results.join(rosters, on="player_id", how="left")

    # Filter to active players
    active_statuses = ["ACT", "RES", "PUP", "NFI"]
    results = results.filter(
        pl.col("current_status").is_in(active_statuses)
        | pl.col("current_status").is_null()
    )

    # Step 7: Write CSVs and print rankings
    _write_projection_csvs(results)

    for pos in ["QB", "RB", "WR", "TE"]:
        n = 40 if pos in ("RB", "WR") else 20
        pos_df = (
            results.filter(pl.col("position_group") == pos)
            .sort("projected_total", descending=True)
            .head(n)
        )
        print(f"\n{'='*72}")
        print(f"  {pos} RANKINGS — 2026 Season-Level Projections (PPR)")
        print(f"{'='*72}")
        print(f"{'Rank':<6}{'Player':<28}{'Team':<6}{'PPG':>7}{'Games':>7}{'Total':>7}")
        print(f"{'-'*6}{'-'*28}{'-'*6}{'-'*7}{'-'*7}{'-'*7}")
        for i, row in enumerate(pos_df.iter_rows(named=True), 1):
            name = (row.get("player_display_name") or "???")[:26]
            team = (row.get("current_team") or "???")[:5]
            ppg = row["projected_ppg"]
            games = row["projected_games"]
            total = row["projected_total"]
            print(f"{i:<6}{name:<28}{team:<6}{ppg:>7.1f}{games:>7.1f}{total:>7}")

    # Overall draft board
    overall = results.sort("projected_total", descending=True).head(60)
    print(f"\n{'='*78}")
    print(f"  OVERALL DRAFT BOARD — Top 60 (PPR, Season-Level Model)")
    print(f"{'='*78}")
    print(f"{'#':<5}{'Pos':<7}{'Player':<28}{'Team':<6}{'PPG':>7}{'Games':>7}{'Total':>7}")
    print(f"{'-'*5}{'-'*7}{'-'*28}{'-'*6}{'-'*7}{'-'*7}{'-'*7}")
    for i, row in enumerate(overall.iter_rows(named=True), 1):
        name = (row.get("player_display_name") or "???")[:26]
        team = (row.get("current_team") or "???")[:5]
        pos = row.get("rank_label", "?")
        ppg = row["projected_ppg"]
        games = row["projected_games"]
        total = row["projected_total"]
        print(f"{i:<5}{pos:<7}{name:<28}{team:<6}{ppg:>7.1f}{games:>7.1f}{total:>7}")


def _write_projection_csvs(results):
    """Write per-position and overall projection CSVs to projections/ dir."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Columns for the CSV output
    csv_cols = ["player_display_name", "position_group", "current_team",
                "projected_ppg", "projected_games", "projected_total", "pos_rank"]
    available = [c for c in csv_cols if c in results.columns]

    # Per-position files
    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = (
            results.filter(pl.col("position_group") == pos)
            .sort("projected_total", descending=True)
            .select(available)
        )
        path = OUTPUT_DIR / f"{pos.lower()}_projections.csv"
        pos_df.write_csv(path)
        print(f"  Wrote {pos_df.shape[0]} {pos}s to {path}")

    # Overall file (all positions combined)
    all_df = results.sort("projected_total", descending=True).select(available)
    path = OUTPUT_DIR / "all_projections.csv"
    all_df.write_csv(path)
    print(f"  Wrote {all_df.shape[0]} total players to {path}")


def _build_projection_features(seasons):
    """Build feature rows for next-season projection.

    Creates one row per player using their most recent season as
    the "prior" season. These rows have prior-season features filled
    but no targets (since the season hasn't happened yet).
    """
    import nflreadpy as nfl
    from nfldata.season_features import _aggregate_to_season, _add_injury_features, _add_player_metadata

    # Get season-level aggregates
    season_df = _aggregate_to_season(seasons)

    # For projection, we want each player's LAST season to become their prior features
    # Get players who played in the most recent season
    max_season = season_df["season"].max()
    print(f"  Projecting from {max_season} season data")

    # Build prior features the same way, then filter to the synthetic "next year" row
    # We'll add a dummy next-season row for each player who played in max_season
    players_latest = season_df.filter(pl.col("season") == max_season)

    # Create dummy rows for max_season + 1
    dummy = players_latest.with_columns(pl.lit(max_season + 1).alias("season"))
    # Set current-season stats to null (they're targets we don't have)
    # Preserve each column's original dtype to avoid schema conflicts on concat
    id_cols = {"player_id", "season", "player_display_name", "position_group"}
    stat_cols = [c for c in dummy.columns if c not in id_cols]
    dummy = dummy.with_columns([pl.lit(None).cast(dummy[c].dtype).alias(c) for c in stat_cols])

    # Append dummy rows and rebuild prior features
    extended = pl.concat([season_df, dummy], how="diagonal")
    extended = extended.sort(["player_id", "season"])

    # Rebuild prior-season features
    from nfldata.season_features import _build_prior_features
    extended = _build_prior_features(extended)

    # Add injury and metadata features
    extended = _add_injury_features(extended, seasons)
    extended = _add_player_metadata(extended, seasons)

    # Filter to the projection rows only
    proj = extended.filter(pl.col("season") == max_season + 1)
    proj = proj.filter(pl.col("prior1_ppg").is_not_null())

    # Add dummy targets (won't be used, but needed for column compatibility)
    proj = proj.with_columns([
        pl.lit(0.0).alias("target_ppg"),
        pl.lit(0.0).alias("target_games"),
        pl.lit(0.0).alias("target_total"),
    ])

    print(f"  Projection rows: {proj.shape[0]} players")
    return proj


if __name__ == "__main__":
    main()
