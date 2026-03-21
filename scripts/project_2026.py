"""
Project 2026 fantasy rankings by position.

Uses the trained model + each player's most recent rolling features
from the 2025 season to estimate expected weekly fantasy points,
then ranks players within each position group.

Roster data is read from rosters.csv — run update_rosters.py first
to populate it, then make manual edits for trades/cuts/signings.
"""

import polars as pl
import numpy as np
from pathlib import Path
from nfldata.features import build_features, get_feature_columns
from nfldata.model import train_model

ROSTER_PATH = Path(__file__).parent.parent / "data" / "rosters.csv"


def get_current_rosters():
    """Load current rosters from rosters.csv.

    The file is populated by update_rosters.py and can be manually edited
    to reflect trades, free agency moves, cuts, and retirements.
    """
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

    # Resolve player names to gsis_id for joining with feature data
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
    # Step 1: Build features on all available data (2018-2025)
    print("Building features (2018-2025)...")
    df = build_features(range(2018, 2026))

    # CR opus: train_model() hardcodes train=<2024, test=2024. But we built features
    # CR opus: through 2025, so 2025 data is loaded but never used for training or testing.
    # CR opus: The model is trained on 2018-2023 and tested on 2024, wasting a full season.
    # Step 2: Train model on 2018-2023, test on 2024 (same as before)
    print("\nTraining model...")
    model, importance = train_model(df)

    # Step 3: For each player, grab their LAST row from the 2025 season
    # This row has rolling features reflecting end-of-season form
    feature_cols = get_feature_columns(df)

    last_2025 = (
        df.filter(pl.col("season") == 2025)
        .sort(["player_id", "week"], descending=[False, True])
        .group_by("player_id")
        .first()
    )
    print(f"\nPlayers with 2025 data: {last_2025.shape[0]}")

    # Step 4: Get current rosters to know team assignments and filter active players
    print("Loading current rosters...")
    rosters = get_current_rosters()
    last_2025 = last_2025.join(rosters, on="player_id", how="left")

    # Filter to active players (on a roster)
    # CR opus: Allowing current_status=null means any player with no roster match
    # CR opus: (e.g., retired players whose gsis_id didn't join) passes through as "active".
    # CR opus: This inflates the player pool with potentially retired/cut players.
    active_statuses = ["ACT", "RES", "PUP", "NFI"]
    last_2025 = last_2025.filter(
        pl.col("current_status").is_in(active_statuses)
        | pl.col("current_status").is_null()
    )
    print(f"Active players: {last_2025.shape[0]}")

    # Step 5: Generate predictions
    X = last_2025.select(feature_cols).to_pandas()
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category")

    preds = model.predict(X)

    # Step 6: Build results dataframe
    id_cols = ["player_id", "player_display_name", "position_group",
               "current_team", "current_status"]
    available = [c for c in id_cols if c in last_2025.columns]

    results = last_2025.select(available).with_columns(
        pl.Series("projected_fppg", np.round(preds, 1))
    )

    # CR opus: Multiplying every player's PPG by 17 assumes all players play every game.
    # CR opus: This ignores injury risk and bye weeks. project_2026_v2.py trains a separate
    # CR opus: games-played model; this v1 script should be deprecated or updated.
    # Add season projected total (17 games)
    results = results.with_columns(
        (pl.col("projected_fppg") * 17).round(0).cast(pl.Int32).alias("projected_total")
    )

    # Step 7: Rank within each position
    results = results.sort("projected_fppg", descending=True)
    results = results.with_columns(
        pl.col("projected_fppg")
        .rank(method="ordinal", descending=True)
        .over("position_group")
        .alias("pos_rank")
    )
    results = results.with_columns(
        (pl.col("position_group") + pl.col("pos_rank").cast(pl.Utf8)).alias("rank_label")
    )

    # Step 8: Print rankings by position
    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = (
            results.filter(pl.col("position_group") == pos)
            .sort("pos_rank")
            .head(40 if pos in ("RB", "WR") else 20)
        )
        print(f"\n{'='*65}")
        print(f"  {pos} RANKINGS — 2026 Projected Fantasy Points (PPR)")
        print(f"{'='*65}")
        print(f"{'Rank':<6}{'Player':<28}{'Team':<6}{'PPG':>7}{'Total':>7}")
        print(f"{'-'*6}{'-'*28}{'-'*6}{'-'*7}{'-'*7}")
        for row in pos_df.iter_rows(named=True):
            name = (row.get("player_display_name") or "???")[:26]
            team = (row.get("current_team") or "???")[:5]
            ppg = row["projected_fppg"]
            total = row["projected_total"]
            rank = row["pos_rank"]
            print(f"{rank:<6}{name:<28}{team:<6}{ppg:>7.1f}{total:>7}")

    # Step 9: Overall draft board (top 60)
    overall = results.sort("projected_fppg", descending=True).head(60)
    print(f"\n{'='*70}")
    print(f"  OVERALL DRAFT BOARD — Top 60 (PPR)")
    print(f"{'='*70}")
    print(f"{'#':<5}{'Pos':<7}{'Player':<28}{'Team':<6}{'PPG':>7}{'Total':>7}")
    print(f"{'-'*5}{'-'*7}{'-'*28}{'-'*6}{'-'*7}{'-'*7}")
    for i, row in enumerate(overall.iter_rows(named=True), 1):
        name = (row.get("player_display_name") or "???")[:26]
        team = (row.get("current_team") or "???")[:5]
        pos = row.get("rank_label", "?")
        ppg = row["projected_fppg"]
        total = row["projected_total"]
        print(f"{i:<5}{pos:<7}{name:<28}{team:<6}{ppg:>7.1f}{total:>7}")


if __name__ == "__main__":
    main()
