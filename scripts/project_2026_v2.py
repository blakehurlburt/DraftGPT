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
    # Step 1: Build season features including 2025 (which becomes prior-season data for 2026)
    # We include 2026 in the range so that players with 2025 data get a "2026 row"
    # with prior-season features from 2025. But 2026 actuals won't exist yet.
    print("Building season features (2010-2025)...")
    df = build_season_features(range(2009, 2026))

    # Step 2: Run walk-forward eval first to see model quality
    print("\n--- Walk-Forward Evaluation (for reference) ---")
    from nfldata.season_model import walk_forward_eval
    walk_forward_eval(df)

    # Step 3: Train final model on all available data
    print("\n--- Training Final Model ---")
    ppg_model, games_model, importance, quantile_models = train_final_model(df)

    # Step 4: Build projection features for 2026
    # Players who played in 2025 will have prior-season features.
    # We need to create "2026 rows" with 2025 as the prior season.
    # The easiest way: rebuild features including a synthetic 2026 entry.
    # But since build_season_features only looks at actual data, we need
    # the 2025 season rows with their prior-season features already computed.
    # The "2025 target rows" in df already have prior1_* features from 2024.
    # We need features where 2025 IS the prior season.

    # Step 5: Get current rosters (needed before projection features for team override)
    print("\nLoading current rosters...")
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

    # Rebuild with 2025 as the most recent season to project from
    print("\nBuilding projection features...")
    proj_df = _build_projection_features(range(2009, 2026), rosters=rosters)

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


def _build_projection_features(seasons, rosters=None):
    """Build feature rows for next-season projection.

    Creates one row per player using their most recent season as
    the "prior" season. These rows have prior-season features filled
    but no targets (since the season hasn't happened yet).

    Args:
        seasons: Iterable of season years for historical data.
        rosters: Optional roster DataFrame with player_id, current_team,
                 and adjustment_ppg columns. Used to set the projection
                 year's team assignment so changed_team is detected correctly.
    """
    import nflreadpy as nfl
    from nfldata.season_features import (
        _aggregate_to_season, _add_injury_features, _add_player_metadata,
        _add_roster_context_features, _build_prior_features,
        _add_combine_draft_features,
    )

    # Get season-level aggregates
    season_df = _aggregate_to_season(seasons)

    # For projection, we want each player's LAST season to become their prior features
    max_season = season_df["season"].max()
    print(f"  Projecting from {max_season} season data")

    # Create dummy rows for max_season + 1
    players_latest = season_df.filter(pl.col("season") == max_season)
    dummy = players_latest.with_columns(pl.lit(max_season + 1).alias("season"))

    # Set current-season stats to null (they're targets we don't have)
    # Keep id cols and team (team will be overridden from rosters if available)
    id_cols = {"player_id", "season", "player_display_name", "position_group", "team"}
    stat_cols = [c for c in dummy.columns if c not in id_cols]
    dummy = dummy.with_columns([pl.lit(None).cast(dummy[c].dtype).alias(c) for c in stat_cols])

    # Override team from current rosters for the projection year
    # This ensures changed_team is correctly detected for players who moved
    if rosters is not None and "current_team" in rosters.columns:
        dummy = dummy.drop("team").join(
            rosters.select(["player_id", pl.col("current_team").alias("team")]),
            on="player_id",
            how="left",
        )
        # Fall back to last season's team if roster doesn't have this player
        fallback = players_latest.select(["player_id", pl.col("team").alias("_fallback_team")])
        dummy = dummy.join(fallback, on="player_id", how="left")
        dummy = dummy.with_columns(
            pl.col("team").fill_null(pl.col("_fallback_team"))
        ).drop("_fallback_team")

    # --- Add rookie rows from draft picks for the projection year ---
    proj_year = max_season + 1
    try:
        draft_picks = nfl.load_draft_picks([proj_year])
        # Filter to offensive skill positions
        skill_positions = {"QB", "RB", "WR", "TE", "HB", "FB"}
        draft_picks = draft_picks.filter(
            pl.col("position").is_in(skill_positions)
            | pl.col("category").str.contains("(?i)offense")
        )
        draft_picks = draft_picks.filter(pl.col("gsis_id").is_not_null())

        # Map draft position to our position_group
        pos_map = {"QB": "QB", "RB": "RB", "HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}
        draft_picks = draft_picks.with_columns(
            pl.col("position").replace_strict(pos_map, default=None).alias("position_group")
        ).filter(pl.col("position_group").is_not_null())

        # Get display names from load_players()
        players_df = nfl.load_players()
        name_map = (
            players_df.select(["gsis_id", "display_name"])
            .drop_nulls()
            .unique(subset=["gsis_id"])
        )
        rookie_rows = (
            draft_picks.select([
                pl.col("gsis_id").alias("player_id"),
                pl.lit(proj_year).alias("season"),
                pl.col("team"),
                pl.col("position_group"),
            ])
            .join(name_map.rename({"gsis_id": "player_id", "display_name": "player_display_name"}),
                  on="player_id", how="left")
        )

        # Exclude any rookies already in the dataset (shouldn't happen, but safety check)
        existing_ids = set(season_df["player_id"].unique().to_list())
        rookie_rows = rookie_rows.filter(~pl.col("player_id").is_in(list(existing_ids)))

        print(f"  Adding {rookie_rows.shape[0]} rookies from {proj_year} draft")
    except Exception as e:
        print(f"  No {proj_year} draft data available ({e}), skipping rookies")
        rookie_rows = None

    # Append dummy rows (and rookies) and rebuild prior features
    parts = [season_df, dummy]
    if rookie_rows is not None and rookie_rows.shape[0] > 0:
        parts.append(rookie_rows)
    extended = pl.concat(parts, how="diagonal")
    extended = extended.sort(["player_id", "season"])

    # Rebuild prior-season features
    extended = _build_prior_features(extended)

    # Add injury, metadata, and roster context features
    extended = _add_injury_features(extended, seasons)
    extended = _add_player_metadata(extended, seasons)
    extended = _add_combine_draft_features(extended)
    extended = _add_roster_context_features(extended, seasons)

    # Filter to the projection rows only
    proj = extended.filter(pl.col("season") == proj_year)
    # Keep veterans (have prior stats) AND rookies (have combine/draft features)
    proj = proj.filter(
        pl.col("prior1_ppg").is_not_null()
        | (pl.col("is_rookie") == 1.0)
    )

    # Metadata join misses projection rows (season=max+1 has no roster data).
    # Fill from the most recent season's roster data for each player.
    meta_cols = ["years_exp", "height", "weight", "draft_number", "age"]
    available_meta = [c for c in meta_cols if c in proj.columns]
    missing_meta = proj.select("player_id").join(
        extended.filter(pl.col("season") == max_season)
        .select(["player_id"] + available_meta)
        .unique(subset=["player_id"]),
        on="player_id",
        how="left",
    )
    # Bump years_exp and age by 1 for the projection season
    if "years_exp" in missing_meta.columns:
        missing_meta = missing_meta.with_columns(
            (pl.col("years_exp") + 1).alias("years_exp")
        )
    if "age" in missing_meta.columns:
        missing_meta = missing_meta.with_columns(
            (pl.col("age") + 1.0).alias("age")
        )
    # Overwrite the null metadata columns
    proj = proj.drop(available_meta).join(missing_meta, on="player_id", how="left")

    # For rookies, fill metadata from load_players() since they have no prior roster entry
    if rookie_rows is not None and rookie_rows.shape[0] > 0:
        players_meta = nfl.load_players()
        rookie_meta = (
            players_meta.select(["gsis_id", "height", "weight"])
            .drop_nulls(subset=["gsis_id"])
            .unique(subset=["gsis_id"])
            .rename({"gsis_id": "player_id"})
        )
        if "height" in rookie_meta.columns and rookie_meta["height"].dtype != pl.Float64:
            rookie_meta = rookie_meta.with_columns(pl.col("height").cast(pl.Float64, strict=False))
        # Fill nulls for rookies only
        proj = proj.join(
            rookie_meta.rename({"height": "_rk_height", "weight": "_rk_weight"}),
            on="player_id", how="left"
        )
        proj = proj.with_columns([
            pl.col("height").fill_null(pl.col("_rk_height")),
            pl.col("weight").fill_null(pl.col("_rk_weight")),
        ]).drop(["_rk_height", "_rk_weight"])
        # Set years_exp = 0 for rookies
        proj = proj.with_columns(
            pl.when(pl.col("is_rookie") == 1.0)
            .then(0)
            .otherwise(pl.col("years_exp"))
            .alias("years_exp")
        )

    # Add manual adjustments from rosters if available
    if rosters is not None and "adjustment_ppg" in rosters.columns:
        adj = rosters.select(["player_id", "adjustment_ppg"])
        proj = proj.join(adj, on="player_id", how="left")
    if "adjustment_ppg" not in proj.columns:
        proj = proj.with_columns(pl.lit(0.0).alias("adjustment_ppg"))

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
