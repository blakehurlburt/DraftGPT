"""
Season-level feature pipeline for fantasy points prediction.

Aggregates weekly player stats to season level, then builds prior-season
features (lagged stats, trends, injury history, player metadata) for
predicting next-season PPG and games played.
"""

import polars as pl
import nflreadpy as nfl

# Maximum regular-season games (17 since 2021, 16 before)
MAX_GAMES = {s: 17 if s >= 2021 else 16 for s in range(2018, 2030)}


def _max_games_for_season(season: int) -> int:
    return 17 if season >= 2021 else 16


def _aggregate_to_season(seasons):
    """Step 1: Load player_stats and aggregate to one row per player-season.

    Filters to regular season weeks 1-17 only (excludes week 18, preseason,
    and playoffs) for performance stats.
    """
    season_list = list(seasons)
    print("Loading player stats...")
    stats = nfl.load_player_stats(season_list)

    # Filter to regular season only, weeks 1-17
    if "season_type" in stats.columns:
        stats = stats.filter(pl.col("season_type") == "REG")
    stats = stats.filter(pl.col("week") <= 17)

    # Filter to offensive skill positions
    if "position_group" in stats.columns:
        stats = stats.filter(pl.col("position_group").is_in(["QB", "RB", "WR", "TE"]))

    # Fill nulls in numeric columns we'll aggregate
    agg_cols = [
        "fantasy_points_ppr", "passing_yards", "rushing_yards", "receiving_yards",
        "targets", "carries", "receptions",
        "passing_tds", "rushing_tds", "receiving_tds",
        "passing_epa", "rushing_epa", "receiving_epa",
        "target_share", "wopr",
    ]
    available_agg = [c for c in agg_cols if c in stats.columns]
    stats = stats.with_columns([pl.col(c).fill_null(0.0).cast(pl.Float64) for c in available_agg])

    # Grab identifying info (first occurrence per player-season)
    id_exprs = [
        pl.col("player_display_name").first().alias("player_display_name"),
        pl.col("position_group").first().alias("position_group"),
    ]

    # Per-game means
    mean_exprs = [
        pl.col("fantasy_points_ppr").mean().alias("ppg"),
        pl.col("passing_yards").mean().alias("pass_ypg"),
        pl.col("rushing_yards").mean().alias("rush_ypg"),
        pl.col("receiving_yards").mean().alias("rec_ypg"),
        pl.col("targets").mean().alias("tgt_pg"),
        pl.col("carries").mean().alias("carries_pg"),
        pl.col("receptions").mean().alias("rec_pg"),
        pl.col("passing_tds").mean().alias("pass_td_pg"),
        pl.col("rushing_tds").mean().alias("rush_td_pg"),
        pl.col("receiving_tds").mean().alias("rec_td_pg"),
        pl.col("passing_epa").mean().alias("pass_epa_pg"),
        pl.col("rushing_epa").mean().alias("rush_epa_pg"),
        pl.col("receiving_epa").mean().alias("rec_epa_pg"),
        pl.col("target_share").mean().alias("target_share_avg"),
        pl.col("wopr").mean().alias("wopr_avg"),
    ]

    # Filter to available columns
    mean_exprs = [e for e in mean_exprs if True]  # all should be available after fill

    # Consistency
    std_exprs = [
        pl.col("fantasy_points_ppr").std().alias("ppg_std"),
    ]

    # Games played
    count_expr = [pl.len().alias("games_played")]

    season_df = (
        stats.group_by(["player_id", "season"])
        .agg(id_exprs + count_expr + mean_exprs + std_exprs)
        .sort(["player_id", "season"])
    )

    print(f"  Season-level: {season_df.shape[0]} player-seasons")
    return season_df


def _build_prior_features(season_df):
    """Step 2: Build prior-season features using lag/lookback.

    For each player-season row, creates features from seasons strictly
    before the target season (no leakage).
    """
    # Per-game stat columns to lag
    pg_stats = [
        "ppg", "pass_ypg", "rush_ypg", "rec_ypg", "tgt_pg", "carries_pg",
        "rec_pg", "pass_epa_pg", "rush_epa_pg", "rec_epa_pg",
        "target_share_avg", "wopr_avg",
    ]

    # Sort to ensure correct lag ordering
    df = season_df.sort(["player_id", "season"])

    # --- Prior 1-year features (most recent season) ---
    prior1_exprs = []
    for stat in pg_stats:
        prior1_exprs.append(
            pl.col(stat).shift(1).over("player_id").alias(f"prior1_{stat}")
        )
    # Prior year consistency and games
    prior1_exprs.append(
        pl.col("ppg_std").shift(1).over("player_id").alias("prior1_ppg_std")
    )
    prior1_exprs.append(
        pl.col("games_played").shift(1).over("player_id").alias("prior_games_played")
    )

    df = df.with_columns(prior1_exprs)

    # --- Prior 2-year average (X-1 and X-2) ---
    prior2_stats = ["ppg", "pass_ypg", "rush_ypg", "rec_ypg"]
    prior2_exprs = []
    for stat in prior2_stats:
        # Average of shift(1) and shift(2)
        prior2_exprs.append(
            ((pl.col(stat).shift(1) + pl.col(stat).shift(2)) / 2.0)
            .over("player_id")
            .alias(f"{stat}_2yr")
        )
    df = df.with_columns(prior2_exprs)

    # --- Trajectory (X-1 minus X-2) ---
    trend_stats = ["ppg", "rush_ypg", "rec_ypg", "tgt_pg"]
    trend_exprs = []
    for stat in trend_stats:
        trend_exprs.append(
            (pl.col(stat).shift(1) - pl.col(stat).shift(2))
            .over("player_id")
            .alias(f"{stat}_trend")
        )
    df = df.with_columns(trend_exprs)

    # --- Career games played rate ---
    # For each row, compute cumulative games / cumulative max possible games
    # across all prior seasons. We need to be careful about season boundaries.
    # We'll compute this via a cumulative sum approach.
    df = df.with_columns(
        pl.col("games_played")
        .shift(1)
        .cum_sum()
        .over("player_id")
        .alias("_cum_games")
    )

    # Cumulative max possible games (each prior season = 16 or 17)
    # We'll use a simpler approach: count prior seasons * avg max games
    df = df.with_columns(
        pl.lit(1)
        .shift(1)
        .cum_sum()
        .over("player_id")
        .alias("_cum_seasons")
    )

    # Map seasons to max games and compute cumulative possible
    df = df.with_columns(
        pl.col("season").map_elements(
            lambda s: _max_games_for_season(s), return_dtype=pl.Int64
        ).alias("_max_games_this_season")
    )
    df = df.with_columns(
        pl.col("_max_games_this_season")
        .shift(1)
        .cum_sum()
        .over("player_id")
        .alias("_cum_max_games")
    )
    df = df.with_columns(
        (pl.col("_cum_games") / pl.col("_cum_max_games")).alias("career_games_rate")
    )

    # --- Best prior season PPG ---
    df = df.with_columns(
        pl.col("ppg")
        .shift(1)
        .cum_max()
        .over("player_id")
        .alias("best_ppg")
    )

    # --- Prior games missed ---
    df = df.with_columns(
        pl.col("season")
        .shift(1)
        .over("player_id")
        .map_elements(lambda s: _max_games_for_season(s) if s is not None else None, return_dtype=pl.Int64)
        .alias("_prior_season_max")
    )
    df = df.with_columns(
        (pl.col("_prior_season_max") - pl.col("prior_games_played")).alias("prior_games_missed")
    )

    # Drop temp columns
    df = df.drop([c for c in df.columns if c.startswith("_")])

    return df


def _add_injury_features(df, seasons):
    """Step 3: Add injury history features from prior season."""
    season_list = list(seasons)
    print("Loading injuries...")
    try:
        inj = nfl.load_injuries(season_list)
    except Exception as e:
        print(f"  Warning: Failed to load injuries: {e}")
        # Add null injury columns
        for col in ["inj_weeks_on_report", "inj_times_out", "inj_times_questionable", "inj_distinct_injuries"]:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
        return df

    # Rename gsis_id to player_id for joining
    if "gsis_id" in inj.columns:
        inj = inj.rename({"gsis_id": "player_id"})

    # Keep all injury data (including playoffs/week 18) per the user's request
    # Aggregate injury data per player per season
    inj_agg = (
        inj.group_by(["player_id", "season"])
        .agg([
            pl.len().alias("inj_weeks_on_report"),
            (pl.col("report_status") == "Out").sum().alias("inj_times_out"),
            (pl.col("report_status") == "Questionable").sum().alias("inj_times_questionable"),
            pl.col("report_primary_injury")
            .drop_nulls()
            .n_unique()
            .alias("inj_distinct_injuries"),
        ])
    )

    # These are prior-season injury features, so shift by joining on season+1
    inj_agg = inj_agg.with_columns(
        (pl.col("season") + 1).cast(pl.Int32).alias("target_season")
    )
    inj_agg = inj_agg.drop("season").rename({"target_season": "season"})

    # Cast to float for consistency
    for col in ["inj_weeks_on_report", "inj_times_out", "inj_times_questionable", "inj_distinct_injuries"]:
        inj_agg = inj_agg.with_columns(pl.col(col).cast(pl.Float64))

    df = df.join(inj_agg, on=["player_id", "season"], how="left")

    return df


def _add_player_metadata(df, seasons):
    """Step 4: Add player metadata from rosters."""
    season_list = list(seasons)
    print("Loading rosters for metadata...")
    rosters = nfl.load_rosters(season_list)

    keep_cols = ["gsis_id", "season", "years_exp", "height", "weight", "draft_number", "birth_date"]
    available = [c for c in keep_cols if c in rosters.columns]
    rosters = rosters.select(available).unique(subset=["gsis_id", "season"])
    rosters = rosters.rename({"gsis_id": "player_id"})

    # Ensure height is numeric
    if "height" in rosters.columns and rosters["height"].dtype != pl.Float64:
        rosters = rosters.with_columns(pl.col("height").cast(pl.Float64, strict=False))

    # Compute age if birth_date available
    if "birth_date" in rosters.columns:
        rosters = rosters.with_columns(
            # Age at start of season (September 1)
            ((pl.date(pl.col("season"), 9, 1) - pl.col("birth_date")).dt.total_days() / 365.25)
            .round(1)
            .alias("age")
        )
        rosters = rosters.drop("birth_date")

    df = df.join(rosters, on=["player_id", "season"], how="left")

    return df


def build_season_features(seasons):
    """Build the full season-level feature matrix.

    Args:
        seasons: Iterable of season years (e.g., range(2018, 2025)).

    Returns:
        Polars DataFrame with one row per player-season, containing
        prior-season features and target columns. Only includes seasons
        where prior-season data exists (drops rookies and first appearances).
    """
    print("=== Building Season-Level Features ===")

    # Step 1: Aggregate to season level
    season_df = _aggregate_to_season(seasons)

    # Step 2: Build prior-season features
    print("Building prior-season features...")
    df = _build_prior_features(season_df)

    # Step 3: Add injury features
    df = _add_injury_features(df, seasons)

    # Step 4: Add player metadata
    df = _add_player_metadata(df, seasons)

    # Step 5: Define targets
    df = df.with_columns([
        pl.col("ppg").alias("target_ppg"),
        pl.col("games_played").alias("target_games"),
        (pl.col("ppg") * pl.col("games_played")).alias("target_total"),
    ])

    # Drop rows without prior-season data (first appearance for each player)
    df = df.filter(pl.col("prior1_ppg").is_not_null())

    # Drop seasons before 2020 as targets (need 2018-2019 as history)
    df = df.filter(pl.col("season") >= 2020)

    # Define feature columns (everything that's not an ID, target, or raw current-season stat)
    print(f"  Final dataset: {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"  Seasons with targets: {sorted(df['season'].unique().to_list())}")

    return df


def get_season_feature_columns(df):
    """Return the list of feature column names for the season model."""
    drop_cols = {
        # IDs
        "player_id", "player_display_name", "position_group", "season",
        # Current-season raw stats (these are targets or leakage)
        "ppg", "pass_ypg", "rush_ypg", "rec_ypg", "tgt_pg", "carries_pg",
        "rec_pg", "pass_td_pg", "rush_td_pg", "rec_td_pg",
        "pass_epa_pg", "rush_epa_pg", "rec_epa_pg",
        "target_share_avg", "wopr_avg", "ppg_std", "games_played",
        # Targets
        "target_ppg", "target_games", "target_total",
    }
    return [c for c in df.columns if c not in drop_cols]
