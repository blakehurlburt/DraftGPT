"""
DST (Defense/Special Teams) feature pipeline for fantasy points prediction.

Uses team-level defensive stats from nfl.load_team_stats(). Each team is
represented as a synthetic player with player_id = "DST_{team_abbr}".
"""

import polars as pl
import nflreadpy as nfl

# Team abbreviation normalization for franchise relocations/renames
_TEAM_NORM = {
    "OAK": "LV",
    "STL": "LA",
    "SD": "LAC",
    "WAS": "WSH",
}

# Points-allowed scoring tiers
_PA_TIERS = [
    (0, 0, 10),      # 0 points allowed: +10
    (1, 6, 7),       # 1-6: +7
    (7, 13, 4),      # 7-13: +4
    (14, 20, 1),     # 14-20: +1
    (21, 27, 0),     # 21-27: 0
    (28, 34, -1),    # 28-34: -1
    (35, 999, -4),   # 35+: -4
]


def _normalize_team(team):
    """Normalize team abbreviation for relocated/renamed franchises."""
    return _TEAM_NORM.get(team, team)


def _pa_score(pts_allowed):
    """Compute points-allowed fantasy score from a points-allowed value."""
    if pts_allowed is None:
        return 0.0
    pts = int(pts_allowed)
    for low, high, score in _PA_TIERS:
        if low <= pts <= high:
            return float(score)
    return -4.0


def _compute_dst_fpts(row):
    """Compute DST fantasy points for a single week.

    Scoring: sack=1, INT=2, fumble recovery=2, def TD=6, safety=2,
             plus points-allowed tier bonus.
    """
    sacks = row.get("def_sacks", 0) or 0
    ints = row.get("def_interceptions", 0) or 0
    fumbles_rec = row.get("fumble_recovery_opp", 0) or 0
    def_tds = row.get("def_tds", 0) or 0
    safeties = row.get("def_safeties", 0) or 0
    pts_allowed = row.get("pts_allowed", 0) or 0

    pts = (
        sacks * 1
        + ints * 2
        + fumbles_rec * 2
        + def_tds * 6
        + safeties * 2
        + _pa_score(pts_allowed)
    )
    return float(pts)


def _aggregate_dst_to_season(seasons):
    """Load team stats and aggregate to one row per team-season for DST."""
    season_list = list(seasons)
    print("Loading team stats for DST...")
    ts = nfl.load_team_stats(season_list)

    if "season_type" in ts.columns:
        ts = ts.filter(pl.col("season_type") == "REG")
    ts = ts.filter(pl.col("week") <= 17)

    # Normalize team abbreviations
    ts = ts.with_columns(
        pl.col("team").map_elements(_normalize_team, return_dtype=pl.Utf8).alias("team")
    )

    # Normalize opponent team abbreviations too
    if "opponent_team" in ts.columns:
        ts = ts.with_columns(
            pl.col("opponent_team").map_elements(_normalize_team, return_dtype=pl.Utf8)
            .alias("opponent_team")
        )

    # Compute points allowed by joining opponent's offensive output
    opp_scoring = ts.select([
        pl.col("team").alias("_join_team"),
        "season", "week",
        (
            (pl.col("passing_tds").fill_null(0) + pl.col("rushing_tds").fill_null(0)) * 7
            + pl.col("fg_made").fill_null(0) * 3
            + pl.col("pat_made").fill_null(0)
            + (pl.col("passing_2pt_conversions").fill_null(0)
               + pl.col("rushing_2pt_conversions").fill_null(0)
               + pl.col("receiving_2pt_conversions").fill_null(0)) * 2
            + pl.col("def_tds").fill_null(0) * 6
            + pl.col("def_safeties").fill_null(0) * 2
        ).alias("_opp_pts"),
    ])
    ts = ts.join(
        opp_scoring,
        left_on=["opponent_team", "season", "week"],
        right_on=["_join_team", "season", "week"],
        how="left",
    )
    ts = ts.with_columns(pl.col("_opp_pts").fill_null(21.0).alias("pts_allowed"))

    def_cols = [
        "def_sacks", "def_interceptions", "fumble_recovery_opp",
        "def_tds", "def_safeties",
    ]

    # Fill nulls
    all_def_cols = def_cols + ["pts_allowed"]
    available = [c for c in all_def_cols if c in ts.columns]
    ts = ts.with_columns([pl.col(c).fill_null(0).cast(pl.Float64) for c in available])

    # Compute weekly DST fantasy points
    ts = ts.with_columns(
        pl.struct(available)
        .map_elements(_compute_dst_fpts, return_dtype=pl.Float64)
        .alias("dst_fpts")
    )

    # Aggregate to season level
    agg_exprs = [
        pl.len().alias("games_played"),
        pl.col("dst_fpts").mean().alias("ppg"),
        pl.col("pts_allowed").mean().alias("pts_allowed_pg"),
    ]

    for col in def_cols:
        if col in ts.columns:
            short = col.replace("def_", "").replace("fumble_recovery_opp", "fumbles_recovered")
            agg_exprs.append(pl.col(col).mean().alias(f"{short}_pg"))

    # Takeaways = interceptions + fumble recoveries
    if "def_interceptions" in ts.columns and "fumble_recovery_opp" in ts.columns:
        agg_exprs.append(
            (pl.col("def_interceptions") + pl.col("fumble_recovery_opp"))
            .mean().alias("takeaways_pg")
        )

    season_df = (
        ts.group_by(["team", "season"])
        .agg(agg_exprs)
        .sort(["team", "season"])
    )

    # Create synthetic player_id and display name
    season_df = season_df.with_columns([
        (pl.lit("DST_") + pl.col("team")).alias("player_id"),
        (pl.col("team") + pl.lit(" DST")).alias("player_display_name"),
        pl.lit("DST").alias("position_group"),
    ])

    print(f"  DST season-level: {season_df.shape[0]} team-seasons")
    return season_df


def _build_dst_prior_features(df):
    """Build prior-season features for DSTs using lag/lookback."""
    df = df.sort(["player_id", "season"])

    # Identify per-game stat columns to lag
    pg_stats = [c for c in df.columns if c.endswith("_pg") and c != "ppg"]
    pg_stats = ["ppg"] + pg_stats

    # Prior 1-year features
    prior1_exprs = []
    for stat in pg_stats:
        prior1_exprs.append(
            pl.col(stat).shift(1).over("player_id").alias(f"prior1_{stat}")
        )
    prior1_exprs.append(
        pl.col("games_played").shift(1).over("player_id").alias("prior_games_played")
    )
    df = df.with_columns(prior1_exprs)

    # Prior 2-year average PPG
    df = df.with_columns(
        ((pl.col("ppg").shift(1) + pl.col("ppg").shift(2)) / 2.0)
        .over("player_id")
        .alias("ppg_2yr")
    )

    # PPG trend
    df = df.with_columns(
        (pl.col("ppg").shift(1) - pl.col("ppg").shift(2))
        .over("player_id")
        .alias("ppg_trend")
    )

    # Career games rate
    df = df.with_columns([
        pl.col("games_played")
        .shift(1).fill_null(0).cum_sum()
        .over("player_id")
        .alias("_cum_games"),
        pl.lit(1)
        .shift(1).fill_null(0).cum_sum()
        .over("player_id")
        .alias("_cum_seasons"),
    ])
    df = df.with_columns(
        pl.col("season").map_elements(
            lambda s: 17 if s >= 2021 else 16, return_dtype=pl.Int64
        ).alias("_max_games")
    )
    df = df.with_columns(
        pl.col("_max_games").shift(1).fill_null(0).cum_sum()
        .over("player_id").alias("_cum_max_games")
    )
    df = df.with_columns(
        (pl.col("_cum_games") / pl.col("_cum_max_games")).alias("career_games_rate")
    )

    # Best prior PPG
    df = df.with_columns(
        pl.col("ppg").shift(1).cum_max().over("player_id").alias("best_ppg")
    )

    # Drop temp columns
    df = df.drop([c for c in df.columns if c.startswith("_")])
    return df


def build_dst_features(seasons):
    """Build the full DST feature matrix for training.

    Returns:
        Polars DataFrame with one row per team-season, containing
        prior-season features and target columns.
    """
    print("=== Building DST Features ===")

    season_df = _aggregate_dst_to_season(seasons)
    df = _build_dst_prior_features(season_df)

    # Targets
    df = df.with_columns([
        pl.col("ppg").alias("target_ppg"),
        pl.col("games_played").alias("target_games"),
        (pl.col("ppg") * pl.col("games_played")).alias("target_total"),
    ])

    # Keep rows with prior-season data
    df = df.filter(pl.col("prior1_ppg").is_not_null())
    df = df.filter(pl.col("season") >= 2011)

    # Fill nulls in feature columns
    for col in df.columns:
        if df[col].dtype in (pl.Float64, pl.Int64) and col not in (
            "player_id", "season", "target_ppg", "target_games", "target_total"
        ):
            df = df.with_columns(pl.col(col).fill_null(0.0))

    print(f"  Final DST dataset: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def build_dst_projection_features(seasons):
    """Build feature rows for next-season DST projection.

    Creates one row per team using their most recent season as prior.
    """
    print("Building DST projection features...")
    season_df = _aggregate_dst_to_season(seasons)
    max_season = season_df["season"].max()
    print(f"  Projecting DSTs from {max_season} season data")

    # Create dummy rows for projection year
    teams_latest = season_df.filter(pl.col("season") == max_season)
    dummy = teams_latest.with_columns(pl.lit(max_season + 1).alias("season"))

    id_cols = {"player_id", "season", "player_display_name", "position_group", "team"}
    stat_cols = [c for c in dummy.columns if c not in id_cols]
    dummy = dummy.with_columns([pl.lit(None).cast(dummy[c].dtype).alias(c) for c in stat_cols])

    extended = pl.concat([season_df, dummy], how="diagonal")
    extended = extended.sort(["player_id", "season"])

    extended = _build_dst_prior_features(extended)

    proj_year = max_season + 1
    proj = extended.filter(pl.col("season") == proj_year)
    proj = proj.filter(pl.col("prior1_ppg").is_not_null())

    # Add dummy targets
    proj = proj.with_columns([
        pl.lit(0.0).alias("target_ppg"),
        pl.lit(0.0).alias("target_games"),
        pl.lit(0.0).alias("target_total"),
    ])

    # Fill nulls
    for col in proj.columns:
        if proj[col].dtype in (pl.Float64, pl.Int64) and col not in (
            "player_id", "season", "target_ppg", "target_games", "target_total"
        ):
            proj = proj.with_columns(pl.col(col).fill_null(0.0))

    print(f"  DST projection rows: {proj.shape[0]} teams")
    return proj


def get_dst_feature_columns(df):
    """Return the list of feature column names for the DST model."""
    drop_cols = {
        "player_id", "player_display_name", "position_group", "season", "team",
        "ppg", "games_played",
        "pts_allowed_pg", "sacks_pg", "interceptions_pg",
        "fumbles_recovered_pg", "tds_pg", "safeties_pg",
        "takeaways_pg", "yards_allowed_pg",
        "target_ppg", "target_games", "target_total",
        "adjustment_ppg",
    }
    return sorted(c for c in df.columns if c not in drop_cols)
