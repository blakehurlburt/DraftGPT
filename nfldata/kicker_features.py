"""
Kicker feature pipeline for fantasy points prediction.

Aggregates weekly kicker stats to season level, computes fantasy points
manually (nflreadpy reports zero for kickers), and builds prior-season
features for predicting next-season PPG and games played.
"""

import polars as pl
import nflreadpy as nfl


def _compute_kicker_fpts(row):
    """Compute kicker fantasy points for a single week.

    Scoring: FG 0-39yd=3, 40-49yd=4, 50+yd=5, PAT made=1,
             FG miss=-1, PAT miss=-1.
    """
    # Field goals by distance bucket
    fg_0_19 = row.get("fg_made_0_19", 0) or 0
    fg_20_29 = row.get("fg_made_20_29", 0) or 0
    fg_30_39 = row.get("fg_made_30_39", 0) or 0
    fg_40_49 = row.get("fg_made_40_49", 0) or 0
    fg_50_59 = row.get("fg_made_50_59", 0) or 0
    fg_60_plus = row.get("fg_made_60_", 0) or 0

    fg_att = row.get("fg_att", 0) or 0
    fg_made = row.get("fg_made", 0) or 0
    fg_missed = fg_att - fg_made

    pat_made = row.get("pat_made", 0) or 0
    pat_att = row.get("pat_att", 0) or 0
    pat_missed = pat_att - pat_made

    pts = (
        (fg_0_19 + fg_20_29 + fg_30_39) * 3
        + fg_40_49 * 4
        + (fg_50_59 + fg_60_plus) * 5
        + pat_made * 1
        + fg_missed * -1
        + pat_missed * -1
    )
    return float(pts)


def _aggregate_kicker_to_season(seasons):
    """Load kicker stats and aggregate to one row per kicker-season."""
    season_list = list(seasons)
    print("Loading kicker stats...")
    stats = nfl.load_player_stats(season_list)

    # Filter to kickers, regular season, weeks 1-17
    stats = stats.filter(pl.col("position") == "K")
    if "season_type" in stats.columns:
        stats = stats.filter(pl.col("season_type") == "REG")
    # CR opus: Hardcoded week <= 17 excludes week 18 of the regular season, which
    # has existed since 2021 (17-game schedule). This drops ~1 game per kicker per
    # season for 2021+, understating games_played and biasing ppg calculations.
    stats = stats.filter(pl.col("week") <= 17)

    # Fill nulls in kicking columns
    kick_cols = [
        "fg_att", "fg_made", "fg_missed",
        "fg_made_0_19", "fg_made_20_29", "fg_made_30_39",
        "fg_made_40_49", "fg_made_50_59", "fg_made_60_",
        "fg_long",
        "pat_att", "pat_made",
    ]
    available = [c for c in kick_cols if c in stats.columns]
    stats = stats.with_columns([pl.col(c).fill_null(0).cast(pl.Float64) for c in available])

    # Compute weekly fantasy points
    stats = stats.with_columns(
        pl.struct(available)
        .map_elements(_compute_kicker_fpts, return_dtype=pl.Float64)
        .alias("kicker_fpts")
    )

    # Aggregate to season level
    season_df = (
        stats.group_by(["player_id", "season"])
        .agg([
            pl.col("player_display_name").first().alias("player_display_name"),
            pl.lit("K").alias("position_group"),
            # CR opus: pl.col("team").last() depends on row ordering within the
            # group_by, but group_by does not guarantee order. Should sort by week
            # before aggregating, or use a different approach to get end-of-season team.
            pl.col("team").last().alias("team"),
            pl.len().alias("games_played"),
            pl.col("kicker_fpts").mean().alias("ppg"),
            pl.col("fg_att").mean().alias("fg_att_pg"),
            pl.col("fg_made").mean().alias("fg_made_pg"),
            pl.col("pat_att").mean().alias("pat_att_pg"),
            pl.col("pat_made").mean().alias("pat_made_pg"),
            pl.col("fg_long").max().alias("fg_long"),
            # FG percentage
            pl.col("fg_made").sum().alias("_fg_made_total"),
            pl.col("fg_att").sum().alias("_fg_att_total"),
            # FG 40+ percentage
            (pl.col("fg_made_40_49").sum() + pl.col("fg_made_50_59").sum()
             + pl.col("fg_made_60_").sum()).alias("_fg40_made"),
            # PAT percentage
            pl.col("pat_made").sum().alias("_pat_made_total"),
            pl.col("pat_att").sum().alias("_pat_att_total"),
        ])
    )

    # CR opus: _fg40_made is accumulated above but never used — no FG 40+ accuracy
    # rate is computed from it. Consider either computing a fg40_pct feature or
    # removing the dead aggregation.
    # Compute percentages
    season_df = season_df.with_columns([
        (pl.col("_fg_made_total") / pl.col("_fg_att_total").clip(lower_bound=1)).alias("fg_pct"),
        (pl.col("_pat_made_total") / pl.col("_pat_att_total").clip(lower_bound=1)).alias("pat_pct"),
    ])

    # Drop temp columns
    season_df = season_df.drop([c for c in season_df.columns if c.startswith("_")])
    season_df = season_df.sort(["player_id", "season"])

    print(f"  Kicker season-level: {season_df.shape[0]} kicker-seasons")
    return season_df


def _build_kicker_prior_features(df):
    """Build prior-season features for kickers using lag/lookback."""
    df = df.sort(["player_id", "season"])

    pg_stats = ["ppg", "fg_att_pg", "fg_made_pg", "pat_att_pg", "pat_made_pg"]
    rate_stats = ["fg_pct", "pat_pct"]

    # Prior 1-year features
    prior1_exprs = []
    for stat in pg_stats + rate_stats:
        prior1_exprs.append(
            pl.col(stat).shift(1).over("player_id").alias(f"prior1_{stat}")
        )
    prior1_exprs.append(
        pl.col("games_played").shift(1).over("player_id").alias("prior_games_played")
    )
    prior1_exprs.append(
        pl.col("fg_long").shift(1).over("player_id").alias("prior1_fg_long")
    )
    df = df.with_columns(prior1_exprs)

    # Prior 2-year average PPG
    df = df.with_columns(
        ((pl.col("ppg").shift(1) + pl.col("ppg").shift(2)) / 2.0)
        .over("player_id")
        .alias("ppg_2yr")
    )

    # PPG trend (year-over-year change)
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


def _add_kicker_team_context(df, seasons):
    """Add team offensive context features that drive kicker volume."""
    season_list = list(seasons)
    print("Loading team stats for kicker context...")
    try:
        ts = nfl.load_team_stats(season_list)
        if "season_type" in ts.columns:
            ts = ts.filter(pl.col("season_type") == "REG")

        team_ctx = (
            ts.group_by(["team", "season"])
            .agg([
                # Points per game drives PAT/FG volume
                # CR opus: (passing_tds + rushing_tds) * 7 is a rough approximation of
            # team points that ignores FGs, PATs, 2-pt conversions, safeties, and
            # defensive/special teams TDs. This consistently underestimates team
            # scoring, which is the key driver of kicker volume.
            ((pl.col("passing_tds") + pl.col("rushing_tds")) * 7).mean().alias("team_pts_pg"),
            ])
        )

        # Prior season team context (shift season +1)
        team_ctx = team_ctx.with_columns(
            (pl.col("season") + 1).cast(pl.Int32).alias("season")
        )

        df = df.join(team_ctx, on=["team", "season"], how="left")
        print("  Added kicker team context")
    except Exception as e:
        print(f"  Warning: Failed to load team stats: {e}")
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("team_pts_pg"))

    # Changed team
    df = df.sort(["player_id", "season"])
    df = df.with_columns(
        pl.col("team").shift(1).over("player_id").alias("_prior_team")
    )
    df = df.with_columns(
        pl.when(pl.col("_prior_team").is_null())
        .then(0.0)
        .when(pl.col("team") != pl.col("_prior_team"))
        .then(1.0)
        .otherwise(0.0)
        .alias("changed_team")
    )
    df = df.drop([c for c in df.columns if c.startswith("_")])
    return df


def _add_kicker_metadata(df, seasons):
    """Add age and years_exp from rosters."""
    season_list = list(seasons)
    print("Loading rosters for kicker metadata...")
    rosters = nfl.load_rosters(season_list)

    keep_cols = ["gsis_id", "season", "years_exp", "birth_date"]
    available = [c for c in keep_cols if c in rosters.columns]
    rosters = rosters.select(available).unique(subset=["gsis_id", "season"])
    rosters = rosters.rename({"gsis_id": "player_id"})

    if "birth_date" in rosters.columns:
        rosters = rosters.with_columns(
            ((pl.date(pl.col("season"), 9, 1) - pl.col("birth_date")).dt.total_days() / 365.25)
            .round(1)
            .alias("age")
        )
        rosters = rosters.drop("birth_date")

    df = df.join(rosters, on=["player_id", "season"], how="left")
    return df


def build_kicker_features(seasons):
    """Build the full kicker feature matrix for training.

    Returns:
        Polars DataFrame with one row per kicker-season, containing
        prior-season features and target columns.
    """
    print("=== Building Kicker Features ===")

    season_df = _aggregate_kicker_to_season(seasons)
    df = _build_kicker_prior_features(season_df)
    df = _add_kicker_team_context(df, seasons)
    df = _add_kicker_metadata(df, seasons)

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

    print(f"  Final kicker dataset: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def build_kicker_projection_features(seasons, rosters=None):
    """Build feature rows for next-season kicker projection.

    Creates one row per kicker using their most recent season as prior.
    """
    print("Building kicker projection features...")
    season_df = _aggregate_kicker_to_season(seasons)
    max_season = season_df["season"].max()
    print(f"  Projecting kickers from {max_season} season data")

    # Create dummy rows for projection year
    players_latest = season_df.filter(pl.col("season") == max_season)
    dummy = players_latest.with_columns(pl.lit(max_season + 1).alias("season"))

    id_cols = {"player_id", "season", "player_display_name", "position_group", "team"}
    stat_cols = [c for c in dummy.columns if c not in id_cols]
    dummy = dummy.with_columns([pl.lit(None).cast(dummy[c].dtype).alias(c) for c in stat_cols])

    # Override team from rosters if available
    if rosters is not None and "current_team" in rosters.columns:
        dummy = dummy.drop("team").join(
            rosters.select(["player_id", pl.col("current_team").alias("team")]),
            on="player_id", how="left",
        )
        fallback = players_latest.select(["player_id", pl.col("team").alias("_fb_team")])
        dummy = dummy.join(fallback, on="player_id", how="left")
        dummy = dummy.with_columns(
            pl.col("team").fill_null(pl.col("_fb_team"))
        ).drop("_fb_team")

    extended = pl.concat([season_df, dummy], how="diagonal")
    extended = extended.sort(["player_id", "season"])

    extended = _build_kicker_prior_features(extended)
    extended = _add_kicker_team_context(extended, seasons)
    extended = _add_kicker_metadata(extended, seasons)

    proj_year = max_season + 1
    proj = extended.filter(pl.col("season") == proj_year)
    proj = proj.filter(pl.col("prior1_ppg").is_not_null())

    # Fill metadata from prior season
    meta_cols = ["years_exp", "age"]
    available_meta = [c for c in meta_cols if c in proj.columns]
    missing_meta = proj.select("player_id").join(
        extended.filter(pl.col("season") == max_season)
        .select(["player_id"] + available_meta)
        .unique(subset=["player_id"]),
        on="player_id", how="left",
    )
    if "years_exp" in missing_meta.columns:
        missing_meta = missing_meta.with_columns((pl.col("years_exp") + 1).alias("years_exp"))
    if "age" in missing_meta.columns:
        missing_meta = missing_meta.with_columns((pl.col("age") + 1.0).alias("age"))
    proj = proj.drop(available_meta).join(missing_meta, on="player_id", how="left")

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

    print(f"  Kicker projection rows: {proj.shape[0]} players")
    return proj


def get_kicker_feature_columns(df):
    """Return the list of feature column names for the kicker model."""
    drop_cols = {
        "player_id", "player_display_name", "position_group", "season", "team",
        "ppg", "fg_att_pg", "fg_made_pg", "pat_att_pg", "pat_made_pg",
        "fg_pct", "pat_pct", "fg_long", "games_played",
        "target_ppg", "target_games", "target_total",
        "adjustment_ppg",
    }
    return sorted(c for c in df.columns if c not in drop_cols)
