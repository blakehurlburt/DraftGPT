"""Season-level feature pipeline for MLB fantasy points prediction.

Builds prior-season features from the Lahman database for predicting
next-season fantasy PPG and games played. Separate pipelines for
batters and pitchers since their stats and scoring differ fundamentally.
"""

import polars as pl
import numpy as np
from . import loader
from .fantasy_scoring import compute_batter_fpts, compute_pitcher_fpts

# MLB regular season game counts
MAX_GAMES_BATTER = 162
MAX_GAMES_PITCHER = 162  # appearances, not starts

# 2020 COVID-shortened season (60 games) — excluded from training and
# evaluation. Removing it entirely means 2021 prior-season features
# fall back to 2019 via the shift mechanism, which is more representative
# than a 60-game sample.
EXCLUDED_SEASONS = {2020}

# Minimum games to include a player in the dataset
_MIN_GAMES_BATTER = 50
_MIN_GAMES_PITCHER = 20


# ---------------------------------------------------------------------------
# Step 1: Aggregate raw stats and compute fantasy points
# ---------------------------------------------------------------------------

def _build_batter_seasons(seasons):
    """Load batting data and compute per-game rates + fantasy points."""
    min_year = min(seasons)
    batting = loader.load_batting(min_year=min_year)
    valid_seasons = [s for s in seasons if s not in EXCLUDED_SEASONS]
    batting = batting.filter(pl.col("yearID").is_in(valid_seasons))

    # Filter to players with meaningful playing time
    batting = batting.filter(pl.col("G") >= _MIN_GAMES_BATTER)

    # Compute derived stats
    batting = batting.with_columns([
        # Plate appearances
        (pl.col("AB") + pl.col("BB") + pl.col("HBP")
         + pl.col("SF").fill_null(0) + pl.col("SH").fill_null(0)).alias("PA"),
    ])

    # Filter out players with almost no PAs (pitchers batting)
    batting = batting.filter(pl.col("PA") >= 150)

    # Rate stats
    batting = batting.with_columns([
        # Batting average
        pl.when(pl.col("AB") > 0)
        .then(pl.col("H") / pl.col("AB"))
        .otherwise(0.0).alias("AVG"),
        # On-base percentage
        pl.when(pl.col("PA") > 0)
        .then((pl.col("H") + pl.col("BB") + pl.col("HBP")) / pl.col("PA"))
        .otherwise(0.0).alias("OBP"),
        # Slugging
        pl.when(pl.col("AB") > 0)
        .then(
            (pl.col("H") - pl.col("2B") - pl.col("3B") - pl.col("HR")  # singles
             + 2 * pl.col("2B") + 3 * pl.col("3B") + 4 * pl.col("HR"))
            / pl.col("AB")
        ).otherwise(0.0).alias("SLG"),
        # Isolated power
        pl.when(pl.col("AB") > 0)
        .then(
            (pl.col("2B") + 2 * pl.col("3B") + 3 * pl.col("HR")) / pl.col("AB")
        ).otherwise(0.0).alias("ISO"),
        # BABIP
        pl.when((pl.col("AB") - pl.col("SO") - pl.col("HR") + pl.col("SF").fill_null(0)) > 0)
        .then(
            (pl.col("H") - pl.col("HR"))
            / (pl.col("AB") - pl.col("SO") - pl.col("HR") + pl.col("SF").fill_null(0))
        ).otherwise(0.0).alias("BABIP"),
        # Strikeout rate
        pl.when(pl.col("PA") > 0)
        .then(pl.col("SO") / pl.col("PA"))
        .otherwise(0.0).alias("K_rate"),
        # Walk rate
        pl.when(pl.col("PA") > 0)
        .then(pl.col("BB") / pl.col("PA"))
        .otherwise(0.0).alias("BB_rate"),
        # Per-game rates
        (pl.col("R") / pl.col("G")).alias("r_pg"),
        (pl.col("HR") / pl.col("G")).alias("hr_pg"),
        (pl.col("RBI") / pl.col("G")).alias("rbi_pg"),
        (pl.col("SB") / pl.col("G")).alias("sb_pg"),
        (pl.col("H") / pl.col("G")).alias("h_pg"),
        (pl.col("BB") / pl.col("G")).alias("bb_pg"),
        (pl.col("AB") / pl.col("G")).alias("ab_pg"),
    ])

    # OPS
    batting = batting.with_columns(
        (pl.col("OBP") + pl.col("SLG")).alias("OPS")
    )

    # Fantasy points (season total, then per-game)
    stat_cols = ["R", "HR", "RBI", "SB", "CS", "BB", "HBP", "H", "2B", "3B", "SO", "GIDP"]

    def _fpts_expr(df):
        rows = df.select(stat_cols).fill_null(0).to_dicts()
        return [compute_batter_fpts(r) for r in rows]

    fpts = _fpts_expr(batting)
    batting = batting.with_columns(pl.Series("fantasy_points", fpts))
    batting = batting.with_columns(
        (pl.col("fantasy_points") / pl.col("G")).alias("ppg")
    )

    # Consistency: std of monthly-ish splits not available in Lahman,
    # so we use season-level variance proxy from year-to-year
    batting = batting.with_columns(
        pl.col("G").alias("games_played"),
    )

    # Rename for consistency with the pipeline
    batting = batting.rename({
        "playerID": "player_id",
        "yearID": "season",
        "teamID": "team",
    })

    return batting


def _build_pitcher_seasons(seasons):
    """Load pitching data and compute per-game rates + fantasy points."""
    min_year = min(seasons)
    pitching = loader.load_pitching(min_year=min_year)
    valid_seasons = [s for s in seasons if s not in EXCLUDED_SEASONS]
    pitching = pitching.filter(pl.col("yearID").is_in(valid_seasons))

    # Filter to meaningful contributors
    pitching = pitching.filter(pl.col("G") >= _MIN_GAMES_PITCHER)

    # Innings pitched
    pitching = pitching.with_columns(
        (pl.col("IPouts") / 3.0).alias("IP"),
    )

    # Filter out very low IP (mop-up guys)
    pitching = pitching.filter(pl.col("IP") >= 30)

    # Determine starter vs reliever from GS ratio
    pitching = pitching.with_columns(
        pl.when(pl.col("G") > 0)
        .then(pl.col("GS") / pl.col("G"))
        .otherwise(0.0).alias("gs_ratio"),
    )
    pitching = pitching.with_columns(
        pl.when(pl.col("gs_ratio") >= 0.5)
        .then(pl.lit("SP"))
        .otherwise(pl.lit("RP"))
        .alias("position_group"),
    )

    # Rate stats
    pitching = pitching.with_columns([
        # ERA already computed in loader
        # WHIP
        pl.when(pl.col("IP") > 0)
        .then((pl.col("BB") + pl.col("H")) / pl.col("IP"))
        .otherwise(None).alias("WHIP"),
        # K/9
        pl.when(pl.col("IP") > 0)
        .then(9.0 * pl.col("SO") / pl.col("IP"))
        .otherwise(0.0).alias("K9"),
        # BB/9
        pl.when(pl.col("IP") > 0)
        .then(9.0 * pl.col("BB") / pl.col("IP"))
        .otherwise(0.0).alias("BB9"),
        # HR/9
        pl.when(pl.col("IP") > 0)
        .then(9.0 * pl.col("HR") / pl.col("IP"))
        .otherwise(0.0).alias("HR9"),
        # K rate (K/BFP)
        pl.when(pl.col("BFP") > 0)
        .then(pl.col("SO") / pl.col("BFP"))
        .otherwise(0.0).alias("K_rate"),
        # BB rate (BB/BFP)
        pl.when(pl.col("BFP") > 0)
        .then(pl.col("BB") / pl.col("BFP"))
        .otherwise(0.0).alias("BB_rate"),
        # FIP: (13*HR + 3*BB - 2*K)/IP + 3.2 (constant approximation)
        pl.when(pl.col("IP") > 0)
        .then(
            (13.0 * pl.col("HR") + 3.0 * pl.col("BB") - 2.0 * pl.col("SO"))
            / pl.col("IP") + 3.2
        ).otherwise(None).alias("FIP"),
        # Per-game rates
        (pl.col("W") / pl.col("G")).alias("w_pg"),
        (pl.col("SV") / pl.col("G")).alias("sv_pg"),
        (pl.col("IP") / pl.col("G")).alias("ip_pg"),
        (pl.col("SO") / pl.col("G")).alias("so_pg"),
    ])

    # Fantasy points
    stat_cols = ["W", "L", "SV", "SO", "IPouts", "ER", "H", "BB", "HBP", "CG", "SHO"]

    def _fpts_expr(df):
        rows = df.select(stat_cols).fill_null(0).to_dicts()
        return [compute_pitcher_fpts(r) for r in rows]

    fpts = _fpts_expr(pitching)
    pitching = pitching.with_columns(pl.Series("fantasy_points", fpts))
    pitching = pitching.with_columns(
        (pl.col("fantasy_points") / pl.col("G")).alias("ppg"),
        pl.col("G").alias("games_played"),
    )

    pitching = pitching.rename({
        "playerID": "player_id",
        "yearID": "season",
        "teamID": "team",
    })

    return pitching


# ---------------------------------------------------------------------------
# Step 2: Determine batter position from Appearances
# ---------------------------------------------------------------------------

def _assign_batter_positions(df, seasons):
    """Assign primary position to batters using Appearances data."""
    appearances = loader.load_appearances(min_year=min(seasons))
    appearances = appearances.filter(pl.col("yearID").is_in(list(seasons)))

    # Position game columns to aggregate
    pos_game_raw = ["G_c", "G_1b", "G_2b", "G_3b", "G_ss", "G_lf", "G_cf", "G_rf", "G_dh"]
    available_cols = [c for c in pos_game_raw if c in appearances.columns]

    # Aggregate across stints (traded players have multiple rows per season)
    appearances = (
        appearances.group_by(["playerID", "yearID"])
        .agg([pl.col(c).fill_null(0).sum() for c in available_cols])
    )

    # Combine OF sub-positions
    appearances = appearances.with_columns(
        (pl.col("G_lf") + pl.col("G_cf") + pl.col("G_rf")).alias("G_of_total")
    )

    pos_game_cols = {
        "G_c": "C", "G_1b": "1B", "G_2b": "2B", "G_3b": "3B",
        "G_ss": "SS", "G_of_total": "OF", "G_dh": "DH",
    }

    # Find primary position per player-season (most games at a position)
    rows = []
    for row in appearances.iter_rows(named=True):
        pid = row["playerID"]
        year = row["yearID"]
        max_g = 0
        best_pos = "DH"
        for col, pos in pos_game_cols.items():
            g = row.get(col, 0) or 0
            if g > max_g:
                max_g = g
                best_pos = pos
        rows.append({"player_id": pid, "season": year, "position_group": best_pos})

    if not rows:
        df = df.with_columns(pl.lit("DH").alias("position_group"))
        return df

    pos_df = pl.DataFrame(rows).unique(subset=["player_id", "season"])
    df = df.join(pos_df, on=["player_id", "season"], how="left")
    df = df.with_columns(pl.col("position_group").fill_null("DH"))

    return df


# ---------------------------------------------------------------------------
# Step 3: Add player metadata (age, height, weight, handedness)
# ---------------------------------------------------------------------------

def _add_player_metadata(df):
    """Add biographical data from People.csv."""
    people = loader.load_people()

    meta = people.select([
        "playerID",
        "birth_date",
        pl.col("height").cast(pl.Float64, strict=False),
        pl.col("weight").cast(pl.Float64, strict=False),
        "bats",
        "throws",
        pl.col("debut").str.to_date("%Y-%m-%d", strict=False).alias("debut_date"),
    ]).rename({"playerID": "player_id"})

    df = df.join(meta, on="player_id", how="left")

    # Compute age at start of season (April 1)
    if "birth_date" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("birth_date").is_not_null())
            .then(
                ((pl.date(pl.col("season"), 4, 1) - pl.col("birth_date")).dt.total_days() / 365.25)
                .round(1)
            ).otherwise(None)
            .alias("age")
        )
        df = df.drop("birth_date")

    # Compute years of MLB experience at start of season
    if "debut_date" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("debut_date").is_not_null())
            .then(
                ((pl.date(pl.col("season"), 4, 1) - pl.col("debut_date")).dt.total_days() / 365.25)
                .round(1)
                .clip(lower_bound=0.0)
            ).otherwise(None)
            .alias("years_exp")
        )
        df = df.drop("debut_date")

    # Encode handedness as numeric
    bat_map = {"R": 0.0, "L": 1.0, "B": 2.0}
    throw_map = {"R": 0.0, "L": 1.0}
    if "bats" in df.columns:
        df = df.with_columns(
            pl.col("bats").replace_strict(bat_map, default=0.0).alias("bats_code")
        ).drop("bats")
    if "throws" in df.columns:
        df = df.with_columns(
            pl.col("throws").replace_strict(throw_map, default=0.0).alias("throws_code")
        ).drop("throws")

    return df


# ---------------------------------------------------------------------------
# Step 4: Add park factors from Teams.csv
# ---------------------------------------------------------------------------

def _add_park_factors(df, seasons):
    """Add park factor for each player's home ballpark."""
    teams = loader.load_teams(min_year=min(seasons))
    teams = teams.filter(pl.col("yearID").is_in(list(seasons)))

    park = teams.select([
        pl.col("teamID").alias("team"),
        pl.col("yearID").alias("season"),
        pl.col("BPF").cast(pl.Float64).alias("park_factor"),
    ]).unique(subset=["team", "season"])

    df = df.join(park, on=["team", "season"], how="left")
    # Normalize: BPF of 100 = neutral, >100 = hitter-friendly
    df = df.with_columns(
        (pl.col("park_factor").fill_null(100.0) / 100.0).alias("park_factor")
    )

    return df


# ---------------------------------------------------------------------------
# Step 5: Build prior-season features (lags, trends, career stats)
# ---------------------------------------------------------------------------

def _build_prior_features_batter(df):
    """Build prior-season features for batters."""
    df = df.sort(["player_id", "season"])

    # Stats to lag
    rate_stats = [
        "ppg", "AVG", "OBP", "SLG", "OPS", "ISO", "BABIP",
        "K_rate", "BB_rate",
        "r_pg", "hr_pg", "rbi_pg", "sb_pg", "h_pg", "ab_pg",
    ]

    # --- Prior 1-year features ---
    prior1_exprs = []
    for stat in rate_stats:
        if stat in df.columns:
            prior1_exprs.append(
                pl.col(stat).shift(1).over("player_id").alias(f"prior1_{stat}")
            )
    prior1_exprs.append(
        pl.col("games_played").shift(1).over("player_id").alias("prior_games_played")
    )
    df = df.with_columns(prior1_exprs)

    # --- Prior 2-year average ---
    # When only 1 prior season exists (sophomore), fall back to prior1 value
    avg2_stats = ["ppg", "AVG", "OBP", "SLG", "OPS", "ISO", "hr_pg", "rbi_pg", "sb_pg"]
    prior2_exprs = []
    for stat in avg2_stats:
        if stat in df.columns:
            prior2_exprs.append(
                pl.when(pl.col(stat).shift(2).over("player_id").is_not_null())
                .then(
                    ((pl.col(stat).shift(1) + pl.col(stat).shift(2)) / 2.0)
                    .over("player_id")
                )
                .otherwise(pl.col(stat).shift(1).over("player_id"))
                .alias(f"{stat}_2yr")
            )
    df = df.with_columns(prior2_exprs)

    # --- Trajectory (year-over-year change) ---
    # 0.0 when only 1 prior season exists (no trend measurable)
    trend_stats = ["ppg", "OPS", "hr_pg", "sb_pg", "K_rate", "BB_rate"]
    trend_exprs = []
    for stat in trend_stats:
        if stat in df.columns:
            trend_exprs.append(
                (pl.col(stat).shift(1) - pl.col(stat).shift(2))
                .over("player_id")
                .fill_null(0.0)
                .alias(f"{stat}_trend")
            )
    df = df.with_columns(trend_exprs)

    # --- Career games played rate ---
    # Cumulative games and seasons prior to current row, computed per player.
    # shift(1) gives prior-season value; we accumulate all prior values.
    df = df.with_columns(
        pl.col("games_played").shift(1).over("player_id").alias("_prior_gp")
    )
    df = df.with_columns(
        pl.col("_prior_gp").fill_null(0).cum_sum().over("player_id").alias("_cum_games")
    )
    df = df.with_columns(
        pl.col("_prior_gp").is_not_null().cast(pl.Int32).cum_sum().over("player_id").alias("_cum_seasons")
    )
    df = df.with_columns(
        pl.when(pl.col("_cum_seasons") > 0)
        .then(pl.col("_cum_games") / (pl.col("_cum_seasons") * MAX_GAMES_BATTER))
        .otherwise(None)
        .alias("career_games_rate")
    )

    # --- Best prior PPG ---
    df = df.with_columns(
        pl.col("ppg").shift(1).over("player_id").alias("_prior_ppg_for_max")
    )
    df = df.with_columns(
        pl.col("_prior_ppg_for_max").cum_max().over("player_id").alias("best_ppg")
    )

    # Clean up temp columns
    df = df.drop([c for c in df.columns if c.startswith("_")])

    return df


def _build_prior_features_pitcher(df):
    """Build prior-season features for pitchers."""
    df = df.sort(["player_id", "season"])

    rate_stats = [
        "ppg", "ERA", "WHIP", "K9", "BB9", "HR9", "FIP",
        "K_rate", "BB_rate",
        "w_pg", "sv_pg", "ip_pg", "so_pg", "gs_ratio",
    ]

    # --- Prior 1-year ---
    prior1_exprs = []
    for stat in rate_stats:
        if stat in df.columns:
            prior1_exprs.append(
                pl.col(stat).shift(1).over("player_id").alias(f"prior1_{stat}")
            )
    prior1_exprs.append(
        pl.col("games_played").shift(1).over("player_id").alias("prior_games_played")
    )
    df = df.with_columns(prior1_exprs)

    # --- Prior 2-year average ---
    # When only 1 prior season exists, fall back to prior1 value
    avg2_stats = ["ppg", "ERA", "WHIP", "K9", "FIP", "w_pg", "sv_pg"]
    prior2_exprs = []
    for stat in avg2_stats:
        if stat in df.columns:
            prior2_exprs.append(
                pl.when(pl.col(stat).shift(2).over("player_id").is_not_null())
                .then(
                    ((pl.col(stat).shift(1) + pl.col(stat).shift(2)) / 2.0)
                    .over("player_id")
                )
                .otherwise(pl.col(stat).shift(1).over("player_id"))
                .alias(f"{stat}_2yr")
            )
    df = df.with_columns(prior2_exprs)

    # --- Trajectory ---
    # 0.0 when only 1 prior season exists
    trend_stats = ["ppg", "ERA", "WHIP", "K9", "K_rate"]
    trend_exprs = []
    for stat in trend_stats:
        if stat in df.columns:
            trend_exprs.append(
                (pl.col(stat).shift(1) - pl.col(stat).shift(2))
                .over("player_id")
                .fill_null(0.0)
                .alias(f"{stat}_trend")
            )
    df = df.with_columns(trend_exprs)

    # --- Career games rate ---
    df = df.with_columns(
        pl.col("games_played").shift(1).over("player_id").alias("_prior_gp")
    )
    df = df.with_columns(
        pl.col("_prior_gp").fill_null(0).cum_sum().over("player_id").alias("_cum_games")
    )
    df = df.with_columns(
        pl.col("_prior_gp").is_not_null().cast(pl.Int32).cum_sum().over("player_id").alias("_cum_seasons")
    )
    df = df.with_columns(
        pl.when(pl.col("_cum_seasons") > 0)
        .then(pl.col("_cum_games") / (pl.col("_cum_seasons") * MAX_GAMES_PITCHER))
        .otherwise(None)
        .alias("career_games_rate")
    )

    # --- Best prior PPG ---
    df = df.with_columns(
        pl.col("ppg").shift(1).over("player_id").alias("_prior_ppg_for_max")
    )
    df = df.with_columns(
        pl.col("_prior_ppg_for_max").cum_max().over("player_id").alias("best_ppg")
    )

    df = df.drop([c for c in df.columns if c.startswith("_")])

    return df


# ---------------------------------------------------------------------------
# Step 6: Roster context features (team changes, competition)
# ---------------------------------------------------------------------------

def _add_roster_context(df, seasons):
    """Add team-change and offensive environment features."""
    df = df.sort(["player_id", "season"])

    # --- changed_team ---
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
    df = df.drop("_prior_team")

    # --- Team offensive environment (from Teams.csv, prior season) ---
    teams = loader.load_teams(min_year=min(seasons))
    teams = teams.filter(pl.col("yearID").is_in(list(seasons)))

    team_ctx = teams.select([
        pl.col("teamID").alias("team"),
        pl.col("yearID").alias("season"),
        pl.when(pl.col("G") > 0).then(pl.col("R").cast(pl.Float64) / pl.col("G")).otherwise(0.0).alias("team_r_pg"),
        pl.when(pl.col("G") > 0).then(pl.col("HR").cast(pl.Float64) / pl.col("G")).otherwise(0.0).alias("team_hr_pg"),
        pl.when(pl.col("G") > 0).then(pl.col("H").cast(pl.Float64) / pl.col("G")).otherwise(0.0).alias("team_h_pg"),
        pl.when(pl.col("G") > 0).then(pl.col("BB").cast(pl.Float64) / pl.col("G")).otherwise(0.0).alias("team_bb_pg"),
    ]).unique(subset=["team", "season"])

    # Shift to prior season
    team_ctx = team_ctx.with_columns(
        (pl.col("season") + 1).cast(pl.Int64).alias("season")
    )
    team_ctx = team_ctx.rename({
        "team_r_pg": "new_team_r_pg",
        "team_hr_pg": "new_team_hr_pg",
        "team_h_pg": "new_team_h_pg",
        "team_bb_pg": "new_team_bb_pg",
    })

    df = df.join(team_ctx, on=["team", "season"], how="left")

    # Fill nulls
    for col in ["changed_team", "new_team_r_pg", "new_team_hr_pg", "new_team_h_pg", "new_team_bb_pg"]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).fill_null(0.0))

    return df


# ---------------------------------------------------------------------------
# Step 7: Add player display name from People
# ---------------------------------------------------------------------------

def _add_display_names(df):
    """Add player_display_name from People.csv."""
    people = loader.load_people()
    names = people.select([
        pl.col("playerID").alias("player_id"),
        (pl.col("nameFirst").fill_null("") + pl.lit(" ") + pl.col("nameLast").fill_null(""))
        .str.strip_chars()
        .alias("player_display_name"),
    ]).unique(subset=["player_id"])

    df = df.join(names, on="player_id", how="left")
    return df


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def build_batter_features(seasons):
    """Build the full batter feature matrix for season-level projection.

    Args:
        seasons: Iterable of season years (e.g., range(2009, 2026)).

    Returns:
        Polars DataFrame with one row per batter-season, containing
        prior-season features and target columns.
    """
    print("=== Building MLB Batter Season Features ===")

    # Step 1: Build season-level stats
    season_df = _build_batter_seasons(seasons)
    print(f"  Batter seasons: {season_df.shape[0]} rows")

    # Step 2: Assign positions
    season_df = _assign_batter_positions(season_df, seasons)

    # Step 3: Add player metadata
    season_df = _add_player_metadata(season_df)

    # Step 4: Add park factors
    season_df = _add_park_factors(season_df, seasons)

    # Step 5: Build prior-season features
    print("Building prior-season features...")
    df = _build_prior_features_batter(season_df)

    # Step 6: Roster context
    df = _add_roster_context(df, seasons)

    # Step 7: Display names
    df = _add_display_names(df)

    # Define targets
    df = df.with_columns([
        pl.col("ppg").alias("target_ppg"),
        pl.col("games_played").alias("target_games"),
        (pl.col("ppg") * pl.col("games_played")).alias("target_total"),
    ])

    # Keep only rows with prior-season data
    df = df.filter(pl.col("prior1_ppg").is_not_null())

    # Drop first 2 years (need lookback)
    min_season = min(seasons)
    df = df.filter(pl.col("season") >= min_season + 2)

    print(f"  Final batter dataset: {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"  Seasons: {sorted(df['season'].unique().to_list())}")

    return df


def build_pitcher_features(seasons):
    """Build the full pitcher feature matrix for season-level projection.

    Args:
        seasons: Iterable of season years (e.g., range(2009, 2026)).

    Returns:
        Polars DataFrame with one row per pitcher-season.
    """
    print("=== Building MLB Pitcher Season Features ===")

    # Step 1: Build season-level stats
    season_df = _build_pitcher_seasons(seasons)
    print(f"  Pitcher seasons: {season_df.shape[0]} rows")

    # Step 3: Add player metadata
    season_df = _add_player_metadata(season_df)

    # Step 4: Add park factors
    season_df = _add_park_factors(season_df, seasons)

    # Step 5: Build prior-season features
    print("Building prior-season features...")
    df = _build_prior_features_pitcher(season_df)

    # Step 6: Roster context
    df = _add_roster_context(df, seasons)

    # Step 7: Display names
    df = _add_display_names(df)

    # Define targets
    df = df.with_columns([
        pl.col("ppg").alias("target_ppg"),
        pl.col("games_played").alias("target_games"),
        (pl.col("ppg") * pl.col("games_played")).alias("target_total"),
    ])

    # Keep only rows with prior-season data
    df = df.filter(pl.col("prior1_ppg").is_not_null())

    min_season = min(seasons)
    df = df.filter(pl.col("season") >= min_season + 2)

    print(f"  Final pitcher dataset: {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"  Seasons: {sorted(df['season'].unique().to_list())}")

    return df


def build_batter_projection_features(seasons):
    """Build projection features for the next season (one row per batter).

    Takes players from the most recent season and creates dummy rows for
    season+1, then rebuilds prior-season features so the model can project.

    Args:
        seasons: Iterable of season years for historical data.

    Returns:
        Polars DataFrame with one row per batter, ready for projection.
    """
    print("=== Building MLB Batter Projection Features ===")

    # Build full historical season data (without prior features yet)
    season_df = _build_batter_seasons(seasons)
    season_df = _assign_batter_positions(season_df, seasons)
    season_df = _add_player_metadata(season_df)
    season_df = _add_park_factors(season_df, seasons)

    max_season = season_df["season"].max()
    proj_year = max_season + 1
    print(f"  Projecting from {max_season} → {proj_year}")

    # Create dummy rows for the projection year
    players_latest = season_df.filter(pl.col("season") == max_season)
    dummy = players_latest.with_columns(pl.lit(proj_year).alias("season"))

    # Null out current-season stats (targets we don't have yet)
    id_cols = {"player_id", "season", "player_display_name", "position_group", "team"}
    stat_cols = [c for c in dummy.columns if c not in id_cols]
    dummy = dummy.with_columns(
        [pl.lit(None).cast(dummy[c].dtype).alias(c) for c in stat_cols]
    )
    # Ensure schema compatibility (lit(proj_year) is i64, season_df may have i32)
    for col in dummy.columns:
        if col in season_df.columns and dummy[col].dtype != season_df[col].dtype:
            dummy = dummy.with_columns(pl.col(col).cast(season_df[col].dtype))

    # Append dummy rows and rebuild prior features
    extended = pl.concat([season_df, dummy], how="diagonal")
    extended = _build_prior_features_batter(extended)
    extended = _add_roster_context(extended, seasons)
    extended = _add_display_names(extended)

    # Filter to projection year only
    proj = extended.filter(pl.col("season") == proj_year)
    proj = proj.filter(pl.col("prior1_ppg").is_not_null())

    # Add dummy targets for column compatibility
    proj = proj.with_columns([
        pl.lit(0.0).alias("target_ppg"),
        pl.lit(0.0).alias("target_games"),
        pl.lit(0.0).alias("target_total"),
    ])

    # Bump age and years_exp by 1 for projection year
    if "age" in proj.columns:
        proj = proj.with_columns((pl.col("age") + 1.0).alias("age"))
    if "years_exp" in proj.columns:
        proj = proj.with_columns((pl.col("years_exp") + 1.0).alias("years_exp"))

    print(f"  Projection rows: {proj.shape[0]} batters")
    return proj


def build_pitcher_projection_features(seasons):
    """Build projection features for the next season (one row per pitcher).

    Args:
        seasons: Iterable of season years for historical data.

    Returns:
        Polars DataFrame with one row per pitcher, ready for projection.
    """
    print("=== Building MLB Pitcher Projection Features ===")

    # Build full historical season data
    season_df = _build_pitcher_seasons(seasons)
    season_df = _add_player_metadata(season_df)
    season_df = _add_park_factors(season_df, seasons)

    max_season = season_df["season"].max()
    proj_year = max_season + 1
    print(f"  Projecting from {max_season} → {proj_year}")

    # Create dummy rows for the projection year
    players_latest = season_df.filter(pl.col("season") == max_season)
    dummy = players_latest.with_columns(pl.lit(proj_year).alias("season"))

    # Null out current-season stats
    id_cols = {"player_id", "season", "player_display_name", "position_group", "team"}
    stat_cols = [c for c in dummy.columns if c not in id_cols]
    dummy = dummy.with_columns(
        [pl.lit(None).cast(dummy[c].dtype).alias(c) for c in stat_cols]
    )
    # Ensure schema compatibility
    for col in dummy.columns:
        if col in season_df.columns and dummy[col].dtype != season_df[col].dtype:
            dummy = dummy.with_columns(pl.col(col).cast(season_df[col].dtype))

    # Append dummy rows and rebuild prior features
    extended = pl.concat([season_df, dummy], how="diagonal")
    extended = _build_prior_features_pitcher(extended)
    extended = _add_roster_context(extended, seasons)
    extended = _add_display_names(extended)

    # Filter to projection year only
    proj = extended.filter(pl.col("season") == proj_year)
    proj = proj.filter(pl.col("prior1_ppg").is_not_null())

    # Add dummy targets
    proj = proj.with_columns([
        pl.lit(0.0).alias("target_ppg"),
        pl.lit(0.0).alias("target_games"),
        pl.lit(0.0).alias("target_total"),
    ])

    # Bump age and years_exp by 1
    if "age" in proj.columns:
        proj = proj.with_columns((pl.col("age") + 1.0).alias("age"))
    if "years_exp" in proj.columns:
        proj = proj.with_columns((pl.col("years_exp") + 1.0).alias("years_exp"))

    print(f"  Projection rows: {proj.shape[0]} pitchers")
    return proj


def get_batter_feature_columns(df):
    """Return feature columns for the batter model."""
    drop_cols = {
        # IDs
        "player_id", "player_display_name", "position_group", "season", "team",
        "lgID",
        # Raw current-season stats (leakage)
        "G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS",
        "BB", "SO", "IBB", "HBP", "SH", "SF", "GIDP", "PA",
        "AVG", "OBP", "SLG", "OPS", "ISO", "BABIP", "K_rate", "BB_rate",
        "r_pg", "hr_pg", "rbi_pg", "sb_pg", "h_pg", "bb_pg", "ab_pg",
        "ppg", "games_played", "fantasy_points",
        # Targets
        "target_ppg", "target_games", "target_total",
    }
    return sorted(c for c in df.columns if c not in drop_cols)


def get_pitcher_feature_columns(df):
    """Return feature columns for the pitcher model."""
    drop_cols = {
        # IDs
        "player_id", "player_display_name", "position_group", "season", "team",
        "lgID",
        # Raw current-season stats (leakage)
        "W", "L", "G", "GS", "CG", "SHO", "SV", "IPouts", "H", "ER",
        "HR", "BB", "SO", "IBB", "WP", "HBP", "BK", "BFP", "GF", "R",
        "SH", "SF", "GIDP", "IP",
        "ERA", "WHIP", "K9", "BB9", "HR9", "FIP",
        "K_rate", "BB_rate", "gs_ratio",
        "w_pg", "sv_pg", "ip_pg", "so_pg",
        "ppg", "games_played", "fantasy_points",
        # Targets
        "target_ppg", "target_games", "target_total",
    }
    return sorted(c for c in df.columns if c not in drop_cols)
