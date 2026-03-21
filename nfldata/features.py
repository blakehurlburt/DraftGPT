"""
Feature engineering pipeline for fantasy points prediction.

Merges nflverse data (player stats, schedules, rosters, snap counts)
into a single player-game-level dataframe with rolling and contextual
features suitable for ML modeling.
"""

import polars as pl
import nflreadpy as nfl


# Stats to compute rolling windows for
ROLLING_STATS = [
    "fantasy_points_ppr",
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "targets",
    "carries",
    "receptions",
    "passing_epa",
    "rushing_epa",
    "receiving_epa",
    "offense_pct",
    "target_share",
    "wopr",
]

# Columns that identify a row (not features)
ID_COLS = [
    "player_id",
    "player_name",
    "player_display_name",
    "game_id",
    "season",
    "week",
    "team",
    "opponent_team",
    "position",
    "position_group",
]


def _load_and_merge(seasons):
    """Load all raw data sources and merge into one player-game dataframe."""
    season_list = list(seasons)

    # --- Base table: player stats ---
    print("Loading player stats...")
    stats = nfl.load_player_stats(season_list)

    # --- Schedules ---
    print("Loading schedules...")
    scheds = nfl.load_schedules(season_list)
    sched_cols = [
        "game_id", "home_team", "away_team",
        "spread_line", "total_line", "roof", "surface",
        "temp", "wind", "home_rest", "away_rest",
    ]
    # Keep only columns that exist
    available = [c for c in sched_cols if c in scheds.columns]
    scheds = scheds.select(available).unique(subset=["game_id"])

    stats = stats.join(scheds, on="game_id", how="left")

    # --- Rosters (for player metadata) ---
    print("Loading rosters...")
    rosters = nfl.load_rosters(season_list)
    roster_cols_map = {
        "gsis_id": "player_id",  # rename for join
    }
    roster_keep = ["gsis_id", "season", "years_exp", "weight", "height", "draft_number"]
    available_roster = [c for c in roster_keep if c in rosters.columns]
    rosters = rosters.select(available_roster).unique(subset=["gsis_id", "season"])
    rosters = rosters.rename({"gsis_id": "player_id"})

    # Ensure height is numeric (already in inches in recent nflreadpy)
    if "height" in rosters.columns and rosters["height"].dtype != pl.Float64:
        rosters = rosters.with_columns(pl.col("height").cast(pl.Float64))

    stats = stats.join(rosters, on=["player_id", "season"], how="left")

    # --- Snap counts (via players table bridge) ---
    print("Loading snap counts...")
    try:
        snaps = nfl.load_snap_counts(season_list)
        players_table = nfl.load_players()

        # Build gsis_id -> pfr_id mapping
        id_map_cols = ["gsis_id", "pfr_id"]
        available_id = [c for c in id_map_cols if c in players_table.columns]
        if len(available_id) == 2:
            id_map = (
                players_table.select(available_id)
                .drop_nulls()
                .unique(subset=["gsis_id"])
            )

            # Add gsis_id to snap counts via pfr_id
            pfr_col = "pfr_player_id" if "pfr_player_id" in snaps.columns else "pfr_id"
            snaps = snaps.join(
                id_map, left_on=pfr_col, right_on="pfr_id", how="inner"
            )

            # Keep relevant snap columns
            snap_keep = ["gsis_id", "game_id", "offense_pct"]
            available_snap = [c for c in snap_keep if c in snaps.columns]
            if "offense_pct" in snaps.columns:
                snaps_deduped = (
                    snaps.select(available_snap)
                    .unique(subset=["gsis_id", "game_id"])
                )
                stats = stats.join(
                    snaps_deduped,
                    left_on=["player_id", "game_id"],
                    right_on=["gsis_id", "game_id"],
                    how="left",
                )
            else:
                print("  Warning: offense_pct not found in snap counts, skipping.")
        else:
            print("  Warning: Could not build ID mapping, skipping snap counts.")
    except Exception as e:
        print(f"  Warning: Failed to load snap counts: {e}")

    return stats


def _add_game_context(df):
    """Derive game-context columns from schedule data."""
    exprs = []

    # is_home
    if "home_team" in df.columns:
        exprs.append(
            (pl.col("team") == pl.col("home_team")).alias("is_home")
        )

    # rest_days
    if "home_rest" in df.columns and "away_rest" in df.columns:
        exprs.append(
            pl.when(pl.col("team") == pl.col("home_team"))
            .then(pl.col("home_rest"))
            .otherwise(pl.col("away_rest"))
            .alias("rest_days")
        )

    # is_dome
    if "roof" in df.columns:
        exprs.append(
            pl.col("roof").is_in(["dome", "closed"]).alias("is_dome")
        )

    # opponent_spread (positive = player's team is favored)
    # spread_line is typically from home team perspective (negative = home favored)
    if "spread_line" in df.columns and "home_team" in df.columns:
        exprs.append(
            pl.when(pl.col("team") == pl.col("home_team"))
            .then(-pl.col("spread_line"))  # home team: negate (spread_line negative = home favored -> positive)
            .otherwise(pl.col("spread_line"))
            .alias("opponent_spread")
        )

    # game_total
    if "total_line" in df.columns:
        exprs.append(pl.col("total_line").alias("game_total"))

    if exprs:
        df = df.with_columns(exprs)

    return df


def _compute_rolling_features(df):
    """Compute rolling averages, trends, and consistency per player."""
    # Sort for rolling computations
    df = df.sort(["player_id", "season", "week"])

    # Fill nulls in rolling stat columns with 0 for computation
    fill_cols = [c for c in ROLLING_STATS if c in df.columns]
    df = df.with_columns([pl.col(c).fill_null(0.0).cast(pl.Float64) for c in fill_cols])

    # We need to compute rolling stats that don't cross season boundaries.
    # Use a season_week rank within each player-season group for window calcs.
    # Polars rolling_mean with group_by handles this if we partition by (player_id, season).

    rolling_exprs = []
    for stat in fill_cols:
        col = pl.col(stat)

        # 3-week rolling average (shift by 1 to exclude current week = lag)
        rolling_exprs.append(
            col.shift(1)
            .rolling_mean(window_size=3, min_periods=1)
            .over("player_id", "season")
            .alias(f"{stat}_roll_3")
        )

        # 8-week rolling average
        rolling_exprs.append(
            col.shift(1)
            .rolling_mean(window_size=8, min_periods=2)
            .over("player_id", "season")
            .alias(f"{stat}_roll_8")
        )

    df = df.with_columns(rolling_exprs)

    # Trend: 3-week avg minus 8-week avg (momentum)
    trend_exprs = []
    for stat in fill_cols:
        r3 = f"{stat}_roll_3"
        r8 = f"{stat}_roll_8"
        # CR opus: "or True" makes this condition always true, rendering the
        # check meaningless. Should just remove the condition entirely.
        if r3 in df.columns or True:  # we just created them
            trend_exprs.append(
                (pl.col(r3) - pl.col(r8)).alias(f"{stat}_trend")
            )
    df = df.with_columns(trend_exprs)

    # Consistency: 5-week rolling std of fantasy_points_ppr
    if "fantasy_points_ppr" in df.columns:
        df = df.with_columns(
            pl.col("fantasy_points_ppr")
            .shift(1)
            .rolling_std(window_size=5, min_periods=2)
            .over("player_id", "season")
            .alias("fpp_std_5")
        )

    return df


def _compute_opponent_features(df):
    """Compute opponent-adjusted features: avg fantasy points allowed by position."""
    if "opponent_team" not in df.columns or "position_group" not in df.columns:
        return df

    # For each game, compute the opponent defense's season-to-date avg fantasy points
    # allowed to the player's position group.
    # We need: for opponent_team X in season S, week W, what's the average
    # fantasy_points_ppr they've allowed to position_group P in weeks < W?

    # First compute per-game points allowed by defense to each position group
    # This is: for each (recent_team_of_opponent, season, week, position_group),
    # sum up fantasy points scored against them.
    # "opponent_team" in the main df is the team the player is playing AGAINST.
    # So points scored by player with opponent_team=X means X allowed those points.

    opp_allowed = (
        df.group_by(["opponent_team", "season", "week", "position_group"])
        .agg(pl.col("fantasy_points_ppr").sum().alias("pts_allowed"))
        .sort(["opponent_team", "season", "week"])
    )

    # Compute expanding mean of pts_allowed up to (but not including) current week
    opp_allowed = opp_allowed.with_columns(
        pl.col("pts_allowed")
        .shift(1)
        .rolling_mean(window_size=20, min_periods=1)
        .over("opponent_team", "season", "position_group")
        .alias("opp_fppg_allowed")
    )

    opp_lookup = opp_allowed.select(
        ["opponent_team", "season", "week", "position_group", "opp_fppg_allowed"]
    )

    df = df.join(
        opp_lookup,
        on=["opponent_team", "season", "week", "position_group"],
        how="left",
    )

    return df


def build_features(seasons):
    """Build the full feature matrix for fantasy points prediction.

    Args:
        seasons: Iterable of season years (e.g., range(2018, 2025)).

    Returns:
        Polars DataFrame with features and target variable, ready for modeling.
    """
    print("=== Building Feature Matrix ===")

    # Step 1: Load & merge
    df = _load_and_merge(seasons)
    print(f"  After merge: {df.shape[0]:,} rows, {df.shape[1]} cols")

    # Filter to offensive positions only
    if "position_group" in df.columns:
        df = df.filter(pl.col("position_group").is_in(["QB", "RB", "WR", "TE"]))
    elif "position" in df.columns:
        # CR opus: "K" is included in the position fallback filter but NOT in the
        # position_group filter above. This is inconsistent — kickers will only be
        # included when position_group is absent. Intentional?
        df = df.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE", "K"]))
    print(f"  After position filter: {df.shape[0]:,} rows")

    # Step 2: Game context
    df = _add_game_context(df)

    # Step 3: Rolling features
    print("Computing rolling features...")
    df = _compute_rolling_features(df)

    # Step 4: Opponent-adjusted features
    print("Computing opponent features...")
    df = _compute_opponent_features(df)

    # CR opus: This drops ALL week-1 rows globally, but rolling features are
    # partitioned by (player_id, season). A player in their 2nd+ season still has
    # no rolling history in week 1 of the new season (correct to drop), but consider
    # that week 2 also has very thin history (1 game for roll_3, roll_8). The
    # min_periods=1 in rolling_mean will produce a "rolling avg" from just 1 data
    # point, which may be noisier than desired.
    # Step 5: Clean up - drop week 1 of each season (no rolling history)
    df = df.filter(pl.col("week") > 1)

    # Drop rows with null target
    df = df.filter(pl.col("fantasy_points_ppr").is_not_null())

    # Convert booleans to int for model compatibility
    bool_cols = [c for c in df.columns if df[c].dtype == pl.Boolean]
    if bool_cols:
        df = df.with_columns([pl.col(c).cast(pl.Int8) for c in bool_cols])

    print(f"  Final dataset: {df.shape[0]:,} rows, {df.shape[1]} cols")

    # Identify feature columns
    drop_cols = {
        "player_id", "player_name", "player_display_name", "game_id",
        "home_team", "away_team", "roof", "surface",
        "home_rest", "away_rest", "spread_line", "total_line",
        "headshot_url",
    }
    # CR opus: raw_stat_cols is computed but never used — the raw current-week
    # stat columns (passing_yards, rushing_yards, etc.) are NOT removed from the
    # returned DataFrame. get_feature_columns() handles exclusion downstream, but
    # this dead code suggests the intent was to drop them here too.
    # Also drop raw stat columns that have rolling versions
    raw_stat_cols = set(ROLLING_STATS)

    feature_and_id = [c for c in df.columns if c not in drop_cols]
    print(f"  Feature + ID columns: {len(feature_and_id)}")

    return df


def get_feature_columns(df):
    """Return the list of feature column names from a features dataframe."""
    drop_cols = {
        "player_id", "player_name", "player_display_name", "game_id",
        "season", "week", "team", "opponent_team",
        "position", "position_group",
        "home_team", "away_team", "roof", "surface",
        "home_rest", "away_rest", "spread_line", "total_line",
        "headshot_url", "fantasy_points_ppr", "fantasy_points",
        # Raw stats that have rolling versions — keep them out of features
        # to avoid leakage (they are current-week values)
        "completions", "attempts", "passing_yards", "passing_tds",
        "passing_interceptions", "sacks_suffered", "passing_air_yards",
        "passing_yards_after_catch", "passing_first_downs", "passing_epa",
        "carries", "rushing_yards", "rushing_tds", "rushing_first_downs",
        "rushing_epa", "receptions", "targets", "receiving_yards",
        "receiving_tds", "receiving_air_yards", "receiving_yards_after_catch",
        "receiving_first_downs", "receiving_epa",
        "target_share", "wopr", "offense_pct",
        "racr", "air_yards_share", "tgt_sh", "wopr_x", "ay_sh",
        "special_teams_tds", "pacr",
        "passing_2pt_conversions", "rushing_2pt_conversions",
        "receiving_2pt_conversions", "receiving_fumbles",
        "receiving_fumbles_lost", "rushing_fumbles", "rushing_fumbles_lost",
        "sack_yards", "sack_fumbles", "sack_fumbles_lost",
        "dakota",
    }
    # Actually, let's be more precise: keep rolling/derived features, drop raw stats
    feature_cols = []
    for c in df.columns:
        if c in drop_cols:
            continue
        # Keep rolling features, trends, std
        if any(c.endswith(s) for s in ["_roll_3", "_roll_8", "_trend", "_std_5"]):
            feature_cols.append(c)
            continue
        # Keep game context features
        if c in [
            "is_home", "rest_days", "is_dome", "opponent_spread",
            "game_total", "temp", "wind",
            "years_exp", "weight", "height", "draft_number",
            "opp_fppg_allowed",
        ]:
            feature_cols.append(c)
            continue

    return feature_cols
