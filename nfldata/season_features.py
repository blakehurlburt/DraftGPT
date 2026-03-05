"""
Season-level feature pipeline for fantasy points prediction.

Aggregates weekly player stats to season level, then builds prior-season
features (lagged stats, trends, injury history, player metadata) for
predicting next-season PPG and games played.
"""

import polars as pl
import nflreadpy as nfl

# Maximum regular-season games (17 since 2021, 16 before)
MAX_GAMES = {s: 17 if s >= 2021 else 16 for s in range(2012, 2030)}

# ---------------------------------------------------------------------------
# Injury classification system
#
# Based on sports medicine research on NFL injuries. Each injury is assigned:
#   severity  : 1 (minor) to 5 (season-ending / catastrophic)
#   perf_hit  : expected next-season performance decline (0.0 = none, 1.0 = total)
#   reinjury  : probability of recurrence (0.0 – 1.0)
#
# The raw `report_primary_injury` strings from the nflverse injury report
# are mapped to canonical types, then looked up in this table.
# ---------------------------------------------------------------------------

# (severity, perf_hit, reinjury_risk)
INJURY_PROFILES = {
    # --- Season-ending / catastrophic (severity 5) ---
    "achilles":       (5, 0.40, 0.10),  # 22-88% decline by position, low re-tear
    "patellar":       (5, 0.35, 0.08),  # 55% RTP rate, severe performance loss

    # --- Severe (severity 4) ---
    "acl":            (4, 0.35, 0.25),  # 33-50% decline, 25% reinjury
    "lisfranc":       (4, 0.21, 0.15),  # 21% decline at 1 year
    "fibula":         (4, 0.15, 0.10),  # lower-leg fracture, extended recovery
    "tibia":          (4, 0.15, 0.10),

    # --- Significant (severity 3) ---
    "high_ankle":     (3, 0.17, 0.15),  # 17% initial decline, lingers
    "back":           (3, 0.10, 0.12),  # recoverable if surgical
    "pectoral":       (3, 0.12, 0.08),  # often season-ending for linemen
    "biceps":         (3, 0.10, 0.08),
    "triceps":        (3, 0.10, 0.08),
    "shoulder":       (3, 0.12, 0.15),  # labrum/rotator cuff, position-dependent
    "collarbone":     (3, 0.08, 0.05),

    # --- Moderate (severity 2) ---
    "hamstring":      (2, 0.05, 0.38),  # minimal per-incident but 38% recurrence!
    "knee":           (2, 0.08, 0.15),  # MCL/meniscus (non-ACL)
    "concussion":     (2, 0.07, 0.15),  # cumulative risk, 7% compensation decline
    "ankle":          (2, 0.05, 0.12),  # low ankle sprain (high ankle caught separately)
    "foot":           (2, 0.05, 0.10),  # general foot (Lisfranc caught separately)
    "hip":            (2, 0.05, 0.10),
    "quadricep":      (2, 0.05, 0.18),
    "elbow":          (2, 0.05, 0.08),
    "ribs":           (2, 0.03, 0.05),
    "neck":           (2, 0.08, 0.10),

    # --- Minor (severity 1) ---
    "calf":           (1, 0.02, 0.25),  # no measurable decline but 19-31% recurrence
    "groin":          (1, 0.02, 0.18),  # 18% recurrence
    "toe":            (1, 0.02, 0.10),
    "hand":           (1, 0.01, 0.05),
    "wrist":          (1, 0.01, 0.05),
    "finger":         (1, 0.01, 0.05),
    "thumb":          (1, 0.01, 0.05),
    "thigh":          (1, 0.03, 0.15),
    "shin":           (1, 0.02, 0.08),
    "heel":           (1, 0.02, 0.08),
    "oblique":        (1, 0.02, 0.12),
    "abdomen":        (1, 0.02, 0.08),
    "chest":          (1, 0.02, 0.05),
    "forearm":        (1, 0.01, 0.05),
    "glute":          (1, 0.02, 0.10),

    # --- Non-injuries (severity 0) ---
    "illness":        (0, 0.00, 0.00),
    "rest":           (0, 0.00, 0.00),
    "personal":       (0, 0.00, 0.00),
    "not_injury":     (0, 0.00, 0.00),
}

# Map raw injury strings (lowercased) to canonical types
# Order matters: first match wins, so check specific strings before general ones
_INJURY_KEYWORD_MAP = [
    # Non-injuries first
    ("not injury", "not_injury"),
    ("illness", "illness"),
    ("personal", "personal"),
    ("resting", "rest"),
    ("coach", "rest"),
    ("covid", "illness"),
    ("suspension", "not_injury"),
    ("inactive", "not_injury"),
    # Specific before general
    ("achilles", "achilles"),
    ("patellar", "patellar"),
    ("lisfranc", "lisfranc"),
    ("acl", "acl"),
    ("pectoral", "pectoral"),
    ("bicep", "biceps"),
    ("tricep", "triceps"),
    ("collarbone", "collarbone"),
    ("fibula", "fibula"),
    ("tibia", "tibia"),
    ("concussion", "concussion"),
    ("hamstring", "hamstring"),
    ("quadricep", "quadricep"),
    ("oblique", "oblique"),
    ("abdomen", "abdomen"),
    ("groin", "groin"),
    ("adductor", "groin"),
    ("calf", "calf"),
    ("shoulder", "shoulder"),
    ("knee", "knee"),
    ("ankle", "ankle"),
    ("hip", "hip"),
    ("back", "back"),
    ("spine", "back"),
    ("neck", "neck"),
    ("foot", "foot"),
    ("toe", "toe"),
    ("heel", "heel"),
    ("shin", "shin"),
    ("hand", "hand"),
    ("wrist", "wrist"),
    ("finger", "finger"),
    ("thumb", "thumb"),
    ("chest", "chest"),
    ("rib", "ribs"),
    ("elbow", "elbow"),
    ("forearm", "forearm"),
    ("thigh", "thigh"),
    ("glute", "glute"),
    ("head", "concussion"),
    ("stinger", "neck"),
]


def _classify_injury(raw: str) -> str:
    """Map a raw injury string to a canonical injury type."""
    if raw is None:
        return "unknown"
    lower = raw.lower().strip()
    if not lower or lower == "--":
        return "unknown"
    for keyword, canonical in _INJURY_KEYWORD_MAP:
        if keyword in lower:
            return canonical
    return "unknown"


def _injury_severity(canonical: str) -> int:
    """Return severity tier (0-5) for a canonical injury type."""
    return INJURY_PROFILES.get(canonical, (1, 0.05, 0.10))[0]


def _injury_perf_hit(canonical: str) -> float:
    """Return expected performance decline (0-1) for a canonical injury type."""
    return INJURY_PROFILES.get(canonical, (1, 0.05, 0.10))[1]


def _injury_reinjury_risk(canonical: str) -> float:
    """Return reinjury probability (0-1) for a canonical injury type."""
    return INJURY_PROFILES.get(canonical, (1, 0.05, 0.10))[2]


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
        pl.col("team").last().alias("team"),  # end-of-season team assignment
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
    """Step 3: Add injury history features from prior season.

    Classifies each injury by type and computes severity-weighted features
    based on sports medicine research:
      - severity tiers (0=non-injury, 1=minor, 2=moderate, 3=significant,
        4=severe, 5=season-ending/catastrophic)
      - expected performance decline priors
      - reinjury risk priors
    """
    season_list = list(seasons)
    _INJ_COLS = [
        "inj_weeks_on_report", "inj_times_out", "inj_times_questionable",
        "inj_distinct_injuries",
        "inj_max_severity", "inj_weighted_severity", "inj_perf_risk",
        "inj_reinjury_risk", "inj_has_major", "inj_has_moderate",
        "inj_has_minor",
    ]
    print("Loading injuries...")
    try:
        inj = nfl.load_injuries(season_list)
    except Exception as e:
        print(f"  Warning: Failed to load injuries: {e}")
        for col in _INJ_COLS:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
        return df

    # Rename gsis_id to player_id for joining
    if "gsis_id" in inj.columns:
        inj = inj.rename({"gsis_id": "player_id"})

    # Classify each injury row
    inj = inj.with_columns(
        pl.col("report_primary_injury")
        .map_elements(_classify_injury, return_dtype=pl.Utf8)
        .alias("injury_type")
    )
    inj = inj.with_columns([
        pl.col("injury_type")
        .map_elements(_injury_severity, return_dtype=pl.Int64)
        .alias("severity"),
        pl.col("injury_type")
        .map_elements(_injury_perf_hit, return_dtype=pl.Float64)
        .alias("perf_hit"),
        pl.col("injury_type")
        .map_elements(_injury_reinjury_risk, return_dtype=pl.Float64)
        .alias("reinjury_prob"),
    ])

    # Filter out non-injuries (illness, rest, personal) for severity features
    real_injuries = inj.filter(pl.col("severity") > 0)

    # Aggregate per player-season
    # Basic counts (from ALL injury report rows including non-injuries)
    basic_agg = (
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

    # Severity-weighted features (from real injuries only)
    severity_agg = (
        real_injuries.group_by(["player_id", "season"])
        .agg([
            # Worst injury suffered that season
            pl.col("severity").max().alias("inj_max_severity"),
            # Weighted severity = sum of (severity * times_reported) / total reports
            # Captures both severity and persistence
            pl.col("severity").mean().alias("inj_weighted_severity"),
            # Max expected performance hit from any single injury type
            pl.col("perf_hit").max().alias("inj_perf_risk"),
            # Max reinjury probability from any injury type
            pl.col("reinjury_prob").max().alias("inj_reinjury_risk"),
            # Binary flags for injury tiers
            (pl.col("severity") >= 4).any().cast(pl.Float64).alias("inj_has_major"),
            (pl.col("severity").is_between(2, 3)).any().cast(pl.Float64).alias("inj_has_moderate"),
            (pl.col("severity") == 1).any().cast(pl.Float64).alias("inj_has_minor"),
        ])
    )

    # Join basic and severity aggregations
    inj_agg = basic_agg.join(severity_agg, on=["player_id", "season"], how="left")

    # These are prior-season injury features, so shift by joining on season+1
    inj_agg = inj_agg.with_columns(
        (pl.col("season") + 1).cast(pl.Int32).alias("target_season")
    )
    inj_agg = inj_agg.drop("season").rename({"target_season": "season"})

    # Cast to float for consistency
    for col in _INJ_COLS:
        if col in inj_agg.columns:
            inj_agg = inj_agg.with_columns(pl.col(col).cast(pl.Float64))

    df = df.join(inj_agg, on=["player_id", "season"], how="left")

    # Fill severity nulls for players who were on the injury report but had
    # no "real" injuries (only illness/rest/personal). These players have
    # inj_weeks_on_report populated but severity features null.
    severity_cols = [
        "inj_max_severity", "inj_has_major", "inj_has_moderate",
        "inj_has_minor",
    ]
    for col in severity_cols:
        if col in df.columns:
            df = df.with_columns(
                pl.when(pl.col("inj_weeks_on_report").is_not_null() & pl.col(col).is_null())
                .then(0.0)
                .otherwise(pl.col(col))
                .alias(col)
            )

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

    # Ensure numeric columns are properly typed (older seasons may have strings)
    if "height" in rosters.columns and rosters["height"].dtype != pl.Float64:
        rosters = rosters.with_columns(pl.col("height").cast(pl.Float64, strict=False))
    if "draft_number" in rosters.columns and rosters["draft_number"].dtype != pl.Float64:
        rosters = rosters.with_columns(pl.col("draft_number").cast(pl.Float64, strict=False))

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


def _add_roster_context_features(df, seasons):
    """Step 5: Add roster-change and team-context features.

    Captures team changes, offensive environment, positional competition
    shifts, and QB changes to help the model account for roster turnover.
    """
    season_list = list(seasons)
    print("Building roster context features...")

    # --- 5a: changed_team (binary: player's team differs from prior season) ---
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

    # --- 5b: Team offensive context (from team stats, prior season) ---
    try:
        ts = nfl.load_team_stats(season_list)
        # Filter to regular season
        if "season_type" in ts.columns:
            ts = ts.filter(pl.col("season_type") == "REG")

        # Aggregate to per-team-season means
        team_off = (
            ts.group_by(["team", "season"])
            .agg([
                pl.col("passing_yards").mean().alias("team_pass_ypg"),
                pl.col("rushing_yards").mean().alias("team_rush_ypg"),
                pl.col("receptions").mean().alias("team_rec_pg"),
                (pl.col("passing_tds") + pl.col("rushing_tds")).mean().alias("team_td_pg"),
            ])
        )

        # These are PRIOR-season team stats, so shift season by +1
        # Then join on the player's CURRENT team — this means:
        # - For players who stayed: their team's prior-year volume
        # - For players who changed: their NEW team's prior-year volume
        team_off = team_off.with_columns(
            (pl.col("season") + 1).cast(pl.Int32).alias("season")
        )
        team_off = team_off.rename({
            "team_pass_ypg": "new_team_pass_ypg",
            "team_rush_ypg": "new_team_rush_ypg",
            "team_rec_pg": "new_team_rec_pg",
            "team_td_pg": "new_team_td_pg",
        })

        df = df.join(team_off, on=["team", "season"], how="left")
        print("  Added team offensive context features")
    except Exception as e:
        print(f"  Warning: Failed to load team stats: {e}")
        for col in ["new_team_pass_ypg", "new_team_rush_ypg", "new_team_rec_pg",
                     "new_team_td_pg"]:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    # --- 5c: Positional competition features ---
    # Compare rosters at each position between consecutive seasons
    # to find departed and arriving players
    roster_by_team = (
        df.select(["player_id", "team", "season", "position_group", "ppg"])
        .filter(pl.col("ppg").is_not_null())
    )

    # For each team-season-position, build the set of player_ids
    # Departures: players at position P on team T in season S-1, not on team T in season S
    # Arrivals: players at position P on team T in season S, not on team T in season S-1
    pos_opp_rows = []
    for season in sorted(df["season"].unique().to_list()):
        prev_season = season - 1
        curr = roster_by_team.filter(pl.col("season") == season)
        prev = roster_by_team.filter(pl.col("season") == prev_season)

        if prev.height == 0 or curr.height == 0:
            continue

        for team in curr["team"].unique().to_list():
            if team is None:
                continue
            for pos in ["QB", "RB", "WR", "TE"]:
                # Players at this position on this team last year
                prev_players = set(
                    prev.filter(
                        (pl.col("team") == team) & (pl.col("position_group") == pos)
                    )["player_id"].to_list()
                )
                # Players at this position on this team this year
                curr_players = set(
                    curr.filter(
                        (pl.col("team") == team) & (pl.col("position_group") == pos)
                    )["player_id"].to_list()
                )

                departed = prev_players - curr_players
                arrived = curr_players - prev_players

                # Sum PPG of departed players (from their prior-season stats)
                vacated = 0.0
                if departed:
                    dep_ppg = prev.filter(
                        pl.col("player_id").is_in(list(departed))
                        & (pl.col("team") == team)
                        & (pl.col("position_group") == pos)
                    )["ppg"].to_list()
                    vacated = sum(dep_ppg)

                # Sum prior-season PPG of arriving players (from wherever they were)
                added = 0.0
                if arrived:
                    arr_ppg = prev.filter(
                        pl.col("player_id").is_in(list(arrived))
                    )["ppg"].to_list()
                    added = sum(arr_ppg)

                pos_opp_rows.append({
                    "team": team,
                    "season": season,
                    "position_group": pos,
                    "pos_vacated_pts": round(vacated, 2),
                    "pos_added_pts": round(added, 2),
                })

    if pos_opp_rows:
        pos_opp_df = pl.DataFrame(pos_opp_rows)
        # Clamp to 0 — vacated/added points can't be negative
        pos_opp_df = pos_opp_df.with_columns([
            pl.col("pos_vacated_pts").clip(lower_bound=0.0),
            pl.col("pos_added_pts").clip(lower_bound=0.0),
        ])
        pos_opp_df = pos_opp_df.with_columns(
            (pl.col("pos_vacated_pts") - pl.col("pos_added_pts")).alias("pos_net_opportunity")
        )
        df = df.join(pos_opp_df, on=["team", "season", "position_group"], how="left")
        print("  Added positional competition features")
    else:
        for col in ["pos_vacated_pts", "pos_added_pts", "pos_net_opportunity"]:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    # --- 5d: QB change features (for WR/TE) ---
    # Identify each team's primary QB per season (most games)
    qb_data = (
        roster_by_team.filter(pl.col("position_group") == "QB")
        .group_by(["team", "season"])
        .agg([
            pl.col("player_id").first().alias("primary_qb"),  # first after sort = most games (from agg)
            pl.col("ppg").first().alias("primary_qb_ppg"),
        ])
    )

    # Build QB change detection: compare primary QB between consecutive seasons
    qb_change_rows = []
    for team in qb_data["team"].unique().to_list():
        if team is None:
            continue
        team_qbs = qb_data.filter(pl.col("team") == team).sort("season")
        seasons_list = team_qbs["season"].to_list()
        qb_ids = team_qbs["primary_qb"].to_list()

        for i in range(1, len(seasons_list)):
            changed = 1.0 if qb_ids[i] != qb_ids[i - 1] else 0.0
            # Get the new QB's prior-season passing ypg
            new_qb_id = qb_ids[i]
            new_qb_prior = roster_by_team.filter(
                (pl.col("player_id") == new_qb_id)
                & (pl.col("season") == seasons_list[i] - 1)
            )
            new_qb_pass_ypg = 0.0
            if new_qb_prior.height > 0:
                # Use pass_ypg from the season-level data if available
                # Fall back to ppg as a proxy
                new_qb_pass_ypg = new_qb_prior["ppg"].to_list()[0] or 0.0

            qb_change_rows.append({
                "team": team,
                "season": seasons_list[i],
                "qb_changed": changed,
                "new_qb_prior_ppg": round(new_qb_pass_ypg, 2),
            })

    if qb_change_rows:
        qb_change_df = pl.DataFrame(qb_change_rows)
        df = df.join(qb_change_df, on=["team", "season"], how="left")
        # Only apply QB features to WR/TE; zero out for QB/RB
        df = df.with_columns([
            pl.when(pl.col("position_group").is_in(["WR", "TE"]))
            .then(pl.col("qb_changed"))
            .otherwise(0.0)
            .alias("qb_changed"),
            pl.when(pl.col("position_group").is_in(["WR", "TE"]))
            .then(pl.col("new_qb_prior_ppg"))
            .otherwise(0.0)
            .alias("new_qb_prior_ppg"),
        ])
        print("  Added QB change features")
    else:
        df = df.with_columns([
            pl.lit(0.0).alias("qb_changed"),
            pl.lit(0.0).alias("new_qb_prior_ppg"),
        ])

    # Clean up temp columns
    df = df.drop([c for c in df.columns if c.startswith("_")])

    # Fill nulls in roster context features
    roster_feat_cols = [
        "changed_team", "new_team_pass_ypg", "new_team_rush_ypg",
        "new_team_rec_pg", "new_team_td_pg", "pos_vacated_pts",
        "pos_added_pts", "pos_net_opportunity", "qb_changed",
        "new_qb_prior_ppg",
    ]
    for col in roster_feat_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).fill_null(0.0))

    return df


def _load_starter_set(seasons):
    """Load depth charts to identify starters (depth_team='1').

    Returns a set of (player_id, season) tuples for players who appeared
    as a first-team starter at any point during the season.
    """
    season_list = list(seasons)
    print("Loading depth charts...")
    try:
        dc = nfl.load_depth_charts(season_list)
    except Exception as e:
        print(f"  Warning: Failed to load depth charts: {e}")
        return set()

    # Filter to offensive skill positions, first team
    starters = dc.filter(
        (pl.col("depth_team") == "1")
        & pl.col("position").is_in(["QB", "RB", "WR", "TE"])
    )

    # Build set of (gsis_id, season) — player was a starter at any point
    starter_pairs = set()
    if "gsis_id" in starters.columns:
        pairs = (
            starters.select(["gsis_id", "season"])
            .unique()
            .rename({"gsis_id": "player_id"})
        )
        for row in pairs.iter_rows():
            starter_pairs.add((row[0], row[1]))

    print(f"  Identified {len(starter_pairs)} player-season starter entries")
    return starter_pairs


# Minimum games to keep a non-starter in the dataset
_MIN_GAMES_NON_STARTER = 6


def build_season_features(seasons):
    """Build the full season-level feature matrix.

    Args:
        seasons: Iterable of season years (e.g., range(2018, 2025)).

    Returns:
        Polars DataFrame with one row per player-season, containing
        prior-season features and target columns. Only includes seasons
        where prior-season data exists (drops rookies and first appearances).

    Filters out fringe/depth players to improve games-played predictions.
    A player-season is kept if EITHER:
      - The player was listed as a depth_team='1' starter that season, OR
      - The player played 6+ games that season (meaningful contributor)
    """
    print("=== Building Season-Level Features ===")

    # Step 1: Aggregate to season level
    season_df = _aggregate_to_season(seasons)

    # Step 1b: Filter to starters and meaningful contributors
    starter_set = _load_starter_set(seasons)
    if starter_set:
        before = season_df.shape[0]
        # Tag each row as starter or not
        season_df = season_df.with_columns(
            pl.struct(["player_id", "season"])
            .map_elements(
                lambda row: (row["player_id"], row["season"]) in starter_set,
                return_dtype=pl.Boolean,
            )
            .alias("_is_starter")
        )
        # Keep: starters OR players with 6+ games
        season_df = season_df.filter(
            pl.col("_is_starter") | (pl.col("games_played") >= _MIN_GAMES_NON_STARTER)
        )
        season_df = season_df.drop("_is_starter")
        print(f"  After starter/contributor filter: {season_df.shape[0]} rows (was {before})")

    # Step 2: Build prior-season features
    print("Building prior-season features...")
    df = _build_prior_features(season_df)

    # Step 3: Add injury features
    df = _add_injury_features(df, seasons)

    # Step 4: Add player metadata
    df = _add_player_metadata(df, seasons)

    # Step 5: Add roster context features
    df = _add_roster_context_features(df, seasons)

    # Step 6: Define targets
    df = df.with_columns([
        pl.col("ppg").alias("target_ppg"),
        pl.col("games_played").alias("target_games"),
        (pl.col("ppg") * pl.col("games_played")).alias("target_total"),
    ])

    # Drop rows without prior-season data (first appearance for each player)
    df = df.filter(pl.col("prior1_ppg").is_not_null())

    # Drop seasons before 2017 as targets (need 2 prior seasons for lookback)
    df = df.filter(pl.col("season") >= 2017)

    # Define feature columns (everything that's not an ID, target, or raw current-season stat)
    print(f"  Final dataset: {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"  Seasons with targets: {sorted(df['season'].unique().to_list())}")

    return df


def get_season_feature_columns(df):
    """Return the list of feature column names for the season model."""
    drop_cols = {
        # IDs and non-features
        "player_id", "player_display_name", "position_group", "season", "team",
        "adjustment_ppg",
        # Current-season raw stats (these are targets or leakage)
        "ppg", "pass_ypg", "rush_ypg", "rec_ypg", "tgt_pg", "carries_pg",
        "rec_pg", "pass_td_pg", "rush_td_pg", "rec_td_pg",
        "pass_epa_pg", "rush_epa_pg", "rec_epa_pg",
        "target_share_avg", "wopr_avg", "ppg_std", "games_played",
        # Targets
        "target_ppg", "target_games", "target_total",
        # --- Pruned features (correlation analysis, 2025-03) ---
        # Near-perfect redundancy (r > 0.95 with a retained feature)
        "prior_games_missed",       # complement of prior_games_played (r=-0.992)
        "prior1_pass_ypg",          # redundant with pass_ypg_2yr (r=0.989)
        "prior1_tgt_pg",            # redundant with prior1_target_share_avg (r=0.983)
        "prior1_carries_pg",        # redundant with rush_ypg_2yr (r=0.979)
        "prior1_rec_pg",            # redundant with prior1_target_share_avg (r=0.960)
        "prior1_wopr_avg",          # redundant with prior1_target_share_avg (r=0.976)
        "prior1_rec_ypg",           # redundant with rec_ypg_2yr (r=0.959)
        "prior1_rush_ypg",          # redundant with rush_ypg_2yr (r=0.969)
        "age",                      # redundant with years_exp (r=0.960)
        # Injury cluster redundancy
        "inj_weighted_severity",    # redundant with inj_max_severity (r=0.895)
        "inj_perf_risk",            # redundant with inj_max_severity (r=0.870)
        # Near-zero target signal
        "pos_net_opportunity",      # target r=0.002; linear combo of kept features
        "inj_has_moderate",         # target r=-0.005
        "inj_times_out",            # target r=0.008
        "inj_reinjury_risk",        # target r=0.018
        "new_team_rush_ypg",        # target r=0.011
        # Borderline redundancy
        "best_ppg",                 # redundant with ppg_2yr (r=0.928)
        # Data quality
        "inj_has_major",            # only 16 positives out of 2928 rows
    }
    return [c for c in df.columns if c not in drop_cols]
