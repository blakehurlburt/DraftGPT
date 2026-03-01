import polars as pl
from .loader import get_player_stats, get_player_game_log

PASSING_COLS = [
    "completions", "attempts", "passing_yards", "passing_tds",
    "passing_interceptions", "sacks_suffered", "passing_air_yards",
    "passing_yards_after_catch", "passing_first_downs", "passing_epa",
]

RUSHING_COLS = [
    "carries", "rushing_yards", "rushing_tds", "rushing_first_downs",
    "rushing_epa",
]

RECEIVING_COLS = [
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "receiving_air_yards", "receiving_yards_after_catch",
    "receiving_first_downs", "receiving_epa",
]

FANTASY_COLS = ["fantasy_points", "fantasy_points_ppr"]

ID_COLS = [
    "player_id", "player_name", "player_display_name",
    "position", "team", "season", "week", "opponent_team",
]


def player_season_summary(player_name, seasons=None):
    """Aggregate a player's stats per season with per-game averages."""
    game_log = get_player_game_log(player_name, seasons)
    if game_log.height == 0:
        return game_log

    all_stat_cols = PASSING_COLS + RUSHING_COLS + RECEIVING_COLS + FANTASY_COLS
    stat_cols = [c for c in all_stat_cols if c in game_log.columns]

    aggs = [pl.col("week").count().alias("games")]
    for col in stat_cols:
        aggs.append(pl.col(col).sum().alias(col))

    summary = game_log.group_by(["season", "player_display_name", "position", "team"]).agg(aggs)

    # Add per-game averages
    for col in stat_cols:
        summary = summary.with_columns(
            (pl.col(col) / pl.col("games")).round(1).alias(f"{col}_per_game")
        )

    return summary.sort("season")


def compare_players(player_names, seasons=None, stat_columns=None):
    """Side-by-side season totals for multiple players."""
    if stat_columns is None:
        stat_columns = PASSING_COLS + RUSHING_COLS + RECEIVING_COLS + FANTASY_COLS

    frames = []
    for name in player_names:
        summary = player_season_summary(name, seasons)
        if summary.height > 0:
            frames.append(summary)

    if not frames:
        return pl.DataFrame()

    combined = pl.concat(frames)
    available = [c for c in ["player_display_name", "season", "games"] + stat_columns
                 if c in combined.columns]
    return combined.select(available).sort(["season", "player_display_name"])


def top_performers(season, week, stat, n=10):
    """Leaderboard for a specific stat in a given week."""
    df = get_player_stats([season], weeks=[week])
    if stat not in df.columns:
        raise ValueError(f"Stat '{stat}' not found. Available: {df.columns}")

    available_id_cols = [c for c in ID_COLS if c in df.columns]
    return df.sort(stat, descending=True).head(n).select(available_id_cols + [stat])
