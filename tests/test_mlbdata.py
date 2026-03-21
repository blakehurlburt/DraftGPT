"""Tests for the mlbdata package.

Tests cover fantasy scoring, data loading, and feature engineering.
"""

import pytest
import polars as pl
from pathlib import Path

from mlbdata.fantasy_scoring import (
    compute_batter_fpts, compute_pitcher_fpts,
    BATTER_POINTS, PITCHER_POINTS,
)
from mlbdata.loader import normalize_team_code, TEAM_CODE_MAP


# ---------------------------------------------------------------------------
# Fantasy scoring tests
# ---------------------------------------------------------------------------

class TestBatterFantasyScoring:
    def test_basic_scoring(self):
        row = {"R": 1, "HR": 1, "RBI": 1, "H": 1}
        # R=1, HR=4, RBI=1, H=1 = 7
        assert compute_batter_fpts(row) == 7.0

    def test_empty_row(self):
        assert compute_batter_fpts({}) == 0.0

    def test_stolen_bases(self):
        row = {"SB": 5, "CS": 2}
        # 5*2 + 2*-1 = 8
        assert compute_batter_fpts(row) == 8.0

    def test_negative_stats(self):
        row = {"SO": 100, "GIDP": 20}
        # 100*-0.5 + 20*-0.5 = -60
        assert compute_batter_fpts(row) == -60.0

    def test_extra_base_hits(self):
        # A double: H=1 + 2B=1 = 2 pts
        row = {"H": 1, "2B": 1}
        assert compute_batter_fpts(row) == 2.0
        # A triple: H=1 + 3B=2 = 3 pts
        row = {"H": 1, "3B": 1}
        assert compute_batter_fpts(row) == 3.0

    def test_custom_scoring(self):
        row = {"HR": 1}
        custom = {"HR": 10}
        assert compute_batter_fpts(row, scoring=custom) == 10.0

    def test_full_season_example(self):
        """Aaron Judge 2022-style line: 133R, 62HR, 131RBI, 16SB, 3CS, 111BB, 175SO, 78H (non-HR singles approx)."""
        row = {
            "R": 133, "HR": 62, "RBI": 131, "SB": 16, "CS": 3,
            "BB": 111, "HBP": 6, "H": 175, "2B": 28, "3B": 0,
            "SO": 175, "GIDP": 10,
        }
        pts = compute_batter_fpts(row)
        assert pts > 0
        # Should be a big positive number for an MVP season
        assert pts > 500


class TestPitcherFantasyScoring:
    def test_basic_scoring(self):
        row = {"W": 1, "SO": 10, "IPouts": 21}  # 7 IP
        # W=5, SO=10, IPouts=21 = 36
        assert compute_pitcher_fpts(row) == 36.0

    def test_empty_row(self):
        assert compute_pitcher_fpts({}) == 0.0

    def test_loss_penalty(self):
        row = {"L": 1}
        assert compute_pitcher_fpts(row) == -3.0

    def test_save_bonus(self):
        row = {"SV": 1}
        assert compute_pitcher_fpts(row) == 5.0

    def test_earned_runs_penalty(self):
        row = {"ER": 5}
        assert compute_pitcher_fpts(row) == -10.0

    def test_complete_game_shutout(self):
        row = {"CG": 1, "SHO": 1}
        # CG=3, SHO=3 = 6
        assert compute_pitcher_fpts(row) == 6.0

    def test_hits_walks_penalty(self):
        row = {"H": 10, "BB": 5, "HBP": 2}
        # H=10*-0.5=-5, BB=5*-1=-5, HBP=2*-0.5=-1 = -11
        assert compute_pitcher_fpts(row) == -11.0


# ---------------------------------------------------------------------------
# Team code normalization tests
# ---------------------------------------------------------------------------

class TestTeamCodes:
    def test_historical_to_modern(self):
        assert normalize_team_code("CHN") == "CHC"
        assert normalize_team_code("CHA") == "CWS"
        assert normalize_team_code("SFN") == "SF"
        assert normalize_team_code("NYN") == "NYM"
        assert normalize_team_code("NYA") == "NYY"
        assert normalize_team_code("LAN") == "LAD"
        assert normalize_team_code("SDN") == "SD"
        assert normalize_team_code("SLN") == "STL"
        assert normalize_team_code("KCA") == "KC"
        assert normalize_team_code("TBA") == "TB"
        assert normalize_team_code("FLO") == "MIA"
        assert normalize_team_code("MON") == "WSN"
        assert normalize_team_code("ANA") == "LAA"
        assert normalize_team_code("CAL") == "LAA"

    def test_modern_codes_passthrough(self):
        for code in ["NYY", "NYM", "LAD", "SF", "CHC", "CWS", "STL", "KC", "TB", "SD"]:
            assert normalize_team_code(code) == code

    def test_unknown_code_passthrough(self):
        assert normalize_team_code("XYZ") == "XYZ"


# ---------------------------------------------------------------------------
# Loader tests (require Lahman data)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data" / "lahman_1871-2025_csv"
ZIP_PATH = Path(__file__).parent.parent / "data" / "lahman_1871-2025_csv.zip"
lahman_available = pytest.mark.skipif(
    not DATA_DIR.exists() and not ZIP_PATH.exists(),
    reason="Lahman data not available",
)


@lahman_available
class TestLoader:
    def test_load_batting(self):
        from mlbdata.loader import load_batting
        df = load_batting(min_year=2024)
        assert isinstance(df, pl.DataFrame)
        assert "playerID" in df.columns
        assert "yearID" in df.columns
        assert "HR" in df.columns
        assert df.shape[0] > 0
        # Should have normalized team codes
        teams = df["teamID"].unique().to_list()
        assert "CHN" not in teams  # should be CHC

    def test_load_pitching(self):
        from mlbdata.loader import load_pitching
        df = load_pitching(min_year=2024)
        assert isinstance(df, pl.DataFrame)
        assert "ERA" in df.columns
        assert df.shape[0] > 0

    def test_load_people(self):
        from mlbdata.loader import load_people
        df = load_people()
        assert isinstance(df, pl.DataFrame)
        assert "playerID" in df.columns
        assert "birth_date" in df.columns
        assert "debut" in df.columns

    def test_load_appearances(self):
        from mlbdata.loader import load_appearances
        df = load_appearances(min_year=2024)
        assert isinstance(df, pl.DataFrame)
        assert "G_c" in df.columns

    def test_load_teams(self):
        from mlbdata.loader import load_teams
        df = load_teams(min_year=2024)
        assert isinstance(df, pl.DataFrame)
        assert "BPF" in df.columns

    def test_stint_aggregation_batting(self):
        """Traded players should have one row per season after aggregation."""
        from mlbdata.loader import load_batting
        df = load_batting(min_year=2010)
        dupes = df.group_by(["playerID", "yearID"]).len().filter(pl.col("len") > 1)
        assert dupes.shape[0] == 0, f"Found {dupes.shape[0]} duplicate player-seasons"

    def test_stint_aggregation_pitching(self):
        from mlbdata.loader import load_pitching
        df = load_pitching(min_year=2010)
        dupes = df.group_by(["playerID", "yearID"]).len().filter(pl.col("len") > 1)
        assert dupes.shape[0] == 0


# ---------------------------------------------------------------------------
# Season features tests (require Lahman data)
# ---------------------------------------------------------------------------

@lahman_available
class TestSeasonFeatures:
    @pytest.fixture(scope="class")
    def batter_df(self):
        from mlbdata.season_features import build_batter_features
        return build_batter_features(range(2022, 2026))

    @pytest.fixture(scope="class")
    def pitcher_df(self):
        from mlbdata.season_features import build_pitcher_features
        return build_pitcher_features(range(2022, 2026))

    def test_batter_shape(self, batter_df):
        assert batter_df.shape[0] > 0
        assert batter_df.shape[1] > 30

    def test_pitcher_shape(self, pitcher_df):
        assert pitcher_df.shape[0] > 0
        assert pitcher_df.shape[1] > 30

    def test_batter_required_columns(self, batter_df):
        required = [
            "player_id", "season", "team", "position_group",
            "target_ppg", "target_games", "target_total",
            "prior1_ppg", "prior_games_played", "age", "years_exp",
            "career_games_rate", "best_ppg", "park_factor",
        ]
        for col in required:
            assert col in batter_df.columns, f"Missing column: {col}"

    def test_pitcher_required_columns(self, pitcher_df):
        required = [
            "player_id", "season", "team", "position_group",
            "target_ppg", "target_games", "target_total",
            "prior1_ppg", "prior_games_played", "age", "years_exp",
            "career_games_rate", "best_ppg",
            "prior1_ERA", "prior1_WHIP", "prior1_K9", "prior1_FIP",
        ]
        for col in required:
            assert col in pitcher_df.columns, f"Missing column: {col}"

    def test_no_duplicate_player_seasons(self, batter_df):
        dupes = batter_df.group_by(["player_id", "season"]).len().filter(pl.col("len") > 1)
        assert dupes.shape[0] == 0

    def test_2020_excluded(self, batter_df):
        seasons = batter_df["season"].unique().to_list()
        assert 2020 not in seasons

    def test_prior1_ppg_not_null(self, batter_df):
        nulls = batter_df.filter(pl.col("prior1_ppg").is_null()).shape[0]
        assert nulls == 0

    def test_career_games_rate_not_null(self, batter_df):
        nulls = batter_df.filter(pl.col("career_games_rate").is_null()).shape[0]
        assert nulls == 0

    def test_batter_positions_valid(self, batter_df):
        valid = {"C", "1B", "2B", "3B", "SS", "OF", "DH"}
        positions = set(batter_df["position_group"].unique().to_list())
        assert positions.issubset(valid), f"Unexpected positions: {positions - valid}"

    def test_pitcher_positions_valid(self, pitcher_df):
        valid = {"SP", "RP"}
        positions = set(pitcher_df["position_group"].unique().to_list())
        assert positions.issubset(valid)

    def test_feature_columns_exclude_targets(self, batter_df):
        from mlbdata.season_features import get_batter_feature_columns
        feat_cols = get_batter_feature_columns(batter_df)
        assert "target_ppg" not in feat_cols
        assert "target_games" not in feat_cols
        assert "player_id" not in feat_cols
        assert "season" not in feat_cols
        assert "ppg" not in feat_cols  # raw stat, not prior

    def test_feature_columns_include_priors(self, batter_df):
        from mlbdata.season_features import get_batter_feature_columns
        feat_cols = get_batter_feature_columns(batter_df)
        assert "prior1_ppg" in feat_cols
        assert "age" in feat_cols
        assert "years_exp" in feat_cols
        assert "career_games_rate" in feat_cols


@lahman_available
class TestProjectionFeatures:
    @pytest.fixture(scope="class")
    def batter_proj(self):
        from mlbdata.season_features import build_batter_projection_features
        return build_batter_projection_features(range(2022, 2026))

    @pytest.fixture(scope="class")
    def pitcher_proj(self):
        from mlbdata.season_features import build_pitcher_projection_features
        return build_pitcher_projection_features(range(2022, 2026))

    def test_one_row_per_batter(self, batter_proj):
        dupes = batter_proj.group_by("player_id").len().filter(pl.col("len") > 1)
        assert dupes.shape[0] == 0, f"Found {dupes.shape[0]} duplicate players"

    def test_one_row_per_pitcher(self, pitcher_proj):
        dupes = pitcher_proj.group_by("player_id").len().filter(pl.col("len") > 1)
        assert dupes.shape[0] == 0

    def test_projection_season(self, batter_proj):
        seasons = batter_proj["season"].unique().to_list()
        assert len(seasons) == 1
        assert seasons[0] == 2026

    def test_has_prior_features(self, batter_proj):
        assert "prior1_ppg" in batter_proj.columns
        nulls = batter_proj.filter(pl.col("prior1_ppg").is_null()).shape[0]
        assert nulls == 0

    def test_has_dummy_targets(self, batter_proj):
        assert "target_ppg" in batter_proj.columns
        assert "target_games" in batter_proj.columns


# ---------------------------------------------------------------------------
# MiLB feature formula tests
# ---------------------------------------------------------------------------

class TestMiLBFormulas:
    def test_fip_includes_hbp(self):
        """FIP formula should include HBP: (13*HR + 3*(BB+HBP) - 2*K)/IP + C."""
        from mlbdata.milb_features import _parse_milb_splits
        splits = [{
            "season": "2023",
            "_sport_id": None,
            "_level": "AAA",
            "stat": {
                "gamesPlayed": 20,
                "inningsPitched": "90.0",
                "outs": 270,
                "wins": 5, "losses": 3, "saves": 0,
                "strikeOuts": 80, "baseOnBalls": 20,
                "hits": 70, "earnedRuns": 30,
                "homeRuns": 8, "hitByPitch": 10,
                "gamesStarted": 15, "completeGames": 0, "shutouts": 0,
            }
        }]
        rows = _parse_milb_splits(splits, before_year=2025, group="pitching")
        assert len(rows) == 1
        fip = rows[0]["FIP"]
        # Manual: (13*8 + 3*(20+10) - 2*80) / 90 + 3.2
        #       = (104 + 90 - 160) / 90 + 3.2 = 34/90 + 3.2 ≈ 3.578
        expected = (13*8 + 3*(20+10) - 2*80) / 90.0 + 3.2
        assert abs(fip - expected) < 0.01

    def test_pa_includes_sf_sh(self):
        """PA fallback should include SF and SH."""
        from mlbdata.milb_features import _parse_milb_splits
        splits = [{
            "season": "2023",
            "_sport_id": None,
            "_level": "AAA",
            "stat": {
                "gamesPlayed": 50,
                "atBats": 180,
                # No plateAppearances — triggers fallback
                "baseOnBalls": 20,
                "hitByPitch": 5,
                "sacFlies": 4,
                "sacBunts": 3,
                "hits": 50, "homeRuns": 10, "doubles": 8, "triples": 2,
                "strikeOuts": 40, "stolenBases": 5, "caughtStealing": 2,
                "rbi": 30, "runs": 25,
            }
        }]
        rows = _parse_milb_splits(splits, before_year=2025, group="hitting")
        assert len(rows) == 1
        # PA fallback = AB + BB + HBP + SF + SH = 180 + 20 + 5 + 4 + 3 = 212
        assert rows[0]["PA"] == 212
