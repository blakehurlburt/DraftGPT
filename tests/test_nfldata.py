"""Tests for the nfldata package.

Tests that require nflreadpy (which depends on network access and a separate
venv) are skipped when the dependency is unavailable. Data file tests always run.
"""

import pytest
from pathlib import Path

try:
    import nflreadpy
    HAS_NFLREADPY = True
except ImportError:
    HAS_NFLREADPY = False

nflreadpy_required = pytest.mark.skipif(
    not HAS_NFLREADPY, reason="nflreadpy not installed"
)


# ---------------------------------------------------------------------------
# Import / API surface tests (require nflreadpy)
# ---------------------------------------------------------------------------

@nflreadpy_required
class TestImports:
    def test_package_imports(self):
        import nfldata
        assert hasattr(nfldata, "get_player_stats")
        assert hasattr(nfldata, "get_player_game_log")
        assert hasattr(nfldata, "player_season_summary")
        assert hasattr(nfldata, "compare_players")
        assert hasattr(nfldata, "top_performers")

    def test_re_exports_nflreadpy(self):
        import nfldata
        assert hasattr(nfldata, "load_player_stats")
        assert hasattr(nfldata, "load_rosters")
        assert hasattr(nfldata, "load_schedules")
        assert hasattr(nfldata, "load_snap_counts")
        assert hasattr(nfldata, "get_current_season")

    def test_cache_module(self):
        from nfldata.cache import configure_cache, clear_cache
        assert callable(configure_cache)
        assert callable(clear_cache)

    def test_loader_module(self):
        from nfldata.loader import get_player_stats, get_player_game_log
        assert callable(get_player_stats)
        assert callable(get_player_game_log)

    def test_stats_module(self):
        from nfldata.stats import (
            player_season_summary, compare_players, top_performers,
            PASSING_COLS, RUSHING_COLS, RECEIVING_COLS, FANTASY_COLS,
        )
        assert callable(player_season_summary)
        assert isinstance(PASSING_COLS, list)
        assert "passing_yards" in PASSING_COLS
        assert "rushing_yards" in RUSHING_COLS
        assert "receiving_yards" in RECEIVING_COLS
        assert "fantasy_points_ppr" in FANTASY_COLS

    def test_features_module(self):
        from nfldata.features import build_features, get_feature_columns
        assert callable(build_features)
        assert callable(get_feature_columns)

    def test_model_module(self):
        from nfldata.model import train_model
        assert callable(train_model)

    def test_season_features_module(self):
        from nfldata.season_features import build_season_features
        assert callable(build_season_features)

    def test_season_model_module(self):
        from nfldata.season_model import walk_forward_eval
        assert callable(walk_forward_eval)


# ---------------------------------------------------------------------------
# Stats column constants (require nflreadpy for import)
# ---------------------------------------------------------------------------

@nflreadpy_required
class TestStatColumns:
    def test_passing_cols_not_empty(self):
        from nfldata.stats import PASSING_COLS
        assert len(PASSING_COLS) > 5

    def test_no_duplicate_cols(self):
        from nfldata.stats import PASSING_COLS, RUSHING_COLS, RECEIVING_COLS, FANTASY_COLS
        all_cols = PASSING_COLS + RUSHING_COLS + RECEIVING_COLS + FANTASY_COLS
        assert len(all_cols) == len(set(all_cols))


# ---------------------------------------------------------------------------
# Cache configuration (requires nflreadpy)
# ---------------------------------------------------------------------------

@nflreadpy_required
class TestCache:
    def test_default_cache_dir_exists(self):
        from nfldata.cache import DEFAULT_CACHE_DIR
        assert DEFAULT_CACHE_DIR.name == ".cache"

    def test_configure_cache_creates_dir(self, tmp_path):
        from nfldata.cache import configure_cache
        cache_dir = tmp_path / "test_cache"
        configure_cache(cache_dir=cache_dir)
        assert cache_dir.exists()


# ---------------------------------------------------------------------------
# Data files (always run — no nflreadpy needed)
# ---------------------------------------------------------------------------

class TestDataFiles:
    def test_projections_exist(self):
        proj = Path("data/projections/all_projections.csv")
        assert proj.exists(), f"Missing {proj}"

    def test_projections_has_required_columns(self):
        import csv
        with open("data/projections/all_projections.csv") as f:
            reader = csv.DictReader(f)
            headers = set(reader.fieldnames)
        required = {
            "player_display_name", "position_group", "projected_ppg",
            "projected_games", "projected_total", "pos_rank",
        }
        assert required.issubset(headers), f"Missing columns: {required - headers}"

    def test_per_position_csvs_exist(self):
        for pos in ["qb", "rb", "wr", "te"]:
            path = Path(f"data/projections/{pos}_projections.csv")
            assert path.exists(), f"Missing {path}"

    def test_rosters_csv_exists(self):
        path = Path("data/rosters.csv")
        assert path.exists(), f"Missing {path}"

    def test_projections_has_players(self):
        import csv
        with open("data/projections/all_projections.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 100, f"Expected >100 players, got {len(rows)}"
        positions = {r["position_group"] for r in rows}
        assert positions == {"QB", "RB", "WR", "TE", "K", "DST"}

    def test_projections_include_2026_rookies(self):
        """Key 2026 rookies must appear in projections."""
        import csv
        with open("data/projections/all_projections.csv") as f:
            rows = list(csv.DictReader(f))
        names = {r["player_display_name"] for r in rows}
        expected_rookies = [
            "Jeremiyah Love",
            "Makai Lemon",
            "Carnell Tate",
            "Jordyn Tyson",
        ]
        for rookie in expected_rookies:
            assert rookie in names, f"2026 rookie {rookie} missing from projections"

    @nflreadpy_required
    def test_no_rookies_in_historical_data(self):
        """2026 combine rookies must not have NFL stats in 2025 or earlier."""
        import nflreadpy as nfl
        import polars as pl

        # Get pfr_ids for known 2026 rookies
        combine = nfl.load_combine([2026])
        rookie_pfr_ids = set(
            combine.filter(pl.col("pfr_id").is_not_null())["pfr_id"].to_list()
        )

        # Bridge pfr_id -> gsis_id
        players = nfl.load_players()
        bridge = (
            players.filter(pl.col("pfr_id").is_in(list(rookie_pfr_ids)))
            .select(["gsis_id", "pfr_id"])
            .drop_nulls(subset=["gsis_id"])
        )
        rookie_gsis_ids = set(bridge["gsis_id"].to_list())

        # These gsis_ids should have no player stats in any prior season
        if rookie_gsis_ids:
            stats = nfl.load_player_stats(list(range(2020, 2026)))
            leaked = stats.filter(
                pl.col("player_id").is_in(list(rookie_gsis_ids))
            )
            assert leaked.shape[0] == 0, (
                f"{leaked.shape[0]} stat rows found for 2026 rookies: "
                f"{leaked['player_display_name'].unique().to_list()[:5]}"
            )
