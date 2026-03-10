"""Tests for the draftassist package."""

import pytest

from draftsim.players import Player, load_players
from draftsim.config import LeagueConfig
from draftsim.draft import DraftState
from draftsim.strategies import STRATEGIES

from draftassist.bridge import (
    _normalize,
    build_player_index,
    config_from_sleeper_meta,
    rebuild_draft_state,
    _make_placeholder,
)
from draftassist.recommender import (
    top_n_picks,
    get_all_recommendations,
    Recommendation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_player(name, pos, total, rank=1, team="TST"):
    return Player(name=name, position=pos, team=team,
                  projected_ppg=total / 17.0, projected_games=17.0,
                  projected_total=total, pos_rank=rank)


@pytest.fixture
def sample_players():
    players = []
    for i, (name, total) in enumerate([
        ("QB Alpha", 340), ("QB Bravo", 310),
    ], 1):
        players.append(_make_player(name, "QB", total, rank=i))
    for i, total in enumerate(range(300, 200, -10), 1):
        players.append(_make_player(f"RB {i}", "RB", total, rank=i))
    for i, total in enumerate(range(290, 190, -10), 1):
        players.append(_make_player(f"WR {i}", "WR", total, rank=i))
    for i, total in enumerate(range(200, 140, -10), 1):
        players.append(_make_player(f"TE {i}", "TE", total, rank=i))
    players.sort(key=lambda p: p.projected_total, reverse=True)
    return players


@pytest.fixture
def small_config():
    return LeagueConfig(num_teams=4, roster_size=7,
                        lineup={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1})


@pytest.fixture
def real_players():
    return load_players()


# ---------------------------------------------------------------------------
# bridge._normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_basic(self):
        assert _normalize("Josh Allen") == "josh allen"

    def test_strips_suffix(self):
        assert _normalize("Todd Gurley II") == "todd gurley"
        assert _normalize("Marvin Harrison Jr.") == "marvin harrison"
        assert _normalize("Odell Beckham Jr") == "odell beckham"

    def test_strips_punctuation(self):
        assert _normalize("Ja'Marr Chase") == "jamarr chase"

    def test_collapses_whitespace(self):
        assert _normalize("  John   Smith  ") == "john smith"


# ---------------------------------------------------------------------------
# bridge.build_player_index
# ---------------------------------------------------------------------------

class TestBuildPlayerIndex:
    def test_matches_by_name_and_position(self):
        players = [_make_player("Josh Allen", "QB", 340)]
        sleeper_db = {
            "100": {"first_name": "Josh", "last_name": "Allen", "position": "QB"},
            "200": {"first_name": "Josh", "last_name": "Allen", "position": "DE"},  # wrong position
        }
        index = build_player_index(players, sleeper_db)
        assert "100" in index
        assert "200" not in index
        assert index["100"].sleeper_id == "100"

    def test_ignores_non_skill_positions(self):
        players = [_make_player("Test Kicker", "QB", 100)]
        sleeper_db = {
            "300": {"first_name": "Test", "last_name": "Kicker", "position": "K"},
        }
        index = build_player_index(players, sleeper_db)
        assert len(index) == 0

    def test_handles_missing_fields(self):
        players = [_make_player("Test", "QB", 100)]
        sleeper_db = {
            "400": {"first_name": "", "last_name": "Test"},  # missing position
            "500": {},  # empty
        }
        index = build_player_index(players, sleeper_db)
        assert len(index) == 0

    def test_real_player_matching(self, real_players):
        """Build index with a realistic sleeper-style DB."""
        sleeper_db = {}
        matched_count = 0
        for i, p in enumerate(real_players[:50]):
            parts = p.name.split(maxsplit=1)
            first = parts[0] if parts else ""
            last = parts[1] if len(parts) > 1 else ""
            sid = str(1000 + i)
            sleeper_db[sid] = {
                "first_name": first,
                "last_name": last,
                "position": p.position,
            }
        index = build_player_index(real_players, sleeper_db)
        # Most should match (some may fail due to suffix normalization)
        assert len(index) >= 40


# ---------------------------------------------------------------------------
# bridge.config_from_sleeper_meta
# ---------------------------------------------------------------------------

class TestConfigFromSleeperMeta:
    def test_basic_config(self):
        meta = {
            "settings": {
                "teams": 10,
                "rounds": 15,
                "roster_positions": [
                    "QB", "RB", "RB", "WR", "WR", "TE", "FLEX",
                    "BN", "BN", "BN", "BN", "BN", "BN", "BN", "BN",
                ],
            }
        }
        config = config_from_sleeper_meta(meta)
        assert config.num_teams == 10
        assert config.roster_size == 15
        assert config.lineup["QB"] == 1
        assert config.lineup["RB"] == 2
        assert config.lineup["WR"] == 2
        assert config.lineup["TE"] == 1
        assert config.lineup["FLEX"] == 1

    def test_superflex(self):
        meta = {
            "settings": {
                "teams": 12,
                "rounds": 15,
                "roster_positions": [
                    "QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX",
                    "BN", "BN", "BN", "BN", "BN", "BN", "BN",
                ],
            }
        }
        config = config_from_sleeper_meta(meta)
        # SUPER_FLEX should be merged into FLEX
        assert config.lineup["FLEX"] == 2
        assert "SUPER_FLEX" not in config.lineup

    def test_defaults_when_empty(self):
        meta = {"settings": {"teams": 8, "rounds": 12}}
        config = config_from_sleeper_meta(meta)
        assert config.num_teams == 8
        assert config.roster_size == 12
        assert "QB" in config.lineup


# ---------------------------------------------------------------------------
# bridge.rebuild_draft_state
# ---------------------------------------------------------------------------

class TestRebuildDraftState:
    def test_replay_picks(self, sample_players, small_config):
        # Build an index for first 2 players
        p1, p2 = sample_players[0], sample_players[1]
        p1.sleeper_id = "100"
        p2.sleeper_id = "200"
        id_to_player = {"100": p1, "200": p2}

        picks = [
            {"pick_no": 1, "player_id": "100", "draft_slot": 1, "round": 1,
             "metadata": {"first_name": "X", "last_name": "Y", "position": "QB", "team": "T"}},
            {"pick_no": 2, "player_id": "200", "draft_slot": 2, "round": 1,
             "metadata": {"first_name": "A", "last_name": "B", "position": "RB", "team": "T"}},
        ]

        state = rebuild_draft_state(small_config, sample_players, picks, id_to_player)
        assert state.current_pick == 2
        assert p1 in state.teams[0]
        assert p2 in state.teams[1]
        assert p1 not in state.available
        assert p2 not in state.available

    def test_unmatched_placeholder(self, sample_players, small_config):
        """Unmatched picks (K/DST) should create placeholders."""
        picks = [
            {"pick_no": 1, "player_id": "999", "draft_slot": 1, "round": 1,
             "metadata": {"first_name": "Tyler", "last_name": "Bass", "position": "K", "team": "BUF"}},
        ]
        state = rebuild_draft_state(small_config, sample_players, picks, {})
        assert state.current_pick == 1
        assert len(state.teams[0]) == 1
        # Placeholder has zero projection
        assert state.teams[0][0].projected_total == 0.0

    def test_empty_picks(self, sample_players, small_config):
        state = rebuild_draft_state(small_config, sample_players, [], {})
        assert state.current_pick == 0
        assert len(state.available) == len(sample_players)


class TestMakePlaceholder:
    def test_basic(self):
        pick = {
            "player_id": "999",
            "metadata": {
                "first_name": "Tyler",
                "last_name": "Bass",
                "position": "K",
                "team": "BUF",
            },
        }
        p = _make_placeholder(pick)
        assert p.name == "Tyler Bass"
        assert p.projected_total == 0.0
        # K is now a supported position
        assert p.position == "K"


# ---------------------------------------------------------------------------
# recommender.top_n_picks
# ---------------------------------------------------------------------------

class TestTopNPicks:
    def test_returns_n_picks(self, sample_players, small_config):
        from draftsim.strategies import pick_bpa
        state = DraftState.create(small_config, sample_players)
        recs = top_n_picks(pick_bpa, state, 0, n=5)
        assert len(recs) == 5
        assert all(isinstance(r, Recommendation) for r in recs)

    def test_ranks_sequential(self, sample_players, small_config):
        from draftsim.strategies import pick_bpa
        state = DraftState.create(small_config, sample_players)
        recs = top_n_picks(pick_bpa, state, 0, n=3)
        assert [r.rank for r in recs] == [1, 2, 3]

    def test_no_duplicates(self, sample_players, small_config):
        from draftsim.strategies import pick_vbd
        state = DraftState.create(small_config, sample_players)
        recs = top_n_picks(pick_vbd, state, 0, n=5, players=sample_players)
        names = [r.player.name for r in recs]
        assert len(names) == len(set(names))

    def test_restores_available(self, sample_players, small_config):
        from draftsim.strategies import pick_bpa
        state = DraftState.create(small_config, sample_players)
        original_count = len(state.available)
        top_n_picks(pick_bpa, state, 0, n=5)
        assert len(state.available) == original_count

    def test_empty_when_complete(self, sample_players, small_config):
        from draftsim.strategies import pick_bpa
        state = DraftState.create(small_config, sample_players)
        while not state.is_complete:
            state.make_pick(state.available[0])
        recs = top_n_picks(pick_bpa, state, 0, n=5)
        assert recs == []


# ---------------------------------------------------------------------------
# recommender.get_all_recommendations
# ---------------------------------------------------------------------------

class TestGetAllRecommendations:
    def test_returns_all_strategies(self, sample_players, small_config):
        state = DraftState.create(small_config, sample_players)
        recs = get_all_recommendations(state, 0, sample_players)
        assert set(recs.keys()) == set(STRATEGIES.keys())
        for name, rec_list in recs.items():
            assert len(rec_list) <= 5
            assert all(isinstance(r, Recommendation) for r in rec_list)

    def test_vbd_values_present(self, sample_players, small_config):
        state = DraftState.create(small_config, sample_players)
        recs = get_all_recommendations(state, 0, sample_players)
        for name, rec_list in recs.items():
            assert len(rec_list) > 0, f"No recommendations for {name}"
            for r in rec_list:
                assert isinstance(r.vbd_value, (int, float))

    def test_with_real_players(self, real_players):
        config = LeagueConfig()
        state = DraftState.create(config, real_players)
        recs = get_all_recommendations(state, 0, real_players)
        assert len(recs) == 5
        for name, rec_list in recs.items():
            assert len(rec_list) == 5
            for r in rec_list:
                assert r.player.projected_total > 0
