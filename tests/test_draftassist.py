"""Tests for the draftassist package."""

import pytest

from draftsim.players import Player, load_players
from draftsim.config import LeagueConfig
from draftsim.draft import DraftState

from draftassist.bridge import (
    _normalize,
    attach_sleeper_projections,
    build_player_index,
    config_from_sleeper_meta,
    default_config_for_sport,
    rebuild_draft_state,
    rebuild_from_manual_picks,
    swap_projection_source,
    _make_placeholder,
)
from draftassist.recommender import (
    top_n_picks,
    get_recommendations,
    Recommendation,
)
from draftassist.scoring import (
    default_ppr_scoring,
    extract_scoring_from_meta,
    sleeper_stats_to_fantasy_points,
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
# recommender.get_recommendations
# ---------------------------------------------------------------------------

class TestGetRecommendations:
    def test_returns_flat_list(self, sample_players, small_config):
        state = DraftState.create(small_config, sample_players)
        recs = get_recommendations(state, 0, sample_players)
        assert isinstance(recs, list)
        assert len(recs) <= 5
        assert all(isinstance(r, Recommendation) for r in recs)

    def test_vbd_values_present(self, sample_players, small_config):
        state = DraftState.create(small_config, sample_players)
        recs = get_recommendations(state, 0, sample_players)
        assert len(recs) > 0
        for r in recs:
            assert isinstance(r.vbd_value, (int, float))

    def test_with_real_players(self, real_players):
        config = LeagueConfig()
        state = DraftState.create(config, real_players)
        recs = get_recommendations(state, 0, real_players)
        assert len(recs) == 5
        for r in recs:
            assert r.player.projected_total > 0


# ---------------------------------------------------------------------------
# Rookies in recommendations / state payload
# ---------------------------------------------------------------------------

class TestRookiesInPayload:
    """Rookies must be surfaced in the UI even when they rank below the
    strategy-scored recommendation cutoff."""

    def test_real_players_have_rookies(self, real_players):
        """Projections CSV must include rookie players."""
        rookies = [p for p in real_players if p.is_rookie]
        assert len(rookies) >= 10, (
            f"Expected >=10 rookies in projections, got {len(rookies)}"
        )

    def test_rookies_in_available_pool(self, real_players):
        """Rookies must be in DraftState.available at draft start."""
        config = LeagueConfig()
        state = DraftState.create(config, real_players)
        available_rookies = [p for p in state.available if p.is_rookie]
        assert len(available_rookies) >= 10, (
            f"Expected >=10 rookies in available pool, got {len(available_rookies)}"
        )

    def test_payload_includes_available_rookies(self, real_players):
        """State payload must include available_rookies list so the UI
        can show them when the Rookies filter is active, even if they
        don't appear in strategy-ranked recommendations."""
        from draftassist.app import _build_state_payload

        config = LeagueConfig()
        state = DraftState.create(config, real_players)
        payload = _build_state_payload(
            state, {"status": "drafting"}, [], user_slot=0,
            players=real_players, skip_recommendations=False,
        )
        assert "available_rookies" in payload, (
            "Payload missing 'available_rookies' key"
        )
        rookies = payload["available_rookies"]
        assert len(rookies) >= 10, (
            f"Expected >=10 available_rookies, got {len(rookies)}"
        )
        # Each rookie entry must have the fields the UI needs
        for r in rookies[:3]:
            assert "name" in r
            assert "position" in r
            assert "projected_total" in r
            assert "is_rookie" in r and r["is_rookie"] is True


# ---------------------------------------------------------------------------
# bridge.default_config_for_sport
# ---------------------------------------------------------------------------

class TestDefaultConfigForSport:
    def test_nfl_defaults(self):
        config = default_config_for_sport("nfl")
        assert config.num_teams == 12
        assert config.roster_size == 15
        assert config.lineup["QB"] == 1
        assert config.lineup["RB"] == 2
        assert config.lineup["WR"] == 2
        assert "FLEX" in config.lineup
        assert config.flex_positions() == ["RB", "WR", "TE"]

    def test_nfl_custom_size(self):
        config = default_config_for_sport("nfl", num_teams=10, roster_size=20)
        assert config.num_teams == 10
        assert config.roster_size == 20

    def test_mlb_defaults(self):
        config = default_config_for_sport("mlb")
        assert config.num_teams == 12
        assert config.lineup["C"] == 1
        assert config.lineup["SP"] == 2
        assert config.lineup["RP"] == 2
        assert config.lineup["OF"] == 3
        assert "FLEX" in config.lineup  # UTIL slot
        # UTIL should be eligible for all batters
        assert "C" in config.flex_positions()
        assert "OF" in config.flex_positions()

    def test_mlb_custom_size(self):
        config = default_config_for_sport("mlb", num_teams=8, roster_size=22)
        assert config.num_teams == 8
        assert config.roster_size == 22


# ---------------------------------------------------------------------------
# bridge.rebuild_from_manual_picks
# ---------------------------------------------------------------------------

class TestRebuildFromManualPicks:
    def test_replays_picks(self, sample_players, small_config):
        p1, p2 = sample_players[0], sample_players[1]
        picks = [
            {"pick_no": 1, "metadata": {
                "first_name": p1.name.split()[0],
                "last_name": " ".join(p1.name.split()[1:]),
                "position": p1.position, "team": p1.team}},
            {"pick_no": 2, "metadata": {
                "first_name": p2.name.split()[0],
                "last_name": " ".join(p2.name.split()[1:]),
                "position": p2.position, "team": p2.team}},
        ]
        state = rebuild_from_manual_picks(small_config, sample_players, picks)
        assert state.current_pick == 2
        assert p1 in state.teams[0]
        assert p2 in state.teams[1]

    def test_empty_picks(self, sample_players, small_config):
        state = rebuild_from_manual_picks(small_config, sample_players, [])
        assert state.current_pick == 0
        assert len(state.available) == len(sample_players)

    def test_undo_replay(self, sample_players, small_config):
        """Simulates undo by replaying with one fewer pick."""
        p1, p2 = sample_players[0], sample_players[1]
        picks = [
            {"pick_no": 1, "metadata": {
                "first_name": p1.name.split()[0],
                "last_name": " ".join(p1.name.split()[1:]),
                "position": p1.position, "team": p1.team}},
            {"pick_no": 2, "metadata": {
                "first_name": p2.name.split()[0],
                "last_name": " ".join(p2.name.split()[1:]),
                "position": p2.position, "team": p2.team}},
        ]
        # Full replay
        state_full = rebuild_from_manual_picks(small_config, sample_players, picks)
        assert state_full.current_pick == 2

        # Undo: replay without last pick
        state_undo = rebuild_from_manual_picks(small_config, sample_players, picks[:1])
        assert state_undo.current_pick == 1
        assert p2 in state_undo.available


# ---------------------------------------------------------------------------
# config.LeagueConfig.flex_eligible
# ---------------------------------------------------------------------------

class TestFlexEligible:
    def test_default_nfl_flex(self):
        config = LeagueConfig()
        assert config.flex_positions() == ["RB", "WR", "TE"]

    def test_custom_flex(self):
        config = LeagueConfig(flex_eligible=["C", "1B", "2B", "3B", "SS", "OF", "DH"])
        assert "C" in config.flex_positions()
        assert "DH" in config.flex_positions()


# ---------------------------------------------------------------------------
# Manual draft state payload
# ---------------------------------------------------------------------------

class TestManualPayload:
    def test_build_payload_manual(self, sample_players, small_config):
        """Payload should work with manual-mode synthetic meta."""
        from draftassist.app import _build_state_payload

        state = DraftState.create(small_config, sample_players)
        meta = {"status": "in_progress"}
        payload = _build_state_payload(
            state, meta, [], user_slot=0,
            players=sample_players, skip_recommendations=False,
        )
        assert payload["draft_status"] == "in_progress"
        assert payload["current_pick"] == 1
        assert payload["num_teams"] == small_config.num_teams


# ---------------------------------------------------------------------------
# Manual draft creation endpoint (error handling)
# ---------------------------------------------------------------------------

class TestCreateManualDraft:
    """Tests for POST /api/create error handling."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from draftassist.app import app
        return TestClient(app)

    def test_create_nfl_succeeds(self, client):
        resp = client.post("/api/create?sport=nfl&num_teams=10&roster_size=15&user_slot=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "connected"
        assert data["mode"] == "manual"
        assert data["sport"] == "nfl"

    def test_create_mlb_missing_projections(self, client):
        """MLB projections file doesn't exist — should return 400 JSON, not 500."""
        resp = client.post("/api/create?sport=mlb&num_teams=10&roster_size=23&user_slot=1")
        assert resp.status_code == 400
        data = resp.json()
        assert "detail" in data
        assert "mlb_projections" in data["detail"].lower() or "not found" in data["detail"].lower()

    def test_create_invalid_sport(self, client):
        resp = client.post("/api/create?sport=nba&num_teams=10&roster_size=15&user_slot=1")
        assert resp.status_code == 400
        data = resp.json()
        assert "detail" in data

    def test_create_returns_json_on_error(self, client):
        """Verify error responses are valid JSON (not HTML 500)."""
        resp = client.post("/api/create?sport=mlb&num_teams=12&roster_size=15&user_slot=1")
        # Must be parseable JSON regardless of status
        data = resp.json()  # should not raise
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# scoring.sleeper_stats_to_fantasy_points
# ---------------------------------------------------------------------------

class TestSleeperScoring:
    def test_default_ppr_qb(self):
        """A QB stat line should produce reasonable fantasy points."""
        stats = {
            "pass_yd": 4200,
            "pass_td": 30,
            "pass_int": 12,
            "rush_yd": 300,
            "rush_td": 3,
        }
        pts = sleeper_stats_to_fantasy_points(stats)
        # 4200*0.04 + 30*4 + 12*(-1) + 300*0.1 + 3*6 = 168+120-12+30+18 = 324
        assert abs(pts - 324.0) < 0.1

    def test_default_ppr_rb(self):
        """An RB stat line with receptions (PPR)."""
        stats = {
            "rush_yd": 1200,
            "rush_td": 10,
            "rec": 50,
            "rec_yd": 400,
            "rec_td": 2,
        }
        pts = sleeper_stats_to_fantasy_points(stats)
        # 1200*0.1 + 10*6 + 50*1 + 400*0.1 + 2*6 = 120+60+50+40+12 = 282
        assert abs(pts - 282.0) < 0.1

    def test_custom_scoring(self):
        """Custom scoring should override defaults."""
        stats = {"rec": 80, "rec_yd": 1000}
        # Half-PPR
        scoring = {"rec": 0.5, "rec_yd": 0.1}
        pts = sleeper_stats_to_fantasy_points(stats, scoring)
        # 80*0.5 + 1000*0.1 = 40 + 100 = 140
        assert abs(pts - 140.0) < 0.1

    def test_empty_stats(self):
        pts = sleeper_stats_to_fantasy_points({})
        assert pts == 0.0

    def test_missing_stat_keys_handled(self):
        """Stats dict may have keys not in scoring — should be ignored."""
        stats = {"pass_yd": 100, "unknown_stat": 999}
        pts = sleeper_stats_to_fantasy_points(stats)
        assert pts == 100 * 0.04

    def test_extract_scoring_from_meta_present(self):
        meta = {
            "settings": {
                "scoring_settings": {"rec": 0.5, "pass_yd": 0.04, "rush_td": 6},
            }
        }
        scoring = extract_scoring_from_meta(meta)
        assert scoring is not None
        assert scoring["rec"] == 0.5

    def test_extract_scoring_from_meta_missing(self):
        meta = {"settings": {}}
        assert extract_scoring_from_meta(meta) is None

    def test_default_ppr_scoring_keys(self):
        scoring = default_ppr_scoring()
        assert "pass_yd" in scoring
        assert "rec" in scoring
        assert scoring["rec"] == 1.0


# ---------------------------------------------------------------------------
# bridge.attach_sleeper_projections + swap_projection_source
# ---------------------------------------------------------------------------

class TestSleeperProjections:
    def _make_players_with_ids(self):
        """Create players with sleeper_id and realistic floor/ceiling."""
        players = []
        for i, (name, pos, total) in enumerate([
            ("Alpha QB", "QB", 340),
            ("Bravo RB", "RB", 280),
            ("Charlie WR", "WR", 260),
            ("Delta TE", "TE", 180),
        ]):
            p = Player(
                name=name, position=pos, team="TST",
                projected_ppg=total / 17, projected_games=17,
                projected_total=total, pos_rank=1,
                total_floor=total * 0.75,
                total_ceiling=total * 1.25,
                sleeper_id=str(100 + i),
            )
            players.append(p)
        players.sort(key=lambda p: p.projected_total, reverse=True)
        return players

    def test_attach_populates_sleeper_fields(self):
        players = self._make_players_with_ids()
        sleeper_proj = {
            "100": {"pass_yd": 4000, "pass_td": 28, "pass_int": 10},
            "101": {"rush_yd": 1100, "rush_td": 8, "rec": 40, "rec_yd": 300, "rec_td": 2},
        }
        matched = attach_sleeper_projections(players, sleeper_proj)
        assert matched == 2
        # QB should have sleeper projections
        qb = next(p for p in players if p.name == "Alpha QB")
        assert qb.sleeper_projected_total > 0
        # WR (id=102) not in sleeper_proj — no sleeper data
        wr = next(p for p in players if p.name == "Charlie WR")
        assert wr.sleeper_projected_total == 0.0

    def test_attach_saves_model_backup(self):
        players = self._make_players_with_ids()
        sleeper_proj = {
            "100": {"pass_yd": 4000, "pass_td": 28, "pass_int": 10},
        }
        attach_sleeper_projections(players, sleeper_proj)
        qb = next(p for p in players if p.name == "Alpha QB")
        assert qb._model_projected_total == 340
        assert qb._model_total_floor == 340 * 0.75

    def test_swap_to_sleeper_and_back(self):
        players = self._make_players_with_ids()
        sleeper_proj = {
            "100": {"pass_yd": 4000, "pass_td": 28, "pass_int": 10},
            "101": {"rush_yd": 1100, "rush_td": 8, "rec": 40, "rec_yd": 300, "rec_td": 2},
        }
        attach_sleeper_projections(players, sleeper_proj)

        # Save original model values
        original_totals = {p.name: p.projected_total for p in players}

        # Swap to sleeper
        swap_projection_source(players, "sleeper")
        qb = next(p for p in players if p.name == "Alpha QB")
        assert qb.projected_total != original_totals["Alpha QB"]
        assert qb.projected_total == qb.sleeper_projected_total

        # Players without sleeper data keep model values
        wr = next(p for p in players if p.name == "Charlie WR")
        assert wr.projected_total == original_totals["Charlie WR"]

        # Swap back to model
        swap_projection_source(players, "model")
        for p in players:
            assert abs(p.projected_total - original_totals[p.name]) < 0.01, (
                f"{p.name}: expected {original_totals[p.name]}, got {p.projected_total}"
            )

    def test_swap_recomputes_pos_rank(self):
        players = self._make_players_with_ids()
        sleeper_proj = {
            "100": {"pass_yd": 1000, "pass_td": 5, "pass_int": 2},  # low QB
            "101": {"rush_yd": 2000, "rush_td": 20, "rec": 80, "rec_yd": 800, "rec_td": 5},  # elite RB
        }
        attach_sleeper_projections(players, sleeper_proj)
        swap_projection_source(players, "sleeper")

        # RB should now rank higher than QB
        rb = next(p for p in players if p.name == "Bravo RB")
        qb = next(p for p in players if p.name == "Alpha QB")
        assert rb.projected_total > qb.projected_total

        # pos_rank should be reassigned
        assert rb.pos_rank == 1  # only RB

    def test_estimated_floor_ceiling_on_sleeper(self):
        players = self._make_players_with_ids()
        sleeper_proj = {
            "100": {"pass_yd": 4000, "pass_td": 28, "pass_int": 10},
        }
        attach_sleeper_projections(players, sleeper_proj)
        swap_projection_source(players, "sleeper")
        qb = next(p for p in players if p.name == "Alpha QB")
        # Floor/ceiling should be estimated (not zero)
        assert qb.total_floor > 0
        assert qb.total_ceiling > qb.total_floor
        assert qb.total_floor < qb.projected_total
        assert qb.total_ceiling > qb.projected_total


# ---------------------------------------------------------------------------
# Projection source in state payload
# ---------------------------------------------------------------------------

class TestProjectionSourcePayload:
    def test_payload_includes_projection_source(self, sample_players, small_config):
        from draftassist.app import _build_state_payload

        state = DraftState.create(small_config, sample_players)
        payload = _build_state_payload(
            state, {"status": "in_progress"}, [], user_slot=0,
            players=sample_players, projection_source="model",
        )
        assert payload["projection_source"] == "model"
        assert payload["floor_estimated"] is False

    def test_payload_floor_estimated_sleeper(self, sample_players, small_config):
        from draftassist.app import _build_state_payload

        state = DraftState.create(small_config, sample_players)
        payload = _build_state_payload(
            state, {"status": "in_progress"}, [], user_slot=0,
            players=sample_players, projection_source="sleeper",
        )
        assert payload["projection_source"] == "sleeper"
        assert payload["floor_estimated"] is True
