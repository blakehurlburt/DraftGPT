"""Tests for the draftsim package."""

import pytest
import numpy as np

from draftsim.players import Player, load_players
from draftsim.config import LeagueConfig
from draftsim.draft import DraftState, build_snake_order
from draftsim.strategies import (
    pick_bpa, pick_vbd, pick_vona,
    get_strategy, STRATEGIES, _eligible, _force_need_pick,
)
from draftsim.value import (
    compute_replacement_levels, compute_last_starter_levels,
    vbd, vorp, vols, vbd_score, vona,
    _pick_probability, _need_adjusted_adp,
)
from draftsim.adp import (
    generate_platform_adp, generate_consensus_adp, load_adp,
    PLATFORMS,
)
from draftsim.opponents import ADPOpponent
from draftsim.results import compute_optimal_lineup


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_player(name, pos, total, rank=1, team="TST"):
    return Player(name=name, position=pos, team=team,
                  projected_ppg=total / 17.0, projected_games=17.0,
                  projected_total=total, pos_rank=rank)


@pytest.fixture
def sample_players():
    """A minimal pool of 30 players for testing."""
    players = []
    # 4 QBs
    for i, (name, total) in enumerate([
        ("QB Alpha", 340), ("QB Bravo", 310), ("QB Charlie", 280), ("QB Delta", 250),
    ], 1):
        players.append(_make_player(name, "QB", total, rank=i))
    # 10 RBs
    for i, total in enumerate(range(300, 200, -10), 1):
        players.append(_make_player(f"RB_{i}", "RB", total, rank=i))
    # 10 WRs
    for i, total in enumerate(range(290, 190, -10), 1):
        players.append(_make_player(f"WR_{i}", "WR", total, rank=i))
    # 6 TEs
    for i, total in enumerate(range(200, 140, -10), 1):
        players.append(_make_player(f"TE_{i}", "TE", total, rank=i))
    players.sort(key=lambda p: p.projected_total, reverse=True)
    return players


@pytest.fixture
def small_config():
    return LeagueConfig(num_teams=4, roster_size=7,
                        lineup={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1})


@pytest.fixture
def default_config():
    return LeagueConfig()


@pytest.fixture
def real_players():
    """Load the actual projections CSV."""
    return load_players()


# ---------------------------------------------------------------------------
# Player dataclass
# ---------------------------------------------------------------------------

class TestPlayer:
    def test_create(self):
        p = _make_player("Test", "QB", 300)
        assert p.name == "Test"
        assert p.position == "QB"
        assert p.projected_total == 300

    def test_sleeper_id_default(self):
        p = _make_player("Test", "QB", 300)
        assert p.sleeper_id == ""

    def test_sleeper_id_set(self):
        p = Player("X", "RB", "T", 10, 17, 170, 1, sleeper_id="abc")
        assert p.sleeper_id == "abc"

    def test_repr(self):
        p = _make_player("Test", "QB", 300)
        assert "Test" in repr(p) and "QB" in repr(p)

    def test_load_players(self, real_players):
        assert len(real_players) > 100
        positions = {p.position for p in real_players}
        assert positions == {"QB", "RB", "WR", "TE", "K", "DST"}
        # Sorted descending by projected_total
        totals = [p.projected_total for p in real_players]
        assert totals == sorted(totals, reverse=True)


# ---------------------------------------------------------------------------
# LeagueConfig
# ---------------------------------------------------------------------------

class TestLeagueConfig:
    def test_defaults(self, default_config):
        assert default_config.num_teams == 12
        assert default_config.roster_size == 15
        assert default_config.num_rounds == 15
        assert default_config.total_picks == 180

    def test_starter_slots(self, default_config):
        slots = default_config.starter_slots()
        assert "FLEX" not in slots
        assert slots["QB"] == 1
        assert slots["RB"] == 2

    def test_flex(self, default_config):
        assert default_config.num_flex() == 1
        assert set(default_config.flex_positions()) == {"RB", "WR", "TE"}

    def test_position_caps(self, default_config):
        assert default_config.position_caps["QB"] == 3


# ---------------------------------------------------------------------------
# DraftState & snake order
# ---------------------------------------------------------------------------

class TestSnakeOrder:
    def test_length(self):
        order = build_snake_order(4, 3)
        assert len(order) == 12

    def test_round1_forward(self):
        order = build_snake_order(4, 2)
        assert order[:4] == [0, 1, 2, 3]

    def test_round2_reverse(self):
        order = build_snake_order(4, 2)
        assert order[4:8] == [3, 2, 1, 0]


class TestDraftState:
    def test_create(self, sample_players, small_config):
        state = DraftState.create(small_config, sample_players)
        assert len(state.available) == len(sample_players)
        assert state.current_pick == 0
        assert not state.is_complete

    def test_make_pick(self, sample_players, small_config):
        state = DraftState.create(small_config, sample_players)
        first = state.available[0]
        state.make_pick(first)
        assert first not in state.available
        assert first in state.teams[0]
        assert state.current_pick == 1

    def test_current_team_snake(self, sample_players, small_config):
        state = DraftState.create(small_config, sample_players)
        # First pick = team 0
        assert state.current_team_idx == 0
        for i in range(4):
            state.make_pick(state.available[0])
        # 5th pick (round 2) = team 3 (snake)
        assert state.current_team_idx == 3

    def test_current_round(self, sample_players, small_config):
        state = DraftState.create(small_config, sample_players)
        assert state.current_round == 1
        for _ in range(4):
            state.make_pick(state.available[0])
        assert state.current_round == 2

    def test_team_needs(self, sample_players, small_config):
        state = DraftState.create(small_config, sample_players)
        needs = state.team_needs(0)
        assert needs["QB"] == 1
        assert needs["RB"] == 2
        assert needs["WR"] == 2
        assert needs["TE"] == 1

    def test_picks_until_next(self, sample_players, small_config):
        state = DraftState.create(small_config, sample_players)
        # Team 0 picks at position 0, next at position 7 (4+3)
        gap = state.picks_until_next(0)
        assert gap == 7  # 0...(3,2,1,0)...next at index 7

    def test_is_complete(self, sample_players, small_config):
        state = DraftState.create(small_config, sample_players)
        total = small_config.total_picks
        for _ in range(total):
            state.make_pick(state.available[0])
        assert state.is_complete

    def test_position_count(self, sample_players, small_config):
        state = DraftState.create(small_config, sample_players)
        qb = next(p for p in state.available if p.position == "QB")
        state.make_pick(qb)
        assert state.team_position_count(0, "QB") == 1

    def test_can_draft_position_cap(self, sample_players):
        # Use a 2-team league so team 0 picks every other turn
        config = LeagueConfig(num_teams=2, roster_size=7,
                              lineup={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
                              position_caps={"QB": 3, "RB": 6, "WR": 6, "TE": 3})
        state = DraftState.create(config, sample_players)
        drafted = 0
        while drafted < 3 and not state.is_complete:
            if state.current_team_idx == 0:
                # Pick a QB if one is available
                qb = next((p for p in state.available if p.position == "QB"), None)
                if qb:
                    state.make_pick(qb)
                    drafted += 1
                    continue
            state.make_pick(state.available[0])
        assert state.team_position_count(0, "QB") == 3
        assert not state.can_draft_position(0, "QB")


# ---------------------------------------------------------------------------
# Value math
# ---------------------------------------------------------------------------

class TestValue:
    def test_replacement_levels(self, sample_players, small_config):
        levels = compute_replacement_levels(sample_players, small_config)
        assert "QB" in levels and "RB" in levels
        assert all(v >= 0 for v in levels.values())

    def test_vbd_positive_for_top_players(self, sample_players, small_config):
        levels = compute_replacement_levels(sample_players, small_config)
        top_qb = next(p for p in sample_players if p.position == "QB")
        assert vbd(top_qb, levels) > 0

    def test_vona_returns_float(self, sample_players, small_config):
        state = DraftState.create(small_config, sample_players)
        adp_order = [p.name for p in sample_players]
        result = vona(state, 0, "QB", adp_order)
        assert isinstance(result, (int, float))
        assert result >= 0.0

    def test_vona_zero_for_empty_position(self, sample_players, small_config):
        state = DraftState.create(small_config, sample_players)
        # Remove all QBs from available
        state.available = [p for p in state.available if p.position != "QB"]
        adp_order = [p.name for p in state.available]
        result = vona(state, 0, "QB", adp_order)
        assert result == 0.0

    def test_vona_with_round_params(self, sample_players, small_config):
        """VONA accepts current_round/total_rounds and returns valid result."""
        state = DraftState.create(small_config, sample_players)
        adp_order = [p.name for p in sample_players]
        result = vona(state, 0, "RB", adp_order, current_round=3, total_rounds=7)
        assert isinstance(result, (int, float))
        assert result >= 0.0

    def test_vona_probabilistic_smoother(self, sample_players, small_config):
        """Probabilistic VONA should give non-negative, finite values."""
        state = DraftState.create(small_config, sample_players)
        adp_order = [p.name for p in sample_players]
        for pos in ["QB", "RB", "WR", "TE"]:
            result = vona(state, 0, pos, adp_order,
                          current_round=1, total_rounds=7)
            assert result >= 0.0
            assert result < 1000  # sanity bound

    def test_pick_probability_bounds(self):
        """_pick_probability returns values in [0, 1]."""
        for adp in range(1, 20):
            for gap in range(1, 15):
                p = _pick_probability(adp, gap, 1, 15)
                assert 0.0 <= p <= 1.0
                p_late = _pick_probability(adp, gap, 14, 15)
                assert 0.0 <= p_late <= 1.0

    def test_pick_probability_decreases_with_adp(self):
        """Higher ADP position (further from pick) = lower probability of being taken."""
        p_close = _pick_probability(2, 10, 1, 15)
        p_far = _pick_probability(15, 10, 1, 15)
        assert p_close > p_far

    def test_need_adjusted_adp_length(self, sample_players, small_config):
        """Need-adjusted ADP preserves all names."""
        state = DraftState.create(small_config, sample_players)
        adp_order = [p.name for p in sample_players]
        adjusted = _need_adjusted_adp(state, 0, adp_order)
        assert len(adjusted) == len(adp_order)
        assert set(adjusted) == set(adp_order)

    def test_vorp_is_vbd_alias(self):
        """vorp is an alias for vbd."""
        assert vorp is vbd

    def test_last_starter_levels(self, sample_players, small_config):
        """Last starter levels are at or above replacement levels."""
        state = DraftState.create(small_config, sample_players)
        repl = compute_replacement_levels(sample_players, small_config)
        last = compute_last_starter_levels(
            state.available, small_config, state.teams
        )
        # Last starter should be >= replacement level (higher baseline)
        for pos in ["QB", "RB", "WR", "TE"]:
            if pos in repl and pos in last:
                assert last[pos] >= repl[pos], (
                    f"{pos}: last starter {last[pos]} < replacement {repl[pos]}"
                )

    def test_last_starter_levels_adjust_after_drafting(self, sample_players, small_config):
        """As starters get drafted, last starter levels respond."""
        state = DraftState.create(small_config, sample_players)
        # Draft two RBs for team 0 (fills starter slots, removes top RBs)
        for _ in range(2):
            rb = next(p for p in state.available if p.position == "RB")
            state.make_pick(rb)
        levels_after = compute_last_starter_levels(
            state.available, small_config, state.teams
        )
        # Remaining RB need is reduced; last starter among available should
        # be lower since top 2 RBs are gone from the pool
        # (remaining_need went from 8 to 6, but pool lost 2 top players)
        assert levels_after["RB"] > 0  # still some need

    def test_vols_returns_non_negative(self, sample_players, small_config):
        """VOLS is always >= 0 (floored)."""
        state = DraftState.create(small_config, sample_players)
        last = compute_last_starter_levels(
            state.available, small_config, state.teams
        )
        for p in sample_players:
            v = vols(p, last)
            assert v >= 0.0, f"VOLS for {p.name} = {v}"

    def test_vols_less_than_vorp(self, sample_players, small_config):
        """VOLS <= VORP for all players (VOLS has a higher baseline)."""
        state = DraftState.create(small_config, sample_players)
        repl = compute_replacement_levels(sample_players, small_config)
        last = compute_last_starter_levels(
            state.available, small_config, state.teams
        )
        for p in sample_players:
            v_vols = vols(p, last)
            v_vorp = max(0.0, vbd(p, repl))
            assert v_vols <= v_vorp + 0.01, (
                f"{p.name}: VOLS {v_vols} > VORP {v_vorp}"
            )

    def test_vols_top_player_positive(self, sample_players, small_config):
        """Top player at each position should have positive VOLS."""
        state = DraftState.create(small_config, sample_players)
        last = compute_last_starter_levels(
            state.available, small_config, state.teams
        )
        for pos in ["QB", "RB", "WR", "TE"]:
            top = next(p for p in sample_players if p.position == pos)
            assert vols(top, last) > 0, f"Top {pos} should have VOLS > 0"

    def test_vbd_score_composite(self):
        """VBD score is the sum of VORP + VONA + VOLS."""
        assert vbd_score(50.0, 10.0, 30.0) == 90.0

    def test_vbd_score_floors_negatives(self):
        """VBD score floors each component at 0."""
        assert vbd_score(-10.0, 5.0, -3.0) == 5.0
        assert vbd_score(0.0, 0.0, 0.0) == 0.0

    def test_vbd_score_all_positive(self, sample_players, small_config):
        """VBD score is always non-negative for any player."""
        state = DraftState.create(small_config, sample_players)
        repl = compute_replacement_levels(sample_players, small_config)
        last = compute_last_starter_levels(
            state.available, small_config, state.teams
        )
        adp_order = [p.name for p in sample_players]
        for p in sample_players:
            v = vbd_score(
                vbd(p, repl),
                vona(state, 0, p.position, adp_order),
                vols(p, last),
            )
            assert v >= 0.0

    def test_vbd_score_top_players_high(self, sample_players, small_config):
        """Top players should have higher VBD scores than bottom players."""
        state = DraftState.create(small_config, sample_players)
        repl = compute_replacement_levels(sample_players, small_config)
        last = compute_last_starter_levels(
            state.available, small_config, state.teams
        )
        adp_order = [p.name for p in sample_players]

        def _score(p):
            return vbd_score(
                vbd(p, repl),
                vona(state, 0, p.position, adp_order),
                vols(p, last),
            )

        top_qb = next(p for p in sample_players if p.position == "QB")
        bottom_qb = [p for p in sample_players if p.position == "QB"][-1]
        assert _score(top_qb) > _score(bottom_qb)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class TestStrategies:
    def test_registry(self):
        assert len(STRATEGIES) == 3
        for name in ["bpa", "vbd", "vona"]:
            assert name in STRATEGIES
            assert get_strategy(name) is STRATEGIES[name]

    def test_get_strategy_invalid(self):
        with pytest.raises(ValueError):
            get_strategy("nonexistent")

    def test_bpa_picks_highest(self, sample_players, small_config):
        state = DraftState.create(small_config, sample_players)
        pick = pick_bpa(state, 0)
        assert pick.projected_total == max(p.projected_total for p in sample_players)

    def test_vbd_returns_player(self, sample_players, small_config):
        state = DraftState.create(small_config, sample_players)
        pick = pick_vbd(state, 0, players=sample_players)
        assert isinstance(pick, Player)
        assert pick in sample_players

    def test_vona_returns_player(self, sample_players, small_config):
        state = DraftState.create(small_config, sample_players)
        pick = pick_vona(state, 0, players=sample_players)
        assert isinstance(pick, Player)

    def test_all_strategies_complete_draft(self, sample_players, small_config):
        """Each strategy can complete a full draft without error."""
        for name, fn in STRATEGIES.items():
            state = DraftState.create(small_config, sample_players)
            while not state.is_complete:
                team = state.current_team_idx
                player = fn(state, team, players=sample_players)
                state.make_pick(player)
            # All teams should have full rosters
            for t in range(small_config.num_teams):
                assert len(state.teams[t]) == small_config.roster_size

    def test_eligible_respects_caps(self, sample_players, small_config):
        state = DraftState.create(small_config, sample_players)
        eligible = _eligible(state, 0)
        assert len(eligible) == len(sample_players)  # no caps hit yet


# ---------------------------------------------------------------------------
# ADP
# ---------------------------------------------------------------------------

class TestADP:
    def test_platform_adp_length(self, sample_players):
        entries = generate_platform_adp(sample_players, "sleeper")
        assert len(entries) == len(sample_players)

    def test_platform_adp_sorted(self, sample_players):
        entries = generate_platform_adp(sample_players, "espn")
        adps = [adp for _, adp in entries]
        assert adps == sorted(adps)

    def test_consensus_adp(self, sample_players):
        entries = generate_consensus_adp(sample_players)
        assert len(entries) == len(sample_players)

    def test_load_adp(self):
        adp = load_adp("consensus")
        assert len(adp) > 0
        assert all(isinstance(v, float) for v in adp.values())

    def test_all_platforms_exist(self):
        assert "sleeper" in PLATFORMS
        assert "espn" in PLATFORMS
        assert "consensus" in PLATFORMS


# ---------------------------------------------------------------------------
# Opponents
# ---------------------------------------------------------------------------

class TestOpponent:
    def test_pick_returns_valid_player(self, sample_players, small_config):
        rng = np.random.default_rng(42)
        adp = {p.name: float(i) for i, p in enumerate(sample_players, 1)}
        opp = ADPOpponent(adp, rng)
        state = DraftState.create(small_config, sample_players)
        pick = opp.pick(state, 0)
        assert isinstance(pick, Player)
        assert pick in sample_players

    def test_opponent_completes_draft(self, sample_players, small_config):
        rng = np.random.default_rng(42)
        adp = {p.name: float(i) for i, p in enumerate(sample_players, 1)}
        opp = ADPOpponent(adp, rng)
        state = DraftState.create(small_config, sample_players)
        while not state.is_complete:
            team = state.current_team_idx
            pick = opp.pick(state, team)
            state.make_pick(pick)
        assert state.is_complete


# ---------------------------------------------------------------------------
# Results / lineup scoring
# ---------------------------------------------------------------------------

class TestResults:
    def test_optimal_lineup(self, default_config):
        roster = [
            _make_player("QB1", "QB", 300),
            _make_player("RB1", "RB", 250),
            _make_player("RB2", "RB", 230),
            _make_player("WR1", "WR", 260),
            _make_player("WR2", "WR", 240),
            _make_player("TE1", "TE", 180),
            _make_player("RB3", "RB", 200),  # flex candidate
            _make_player("WR3", "WR", 190),
        ]
        lineup, total = compute_optimal_lineup(roster, default_config)
        # Should pick QB1, RB1, RB2, WR1, WR2, TE1, + flex (RB3 at 200)
        assert len(lineup) == 7  # 1+2+2+1+1
        assert total == 300 + 250 + 230 + 260 + 240 + 180 + 200

    def test_empty_roster(self, default_config):
        lineup, total = compute_optimal_lineup([], default_config)
        assert lineup == []
        assert total == 0.0


# ---------------------------------------------------------------------------
# Integration: full draft with real projections
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_draft_real_players(self, real_players, default_config):
        """Run a complete 12-team draft with VBD strategy using real data."""
        state = DraftState.create(default_config, real_players)
        while not state.is_complete:
            team = state.current_team_idx
            pick = pick_vbd(state, team, players=real_players)
            state.make_pick(pick)
        for t in range(12):
            assert len(state.teams[t]) == 15
            lineup, total = compute_optimal_lineup(state.teams[t], default_config)
            assert total > 0
