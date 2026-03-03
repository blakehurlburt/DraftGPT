"""Snake draft engine — state machine for simulating drafts."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import LeagueConfig
from .players import Player


def build_snake_order(num_teams: int, num_rounds: int) -> list[int]:
    """Generate snake draft pick order.

    Even rounds (0-indexed) go 0..N-1, odd rounds go N-1..0.

    Returns:
        List of team indices in pick order.
    """
    order = []
    for rnd in range(num_rounds):
        if rnd % 2 == 0:
            order.extend(range(num_teams))
        else:
            order.extend(range(num_teams - 1, -1, -1))
    return order


@dataclass
class DraftState:
    """Mutable state for a snake draft."""

    config: LeagueConfig
    available: list[Player] = field(default_factory=list)
    teams: list[list[Player]] = field(default_factory=list)
    pick_order: list[int] = field(default_factory=list)
    current_pick: int = 0

    @classmethod
    def create(cls, config: LeagueConfig, players: list[Player]) -> "DraftState":
        """Initialize a fresh draft state."""
        pick_order = build_snake_order(config.num_teams, config.num_rounds)
        return cls(
            config=config,
            available=list(players),  # copy so original is unmodified
            teams=[[] for _ in range(config.num_teams)],
            pick_order=pick_order,
            current_pick=0,
        )

    @property
    def is_complete(self) -> bool:
        return self.current_pick >= len(self.pick_order)

    @property
    def current_team_idx(self) -> int:
        return self.pick_order[self.current_pick]

    @property
    def current_round(self) -> int:
        """1-indexed round number."""
        return self.current_pick // self.config.num_teams + 1

    def make_pick(self, player: Player) -> None:
        """Draft a player: remove from available, add to team, advance pick."""
        team_idx = self.current_team_idx
        self.available.remove(player)
        self.teams[team_idx].append(player)
        self.current_pick += 1

    def team_roster(self, team_idx: int) -> list[Player]:
        return self.teams[team_idx]

    def team_position_count(self, team_idx: int, position: str) -> int:
        return sum(1 for p in self.teams[team_idx] if p.position == position)

    def can_draft_position(self, team_idx: int, position: str) -> bool:
        """Check if team hasn't hit position cap."""
        cap = self.config.position_caps.get(position, self.config.roster_size)
        return self.team_position_count(team_idx, position) < cap

    def team_needs(self, team_idx: int) -> dict[str, int]:
        """Remaining starter slots needed by position.

        Returns dict mapping position to number of unfilled starter spots.
        """
        needs = {}
        roster = self.teams[team_idx]
        starter_slots = self.config.starter_slots()

        for pos, required in starter_slots.items():
            have = sum(1 for p in roster if p.position == pos)
            remaining = max(0, required - have)
            if remaining > 0:
                needs[pos] = remaining

        # FLEX: check if we have enough RB/WR/TE beyond starters
        flex_needed = self.config.num_flex()
        if flex_needed > 0:
            flex_surplus = 0
            for pos in self.config.flex_positions():
                have = sum(1 for p in roster if p.position == pos)
                required = starter_slots.get(pos, 0)
                flex_surplus += max(0, have - required)
            flex_remaining = max(0, flex_needed - flex_surplus)
            if flex_remaining > 0:
                needs["FLEX"] = flex_remaining

        return needs

    def picks_until_next(self, team_idx: int) -> int:
        """How many picks until this team picks again after current pick.

        Returns the gap (number of other picks between now and next turn).
        Returns roster_size * num_teams if no more picks.
        """
        if self.is_complete:
            return self.config.total_picks

        for i in range(self.current_pick + 1, len(self.pick_order)):
            if self.pick_order[i] == team_idx:
                return i - self.current_pick
        return self.config.total_picks

    def available_at_position(self, position: str) -> list[Player]:
        """Get available players at a position, sorted by projected_total desc."""
        return [p for p in self.available if p.position == position]

    def picks_remaining(self, team_idx: int) -> int:
        """How many picks this team has left including current."""
        return sum(
            1 for i in range(self.current_pick, len(self.pick_order))
            if self.pick_order[i] == team_idx
        )
