"""League configuration for draft simulation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LeagueConfig:
    """Fantasy league settings."""

    num_teams: int = 12
    roster_size: int = 15
    # CR opus: roster_size=15 but lineup sums to 9 starters (QB:1+RB:2+WR:2+TE:1+FLEX:1+K:1+DST:1).
    # That leaves 6 bench spots, which is fine, but nothing enforces roster_size >= sum(lineup).
    # If someone sets roster_size=8, the draft ends before filling all starter slots.
    lineup: dict = field(
        default_factory=lambda: {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1}
    )
    scoring: str = "ppr"

    # Position caps to prevent absurd rosters
    position_caps: dict = field(
        default_factory=lambda: {"QB": 3, "RB": 6, "WR": 6, "TE": 3, "K": 2, "DST": 2}
    )
    # Positions eligible for FLEX slots
    flex_eligible: list = field(
        default_factory=lambda: ["RB", "WR", "TE"]
    )

    @property
    def num_rounds(self) -> int:
        return self.roster_size

    @property
    def total_picks(self) -> int:
        return self.num_teams * self.roster_size

    def starter_slots(self) -> dict[str, int]:
        """Return number of starter slots per position (FLEX counted separately)."""
        return {k: v for k, v in self.lineup.items() if k != "FLEX"}

    def flex_positions(self) -> list[str]:
        """Positions eligible for FLEX."""
        return self.flex_eligible

    def num_flex(self) -> int:
        return self.lineup.get("FLEX", 0)
