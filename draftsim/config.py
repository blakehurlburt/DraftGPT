"""League configuration for draft simulation."""

from __future__ import annotations

from dataclasses import dataclass, field


# Flex slot names used in lineup dicts (not real positions).
_FLEX_SLOTS = {"FLEX", "PFLEX"}


@dataclass
class LeagueConfig:
    """Fantasy league settings."""

    num_teams: int = 12
    roster_size: int = 15
    lineup: dict = field(
        default_factory=lambda: {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1}
    )
    scoring: str = "ppr"

    # Position caps to prevent absurd rosters
    position_caps: dict = field(
        default_factory=lambda: {"QB": 3, "RB": 6, "WR": 6, "TE": 3, "K": 2, "DST": 2}
    )
    # Positions eligible for flex slots.
    # Accepts a list (backward compat, applies to FLEX) or a dict mapping
    # flex slot name -> list of eligible positions.
    flex_eligible: list | dict = field(
        default_factory=lambda: ["RB", "WR", "TE"]
    )

    def __post_init__(self):
        # Normalize list form to dict keyed by "FLEX"
        if isinstance(self.flex_eligible, list):
            self.flex_eligible = {"FLEX": self.flex_eligible}

    @property
    def num_rounds(self) -> int:
        return self.roster_size

    @property
    def total_picks(self) -> int:
        return self.num_teams * self.roster_size

    def starter_slots(self) -> dict[str, int]:
        """Return number of starter slots per position (flex slots excluded)."""
        return {k: v for k, v in self.lineup.items() if k not in _FLEX_SLOTS}

    def flex_positions(self) -> list[str]:
        """All positions eligible for any flex slot (union)."""
        seen: set[str] = set()
        result: list[str] = []
        for positions in self.flex_eligible.values():
            for p in positions:
                if p not in seen:
                    seen.add(p)
                    result.append(p)
        return result

    def num_flex(self) -> int:
        """Total flex slot count across all flex types."""
        return sum(self.lineup.get(slot, 0) for slot in self.flex_eligible)

    def flex_slot_info(self) -> list[tuple[str, int, list[str]]]:
        """Return [(slot_name, count, eligible_positions), ...] for each flex type."""
        return [
            (slot, self.lineup.get(slot, 0), positions)
            for slot, positions in self.flex_eligible.items()
            if self.lineup.get(slot, 0) > 0
        ]
