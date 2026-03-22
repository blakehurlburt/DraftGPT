"""ADP-based opponent modeling for draft simulation."""

import re

import numpy as np

from .config import LeagueConfig
from .draft import DraftState
from .players import Player


def _normalise_name(name: str) -> str:
    """Lowercase, strip suffixes like Jr./III/II, collapse whitespace."""
    name = name.lower().strip()
    name = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv|v)$", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


class ADPOpponent:
    """Opponent that drafts based on ADP with noise.

    Behavior:
    - 70% chance: take best available by noisy ADP
    - 30% chance: reach for a positional need
    - Always respects roster caps
    """

    def __init__(
        self,
        adp: dict[str, float],
        rng: np.random.Generator,
        noise_std: float = 3.0,
    ):
        # CR opus: noise_std=3.0 is in ADP units (pick positions). This means a player
        # with ADP 10 could be valued as ADP 7 or 13. In early rounds this is reasonable,
        # but in late rounds where ADP spread is wider, 3.0 picks of noise is too tight —
        # opponents become too predictable. Consider scaling noise_std by round.
        self.adp = adp
        self.rng = rng
        self.noise_std = noise_std

    def pick(self, state: DraftState, team_idx: int) -> Player:
        """Select a player for this opponent team."""
        eligible = [
            p for p in state.available
            if state.can_draft_position(team_idx, p.position)
        ]
        if not eligible:
            # CR opus: If state.available is also empty (all players drafted), this will
            # raise IndexError. Should check `state.available` is non-empty or raise a
            # clear error / return None.
            return state.available[0]  # fallback

        # CR opus: The 70/30 ADP-vs-need split is constant throughout the draft. In
        # reality, opponents reach for need more often in later rounds. Consider making
        # the need probability increase as the draft progresses.
        if self.rng.random() < 0.7:
            return self._adp_pick(eligible)
        else:
            return self._need_pick(state, team_idx, eligible)

    def _adp_pick(self, eligible: list[Player]) -> Player:
        """Pick best available by noisy ADP."""
        def noisy_adp(p: Player) -> float:
            base = self.adp.get(_normalise_name(p.name), 999.0)
            return base + self.rng.normal(0, self.noise_std)

        return min(eligible, key=noisy_adp)

    def _need_pick(self, state: DraftState, team_idx: int, eligible: list[Player]) -> Player:
        """Reach for a positional need."""
        needs = state.team_needs(team_idx)
        if not needs:
            return self._adp_pick(eligible)

        # Find highest-priority need (fewest filled starters)
        need_positions = set(needs.keys())
        # Flex needs can be filled by their eligible positions
        for slot_name, _count, eligible_pos in state.config.flex_slot_info():
            if slot_name in need_positions:
                need_positions.discard(slot_name)
                need_positions.update(eligible_pos)

        need_players = [p for p in eligible if p.position in need_positions]
        if not need_players:
            return self._adp_pick(eligible)

        # Pick best need player by noisy ADP
        def noisy_adp(p: Player) -> float:
            base = self.adp.get(_normalise_name(p.name), 999.0)
            return base + self.rng.normal(0, self.noise_std)

        return min(need_players, key=noisy_adp)
