"""ADP-based opponent modeling for draft simulation."""

import numpy as np

from .config import LeagueConfig
from .draft import DraftState
from .players import Player


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

        if self.rng.random() < 0.7:
            return self._adp_pick(eligible)
        else:
            return self._need_pick(state, team_idx, eligible)

    def _adp_pick(self, eligible: list[Player]) -> Player:
        """Pick best available by noisy ADP."""
        # CR opus: Players missing from ADP (base=999.0) get a noisy value around 999,
        # making them essentially never picked. This is fine for known players but means
        # any projection-only player (not in FantasyPros CSV) is invisible to opponents,
        # potentially over-valuing them for the user since opponents will never draft them.
        def noisy_adp(p: Player) -> float:
            base = self.adp.get(p.name, 999.0)
            return base + self.rng.normal(0, self.noise_std)

        return min(eligible, key=noisy_adp)

    def _need_pick(self, state: DraftState, team_idx: int, eligible: list[Player]) -> Player:
        """Reach for a positional need."""
        needs = state.team_needs(team_idx)
        if not needs:
            return self._adp_pick(eligible)

        # Find highest-priority need (fewest filled starters)
        need_positions = set(needs.keys())
        # FLEX need can be filled by RB/WR/TE
        if "FLEX" in need_positions:
            need_positions.discard("FLEX")
            need_positions.update(state.config.flex_positions())

        need_players = [p for p in eligible if p.position in need_positions]
        if not need_players:
            return self._adp_pick(eligible)

        # Pick best need player by noisy ADP
        def noisy_adp(p: Player) -> float:
            base = self.adp.get(p.name, 999.0)
            return base + self.rng.normal(0, self.noise_std)

        return min(need_players, key=noisy_adp)
