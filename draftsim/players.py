"""Player pool loading from projections CSV."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


# Normalise team abbreviations so all code uses a single canonical form
_TEAM_NORMALIZE = {"LA": "LAR"}


@dataclass
class Player:
    """A projected player available for drafting."""

    name: str
    position: str  # QB, RB, WR, TE
    team: str
    projected_ppg: float
    projected_games: float
    projected_total: float
    pos_rank: int
    total_floor: float = 0.0   # 10th percentile season total
    total_ceiling: float = 0.0  # 90th percentile season total
    bye_week: int = 0
    sleeper_id: str = ""

    @property
    def upside(self) -> float:
        """Projection spread: ceiling - floor."""
        return self.total_ceiling - self.total_floor

    def __repr__(self) -> str:
        return f"{self.name} ({self.position}{self.pos_rank}, {self.team}) {self.projected_total:.0f}pts"


def load_players(
    projections_path: str | Path = "data/projections/all_projections.csv",
    min_total: float = 20.0,
) -> list[Player]:
    """Load player projections from CSV into Player objects.

    Args:
        projections_path: Path to all_projections.csv
        min_total: Minimum projected total points to include

    Returns:
        List of Player objects sorted by projected_total descending
    """
    players = []
    path = Path(projections_path)
    if not path.exists():
        raise FileNotFoundError(f"Projections file not found: {path}")

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total = float(row["projected_total"])
            if total < min_total:
                continue
            position = row["position_group"]
            if position not in ("QB", "RB", "WR", "TE", "K", "DST"):
                continue
            players.append(
                Player(
                    name=row["player_display_name"],
                    position=position,
                    team=_TEAM_NORMALIZE.get(row.get("current_team", ""), row.get("current_team", "")),
                    projected_ppg=float(row["projected_ppg"]),
                    projected_games=float(row["projected_games"]),
                    projected_total=total,
                    pos_rank=int(row["pos_rank"]),
                    total_floor=float(row.get("total_floor", 0)),
                    total_ceiling=float(row.get("total_ceiling", 0)),
                )
            )

    players.sort(key=lambda p: p.projected_total, reverse=True)

    # Assign bye weeks from ADP CSV
    from .adp import load_bye_weeks
    try:
        bye_weeks = load_bye_weeks()
        for p in players:
            p.bye_week = bye_weeks.get(p.team, 0)
    except FileNotFoundError:
        pass  # ADP file missing — leave bye_week as 0

    return players
