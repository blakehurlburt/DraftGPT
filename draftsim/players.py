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
    is_rookie: bool = False
    sleeper_id: str = ""
    # Sleeper projection values (populated at connect time)
    sleeper_projected_total: float = 0.0
    sleeper_projected_ppg: float = 0.0
    sleeper_projected_games: float = 0.0
    # FanGraphs projection values (populated for MLB drafts)
    fangraphs_projected_total: float = 0.0
    fangraphs_projected_ppg: float = 0.0
    fangraphs_projected_games: float = 0.0
    # Model backup (saved before first swap, used to restore)
    _model_projected_total: float = 0.0
    _model_projected_ppg: float = 0.0
    _model_projected_games: float = 0.0
    _model_total_floor: float = 0.0
    _model_total_ceiling: float = 0.0

    @property
    def upside(self) -> float:
        """Projection spread: ceiling - floor."""
        return self.total_ceiling - self.total_floor

    def __repr__(self) -> str:
        return f"{self.name} ({self.position}{self.pos_rank}, {self.team}) {self.projected_total:.0f}pts"


_NFL_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DST"}
_MLB_POSITIONS = {"C", "1B", "2B", "3B", "SS", "OF", "SP", "RP", "DH"}


def load_players(
    projections_path: str | Path | None = None,
    min_total: float = 20.0,
    sport: str = "nfl",
) -> list[Player]:
    """Load player projections from CSV into Player objects.

    Args:
        projections_path: Path to projections CSV. Defaults per sport.
        min_total: Minimum projected total points to include
        sport: "nfl" or "mlb"

    Returns:
        List of Player objects sorted by projected_total descending
    """
    if projections_path is None:
        if sport == "mlb":
            projections_path = "data/projections/mlb_projections.csv"
        else:
            projections_path = "data/projections/all_projections.csv"

    valid_positions = _MLB_POSITIONS if sport == "mlb" else _NFL_POSITIONS

    players = []
    path = Path(projections_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Projections file not found: {path}. "
            + ("Run scripts/train_mlb.py to generate MLB projections." if sport == "mlb" else "")
        )

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total = float(row["projected_total"])
            if total < min_total:
                continue
            position = row["position_group"]
            if position not in valid_positions:
                continue
            raw_team = row.get("current_team", "") or row.get("team", "") or ""
            team = _TEAM_NORMALIZE.get(raw_team, raw_team)
            players.append(
                Player(
                    name=row["player_display_name"],
                    position=position,
                    team=team,
                    projected_ppg=float(row["projected_ppg"]),
                    projected_games=float(row["projected_games"]),
                    projected_total=total,
                    pos_rank=int(row["pos_rank"]),
                    total_floor=float(row.get("total_floor", 0)),
                    total_ceiling=float(row.get("total_ceiling", 0)),
                    # CR opus: `not raw_team` flags a player as a rookie when their team is
                    # empty/missing. This is a fragile heuristic — an empty team string could
                    # indicate bad data rather than rookie status, and actual rookies on known
                    # teams will be missed. Consider using an explicit "is_rookie" CSV column.
                    is_rookie=not raw_team if sport == "nfl" else False,
                )
            )

    players.sort(key=lambda p: p.projected_total, reverse=True)

    # Assign bye weeks from ADP CSV (NFL only)
    if sport == "nfl":
        from .adp import load_bye_weeks
        try:
            bye_weeks = load_bye_weeks()
            for p in players:
                p.bye_week = bye_weeks.get(p.team, 0)
        except FileNotFoundError:
            pass  # ADP file missing — leave bye_week as 0

    return players
