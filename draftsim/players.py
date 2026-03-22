"""Player pool loading from projections CSV."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


# Normalise team abbreviations so all code uses a single canonical form
# CR opus: Only "LA"->"LAR" is normalised here, but adp.py has a separate
# _TEAM_ALIASES = {"JAC": "JAX"}. These two maps are not shared, so if the
# projections CSV uses "JAC" it won't be normalised, and if the ADP CSV uses
# "LA" it won't be normalised either. Both maps should be unified.
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
    age: int | None = None
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

    # Assign ages from Lahman People.csv (MLB only)
    if sport == "mlb":
        try:
            _attach_mlb_ages(players)
        except FileNotFoundError:
            pass

    return players


def _attach_mlb_ages(players: list[Player]) -> None:
    """Attach age to MLB players from Lahman People.csv."""
    from datetime import date

    people_path = Path("data/lahman_1871-2025_csv/People.csv")
    if not people_path.exists():
        return

    # Build name -> (birthYear, birthMonth, birthDay) lookup
    name_to_birth: dict[str, tuple[int, int, int]] = {}
    with open(people_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by, bm, bd = row.get("birthYear", ""), row.get("birthMonth", ""), row.get("birthDay", "")
            if not by or not bm or not bd:
                continue
            try:
                first = row.get("nameFirst", "")
                last = row.get("nameLast", "")
                full = f"{first} {last}".strip()
                if full:
                    name_to_birth[full.lower()] = (int(by), int(bm), int(bd))
            except (ValueError, TypeError):
                continue

    today = date.today()
    for p in players:
        birth = name_to_birth.get(p.name.lower())
        if birth:
            by, bm, bd = birth
            try:
                bdate = date(by, bm, bd)
                age = (today - bdate).days // 365
                p.age = age
            except ValueError:
                continue
