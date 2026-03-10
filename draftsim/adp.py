"""ADP data from FantasyPros aggregated rankings.

Loads real ADP data from a FantasyPros CSV export containing per-platform
columns (ESPN, Sleeper, CBS, NFL, RTSports, Fantrax) plus an AVG consensus.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional

from .players import Player

ADP_CSV = Path(__file__).parent.parent / "data" / "FantasyPros_2025_Overall_ADP_Rankings.csv"

# Map CSV column names to our platform keys
PLATFORM_COLUMNS = {
    "consensus": "AVG",
    "espn": "ESPN",
    "sleeper": "Sleeper",
    "cbs": "CBS",
    "nfl": "NFL",
    "rtsports": "RTSports",
    "fantrax": "Fantrax",
}

# Platforms exposed in the UI
PLATFORMS = {"sleeper", "espn", "consensus"}

_POS_RE = re.compile(r"^([A-Z]+)\d*$")


def _parse_pos(raw: str) -> str:
    """Strip positional rank suffix: 'WR3' -> 'WR'."""
    m = _POS_RE.match(raw.strip())
    return m.group(1) if m else raw.strip()


def _normalise(name: str) -> str:
    """Lowercase, strip suffixes like Jr./III/II, collapse whitespace."""
    name = name.lower().strip()
    name = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv|v)$", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def _load_csv(path: Path | None = None) -> list[dict]:
    """Read the FantasyPros CSV and return list of row dicts."""
    path = path or ADP_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"ADP file not found: {path}. "
            "Download from FantasyPros and place in data/."
        )
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip empty/trailing rows
            if not row.get("Player"):
                continue
            rows.append(row)
    return rows


def _build_adp_lookup(csv_rows: list[dict], column: str) -> dict[str, float]:
    """Build normalised-name -> ADP float from a specific column."""
    lookup: dict[str, float] = {}
    for row in csv_rows:
        name = _normalise(row.get("Player", ""))
        if not name:
            continue
        raw = row.get(column, "").strip()
        if not raw:
            continue
        try:
            lookup[name] = float(raw)
        except ValueError:
            continue
    return lookup


def load_adp_for_platform(
    players: list[Player],
    platform: str = "consensus",
    csv_path: Path | None = None,
) -> list[tuple[Player, float]]:
    """Load real ADP for a platform, matched against our player list.

    Args:
        players: Our player list (from projections)
        platform: One of 'consensus', 'espn', 'sleeper', 'cbs', 'nfl', etc.
        csv_path: Override path to the FantasyPros CSV

    Returns:
        List of (player, adp) sorted by ADP ascending.
        Players not found in ADP data get adp = 999.0.
    """
    col = PLATFORM_COLUMNS.get(platform)
    if col is None:
        raise ValueError(
            f"Unknown platform '{platform}'. "
            f"Choose from: {sorted(PLATFORM_COLUMNS.keys())}"
        )

    csv_rows = _load_csv(csv_path)
    lookup = _build_adp_lookup(csv_rows, col)

    # Also build a team+pos lookup for disambiguation
    pos_team_lookup: dict[tuple[str, str], float] = {}
    for row in csv_rows:
        name = _normalise(row.get("Player", ""))
        pos = _parse_pos(row.get("POS", ""))
        team = row.get("Team", "").strip()
        raw = row.get(col, "").strip()
        if name and raw:
            try:
                pos_team_lookup[(name, pos, team)] = float(raw)
            except ValueError:
                pass

    result = []
    for p in players:
        norm = _normalise(p.name)
        # Try exact normalised match first
        adp = lookup.get(norm)
        # Try with position+team for disambiguation
        if adp is None:
            adp = pos_team_lookup.get((norm, p.position, p.team))
        # Fallback: unranked
        if adp is None:
            adp = 999.0
        result.append((p, adp))

    result.sort(key=lambda x: x[1])
    return result


# ── Legacy-compatible API (used by app.py and simulate.py) ──────────────

def generate_consensus_adp(
    players: list[Player],
    rng=None,  # ignored — kept for API compat
) -> list[tuple[Player, float]]:
    """Return consensus ADP from FantasyPros AVG column."""
    return load_adp_for_platform(players, "consensus")


def generate_platform_adp(
    players: list[Player],
    platform: str,
    rng=None,  # ignored — kept for API compat
) -> list[tuple[Player, float]]:
    """Return platform-specific ADP from FantasyPros."""
    return load_adp_for_platform(players, platform)


# Normalise ADP team abbreviations to match projections CSV
_TEAM_ALIASES = {"JAC": "JAX", "LAR": "LA"}


def load_bye_weeks(csv_path: Path | None = None) -> dict[str, int]:
    """Return {team_abbr: bye_week} from ADP CSV."""
    csv_rows = _load_csv(csv_path)
    team_bye: dict[str, int] = {}
    for row in csv_rows:
        team = _TEAM_ALIASES.get(row.get("Team", "").strip(), row.get("Team", "").strip())
        bye = row.get("Bye", "").strip()
        if team and bye and team not in team_bye:
            try:
                team_bye[team] = int(bye)
            except ValueError:
                continue
    return team_bye


def load_adp(
    platform: str = "consensus",
    adp_dir: str | Path = "data/adp",
) -> dict[str, float]:
    """Load ADP into a name->adp dict (legacy interface).

    Note: adp_dir is ignored; data comes from FantasyPros CSV.
    """
    csv_rows = _load_csv()
    col = PLATFORM_COLUMNS.get(platform)
    if col is None:
        raise ValueError(f"Unknown platform '{platform}'")

    result: dict[str, float] = {}
    for row in csv_rows:
        name = row.get("Player", "").strip()
        raw = row.get(col, "").strip()
        if name and raw:
            try:
                result[name] = float(raw)
            except ValueError:
                continue
    return result
