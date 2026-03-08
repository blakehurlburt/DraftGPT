"""Fantasy baseball scoring systems.

Provides configurable point values for converting raw stats to fantasy points.
Default is a standard ESPN-style points league.
"""

# Standard ESPN points league scoring
BATTER_POINTS = {
    "R": 1,
    "HR": 4,
    "RBI": 1,
    "SB": 2,
    "CS": -1,
    "BB": 1,
    "HBP": 1,
    "H": 1,       # singles, doubles, triples, HRs each get base hit point
    "2B": 1,      # extra point for doubles (total 2 per double)
    "3B": 2,      # extra points for triples (total 3 per triple)
    # HR extra point handled via HR weight above
    "SO": -0.5,
    "GIDP": -0.5,
}

PITCHER_POINTS = {
    "W": 5,
    "L": -3,
    "SV": 5,
    "SO": 1,       # strikeouts by pitcher
    "IPouts": 1,   # 1 point per out recorded (3 per IP)
    "ER": -2,
    "H": -0.5,     # hits allowed
    "BB": -1,      # walks allowed
    "HBP": -0.5,
    "CG": 3,
    "SHO": 3,      # bonus on top of CG
}


def compute_batter_fpts(row: dict, scoring: dict = None) -> float:
    """Compute fantasy points for a batter season from a dict of raw stats."""
    s = scoring or BATTER_POINTS
    pts = 0.0
    pts += row.get("R", 0) * s.get("R", 0)
    pts += row.get("HR", 0) * s.get("HR", 0)
    pts += row.get("RBI", 0) * s.get("RBI", 0)
    pts += row.get("SB", 0) * s.get("SB", 0)
    pts += row.get("CS", 0) * s.get("CS", 0)
    pts += row.get("BB", 0) * s.get("BB", 0)
    pts += row.get("HBP", 0) * s.get("HBP", 0)
    pts += row.get("H", 0) * s.get("H", 0)
    # Extra-base hit bonuses (on top of base H point)
    pts += row.get("2B", 0) * s.get("2B", 0)
    pts += row.get("3B", 0) * s.get("3B", 0)
    pts += row.get("SO", 0) * s.get("SO", 0)
    pts += row.get("GIDP", 0) * s.get("GIDP", 0)
    return pts


def compute_pitcher_fpts(row: dict, scoring: dict = None) -> float:
    """Compute fantasy points for a pitcher season from a dict of raw stats."""
    s = scoring or PITCHER_POINTS
    pts = 0.0
    pts += row.get("W", 0) * s.get("W", 0)
    pts += row.get("L", 0) * s.get("L", 0)
    pts += row.get("SV", 0) * s.get("SV", 0)
    pts += row.get("SO", 0) * s.get("SO", 0)
    pts += row.get("IPouts", 0) * s.get("IPouts", 0)
    pts += row.get("ER", 0) * s.get("ER", 0)
    pts += row.get("H", 0) * s.get("H", 0)
    pts += row.get("BB", 0) * s.get("BB", 0)
    pts += row.get("HBP", 0) * s.get("HBP", 0)
    pts += row.get("CG", 0) * s.get("CG", 0)
    pts += row.get("SHO", 0) * s.get("SHO", 0)
    return pts
