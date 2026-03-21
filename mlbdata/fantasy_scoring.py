"""Fantasy baseball scoring systems.

Provides configurable point values for converting raw stats to fantasy points.
Default is Yahoo points league scoring.

Batter scoring uses explicit 1B/2B/3B/HR weights (NOT an "H" catchall)
so home runs aren't double-counted.
"""

# Yahoo points league scoring — batters
# Uses explicit hit types: 1B (singles), 2B, 3B, HR
BATTER_POINTS = {
    "1B": 2.6,
    "2B": 5.2,
    "3B": 7.8,
    "HR": 10.4,
    "R": 1.9,
    "RBI": 1.9,
    "BB": 2.6,
    "SB": 4.2,
    "HBP": 2.6,
}

# Yahoo points league scoring — pitchers
PITCHER_POINTS = {
    "SV": 8,
    "W": 8,
    "SO": 3,       # strikeouts by pitcher
    "ER": -3,
    "IPouts": 1,   # 1 point per out recorded (3 per IP)
    "BB": -1.3,
    "H": -1.3,     # hits allowed
    "HBP": -1.3,
}


def compute_batter_fpts(row: dict, scoring: dict = None) -> float:
    """Compute fantasy points for a batter season from a dict of raw stats.

    Computes singles (1B) as H - 2B - 3B - HR to avoid double-counting.
    """
    s = scoring or BATTER_POINTS
    pts = 0.0

    # Compute singles from hits minus extra-base hits
    h = row.get("H", 0)
    doubles = row.get("2B", 0)
    triples = row.get("3B", 0)
    hr = row.get("HR", 0)
    singles = h - doubles - triples - hr

    pts += singles * s.get("1B", 0)
    pts += doubles * s.get("2B", 0)
    pts += triples * s.get("3B", 0)
    pts += hr * s.get("HR", 0)
    pts += row.get("R", 0) * s.get("R", 0)
    pts += row.get("RBI", 0) * s.get("RBI", 0)
    pts += row.get("BB", 0) * s.get("BB", 0)
    pts += row.get("SB", 0) * s.get("SB", 0)
    pts += row.get("HBP", 0) * s.get("HBP", 0)
    pts += row.get("CS", 0) * s.get("CS", 0)
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
