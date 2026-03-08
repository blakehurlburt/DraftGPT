"""Data loading functions for MLB analysis using the Lahman database.

Reads CSV files from data/lahman_1871-2025_csv/ and returns Polars DataFrames.
"""

import polars as pl
from pathlib import Path
import zipfile

DATA_DIR = Path(__file__).parent.parent / "data" / "lahman_1871-2025_csv"
ZIP_PATH = Path(__file__).parent.parent / "data" / "lahman_1871-2025_csv.zip"

# Lahman uses historical 3-char codes; map to modern abbreviations
TEAM_CODE_MAP = {
    "ANA": "LAA", "CAL": "LAA", "MON": "WSN", "FLO": "MIA",
    "TBA": "TB",  "KCA": "KC",  "SLN": "STL", "SFN": "SF",
    "NYN": "NYM", "NYA": "NYY", "LAN": "LAD", "SDN": "SD",
    "CHN": "CHC", "CHA": "CWS", "WAS": "WSN", "ATH": "OAK",
    "OAK": "OAK", "SEA": "SEA", "MIN": "MIN", "CLE": "CLE",
    "DET": "DET", "BOS": "BOS", "BAL": "BAL", "TOR": "TOR",
    "TEX": "TEX", "HOU": "HOU", "COL": "COL", "ARI": "ARI",
    "CIN": "CIN", "PIT": "PIT", "MIL": "MIL", "ATL": "ATL",
    "PHI": "PHI", "MIA": "MIA", "TB": "TB",   "KC": "KC",
    "STL": "STL", "SF": "SF",   "NYM": "NYM", "NYY": "NYY",
    "LAD": "LAD", "SD": "SD",   "CHC": "CHC", "CWS": "CWS",
    "WSN": "WSN", "LAA": "LAA",
}


def normalize_team_code(code: str) -> str:
    """Convert a Lahman team code to a modern abbreviation."""
    return TEAM_CODE_MAP.get(code, code)


def _ensure_extracted():
    """Extract the Lahman zip if the CSV directory doesn't exist."""
    if DATA_DIR.exists():
        return
    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Lahman data not found. Expected zip at {ZIP_PATH}"
        )
    print(f"Extracting {ZIP_PATH.name}...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR.parent)


def _read_csv(filename: str) -> pl.DataFrame:
    """Read a Lahman CSV file, extracting from zip if needed."""
    _ensure_extracted()
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Lahman data not found at {path}")
    return pl.read_csv(path, infer_schema_length=10000)


def load_batting(min_year: int = 2000) -> pl.DataFrame:
    """Load batting stats, combining stints for traded players."""
    df = _read_csv("Batting.csv")
    df = df.filter(pl.col("yearID") >= min_year)

    # Players traded mid-season have multiple stint rows — aggregate them
    id_cols = ["playerID", "yearID"]
    # For teamID, keep the last stint (end-of-season team)
    sum_cols = [
        "G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS",
        "BB", "SO", "IBB", "HBP", "SH", "SF", "GIDP",
    ]
    available_sum = [c for c in sum_cols if c in df.columns]

    df = (
        df.sort(["playerID", "yearID", "stint"])
        .group_by(id_cols)
        .agg(
            [pl.col("teamID").last().alias("teamID"),
             pl.col("lgID").last().alias("lgID")]
            + [pl.col(c).sum() for c in available_sum]
        )
    )
    # Normalize team codes to modern abbreviations
    df = df.with_columns(
        pl.col("teamID").replace_strict(TEAM_CODE_MAP, default=pl.col("teamID"))
    )
    return df


def load_pitching(min_year: int = 2000) -> pl.DataFrame:
    """Load pitching stats, combining stints for traded players."""
    df = _read_csv("Pitching.csv")
    df = df.filter(pl.col("yearID") >= min_year)

    id_cols = ["playerID", "yearID"]
    sum_cols = [
        "W", "L", "G", "GS", "CG", "SHO", "SV", "IPouts", "H", "ER",
        "HR", "BB", "SO", "IBB", "WP", "HBP", "BK", "BFP", "GF", "R",
        "SH", "SF", "GIDP",
    ]
    available_sum = [c for c in sum_cols if c in df.columns]

    # ERA and BAOpp need to be recomputed after aggregation, not summed
    df = (
        df.sort(["playerID", "yearID", "stint"])
        .group_by(id_cols)
        .agg(
            [pl.col("teamID").last().alias("teamID"),
             pl.col("lgID").last().alias("lgID")]
            + [pl.col(c).sum() for c in available_sum]
        )
    )

    # Recompute ERA: 9 * ER / IP (IPouts / 3 = IP)
    df = df.with_columns(
        pl.when(pl.col("IPouts") > 0)
        .then(9.0 * pl.col("ER") / (pl.col("IPouts") / 3.0))
        .otherwise(None)
        .alias("ERA")
    )
    # Normalize team codes to modern abbreviations
    df = df.with_columns(
        pl.col("teamID").replace_strict(TEAM_CODE_MAP, default=pl.col("teamID"))
    )
    return df


def load_people() -> pl.DataFrame:
    """Load player biographical data (name, birth date, height, weight, etc.)."""
    df = _read_csv("People.csv")
    # Compute birth_date from components
    df = df.with_columns(
        pl.when(
            pl.col("birthYear").is_not_null()
            & pl.col("birthMonth").is_not_null()
            & pl.col("birthDay").is_not_null()
        )
        .then(
            pl.date(pl.col("birthYear"), pl.col("birthMonth"), pl.col("birthDay"))
        )
        .otherwise(None)
        .alias("birth_date")
    )
    return df


def load_appearances(min_year: int = 2000) -> pl.DataFrame:
    """Load appearances data (games by position)."""
    df = _read_csv("Appearances.csv")
    return df.filter(pl.col("yearID") >= min_year)


def load_teams(min_year: int = 2000) -> pl.DataFrame:
    """Load team-level stats including park factors."""
    df = _read_csv("Teams.csv")
    df = df.filter(pl.col("yearID") >= min_year)
    df = df.with_columns(
        pl.col("teamID").replace_strict(TEAM_CODE_MAP, default=pl.col("teamID"))
    )
    return df


def load_fielding(min_year: int = 2000) -> pl.DataFrame:
    """Load fielding stats."""
    df = _read_csv("Fielding.csv")
    return df.filter(pl.col("yearID") >= min_year)


def load_salaries(min_year: int = 2000) -> pl.DataFrame:
    """Load salary data."""
    df = _read_csv("Salaries.csv")
    return df.filter(pl.col("yearID") >= min_year)
