"""Map between Lahman playerIDs and MLB Stats API person IDs.

The Lahman database uses string IDs like 'troutmi01' while the MLB Stats
API uses numeric IDs like 545361.  This module bridges the two by matching
on (first name, last name, birth date).
"""

import json
from pathlib import Path

import polars as pl

from . import loader
from .milb_api import CACHE_DIR, fetch_players_at_level, SPORT_IDS

MAP_PATH = CACHE_DIR / "id_map.json"


def _normalize_name(name: str) -> str:
    """Lowercase, strip accents/punctuation for fuzzy matching."""
    import unicodedata
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return name.lower().strip().replace(".", "").replace("'", "").replace("-", " ")


def build_id_map(seasons: range, verbose: bool = False) -> dict[str, int]:
    """Build a mapping from Lahman playerID to MLB API person ID.

    Matches on (normalized first name, last name, birth year, birth month).
    Birth day is used as a tiebreaker when available but not required for
    the initial match (some sources disagree on exact day).

    Returns:
        Dict mapping lahman_id -> mlb_api_id.
    """
    # Load Lahman people for the match
    people = loader.load_people()
    lahman = people.select([
        "playerID",
        pl.col("nameFirst").fill_null(""),
        pl.col("nameLast").fill_null(""),
        pl.col("birthYear").cast(pl.Int64, strict=False),
        pl.col("birthMonth").cast(pl.Int64, strict=False),
        pl.col("birthDay").cast(pl.Int64, strict=False),
    ]).to_dicts()

    # Build lookup: (normalized_first, normalized_last, birth_year, birth_month) -> playerID
    lahman_lookup: dict[tuple, list[dict]] = {}
    for row in lahman:
        key = (
            _normalize_name(row["nameFirst"]),
            _normalize_name(row["nameLast"]),
            row["birthYear"],
            row["birthMonth"],
        )
        lahman_lookup.setdefault(key, []).append(row)

    # Collect all MiLB API players across levels and seasons
    api_players: dict[int, dict] = {}  # api_id -> player info
    for level_name, sport_id in SPORT_IDS.items():
        if level_name == "MLB":
            continue
        for season in seasons:
            players = fetch_players_at_level(sport_id, season)
            for p in players:
                pid = p.get("id")
                if pid and pid not in api_players:
                    api_players[pid] = p

    # Also collect MLB-level players
    for season in seasons:
        try:
            players = fetch_players_at_level(1, season)
            for p in players:
                pid = p.get("id")
                if pid and pid not in api_players:
                    api_players[pid] = p
        except Exception:
            # CR opus: Bare `except Exception: pass` silently swallows HTTP errors,
            # CR opus: timeouts, and JSON decode failures. At minimum log a warning
            # CR opus: so callers know MLB-level players for a season were skipped.
            pass

    if verbose:
        print(f"  Lahman players: {len(lahman)}")
        print(f"  API players collected: {len(api_players)}")

    # Match
    id_map: dict[str, int] = {}
    matched = 0
    for api_id, p in api_players.items():
        full_name = p.get("fullName", "")
        # CR opus: split(maxsplit=1) puts the entire rest of the name into `last`.
        # CR opus: For players with suffixes like "Ronald Acuna Jr." this makes
        # CR opus: last="acuna jr" which won't match Lahman's nameLast="Acuna".
        # CR opus: Consider stripping known suffixes (Jr., Sr., II, III, IV).
        parts = full_name.split(maxsplit=1)
        first = _normalize_name(parts[0]) if parts else ""
        last = _normalize_name(parts[1]) if len(parts) > 1 else ""

        birth_date = p.get("birthDate", "")  # "YYYY-MM-DD"
        if not birth_date or len(birth_date) < 7:
            continue
        try:
            by = int(birth_date[:4])
            bm = int(birth_date[5:7])
        except (ValueError, IndexError):
            continue

        key = (first, last, by, bm)
        candidates = lahman_lookup.get(key, [])
        if len(candidates) == 1:
            lahman_id = candidates[0]["playerID"]
            id_map[lahman_id] = api_id
            matched += 1
        elif len(candidates) > 1:
            # Tiebreak on birth day
            bd_parts = birth_date.split("-")
            bd = int(bd_parts[2]) if len(bd_parts) == 3 else None
            for c in candidates:
                if bd and c.get("birthDay") == bd:
                    # CR opus: If multiple candidates match on birth day too, only
                    # CR opus: the first is mapped and the rest are silently skipped.
                    # CR opus: This can produce wrong mappings for common names.
                    id_map[c["playerID"]] = api_id
                    matched += 1
                    break
            else:
                # CR opus: Falling back to candidates[0] when the tiebreaker fails
                # CR opus: arbitrarily maps to the wrong player. Better to skip the
                # CR opus: match entirely and log a warning.
                # Just take the first candidate
                id_map[candidates[0]["playerID"]] = api_id
                matched += 1

    if verbose:
        print(f"  Matched: {matched}")

    return id_map


def save_id_map(id_map: dict[str, int]):
    """Save the ID mapping to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MAP_PATH.write_text(json.dumps(id_map, indent=1))


def load_id_map() -> dict[str, int]:
    """Load the ID mapping from disk."""
    if not MAP_PATH.exists():
        return {}
    return json.loads(MAP_PATH.read_text())


# CR opus: If build_id_map produces duplicate api_id values (two Lahman IDs
# CR opus: mapping to the same MLB API ID), this reverse map silently drops one.
# CR opus: Consider validating uniqueness of values in the forward map.
def get_reverse_map() -> dict[int, str]:
    """Return API ID -> Lahman ID mapping."""
    return {v: k for k, v in load_id_map().items()}
