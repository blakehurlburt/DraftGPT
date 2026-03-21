"""Convert Sleeper raw stat projections to fantasy points."""

from __future__ import annotations


def default_ppr_scoring() -> dict[str, float]:
    """Standard PPR scoring multipliers."""
    return {
        "pass_yd": 0.04,
        "pass_td": 4.0,
        "pass_int": -1.0,
        "pass_2pt": 2.0,
        "rush_yd": 0.1,
        "rush_td": 6.0,
        "rush_2pt": 2.0,
        "rec": 1.0,
        "rec_yd": 0.1,
        "rec_td": 6.0,
        "rec_2pt": 2.0,
        "fum_lost": -2.0,
        # Kicker
        "fgm": 3.0,
        "fgmiss": -1.0,
        "xpm": 1.0,
        "xpmiss": -1.0,
        # DST
        "sack": 1.0,
        "int": 2.0,
        "fum_rec": 2.0,
        "def_td": 6.0,
        "safe": 2.0,
        "blk_kick": 2.0,
    }


def extract_scoring_from_meta(meta: dict) -> dict[str, float] | None:
    """Pull scoring_settings from Sleeper draft metadata.

    Returns None if settings are unavailable (fall back to default PPR).
    """
    settings = meta.get("settings", {})
    scoring = settings.get("scoring_settings")
    if not scoring or not isinstance(scoring, dict):
        return None
    # Sleeper scoring_settings keys map directly to stat names
    # CR opus: `if v` filters out scoring multipliers with value 0 or 0.0, but a league
    # may intentionally set a category to 0 points (e.g., rec=0 for standard scoring).
    # Filtering these out causes the default PPR value to be used for that category
    # instead. Should use `if v is not None` instead of `if v`.
    return {k: float(v) for k, v in scoring.items() if v}


def sleeper_stats_to_fantasy_points(
    stats: dict,
    scoring: dict[str, float] | None = None,
) -> float:
    """Convert a Sleeper stat projection dict to total fantasy points.

    Args:
        stats: Raw stat dict from Sleeper projections API.
        scoring: Scoring multipliers (from league settings or default PPR).

    Returns:
        Total projected fantasy points for the season.
    """
    if scoring is None:
        scoring = default_ppr_scoring()

    total = 0.0
    for stat_key, multiplier in scoring.items():
        val = stats.get(stat_key)
        if val is not None:
            try:
                total += float(val) * multiplier
            except (ValueError, TypeError):
                pass
    return total
