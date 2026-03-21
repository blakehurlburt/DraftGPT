"""Convert Sleeper raw stat projections to fantasy points."""

from __future__ import annotations


def default_ppr_scoring() -> dict[str, float]:
    """Standard Sleeper PPR scoring multipliers."""
    return {
        # Passing
        "pass_yd": 0.04,       # 1 point per 25 yards
        "pass_td": 4.0,
        "pass_int": -2.0,
        "pass_2pt": 2.0,
        # Rushing
        "rush_yd": 0.1,
        "rush_td": 6.0,
        "rush_2pt": 2.0,
        # Receiving
        "rec": 1.0,            # PPR
        "rec_yd": 0.1,
        "rec_td": 6.0,
        "rec_2pt": 2.0,
        # Fumbles
        "fum_lost": -2.0,
        # Kicker — distance-based FG scoring
        "fgm_0_39": 3.0,      # computed: fgm - fgm_40_49 - fgm_50p
        "fgm_40_49": 4.0,
        "fgm_50p": 5.0,
        "xpm": 1.0,
        "xpmiss": -1.0,
        # DST
        "sack": 1.0,
        "int": 2.0,
        "fum_rec": 2.0,
        "def_td": 6.0,
        "def_kr_td": 6.0,     # kick return TD
        "pr_td": 6.0,         # punt return TD
        "safe": 2.0,
        "blk_kick": 2.0,
        "pts_allow_0": 10.0,  # shutout bonus (flag=1 when 0 pts allowed)
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

    # --- Kicker FG distance handling ---
    # Sleeper doesn't provide fgm_0_39 directly. Compute it when distance
    # breakdowns are available so the scoring dict can price it correctly.
    if "fgm_0_39" not in stats and "fgm_0_39" in scoring:
        fgm_40_49 = float(stats.get("fgm_40_49", 0))
        fgm_50p = float(stats.get("fgm_50p", 0))
        fgm_total = stats.get("fgm")

        if fgm_total is not None:
            # Total FG exists — derive short-range FGs by subtraction
            fgm_0_39 = float(fgm_total) - fgm_40_49 - fgm_50p
            if fgm_0_39 > 0:
                stats = {**stats, "fgm_0_39": fgm_0_39}
        elif fgm_40_49 or fgm_50p:
            # Distance breakdowns exist but no total — short-range FGs unknown,
            # only score the distance buckets we have (no fgm_0_39 injected).
            pass
        # If neither total nor breakdowns exist, nothing to score.

    total = 0.0
    for stat_key, multiplier in scoring.items():
        val = stats.get(stat_key)
        if val is not None:
            try:
                total += float(val) * multiplier
            except (ValueError, TypeError):
                pass
    return total
