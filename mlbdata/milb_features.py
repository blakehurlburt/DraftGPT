"""MiLB feature engineering for MLB projection models.

Computes minor-league features from cached API data, keyed by
(player_id, season) where player_id is a Lahman ID and season is the
MLB season being *predicted*.  All MiLB data used for a given row is
strictly from years before that season (walk-forward safe).
"""

import json
from pathlib import Path

import polars as pl

from .milb_api import CACHE_DIR, LEVEL_RANK, SPORT_ID_TO_LEVEL
from .id_mapping import load_id_map, get_reverse_map
from .fantasy_scoring import compute_batter_fpts, compute_pitcher_fpts

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_cached_stats(group: str = "hitting") -> dict[int, list[dict]]:
    """Load all cached player stats from disk.

    Returns:
        Dict mapping mlb_api_id -> list of stat splits.
    """
    stats_dir = CACHE_DIR / "stats"
    if not stats_dir.exists():
        return {}

    result: dict[int, list[dict]] = {}
    suffix = f"_{group}.json"
    for path in stats_dir.iterdir():
        if not path.name.endswith(suffix):
            continue
        try:
            api_id = int(path.name.replace(suffix, ""))
        except ValueError:
            continue
        try:
            splits = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if splits:
            result[api_id] = splits

    return result


def _parse_milb_splits(
    splits: list[dict],
    before_year: int,
    group: str = "hitting",
) -> list[dict]:
    """Extract MiLB-level stat lines from raw API splits.

    Only includes seasons strictly before `before_year` and at
    non-MLB levels.
    """
    rows = []
    for s in splits:
        season_str = s.get("season")
        if not season_str:
            continue
        try:
            season = int(season_str)
        except (ValueError, TypeError):
            continue

        if season >= before_year:
            continue

        # Determine the level
        sport_id = s.get("_sport_id")
        level = s.get("_level") or SPORT_ID_TO_LEVEL.get(sport_id, "")
        if not level or level == "MLB":
            continue

        stat = s.get("stat", {})
        if not stat:
            continue

        # Require meaningful playing time
        games = _int(stat.get("gamesPlayed", 0))
        if games < 10:
            continue

        row = {
            "season": season,
            "level": level,
            "level_rank": LEVEL_RANK.get(level, 0),
            "games": games,
        }

        if group == "hitting":
            ab = _int(stat.get("atBats", 0))
            sf = _int(stat.get("sacFlies", 0))
            sh = _int(stat.get("sacBunts", 0))
            pa = _int(stat.get("plateAppearances", 0)) or (ab + _int(stat.get("baseOnBalls", 0)) + _int(stat.get("hitByPitch", 0)) + sf + sh)
            h = _int(stat.get("hits", 0))
            hr = _int(stat.get("homeRuns", 0))
            doubles = _int(stat.get("doubles", 0))
            triples = _int(stat.get("triples", 0))
            bb = _int(stat.get("baseOnBalls", 0))
            so = _int(stat.get("strikeOuts", 0))
            sb = _int(stat.get("stolenBases", 0))
            cs = _int(stat.get("caughtStealing", 0))
            rbi = _int(stat.get("rbi", 0))
            runs = _int(stat.get("runs", 0))
            hbp = _int(stat.get("hitByPitch", 0))
            gidp = _int(stat.get("groundIntoDoublePlay", 0))

            row.update({
                "AB": ab, "PA": pa, "H": h, "HR": hr, "2B": doubles,
                "3B": triples, "BB": bb, "SO": so, "SB": sb, "CS": cs,
                "RBI": rbi, "R": runs, "HBP": hbp, "GIDP": gidp,
                "AVG": h / ab if ab > 0 else 0.0,
                "OBP": (h + bb + hbp) / pa if pa > 0 else 0.0,
                "SLG": ((h - doubles - triples - hr) + 2*doubles + 3*triples + 4*hr) / ab if ab > 0 else 0.0,
                "K_rate": so / pa if pa > 0 else 0.0,
                "BB_rate": bb / pa if pa > 0 else 0.0,
            })
            row["ISO"] = row["SLG"] - row["AVG"]
            row["OPS"] = row["OBP"] + row["SLG"]
            fpts = compute_batter_fpts(row)
            row["fpts"] = fpts
            row["ppg"] = fpts / games if games > 0 else 0.0

        elif group == "pitching":
            ipouts = _int(stat.get("outs", 0))
            ip = ipouts / 3.0 if ipouts > 0 else _float(stat.get("inningsPitched", "0"))
            w = _int(stat.get("wins", 0))
            l_ = _int(stat.get("losses", 0))
            sv = _int(stat.get("saves", 0))
            so = _int(stat.get("strikeOuts", 0))
            bb = _int(stat.get("baseOnBalls", 0))
            h = _int(stat.get("hits", 0))
            er = _int(stat.get("earnedRuns", 0))
            hr = _int(stat.get("homeRuns", 0))
            hbp = _int(stat.get("hitByPitch", 0))
            gs = _int(stat.get("gamesStarted", 0))
            cg = _int(stat.get("completeGames", 0))
            sho = _int(stat.get("shutouts", 0))

            row.update({
                "IP": ip, "IPouts": ipouts, "W": w, "L": l_, "SV": sv,
                "SO": so, "BB": bb, "H": h, "ER": er, "HR": hr,
                "HBP": hbp, "GS": gs, "CG": cg, "SHO": sho,
                "ERA": 9.0 * er / ip if ip > 0 else 0.0,
                "WHIP": (bb + h) / ip if ip > 0 else 0.0,
                "K9": 9.0 * so / ip if ip > 0 else 0.0,
                "BB9": 9.0 * bb / ip if ip > 0 else 0.0,
                "FIP": (13*hr + 3*(bb + hbp) - 2*so) / ip + 3.2 if ip > 0 else 0.0,
            })
            fpts = compute_pitcher_fpts(row)
            row["fpts"] = fpts
            row["ppg"] = fpts / games if games > 0 else 0.0

        rows.append(row)

    return rows


def _int(v) -> int:
    try:
        return int(v) if v is not None else 0
    except (ValueError, TypeError):
        return 0


def _float(v) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (ValueError, TypeError):
        return 0.0


def _weighted_avg(rows: list[dict], key: str, weight_key: str = "PA") -> float:
    """Compute weighted average of a stat across rows."""
    total_w = sum(r.get(weight_key, 0) for r in rows)
    if total_w == 0:
        return 0.0
    # CR opus: Default weight_key is "PA" but pitcher features call this with
    # CR opus: weight_key="IP". If a pitcher row is missing "IP" (e.g. due to
    # CR opus: a parsing error), it silently gets weight=0 and is excluded.
    return sum(r.get(key, 0) * r.get(weight_key, 0) for r in rows) / total_w


# ---------------------------------------------------------------------------
# Batter features
# ---------------------------------------------------------------------------

def _compute_batter_milb_features(splits: list[dict], before_year: int) -> dict | None:
    """Compute MiLB features for a single batter from their stat splits.

    Only uses data from seasons < before_year.
    """
    rows = _parse_milb_splits(splits, before_year, group="hitting")
    if not rows:
        return None

    # Sort by level rank desc, then season desc
    rows.sort(key=lambda r: (r["level_rank"], r["season"]), reverse=True)

    highest_rank = rows[0]["level_rank"]
    highest_level = rows[0]["level"]

    # CR opus: This computes the earliest season at the highest level, but calls
    # CR opus: it "age at highest level" in the comment — it's actually just the
    # CR opus: year. It's used for milb_level_jump_recency = before_year - earliest,
    # CR opus: which measures how many years ago the player first reached this level.
    # CR opus: But without birth year, this is not actually "age" — it's tenure.
    # Find age at highest level (earliest season at that level)
    earliest_at_highest = min(
        (r["season"] for r in rows if r["level_rank"] == highest_rank),
        default=before_year,
    )

    # Stats at the highest level (most recent 2 seasons)
    top_rows = [r for r in rows if r["level_rank"] == highest_rank][:2]
    # Stats at second-highest level for trend computation
    second_rank = highest_rank - 1
    second_rows = [r for r in rows if r["level_rank"] == second_rank][:2]

    total_pa = sum(r.get("PA", 0) for r in top_rows)
    weight_key = "PA"

    features = {
        "milb_highest_level": highest_rank,
        "milb_seasons_total": len(set(r["season"] for r in rows)),
        "milb_pa": total_pa,
        "milb_avg": _weighted_avg(top_rows, "AVG", weight_key),
        "milb_obp": _weighted_avg(top_rows, "OBP", weight_key),
        "milb_slg": _weighted_avg(top_rows, "SLG", weight_key),
        "milb_iso": _weighted_avg(top_rows, "ISO", weight_key),
        "milb_ops": _weighted_avg(top_rows, "OPS", weight_key),
        "milb_k_rate": _weighted_avg(top_rows, "K_rate", weight_key),
        "milb_bb_rate": _weighted_avg(top_rows, "BB_rate", weight_key),
        "milb_hr_rate": (
            sum(r.get("HR", 0) for r in top_rows) / total_pa
            if total_pa > 0 else 0.0
        ),
        "milb_sb_pg": (
            sum(r.get("SB", 0) for r in top_rows)
            / sum(r.get("games", 0) for r in top_rows)
            if sum(r.get("games", 0) for r in top_rows) > 0 else 0.0
        ),
        "milb_ppg": _weighted_avg(top_rows, "ppg", "games"),
        "milb_level_jump_recency": before_year - earliest_at_highest,
    }

    # Level-over-level OPS trend
    if second_rows and top_rows:
        top_ops = _weighted_avg(top_rows, "OPS", weight_key)
        sec_ops = _weighted_avg(second_rows, "OPS", weight_key)
        features["milb_ops_trend"] = top_ops - sec_ops
        top_k = _weighted_avg(top_rows, "K_rate", weight_key)
        sec_k = _weighted_avg(second_rows, "K_rate", weight_key)
        features["milb_k_trend"] = top_k - sec_k
    else:
        features["milb_ops_trend"] = 0.0
        features["milb_k_trend"] = 0.0

    return features


# ---------------------------------------------------------------------------
# Pitcher features
# ---------------------------------------------------------------------------

def _compute_pitcher_milb_features(splits: list[dict], before_year: int) -> dict | None:
    """Compute MiLB features for a single pitcher from their stat splits."""
    rows = _parse_milb_splits(splits, before_year, group="pitching")
    if not rows:
        return None

    rows.sort(key=lambda r: (r["level_rank"], r["season"]), reverse=True)

    highest_rank = rows[0]["level_rank"]
    top_rows = [r for r in rows if r["level_rank"] == highest_rank][:2]
    second_rank = highest_rank - 1
    second_rows = [r for r in rows if r["level_rank"] == second_rank][:2]

    earliest_at_highest = min(
        (r["season"] for r in rows if r["level_rank"] == highest_rank),
        default=before_year,
    )

    total_ip = sum(r.get("IP", 0) for r in top_rows)
    weight_key = "IP"

    features = {
        "milb_highest_level": highest_rank,
        "milb_seasons_total": len(set(r["season"] for r in rows)),
        "milb_ip": total_ip,
        "milb_era": _weighted_avg(top_rows, "ERA", weight_key),
        "milb_whip": _weighted_avg(top_rows, "WHIP", weight_key),
        "milb_k9": _weighted_avg(top_rows, "K9", weight_key),
        "milb_bb9": _weighted_avg(top_rows, "BB9", weight_key),
        "milb_fip": _weighted_avg(top_rows, "FIP", weight_key),
        "milb_ppg": _weighted_avg(top_rows, "ppg", "games"),
        "milb_level_jump_recency": before_year - earliest_at_highest,
    }

    # Level-over-level ERA trend (negative = improvement)
    if second_rows and top_rows:
        features["milb_era_trend"] = (
            _weighted_avg(top_rows, "ERA", weight_key)
            - _weighted_avg(second_rows, "ERA", weight_key)
        )
        features["milb_k9_trend"] = (
            _weighted_avg(top_rows, "K9", weight_key)
            - _weighted_avg(second_rows, "K9", weight_key)
        )
    else:
        features["milb_era_trend"] = 0.0
        features["milb_k9_trend"] = 0.0

    return features


# ---------------------------------------------------------------------------
# Draft features (shared by batters and pitchers)
# ---------------------------------------------------------------------------

def _load_draft_data() -> dict[int, dict]:
    """Load all cached draft data and index by API person ID.

    Returns:
        Dict mapping mlb_api_id -> {draft_year, draft_round, draft_pick, ...}
    """
    draft_dir = CACHE_DIR / "draft"
    if not draft_dir.exists():
        return {}

    index: dict[int, dict] = {}
    for path in draft_dir.iterdir():
        if not path.name.endswith(".json"):
            continue
        try:
            picks = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        for pick in picks:
            person = pick.get("person", {})
            pid = person.get("id")
            if not pid:
                continue
            # Keep earliest draft entry per player
            if pid in index:
                continue
            index[pid] = {
                # CR opus: int(path.stem) will crash if the filename is not a
                # CR opus: valid integer (e.g., "2024_supplemental.json"). Should
                # CR opus: be wrapped in try/except like the other file parsers.
                "draft_year": int(path.stem),
                "draft_round": _int(pick.get("pickRound", 99)),
                "draft_pick": _int(pick.get("pickNumber", 999)),
            }

    return index


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def build_milb_batter_features(seasons) -> pl.DataFrame | None:
    """Build MiLB features for batters, keyed by (player_id, season).

    For each (player_id, season), uses only MiLB data from years
    strictly before that season.

    Args:
        seasons: Iterable of MLB seasons to build features for.

    Returns:
        Polars DataFrame with columns [player_id, season, milb_*, draft_*],
        or None if no MiLB data is available.
    """
    id_map = load_id_map()
    if not id_map:
        print("  [MiLB] No ID map found — run scripts/fetch_milb.py first")
        return None

    # CR opus: `reverse_map` is computed here but never used in this function.
    reverse_map = get_reverse_map()
    all_hitting = _load_cached_stats("hitting")
    draft_index = _load_draft_data()

    if not all_hitting:
        print("  [MiLB] No cached hitting stats found")
        return None

    rows = []
    season_list = sorted(seasons)

    # CR opus: This iterates all players in id_map (batters AND pitchers) against
    # CR opus: hitting stats. Most pitchers will simply have no hitting splits and
    # CR opus: be skipped, but it wastes cycles. Consider filtering to known batters.
    for lahman_id, api_id in id_map.items():
        splits = all_hitting.get(api_id)
        if not splits:
            continue

        draft_info = draft_index.get(api_id, {})

        for season in season_list:
            feats = _compute_batter_milb_features(splits, before_year=season)
            if feats is None:
                continue

            feats["player_id"] = lahman_id
            feats["season"] = season
            feats["draft_round"] = draft_info.get("draft_round", 0)
            feats["draft_pick"] = draft_info.get("draft_pick", 0)
            rows.append(feats)

    if not rows:
        print("  [MiLB] No batter features generated")
        return None

    df = pl.DataFrame(rows)
    print(f"  [MiLB] Batter features: {df.shape[0]} rows for {df['player_id'].n_unique()} players")
    return df


def build_milb_pitcher_features(seasons) -> pl.DataFrame | None:
    """Build MiLB features for pitchers, keyed by (player_id, season).

    Same interface as build_milb_batter_features.
    """
    id_map = load_id_map()
    if not id_map:
        print("  [MiLB] No ID map found — run scripts/fetch_milb.py first")
        return None

    # CR opus: `reverse_map` is computed here but never used in this function.
    reverse_map = get_reverse_map()
    all_pitching = _load_cached_stats("pitching")
    draft_index = _load_draft_data()

    if not all_pitching:
        print("  [MiLB] No cached pitching stats found")
        return None

    rows = []
    season_list = sorted(seasons)

    for lahman_id, api_id in id_map.items():
        splits = all_pitching.get(api_id)
        if not splits:
            continue

        draft_info = draft_index.get(api_id, {})

        for season in season_list:
            feats = _compute_pitcher_milb_features(splits, before_year=season)
            if feats is None:
                continue

            feats["player_id"] = lahman_id
            feats["season"] = season
            feats["draft_round"] = draft_info.get("draft_round", 0)
            feats["draft_pick"] = draft_info.get("draft_pick", 0)
            rows.append(feats)

    if not rows:
        print("  [MiLB] No pitcher features generated")
        return None

    df = pl.DataFrame(rows)
    print(f"  [MiLB] Pitcher features: {df.shape[0]} rows for {df['player_id'].n_unique()} players")
    return df
