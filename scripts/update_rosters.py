"""
Fetch the latest rosters from nflverse and write a summary to rosters.csv.

Usage:
    python update_rosters.py              # refresh rosters.csv from nflverse
    python update_rosters.py --diff       # show changes vs. current rosters.csv
    python update_rosters.py --season 2025  # specify season (default: latest)

After running, review rosters.csv and make any manual edits (trades not yet
reflected in nflverse, retirements, etc.), then re-run project_2026_v2.py.
"""

import argparse
from pathlib import Path
import polars as pl
import nflreadpy as nfl

ROSTER_PATH = Path(__file__).parent.parent / "data" / "rosters.csv"
HEADER_COMMENT = """# Roster overrides for 2026 projections (auto-generated + manual edits)
# Edit this file to reflect trades, cuts, signings, and retirements.
# Columns: gsis_id, pfr_id, player_name, team, position, status, rookie_year,
#          draft_round, draft_number, draft_club, adjustment_ppg
#   gsis_id: NFL unique player ID (do not edit — used for matching)
#   pfr_id: Pro Football Reference ID (used to match combine/draft rookies)
#   status: ACT = active, TRADE = moved teams, CUT = released, RET = retired
#   draft_number: actual overall NFL draft pick (not fantasy ADP)
#   adjustment_ppg: manual PPG adjustment (e.g. +2.0 for scheme upgrade, -1.5 for coaching downgrade)
# Lines starting with # are ignored. Re-run update_rosters.py to refresh.
"""

ROSTER_COLUMNS = [
    "gsis_id", "pfr_id", "player_name", "team", "position", "status",
    "rookie_year", "draft_round", "draft_number", "draft_club",
    "adjustment_ppg",
]

# nflverse uses these alternate abbreviations in roster and draft datasets.
TEAM_ALIASES = {
    "AZ": "ARI", "LA": "LAR", "LVR": "LV", "NOR": "NO",
    "TAM": "TB", "SFO": "SF", "GNB": "GB", "KAN": "KC",
    "NWE": "NE",
}


def _canonical_team_expr(column):
    return pl.col(column).replace(TEAM_ALIASES)


def fetch_latest_rosters(season=None):
    """Pull the most recent roster snapshot from nflverse."""
    if season is None:
        season = nfl.get_current_season()
    print(f"Fetching {season} rosters from nflverse...")

    rosters = nfl.load_rosters([season])
    roster_cols = [
        "full_name", "team", "position", "status", "gsis_id", "pfr_id",
        "rookie_year", "draft_club", "draft_number",
    ]
    rosters = rosters.select([c for c in roster_cols if c in rosters.columns]).rename(
        {"full_name": "player_name"}
    )

    # Only keep offensive skill positions relevant to fantasy
    # CR opus: This filters out DST/DEF players entirely. project_2026_v2.py generates
    # CR opus: DST projections but won't find them in rosters.csv since they're excluded here.
    rosters = rosters.filter(
        pl.col("position").is_in(["QB", "RB", "WR", "TE", "K"])
    )

    # The current-season roster feed can temporarily omit PFR IDs even when
    # nflverse's player registry already has them (notably for new rookies).
    player_ids = (
        nfl.load_players()
        .select(["gsis_id", pl.col("pfr_id").alias("_players_pfr_id")])
        .drop_nulls(subset=["gsis_id"])
        .unique(subset=["gsis_id"])
    )
    rosters = rosters.join(player_ids, on="gsis_id", how="left").with_columns(
        pl.col("pfr_id").fill_null(pl.col("_players_pfr_id")).alias("pfr_id")
    ).drop("_players_pfr_id")

    # draft_number in the roster feed is the actual overall NFL pick. Join it
    # to the draft results to add the round; 2026 draft GSIS IDs are provisional
    # and do not match the roster GSIS IDs, so they are not a safe join key.
    draft = nfl.load_draft_picks([season]).select([
        pl.col("pick").alias("draft_number"),
        pl.col("round").alias("draft_round"),
    ]).drop_nulls(subset=["draft_number"]).unique(subset=["draft_number"])
    rosters = rosters.join(draft, on="draft_number", how="left")

    rosters = rosters.with_columns([
        _canonical_team_expr("team").alias("team"),
        _canonical_team_expr("draft_club").alias("draft_club"),
    ])

    # Sort by position priority then name
    pos_order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "K": 4}
    rosters = rosters.with_columns(
        pl.col("position").replace_strict(pos_order, default=9).alias("_pos_ord")
    ).sort(["_pos_ord", "player_name"]).drop("_pos_ord")

    print(f"  Found {rosters.shape[0]} players across {rosters['team'].n_unique()} teams")
    return rosters


def load_existing_overrides():
    """Load manually edited rows from existing rosters.csv."""
    if not ROSTER_PATH.exists():
        return pl.DataFrame({"player_name": [], "team": [], "position": [], "status": []})
    return pl.read_csv(ROSTER_PATH, comment_prefix="#")


def write_rosters(df):
    """Write roster dataframe to CSV with header comments."""
    # Ensure adjustment_ppg column exists
    if "adjustment_ppg" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("adjustment_ppg"))
    for column in ROSTER_COLUMNS:
        if column not in df.columns:
            df = df.with_columns(pl.lit(None).alias(column))
    csv_body = df.select(ROSTER_COLUMNS).write_csv()
    with open(ROSTER_PATH, "w") as f:
        f.write(HEADER_COMMENT)
        f.write(csv_body)
    print(f"  Wrote {df.shape[0]} rows to {ROSTER_PATH}")


def show_diff(old, new):
    """Print differences between old and new rosters."""
    old_dict = {r["gsis_id"]: r for r in old.iter_rows(named=True)}
    new_dict = {r["gsis_id"]: r for r in new.iter_rows(named=True)}

    added = set(new_dict) - set(old_dict)
    removed = set(old_dict) - set(new_dict)
    common = set(old_dict) & set(new_dict)

    changes = []
    for player_id in common:
        o, n = old_dict[player_id], new_dict[player_id]
        name = n.get("player_name") or o.get("player_name") or player_id
        if o.get("team") != n.get("team"):
            changes.append(f"  {name}: {o.get('team')} → {n.get('team')}")
        if o.get("status") != n.get("status"):
            changes.append(f"  {name}: status {o.get('status')} → {n.get('status')}")

    if not added and not removed and not changes:
        print("\nNo changes detected.")
        return

    if added:
        print(f"\n+ {len(added)} new player(s):")
        for player_id in sorted(added):
            r = new_dict[player_id]
            print(f"  + {r.get('player_name')} ({r.get('team')}, {r.get('position')})")
    if removed:
        print(f"\n- {len(removed)} removed player(s):")
        for player_id in sorted(removed):
            print(f"  - {old_dict[player_id].get('player_name')}")
    if changes:
        print(f"\n~ {len(changes)} change(s):")
        for c in changes:
            print(c)


def main():
    parser = argparse.ArgumentParser(description="Update rosters.csv from nflverse")
    parser.add_argument("--diff", action="store_true",
                        help="Show changes vs. current rosters.csv without writing")
    parser.add_argument("--season", type=int, default=None,
                        help="Season year to fetch (default: current)")
    parser.add_argument("--keep-manual", action="store_true",
                        help="Preserve manual overrides (CUT/RET/TRADE rows) from existing file")
    args = parser.parse_args()

    new_rosters = fetch_latest_rosters(args.season)

    if args.diff:
        old = load_existing_overrides()
        show_diff(old, new_rosters)
        return

    if args.keep_manual:
        old = load_existing_overrides()
        # CR opus: If the old CSV has no "status" column (e.g., corrupted/empty file),
        # CR opus: this filter will raise a ColumnNotFoundError with no helpful message.
        # Keep manually-set CUT/RET/TRADE rows that aren't in nflverse
        manual_rows = old.filter(pl.col("status").is_in(["CUT", "RET", "TRADE"]))
        if manual_rows.shape[0] > 0:
            manual_ids = manual_rows["gsis_id"].drop_nulls().to_list()
            new_rosters = new_rosters.filter(~pl.col("gsis_id").is_in(manual_ids))
            # Resolve gsis_id for manual rows if missing
            if "gsis_id" not in manual_rows.columns:
                id_map = new_rosters.select(["gsis_id", "player_name"]).head(0)  # empty, just for schema
                players = nfl.load_players().select(["gsis_id", "display_name", "position"]).drop_nulls()
                id_map = (
                    players.unique(subset=["display_name", "position"])
                    .rename({"display_name": "player_name"})
                )
                manual_rows = manual_rows.join(id_map, on=["player_name", "position"], how="left")
            new_rosters = pl.concat([new_rosters, manual_rows.select(new_rosters.columns)], how="diagonal")
            print(f"  Preserved {manual_rows.shape[0]} manual override(s)")

        # Preserve any non-zero adjustment_ppg values from the old file
        if "adjustment_ppg" in old.columns:
            adj = old.filter(pl.col("adjustment_ppg") != 0.0).select(["gsis_id", "adjustment_ppg"])
            if adj.shape[0] > 0:
                if "adjustment_ppg" in new_rosters.columns:
                    new_rosters = new_rosters.drop("adjustment_ppg")
                new_rosters = new_rosters.join(adj, on="gsis_id", how="left")
                print(f"  Preserved {adj.shape[0]} manual adjustment(s)")

    write_rosters(new_rosters)
    print("Done. Review the file and make manual edits as needed.")


if __name__ == "__main__":
    main()
