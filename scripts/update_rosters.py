"""
Fetch the latest rosters from nflverse and write a summary to rosters.csv.

Usage:
    python update_rosters.py              # refresh rosters.csv from nflverse
    python update_rosters.py --diff       # show changes vs. current rosters.csv
    python update_rosters.py --season 2025  # specify season (default: latest)

After running, review rosters.csv and make any manual edits (trades not yet
reflected in nflverse, retirements, etc.), then re-run project_2026.py.
"""

import argparse
from pathlib import Path
import polars as pl
import nflreadpy as nfl

ROSTER_PATH = Path(__file__).parent.parent / "data" / "rosters.csv"
HEADER_COMMENT = """# Roster overrides for 2026 projections (auto-generated + manual edits)
# Edit this file to reflect trades, cuts, signings, and retirements.
# Columns: gsis_id, player_name, team, position, status, adjustment_ppg
#   gsis_id: NFL unique player ID (do not edit — used for matching)
#   status: ACT = active, TRADE = moved teams, CUT = released, RET = retired
#   adjustment_ppg: manual PPG adjustment (e.g. +2.0 for scheme upgrade, -1.5 for coaching downgrade)
# Lines starting with # are ignored. Re-run update_rosters.py to refresh.
"""


def fetch_latest_rosters(season=None):
    """Pull the most recent roster snapshot from nflverse."""
    if season is None:
        season = nfl.get_current_season()
    print(f"Fetching {season} rosters from nflverse...")

    rosters = nfl.load_rosters([season])
    rosters = rosters.select([
        "full_name", "team", "position", "status", "gsis_id",
    ]).rename({"full_name": "player_name"})

    # Only keep offensive skill positions relevant to fantasy
    # CR opus: This filters out DST/DEF players entirely. project_2026_v2.py generates
    # CR opus: DST projections but won't find them in rosters.csv since they're excluded here.
    rosters = rosters.filter(
        pl.col("position").is_in(["QB", "RB", "WR", "TE", "K"])
    )

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
    csv_body = df.select(["gsis_id", "player_name", "team", "position", "status", "adjustment_ppg"]).write_csv()
    with open(ROSTER_PATH, "w") as f:
        f.write(HEADER_COMMENT)
        f.write(csv_body)
    print(f"  Wrote {df.shape[0]} rows to {ROSTER_PATH}")


def show_diff(old, new):
    """Print differences between old and new rosters."""
    # CR opus: Keying on player_name alone will collide when two players share the same
    # CR opus: name (e.g., multiple "Mike Williams"). Should key on gsis_id instead.
    old_dict = {r["player_name"]: r for r in old.iter_rows(named=True)}
    new_dict = {r["player_name"]: r for r in new.iter_rows(named=True)}

    added = set(new_dict) - set(old_dict)
    removed = set(old_dict) - set(new_dict)
    common = set(old_dict) & set(new_dict)

    changes = []
    for name in common:
        o, n = old_dict[name], new_dict[name]
        if o.get("team") != n.get("team"):
            changes.append(f"  {name}: {o.get('team')} → {n.get('team')}")
        if o.get("status") != n.get("status"):
            changes.append(f"  {name}: status {o.get('status')} → {n.get('status')}")

    if not added and not removed and not changes:
        print("\nNo changes detected.")
        return

    if added:
        print(f"\n+ {len(added)} new player(s):")
        for name in sorted(added):
            r = new_dict[name]
            print(f"  + {name} ({r.get('team')}, {r.get('position')})")
    if removed:
        print(f"\n- {len(removed)} removed player(s):")
        for name in sorted(removed):
            print(f"  - {name}")
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
            manual_names = manual_rows["player_name"].to_list()
            # CR opus: Filtering by player_name is fragile — a manual "CUT" for "Josh Allen (QB)"
            # CR opus: would also remove a different "Josh Allen (TE)" from the new roster.
            # CR opus: Should filter by gsis_id instead.
            new_rosters = new_rosters.filter(~pl.col("player_name").is_in(manual_names))
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
            adj = old.filter(pl.col("adjustment_ppg") != 0.0).select(["player_name", "adjustment_ppg"])
            if adj.shape[0] > 0:
                if "adjustment_ppg" in new_rosters.columns:
                    new_rosters = new_rosters.drop("adjustment_ppg")
                new_rosters = new_rosters.join(adj, on="player_name", how="left")
                print(f"  Preserved {adj.shape[0]} manual adjustment(s)")

    write_rosters(new_rosters)
    print("Done. Review the file and make manual edits as needed.")


if __name__ == "__main__":
    main()
