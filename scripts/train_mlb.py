"""Train MLB fantasy projection models and generate rankings.

Uses the Lahman database (2010-2025) to train season-level XGBoost models
for batters and pitchers separately, then projects 2026 fantasy points.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl

from mlbdata.season_features import (
    build_batter_features, build_pitcher_features,
    build_batter_projection_features, build_pitcher_projection_features,
    get_batter_feature_columns, get_pitcher_feature_columns,
    MAX_GAMES_BATTER, MAX_GAMES_PITCHER,
)
from modelcore.season_model import (
    walk_forward_eval, train_final_model, project_season,
)

SEASONS = range(2009, 2026)


def run_model(df, feature_cols_fn, max_games, label):
    """Run walk-forward eval and train final model for a player type."""
    print(f"\n{'='*60}")
    print(f"  {label} Model")
    print(f"{'='*60}")

    print(f"\n--- Walk-Forward Evaluation ({label}s) ---")
    walk_forward_eval(df, feature_cols_fn, max_games)

    print(f"\n--- Training Final {label} Model ---")
    return train_final_model(df, feature_cols_fn)


def print_rankings(results, label, positions=None):
    """Print formatted rankings."""
    has_quantiles = "ppg_floor" in results.columns

    if positions:
        for pos in positions:
            pos_df = (
                results.filter(pl.col("position_group") == pos)
                .sort("projected_total", descending=True)
                .head(30)
            )
            if pos_df.shape[0] == 0:
                continue
            print(f"\n{'='*80}")
            print(f"  {pos} RANKINGS — 2026 MLB {label} Projections")
            print(f"{'='*80}")
            if has_quantiles:
                print(f"{'#':<5}{'Player':<26}{'Team':<6}{'PPG':>7}{'Floor':>7}{'Ceil':>7}{'Games':>7}{'Total':>7}")
                print(f"{'-'*5}{'-'*26}{'-'*6}{'-'*7}{'-'*7}{'-'*7}{'-'*7}{'-'*7}")
            else:
                print(f"{'#':<5}{'Player':<26}{'Team':<6}{'PPG':>7}{'Games':>7}{'Total':>7}")
                print(f"{'-'*5}{'-'*26}{'-'*6}{'-'*7}{'-'*7}{'-'*7}")
            for i, row in enumerate(pos_df.iter_rows(named=True), 1):
                name = (row.get("player_display_name") or "???")[:24]
                team = (row.get("team") or "???")[:5]
                ppg = row["projected_ppg"]
                games = row["projected_games"]
                total = row["projected_total"]
                if has_quantiles:
                    floor = row.get("ppg_floor", 0)
                    ceil = row.get("ppg_ceiling", 0)
                    print(f"{i:<5}{name:<26}{team:<6}{ppg:>7.1f}{floor:>7.1f}{ceil:>7.1f}{games:>7.1f}{total:>7}")
                else:
                    print(f"{i:<5}{name:<26}{team:<6}{ppg:>7.1f}{games:>7.1f}{total:>7}")

    # Overall
    overall = results.sort("projected_total", descending=True).head(50)
    print(f"\n{'='*85}")
    print(f"  OVERALL {label.upper()} BOARD — Top 50")
    print(f"{'='*85}")
    if has_quantiles:
        print(f"{'#':<5}{'Pos':<5}{'Player':<26}{'Team':<6}{'PPG':>7}{'Floor':>7}{'Ceil':>7}{'Games':>7}{'Total':>7}")
        print(f"{'-'*5}{'-'*5}{'-'*26}{'-'*6}{'-'*7}{'-'*7}{'-'*7}{'-'*7}{'-'*7}")
    else:
        print(f"{'#':<5}{'Pos':<5}{'Player':<26}{'Team':<6}{'PPG':>7}{'Games':>7}{'Total':>7}")
        print(f"{'-'*5}{'-'*5}{'-'*26}{'-'*6}{'-'*7}{'-'*7}{'-'*7}")
    for i, row in enumerate(overall.iter_rows(named=True), 1):
        name = (row.get("player_display_name") or "???")[:24]
        team = (row.get("team") or "???")[:5]
        pos = row.get("position_group", "?")
        ppg = row["projected_ppg"]
        games = row["projected_games"]
        total = row["projected_total"]
        if has_quantiles:
            floor = row.get("ppg_floor", 0)
            ceil = row.get("ppg_ceiling", 0)
            print(f"{i:<5}{pos:<5}{name:<26}{team:<6}{ppg:>7.1f}{floor:>7.1f}{ceil:>7.1f}{games:>7.1f}{total:>7}")
        else:
            print(f"{i:<5}{pos:<5}{name:<26}{team:<6}{ppg:>7.1f}{games:>7.1f}{total:>7}")


def main():
    print("=" * 60)
    print("  MLB Fantasy Projection Model")
    print("=" * 60)

    # --- Batters ---
    print("\nBuilding batter features (2009-2025)...")
    batter_df = build_batter_features(SEASONS)

    bat_ppg, bat_games, bat_imp, bat_q = run_model(
        batter_df, get_batter_feature_columns, MAX_GAMES_BATTER, "Batter"
    )

    # Build projection features (one row per player, latest season only)
    print("\nBuilding batter projection features...")
    bat_proj_df = build_batter_projection_features(SEASONS)

    bat_results = project_season(
        bat_ppg, bat_games, bat_proj_df, get_batter_feature_columns,
        MAX_GAMES_BATTER, bat_q
    )

    print_rankings(bat_results, "Batter",
                   positions=["C", "1B", "2B", "3B", "SS", "OF", "DH"])

    # --- Pitchers ---
    print("\n\nBuilding pitcher features (2009-2025)...")
    pitcher_df = build_pitcher_features(SEASONS)

    pit_ppg, pit_games, pit_imp, pit_q = run_model(
        pitcher_df, get_pitcher_feature_columns, MAX_GAMES_PITCHER, "Pitcher"
    )

    # Build projection features (one row per player, latest season only)
    print("\nBuilding pitcher projection features...")
    pit_proj_df = build_pitcher_projection_features(SEASONS)

    pit_results = project_season(
        pit_ppg, pit_games, pit_proj_df, get_pitcher_feature_columns,
        MAX_GAMES_PITCHER, pit_q
    )

    print_rankings(pit_results, "Pitcher", positions=["SP", "RP"])

    # --- Save combined projections CSV for draft assistant ---
    output_path = Path(__file__).parent.parent / "data" / "projections" / "mlb_projections.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Combine batters and pitchers
    combined = pl.concat([bat_results, pit_results])

    # Filter to positive projections only
    combined = combined.filter(pl.col("projected_total") > 0)

    # Rename columns to match the expected schema
    rename_map = {}
    if "ppg_floor" in combined.columns:
        rename_map["ppg_floor"] = "ppg_floor_raw"
    if "ppg_ceiling" in combined.columns:
        rename_map["ppg_ceiling"] = "ppg_ceiling_raw"
    if rename_map:
        combined = combined.rename(rename_map)

    # Add total_floor / total_ceiling from quantile columns if available
    if "total_floor" not in combined.columns:
        combined = combined.with_columns(pl.lit(0).alias("total_floor"))
    if "total_ceiling" not in combined.columns:
        combined = combined.with_columns(pl.lit(0).alias("total_ceiling"))

    # Select and write
    out_cols = ["player_display_name", "position_group", "team",
                "projected_ppg", "projected_games", "projected_total",
                "pos_rank", "total_floor", "total_ceiling"]
    available_out = [c for c in out_cols if c in combined.columns]
    combined.select(available_out).sort("projected_total", descending=True).write_csv(str(output_path))
    print(f"\nSaved {combined.shape[0]} projections to {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
