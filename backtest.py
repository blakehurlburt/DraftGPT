"""
Walk-forward backtest for the season-level fantasy model.

For each test season (2022-2024), trains on all prior seasons and evaluates
predictions against actuals. Prints per-season accuracy, overall metrics,
and best/worst predictions for analysis.
"""

from nfldata.season_features import build_season_features
from nfldata.season_model import walk_forward_eval


def main():
    # Build season-level features (2018-2019 provide history, 2020+ are targets)
    df = build_season_features(range(2018, 2025))

    # Run walk-forward evaluation
    results = walk_forward_eval(df)

    # Analyze best and worst predictions
    results = results.with_columns(
        (results["pred_total"] - results["actual_total"]).alias("error"),
        (results["pred_total"] - results["actual_total"]).abs().alias("abs_error"),
    )

    print("\n=== Biggest Over-Predictions (model too optimistic) ===")
    over = results.sort("error", descending=True).head(15)
    print(f"{'Player':<28}{'Pos':<5}{'Year':>5}{'Pred':>8}{'Actual':>8}{'Error':>8}")
    print("-" * 62)
    for row in over.iter_rows(named=True):
        name = (row["player_display_name"] or "???")[:26]
        print(f"{name:<28}{row['position_group']:<5}{row['season']:>5}"
              f"{row['pred_total']:>8.1f}{row['actual_total']:>8.1f}{row['error']:>8.1f}")

    print("\n=== Biggest Under-Predictions (model too pessimistic) ===")
    under = results.sort("error").head(15)
    print(f"{'Player':<28}{'Pos':<5}{'Year':>5}{'Pred':>8}{'Actual':>8}{'Error':>8}")
    print("-" * 62)
    for row in under.iter_rows(named=True):
        name = (row["player_display_name"] or "???")[:26]
        print(f"{name:<28}{row['position_group']:<5}{row['season']:>5}"
              f"{row['pred_total']:>8.1f}{row['actual_total']:>8.1f}{row['error']:>8.1f}")

    print("\n=== Most Accurate Predictions ===")
    accurate = results.sort("abs_error").head(15)
    print(f"{'Player':<28}{'Pos':<5}{'Year':>5}{'Pred':>8}{'Actual':>8}{'Error':>8}")
    print("-" * 62)
    for row in accurate.iter_rows(named=True):
        name = (row["player_display_name"] or "???")[:26]
        print(f"{name:<28}{row['position_group']:<5}{row['season']:>5}"
              f"{row['pred_total']:>8.1f}{row['actual_total']:>8.1f}{row['error']:>8.1f}")


if __name__ == "__main__":
    main()
