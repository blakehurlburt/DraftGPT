"""
Walk-forward backtest for the season-level fantasy model.

Runs both the standard walk-forward and the two-stage residual-stacking
variant, then compares results side by side.
"""

from nfldata.season_features import build_season_features
from nfldata.season_model import walk_forward_eval, walk_forward_with_residuals


def _print_analysis(results, label):
    """Print best/worst predictions for a results set."""
    results = results.with_columns(
        (results["pred_total"] - results["actual_total"]).alias("error"),
        (results["pred_total"] - results["actual_total"]).abs().alias("abs_error"),
    )

    print(f"\n=== {label}: Biggest Over-Predictions ===")
    over = results.sort("error", descending=True).head(10)
    print(f"{'Player':<28}{'Pos':<5}{'Year':>5}{'Pred':>8}{'Actual':>8}{'Error':>8}")
    print("-" * 62)
    for row in over.iter_rows(named=True):
        name = (row["player_display_name"] or "???")[:26]
        print(f"{name:<28}{row['position_group']:<5}{row['season']:>5}"
              f"{row['pred_total']:>8.1f}{row['actual_total']:>8.1f}{row['error']:>8.1f}")

    print(f"\n=== {label}: Biggest Under-Predictions ===")
    under = results.sort("error").head(10)
    print(f"{'Player':<28}{'Pos':<5}{'Year':>5}{'Pred':>8}{'Actual':>8}{'Error':>8}")
    print("-" * 62)
    for row in under.iter_rows(named=True):
        name = (row["player_display_name"] or "???")[:26]
        print(f"{name:<28}{row['position_group']:<5}{row['season']:>5}"
              f"{row['pred_total']:>8.1f}{row['actual_total']:>8.1f}{row['error']:>8.1f}")


def main():
    df = build_season_features(range(2018, 2026))

    # --- Standard walk-forward ---
    print("\n" + "=" * 70)
    print("  STANDARD WALK-FORWARD (baseline)")
    print("=" * 70)
    # CR opus: walk_forward_eval() in modelcore requires (df, feature_cols_fn, max_games)
    # CR opus: but the nfldata.season_model wrapper is called here with only (df).
    # CR opus: This works, but note this import is from nfldata.season_model, not modelcore.
    results_std = walk_forward_eval(df)
    _print_analysis(results_std, "Standard")

    # --- Two-stage residual walk-forward ---
    print("\n" + "=" * 70)
    print("  TWO-STAGE RESIDUAL WALK-FORWARD")
    print("=" * 70)
    results_resid, residual_map = walk_forward_with_residuals(df)
    _print_analysis(results_resid, "Residual")

    # --- Side-by-side comparison ---
    print("\n" + "=" * 70)
    print("  COMPARISON: Standard vs Residual-Stacking")
    print("=" * 70)
    from sklearn.metrics import r2_score, mean_absolute_error
    import numpy as np

    for label, res in [("Standard", results_std), ("Residual", results_resid)]:
        ppg_r2 = r2_score(res["actual_ppg"].to_list(), res["pred_ppg"].to_list())
        ppg_mae = mean_absolute_error(res["actual_ppg"].to_list(), res["pred_ppg"].to_list())
        games_r2 = r2_score(res["actual_games"].to_list(), res["pred_games"].to_list())
        games_mae = mean_absolute_error(res["actual_games"].to_list(), res["pred_games"].to_list())
        total_r2 = r2_score(res["actual_total"].to_list(), res["pred_total"].to_list())
        total_mae = mean_absolute_error(res["actual_total"].to_list(), res["pred_total"].to_list())
        print(f"\n  {label:>10}:")
        print(f"    PPG   — R²: {ppg_r2:.3f}, MAE: {ppg_mae:.2f}")
        print(f"    Games — R²: {games_r2:.3f}, MAE: {games_mae:.2f}")
        print(f"    Total — R²: {total_r2:.3f}, MAE: {total_mae:.2f}")


if __name__ == "__main__":
    main()
