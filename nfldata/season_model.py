"""
Season-level model training and walk-forward evaluation.

Trains two XGBoost models (PPG and games played) using walk-forward
validation: for each test season X, train only on seasons < X.
"""

import polars as pl
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .season_features import get_season_feature_columns


def _train_xgb(X_train, y_train, X_val, y_val, label=""):
    """Train a single XGBRegressor with early stopping."""
    model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=50,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def _eval_metrics(y_true, y_pred):
    """Compute MAE, RMSE, R² for a set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def walk_forward_eval(df, min_train_seasons=2):
    """Run walk-forward evaluation across multiple test seasons.

    For each test season X, trains on all seasons < X and evaluates on X.

    Args:
        df: Polars DataFrame from build_season_features().
        min_train_seasons: Minimum number of training seasons required.

    Returns:
        Polars DataFrame with all predictions and actuals.
    """
    feature_cols = get_season_feature_columns(df)
    all_seasons = sorted(df["season"].unique().to_list())
    min_season = min(all_seasons)

    print(f"\n=== Walk-Forward Evaluation ===")
    print(f"Features: {len(feature_cols)}")
    print(f"Available seasons: {all_seasons}")

    # Test seasons: those with enough training history
    test_seasons = [s for s in all_seasons if s - min_season >= min_train_seasons]
    print(f"Test seasons: {test_seasons}\n")

    all_results = []
    season_metrics = {"ppg": [], "games": [], "total": []}

    for test_season in test_seasons:
        train_df = df.filter(pl.col("season") < test_season)
        test_df = df.filter(pl.col("season") == test_season)

        print(f"--- Season {test_season} ---")
        print(f"  Train: {train_df.shape[0]} rows ({min_season}-{test_season - 1})")
        print(f"  Test:  {test_df.shape[0]} rows")

        # Prepare data
        X_train = train_df.select(feature_cols).to_pandas()
        X_test = test_df.select(feature_cols).to_pandas()

        y_train_ppg = train_df["target_ppg"].to_pandas()
        y_test_ppg = test_df["target_ppg"].to_pandas()

        y_train_games = train_df["target_games"].to_pandas()
        y_test_games = test_df["target_games"].to_pandas()

        # Train PPG model
        ppg_model = _train_xgb(X_train, y_train_ppg, X_test, y_test_ppg, "PPG")
        pred_ppg = ppg_model.predict(X_test)

        # Train games model
        games_model = _train_xgb(X_train, y_train_games, X_test, y_test_games, "Games")
        pred_games = games_model.predict(X_test)

        # Clamp games to reasonable range
        pred_games = np.clip(pred_games, 0, 17)

        # Compute predicted total
        pred_total = pred_ppg * pred_games
        actual_total = y_test_ppg.values * y_test_games.values

        # Metrics
        ppg_m = _eval_metrics(y_test_ppg, pred_ppg)
        games_m = _eval_metrics(y_test_games, pred_games)
        total_m = _eval_metrics(actual_total, pred_total)

        print(f"  PPG   — MAE: {ppg_m['mae']:.2f}, RMSE: {ppg_m['rmse']:.2f}, R²: {ppg_m['r2']:.3f}")
        print(f"  Games — MAE: {games_m['mae']:.2f}, RMSE: {games_m['rmse']:.2f}, R²: {games_m['r2']:.3f}")
        print(f"  Total — MAE: {total_m['mae']:.1f}, RMSE: {total_m['rmse']:.1f}, R²: {total_m['r2']:.3f}")

        season_metrics["ppg"].append(ppg_m)
        season_metrics["games"].append(games_m)
        season_metrics["total"].append(total_m)

        # Store results
        id_cols = ["player_id", "player_display_name", "position_group", "season"]
        result = test_df.select(id_cols).with_columns([
            pl.Series("pred_ppg", pred_ppg.round(2)),
            pl.Series("actual_ppg", y_test_ppg.values),
            pl.Series("pred_games", pred_games.round(1)),
            pl.Series("actual_games", y_test_games.values.astype(float)),
            pl.Series("pred_total", pred_total.round(1)),
            pl.Series("actual_total", actual_total.round(1)),
        ])
        all_results.append(result)

    # Overall metrics
    results_df = pl.concat(all_results)
    print(f"\n=== Overall Metrics ({len(test_seasons)} seasons) ===")

    for target_name in ["ppg", "games", "total"]:
        avg_mae = np.mean([m["mae"] for m in season_metrics[target_name]])
        avg_rmse = np.mean([m["rmse"] for m in season_metrics[target_name]])
        avg_r2 = np.mean([m["r2"] for m in season_metrics[target_name]])
        print(f"  {target_name.upper():>5} — Avg MAE: {avg_mae:.2f}, Avg RMSE: {avg_rmse:.2f}, Avg R²: {avg_r2:.3f}")

    # Overall R² across all test data combined
    overall_ppg_r2 = r2_score(results_df["actual_ppg"].to_list(), results_df["pred_ppg"].to_list())
    overall_total_r2 = r2_score(results_df["actual_total"].to_list(), results_df["pred_total"].to_list())
    print(f"\n  Combined R² — PPG: {overall_ppg_r2:.3f}, Total: {overall_total_r2:.3f}")

    return results_df


def train_final_model(df):
    """Train on ALL available data for production predictions.

    Args:
        df: Polars DataFrame from build_season_features().

    Returns:
        Tuple of (ppg_model, games_model, feature_importance_df).
    """
    feature_cols = get_season_feature_columns(df)
    print(f"\nTraining final models on {df.shape[0]} rows, {len(feature_cols)} features...")

    X = df.select(feature_cols).to_pandas()
    y_ppg = df["target_ppg"].to_pandas()
    y_games = df["target_games"].to_pandas()

    # Train with no early stopping (use all data)
    ppg_model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    ppg_model.fit(X, y_ppg)

    games_model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    games_model.fit(X, y_games)

    # Feature importance (from PPG model, which is the primary one)
    importance = pd.DataFrame({
        "feature": feature_cols,
        "ppg_importance": ppg_model.feature_importances_,
        "games_importance": games_model.feature_importances_,
    }).sort_values("ppg_importance", ascending=False).reset_index(drop=True)

    print("  Top 10 features (PPG model):")
    for _, row in importance.head(10).iterrows():
        print(f"    {row['feature']:<30} {row['ppg_importance']:.4f}")

    return ppg_model, games_model, pl.from_pandas(importance)


def project_season(ppg_model, games_model, features_df):
    """Project next season's PPG, games, and total for each player.

    Args:
        ppg_model: Fitted XGBRegressor for PPG.
        games_model: Fitted XGBRegressor for games played.
        features_df: Polars DataFrame with feature columns for the projection season.

    Returns:
        Polars DataFrame with player info and projections, ranked by position.
    """
    from .season_features import get_season_feature_columns
    feature_cols = get_season_feature_columns(features_df)

    X = features_df.select(feature_cols).to_pandas()

    pred_ppg = ppg_model.predict(X)
    pred_games = np.clip(games_model.predict(X), 0, 17)
    pred_total = pred_ppg * pred_games

    id_cols = ["player_id", "player_display_name", "position_group"]
    available = [c for c in id_cols if c in features_df.columns]

    results = features_df.select(available).with_columns([
        pl.Series("projected_ppg", np.round(pred_ppg, 1)),
        pl.Series("projected_games", np.round(pred_games, 1)),
        pl.Series("projected_total", np.round(pred_total, 0).astype(int)),
    ])

    # Rank within position
    results = results.sort("projected_total", descending=True)
    results = results.with_columns(
        pl.col("projected_total")
        .rank(method="ordinal", descending=True)
        .over("position_group")
        .alias("pos_rank")
    )
    results = results.with_columns(
        (pl.col("position_group") + pl.col("pos_rank").cast(pl.Utf8)).alias("rank_label")
    )

    return results
