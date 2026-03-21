"""
Model training and evaluation for fantasy points prediction.

Uses XGBoost regression with temporal train/test split to predict
next-week fantasy points (PPR).
"""

import polars as pl
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .features import get_feature_columns


def train_model(df, target_col="fantasy_points_ppr"):
    """Train an XGBoost model to predict fantasy points.

    Uses a temporal split: 2018-2023 for training, 2024 for testing.

    Args:
        df: Polars DataFrame from build_features().
        target_col: Name of the target column.

    Returns:
        Tuple of (fitted XGBRegressor, feature importance DataFrame).
    """
    feature_cols = get_feature_columns(df)
    print(f"\nUsing {len(feature_cols)} features:")
    for c in feature_cols:
        print(f"  - {c}")

    # Temporal train/test split — use latest season as test
    max_season = df["season"].max()
    train_df = df.filter(pl.col("season") < max_season)
    test_df = df.filter(pl.col("season") == max_season)
    print(f"\nTrain: {train_df.shape[0]:,} rows (<{max_season})")
    print(f"Test:  {test_df.shape[0]:,} rows ({max_season})")

    # Use second-to-last season as validation for early stopping (no test leakage)
    val_season = max_season - 1
    train_proper = train_df.filter(pl.col("season") < val_season)
    val_df = train_df.filter(pl.col("season") == val_season)

    # Convert to pandas for XGBoost
    X_train = train_proper.select(feature_cols).to_pandas()
    y_train = train_proper.select(target_col).to_pandas()[target_col]
    X_val = val_df.select(feature_cols).to_pandas()
    y_val = val_df.select(target_col).to_pandas()[target_col]
    X_test = test_df.select(feature_cols).to_pandas()
    y_test = test_df.select(target_col).to_pandas()[target_col]

    # Handle any remaining categorical/string columns
    for col in X_train.columns:
        if X_train[col].dtype == "object":
            X_train[col] = X_train[col].astype("category")
            X_val[col] = X_val[col].astype("category")
            X_test[col] = X_test[col].astype("category")

    # Train XGBoost
    # CR opus: enable_categorical=True requires ALL categorical columns to be pandas
    # CategoricalDtype. The loop above only converts "object" dtype columns, but
    # a column might already be a non-object non-categorical type (e.g., string)
    # that XGBoost can't handle. Consider explicit categorical encoding instead.
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        enable_categorical=True,
        early_stopping_rounds=50,
        verbosity=1,
    )

    print("\nTraining XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n=== Test Set Metrics ({max_season} season) ===")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.4f}")

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    importance_pl = pl.from_pandas(importance)

    return model, importance_pl


def predict_week(model, features_df, season, week):
    """Generate predictions for a specific week.

    Args:
        model: Fitted XGBRegressor.
        features_df: Polars DataFrame from build_features().
        season: Season year.
        week: Week number.

    Returns:
        Polars DataFrame with player info and predictions.
    """
    feature_cols = get_feature_columns(features_df)

    week_df = features_df.filter(
        (pl.col("season") == season) & (pl.col("week") == week)
    )

    if week_df.shape[0] == 0:
        print(f"No data found for season {season}, week {week}")
        return pl.DataFrame()

    X = week_df.select(feature_cols).to_pandas()

    # Handle categoricals
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category")

    preds = model.predict(X)

    id_cols = ["player_name", "player_display_name", "position_group",
               "team", "season", "week"]
    available_ids = [c for c in id_cols if c in week_df.columns]

    result = week_df.select(available_ids).with_columns(
        pl.Series("predicted_fpp", preds)
    )

    return result.sort("predicted_fpp", descending=True)
