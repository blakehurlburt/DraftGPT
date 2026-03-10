"""Sport-agnostic season-level model training and evaluation.

Trains two XGBoost models (PPG and games played) using walk-forward
validation: for each test season X, train only on seasons < X.

All functions require a `feature_cols_fn` callable that extracts the
feature column names from a DataFrame. This is the only sport-specific
dependency — NFL and MLB each provide their own implementation.
"""

import polars as pl
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_xgb(X_train, y_train, X_val, y_val, label="", quantile=None,
              model_params=None):
    """Train a single XGBRegressor with early stopping.

    Args:
        quantile: If set (0-1), trains a quantile regression model instead
                  of mean regression. E.g., 0.1 for floor, 0.9 for ceiling.
        model_params: Optional dict of XGBoost hyperparameters to override defaults.
    """
    kwargs = dict(
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
    if model_params:
        kwargs.update(model_params)
    if quantile is not None:
        kwargs["objective"] = "reg:quantileerror"
        kwargs["quantile_alpha"] = quantile
    model = XGBRegressor(**kwargs)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def eval_metrics(y_true, y_pred):
    """Compute MAE, RMSE, R² for a set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def walk_forward_eval(df, feature_cols_fn, max_games, min_train_seasons=2,
                      model_params=None):
    """Run walk-forward evaluation across multiple test seasons.

    For each test season X, trains on all seasons < X and evaluates on X.

    Args:
        df: Polars DataFrame with feature columns, targets, and season/player IDs.
        feature_cols_fn: Callable(df) -> list[str] returning feature column names.
        max_games: Upper bound for game count predictions (17 for NFL, 162 for MLB).
        min_train_seasons: Minimum number of training seasons required.
        model_params: Optional dict of XGBoost hyperparameters to override defaults.

    Returns:
        Polars DataFrame with all predictions and actuals.
    """
    feature_cols = feature_cols_fn(df)
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
        ppg_model = train_xgb(X_train, y_train_ppg, X_test, y_test_ppg, "PPG",
                              model_params=model_params)
        pred_ppg = ppg_model.predict(X_test)

        # Train games model
        games_model = train_xgb(X_train, y_train_games, X_test, y_test_games, "Games",
                                model_params=model_params)
        pred_games = np.clip(games_model.predict(X_test), 0, max_games)

        # Compute predicted total
        pred_total = pred_ppg * pred_games
        actual_total = y_test_ppg.values * y_test_games.values

        # Metrics
        ppg_m = eval_metrics(y_test_ppg, pred_ppg)
        games_m = eval_metrics(y_test_games, pred_games)
        total_m = eval_metrics(actual_total, pred_total)

        print(f"  PPG   — MAE: {ppg_m['mae']:.2f}, RMSE: {ppg_m['rmse']:.2f}, R²: {ppg_m['r2']:.3f}")
        print(f"  Games — MAE: {games_m['mae']:.2f}, RMSE: {games_m['rmse']:.2f}, R²: {games_m['r2']:.3f}")
        print(f"  Total — MAE: {total_m['mae']:.1f}, RMSE: {total_m['rmse']:.1f}, R²: {total_m['r2']:.3f}")

        season_metrics["ppg"].append(ppg_m)
        season_metrics["games"].append(games_m)
        season_metrics["total"].append(total_m)

        # Store results
        id_cols = ["player_id", "player_display_name", "position_group", "season"]
        available_ids = [c for c in id_cols if c in test_df.columns]
        result = test_df.select(available_ids).with_columns([
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


def walk_forward_with_residuals(df, feature_cols_fn, max_games, min_train_seasons=2):
    """Two-stage walk-forward: stage 1 generates residuals, stage 2 uses them.

    Stage 1: Normal walk-forward to compute per-player prediction errors.
    Stage 2: Re-run walk-forward with prior-year residuals as extra features.
             "How wrong was the model about this player last year?" lets the
             model correct systematic per-player biases.

    This is a form of stacking with no leakage — each residual comes from
    a model that never saw the target year.

    Args:
        df: Polars DataFrame with feature columns, targets, and season/player IDs.
        feature_cols_fn: Callable(df) -> list[str] returning feature column names.
        max_games: Upper bound for game count predictions.
        min_train_seasons: Minimum training seasons required.

    Returns:
        Tuple of (stage2_results_df, residual_lookup) where residual_lookup
        maps (player_id, season) -> (ppg_residual, games_residual).
    """
    feature_cols = feature_cols_fn(df)
    all_seasons = sorted(df["season"].unique().to_list())
    min_season = min(all_seasons)
    test_seasons = [s for s in all_seasons if s - min_season >= min_train_seasons]

    # ---- Stage 1: normal walk-forward to collect residuals ----
    print(f"\n=== Stage 1: Base Walk-Forward (collecting residuals) ===")
    residual_map = {}

    for test_season in test_seasons:
        train_df = df.filter(pl.col("season") < test_season)
        test_df = df.filter(pl.col("season") == test_season)

        X_train = train_df.select(feature_cols).to_pandas()
        X_test = test_df.select(feature_cols).to_pandas()

        ppg_model = train_xgb(X_train, train_df["target_ppg"].to_pandas(),
                              X_test, test_df["target_ppg"].to_pandas())
        games_model = train_xgb(X_train, train_df["target_games"].to_pandas(),
                                X_test, test_df["target_games"].to_pandas())

        pred_ppg = ppg_model.predict(X_test)
        pred_games = np.clip(games_model.predict(X_test), 0, max_games)

        actual_ppg = test_df["target_ppg"].to_list()
        actual_games = test_df["target_games"].to_list()
        player_ids = test_df["player_id"].to_list()

        for pid, ap, pp, ag, pg in zip(player_ids, actual_ppg, pred_ppg,
                                        actual_games, pred_games):
            residual_map[(pid, test_season)] = (ap - pp, ag - pg)

    print(f"  Collected {len(residual_map)} player-season residuals")

    # ---- Stage 2: walk-forward with residual features ----
    print(f"\n=== Stage 2: Walk-Forward with Residual Features ===")

    all_results = []
    season_metrics = {"ppg": [], "games": [], "total": []}

    for test_season in test_seasons:
        train_df = df.filter(pl.col("season") < test_season)
        test_df = df.filter(pl.col("season") == test_season)

        def _get_residuals(df_slice):
            pids = df_slice["player_id"].to_list()
            seasons = df_slice["season"].to_list()
            ppg_res, games_res = [], []
            for pid, s in zip(pids, seasons):
                r = residual_map.get((pid, s - 1))
                if r is not None:
                    ppg_res.append(r[0])
                    games_res.append(r[1])
                else:
                    ppg_res.append(0.0)
                    games_res.append(0.0)
            return ppg_res, games_res

        train_ppg_res, train_games_res = _get_residuals(train_df)
        test_ppg_res, test_games_res = _get_residuals(test_df)

        X_train = train_df.select(feature_cols).to_pandas()
        X_train["prior_ppg_residual"] = train_ppg_res
        X_train["prior_games_residual"] = train_games_res

        X_test = test_df.select(feature_cols).to_pandas()
        X_test["prior_ppg_residual"] = test_ppg_res
        X_test["prior_games_residual"] = test_games_res

        y_train_ppg = train_df["target_ppg"].to_pandas()
        y_test_ppg = test_df["target_ppg"].to_pandas()
        y_train_games = train_df["target_games"].to_pandas()
        y_test_games = test_df["target_games"].to_pandas()

        ppg_model = train_xgb(X_train, y_train_ppg, X_test, y_test_ppg)
        pred_ppg = ppg_model.predict(X_test)

        games_model = train_xgb(X_train, y_train_games, X_test, y_test_games)
        pred_games = np.clip(games_model.predict(X_test), 0, max_games)

        pred_total = pred_ppg * pred_games
        actual_total = y_test_ppg.values * y_test_games.values

        ppg_m = eval_metrics(y_test_ppg, pred_ppg)
        games_m = eval_metrics(y_test_games, pred_games)
        total_m = eval_metrics(actual_total, pred_total)

        print(f"--- Season {test_season} ---")
        print(f"  Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")
        print(f"  PPG   — MAE: {ppg_m['mae']:.2f}, RMSE: {ppg_m['rmse']:.2f}, R²: {ppg_m['r2']:.3f}")
        print(f"  Games — MAE: {games_m['mae']:.2f}, RMSE: {games_m['rmse']:.2f}, R²: {games_m['r2']:.3f}")
        print(f"  Total — MAE: {total_m['mae']:.1f}, RMSE: {total_m['rmse']:.1f}, R²: {total_m['r2']:.3f}")

        season_metrics["ppg"].append(ppg_m)
        season_metrics["games"].append(games_m)
        season_metrics["total"].append(total_m)

        # Update residual map with stage-2 residuals
        player_ids = test_df["player_id"].to_list()
        actual_ppg_vals = y_test_ppg.values
        actual_games_vals = y_test_games.values
        for pid, ap, pp, ag, pg in zip(player_ids, actual_ppg_vals, pred_ppg,
                                        actual_games_vals, pred_games):
            residual_map[(pid, test_season)] = (ap - pp, ag - pg)

        id_cols = ["player_id", "player_display_name", "position_group", "season"]
        available_ids = [c for c in id_cols if c in test_df.columns]
        result = test_df.select(available_ids).with_columns([
            pl.Series("pred_ppg", pred_ppg.round(2)),
            pl.Series("actual_ppg", y_test_ppg.values),
            pl.Series("pred_games", pred_games.round(1)),
            pl.Series("actual_games", y_test_games.values.astype(float)),
            pl.Series("pred_total", pred_total.round(1)),
            pl.Series("actual_total", actual_total.round(1)),
        ])
        all_results.append(result)

    results_df = pl.concat(all_results)

    print(f"\n=== Stage 2 Overall Metrics ({len(test_seasons)} seasons) ===")
    for target_name in ["ppg", "games", "total"]:
        avg_mae = np.mean([m["mae"] for m in season_metrics[target_name]])
        avg_r2 = np.mean([m["r2"] for m in season_metrics[target_name]])
        print(f"  {target_name.upper():>5} — Avg MAE: {avg_mae:.2f}, Avg R²: {avg_r2:.3f}")

    overall_ppg_r2 = r2_score(results_df["actual_ppg"].to_list(), results_df["pred_ppg"].to_list())
    overall_total_r2 = r2_score(results_df["actual_total"].to_list(), results_df["pred_total"].to_list())
    print(f"\n  Combined R² — PPG: {overall_ppg_r2:.3f}, Total: {overall_total_r2:.3f}")

    return results_df, residual_map


def train_final_model(df, feature_cols_fn, quantiles=(0.1, 0.5, 0.9),
                      model_params=None):
    """Train on ALL available data for production predictions.

    Args:
        df: Polars DataFrame with feature columns and targets.
        feature_cols_fn: Callable(df) -> list[str] returning feature column names.
        quantiles: Tuple of quantiles for floor/ceiling models.
        model_params: Optional dict of XGBoost hyperparameters to override defaults.

    Returns:
        Tuple of (ppg_model, games_model, feature_importance_df, quantile_models).
    """
    feature_cols = feature_cols_fn(df)
    print(f"\nTraining final models on {df.shape[0]} rows, {len(feature_cols)} features...")

    X = df.select(feature_cols).to_pandas()
    y_ppg = df["target_ppg"].to_pandas()
    y_games = df["target_games"].to_pandas()

    _final_params = dict(
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
    if model_params:
        _final_params.update(model_params)

    ppg_model = XGBRegressor(**_final_params)
    ppg_model.fit(X, y_ppg)

    games_model = XGBRegressor(**_final_params)
    games_model.fit(X, y_games)

    # Train quantile models for PPG floor/ceiling
    quantile_models = {}
    for q in quantiles:
        if q < 0.5:
            label = "floor"
        elif q == 0.5:
            label = "median"
        else:
            label = "ceiling"
        print(f"  Training PPG {label} model (q={q})...")
        q_params = {**_final_params, "objective": "reg:quantileerror", "quantile_alpha": q}
        q_model = XGBRegressor(**q_params)
        q_model.fit(X, y_ppg)
        quantile_models[q] = q_model

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "ppg_importance": ppg_model.feature_importances_,
        "games_importance": games_model.feature_importances_,
    }).sort_values("ppg_importance", ascending=False).reset_index(drop=True)

    print("  Top 10 features (PPG model):")
    for _, row in importance.head(10).iterrows():
        print(f"    {row['feature']:<30} {row['ppg_importance']:.4f}")

    return ppg_model, games_model, pl.from_pandas(importance), quantile_models


def project_season(ppg_model, games_model, features_df, feature_cols_fn,
                   max_games, quantile_models=None):
    """Project next season's PPG, games, and total for each player.

    Args:
        ppg_model: Fitted XGBRegressor for PPG (mean).
        games_model: Fitted XGBRegressor for games played.
        features_df: Polars DataFrame with feature columns for the projection season.
        feature_cols_fn: Callable(df) -> list[str] returning feature column names.
        max_games: Upper bound for game count predictions.
        quantile_models: Optional dict of {quantile: model} for floor/median/ceiling.

    Returns:
        Polars DataFrame with player info and projections, ranked by position.
    """
    feature_cols = feature_cols_fn(features_df)

    X = features_df.select(feature_cols).to_pandas()

    pred_ppg = ppg_model.predict(X)
    pred_games = np.clip(games_model.predict(X), 0, max_games)

    # Apply manual PPG adjustments if present
    adj = np.zeros(len(pred_ppg))
    if "adjustment_ppg" in features_df.columns:
        adj = features_df["adjustment_ppg"].fill_null(0.0).to_numpy()
        nonzero = np.count_nonzero(adj)
        if nonzero > 0:
            print(f"  Applying {nonzero} manual PPG adjustment(s)")
            pred_ppg = pred_ppg + adj

    pred_total = pred_ppg * pred_games

    id_cols = ["player_id", "player_display_name", "position_group", "team"]
    available = [c for c in id_cols if c in features_df.columns]

    results = features_df.select(available).with_columns([
        pl.Series("projected_ppg", np.round(pred_ppg, 1)),
        pl.Series("projected_games", np.round(pred_games, 1)),
        pl.Series("projected_total", np.round(pred_total, 0).astype(int)),
    ])

    # Add quantile projections (floor/median/ceiling)
    if quantile_models:
        for q, q_model in sorted(quantile_models.items()):
            q_pred = q_model.predict(X) + adj
            q_total = q_pred * pred_games
            if q == 0.1:
                label = "floor"
            elif q == 0.5:
                label = "median"
            elif q == 0.9:
                label = "ceiling"
            else:
                label = f"q{int(q*100)}"
            results = results.with_columns([
                pl.Series(f"ppg_{label}", np.round(q_pred, 1)),
                pl.Series(f"total_{label}", np.round(q_total, 0).astype(int)),
            ])

    # Rank within position
    results = results.sort("projected_total", descending=True)
    if "position_group" in results.columns:
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
