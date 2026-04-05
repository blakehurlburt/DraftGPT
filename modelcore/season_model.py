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


def calibrate_predictions(wf_results_df):
    """Build position-specific quantile-mapping calibration from walk-forward results.

    For each position, computes the empirical CDF of predicted PPGs and actual
    PPGs from walk-forward evaluation, then returns a function that maps new
    predictions through np.interp to stretch compressed distributions back
    toward historical reality.

    Args:
        wf_results_df: Polars DataFrame from walk_forward_eval() containing
            at minimum: position_group, pred_ppg, actual_ppg.

    Returns:
        A callable calibration_fn(pred_ppg_array, positions_array) -> calibrated array.
        Also works as calibration_fn(pred_ppg_array, positions_array) where both are
        numpy arrays of the same length.
    """
    calibration_maps = {}

    positions = wf_results_df["position_group"].unique().to_list()
    for pos in positions:
        pos_df = wf_results_df.filter(pl.col("position_group") == pos)
        pred = np.sort(pos_df["pred_ppg"].to_numpy())
        actual = np.sort(pos_df["actual_ppg"].to_numpy())

        if len(pred) < 10:
            # Not enough data to calibrate — skip this position
            continue

        # Both arrays are sorted ascending. pred[i] maps to actual[i] by quantile.
        # np.interp(new_pred, pred, actual) gives the calibrated value.
        calibration_maps[pos] = (pred, actual)

    print(f"\n  Calibration maps built for: {sorted(calibration_maps.keys())}")
    for pos, (pred, actual) in sorted(calibration_maps.items()):
        print(f"    {pos}: {len(pred)} samples, "
              f"pred range [{pred[0]:.1f}, {pred[-1]:.1f}], "
              f"actual range [{actual[0]:.1f}, {actual[-1]:.1f}]")

    def calibration_fn(pred_ppg, positions):
        """Apply quantile-mapped calibration to predicted PPG values.

        Args:
            pred_ppg: numpy array of predicted PPG values.
            positions: numpy array or list of position strings (same length).

        Returns:
            numpy array of calibrated PPG values.
        """
        calibrated = pred_ppg.copy()
        for pos, (sorted_pred, sorted_actual) in calibration_maps.items():
            mask = np.array([p == pos for p in positions])
            if mask.any():
                calibrated[mask] = np.interp(pred_ppg[mask], sorted_pred, sorted_actual)
        return calibrated

    return calibration_fn


def _compute_elite_weights(df_slice, elite_weight, top_n=12):
    """Compute sample weights: top-N players per position get elite_weight, others 1.0.

    Uses ``target_ppg`` within each ``position_group`` to identify elite players.
    Falls back to uniform weights (None) when the required columns are absent.

    Args:
        df_slice: Polars DataFrame containing at least ``position_group`` and ``target_ppg``.
        elite_weight: Weight assigned to elite players (non-elite always get 1.0).
        top_n: Number of elite players per position group.

    Returns:
        numpy array of sample weights, or None if weighting cannot be applied.
    """
    if elite_weight is None or elite_weight <= 1.0:
        return None
    if "position_group" not in df_slice.columns or "target_ppg" not in df_slice.columns:
        return None

    ranked = df_slice.with_columns(
        pl.col("target_ppg")
        .rank(method="ordinal", descending=True)
        .over("position_group")
        .alias("_elite_rank")
    )
    is_elite = ranked["_elite_rank"].to_numpy() <= top_n
    weights = np.where(is_elite, elite_weight, 1.0)
    return weights


def train_xgb(X_train, y_train, X_val, y_val, label="", quantile=None,
              model_params=None, sample_weight=None):
    """Train a single XGBRegressor with early stopping.

    Args:
        quantile: If set (0-1), trains a quantile regression model instead
                  of mean regression. E.g., 0.1 for floor, 0.9 for ceiling.
        model_params: Optional dict of XGBoost hyperparameters to override defaults.
        sample_weight: Optional array of per-sample weights for the training set.
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
        sample_weight=sample_weight,
    )
    return model


def eval_metrics(y_true, y_pred):
    """Compute MAE, RMSE, R² for a set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def walk_forward_eval(df, feature_cols_fn, max_games, min_train_seasons=2,
                      model_params=None, elite_weight=None):
    """Run walk-forward evaluation across multiple test seasons.

    For each test season X, trains on all seasons < X and evaluates on X.

    Args:
        df: Polars DataFrame with feature columns, targets, and season/player IDs.
        feature_cols_fn: Callable(df) -> list[str] returning feature column names.
        max_games: Upper bound for game count predictions (17 for NFL, 162 for MLB).
        min_train_seasons: Minimum number of training seasons required.
        model_params: Optional dict of XGBoost hyperparameters to override defaults.
        elite_weight: If set (> 1.0), top-12 players per position (by target_ppg)
            receive this weight in the loss function. Default None (uniform weights).

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

        # Prepare data — use last training season as validation for early stopping
        # to avoid data leakage from using the test set
        train_seasons = sorted(train_df["season"].unique().to_list())
        val_season = train_seasons[-1]
        train_proper = train_df.filter(pl.col("season") < val_season)
        val_df = train_df.filter(pl.col("season") == val_season)

        X_train = train_proper.select(feature_cols).to_pandas()
        X_val = val_df.select(feature_cols).to_pandas()
        X_test = test_df.select(feature_cols).to_pandas()

        y_train_ppg = train_proper["target_ppg"].to_pandas()
        y_val_ppg = val_df["target_ppg"].to_pandas()
        y_test_ppg = test_df["target_ppg"].to_pandas()

        y_train_games = train_proper["target_games"].to_pandas()
        y_val_games = val_df["target_games"].to_pandas()
        y_test_games = test_df["target_games"].to_pandas()

        # Compute elite sample weights for the training-proper split
        sw = _compute_elite_weights(train_proper, elite_weight)

        # Train PPG model (early stopping on held-out validation season)
        ppg_model = train_xgb(X_train, y_train_ppg, X_val, y_val_ppg, "PPG",
                              model_params=model_params, sample_weight=sw)
        pred_ppg = ppg_model.predict(X_test)

        # Train games model
        games_model = train_xgb(X_train, y_train_games, X_val, y_val_games, "Games",
                                model_params=model_params, sample_weight=sw)
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
    # CR opus: If all_results is empty (no test seasons met min_train_seasons), pl.concat
    # CR opus: will raise on an empty list. Should guard with `if not all_results: return`.
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


def walk_forward_with_residuals(df, feature_cols_fn, max_games, min_train_seasons=2,
                                elite_weight=None):
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
        elite_weight: If set (> 1.0), top-12 players per position (by target_ppg)
            receive this weight in the loss function. Default None (uniform weights).

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

        # Use last training season as validation for early stopping
        s1_train_seasons = sorted(train_df["season"].unique().to_list())
        s1_val_season = s1_train_seasons[-1]
        s1_train_proper = train_df.filter(pl.col("season") < s1_val_season)
        s1_val_df = train_df.filter(pl.col("season") == s1_val_season)

        X_train = s1_train_proper.select(feature_cols).to_pandas()
        X_val = s1_val_df.select(feature_cols).to_pandas()
        X_test = test_df.select(feature_cols).to_pandas()

        s1_sw = _compute_elite_weights(s1_train_proper, elite_weight)

        ppg_model = train_xgb(X_train, s1_train_proper["target_ppg"].to_pandas(),
                              X_val, s1_val_df["target_ppg"].to_pandas(),
                              sample_weight=s1_sw)
        games_model = train_xgb(X_train, s1_train_proper["target_games"].to_pandas(),
                                X_val, s1_val_df["target_games"].to_pandas(),
                                sample_weight=s1_sw)

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

        # Split training into train-proper + validation for early stopping
        s2_train_seasons = sorted(train_df["season"].unique().to_list())
        s2_val_season = s2_train_seasons[-1]
        s2_train_proper = train_df.filter(pl.col("season") < s2_val_season)
        s2_val_df = train_df.filter(pl.col("season") == s2_val_season)

        s2_train_ppg_res, s2_train_games_res = _get_residuals(s2_train_proper)
        s2_val_ppg_res, s2_val_games_res = _get_residuals(s2_val_df)

        X_train = s2_train_proper.select(feature_cols).to_pandas()
        X_train["prior_ppg_residual"] = s2_train_ppg_res
        X_train["prior_games_residual"] = s2_train_games_res

        X_val = s2_val_df.select(feature_cols).to_pandas()
        X_val["prior_ppg_residual"] = s2_val_ppg_res
        X_val["prior_games_residual"] = s2_val_games_res

        X_test = test_df.select(feature_cols).to_pandas()
        X_test["prior_ppg_residual"] = test_ppg_res
        X_test["prior_games_residual"] = test_games_res

        y_train_ppg = s2_train_proper["target_ppg"].to_pandas()
        y_val_ppg = s2_val_df["target_ppg"].to_pandas()
        y_test_ppg = test_df["target_ppg"].to_pandas()
        y_train_games = s2_train_proper["target_games"].to_pandas()
        y_val_games = s2_val_df["target_games"].to_pandas()
        y_test_games = test_df["target_games"].to_pandas()

        s2_sw = _compute_elite_weights(s2_train_proper, elite_weight)

        ppg_model = train_xgb(X_train, y_train_ppg, X_val, y_val_ppg,
                              sample_weight=s2_sw)
        pred_ppg = ppg_model.predict(X_test)

        games_model = train_xgb(X_train, y_train_games, X_val, y_val_games,
                                sample_weight=s2_sw)
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
                      model_params=None, elite_weight=None):
    """Train on ALL available data for production predictions.

    Args:
        df: Polars DataFrame with feature columns and targets.
        feature_cols_fn: Callable(df) -> list[str] returning feature column names.
        quantiles: Tuple of quantiles for floor/ceiling models.
        model_params: Optional dict of XGBoost hyperparameters to override defaults.
        elite_weight: If set (> 1.0), top-12 players per position (by target_ppg)
            receive this weight in the loss function. Default None (uniform weights).

    Returns:
        Tuple of (ppg_model, games_model, feature_importance_df, quantile_models).
    """
    feature_cols = feature_cols_fn(df)
    print(f"\nTraining final models on {df.shape[0]} rows, {len(feature_cols)} features...")

    X = df.select(feature_cols).to_pandas()
    y_ppg = df["target_ppg"].to_pandas()
    y_games = df["target_games"].to_pandas()

    sw = _compute_elite_weights(df, elite_weight)
    if sw is not None:
        print(f"  Elite weighting: {np.sum(sw > 1.0):.0f} elite samples (weight={elite_weight})")

    # CR opus: n_estimators=300 with no early stopping means the final model uses a
    # CR opus: fixed iteration count. During walk-forward, early stopping typically halts
    # CR opus: at ~150-250 rounds. If 300 is more than needed the model will overfit the
    # CR opus: training data. Consider using cross-validated early stopping here too.
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
    ppg_model.fit(X, y_ppg, sample_weight=sw)

    games_model = XGBRegressor(**_final_params)
    games_model.fit(X, y_games, sample_weight=sw)

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
        q_model.fit(X, y_ppg, sample_weight=sw)
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
                   max_games, quantile_models=None, calibration_fn=None):
    """Project next season's PPG, games, and total for each player.

    Args:
        ppg_model: Fitted XGBRegressor for PPG (mean).
        games_model: Fitted XGBRegressor for games played.
        features_df: Polars DataFrame with feature columns for the projection season.
            Recognised adjustment columns (all optional, default 0):
            - adjustment_ppg: additive PPG shift
            - adjustment_games: additive games shift (result clipped to [0, max_games])
            - adjustment_volatility: multiplicative spread factor on floor/ceiling
              distance from median (e.g. 0.15 = 15% wider range)
        feature_cols_fn: Callable(df) -> list[str] returning feature column names.
        max_games: Upper bound for game count predictions.
        quantile_models: Optional dict of {quantile: model} for floor/median/ceiling.
        calibration_fn: Optional callable from calibrate_predictions() that applies
            quantile-mapped calibration to stretch compressed distributions.

    Returns:
        Polars DataFrame with player info and projections, ranked by position.
    """
    feature_cols = feature_cols_fn(features_df)

    X = features_df.select(feature_cols).to_pandas()

    pred_ppg = ppg_model.predict(X)
    pred_games = np.clip(games_model.predict(X), 0, max_games)

    # Apply quantile-mapped calibration before manual adjustments
    if calibration_fn is not None and "position_group" in features_df.columns:
        positions = features_df["position_group"].to_list()
        pred_ppg = calibration_fn(pred_ppg, positions)
        print(f"  Applied quantile-mapped calibration to {len(pred_ppg)} predictions")

    # Apply manual PPG adjustments if present
    adj = np.zeros(len(pred_ppg))
    if "adjustment_ppg" in features_df.columns:
        adj = features_df["adjustment_ppg"].fill_null(0.0).to_numpy()
        nonzero = np.count_nonzero(adj)
        if nonzero > 0:
            print(f"  Applying {nonzero} manual PPG adjustment(s)")
            pred_ppg = pred_ppg + adj

    # Apply manual games adjustments if present
    adj_games = np.zeros(len(pred_games))
    if "adjustment_games" in features_df.columns:
        adj_games = features_df["adjustment_games"].fill_null(0.0).to_numpy()
        nonzero = np.count_nonzero(adj_games)
        if nonzero > 0:
            print(f"  Applying {nonzero} manual games adjustment(s)")
            pred_games = np.clip(pred_games + adj_games, 0, max_games)

    # Load volatility adjustment for use in quantile loop below
    adj_vol = np.zeros(len(pred_ppg))
    if "adjustment_volatility" in features_df.columns:
        adj_vol = features_df["adjustment_volatility"].fill_null(0.0).to_numpy()
        nonzero = np.count_nonzero(adj_vol)
        if nonzero > 0:
            print(f"  Applying {nonzero} manual volatility adjustment(s)")

    pred_total = pred_ppg * pred_games

    id_cols = ["player_id", "player_display_name", "position_group", "team"]
    available = [c for c in id_cols if c in features_df.columns]

    results = features_df.select(available).with_columns([
        pl.Series("projected_ppg", np.round(pred_ppg, 1)),
        pl.Series("projected_games", np.round(pred_games, 1)),
        # CR opus: np.round(...).astype(int) on negative values (e.g. negative PPG *
        # CR opus: games) could produce negative projected totals. No clipping to >= 0.
        pl.Series("projected_total", np.round(pred_total, 0).astype(int)),
    ])

    # Add quantile projections (floor/median/ceiling)
    if quantile_models:
        # First pass: compute median for volatility scaling reference
        median_pred = None
        if 0.5 in quantile_models:
            median_pred = quantile_models[0.5].predict(X)
            if calibration_fn is not None and "position_group" in features_df.columns:
                positions = features_df["position_group"].to_list()
                median_pred = calibration_fn(median_pred, positions)
            median_pred = median_pred + adj  # shift by PPG adjustment

        for q, q_model in sorted(quantile_models.items()):
            q_pred = q_model.predict(X)
            if calibration_fn is not None and "position_group" in features_df.columns:
                positions = features_df["position_group"].to_list()
                q_pred = calibration_fn(q_pred, positions)
            q_pred = q_pred + adj

            # Apply volatility adjustment to floor/ceiling (scale distance from median)
            if median_pred is not None and q != 0.5 and np.any(adj_vol != 0):
                q_pred = median_pred + (q_pred - median_pred) * (1 + adj_vol)

            q_total = q_pred * pred_games
            # CR opus: These labels only match the exact values 0.1, 0.5, 0.9. If
            # CR opus: train_final_model is called with different quantiles (e.g., 0.25, 0.75),
            # CR opus: they fall through to q25/q75 labels, but downstream code in train_mlb.py
            # CR opus: and project_2026_v2.py checks for "ppg_floor"/"ppg_ceiling" by name,
            # CR opus: so non-default quantiles would silently produce unused columns.
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
