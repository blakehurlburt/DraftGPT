"""NFL season-level model — thin wrapper around modelcore with NFL defaults.

Provides the same API as before so existing NFL scripts don't need changes,
but delegates all modeling logic to modelcore.season_model.
"""

from .season_features import get_season_feature_columns
from modelcore.season_model import (
    train_xgb as _train_xgb,
    eval_metrics as _eval_metrics,
    walk_forward_eval as _walk_forward_eval,
    walk_forward_with_residuals as _walk_forward_with_residuals,
    train_final_model as _train_final_model,
    project_season as _project_season,
    calibrate_predictions,
)

NFL_MAX_GAMES = 17


def walk_forward_eval(df, min_train_seasons=2):
    return _walk_forward_eval(df, get_season_feature_columns, NFL_MAX_GAMES,
                              min_train_seasons=min_train_seasons)


def walk_forward_with_residuals(df, min_train_seasons=2):
    return _walk_forward_with_residuals(df, get_season_feature_columns, NFL_MAX_GAMES,
                                        min_train_seasons=min_train_seasons)


def train_final_model(df, quantiles=(0.1, 0.5, 0.9)):
    return _train_final_model(df, get_season_feature_columns, quantiles=quantiles)


def project_season(ppg_model, games_model, features_df, quantile_models=None,
                   calibration_fn=None):
    return _project_season(ppg_model, games_model, features_df,
                           get_season_feature_columns, NFL_MAX_GAMES,
                           quantile_models=quantile_models,
                           calibration_fn=calibration_fn)
