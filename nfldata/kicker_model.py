"""NFL kicker model — thin wrapper around modelcore with kicker defaults.

Uses reduced-complexity XGBoost parameters suited for the small kicker
sample size (~32 kickers per season).
"""

from .kicker_features import get_kicker_feature_columns
from modelcore.season_model import (
    walk_forward_eval as _walk_forward_eval,
    train_final_model as _train_final_model,
    project_season as _project_season,
)

NFL_MAX_GAMES = 17

# Reduced complexity for small sample sizes
_KICKER_PARAMS = {
    "max_depth": 3,
    "min_child_weight": 10,
    "n_estimators": 200,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "learning_rate": 0.03,
}


def walk_forward_eval(df, min_train_seasons=2):
    return _walk_forward_eval(df, get_kicker_feature_columns, NFL_MAX_GAMES,
                              min_train_seasons=min_train_seasons,
                              model_params=_KICKER_PARAMS)


def train_final_model(df, quantiles=(0.1, 0.5, 0.9)):
    return _train_final_model(df, get_kicker_feature_columns,
                              quantiles=quantiles, model_params=_KICKER_PARAMS)


def project_season(ppg_model, games_model, features_df, quantile_models=None):
    return _project_season(ppg_model, games_model, features_df,
                           get_kicker_feature_columns, NFL_MAX_GAMES,
                           quantile_models=quantile_models)
