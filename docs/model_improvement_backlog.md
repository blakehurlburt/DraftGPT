# Model Improvement Backlog

Approaches identified during the March 2026 model-vs-Sleeper comparison analysis.
Items below are **not yet implemented** but worth revisiting in future iterations.

## Deferred Approaches (ranked by expected impact)

### 1. Optuna Hyperparameter Tuning
**Impact: HIGH | Difficulty: MEDIUM**

Current XGBoost params are untuned defaults. Key levers for fixing tail compression:
- `max_depth` 5 -> 7-9 (deeper trees capture elite-player patterns)
- `min_child_weight` 5 -> 1-3 (allows leaf nodes for rare elite patterns)
- `reg_lambda` 1.0 -> 0.3-0.5 (less L2 regularization = less pull toward mean)
- `learning_rate` 0.05 -> 0.01-0.03 with `n_estimators` 1000+ (more granular corrections)

Optimize a composite metric: overall MAE + tail accuracy (MAE on top-5 and bottom-25 per position).
Walk-forward structure already exists in `modelcore/season_model.py` — wrap with Optuna study.

### 2. Consensus/Sleeper Projections as a Training Feature
**Impact: HIGH | Difficulty: EASY (if historical data available)**

Historical consensus projections (FantasyPros ECR, Sleeper, etc.) from 2018-2025 encode expert
knowledge about roles, coaching schemes, and qualitative factors that statistical features miss.
They also have the correct distribution shape (not compressed).

Challenge: need historical projection data. Sources: FantasyPros historical ECR, cached Sleeper
projections from prior years.

Simpler variant: blend current-year Sleeper projection with model output at a learned weight.

### 3. LightGBM as Drop-In Replacement
**Impact: MEDIUM | Difficulty: EASY**

LightGBM's leaf-wise growth (vs XGBoost's level-wise) naturally handles extremes better.
Can grow deeper, more specialized leaves for tail cases without increasing overall tree depth.
Also has native categorical handling (no ordinal encoding for college_conf_code) and faster training.

Swap: replace `XGBRegressor` with `LGBMRegressor` in `train_xgb()`. Quantile loss params differ.

### 4. Ridge Regression Ensemble Blend
**Impact: MEDIUM | Difficulty: MEDIUM**

Linear models don't compress tails the way tree models do. A ridge regression on the same features
produces predictions proportional to input features. Blending XGBoost (nonlinear interactions) with
ridge (scale preservation) at 70/30 or learned weight often outperforms either alone.

Residual stacking in `walk_forward_with_residuals()` provides existing infrastructure.

### 5. Separate Elite vs Replacement-Level Models
**Impact: MEDIUM | Difficulty: MEDIUM**

Train separate models for top-24 at position vs rest. Elite model sees tighter distribution and
learns finer distinctions among stars. Soft version: add a "prior_elite" binary feature.

### 6. Contract/ADP as Usage Proxy Features
**Impact: MEDIUM | Difficulty: MEDIUM | Needs new data**

Contract APY (from OverTheCap/Spotrac) correlates with usage and opportunity. Prior-year fantasy
ADP encodes market consensus about expected role. Both available historically. Add in
`_add_player_metadata()` or new step in `build_season_features()`.

### 7. Neural Network (MLP) Ensemble Member
**Impact: LOW-MEDIUM | Difficulty: HARD**

Simple 2-3 layer MLP (128-64-1) with dropout as a third ensemble member. Neural nets handle
continuous features differently and may preserve extremes better. But ~3000 training rows means
it will likely underperform XGBoost as a standalone model. Value is primarily as ensemble diversifier.

### 8. Mixture Density Networks
**Impact: LOW | Difficulty: HARD**

Model output as mixture of Gaussians for multi-modal outcomes (plays full season vs gets injured).
Theoretically appealing but ~3000 training samples makes this impractical. Skip unless all other
approaches plateau.

## Already Implemented (March 2026)

- [x] Quantile-mapped calibration (post-processing to fix distribution compression)
- [x] Sample weighting for elite players (top-12 get higher loss weight)
- [x] College data pipeline bugfixes (games counter, API error handling, name matching)
- [x] Sleeper scoring fix (distance-based kicker FGs, -2 INT, DST completeness)
- [x] Model vs Sleeper comparison script (`scripts/compare_projections.py`)

## Root Cause Analysis

The compression is fundamentally a **loss function + regularization** issue:
- MSE loss penalizes large residuals quadratically, pulling predictions toward conditional mean
- `reg_lambda=1.0` and `max_depth=5` further constrain leaf values
- PPG x games multiplication compounds it (both slightly compressed -> total doubly compressed)

Best attack: calibration (immediate fix) + weighted loss (training fix) + linear ensemble (structural fix).
