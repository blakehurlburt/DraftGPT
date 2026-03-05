# AGENTS.md - Notes for AI Assistants Working on DraftGPT

## Project Structure

- `nfldata/` — Core prediction library (features, models, stats)
- `draftassist/` — Live draft assistant web app (FastAPI + JS frontend)
- `scripts/` — CLI scripts for training, projecting, roster management
- `data/` — Rosters, projections, ADP data
- `tests/` — pytest test suite

## Key Files

- `nfldata/season_features.py` — Season-level feature engineering (the main feature pipeline)
- `nfldata/season_model.py` — XGBoost model training, walk-forward eval, projections
- `nfldata/features.py` — Weekly-level feature engineering (game-by-game predictions)
- `nfldata/model.py` — Weekly-level XGBoost model
- `scripts/project_2026_v2.py` — Season-level projection script (the primary one)
- `scripts/project_2026.py` — Weekly-level projection script (older approach)
- `scripts/update_rosters.py` — Roster refresh from nflverse
- `data/rosters.csv` — Roster overrides with manual adjustments

## Python Environment

- **Use `.venv/`** — The project uses a venv at `.venv/` with Python 3.12 (homebrew)
- Run scripts with `.venv/bin/python scripts/whatever.py`
- Key dependencies: `nflreadpy`, `polars`, `xgboost`, `scikit-learn`, `pandas`, `numpy`

## Data Library: nflreadpy

- This is the nflverse data library — provides NFL stats, rosters, injuries, depth charts, etc.
- It uses a filesystem cache (`.cache/` directory with parquet files, 24-hour TTL)
- Main functions: `load_player_stats()`, `load_rosters()`, `load_team_stats()`, `load_injuries()`, `load_depth_charts()`, `load_snap_counts()`, `load_players()`
- All return Polars DataFrames
- Player ID: `player_id` (GSIS ID) is the primary key throughout the codebase

## Polars, Not Pandas

- The codebase uses **Polars** for data processing, converting to pandas only for XGBoost `fit()`/`predict()`
- Be careful with Polars syntax (it's not pandas): `.filter()`, `.with_columns()`, `.select()`, `pl.col()`, etc.
- Watch out for schema mismatches when using `pl.concat()` — use `how="diagonal"` if columns differ

## Modeling Architecture

### Season-Level (Primary)
- Two XGBoost regressors: one for PPG, one for games played
- Total = PPG * games
- Walk-forward validation: train on seasons < X, test on season X
- Two-stage residual stacking available (`walk_forward_with_residuals()`)
- Features are lagged by 1 season (prior1_*) or 2 seasons (*_2yr) — no leakage

### Weekly-Level (Secondary)
- Single XGBoost regressor for next-week fantasy_points_ppr
- Rolling features (3-week, 8-week averages) with lag-1 to prevent leakage

## Roster Context Features (Added March 2026)

New features in `season_features.py` → `_add_roster_context_features()`:
- `changed_team`, team offensive volume, positional competition, QB changes
- These are computed by comparing roster composition between consecutive seasons
- For projections, team assignments come from `rosters.csv` (current rosters), not historical data

### Known Issues & Gotchas

1. **Projection year team assignment**: The `_build_projection_features()` function in `project_2026_v2.py` creates synthetic "next year" rows. The `team` column for these rows must come from `rosters.csv` (the current roster), not from the prior season. This is handled by the `rosters` parameter — make sure it's passed correctly.

2. **QB identification for `qb_changed`**: The current approach identifies the "primary QB" as the first QB in the group-by result (which is the one with the most games due to prior sorting). This is a rough heuristic — if a team has a mid-season QB change, it picks one of them. A more robust approach would use games started.

3. **Positional competition features loop**: The `_add_roster_context_features()` function uses a Python loop over teams/seasons/positions to compute departures and arrivals. This is O(teams * seasons * positions) and is the slowest part of feature building (~5-10 seconds). It could be vectorized with Polars joins if performance becomes an issue.

4. **Rookies**: The model cannot project rookies because it requires prior-season data (`prior1_ppg` must be non-null). Rookies must be handled via manual `adjustment_ppg` in `rosters.csv` or a separate rookie model.

5. **`team` column in aggregation**: The `team` column was added to `_aggregate_to_season()` using `.last()` — this gives the player's last team of the season. For mid-season trades, this is the destination team.

6. **`adjustment_ppg` must be excluded from features**: It's listed in `get_season_feature_columns()` drop_cols to prevent it from being used as a model feature. If you add new non-feature columns, add them to drop_cols too.

7. **Walk-forward eval uses all features**: When running `walk_forward_eval()` after adding new features, the new features are automatically included. To compare with/without, drop the feature columns from the DataFrame before calling eval (see the comparison script pattern).

## Testing

```bash
.venv/bin/python -m pytest tests/ -v
```

For model evaluation:
```bash
.venv/bin/python -c "
from nfldata.season_features import build_season_features
from nfldata.season_model import walk_forward_eval
df = build_season_features(range(2018, 2025))
walk_forward_eval(df)
"
```
