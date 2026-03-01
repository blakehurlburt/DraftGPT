# Fantasy Football Projections

XGBoost model trained on 7 seasons of nflverse data (2018-2024) to predict weekly PPR fantasy points per player. Uses rolling stats, game context, and opponent-adjusted features to project 2026 rankings by position.

## Quick Start

```bash
# install dependencies (use the project venv)
.venv/bin/pip install -r requirements.txt

# train the model (2018-2024 data, prints test metrics)
.venv/bin/python train.py

# refresh rosters from nflverse, then generate 2026 rankings
.venv/bin/python update_rosters.py
.venv/bin/python project_2026.py
```

## Project Structure

```
.
├── train.py               # Train model, print accuracy metrics
├── project_2026.py        # Generate 2026 fantasy rankings by position
├── update_rosters.py      # Fetch latest rosters from nflverse → rosters.csv
├── rosters.csv            # Current player rosters (auto-generated + manually editable)
├── main.py                # Interactive data exploration examples
├── requirements.txt       # Python dependencies
│
└── nfldata/               # Core library
    ├── __init__.py        # Public API, re-exports nflreadpy functions
    ├── loader.py          # get_player_stats(), get_player_game_log()
    ├── stats.py           # player_season_summary(), compare_players(), top_performers()
    ├── features.py        # Feature engineering pipeline (build_features)
    ├── model.py           # XGBoost training & prediction (train_model, predict_week)
    └── cache.py           # Filesystem cache config for nflreadpy
```

## Updating Rosters

Rosters are stored in `rosters.csv` and used by `project_2026.py` to determine current team assignments and filter out inactive players.

### Refresh from nflverse

```bash
# Pull latest roster data (overwrites rosters.csv)
.venv/bin/python update_rosters.py

# Preview changes before writing
.venv/bin/python update_rosters.py --diff

# Refresh but keep your manual CUT/RET/TRADE edits
.venv/bin/python update_rosters.py --keep-manual
```

### Manual edits

Open `rosters.csv` in any editor or spreadsheet app and change rows directly:

| Scenario | What to do |
|---|---|
| Player traded | Change the `team` column to new team |
| Player cut/released | Set `status` to `CUT` |
| Player retired | Set `status` to `RET` |
| Free agent signing | Change `team` and keep `status` as `ACT` |

Lines starting with `#` are comments and ignored. Player names must match nflverse display names exactly (e.g. "Ja'Marr Chase", not "Jamarr Chase").

After editing, re-run `python project_2026.py` to regenerate rankings.

## How the Model Works

### Feature Pipeline (`nfldata/features.py`)

Merges four nflverse data sources into one player-game-level dataframe:

| Source | Join Key | Fields Used |
|---|---|---|
| `load_player_stats()` | base table | all box score stats, fantasy points |
| `load_schedules()` | `game_id` | spread, total, roof, surface, temp, wind, rest days |
| `load_rosters()` | `player_id` + `season` | years_exp, height, weight, draft_number |
| `load_snap_counts()` | `game_id` + `pfr_id` (via players table) | offense_pct |

Then engineers ~50 features:

- **Rolling averages** (3-week and 8-week) for: fantasy points, passing/rushing/receiving yards, targets, carries, receptions, EPA, snap share, target share, WOPR
- **Trends**: 3-week avg minus 8-week avg (momentum signal)
- **Consistency**: 5-week rolling standard deviation of fantasy points
- **Game context**: is_home, rest_days, is_dome, spread, game total, temp, wind
- **Player metadata**: years_exp, height, weight, draft_number
- **Opponent-adjusted**: season-to-date fantasy points allowed by opponent to the player's position group

Rolling windows do not cross season boundaries. All features use only prior-week data (no leakage).

### Model (`nfldata/model.py`)

- **Algorithm**: XGBRegressor (gradient-boosted trees)
- **Train/test split**: temporal — 2018-2023 train, 2024 test
- **Early stopping**: monitors validation RMSE, stops after 50 rounds without improvement
- **Test performance**: MAE ~4.5, RMSE ~6.1, R-squared ~0.43

### Projection (`project_2026.py`)

For each player, takes their last game row from the 2025 season (which carries the rolling feature history), feeds it through the model, and outputs projected weekly PPR points. Rankings are printed by position (QB/RB/WR/TE top 20-40) and as an overall top-60 draft board.

## Data Exploration (`nfldata` library)

The `nfldata` package also provides helpers for ad-hoc analysis:

```python
from nfldata import get_player_stats, player_season_summary, compare_players, top_performers

# Weekly stats for a player
log = get_player_stats(seasons=[2024], weeks=[1, 2, 3])

# Season summary with per-game averages
summary = player_season_summary("Ja'Marr Chase", seasons=[2023, 2024])

# Side-by-side comparison
comp = compare_players(["Josh Allen", "Patrick Mahomes"], seasons=[2024])

# Week 1 leaderboard
top = top_performers(season=2024, week=1, stat="fantasy_points_ppr", n=10)
```

All functions return Polars DataFrames. The full nflreadpy API is re-exported through `nfldata` — see `nfldata/__init__.py` for the complete list.

## Requirements

- Python 3.10+
- nflreadpy, polars, pandas, pyarrow, xgboost, scikit-learn
- libomp (macOS: `brew install libomp` — required by XGBoost)
