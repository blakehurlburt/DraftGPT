# Fantasy Sports Projections & Draft Tools

XGBoost models for NFL and MLB fantasy projections. Includes a Monte Carlo draft simulator and a live Sleeper draft assistant with real-time recommendations.

## Quick Start

```bash
# install dependencies (use the project venv — requires Python 3.10+)
.venv/bin/pip install -r requirements.txt

# train the NFL model (2009-2025 data, prints test metrics)
.venv/bin/python scripts/train.py

# refresh rosters from nflverse, then generate 2026 rankings
.venv/bin/python scripts/update_rosters.py
.venv/bin/python scripts/project_2026_v2.py

# run Monte Carlo draft simulation
.venv/bin/python run_draftsim.py --compare

# launch live Sleeper draft assistant
.venv/bin/python run_draftassist.py
```

## Project Structure

```
.
├── nfldata/                    # NFL data loading, feature engineering, ML model
│   ├── __init__.py             # Public API, re-exports nflreadpy functions
│   ├── loader.py               # get_player_stats(), get_player_game_log()
│   ├── stats.py                # player_season_summary(), compare_players()
│   ├── features.py             # Weekly feature engineering pipeline
│   ├── season_features.py      # Season-level feature engineering
│   ├── model.py                # XGBoost weekly model (train_model)
│   ├── season_model.py         # Season-level model (walk-forward eval)
│   └── cache.py                # Filesystem cache config for nflreadpy
│
├── mlbdata/                    # MLB data loading and feature engineering
│
├── modelcore/                  # Shared ML utilities (used by NFL + MLB models)
│
├── draftsim/                   # Monte Carlo snake draft simulator
│   ├── players.py              # Player dataclass, load from projections CSV
│   ├── config.py               # LeagueConfig (teams, lineup, caps)
│   ├── draft.py                # DraftState, snake order, make_pick
│   ├── strategies.py           # 5 strategies: BPA, VBD, VONA, Zero-RB, Robust-RB
│   ├── value.py                # VBD, VONA, replacement levels, variance bonus
│   ├── adp.py                  # Real ADP from FantasyPros (ESPN/Sleeper/CBS + consensus)
│   ├── opponents.py            # ADP-based opponent modeling
│   ├── simulate.py             # Monte Carlo simulation runner
│   └── results.py              # Lineup scoring, result formatting
│
├── draftassist/                # Live Sleeper draft assistant (web UI)
│   ├── sleeper.py              # Async Sleeper API client + player cache
│   ├── bridge.py               # Maps Sleeper data to DraftState
│   ├── recommender.py          # Top-N picks, multi-strategy recommendations
│   ├── app.py                  # FastAPI app: routes, SSE, poll loop
│   ├── static/                 # Vanilla HTML/JS/CSS frontend
│   └── README.md               # Detailed usage guide
│
├── run_draftsim.py             # Monte Carlo draft simulation CLI
├── run_draftassist.py          # Launch Sleeper draft assistant server
│
├── scripts/                    # CLI scripts
│   ├── train.py                # Train NFL weekly prediction model
│   ├── train_mlb.py            # Train MLB projection model
│   ├── backtest.py             # Walk-forward backtest (standard + residual)
│   ├── project_2026.py         # Weekly-model 2026 rankings
│   ├── project_2026_v2.py      # Season-model 2026 rankings + CSV export
│   ├── update_rosters.py       # Fetch rosters from nflverse → data/rosters.csv
│   └── main.py                 # Interactive data exploration examples
│
├── data/                       # Data files (see data/README.md for sources)
│   ├── FantasyPros_*.csv       # Real ADP rankings (manual download)
│   ├── projections/            # Per-position and combined projection CSVs
│   ├── rosters.csv             # Current rosters (auto-generated + manual edits)
│   └── lahman_1871-2025_csv/   # Lahman baseball database
│
├── tests/                      # Test suite (pytest)
│   ├── test_draftsim.py
│   ├── test_draftassist.py
│   └── test_nfldata.py
│
└── requirements.txt
```

## Three Tools

### 1. nfldata — Projection Model

Loads nflverse data (2009-2025), engineers features, and trains an XGBoost model to predict fantasy points. Includes quantile regression for floor (10th percentile), median (50th), and ceiling (90th percentile) projections.

```bash
.venv/bin/python scripts/train.py                    # train + evaluate
.venv/bin/python scripts/backtest.py                  # walk-forward backtest
.venv/bin/python scripts/project_2026_v2.py           # generate 2026 projections
```

### 2. draftsim — Draft Simulator

Monte Carlo snake draft simulator with 5 strategies, ADP-based opponent modeling, and variance-aware valuation (balances floor vs ceiling picks based on roster composition and round).

```bash
.venv/bin/python run_draftsim.py --compare    # all 5 strategies
.venv/bin/python run_draftsim.py --slot 3 --strategy vona --sims 10000
```

### 3. draftassist — Live Draft Assistant

Web UI that connects to a real Sleeper draft via polling, shows real-time pick updates via SSE, and recommends picks using all 5 strategies with configurable risk profiles (safe/balanced/aggressive). See [draftassist/README.md](draftassist/README.md) for full details.

```bash
.venv/bin/python run_draftassist.py           # http://localhost:8000
```

## Updating Data

See [data/README.md](data/README.md) for a full list of data sources and how to refresh each one. The draft assistant welcome page also shows when each data file was last updated.

```bash
.venv/bin/python scripts/update_rosters.py              # refresh from nflverse
.venv/bin/python scripts/update_rosters.py --diff       # preview changes
.venv/bin/python scripts/update_rosters.py --keep-manual  # preserve manual edits
```

## Running Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

## Requirements

- Python 3.10+
- nflreadpy, polars, pandas, pyarrow, xgboost, scikit-learn
- fastapi, uvicorn, httpx, sse-starlette (for draftassist)
- libomp (macOS: `brew install libomp` — required by XGBoost)
