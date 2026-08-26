# Data Directory

All data files used by the draft assistant and ML models live here.
The web app's welcome page shows when each file was last updated.

## Current Data Sources

### NFL Fantasy

| File | Source | How to Update |
|------|--------|---------------|
| `FantasyPros_2026_Overall_ADP_Rankings.csv` | [FantasyPros PPR ADP](https://www.fantasypros.com/nfl/adp/ppr-overall.php) | Current PPR consensus and per-platform ADP (ESPN, Sleeper, CBS, RTSports, Fantrax), refreshed August 26, 2026. |
| `projections/*.csv` | Model output | Run `.venv/bin/python scripts/project_2026_v2.py`. Generates QB/RB/WR/TE/K/DST and overall projections, including explicit rookie draft metadata. |
| `rosters.csv` | [nflverse](https://github.com/nflverse/nflverse-data) | Run `.venv/bin/python scripts/update_rosters.py --season 2026`. Fetches current teams/statuses plus actual NFL draft round and overall pick. |

### MLB

| File | Source | How to Update |
|------|--------|---------------|
| `lahman_1871-2025_csv/` | [Lahman Baseball Database](https://sabr.app.box.com/s/y1prhc795jk8zvmelfd3jq7tl389y6cd) | Manual download. Unzip into this directory. |

## Adding a New Data Source

1. Place the file(s) in this directory (or a subdirectory).
2. Register it in `draftassist/app.py` in the `_DATA_SOURCES` list so it shows
   on the welcome page with freshness tracking:
   ```python
   {
       "name": "Human-readable name",
       "file": DATA_DIR / "filename.csv",  # path to check for last-modified
       "url": "https://...",               # source URL (empty string if script-generated)
       "how": "Manual download / Run script_name.py",
   }
   ```
3. Update this README with the source info.
