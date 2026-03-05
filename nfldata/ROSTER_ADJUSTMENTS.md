# Roster Change Features & Manual Adjustments

## Overview

The season-level model now incorporates roster context features that capture team changes, offensive environment, positional competition, and QB changes. It also supports manual PPG adjustments for factors the model can't capture.

## Roster Context Features (Automatic)

These features are computed automatically from historical data in `season_features.py`:

| Feature | Description |
|---------|-------------|
| `changed_team` | Binary: player switched teams between seasons |
| `new_team_pass_att_pg` | Player's current team's pass attempts/game (prior year) |
| `new_team_rush_att_pg` | Player's current team's rush attempts/game (prior year) |
| `new_team_td_pg` | Player's current team's TDs/game (prior year) |
| `pos_vacated_pts` | Total PPG from same-position players who LEFT the team |
| `pos_added_pts` | Total prior PPG from same-position players who JOINED the team |
| `pos_net_opportunity` | `vacated - added` (positive = more opportunity) |
| `qb_changed` | Binary: team's starting QB changed (WR/TE only) |
| `new_qb_prior_ppg` | New QB's prior-season fantasy PPG (WR/TE only) |

## Manual Adjustments

Edit `data/rosters.csv` to set the `adjustment_ppg` column for any player. This adds (or subtracts) a fixed PPG value to the model's prediction.

### When to use manual adjustments

- **Coaching changes**: New OC installs a pass-heavy or run-heavy scheme
- **Scheme fit**: Player moving to a system that better suits their skills
- **Holdouts/suspensions**: Expected missed time not reflected in injury data
- **Camp reports**: Breakout or decline signals from training camp
- **Rookie projections**: The model can't project rookies (no prior data); use manual adjustments in rosters.csv

### Example

```csv
player_name,team,position,status,adjustment_ppg
Saquon Barkley,PHI,RB,ACT,2.0
Davante Adams,NYJ,WR,ACT,-1.5
```

- Saquon gets +2.0 PPG boost (upgraded to elite offensive line)
- Davante gets -1.5 PPG penalty (QB downgrade concerns)

### Refreshing rosters

When running `scripts/update_rosters.py --keep-manual`, your manual `adjustment_ppg` values are preserved. Without `--keep-manual`, they reset to 0.

## How it works in the pipeline

1. `build_season_features()` in `season_features.py` computes all roster context features automatically
2. `project_2026_v2.py` loads `adjustment_ppg` from `rosters.csv` and passes it through to `project_season()`
3. `project_season()` in `season_model.py` applies `adjustment_ppg` after model prediction: `final_ppg = model_ppg + adjustment_ppg`
4. The adjusted PPG is then multiplied by predicted games to get `projected_total`
