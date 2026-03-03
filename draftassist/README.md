# Draft Assistant — Sleeper Integration

A live draft assistant that connects to a real [Sleeper](https://sleeper.com) fantasy football draft, tracks picks in real-time, and recommends your next pick using five different draft strategies.

## Prerequisites

- Python 3.9+
- The `data/projections/all_projections.csv` file (player projections used by the draft engine)

## Install

From the `football/` project root:

```
pip install fastapi uvicorn httpx sse-starlette
```

## Start the Server

```
python scripts/run_draftassist.py
```

The server starts at **http://localhost:8000**.

## Connect to a Sleeper Draft

1. Open http://localhost:8000 in your browser.
2. Enter your **Sleeper Draft ID** and your **draft slot number** (1-indexed).
3. Click **Connect**.

### Where to find your Draft ID

On Sleeper, open your draft lobby. The draft ID is the long numeric string in the URL:

```
https://sleeper.com/draft/nfl/1234567890123456789
                              └─── this is your draft ID
```

You can also find it via the Sleeper API — it appears in the response from `GET https://api.sleeper.app/v1/league/{league_id}/drafts`.

### Where to find your slot number

Your slot is your position in the draft order (1 = first pick, 2 = second pick, etc.). This is visible in the Sleeper draft lobby before the draft starts. If the draft has already started, check the draft board to see which column is yours.

## What You See

### Status Bar
Shows the current round, overall pick number, and how many picks until your next turn. A pulsing green **YOUR TURN** indicator appears when you're on the clock.

### Recommendations Panel
When it's your turn, the assistant shows your top 5 pick recommendations under each strategy tab:

| Strategy | Approach |
|----------|----------|
| **BPA** | Best Player Available — highest projected points |
| **VBD** | Value Based Drafting — highest value over positional replacement level |
| **VONA** | Value Over Next Available — factors in how much a position's value drops by your next pick |
| **Zero-RB** | Avoids RBs in rounds 1-4, loads up on WR/TE/QB early |
| **Robust-RB** | Prioritizes RBs in rounds 1-3 if they offer near-top value |

Each recommendation shows the player name, position, team, projected season points, and VBD score.

### My Roster
Your drafted players so far, plus chips showing unfilled starter needs (e.g., "RB: 2 needed").

### Recent Picks
A scrolling ticker of the last 15 picks across all teams. Your picks are highlighted.

### Browser Notifications
The app requests notification permission on load. When it's your turn, you'll get a desktop notification so you can keep the tab in the background.

## How It Works

1. **Connect** hits the Sleeper API to fetch draft settings, existing picks, and the full NFL player database (~5MB, cached locally for 24 hours in `.cache/sleeper_players.json`).
2. Players from your `data/projections/all_projections.csv` are matched to Sleeper player IDs by normalized name + position.
3. A `DraftState` is built by replaying all Sleeper picks in order. Unmatched players (kickers, defenses, IDP) are inserted as zero-projection placeholders so the draft state stays in sync.
4. A **background poll loop** checks the Sleeper picks endpoint every 3 seconds. When new picks are detected, the state is rebuilt and pushed to the browser via Server-Sent Events (SSE).
5. Recommendations are generated only when it's your turn, by running each of the 5 strategies iteratively to produce a ranked top-5 list.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/connect?draft_id=...&user_slot=...` | Connect to a draft (slot is 1-indexed) |
| `GET` | `/api/state` | Current state snapshot (JSON) |
| `GET` | `/api/stream` | SSE stream — emits `draft_update` events on new picks |
| `GET` | `/` | Web UI |

## File Structure

```
draftassist/
├── __init__.py
├── sleeper.py         # Async Sleeper API client + player cache
├── bridge.py          # Maps Sleeper data to draftsim DraftState
├── recommender.py     # Top-N picks wrapper, multi-strategy recommendations
├── app.py             # FastAPI app: routes, SSE, poll loop
└── static/
    ├── index.html     # Single-page draft assistant UI
    ├── style.css      # Dark theme styles
    └── app.js         # EventSource listener, DOM updates
```

## Troubleshooting

**"Not connected to a draft"** — You need to POST to `/api/connect` first (the Connect button does this). Make sure the draft ID is correct.

**Few players matched** — The player matching uses normalized names. If your projections CSV has different name formats than Sleeper (e.g., "Gabriel Davis" vs "Gabe Davis"), those players won't match and will be treated as unknown picks.

**Picks out of sync** — The tool rebuilds state from scratch on every poll cycle, so it self-corrects. If picks for kickers/defenses/IDP appear, they are handled as zero-projection placeholders and won't affect recommendations.

**Slow first connection** — The first connect fetches Sleeper's full player database (~5MB). This is cached to `.cache/sleeper_players.json` for 24 hours, so subsequent connections are fast.
