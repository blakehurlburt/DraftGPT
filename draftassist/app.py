"""FastAPI application: routes, SSE streaming, poll loop."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from draftsim.players import Player, load_players
from draftsim.value import compute_replacement_levels, vbd

from .bridge import build_player_index, config_from_sleeper_meta, rebuild_draft_state
from .recommender import get_all_recommendations
from .sleeper import fetch_all_players, fetch_draft_meta, fetch_draft_picks

STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class DraftSession:
    """Holds live draft state for polling."""

    draft_id: str = ""
    user_slot: int = 0  # 0-indexed draft slot
    players: list[Player] = field(default_factory=list)
    id_to_player: dict[str, Player] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    last_pick_count: int = 0
    state_payload: dict = field(default_factory=dict)
    poll_task: asyncio.Task | None = None
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    connected: bool = False


session = DraftSession()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Cancel poll task on shutdown
    if session.poll_task and not session.poll_task.done():
        session.poll_task.cancel()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _build_state_payload(state, meta, picks, user_slot, players, adp_order=None):
    """Build the JSON payload describing current draft state."""
    config = state.config
    is_complete = state.is_complete
    current_pick = state.current_pick + 1  # 1-indexed for display
    current_round = state.current_round
    current_team = state.current_team_idx if not is_complete else -1
    is_my_turn = (not is_complete) and (state.current_team_idx == user_slot)
    picks_until = state.picks_until_next(user_slot) if not is_complete else 0

    # Recommendations (only when it's user's turn)
    recs = {}
    if is_my_turn:
        all_recs = get_all_recommendations(state, user_slot, players, adp_order)
        for strat_name, rec_list in all_recs.items():
            recs[strat_name] = [
                {
                    "rank": r.rank,
                    "name": r.player.name,
                    "position": r.player.position,
                    "team": r.player.team,
                    "projected_total": round(r.player.projected_total, 1),
                    "vbd": round(r.vbd_value, 1),
                }
                for r in rec_list
            ]

    # Recent picks (last 15)
    recent = []
    sorted_picks = sorted(picks, key=lambda p: p.get("pick_no", 0))
    for p in sorted_picks[-15:]:
        pm = p.get("metadata", {})
        recent.append({
            "pick_no": p.get("pick_no", 0),
            "round": p.get("round", 0),
            "draft_slot": p.get("draft_slot", 0),
            "player_name": f"{pm.get('first_name', '')} {pm.get('last_name', '')}".strip(),
            "position": pm.get("position", ""),
            "team": pm.get("team", ""),
            "is_user": p.get("draft_slot", -1) == user_slot + 1,  # Sleeper uses 1-indexed
        })

    # User roster
    user_roster = []
    if not is_complete and user_slot < len(state.teams):
        roster = state.teams[user_slot]
    elif user_slot < len(state.teams):
        roster = state.teams[user_slot]
    else:
        roster = []

    replacement = compute_replacement_levels(players, config) if players else {}
    for p in roster:
        user_roster.append({
            "name": p.name,
            "position": p.position,
            "team": p.team,
            "projected_total": round(p.projected_total, 1),
            "vbd": round(vbd(p, replacement), 1),
        })

    # Team needs
    needs = state.team_needs(user_slot) if user_slot < config.num_teams else {}

    return {
        "current_pick": current_pick,
        "total_picks": config.total_picks,
        "current_round": current_round,
        "total_rounds": config.num_rounds,
        "current_team": current_team,
        "is_my_turn": is_my_turn,
        "is_complete": is_complete,
        "picks_until_next": picks_until,
        "num_teams": config.num_teams,
        "user_slot": user_slot,
        "picks_made": len(picks),
        "recommendations": recs,
        "recent_picks": recent,
        "user_roster": user_roster,
        "team_needs": needs,
        "draft_status": meta.get("status", "unknown"),
    }


async def _poll_loop():
    """Background task: poll Sleeper every 3s, push SSE on new picks."""
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            await asyncio.sleep(3)
            if not session.connected:
                continue
            try:
                picks = await fetch_draft_picks(client, session.draft_id)
                pick_count = len(picks)

                if pick_count != session.last_pick_count:
                    session.last_pick_count = pick_count

                    # Rebuild state from scratch
                    config = config_from_sleeper_meta(session.meta)
                    state = rebuild_draft_state(
                        config, session.players, picks, session.id_to_player
                    )

                    adp_order = [
                        p.name for p in sorted(
                            session.players,
                            key=lambda p: p.projected_total,
                            reverse=True,
                        )
                    ]

                    payload = _build_state_payload(
                        state, session.meta, picks,
                        session.user_slot, session.players, adp_order,
                    )
                    session.state_payload = payload

                    # Push to all SSE subscribers
                    for q in list(session.subscribers):
                        try:
                            q.put_nowait(payload)
                        except asyncio.QueueFull:
                            pass

            except Exception as e:
                print(f"[poll] Error: {e}")


@app.post("/api/connect")
async def connect_draft(
    draft_id: str = Query(...),
    user_slot: int = Query(..., description="1-indexed draft slot"),
):
    """Connect to a Sleeper draft. Fetches meta, picks, builds state."""
    # Cancel existing poll if any
    if session.poll_task and not session.poll_task.done():
        session.poll_task.cancel()

    async with httpx.AsyncClient(timeout=30) as client:
        meta = await fetch_draft_meta(client, draft_id)
        picks = await fetch_draft_picks(client, draft_id)
        sleeper_players = await fetch_all_players(client)

    players = load_players()
    id_to_player = build_player_index(players, sleeper_players)

    config = config_from_sleeper_meta(meta)
    state = rebuild_draft_state(config, players, picks, id_to_player)

    # Store in session (0-indexed internally)
    slot_0 = user_slot - 1
    session.draft_id = draft_id
    session.user_slot = slot_0
    session.players = players
    session.id_to_player = id_to_player
    session.meta = meta
    session.last_pick_count = len(picks)
    session.connected = True

    adp_order = [
        p.name for p in sorted(
            players, key=lambda p: p.projected_total, reverse=True
        )
    ]
    payload = _build_state_payload(
        state, meta, picks, slot_0, players, adp_order
    )
    session.state_payload = payload

    # Start poll loop
    session.poll_task = asyncio.create_task(_poll_loop())

    return JSONResponse({
        "status": "connected",
        "draft_id": draft_id,
        "user_slot": user_slot,
        "num_teams": config.num_teams,
        "rounds": config.num_rounds,
        "picks_made": len(picks),
        "players_matched": len(id_to_player),
        "total_players": len(players),
        "draft_status": meta.get("status", "unknown"),
    })


@app.get("/api/state")
async def get_state():
    """Current state snapshot for initial load / reconnect."""
    if not session.connected:
        return JSONResponse({"error": "Not connected to a draft"}, status_code=400)
    return JSONResponse(session.state_payload)


@app.get("/api/stream")
async def stream(request: Request):
    """SSE endpoint: yields draft_update events on new picks."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    session.subscribers.append(queue)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15)
                    yield {"event": "draft_update", "data": _json_dumps(payload)}
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {"event": "ping", "data": ""}
        finally:
            if queue in session.subscribers:
                session.subscribers.remove(queue)

    return EventSourceResponse(event_generator())


def _json_dumps(obj):
    import json
    return json.dumps(obj)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main UI."""
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text())
