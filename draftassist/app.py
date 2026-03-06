"""FastAPI application: routes, SSE streaming, poll loop."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from draftsim.adp import generate_consensus_adp, generate_platform_adp
from draftsim.players import Player, load_players
from draftsim.value import compute_replacement_levels, vbd

from .bridge import build_player_index, config_from_sleeper_meta, rebuild_draft_state
from .recommender import get_all_recommendations
from .sleeper import fetch_all_players, fetch_draft_meta, fetch_draft_picks

log = logging.getLogger("draftassist")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

STATIC_DIR = Path(__file__).parent / "static"
SESSION_TIMEOUT = 2 * 60 * 60  # 2 hours in seconds
CLEANUP_INTERVAL = 5 * 60  # check every 5 minutes


@dataclass
class DraftSession:
    """Holds live draft state for polling."""

    draft_id: str = ""
    user_slot: int = 0  # 0-indexed draft slot
    players: list[Player] = field(default_factory=list)
    id_to_player: dict[str, Player] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    last_pick_count: int = 0
    last_meta_refresh: float = 0.0
    state_payload: dict = field(default_factory=dict)
    poll_task: asyncio.Task | None = None
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    connected: bool = False
    adp_platform: str = "consensus"
    adp_order: list[str] = field(default_factory=list)
    risk_profile: str = "balanced"  # "safe", "balanced", "aggressive"
    last_activity: float = field(default_factory=time.monotonic)


sessions: dict[str, DraftSession] = {}
_cleanup_task: asyncio.Task | None = None


def _get_session(session_id: str) -> DraftSession:
    """Look up a session by ID or raise 404."""
    sess = sessions.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    sess.last_activity = time.monotonic()
    return sess


async def _cleanup_loop():
    """Periodically remove sessions inactive for >2 hours."""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL)
        now = time.monotonic()
        expired = [
            sid for sid, sess in sessions.items()
            if (now - sess.last_activity) > SESSION_TIMEOUT
        ]
        for sid in expired:
            sess = sessions.pop(sid, None)
            if sess and sess.poll_task and not sess.poll_task.done():
                sess.poll_task.cancel()
            log.info("Cleaned up expired session %s", sid)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cleanup_task
    _cleanup_task = asyncio.create_task(_cleanup_loop())
    yield
    # Cancel cleanup task
    if _cleanup_task and not _cleanup_task.done():
        _cleanup_task.cancel()
    # Cancel all session poll tasks on shutdown
    for sid, sess in sessions.items():
        if sess.poll_task and not sess.poll_task.done():
            sess.poll_task.cancel()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _compute_adp_order(players: list[Player], platform: str) -> list[str]:
    """Generate ADP-ordered player name list for a given platform."""
    sorted_by_proj = sorted(players, key=lambda p: p.projected_total, reverse=True)
    if platform == "consensus":
        entries = generate_consensus_adp(sorted_by_proj)
    else:
        entries = generate_platform_adp(sorted_by_proj, platform)
    return [p.name for p, _adp in entries]


def _build_state_payload(state, meta, picks, user_slot, players, adp_order=None,
                         skip_recommendations=False, risk_profile="balanced"):
    """Build the JSON payload describing current draft state."""
    config = state.config
    is_complete = state.is_complete
    current_pick = state.current_pick + 1  # 1-indexed for display
    current_round = state.current_round
    current_team = state.current_team_idx if not is_complete else -1
    is_my_turn = (not is_complete) and (state.current_team_idx == user_slot)
    picks_until = state.picks_until_next(user_slot) if not is_complete else 0

    # Build ADP rank lookup (1-indexed)
    adp_rank = {}
    if adp_order:
        adp_rank = {name: i + 1 for i, name in enumerate(adp_order)}

    # Recommendations (only when it's user's turn and not skipped)
    recs = {}
    if is_my_turn and not skip_recommendations:
        all_recs = get_all_recommendations(
            state, user_slot, players, adp_order, n=30, risk_profile=risk_profile,
        )
        for strat_name, rec_list in all_recs.items():
            recs[strat_name] = [
                {
                    "rank": r.rank,
                    "name": r.player.name,
                    "position": r.player.position,
                    "team": r.player.team,
                    "projected_total": round(r.player.projected_total, 1),
                    "total_floor": round(r.player.total_floor, 1),
                    "total_ceiling": round(r.player.total_ceiling, 1),
                    "vbd": round(r.vbd_value, 1),
                    "strategy_score": round(r.strategy_score, 1),
                    "adp": adp_rank.get(r.player.name, 999),
                }
                for r in rec_list
            ]

    # All picks sorted by pick number
    all_picks_list = []
    sorted_picks = sorted(picks, key=lambda p: p.get("pick_no", 0))
    for p in sorted_picks:
        pm = p.get("metadata", {})
        all_picks_list.append({
            "pick_no": p.get("pick_no", 0),
            "round": p.get("round", 0),
            "draft_slot": p.get("draft_slot", 0),
            "player_name": f"{pm.get('first_name', '')} {pm.get('last_name', '')}".strip(),
            "position": pm.get("position", ""),
            "team": pm.get("team", ""),
            "is_user": p.get("draft_slot", -1) == user_slot + 1,  # Sleeper uses 1-indexed
        })
    recent = all_picks_list[-15:]

    # User roster
    user_roster = []
    if user_slot < len(state.teams):
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
        "all_picks": all_picks_list,
        "user_roster": user_roster,
        "team_needs": needs,
        "draft_status": meta.get("status", "unknown"),
    }


def _refresh_state(sess: DraftSession, picks):
    """Rebuild draft state from picks and update session payload."""
    config = config_from_sleeper_meta(sess.meta)
    state = rebuild_draft_state(
        config, sess.players, picks, sess.id_to_player
    )
    payload = _build_state_payload(
        state, sess.meta, picks,
        sess.user_slot, sess.players, sess.adp_order,
        risk_profile=sess.risk_profile,
    )
    sess.state_payload = payload
    return payload


def _push_to_subscribers(sess: DraftSession, payload):
    """Push payload to all SSE subscriber queues."""
    dead = []
    for q in sess.subscribers:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass
    # Clean up dead queues (shouldn't happen, but defensive)
    for q in dead:
        sess.subscribers.remove(q)


async def _poll_loop(sess: DraftSession):
    """Background task: poll Sleeper every 1s, push SSE on new picks."""
    log.info("Poll loop started for draft %s", sess.draft_id)
    meta_refresh_interval = 30  # seconds between meta refreshes
    while True:
        await asyncio.sleep(1)
        if not sess.connected:
            continue
        try:
            now = time.monotonic()
            need_meta = (now - sess.last_meta_refresh) >= meta_refresh_interval

            async with httpx.AsyncClient(timeout=10) as client:
                if need_meta:
                    # Fetch picks and meta concurrently
                    picks, meta = await asyncio.gather(
                        fetch_draft_picks(client, sess.draft_id),
                        fetch_draft_meta(client, sess.draft_id),
                    )
                    sess.meta = meta
                    sess.last_meta_refresh = now
                else:
                    picks = await fetch_draft_picks(client, sess.draft_id)

            pick_count = len(picks)
            log.info(
                "Poll: %d picks (last seen: %d), status: %s, subscribers: %d",
                pick_count, sess.last_pick_count,
                sess.meta.get("status", "?"), len(sess.subscribers),
            )

            if pick_count != sess.last_pick_count:
                sess.last_pick_count = pick_count

                # Push state immediately WITHOUT recommendations
                config = config_from_sleeper_meta(sess.meta)
                state = rebuild_draft_state(
                    config, sess.players, picks, sess.id_to_player
                )
                payload = _build_state_payload(
                    state, sess.meta, picks,
                    sess.user_slot, sess.players, sess.adp_order,
                    skip_recommendations=True,
                    risk_profile=sess.risk_profile,
                )
                sess.state_payload = payload
                _push_to_subscribers(sess, payload)
                log.info("Pushed update: pick %d", pick_count)

                # If it's the user's turn, compute recommendations and push again
                is_my_turn = (
                    not state.is_complete
                    and state.current_team_idx == sess.user_slot
                )
                if is_my_turn:
                    payload = _build_state_payload(
                        state, sess.meta, picks,
                        sess.user_slot, sess.players, sess.adp_order,
                        risk_profile=sess.risk_profile,
                    )
                    sess.state_payload = payload
                    _push_to_subscribers(sess, payload)
                    log.info("Pushed recommendations for pick %d", pick_count)

        except httpx.HTTPError as e:
            log.warning("Poll HTTP error: %s", e)
        except Exception as e:
            log.exception("Poll loop error: %s", e)


def _ensure_poll_running(sess: DraftSession):
    """Make sure the background poll task is alive."""
    if sess.poll_task is None or sess.poll_task.done():
        if sess.poll_task and sess.poll_task.done():
            # Log if it died with an exception
            exc = sess.poll_task.exception() if not sess.poll_task.cancelled() else None
            if exc:
                log.error("Poll task died with exception: %s", exc)
            log.info("Restarting poll loop")
        sess.poll_task = asyncio.create_task(_poll_loop(sess))


@app.post("/api/connect")
async def connect_draft(
    draft_id: str = Query(...),
    user_slot: int = Query(..., description="1-indexed draft slot"),
):
    """Connect to a Sleeper draft. Creates a new session."""
    session_id = str(uuid.uuid4())
    sess = DraftSession()
    sessions[session_id] = sess

    log.info("Connecting session %s to draft %s, slot %d", session_id, draft_id, user_slot)

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
    sess.draft_id = draft_id
    sess.user_slot = slot_0
    sess.players = players
    sess.id_to_player = id_to_player
    sess.meta = meta
    sess.last_pick_count = len(picks)
    sess.last_meta_refresh = time.monotonic()
    sess.connected = True
    sess.adp_platform = "consensus"
    sess.adp_order = _compute_adp_order(players, "consensus")
    sess.last_activity = time.monotonic()

    payload = _build_state_payload(
        state, meta, picks, slot_0, players, sess.adp_order,
        risk_profile=sess.risk_profile,
    )
    sess.state_payload = payload

    # Start poll loop
    _ensure_poll_running(sess)

    log.info(
        "Connected: %d teams, %d rounds, %d picks, %d players matched",
        config.num_teams, config.num_rounds, len(picks), len(id_to_player),
    )

    return JSONResponse({
        "status": "connected",
        "session_id": session_id,
        "draft_id": draft_id,
        "user_slot": user_slot,
        "num_teams": config.num_teams,
        "rounds": config.num_rounds,
        "picks_made": len(picks),
        "players_matched": len(id_to_player),
        "total_players": len(players),
        "draft_status": meta.get("status", "unknown"),
    })


@app.post("/api/adp")
async def set_adp_platform(
    session_id: str = Query(..., description="Session ID"),
    platform: str = Query("consensus", description="ADP platform: consensus, sleeper, espn, yahoo"),
):
    """Switch the ADP source used for opponent modeling."""
    sess = _get_session(session_id)
    if not sess.connected:
        return JSONResponse({"error": "Not connected to a draft"}, status_code=400)

    valid = {"consensus", "sleeper", "espn", "yahoo"}
    if platform not in valid:
        return JSONResponse({"error": f"Invalid platform. Choose from: {valid}"}, status_code=400)

    sess.adp_platform = platform
    sess.adp_order = _compute_adp_order(sess.players, platform)

    # Rebuild current state payload with new ADP
    async with httpx.AsyncClient(timeout=10) as client:
        picks = await fetch_draft_picks(client, sess.draft_id)

    _refresh_state(sess, picks)

    return JSONResponse({"status": "ok", "platform": platform})


@app.post("/api/risk")
async def set_risk_profile(
    session_id: str = Query(..., description="Session ID"),
    profile: str = Query("balanced", description="Risk profile: safe, balanced, aggressive"),
):
    """Switch the risk/variance profile for recommendations."""
    sess = _get_session(session_id)
    if not sess.connected:
        return JSONResponse({"error": "Not connected to a draft"}, status_code=400)

    valid = {"safe", "balanced", "aggressive"}
    if profile not in valid:
        return JSONResponse({"error": f"Invalid profile. Choose from: {valid}"}, status_code=400)

    sess.risk_profile = profile

    # Rebuild current state payload with new risk profile
    async with httpx.AsyncClient(timeout=10) as client:
        picks = await fetch_draft_picks(client, sess.draft_id)

    _refresh_state(sess, picks)

    return JSONResponse({"status": "ok", "profile": profile})


@app.get("/api/more")
async def get_more_recommendations(
    session_id: str = Query(..., description="Session ID"),
    n: int = Query(10, description="Number of additional recommendations"),
    offset: int = Query(10, description="Skip first N recommendations"),
):
    """Fetch additional recommendations beyond the initial set."""
    sess = _get_session(session_id)
    if not sess.connected:
        return JSONResponse({"error": "Not connected to a draft"}, status_code=400)

    async with httpx.AsyncClient(timeout=10) as client:
        picks = await fetch_draft_picks(client, sess.draft_id)

    config = config_from_sleeper_meta(sess.meta)
    state = rebuild_draft_state(config, sess.players, picks, sess.id_to_player)

    slot = sess.user_slot
    if state.is_complete or state.current_team_idx != slot:
        return JSONResponse({"recommendations": {}})

    total = offset + n
    all_recs = get_all_recommendations(
        state, slot, sess.players, sess.adp_order, n=total,
        risk_profile=sess.risk_profile,
    )

    # Build ADP rank lookup
    adp_rank = {name: i + 1 for i, name in enumerate(sess.adp_order)} if sess.adp_order else {}

    recs = {}
    for strat_name, rec_list in all_recs.items():
        # Only return recs beyond the offset
        recs[strat_name] = [
            {
                "rank": r.rank,
                "name": r.player.name,
                "position": r.player.position,
                "team": r.player.team,
                "projected_total": round(r.player.projected_total, 1),
                "total_floor": round(r.player.total_floor, 1),
                "total_ceiling": round(r.player.total_ceiling, 1),
                "vbd": round(r.vbd_value, 1),
                "strategy_score": round(r.strategy_score, 1),
                "adp": adp_rank.get(r.player.name, 999),
            }
            for r in rec_list[offset:]
        ]

    return JSONResponse({"recommendations": recs})


@app.get("/api/state")
async def get_state(
    session_id: str = Query(..., description="Session ID"),
):
    """Current state snapshot for initial load / reconnect."""
    sess = _get_session(session_id)
    if not sess.connected:
        return JSONResponse({"error": "Not connected to a draft"}, status_code=400)
    # Make sure poll is still running
    _ensure_poll_running(sess)
    return JSONResponse(sess.state_payload)


@app.get("/api/stream")
async def stream(
    request: Request,
    session_id: str = Query(..., description="Session ID"),
):
    """SSE endpoint: yields draft_update events on new picks."""
    sess = _get_session(session_id)
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    sess.subscribers.append(queue)
    log.info("SSE subscriber connected (total: %d)", len(sess.subscribers))

    # Make sure poll is running
    _ensure_poll_running(sess)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15)
                    yield {"event": "draft_update", "data": json.dumps(payload)}
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {"event": "ping", "data": ""}
        finally:
            if queue in sess.subscribers:
                sess.subscribers.remove(queue)
            log.info("SSE subscriber disconnected (remaining: %d)", len(sess.subscribers))

    return EventSourceResponse(event_generator())


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main UI."""
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text())
