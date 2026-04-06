"""FastAPI application: routes, SSE streaming, poll loop."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from draftsim.adp import generate_consensus_adp, generate_platform_adp
from draftsim.config import LeagueConfig
from draftsim.draft import DraftState, build_snake_order
from draftsim.players import Player, load_players
from draftsim.value import compute_dynamic_replacement_levels, compute_replacement_levels, vbd

from draftsim.live_sim import SimSnapshot, run_live_simulation

from .bridge import (
    attach_fangraphs_projections, attach_sleeper_projections,
    build_player_index, config_from_sleeper_meta,
    default_config_for_sport, rebuild_draft_state, rebuild_from_manual_picks,
    swap_projection_source,
)
from .fangraphs import fetch_fangraphs_projections
from .recommender import get_recommendations
from .scoring import extract_scoring_from_meta
from .sleeper import (
    fetch_all_players, fetch_draft_meta, fetch_draft_picks, fetch_projections,
    fetch_league_info, fetch_league_rosters, fetch_league_users,
)
from .trade import (
    TradeSession, TradeTeam, evaluate_trade, resolve_sleeper_rosters,
    roster_needs, best_at_position, player_trade_value,
)
from .yahoo import (
    fetch_yahoo_league_info, fetch_yahoo_rosters,
    yahoo_config_from_positions, resolve_yahoo_rosters,
)

log = logging.getLogger("draftassist")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

STATIC_DIR = Path(__file__).parent / "static"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
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
    # Live simulation state
    sim_task: asyncio.Task | None = None
    sim_cancel: asyncio.Event = field(default_factory=asyncio.Event)
    sim_snapshot: dict | None = None
    # Projection source
    projection_source: str = "model"  # "model", "sleeper", or "fangraphs"
    sleeper_projections_matched: int = 0  # how many players have Sleeper data
    fangraphs_projections_matched: int = 0  # how many players have FanGraphs data
    # Manual draft mode
    mode: str = "sleeper"  # "sleeper" or "manual"
    sport: str = "nfl"
    picks: list[dict] = field(default_factory=list)  # manual mode picks
    draft_state: DraftState | None = None  # live state for manual mode
    config: LeagueConfig | None = None  # league config for manual mode


# CR opus: Module-level mutable state — all sessions share the same `players` list
# objects and Player instances in memory. Mutations in swap_projection_source or
# attach_sleeper_projections on one session's players affect ALL sessions that
# loaded the same player pool. Each session should own its own deep copy.
sessions: dict[str, DraftSession] = {}
trade_sessions: dict[str, TradeSession] = {}
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
            if sess:
                if sess.poll_task and not sess.poll_task.done():
                    sess.poll_task.cancel()
                if sess.sim_task and not sess.sim_task.done():
                    sess.sim_cancel.set()
                    sess.sim_task.cancel()
            log.info("Cleaned up expired session %s", sid)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cleanup_task
    _cleanup_task = asyncio.create_task(_cleanup_loop())
    yield
    # Cancel cleanup task
    if _cleanup_task and not _cleanup_task.done():
        _cleanup_task.cancel()
    # Cancel all session poll/sim tasks on shutdown
    for sid, sess in sessions.items():
        if sess.poll_task and not sess.poll_task.done():
            sess.poll_task.cancel()
        if sess.sim_task and not sess.sim_task.done():
            sess.sim_cancel.set()
            sess.sim_task.cancel()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Data sources tracked for freshness display
# CR opus: Filename hardcodes "2025" — should be updated or parameterized for 2026 season.
_DATA_SOURCES = [
    {
        "name": "ADP Rankings",
        "file": DATA_DIR / "FantasyPros_2025_Overall_ADP_Rankings.csv",
        "url": "https://www.fantasypros.com/nfl/adp/ppr-overall.php",
        "how": "Manual download (CSV export)",
    },
    {
        "name": "Player Projections",
        "file": DATA_DIR / "projections" / "all_projections.csv",
        "url": "",
        "how": "Run scripts/project_2026_v2.py",
    },
    {
        "name": "Rosters",
        "file": DATA_DIR / "rosters.csv",
        "url": "",
        "how": "Run scripts/update_rosters.py (fetches from nflverse)",
    },
    {
        "name": "Lahman Baseball DB",
        "file": DATA_DIR / "lahman_1871-2025_csv",
        "url": "https://sabr.app.box.com/s/y1prhc795jk8zvmelfd3jq7tl389y6cd",
        "how": "Manual download + unzip",
    },
]


@app.get("/api/data-info")
async def data_info():
    """Return last-modified timestamps for data files."""
    sources = []
    for src in _DATA_SOURCES:
        path = src["file"]
        if path.exists():
            mtime = os.path.getmtime(path)
            last_modified = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        else:
            last_modified = None
        sources.append({
            "name": src["name"],
            "last_modified": last_modified,
            "url": src.get("url", ""),
            "how": src.get("how", ""),
        })
    return JSONResponse({"sources": sources})


def _compute_adp_order(players: list[Player], platform: str, sport: str = "nfl") -> list[str]:
    """Generate ADP-ordered player name list for a given platform."""
    sorted_by_proj = sorted(players, key=lambda p: p.projected_total, reverse=True)
    if sport != "nfl":
        # No real ADP data for non-NFL sports; use projection ranking
        return [p.name for p in sorted_by_proj]
    if platform == "consensus":
        entries = generate_consensus_adp(sorted_by_proj)
    else:
        entries = generate_platform_adp(sorted_by_proj, platform)
    return [p.name for p, _adp in entries]


def _build_state_payload(state, meta, picks, user_slot, players, adp_order=None,
                         skip_recommendations=False, risk_profile="balanced",
                         projection_source="model"):
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

    # Recommendations — always compute so user can plan ahead
    # Pre-compute for all 3 risk profiles so frontend can toggle client-side
    recs = {}
    if not is_complete and not skip_recommendations:
        for rp in ("safe", "balanced", "aggressive"):
            rec_list = get_recommendations(
                state, user_slot, players, adp_order, n=30, risk_profile=rp,
            )
            recs[rp] = [
                {
                    "rank": r.rank,
                    "name": r.player.name,
                    "position": r.player.position,
                    "team": r.player.team,
                    "projected_total": round(r.player.projected_total, 1),
                    "total_floor": round(r.player.total_floor, 1),
                    "total_ceiling": round(r.player.total_ceiling, 1),
                    "projected_games": round(r.player.projected_games, 1) if r.player.projected_games > 0 else None,
                    "vorp": round(r.vbd_value, 1),
                    "vona": round(r.vona_value, 1),
                    "vols": round(r.vols_value, 1),
                    "vbd_score": round(r.vbd_score_value, 1),
                    "bye_week": r.player.bye_week,
                    "is_rookie": r.player.is_rookie,
                    "age": r.player.age,
                    "adp": adp_rank.get(r.player.name, 999),
                }
                for r in rec_list
            ]

    # All picks sorted by pick number
    rookie_names = {p.name for p in players if p.is_rookie}
    all_picks_list = []
    sorted_picks = sorted(picks, key=lambda p: p.get("pick_no", 0))
    for p in sorted_picks:
        pm = p.get("metadata", {})
        pname = f"{pm.get('first_name', '')} {pm.get('last_name', '')}".strip()
        all_picks_list.append({
            "pick_no": p.get("pick_no", 0),
            "round": p.get("round", 0),
            "draft_slot": p.get("draft_slot", 0),
            "player_name": pname,
            "position": pm.get("position", ""),
            "team": pm.get("team", ""),
            "is_rookie": pname in rookie_names,
            # CR opus: In manual mode, draft_slot is already 1-indexed (set in manual_pick),
            # so this comparison is correct. But if manual mode ever changes to 0-indexed
            # draft_slot, this will break. The "Sleeper uses 1-indexed" comment is misleading
            # since this code path is also used for manual drafts.
            "is_user": p.get("draft_slot", -1) == user_slot + 1,  # Sleeper uses 1-indexed
        })
    recent = all_picks_list[-15:]

    # User roster
    user_roster = []
    if user_slot < len(state.teams):
        roster = state.teams[user_slot]
    else:
        roster = []

    replacement = (
        compute_dynamic_replacement_levels(state.available, config, state.teams)
        if players else {}
    )
    for p in roster:
        user_roster.append({
            "name": p.name,
            "position": p.position,
            "team": p.team,
            "projected_total": round(p.projected_total, 1),
            "bye_week": p.bye_week,
            "is_rookie": p.is_rookie,
            "age": p.age,
            "vbd": round(vbd(p, replacement), 1),
        })

    # Team needs — map internal flex names to user-friendly labels
    _NEEDS_DISPLAY = {"FLEX": "Util", "PFLEX": "P"}
    needs = {
        _NEEDS_DISPLAY.get(k, k): v
        for k, v in (state.team_needs(user_slot) if user_slot < config.num_teams else {}).items()
    }

    # Available rookies — surfaced separately so the UI can show them
    # when the Rookies filter is active, even if they don't rank in the
    # strategy-scored top-N recommendations
    available_rookies = []
    for p in sorted(state.available, key=lambda p: p.projected_total, reverse=True):
        if p.is_rookie:
            available_rookies.append({
                "name": p.name,
                "position": p.position,
                "team": p.team,
                "projected_total": round(p.projected_total, 1),
                "total_floor": round(p.total_floor, 1),
                "total_ceiling": round(p.total_ceiling, 1),
                "projected_games": round(p.projected_games, 1) if p.projected_games > 0 else None,
                "bye_week": p.bye_week,
                "is_rookie": True,
                "age": p.age,
                "adp": adp_rank.get(p.name, 999),
            })

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
        "projection_source": projection_source,
        # CR opus: FanGraphs projections also use estimated floor/ceiling (synthesized
        # from position ratios), but this flag only checks for "sleeper". Should also
        # be True when projection_source == "fangraphs".
        "floor_estimated": projection_source == "sleeper",
        "available_rookies": available_rookies,
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
        projection_source=sess.projection_source,
    )
    sess.state_payload = payload
    return payload


async def _get_picks_and_config(sess: DraftSession):
    """Return (picks, config) from either manual state or Sleeper API."""
    if sess.mode == "manual":
        return sess.picks, sess.config
    async with httpx.AsyncClient(timeout=10) as client:
        picks = await fetch_draft_picks(client, sess.draft_id)
    return picks, config_from_sleeper_meta(sess.meta)


def _refresh_manual_state(sess: DraftSession):
    """Rebuild manual draft state from picks and update session payload."""
    state = rebuild_from_manual_picks(sess.config, sess.players, sess.picks)
    sess.draft_state = state
    meta = {"status": "complete" if state.is_complete else "in_progress"}
    payload = _build_state_payload(
        state, meta, sess.picks,
        sess.user_slot, sess.players, sess.adp_order,
        risk_profile=sess.risk_profile,
        projection_source=sess.projection_source,
    )
    sess.state_payload = payload
    return state, payload


def _push_to_subscribers(sess: DraftSession, payload):
    """Push payload to all SSE subscriber queues."""
    # CR opus: `dead` list is always empty — nothing ever appends to it.
    # QueueFull exceptions silently drop updates; consider logging or evicting stale subscribers.
    dead = []
    for q in sess.subscribers:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass
    # Clean up dead queues (shouldn't happen, but defensive)
    for q in dead:
        sess.subscribers.remove(q)


def _push_sim_to_subscribers(sess: DraftSession, payload: dict):
    """Push a sim_update event to all SSE subscriber queues."""
    wrapped = {"__event__": "sim_update", **payload}
    # CR opus: Same dead-list bug as _push_to_subscribers — `dead` is never populated.
    dead = []
    for q in sess.subscribers:
        try:
            q.put_nowait(wrapped)
        except asyncio.QueueFull:
            pass
    for q in dead:
        sess.subscribers.remove(q)


async def _cancel_sim(sess: DraftSession):
    """Cancel any running simulation."""
    if sess.sim_task and not sess.sim_task.done():
        sess.sim_cancel.set()
        try:
            await asyncio.wait_for(sess.sim_task, timeout=5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            sess.sim_task.cancel()
    sess.sim_task = None


def _start_sim(sess: DraftSession, state: DraftState):
    """Start a live simulation task for the current draft state."""
    sess.sim_cancel = asyncio.Event()
    sess.sim_snapshot = None

    strategy_names = ["vbd", "vona", "bpa"]

    async def _on_snapshot(snap: SimSnapshot):
        payload = snap.to_dict()
        sess.sim_snapshot = payload
        _push_sim_to_subscribers(sess, payload)

    async def _run():
        try:
            await run_live_simulation(
                state=state,
                user_slot=sess.user_slot,
                players=sess.players,
                adp_order=sess.adp_order or None,
                strategies=strategy_names,
                cancel_event=sess.sim_cancel,
                on_snapshot=_on_snapshot,
                batch_size=50,
                max_sims=500,
                top_n_candidates=15,
                opponent_platform=sess.adp_platform,
            )
        except Exception as exc:
            log.exception("Live sim error: %s", exc)

    sess.sim_task = asyncio.create_task(_run())


async def _poll_loop(sess: DraftSession):
    """Background task: poll Sleeper every 1s, push SSE on new picks."""
    log.info("Poll loop started for draft %s", sess.draft_id)
    meta_refresh_interval = 30  # seconds between meta refreshes
    # CR opus: This loop creates a new httpx.AsyncClient on every 1-second poll iteration.
    # This prevents HTTP connection reuse and adds overhead. Consider creating one client
    # for the lifetime of the poll loop.
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

                # Cancel any running simulation
                await _cancel_sim(sess)

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
                    projection_source=sess.projection_source,
                )
                sess.state_payload = payload
                _push_to_subscribers(sess, payload)
                log.info("Pushed update: pick %d", pick_count)

                # Push again with recommendations (computed async)
                if not state.is_complete:
                    payload = _build_state_payload(
                        state, sess.meta, picks,
                        sess.user_slot, sess.players, sess.adp_order,
                        risk_profile=sess.risk_profile,
                        projection_source=sess.projection_source,
                    )
                    sess.state_payload = payload
                    _push_to_subscribers(sess, payload)
                    log.info("Pushed recommendations for pick %d", pick_count)

                # Start sim if user picks within 3 picks or it's their turn
                picks_until = state.picks_until_next(sess.user_slot) if not state.is_complete else 999
                if not state.is_complete and picks_until <= 3:
                    _start_sim(sess, state)
                    log.info("Started live sim (picks_until=%d)", picks_until)

        except httpx.HTTPError as e:
            # CR opus: No backoff on repeated HTTP errors — if the Sleeper API goes down,
            # this fires a request every 1 second indefinitely with no exponential backoff.
            log.warning("Poll HTTP error: %s", e)
        except Exception as e:
            log.exception("Poll loop error: %s", e)


def _ensure_poll_running(sess: DraftSession):
    """Make sure the background poll task is alive."""
    if sess.poll_task is None or sess.poll_task.done():
        if sess.poll_task and sess.poll_task.done():
            # Log if it died with an exception
            # CR opus: Calling .exception() on a done task that raised will re-raise
            # the exception if it hasn't been retrieved yet. This is wrapped in a
            # conditional so it's fine, but note that the poll loop's `while True`
            # with bare `except Exception` means it should never actually die —
            # only CancelledError would escape. If it does die, this restarts it
            # with no limit, which could cause infinite rapid restart loops.
            exc = sess.poll_task.exception() if not sess.poll_task.cancelled() else None
            if exc:
                log.error("Poll task died with exception: %s", exc)
            log.info("Restarting poll loop")
        sess.poll_task = asyncio.create_task(_poll_loop(sess))


@app.post("/api/connect")
async def connect_draft(
    # CR opus: draft_id is not validated — any string is passed directly into Sleeper API URL.
    # Should validate format (e.g., numeric Sleeper draft IDs) to prevent SSRF or unexpected errors.
    draft_id: str = Query(...),
    # CR opus: No upper-bound validation on user_slot. A value > num_teams will silently produce
    # wrong results (the 0-indexed slot will be out of range for state.teams).
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
        # Pre-fetch Sleeper projections for instant toggling later
        try:
            sleeper_proj = await fetch_projections(client)
        except Exception:
            sleeper_proj = {}
            log.warning("Failed to fetch Sleeper projections — tab will be disabled")

    # CR opus: load_players() with no sport argument defaults to NFL, but there's no
    # explicit sport parameter on this endpoint. If the Sleeper draft is for a non-NFL
    # sport, the wrong player pool will be loaded.
    players = load_players()
    id_to_player = build_player_index(players, sleeper_players)

    # Attach Sleeper projections to players (also saves model backups)
    scoring = extract_scoring_from_meta(meta)
    sleeper_matched = attach_sleeper_projections(players, sleeper_proj, scoring)

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
    sess.adp_order = _compute_adp_order(players, "consensus", sport="nfl")
    sess.last_activity = time.monotonic()
    sess.sleeper_projections_matched = sleeper_matched

    payload = _build_state_payload(
        state, meta, picks, slot_0, players, sess.adp_order,
        risk_profile=sess.risk_profile,
        projection_source=sess.projection_source,
    )
    sess.state_payload = payload

    # Start poll loop
    _ensure_poll_running(sess)

    log.info(
        "Connected: %d teams, %d rounds, %d picks, %d players matched, "
        "%d sleeper projections",
        config.num_teams, config.num_rounds, len(picks), len(id_to_player),
        sleeper_matched,
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
        "sleeper_projections_available": sleeper_matched > 0,
        "sleeper_projections_matched": sleeper_matched,
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

    valid = {"consensus", "sleeper", "espn", "cbs", "nfl", "rtsports", "fantrax"}
    if platform not in valid:
        return JSONResponse({"error": f"Invalid platform. Choose from: {valid}"}, status_code=400)

    sess.adp_platform = platform
    sess.adp_order = _compute_adp_order(sess.players, platform, sport=sess.sport)

    # Rebuild current state payload with new ADP
    if sess.mode == "manual":
        _refresh_manual_state(sess)
    else:
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
    if sess.mode == "manual":
        _refresh_manual_state(sess)
    else:
        async with httpx.AsyncClient(timeout=10) as client:
            picks = await fetch_draft_picks(client, sess.draft_id)
        _refresh_state(sess, picks)

    return JSONResponse({"status": "ok", "profile": profile})


@app.post("/api/slot")
async def change_slot(
    session_id: str = Query(..., description="Session ID"),
    user_slot: int = Query(..., description="New 1-indexed draft slot"),
):
    """Change the user's draft slot mid-draft."""
    sess = _get_session(session_id)
    if not sess.connected:
        return JSONResponse({"error": "Not connected to a draft"}, status_code=400)

    # Determine num_teams from whichever mode we're in
    if sess.mode == "manual":
        num_teams = sess.config.num_teams
    else:
        num_teams = config_from_sleeper_meta(sess.meta).num_teams

    if user_slot < 1 or user_slot > num_teams:
        return JSONResponse(
            {"error": f"Slot must be between 1 and {num_teams}"},
            status_code=400,
        )

    slot_0 = user_slot - 1
    sess.user_slot = slot_0

    # Rebuild payload with new slot perspective
    if sess.mode == "manual":
        _refresh_manual_state(sess)
    else:
        async with httpx.AsyncClient(timeout=10) as client:
            picks = await fetch_draft_picks(client, sess.draft_id)
        _refresh_state(sess, picks)

    _push_to_subscribers(sess, sess.state_payload)

    log.info("Session %s changed slot to %d", session_id, user_slot)
    return JSONResponse({"status": "ok", "user_slot": user_slot})


@app.post("/api/projections")
async def set_projection_source(
    session_id: str = Query(..., description="Session ID"),
    source: str = Query("model", description="Projection source: model, sleeper, or fangraphs"),
):
    """Switch between model, Sleeper, and FanGraphs projections."""
    sess = _get_session(session_id)
    if not sess.connected:
        return JSONResponse({"error": "Not connected to a draft"}, status_code=400)

    valid = {"model", "sleeper", "fangraphs"}
    if source not in valid:
        return JSONResponse({"error": f"Invalid source. Choose from: {valid}"}, status_code=400)

    if source == "sleeper" and sess.sleeper_projections_matched == 0:
        return JSONResponse(
            {"error": "Sleeper projections not available for this session"},
            status_code=400,
        )
    if source == "fangraphs" and sess.fangraphs_projections_matched == 0:
        return JSONResponse(
            {"error": "FanGraphs projections not available for this session"},
            status_code=400,
        )

    swap_projection_source(sess.players, source)
    sess.projection_source = source

    # Recalculate ADP order (uses current projected_total)
    sess.adp_order = _compute_adp_order(sess.players, sess.adp_platform, sport=sess.sport)

    # Cancel running sim and rebuild state
    await _cancel_sim(sess)

    if sess.mode == "manual":
        _refresh_manual_state(sess)
    else:
        async with httpx.AsyncClient(timeout=10) as client:
            picks = await fetch_draft_picks(client, sess.draft_id)
        _refresh_state(sess, picks)

    return JSONResponse({"status": "ok", "source": source})


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

    if sess.mode == "manual":
        picks, config = sess.picks, sess.config
        state = rebuild_from_manual_picks(config, sess.players, picks)
    else:
        async with httpx.AsyncClient(timeout=10) as client:
            picks = await fetch_draft_picks(client, sess.draft_id)
        config = config_from_sleeper_meta(sess.meta)
        state = rebuild_draft_state(config, sess.players, picks, sess.id_to_player)

    slot = sess.user_slot
    if state.is_complete:
        return JSONResponse({"recommendations": {}})

    total = offset + n

    # Build ADP rank lookup
    adp_rank = {name: i + 1 for i, name in enumerate(sess.adp_order)} if sess.adp_order else {}

    recs = {}
    for rp in ("safe", "balanced", "aggressive"):
        rec_list = get_recommendations(
            state, slot, sess.players, sess.adp_order, n=total,
            risk_profile=rp,
        )
        recs[rp] = [
            {
                "rank": r.rank,
                "name": r.player.name,
                "position": r.player.position,
                "team": r.player.team,
                "projected_total": round(r.player.projected_total, 1),
                "total_floor": round(r.player.total_floor, 1),
                "total_ceiling": round(r.player.total_ceiling, 1),
                "projected_games": round(r.player.projected_games, 1) if r.player.projected_games > 0 else None,
                "vorp": round(r.vbd_value, 1),
                "vona": round(r.vona_value, 1),
                "vols": round(r.vols_value, 1),
                "vbd_score": round(r.vbd_score_value, 1),
                "bye_week": r.player.bye_week,
                "is_rookie": r.player.is_rookie,
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
    # Make sure poll is still running (Sleeper mode only)
    if sess.mode == "sleeper":
        _ensure_poll_running(sess)
    payload = dict(sess.state_payload)
    payload["sport"] = sess.sport
    payload["mode"] = sess.mode
    return JSONResponse(payload)


@app.get("/api/sim")
async def get_sim(
    session_id: str = Query(..., description="Session ID"),
):
    """Return latest simulation snapshot."""
    sess = _get_session(session_id)
    if sess.sim_snapshot:
        return JSONResponse(sess.sim_snapshot)
    return JSONResponse({"sims_completed": 0, "sims_target": 0, "strategies": {}})


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

    # Make sure poll is running (Sleeper mode only)
    if sess.mode == "sleeper":
        _ensure_poll_running(sess)

    async def event_generator():
        try:
            while True:
                # CR opus: request.is_disconnected() is awaited on every loop iteration,
                # but it only runs AFTER the 15-second wait_for timeout. A disconnected
                # client won't be detected for up to 15 seconds after disconnect.
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15)
                    # Check if this is a sim_update event (tagged by _push_sim_to_subscribers)
                    if isinstance(payload, dict) and payload.get("__event__") == "sim_update":
                        sim_data = {k: v for k, v in payload.items() if k != "__event__"}
                        yield {"event": "sim_update", "data": json.dumps(sim_data)}
                    else:
                        yield {"event": "draft_update", "data": json.dumps(payload)}
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {"event": "ping", "data": ""}
        finally:
            if queue in sess.subscribers:
                sess.subscribers.remove(queue)
            log.info("SSE subscriber disconnected (remaining: %d)", len(sess.subscribers))

    return EventSourceResponse(event_generator())


@app.post("/api/create")
async def create_manual_draft(
    request: Request,
    sport: str = Query("nfl", description="Sport: nfl or mlb"),
    num_teams: int = Query(12, ge=4, le=20),
    roster_size: int = Query(15, ge=10, le=25),
    user_slot: int = Query(1, ge=1, le=20, description="1-indexed draft slot"),
):
    """Create a manual draft session (no Sleeper connection).

    Optionally accepts a JSON body with a list of player names to replay
    (e.g. when restoring a draft after server restart).
    """
    # Parse optional pick names from request body
    pick_names: list[str] = []
    if request.headers.get("content-type", "").startswith("application/json"):
        try:
            body = await request.json()
            if isinstance(body, list):
                pick_names = [str(n) for n in body]
        except Exception:
            pass
    if sport not in ("nfl", "mlb"):
        return JSONResponse({"detail": f"Unsupported sport: {sport}"}, status_code=400)

    session_id = str(uuid.uuid4())
    sess = DraftSession()
    sessions[session_id] = sess

    try:
        players = load_players(sport=sport)
    except FileNotFoundError as e:
        sessions.pop(session_id, None)
        return JSONResponse({"detail": str(e)}, status_code=400)

    # Fetch external projections for the projection toggle
    sleeper_matched = 0
    fangraphs_matched = 0
    if sport == "nfl":
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                sleeper_players = await fetch_all_players(client)
                sleeper_proj = await fetch_projections(client)
            build_player_index(players, sleeper_players)  # sets sleeper_id on each player
            sleeper_matched = attach_sleeper_projections(players, sleeper_proj, scoring=None)
        except Exception:
            log.warning("Failed to fetch Sleeper projections for manual draft")
    elif sport == "mlb":
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                fg_proj = await fetch_fangraphs_projections(client)
            fangraphs_matched = attach_fangraphs_projections(players, fg_proj)
        except Exception:
            log.warning("Failed to fetch FanGraphs projections for manual draft")

    config = default_config_for_sport(sport, num_teams, roster_size)

    # Replay saved picks if provided
    if pick_names:
        # Build pick dicts from player names so rebuild_from_manual_picks can match
        picks = []
        for i, name in enumerate(pick_names):
            parts = name.split(maxsplit=1)
            picks.append({
                "pick_no": i + 1,
                "metadata": {
                    "first_name": parts[0] if parts else "",
                    "last_name": parts[1] if len(parts) > 1 else "",
                },
            })
        state = rebuild_from_manual_picks(config, players, picks)
        # Build sess.picks in the same format as manual_pick creates
        sess_picks = []
        name_lookup = {p.name.lower(): p for p in players}
        snake_order = build_snake_order(num_teams, roster_size)
        for i, name in enumerate(pick_names):
            matched = name_lookup.get(name.lower())
            pick_no = i + 1
            round_no = (pick_no - 1) // num_teams + 1
            draft_slot = snake_order[i] + 1 if i < len(snake_order) else 0  # 1-indexed
            parts = name.split(maxsplit=1)
            sess_picks.append({
                "pick_no": pick_no,
                "round": round_no,
                "draft_slot": draft_slot,
                "player_id": f"manual-{pick_no}",
                "metadata": {
                    "first_name": parts[0] if parts else "",
                    "last_name": parts[1] if len(parts) > 1 else "",
                    "position": matched.position if matched else "",
                    "team": matched.team if matched else "",
                },
            })
        sess.picks = sess_picks
    else:
        state = DraftState.create(config, players)

    slot_0 = user_slot - 1
    sess.mode = "manual"
    sess.sport = sport
    sess.user_slot = slot_0
    sess.players = players
    sess.config = config
    sess.draft_state = state
    sess.connected = True
    sess.adp_platform = "consensus"
    sess.adp_order = _compute_adp_order(players, "consensus", sport=sport)
    sess.last_activity = time.monotonic()
    sess.sleeper_projections_matched = sleeper_matched
    sess.fangraphs_projections_matched = fangraphs_matched

    meta = {"status": "complete" if state.is_complete else "in_progress"}
    payload = _build_state_payload(
        state, meta, sess.picks, slot_0, players, sess.adp_order,
        risk_profile=sess.risk_profile,
        projection_source=sess.projection_source,
    )
    sess.state_payload = payload

    log.info(
        "Created manual %s draft: %d teams, %d rounds, slot %d, %d picks restored",
        sport, num_teams, roster_size, user_slot, len(pick_names),
    )

    return JSONResponse({
        "status": "connected",
        "session_id": session_id,
        "draft_id": f"manual-{session_id[:8]}",
        "user_slot": user_slot,
        "num_teams": num_teams,
        "rounds": roster_size,
        "picks_made": len(sess.picks),
        "players_matched": len(players),
        "total_players": len(players),
        "draft_status": meta["status"],
        "mode": "manual",
        "sport": sport,
        "sleeper_projections_available": sleeper_matched > 0,
        "sleeper_projections_matched": sleeper_matched,
        "fangraphs_projections_available": fangraphs_matched > 0,
        "fangraphs_projections_matched": fangraphs_matched,
    })


@app.post("/api/pick")
async def manual_pick(
    session_id: str = Query(...),
    player_name: str = Query(..., description="Exact player name"),
):
    # CR opus: No concurrency guard — if two rapid /api/pick calls arrive for the same
    # session, both will read draft_state.available, both may find the same player,
    # and both may call make_pick. The `await _cancel_sim` below yields control,
    # creating a window for interleaving. Consider an asyncio.Lock per session.
    """Enter a pick in manual draft mode."""
    sess = _get_session(session_id)
    if sess.mode != "manual":
        return JSONResponse({"error": "Not a manual draft"}, status_code=400)
    if sess.draft_state is None or sess.draft_state.is_complete:
        return JSONResponse({"error": "Draft is complete"}, status_code=400)

    # Find player by exact name (case-insensitive)
    target = player_name.lower()
    matched = None
    for p in sess.draft_state.available:
        if p.name.lower() == target:
            matched = p
            break

    if matched is None:
        return JSONResponse({"error": f"Player '{player_name}' not found in available pool"}, status_code=404)

    state = sess.draft_state
    pick_no = state.current_pick + 1
    current_round = state.current_round
    draft_slot = state.current_team_idx + 1  # 1-indexed

    state.make_pick(matched)

    # Build Sleeper-compatible pick dict
    name_parts = matched.name.split(maxsplit=1)
    pick_dict = {
        "pick_no": pick_no,
        "round": current_round,
        "draft_slot": draft_slot,
        "player_id": f"manual-{pick_no}",
        "metadata": {
            "first_name": name_parts[0] if name_parts else "",
            "last_name": name_parts[1] if len(name_parts) > 1 else "",
            "position": matched.position,
            "team": matched.team,
        },
    }
    sess.picks.append(pick_dict)

    # Cancel any running sim
    await _cancel_sim(sess)

    # Rebuild payload
    meta = {"status": "complete" if state.is_complete else "in_progress"}
    payload = _build_state_payload(
        state, meta, sess.picks,
        sess.user_slot, sess.players, sess.adp_order,
        risk_profile=sess.risk_profile,
        projection_source=sess.projection_source,
    )
    sess.state_payload = payload
    _push_to_subscribers(sess, payload)

    # Start sim if user picks within 3 picks
    if not state.is_complete:
        picks_until = state.picks_until_next(sess.user_slot)
        if picks_until <= 3:
            _start_sim(sess, state)

    return JSONResponse({
        "status": "ok",
        "pick_no": pick_no,
        "player_name": matched.name,
        "position": matched.position,
    })


@app.post("/api/undo")
async def undo_pick(
    session_id: str = Query(...),
):
    """Undo the last pick in manual draft mode."""
    sess = _get_session(session_id)
    if sess.mode != "manual":
        return JSONResponse({"error": "Not a manual draft"}, status_code=400)
    if not sess.picks:
        return JSONResponse({"error": "No picks to undo"}, status_code=400)

    sess.picks.pop()

    await _cancel_sim(sess)

    # Rebuild state from remaining picks
    state, payload = _refresh_manual_state(sess)

    _push_to_subscribers(sess, payload)

    return JSONResponse({
        "status": "ok",
        "picks_remaining": len(sess.picks),
    })


@app.get("/api/search")
async def search_players(
    session_id: str = Query(...),
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
):
    """Search available players by name substring."""
    sess = _get_session(session_id)
    if not sess.connected:
        return JSONResponse({"error": "Not connected to a draft"}, status_code=400)

    # Use manual state or rebuild from Sleeper
    if sess.mode == "manual" and sess.draft_state:
        available = sess.draft_state.available
    else:
        picks, config = await _get_picks_and_config(sess)
        state = rebuild_draft_state(config, sess.players, picks, sess.id_to_player)
        available = state.available

    query = q.lower()
    results = []
    for p in sorted(available, key=lambda p: p.projected_total, reverse=True):
        if query in p.name.lower():
            results.append({
                "name": p.name,
                "position": p.position,
                "team": p.team,
                "projected_total": round(p.projected_total, 1),
                "bye_week": p.bye_week,
                "is_rookie": p.is_rookie,
            })
            if len(results) >= limit:
                break

    return JSONResponse({"results": results})


@app.get("/adjustments", response_class=HTMLResponse)
async def adjustments_page():
    """Serve the projection adjustments editor."""
    html_path = STATIC_DIR / "adjustments.html"
    return HTMLResponse(html_path.read_text())


_NFL_ADJUSTMENTS_PATH = DATA_DIR / "nfl_adjustments.json"


def _load_adjustments() -> dict:
    """Load NFL adjustments from JSON file."""
    if _NFL_ADJUSTMENTS_PATH.exists():
        with open(_NFL_ADJUSTMENTS_PATH) as f:
            return json.load(f)
    return {}


@app.get("/api/nfl-adjustments")
async def get_nfl_adjustments():
    """Load all NFL players with base projections and existing adjustments."""
    import csv as csv_mod

    proj_path = DATA_DIR / "projections" / "all_projections.csv"
    if not proj_path.exists():
        raise HTTPException(status_code=404, detail="Projections file not found")

    players = []
    with open(proj_path, newline="") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            pos = row.get("position_group", "")
            if pos not in ("QB", "RB", "WR", "TE"):
                continue
            total = float(row.get("projected_total", 0))
            if total < 20:
                continue
            players.append({
                "name": row["player_display_name"],
                "position": pos,
                "team": row.get("current_team", ""),
                "projected_ppg": round(float(row.get("projected_ppg", 0)), 1),
                "ppg_median": round(float(row.get("ppg_median", 0)), 1),
                "ppg_floor": round(float(row.get("ppg_floor", 0)), 1),
                "ppg_ceiling": round(float(row.get("ppg_ceiling", 0)), 1),
                "projected_games": round(float(row.get("projected_games", 0)), 1),
                "projected_total": int(total),
                "total_floor": int(float(row.get("total_floor", 0))),
                "total_ceiling": int(float(row.get("total_ceiling", 0))),
                "pos_rank": int(row.get("pos_rank", 0)),
            })

    adjustments = _load_adjustments()

    return JSONResponse({"players": players, "adjustments": adjustments})


@app.post("/api/nfl-adjustments")
async def save_nfl_adjustments(request: Request):
    """Save adjustments to disk."""
    body = await request.json()
    adjustments = body.get("adjustments", {})

    # Strip all-zero entries
    cleaned = {}
    for name, adj in adjustments.items():
        ppg = adj.get("adjustment_ppg", 0) or 0
        games = adj.get("adjustment_games", 0) or 0
        vol = adj.get("adjustment_volatility", 0) or 0
        note = adj.get("note", "") or ""
        if ppg or games or vol or note:
            cleaned[name] = {
                "adjustment_ppg": ppg,
                "adjustment_games": games,
                "adjustment_volatility": vol,
                "note": note,
            }

    with open(_NFL_ADJUSTMENTS_PATH, "w") as f:
        json.dump(cleaned, f, indent=2)

    log.info("Saved %d NFL adjustment(s) to %s", len(cleaned), _NFL_ADJUSTMENTS_PATH)
    return JSONResponse({"status": "ok", "count": len(cleaned)})


# ---------------------------------------------------------------------------
# Trade Calculator API
# ---------------------------------------------------------------------------

def _get_trade_session(session_id: str) -> TradeSession:
    """Look up a trade session by ID or raise 404."""
    sess = trade_sessions.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Trade session not found")
    return sess


@app.post("/api/trade/connect-sleeper")
async def trade_connect_sleeper(request: Request):
    """Connect to a Sleeper league and load rosters for trade analysis."""
    body = await request.json()
    league_id = str(body.get("league_id", "")).strip()
    sport = body.get("sport", "nfl")

    if not league_id:
        raise HTTPException(status_code=400, detail="league_id is required")

    async with httpx.AsyncClient(timeout=30) as client:
        # Fetch league info, rosters, users, and player DB in parallel
        league_info, raw_rosters, users, all_sleeper = await asyncio.gather(
            fetch_league_info(client, league_id),
            fetch_league_rosters(client, league_id),
            fetch_league_users(client, league_id),
            fetch_all_players(client),
        )

    # Build league config from league settings
    settings = league_info.get("settings", {})
    roster_positions = settings.get("roster_positions", [])
    num_teams = settings.get("num_teams", len(raw_rosters))

    if roster_positions:
        # Build config from league info (similar to config_from_sleeper_meta)
        from draftsim.config import LeagueConfig
        _SLOT_MAP = {
            "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "K": "K",
            "DEF": "DST", "FLEX": "FLEX", "SUPER_FLEX": "FLEX",
            "REC_FLEX": "FLEX", "BN": None, "IR": None,
            # MLB slots
            "C": "C", "1B": "1B", "2B": "2B", "3B": "3B", "SS": "SS",
            "OF": "OF", "UTIL": "FLEX", "SP": "SP", "RP": "RP", "P": "PFLEX",
        }
        lineup: dict[str, int] = {}
        roster_size = 0
        for slot in roster_positions:
            roster_size += 1
            mapped = _SLOT_MAP.get(slot)
            if mapped:
                lineup[mapped] = lineup.get(mapped, 0) + 1

        if not lineup:
            config = default_config_for_sport(sport, num_teams=num_teams)
        else:
            config = LeagueConfig(
                num_teams=num_teams,
                roster_size=roster_size,
                lineup=lineup,
            )
    else:
        config = default_config_for_sport(sport, num_teams=num_teams)

    # Load our projection data
    players = load_players(sport=sport)

    # Build player index for ID resolution
    id_to_player = build_player_index(players, all_sleeper)

    # Resolve rosters
    teams = resolve_sleeper_rosters(
        raw_rosters, users, all_sleeper, players, id_to_player,
    )

    # Create trade session
    session_id = str(uuid.uuid4())[:8]
    sess = TradeSession(
        league_id=league_id,
        platform="sleeper",
        sport=sport,
        config=config,
        teams=teams,
        all_players=players,
    )
    trade_sessions[session_id] = sess

    # Build response
    team_list = []
    for t in teams:
        needs = roster_needs(t, config)
        team_list.append({
            "team_id": t.team_id,
            "name": t.name,
            "roster_count": len(t.roster),
            "needs": needs,
        })

    log.info(
        "Trade session %s: connected to Sleeper league %s (%d teams, %s)",
        session_id, league_id, len(teams), sport,
    )

    return JSONResponse({
        "session_id": session_id,
        "league_name": league_info.get("name", league_id),
        "sport": sport,
        "num_teams": len(teams),
        "teams": team_list,
    })


@app.post("/api/trade/connect-yahoo")
async def trade_connect_yahoo(request: Request):
    """Connect to a Yahoo Fantasy league and load rosters for trade analysis."""
    body = await request.json()
    league_id = str(body.get("league_id", "")).strip()
    sport = body.get("sport", "nfl")
    consumer_key = body.get("consumer_key")
    consumer_secret = body.get("consumer_secret")

    if not league_id:
        raise HTTPException(status_code=400, detail="league_id is required")

    try:
        league_info = fetch_yahoo_league_info(
            league_id, sport, consumer_key, consumer_secret,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error("Yahoo connect error: %s", e)
        raise HTTPException(status_code=502, detail=f"Yahoo API error: {e}")

    # Build config from roster positions
    config = yahoo_config_from_positions(
        league_info["roster_positions"],
        league_info["num_teams"],
        sport,
    )

    # Fetch rosters for all teams
    team_keys = [t["team_key"] for t in league_info["teams"] if t.get("team_key")]
    if not team_keys:
        team_keys = [t["team_id"] for t in league_info["teams"]]

    try:
        raw_rosters = fetch_yahoo_rosters(
            league_id, sport, team_keys, consumer_key, consumer_secret,
        )
    except Exception as e:
        log.error("Yahoo roster fetch error: %s", e)
        raise HTTPException(status_code=502, detail=f"Yahoo roster error: {e}")

    # Load projection data and resolve rosters
    players = load_players(sport=sport)
    teams = resolve_yahoo_rosters(league_info, raw_rosters, players)

    # Create trade session
    session_id = str(uuid.uuid4())[:8]
    sess = TradeSession(
        league_id=league_id,
        platform="yahoo",
        sport=sport,
        config=config,
        teams=teams,
        all_players=players,
    )
    trade_sessions[session_id] = sess

    team_list = []
    for t in teams:
        needs = roster_needs(t, config)
        team_list.append({
            "team_id": t.team_id,
            "name": t.name,
            "roster_count": len(t.roster),
            "needs": needs,
        })

    log.info(
        "Trade session %s: connected to Yahoo league %s (%d teams, %s)",
        session_id, league_id, len(teams), sport,
    )

    return JSONResponse({
        "session_id": session_id,
        "league_name": league_info.get("name", league_id),
        "sport": sport,
        "num_teams": len(teams),
        "teams": team_list,
    })


@app.get("/api/trade/teams")
async def trade_teams(session_id: str = Query(...)):
    """Get all teams with rosters and needs for a trade session."""
    sess = _get_trade_session(session_id)

    teams = []
    for t in sess.teams:
        needs = roster_needs(t, sess.config)
        roster_players = []
        for p in sorted(t.roster, key=lambda x: x.projected_total, reverse=True):
            roster_players.append({
                "name": p.name,
                "position": p.position,
                "team": p.team,
                "sleeper_id": p.sleeper_id,
                "projected_ppg": round(p.projected_ppg, 1),
                "projected_total": round(p.projected_total, 1),
                "total_floor": round(p.total_floor, 1),
                "total_ceiling": round(p.total_ceiling, 1),
                "pos_rank": p.pos_rank,
                "age": p.age,
                "trade_value": round(
                    player_trade_value(p, sess.replacement_levels), 1,
                ),
            })

        teams.append({
            "team_id": t.team_id,
            "name": t.name,
            "roster": roster_players,
            "needs": needs,
        })

    return JSONResponse({"teams": teams})


@app.post("/api/trade/evaluate")
async def trade_evaluate(request: Request):
    """Evaluate a proposed trade between two teams."""
    body = await request.json()
    session_id = body.get("session_id", "")
    sess = _get_trade_session(session_id)

    team_a_id = str(body.get("team_a", ""))
    team_b_id = str(body.get("team_b", ""))
    a_gives = [str(x) for x in body.get("team_a_gives", [])]
    b_gives = [str(x) for x in body.get("team_b_gives", [])]

    if not team_a_id or not team_b_id:
        raise HTTPException(status_code=400, detail="team_a and team_b are required")
    if not a_gives and not b_gives:
        raise HTTPException(status_code=400, detail="At least one side must give players")

    ros_fraction = float(body.get("ros_fraction", 1.0))

    result = evaluate_trade(sess, team_a_id, team_b_id, a_gives, b_gives, ros_fraction)
    return JSONResponse(result)


@app.get("/api/trade/free-agents")
async def trade_free_agents(
    session_id: str = Query(...),
    limit: int = Query(10, ge=1, le=50),
):
    """Get best free agents at each position."""
    sess = _get_trade_session(session_id)
    fa = sess.free_agents
    by_pos = best_at_position(fa, sess.replacement_levels)

    # Serialize
    result: dict[str, list[dict]] = {}
    for pos, plist in by_pos.items():
        result[pos] = [
            {
                "name": entry["player"].name,
                "position": entry["player"].position,
                "team": entry["player"].team,
                "sleeper_id": entry["player"].sleeper_id,
                "projected_ppg": round(entry["player"].projected_ppg, 1),
                "projected_total": round(entry["player"].projected_total, 1),
                "pos_rank": entry["player"].pos_rank,
                "vorp": entry["vorp"],
            }
            for entry in plist[:limit]
        ]

    return JSONResponse({"free_agents": result})


@app.get("/api/trade/players")
async def trade_search_players(
    session_id: str = Query(...),
    q: str = Query("", min_length=0),
    position: str = Query(""),
):
    """Search players across all rosters in the trade session."""
    sess = _get_trade_session(session_id)

    results = []
    query = q.lower().strip()

    for team in sess.teams:
        for p in team.roster:
            if query and query not in p.name.lower():
                continue
            if position and p.position != position:
                continue
            results.append({
                "name": p.name,
                "position": p.position,
                "team": p.team,
                "sleeper_id": p.sleeper_id,
                "projected_ppg": round(p.projected_ppg, 1),
                "projected_total": round(p.projected_total, 1),
                "pos_rank": p.pos_rank,
                "trade_value": round(
                    player_trade_value(p, sess.replacement_levels), 1,
                ),
                "roster_team_id": team.team_id,
                "roster_team_name": team.name,
            })

    results.sort(key=lambda x: x["trade_value"], reverse=True)
    return JSONResponse({"results": results[:50]})


@app.get("/trade", response_class=HTMLResponse)
async def trade():
    """Serve the trade calculator UI."""
    html_path = STATIC_DIR / "trade.html"
    return HTMLResponse(html_path.read_text())


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main UI."""
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text())
