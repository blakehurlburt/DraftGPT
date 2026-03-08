// Draft Assistant — Frontend
(function () {
    "use strict";

    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    // DOM refs
    const welcomeScreen = $("#welcome-screen");
    const connectionBar = $("#connection-bar");
    const connectBtn = $("#connect-btn");
    const draftIdInput = $("#draft-id");
    const userSlotInput = $("#user-slot");
    const connectStatus = $("#connect-status");
    const connectedDraftLabel = $("#connected-draft-label");
    const statusBar = $("#status-bar");
    const mainContent = $("#main-content");
    const turnIndicator = $("#turn-indicator");
    const roundDisplay = $("#round-display");
    const pickDisplay = $("#pick-display");
    const picksUntil = $("#picks-until");
    const draftStatusBadge = $("#draft-status-badge");
    const recBody = $("#rec-body");
    const rosterBody = $("#roster-body");
    const rosterNeeds = $("#roster-needs");
    const tickerList = $("#ticker-list");
    const draftBoard = $("#draft-board");
    const posFilter = $("#pos-filter");
    const scoreHeader = $("#score-header");
    const showMoreBtn = $("#show-more-btn");

    let currentStrategy = "vbd";
    let currentAdp = "consensus";
    let currentRisk = "balanced";  // "safe", "balanced", "aggressive"
    let currentSort = "rank";  // "rank" or "adp"
    let posFilters = new Set(["QB", "RB", "WR", "TE"]);
    let currentState = null;
    let eventSource = null;
    let notificationsEnabled = false;
    let connectedDraftId = null;
    let connectedSlot = null;
    let sessionId = null;
    let currentView = "ticker";  // "ticker" or "board"

    // Extra recommendations fetched via "show more"
    let extraRecs = {};       // strategy -> array of additional recs
    let prefetchedRecs = {};  // strategy -> array (prefetched next batch)
    let totalLoaded = 10;     // how many recs are currently loaded (initial)
    let displayCount = 10;    // how many rows to show after filtering
    let prefetching = false;

    // Helper to build URL with session_id
    function apiUrl(path, params = {}) {
        const url = new URL(path, window.location.origin);
        if (sessionId) url.searchParams.set("session_id", sessionId);
        for (const [k, v] of Object.entries(params)) {
            url.searchParams.set(k, v);
        }
        return url.toString();
    }

    // Request notification permission
    if ("Notification" in window && Notification.permission === "default") {
        Notification.requestPermission().then((perm) => {
            notificationsEnabled = perm === "granted";
        });
    } else if ("Notification" in window && Notification.permission === "granted") {
        notificationsEnabled = true;
    }

    // Extract draft ID from a Sleeper URL or raw ID
    function parseDraftId(input) {
        const trimmed = input.trim();
        const urlMatch = trimmed.match(
            /sleeper\.com\/(?:draft\/nfl\/|i\/)(\d+)/
        );
        if (urlMatch) return urlMatch[1];
        if (/^\d+$/.test(trimmed)) return trimmed;
        const trailingMatch = trimmed.match(/\/(\d{10,})(?:[/?#]|$)/);
        if (trailingMatch) return trailingMatch[1];
        return trimmed;
    }

    // Score column labels per strategy
    const SCORE_LABELS = {
        "vbd": "VBD",
        "vona": "VONA",
        "bpa": "Proj Pts",
        "zero-rb": "VBD",
        "robust-rb": "VBD",
    };

    const SCORE_TOOLTIPS = {
        "vbd": "Value over positional replacement level",
        "vona": "Combined VBD + positional drop-off urgency",
        "bpa": "Total projected season points",
        "zero-rb": "Value over positional replacement level",
        "robust-rb": "Value over positional replacement level",
    };

    function updateScoreHeader() {
        scoreHeader.textContent = SCORE_LABELS[currentStrategy] || "Score";
        scoreHeader.title = SCORE_TOOLTIPS[currentStrategy] || "";
    }

    // Strategy tabs
    $$(".tab").forEach((tab) => {
        tab.addEventListener("click", () => {
            $$(".tab").forEach((t) => t.classList.remove("active"));
            tab.classList.add("active");
            currentStrategy = tab.dataset.strategy;
            updateScoreHeader();
            if (currentState) renderRecommendations(currentState);
        });
    });

    // ADP source tabs
    $$(".adp-tab").forEach((tab) => {
        tab.addEventListener("click", async () => {
            $$(".adp-tab").forEach((t) => t.classList.remove("active"));
            tab.classList.add("active");
            currentAdp = tab.dataset.adp;

            try {
                await fetch(
                    apiUrl("/api/adp", { platform: currentAdp }),
                    { method: "POST" }
                );
                const stateResp = await fetch(apiUrl("/api/state"));
                const state = await stateResp.json();
                resetExtraRecs();
                updateUI(state);
            } catch (err) {
                console.error("ADP switch failed:", err);
            }
        });
    });

    // Risk profile tabs — client-side toggle (all risk levels pre-computed)
    $$(".risk-tab").forEach((tab) => {
        tab.addEventListener("click", () => {
            $$(".risk-tab").forEach((t) => t.classList.remove("active"));
            tab.classList.add("active");
            currentRisk = tab.dataset.risk;
            if (currentState) renderRecommendations(currentState);
        });
    });

    // View toggle (ticker / board)
    $$(".view-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            $$(".view-btn").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            currentView = btn.dataset.view;
            if (currentView === "board") {
                tickerList.classList.add("hidden");
                draftBoard.classList.remove("hidden");
            } else {
                tickerList.classList.remove("hidden");
                draftBoard.classList.add("hidden");
            }
            if (currentState) {
                renderTicker(currentState);
                renderBoard(currentState);
            }
        });
    });

    // Sortable column headers
    $$("th.sortable").forEach((th) => {
        th.addEventListener("click", () => {
            currentSort = th.dataset.sort;
            $$("th.sortable").forEach((t) => t.classList.remove("active"));
            th.classList.add("active");
            if (currentState) renderRecommendations(currentState);
        });
    });

    // Position filter checkboxes
    posFilter.querySelectorAll("input[type=checkbox]").forEach((cb) => {
        cb.addEventListener("change", () => {
            if (cb.checked) {
                posFilters.add(cb.value);
            } else {
                posFilters.delete(cb.value);
            }
            if (currentState) renderRecommendations(currentState);
        });
    });

    // Show more button
    showMoreBtn.addEventListener("click", async () => {
        displayCount += 10;

        // Check if we have enough filtered recs already loaded
        const allRecs = getAllRecsForStrategy(currentStrategy);
        const availableFiltered = allRecs.filter((r) => posFilters.has(r.position)).length;

        if (availableFiltered >= displayCount) {
            // Enough data already loaded, just re-render
            renderRecommendations(currentState);
            return;
        }

        // Need more data from backend
        if (prefetchedRecs && Object.keys(prefetchedRecs).length) {
            _mergeNestedRecs(prefetchedRecs, extraRecs);
            const anyRisk = Object.values(prefetchedRecs)[0] || {};
            totalLoaded += Object.values(anyRisk)[0]?.length || 0;
            prefetchedRecs = {};
            renderRecommendations(currentState);
            prefetchMore();
            return;
        }

        // Otherwise fetch now
        showMoreBtn.textContent = "Loading...";
        showMoreBtn.disabled = true;
        try {
            const resp = await fetch(apiUrl("/api/more", { offset: totalLoaded, n: displayCount }));
            const data = await resp.json();
            const moreRecs = data.recommendations || {};
            _mergeNestedRecs(moreRecs, extraRecs);
            totalLoaded += displayCount;
            renderRecommendations(currentState);
            prefetchMore();
        } catch (err) {
            console.error("Show more failed:", err);
        } finally {
            showMoreBtn.disabled = false;
        }
    });

    function resetExtraRecs() {
        extraRecs = {};
        prefetchedRecs = {};
        totalLoaded = 10;
        displayCount = 10;
    }

    function getVisibleCount() {
        // Count how many rows are currently visible after filtering
        const recs = getAllRecsForStrategy(currentStrategy);
        return recs.filter((r) => posFilters.has(r.position)).length || 10;
    }

    function getAllRecsForStrategy(strategy) {
        // recommendations is nested: { risk: { strategy: [...] } }
        const riskRecs = (currentState?.recommendations || {})[currentRisk] || {};
        const base = riskRecs[strategy] || [];
        const extra = (extraRecs[currentRisk] || {})[strategy] || [];
        return [...base, ...extra];
    }

    function _mergeNestedRecs(source, target) {
        // Merge { risk: { strategy: [...] } } from source into target
        for (const [risk, strats] of Object.entries(source)) {
            if (!target[risk]) target[risk] = {};
            for (const [strat, recs] of Object.entries(strats)) {
                if (!target[risk][strat]) target[risk][strat] = [];
                target[risk][strat].push(...recs);
            }
        }
    }

    async function prefetchMore() {
        if (prefetching) return;
        prefetching = true;
        try {
            const visible = getVisibleCount();
            const resp = await fetch(apiUrl("/api/more", { offset: totalLoaded, n: visible }));
            const data = await resp.json();
            prefetchedRecs = data.recommendations || {};
        } catch (err) {
            console.error("Prefetch failed:", err);
        } finally {
            prefetching = false;
        }
    }

    // Main connect function
    async function connectToDraft(draftId, slot, existingSessionId) {
        connectBtn.disabled = true;
        connectStatus.textContent = "Connecting...";
        connectStatus.style.color = "";

        try {
            // If we have an existing session_id (from URL), try to reconnect
            if (existingSessionId) {
                sessionId = existingSessionId;
                const stateResp = await fetch(apiUrl("/api/state"));
                if (stateResp.ok) {
                    // Session still alive, reconnect
                    connectedDraftId = draftId;
                    connectedSlot = slot;
                    showConnectedUI(draftId, slot, null);
                    const state = await stateResp.json();
                    resetExtraRecs();
                    updateUI(state);
                    startSSE();
                    return;
                }
                // Session expired, fall through to create new one
                sessionId = null;
            }

            const resp = await fetch(
                `/api/connect?draft_id=${encodeURIComponent(draftId)}&user_slot=${slot}`,
                { method: "POST" }
            );
            const data = await resp.json();

            if (!resp.ok) {
                throw new Error(data.detail || "Connection failed");
            }

            sessionId = data.session_id;
            connectedDraftId = draftId;
            connectedSlot = slot;

            showConnectedUI(draftId, slot, data);

            const stateResp = await fetch(apiUrl("/api/state"));
            const state = await stateResp.json();
            resetExtraRecs();
            updateUI(state);

            startSSE();
        } catch (err) {
            connectStatus.textContent = `Error: ${err.message}`;
            connectStatus.style.color = "#ff5252";
            connectBtn.disabled = false;
        }
    }

    function showConnectedUI(draftId, slot, data) {
        const url = new URL(window.location);
        url.searchParams.set("draft_id", draftId);
        url.searchParams.set("slot", slot);
        url.searchParams.set("session_id", sessionId);
        window.history.replaceState({}, "", url);

        welcomeScreen.classList.add("hidden");
        connectionBar.classList.remove("hidden");
        const details = data
            ? `${data.num_teams} teams, ${data.players_matched} players`
            : "";
        connectedDraftLabel.textContent =
            `Draft ${draftId.slice(-8)} — Slot ${slot}` +
            (details ? ` — ${details}` : "");
        statusBar.classList.remove("hidden");
        mainContent.classList.remove("hidden");
    }

    // Connect button click
    connectBtn.addEventListener("click", () => {
        const draftId = parseDraftId(draftIdInput.value);
        const slot = parseInt(userSlotInput.value, 10);
        if (!draftId) {
            connectStatus.textContent = "Enter a draft ID or Sleeper URL";
            return;
        }
        if (!slot || slot < 1) {
            connectStatus.textContent = "Enter a valid slot";
            return;
        }
        connectToDraft(draftId, slot, null);
    });

    function startSSE() {
        if (eventSource) eventSource.close();

        eventSource = new EventSource(apiUrl("/api/stream"));

        eventSource.addEventListener("draft_update", (e) => {
            const state = JSON.parse(e.data);
            resetExtraRecs();  // new picks invalidate extra recs
            updateUI(state);
        });

        eventSource.onerror = () => {
            console.warn("SSE connection error — browser will auto-reconnect");
        };
    }

    function updateUI(state) {
        currentState = state;

        roundDisplay.textContent = `Round ${state.current_round} / ${state.total_rounds}`;
        pickDisplay.textContent = `Pick ${state.current_pick} / ${state.total_picks}`;
        draftStatusBadge.textContent = state.draft_status;

        if (state.is_complete) {
            turnIndicator.classList.add("hidden");
            picksUntil.textContent = "Draft Complete";
        } else if (state.is_my_turn) {
            turnIndicator.classList.remove("hidden");
            picksUntil.textContent = "On the clock!";
            sendTurnNotification();
        } else {
            turnIndicator.classList.add("hidden");
            picksUntil.textContent = `${state.picks_until_next} picks until your turn`;
        }

        renderRecommendations(state);
        renderRoster(state);
        renderTicker(state);
        renderBoard(state);

        // Prefetch more recs in background when it's our turn
        if (state.is_my_turn && !Object.keys(prefetchedRecs).length) {
            prefetchMore();
        }
    }

    function renderRecommendations(state) {
        const allRecs = getAllRecsForStrategy(currentStrategy);

        // Apply position filter first, then cap to displayCount
        let filtered = allRecs.filter((r) => posFilters.has(r.position));

        // Apply sort
        if (currentSort === "adp") {
            filtered = filtered.slice().sort((a, b) => a.adp - b.adp);
        } else {
            filtered = filtered.slice().sort((a, b) => a.rank - b.rank);
        }

        // Cap to displayCount visible rows
        const hasMore = filtered.length > displayCount;
        filtered = filtered.slice(0, displayCount);

        if (!filtered.length) {
            const msg = !allRecs.length
                ? (state.is_my_turn ? "No recommendations available" : "Recommendations appear on your turn")
                : "No players match current filters";
            recBody.innerHTML =
                '<tr><td colspan="9" class="empty-msg">' + msg + "</td></tr>";
            showMoreBtn.classList.add("hidden");
            return;
        }

        recBody.innerHTML = filtered
            .map(
                (r) => `
            <tr>
                <td>${r.rank}</td>
                <td><strong>${r.name}</strong></td>
                <td><span class="pos-badge pos-${r.position}">${r.position}</span></td>
                <td>${r.team}</td>
                <td>${r.projected_total}</td>
                <td class="range-floor">${r.total_floor || "—"}</td>
                <td class="range-ceil">${r.total_ceiling || "—"}</td>
                <td>${r.strategy_score ?? "—"}</td>
                <td>${r.adp && r.adp < 999 ? r.adp : "—"}</td>
            </tr>`
            )
            .join("");

        // Show "show more" button when it's our turn and there are more to show
        if (state.is_my_turn && hasMore) {
            showMoreBtn.classList.remove("hidden");
            showMoreBtn.textContent = `Show ${displayCount} more`;
        } else {
            showMoreBtn.classList.add("hidden");
        }
    }

    function renderRoster(state) {
        const roster = state.user_roster || [];
        const needs = state.team_needs || {};

        rosterNeeds.innerHTML = Object.entries(needs)
            .map(([pos, count]) => {
                const urgent = count >= 2 ? "urgent" : "";
                return `<span class="need-chip ${urgent}">${pos}: ${count} needed</span>`;
            })
            .join("");

        if (!roster.length) {
            rosterBody.innerHTML =
                '<tr><td colspan="4" class="empty-msg">No picks yet</td></tr>';
            return;
        }

        rosterBody.innerHTML = roster
            .map(
                (p) => `
            <tr>
                <td><strong>${p.name}</strong></td>
                <td><span class="pos-badge pos-${p.position}">${p.position}</span></td>
                <td>${p.team}</td>
                <td>${p.projected_total}</td>
            </tr>`
            )
            .join("");
    }

    function renderTicker(state) {
        const picks = state.recent_picks || [];

        tickerList.innerHTML = picks
            .slice()
            .reverse()
            .map(
                (p) => `
            <li class="${p.is_user ? "user-pick" : ""}">
                <span class="pick-num">${p.pick_no}.</span>
                <span class="pos-badge pos-${p.position}">${p.position}</span>
                <span>${p.player_name}</span>
                <span style="color:#4a5a6a">${p.team}</span>
                <span class="team-slot">Team ${p.draft_slot}</span>
            </li>`
            )
            .join("");
    }

    function renderBoard(state) {
        const picks = state.all_picks || [];
        const numTeams = state.num_teams || 12;
        const totalRounds = state.total_rounds || 15;
        const userSlot = (state.user_slot || 0) + 1; // 1-indexed for display
        const currentPick = state.current_pick || 0;

        // Build grid: board[round][slot] = pick
        const board = {};
        for (const p of picks) {
            if (!board[p.round]) board[p.round] = {};
            board[p.round][p.draft_slot] = p;
        }

        // Header row
        let html = '<table class="board-table"><thead><tr><th class="round-col">Rd</th>';
        for (let s = 1; s <= numTeams; s++) {
            const isUser = s === userSlot;
            html += `<th class="${isUser ? "user-col" : ""}">Team ${s}${isUser ? " (You)" : ""}</th>`;
        }
        html += "</tr></thead><tbody>";

        // Body rows
        for (let r = 1; r <= totalRounds; r++) {
            html += `<tr><td class="round-cell">${r}</td>`;
            for (let s = 1; s <= numTeams; s++) {
                const p = board[r]?.[s];
                const isUserCol = s === userSlot;
                if (p) {
                    html += `<td class="board-pick${isUserCol ? " user-col" : ""}">` +
                        `<span class="bp-name">${p.player_name}</span>` +
                        `<span class="bp-meta"><span class="pos-badge pos-${p.position}">${p.position}</span> ${p.team}</span>` +
                        `</td>`;
                } else {
                    // Empty cell — check if this is the current pick
                    const pickNo = calcPickNo(r, s, numTeams, totalRounds);
                    const isCurrent = pickNo === currentPick;
                    html += `<td class="board-pick empty-pick${isUserCol ? " user-col" : ""}${isCurrent ? " current-pick" : ""}">` +
                        `${isCurrent ? "&#9654;" : "&mdash;"}</td>`;
                }
            }
            html += "</tr>";
        }
        html += "</tbody></table>";
        draftBoard.innerHTML = html;
    }

    function calcPickNo(round, slot, numTeams, totalRounds) {
        // Snake draft: odd rounds go 1..N, even rounds go N..1
        if (round > totalRounds) return -1;
        if (round % 2 === 1) {
            return (round - 1) * numTeams + slot;
        } else {
            return (round - 1) * numTeams + (numTeams - slot + 1);
        }
    }

    function sendTurnNotification() {
        if (notificationsEnabled) {
            new Notification("Draft Assistant", {
                body: "It's your turn to pick!",
                icon: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🏈</text></svg>",
            });
        }
    }

    // Data freshness: show when data files were last updated
    (async function loadDataFreshness() {
        const container = $("#data-freshness");
        if (!container) return;
        try {
            const resp = await fetch("/api/data-info");
            if (!resp.ok) return;
            const data = await resp.json();
            if (!data.sources || data.sources.length === 0) return;

            let html = '<div class="data-freshness-title">Data Sources</div>';
            const now = Date.now();
            const STALE_MS = 14 * 24 * 60 * 60 * 1000; // 14 days

            for (const src of data.sources) {
                let dateStr = "not found";
                let cls = "";
                if (src.last_modified) {
                    const d = new Date(src.last_modified);
                    const age = now - d.getTime();
                    dateStr = d.toLocaleDateString(undefined, {
                        month: "short", day: "numeric", year: "numeric",
                    });
                    cls = age > STALE_MS ? "stale" : "fresh";
                }
                const howText = src.how ? `<span class="data-source-how">${src.how}</span>` : "";
                const linkText = src.url
                    ? `<a class="data-source-link" href="${src.url}" target="_blank" rel="noopener">source</a>`
                    : "";
                html += `<div class="data-source-row">
                    <span class="data-source-name">${src.name} ${howText}</span>
                    <span>
                        <span class="data-source-date ${cls}">${dateStr}</span>
                        ${linkText}
                    </span>
                </div>`;
            }
            container.innerHTML = html;
        } catch (e) {
            // Silently ignore — non-critical UI
        }
    })();

    // Auto-reconnect: check URL params on page load
    (function autoConnect() {
        const params = new URLSearchParams(window.location.search);
        const draftId = params.get("draft_id");
        const slot = params.get("slot");
        const savedSessionId = params.get("session_id");
        if (draftId && slot) {
            draftIdInput.value = draftId;
            userSlotInput.value = slot;
            connectToDraft(draftId, parseInt(slot, 10), savedSessionId || null);
        }
    })();
})();
