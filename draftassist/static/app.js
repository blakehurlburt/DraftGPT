// Draft Assistant — Frontend
(function () {
    "use strict";

    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    // DOM refs
    const connectBtn = $("#connect-btn");
    const draftIdInput = $("#draft-id");
    const userSlotInput = $("#user-slot");
    const connectStatus = $("#connect-status");
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

    let currentStrategy = "bpa";
    let currentState = null;
    let eventSource = null;
    let notificationsEnabled = false;

    // Request notification permission
    if ("Notification" in window && Notification.permission === "default") {
        Notification.requestPermission().then((perm) => {
            notificationsEnabled = perm === "granted";
        });
    } else if ("Notification" in window && Notification.permission === "granted") {
        notificationsEnabled = true;
    }

    // Strategy tabs
    $$(".tab").forEach((tab) => {
        tab.addEventListener("click", () => {
            $$(".tab").forEach((t) => t.classList.remove("active"));
            tab.classList.add("active");
            currentStrategy = tab.dataset.strategy;
            if (currentState) renderRecommendations(currentState);
        });
    });

    // Connect button
    connectBtn.addEventListener("click", async () => {
        const draftId = draftIdInput.value.trim();
        const slot = parseInt(userSlotInput.value, 10);
        if (!draftId) {
            connectStatus.textContent = "Enter a draft ID";
            return;
        }
        if (!slot || slot < 1) {
            connectStatus.textContent = "Enter a valid slot";
            return;
        }

        connectBtn.disabled = true;
        connectStatus.textContent = "Connecting...";

        try {
            const resp = await fetch(
                `/api/connect?draft_id=${encodeURIComponent(draftId)}&user_slot=${slot}`,
                { method: "POST" }
            );
            const data = await resp.json();

            if (!resp.ok) {
                throw new Error(data.detail || "Connection failed");
            }

            connectStatus.textContent =
                `Connected — ${data.num_teams} teams, ${data.rounds} rounds, ` +
                `${data.players_matched} players matched`;
            statusBar.classList.remove("hidden");
            mainContent.classList.remove("hidden");

            // Load initial state
            const stateResp = await fetch("/api/state");
            const state = await stateResp.json();
            updateUI(state);

            // Start SSE
            startSSE();
        } catch (err) {
            connectStatus.textContent = `Error: ${err.message}`;
        } finally {
            connectBtn.disabled = false;
        }
    });

    function startSSE() {
        if (eventSource) eventSource.close();

        eventSource = new EventSource("/api/stream");

        eventSource.addEventListener("draft_update", (e) => {
            const state = JSON.parse(e.data);
            updateUI(state);
        });

        eventSource.onerror = () => {
            connectStatus.textContent = "SSE disconnected — reconnecting...";
        };
    }

    function updateUI(state) {
        currentState = state;

        // Status bar
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
    }

    function renderRecommendations(state) {
        const recs = state.recommendations || {};
        const stratRecs = recs[currentStrategy] || [];

        if (!stratRecs.length) {
            recBody.innerHTML =
                '<tr><td colspan="6" class="empty-msg">' +
                (state.is_my_turn ? "No recommendations available" : "Recommendations appear on your turn") +
                "</td></tr>";
            return;
        }

        recBody.innerHTML = stratRecs
            .map(
                (r) => `
            <tr>
                <td>${r.rank}</td>
                <td><strong>${r.name}</strong></td>
                <td><span class="pos-badge pos-${r.position}">${r.position}</span></td>
                <td>${r.team}</td>
                <td>${r.projected_total}</td>
                <td>${r.vbd}</td>
            </tr>`
            )
            .join("");
    }

    function renderRoster(state) {
        const roster = state.user_roster || [];
        const needs = state.team_needs || {};

        // Needs chips
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
            .reverse()
            .map(
                (p) => `
            <li class="${p.is_user ? "user-pick" : ""}">
                <span class="pick-num">${p.pick_no}.</span>
                <span class="pos-badge pos-${p.position}">${p.position}</span>
                <span>${p.player_name}</span>
                <span style="color:#4a5a6a">${p.team}</span>
            </li>`
            )
            .join("");
    }

    function sendTurnNotification() {
        if (notificationsEnabled) {
            new Notification("Draft Assistant", {
                body: "It's your turn to pick!",
                icon: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🏈</text></svg>",
            });
        }
    }
})();
