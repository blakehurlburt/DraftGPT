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
    const headerDraftId = $("#header-draft-id");
    const headerSlot = $("#header-slot");
    const headerTeams = $("#header-teams");
    const headerConnectBtn = $("#header-connect-btn");
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
    const showMoreBtn = $("#show-more-btn");
    const simPanel = $("#sim-panel");
    const simContent = $("#sim-content");
    const simToggle = $("#sim-toggle");
    const simProgressFill = $("#sim-progress-fill");
    const simProgressText = $("#sim-progress-text");
    const simStrategies = $("#sim-strategies");
    let currentStrategy = "vona";  // strategy used for generating recommendations
    let currentAdp = "consensus";
    let currentRisk = "balanced";  // "safe", "balanced", "aggressive"
    let currentProj = "model";    // "model" or "sleeper"
    let currentValueMode = "vbd_score";  // "vorp", "vona", "vols", "vbd_score"
    let currentSort = "value";  // "value", "adp", or "rank"
    let posFilters = new Set(["QB", "RB", "WR", "TE"]);  // reset dynamically per sport
    let rookieOnly = false;
    let currentState = null;
    let eventSource = null;
    let notificationsEnabled = false;
    let connectedDraftId = null;
    let connectedSlot = null;
    let sessionId = null;
    let currentView = "ticker";  // "ticker" or "board"
    let simData = null;  // latest sim snapshot
    let simCollapsed = false;
    let rosterSort = null;  // null = draft order, or {key, asc}
    let draftMode = "sleeper";  // "sleeper" or "manual"
    let draftSport = "nfl";     // "nfl" or "mlb"

    const NFL_POSITIONS = ["QB", "RB", "WR", "TE"];
    const MLB_POSITIONS = ["C", "1B", "2B", "3B", "SS", "OF", "DH", "SP", "RP"];

    /**
     * Rebuild position filter chips, projection/ADP tabs, and column
     * visibility based on the current sport (draftSport).
     */
    function initSportUI() {
        const positions = draftSport === "mlb" ? MLB_POSITIONS : NFL_POSITIONS;
        posFilters = new Set(positions);
        rookieOnly = false;

        // Rebuild position filter chips
        const container = posFilter;
        // Keep the label, remove everything else
        while (container.children.length > 1) container.removeChild(container.lastChild);

        positions.forEach((pos) => {
            const label = document.createElement("label");
            label.className = "pos-toggle";
            const cb = document.createElement("input");
            cb.type = "checkbox";
            cb.value = pos;
            cb.checked = true;
            cb.addEventListener("change", () => {
                posFilters[cb.checked ? "add" : "delete"](pos);
                displayCount = 10;
                if (currentState) renderRecommendations(currentState);
            });
            const chip = document.createElement("span");
            chip.className = `pos-chip pos-${pos}`;
            chip.textContent = pos;
            label.appendChild(cb);
            label.appendChild(chip);
            container.appendChild(label);
        });

        // Rookies toggle — only for NFL
        if (draftSport !== "mlb") {
            const label = document.createElement("label");
            label.className = "pos-toggle";
            label.id = "rookie-filter";
            const cb = document.createElement("input");
            cb.type = "checkbox";
            cb.id = "rookie-only";
            cb.addEventListener("change", (e) => {
                rookieOnly = e.target.checked;
                displayCount = 10;
                if (currentState) renderRecommendations(currentState);
            });
            const chip = document.createElement("span");
            chip.className = "pos-chip rookie-chip";
            chip.textContent = "Rookies";
            label.appendChild(cb);
            label.appendChild(chip);
            container.appendChild(label);
        }

        // Sleeper projection tab — hide for MLB
        const sleeperProjTab = document.querySelector('.proj-tab[data-proj="sleeper"]');
        if (sleeperProjTab) {
            sleeperProjTab.classList.toggle("hidden", draftSport === "mlb");
            if (draftSport === "mlb") {
                currentProj = "model";
                sleeperProjTab.classList.remove("active");
                document.querySelector('.proj-tab[data-proj="model"]')?.classList.add("active");
            }
        }

        // ADP tabs — hide for MLB (no real ADP data)
        const adpTabs = document.getElementById("adp-tabs");
        if (adpTabs) {
            adpTabs.classList.toggle("hidden", draftSport === "mlb");
        }
    }

    // Extra recommendations fetched via "show more"
    let extraRecs = {};       // strategy -> array of additional recs
    let prefetchedRecs = {};  // strategy -> array (prefetched next batch)
    let totalLoaded = 30;     // how many recs are currently loaded (initial — backend sends 30)
    let displayCount = 10;    // how many rows to show after filtering
    let prefetching = false;
    let backendExhausted = false;

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

    // ── Mode tabs (welcome screen) ─────────────────────────────────
    const sleeperForm = $("#sleeper-form");
    const manualForm = $("#manual-form");
    const manualStartBtn = $("#manual-start-btn");
    const manualStatus = $("#manual-status");
    const pickSearchInput = $("#pick-search-input");
    const pickSearchResults = $("#pick-search-results");
    const undoBtn = $("#undo-btn");
    const manualPickControls = $("#manual-pick-controls");

    $$(".mode-tab").forEach((tab) => {
        tab.addEventListener("click", () => {
            $$(".mode-tab").forEach((t) => t.classList.remove("active"));
            tab.classList.add("active");
            const mode = tab.dataset.mode;
            if (mode === "sleeper") {
                sleeperForm.classList.remove("hidden");
                manualForm.classList.add("hidden");
            } else {
                sleeperForm.classList.add("hidden");
                manualForm.classList.remove("hidden");
            }
        });
    });

    // Update defaults when sport changes
    $$('input[name="sport"]').forEach((radio) => {
        radio.addEventListener("change", () => {
            const sport = radio.value;
            const teamsInput = $("#manual-teams");
            const roundsInput = $("#manual-rounds");
            if (sport === "mlb") {
                teamsInput.value = 10;
                roundsInput.value = 23;  // MLB rosters are larger
            } else {
                teamsInput.value = 12;
                roundsInput.value = 15;
            }
        });
    });

    // Manual draft creation
    if (manualStartBtn) {
        manualStartBtn.addEventListener("click", async () => {
            const sport = document.querySelector('input[name="sport"]:checked')?.value || "nfl";
            const numTeams = parseInt($("#manual-teams").value, 10) || 12;
            const rosterSize = parseInt($("#manual-rounds").value, 10) || 15;
            const slot = parseInt($("#manual-slot").value, 10) || 1;

            manualStartBtn.disabled = true;
            manualStatus.textContent = "Creating draft...";
            manualStatus.style.color = "";

            try {
                const resp = await fetch(
                    `/api/create?sport=${sport}&num_teams=${numTeams}&roster_size=${rosterSize}&user_slot=${slot}`,
                    { method: "POST" }
                );
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.detail || "Failed to create draft");

                sessionId = data.session_id;
                draftMode = "manual";
                draftSport = sport;
                connectedDraftId = data.draft_id;
                connectedSlot = slot;

                initSportUI();
                showConnectedUI(data.draft_id, slot, data);

                // Show manual controls
                manualPickControls.classList.remove("hidden");

                const stateResp = await fetch(apiUrl("/api/state"));
                const state = await stateResp.json();
                resetExtraRecs();
                updateUI(state);
                startSSE();
            } catch (err) {
                manualStatus.textContent = `Error: ${err.message}`;
                manualStatus.style.color = "#ff5252";
            } finally {
                manualStartBtn.disabled = false;
            }
        });
    }

    // ── Manual pick search ──────────────────────────────────────────
    let searchDebounce = null;

    if (pickSearchInput) {
        pickSearchInput.addEventListener("input", () => {
            clearTimeout(searchDebounce);
            const q = pickSearchInput.value.trim();
            if (q.length < 2) {
                pickSearchResults.classList.add("hidden");
                return;
            }
            searchDebounce = setTimeout(async () => {
                try {
                    const resp = await fetch(apiUrl("/api/search", { q, limit: 10 }));
                    const data = await resp.json();
                    renderSearchResults(data.results || []);
                } catch (err) {
                    console.error("Search failed:", err);
                }
            }, 200);
        });

        // Close dropdown on click outside
        document.addEventListener("click", (e) => {
            if (!e.target.closest(".pick-search-wrapper")) {
                pickSearchResults.classList.add("hidden");
            }
        });
    }

    function renderSearchResults(results) {
        if (!results.length) {
            pickSearchResults.innerHTML = '<div class="search-empty">No players found</div>';
            pickSearchResults.classList.remove("hidden");
            return;
        }
        pickSearchResults.innerHTML = results.map((r) =>
            `<div class="search-result" data-name="${r.name.replace(/"/g, '&quot;')}">
                <span class="pos-badge pos-${r.position}">${r.position}</span>
                <span class="search-result-name">${r.name}</span>
                <span class="search-result-team">${r.team}</span>
                <span class="search-result-pts">${r.projected_total}</span>
            </div>`
        ).join("");
        pickSearchResults.classList.remove("hidden");

        // Bind click handlers
        pickSearchResults.querySelectorAll(".search-result").forEach((el) => {
            el.addEventListener("click", () => makePick(el.dataset.name));
        });
    }

    async function makePick(playerName) {
        pickSearchResults.classList.add("hidden");
        pickSearchInput.value = "";
        try {
            const resp = await fetch(
                apiUrl("/api/pick", { player_name: playerName }),
                { method: "POST" }
            );
            const data = await resp.json();
            if (!resp.ok) {
                console.error("Pick failed:", data.error);
            }
        } catch (err) {
            console.error("Pick error:", err);
        }
    }

    if (undoBtn) {
        undoBtn.addEventListener("click", async () => {
            try {
                const resp = await fetch(apiUrl("/api/undo"), { method: "POST" });
                const data = await resp.json();
                if (!resp.ok) console.error("Undo failed:", data.error);
            } catch (err) {
                console.error("Undo error:", err);
            }
        });
    }

    // ── Column definitions ───────────────────────────────────────────
    // Each column: { key, label, title?, cls?, sortKey?, id?, hidden?(ctx),
    //               render(row, ctx) -> html string for <td> contents }
    // "ctx" = { showSim, currentStrategy, getSimValue }

    function rookieBadge(row) {
        return row.is_rookie ? ' <span class="rookie-badge" title="Rookie">R</span>' : "";
    }

    const REC_COLUMNS = [
        {
            key: "rank", label: "#", sortKey: "rank",
            title: "Strategy recommendation rank (click to sort)",
            cls: "sortable",
            render: (r) => r.rank,
        },
        {
            key: "name", label: "Player",
            render: (r) => `<strong>${r.name}</strong>${rookieBadge(r)}`,
        },
        {
            key: "position", label: "Pos",
            render: (r) => `<span class="pos-badge pos-${r.position}">${r.position}</span>`,
        },
        {
            key: "team", label: "Team",
            render: (r) => r.team,
        },
        {
            key: "bye_week", label: "Bye", title: "Bye week",
            hidden: () => draftSport === "mlb",
            render: (r) => r.bye_week || "—",
        },
        {
            key: "projected_total", label: "Proj",
            title: "Projected season total points",
            render: (r) => r.projected_total,
        },
        {
            key: "total_floor", label: "Floor",
            title: "10th percentile projection (pessimistic)",
            tdClass: "range-floor",
            render: (r, ctx) => {
                if (!r.total_floor) return "—";
                const prefix = ctx.floorEstimated ? "~" : "";
                return prefix + r.total_floor;
            },
            dynamicTdClass: (r, ctx) => ctx.floorEstimated ? "range-floor estimated" : "range-floor",
        },
        {
            key: "total_ceiling", label: "Ceil",
            title: "90th percentile projection (optimistic)",
            tdClass: "range-ceil",
            render: (r, ctx) => {
                if (!r.total_ceiling) return "—";
                const prefix = ctx.floorEstimated ? "~" : "";
                return prefix + r.total_ceiling;
            },
            dynamicTdClass: (r, ctx) => ctx.floorEstimated ? "range-ceil estimated" : "range-ceil",
        },
        {
            key: "value", label: "VBD", id: "value-header",
            sortKey: "value",
            title: "Composite VBD Score — aggregates VORP + VONA + VOLS",
            cls: "sortable active",
            render: (r, ctx) => {
                const v = r[ctx.valueMode];
                return v != null && v > 0 ? v.toFixed(1) : "—";
            },
        },
        {
            key: "adp", label: "ADP", sortKey: "adp",
            title: "Average Draft Position (click to sort)",
            cls: "sortable",
            render: (r) => r.adp && r.adp < 999 ? r.adp : "—",
        },
        {
            key: "sim", label: "Sim", id: "sim-value-header",
            title: "Expected lineup total if you pick this player (from Monte Carlo sim)",
            hidden: (ctx) => !ctx.showSim,
            render: (r, ctx) => {
                const sv = ctx.getSimValue(r.name);
                return sv != null ? sv.toFixed(1) : "—";
            },
            tdClass: "sim-value",
        },
    ];

    const ROSTER_COLUMNS = [
        {
            key: "name", label: "Player", sortKey: "name",
            render: (p) => `<strong>${p.name}</strong>${rookieBadge(p)}`,
        },
        {
            key: "position", label: "Pos", sortKey: "position",
            render: (p) => `<span class="pos-badge pos-${p.position}">${p.position}</span>`,
        },
        {
            key: "team", label: "Team", sortKey: "team",
            render: (p) => p.team,
        },
        {
            key: "bye_week", label: "Bye", sortKey: "bye_week",
            hidden: () => draftSport === "mlb",
            render: (p) => p.bye_week || "—",
        },
        {
            key: "projected_total", label: "Proj Pts", sortKey: "projected_total",
            render: (p) => p.projected_total,
        },
    ];

    function renderTableHeader(columns, theadEl, ctx = {}) {
        if (!theadEl) return;
        const ths = columns
            .filter((c) => !c.hidden || !c.hidden(ctx))
            .map((c) => {
                const attrs = [];
                if (c.id) attrs.push(`id="${c.id}"`);
                const classes = c.cls ? c.cls.split(" ") : [];
                if (c.sortKey && !classes.includes("sortable")) classes.push("sortable");
                if (classes.length) attrs.push(`class="${classes.join(" ")}"`);
                if (c.sortKey) attrs.push(`data-sort="${c.sortKey}"`);
                if (c.title) attrs.push(`title="${c.title}"`);
                return `<th ${attrs.join(" ")}>${c.label}</th>`;
            })
            .join("");
        theadEl.innerHTML = `<tr>${ths}</tr>`;
    }

    function renderTableRow(columns, row, ctx = {}) {
        const tds = columns
            .filter((c) => !c.hidden || !c.hidden(ctx))
            .map((c) => {
                const clsVal = c.dynamicTdClass ? c.dynamicTdClass(row, ctx) : (c.tdClass || "");
                const cls = clsVal ? ` class="${clsVal}"` : "";
                return `<td${cls}>${c.render(row, ctx)}</td>`;
            })
            .join("");
        return `<tr>${tds}</tr>`;
    }

    function visibleColCount(columns, ctx = {}) {
        return columns.filter((c) => !c.hidden || !c.hidden(ctx)).length;
    }

    const VALUE_LABELS = {
        "vbd_score": "VBD",
        "vorp": "VORP",
        "vona": "VONA",
        "vols": "VOLS",
    };

    const VALUE_TOOLTIPS = {
        "vbd_score": "Composite VBD Score — aggregates VORP + VONA + VOLS into one number for cross-position comparison",
        "vorp": "Value Over Replacement Player — points above the best waiver-wire player at this position",
        "vona": "Value Over Next Available — how much this position's value drops before your next pick",
        "vols": "Value Over Last Starter — points above the worst starter at this position across the league",
    };

    function updateValueHeader() {
        const el = $("#value-header");
        if (el) {
            el.textContent = VALUE_LABELS[currentValueMode] || "Value";
            el.title = VALUE_TOOLTIPS[currentValueMode] || "";
        }
    }

    // Value mode tabs (VBD / VORP / VONA / VOLS)
    $$(".tab").forEach((tab) => {
        tab.addEventListener("click", () => {
            $$(".tab").forEach((t) => t.classList.remove("active"));
            tab.classList.add("active");
            currentValueMode = tab.dataset.value;
            if (currentState) renderRecommendations(currentState);
        });
    });

    // Projection source tabs
    $$(".proj-tab").forEach((tab) => {
        tab.addEventListener("click", async () => {
            if (tab.disabled) return;
            $$(".proj-tab").forEach((t) => t.classList.remove("active"));
            tab.classList.add("active");
            currentProj = tab.dataset.proj;

            try {
                await fetch(
                    apiUrl("/api/projections", { source: currentProj }),
                    { method: "POST" }
                );
                const stateResp = await fetch(apiUrl("/api/state"));
                const state = await stateResp.json();
                resetExtraRecs();
                updateUI(state);
            } catch (err) {
                console.error("Projection switch failed:", err);
            }
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

    // Sim panel toggle
    simToggle.addEventListener("click", () => {
        simCollapsed = !simCollapsed;
        simContent.classList.toggle("hidden", simCollapsed);
        simToggle.textContent = simCollapsed ? "Show" : "Hide";
    });

    // Position filter checkboxes and rookie filter are now created
    // dynamically by initSportUI() — no static listeners needed here.

    // Initialize default position filters for NFL on page load
    initSportUI();

    // Show more button
    showMoreBtn.addEventListener("click", async () => {
        displayCount += 10;

        // Check if we have enough filtered recs already loaded
        const allRecs = getAllRecsForStrategy(currentStrategy);
        const availableFiltered = allRecs.filter((r) => posFilters.has(r.position) && (!rookieOnly || r.is_rookie)).length;

        if (availableFiltered >= displayCount) {
            // Enough data already loaded, just re-render
            renderRecommendations(currentState);
            return;
        }

        // Need more data from backend
        if (prefetchedRecs && Object.keys(prefetchedRecs).length) {
            _mergeNestedRecs(prefetchedRecs, extraRecs);
            const anyRisk = Object.values(prefetchedRecs)[0] || [];
            const prefetchCount = anyRisk.length || 0;
            totalLoaded += prefetchCount;
            if (prefetchCount < displayCount) backendExhausted = true;
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
            const anyRisk = Object.values(moreRecs)[0] || [];
            totalLoaded += anyRisk.length;
            if (anyRisk.length < displayCount) backendExhausted = true;
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
        totalLoaded = 30;
        displayCount = 10;
        backendExhausted = false;
    }

    function getVisibleCount() {
        // Count how many rows are currently visible after filtering
        const recs = getAllRecsForStrategy(currentStrategy);
        return recs.filter((r) => posFilters.has(r.position) && (!rookieOnly || r.is_rookie)).length || 10;
    }

    function getAllRecsForStrategy(_strategy) {
        // recommendations is flat: { risk: [...] }
        const base = (currentState?.recommendations || {})[currentRisk] || [];
        const extra = extraRecs[currentRisk] || [];
        return [...base, ...extra];
    }

    function _mergeNestedRecs(source, target) {
        // Merge { risk: [...] } from source into target
        for (const [risk, recs] of Object.entries(source)) {
            if (!target[risk]) target[risk] = [];
            target[risk].push(...recs);
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
                    const state = await stateResp.json();
                    connectedDraftId = draftId;
                    connectedSlot = slot;
                    draftMode = state.mode || "sleeper";
                    draftSport = state.sport || "nfl";
                    initSportUI();
                    showConnectedUI(draftId, slot, null);
                    if (draftMode === "manual") {
                        manualPickControls.classList.remove("hidden");
                    }
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
            draftMode = "sleeper";
            draftSport = "nfl";
            connectedDraftId = draftId;
            connectedSlot = slot;

            initSportUI();

            // Enable/disable Sleeper projection tab based on availability
            const sleeperProjTab = document.querySelector('.proj-tab[data-proj="sleeper"]');
            if (sleeperProjTab) {
                const available = data.sleeper_projections_available !== false;
                sleeperProjTab.disabled = !available;
                sleeperProjTab.title = available
                    ? `Sleeper's season-long projections (${data.sleeper_projections_matched || 0} players)`
                    : "Sleeper projections not available";
            }

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
        if (draftMode === "manual") url.searchParams.set("mode", "manual");
        window.history.replaceState({}, "", url);

        welcomeScreen.classList.add("hidden");
        connectionBar.classList.remove("hidden");
        statusBar.classList.remove("hidden");
        mainContent.classList.remove("hidden");

        const sleeperInfo = $("#sleeper-connect-info");
        const manualInfo = $("#manual-connect-info");

        if (draftMode === "manual") {
            // Hide Sleeper header, show manual header
            sleeperInfo.classList.add("hidden");
            manualInfo.classList.remove("hidden");
            const sportLabel = draftSport === "mlb" ? "MLB" : "NFL";
            const teamsText = data ? `${data.num_teams} teams, ${data.rounds} rounds, slot ${slot}` : "";
            $("#header-manual-label").textContent = `Manual ${sportLabel} Draft`;
            $("#header-teams-manual").textContent = teamsText;
            manualPickControls.classList.remove("hidden");
        } else {
            sleeperInfo.classList.remove("hidden");
            manualInfo.classList.add("hidden");
            headerDraftId.value = draftId;
            headerSlot.value = slot;
            headerTeams.textContent = data ? `${data.num_teams} teams` : "";
        }
    }

    // Connect button click (welcome screen)
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

    // Header connect button (reconnect / change draft)
    headerConnectBtn.addEventListener("click", () => {
        const draftId = parseDraftId(headerDraftId.value);
        const slot = parseInt(headerSlot.value, 10);
        if (!draftId || !slot || slot < 1) return;
        if (eventSource) eventSource.close();
        sessionId = null;
        connectToDraft(draftId, slot, null);
    });

    function startSSE() {
        if (eventSource) eventSource.close();

        eventSource = new EventSource(apiUrl("/api/stream"));

        eventSource.addEventListener("draft_update", (e) => {
            const state = JSON.parse(e.data);
            resetExtraRecs();  // new picks invalidate extra recs
            simData = null;  // new picks invalidate sim results
            updateUI(state);
        });

        eventSource.addEventListener("sim_update", (e) => {
            simData = JSON.parse(e.data);
            renderSimInsights(simData);
            // Update sim value column in existing rec table if visible
            if (currentState) renderRecommendations(currentState);
        });

        eventSource.onerror = () => {
            console.warn("SSE connection error — browser will auto-reconnect");
        };
    }

    function updateUI(state) {
        currentState = state;

        // Sync projection source tab from server state
        if (state.projection_source && state.projection_source !== currentProj) {
            currentProj = state.projection_source;
            $$(".proj-tab").forEach((t) => {
                t.classList.toggle("active", t.dataset.proj === currentProj);
            });
        }

        roundDisplay.textContent = `Round ${state.current_round} / ${state.total_rounds}`;
        pickDisplay.textContent = `Pick ${state.current_pick} / ${state.total_picks}`;
        draftStatusBadge.textContent = state.draft_status;

        if (state.is_complete) {
            turnIndicator.classList.add("hidden");
            picksUntil.textContent = "Draft Complete";
        } else if (state.is_my_turn) {
            turnIndicator.classList.remove("hidden");
            if (draftMode === "manual") {
                picksUntil.textContent = "Your turn — search and pick!";
            } else {
                picksUntil.textContent = "On the clock!";
                sendTurnNotification();
            }
        } else {
            turnIndicator.classList.add("hidden");
            const teamNum = (state.current_team ?? -1) + 1;
            if (draftMode === "manual" && teamNum > 0) {
                picksUntil.textContent = `Pick ${state.current_pick}: Team ${teamNum}'s turn (${state.picks_until_next} until yours)`;
            } else {
                picksUntil.textContent = `${state.picks_until_next} picks until your turn`;
            }
        }

        renderRecommendations(state);
        renderRoster(state);
        renderTicker(state);
        renderBoard(state);
        renderSimInsights(simData);

        // Prefetch more recs in background when it's our turn
        if (state.is_my_turn && !Object.keys(prefetchedRecs).length) {
            prefetchMore();
        }
    }

    function renderRecommendations(state) {
        const allRecs = getAllRecsForStrategy(currentStrategy);
        const floorEstimated = !!(state.floor_estimated);
        const ctx = { showSim: !!simData, currentStrategy, getSimValue, valueMode: currentValueMode, floorEstimated };

        // Render header (updates value label, sim column visibility)
        renderTableHeader(REC_COLUMNS, $("#rec-head"), ctx);
        updateValueHeader();

        // Re-bind sortable header clicks (header was just re-rendered)
        $$("#rec-head th.sortable").forEach((th) => {
            th.addEventListener("click", () => {
                currentSort = th.dataset.sort;
                $$("#rec-head th.sortable").forEach((t) => t.classList.remove("active"));
                th.classList.add("active");
                if (currentState) renderRecommendations(currentState);
            });
        });

        // When rookieOnly is active, merge available_rookies from state
        // so rookies appear even if they didn't rank in strategy-scored recs
        let pool = allRecs;
        if (rookieOnly && currentState?.available_rookies) {
            const recNames = new Set(allRecs.map((r) => r.name));
            const extras = currentState.available_rookies
                .filter((r) => !recNames.has(r.name))
                .map((r, i) => ({ ...r, rank: allRecs.length + i + 1 }));
            pool = [...allRecs, ...extras];
        }

        // Apply position filter first, then cap to displayCount
        let filtered = pool.filter((r) => posFilters.has(r.position) && (!rookieOnly || r.is_rookie));

        // Apply sort
        if (currentSort === "adp") {
            filtered = filtered.slice().sort((a, b) => a.adp - b.adp);
        } else if (currentSort === "value") {
            const key = currentValueMode;
            filtered = filtered.slice().sort((a, b) => (b[key] || 0) - (a[key] || 0));
        } else {
            filtered = filtered.slice().sort((a, b) => a.rank - b.rank);
        }

        // Cap to displayCount visible rows
        const hasMore = filtered.length > displayCount;
        filtered = filtered.slice(0, displayCount);

        const colSpan = visibleColCount(REC_COLUMNS, ctx);

        if (!filtered.length) {
            const msg = !allRecs.length
                ? (state.is_my_turn ? "No recommendations available" : "Recommendations appear on your turn")
                : "No players match current filters";
            recBody.innerHTML =
                `<tr><td colspan="${colSpan}" class="empty-msg">${msg}</td></tr>`;
            showMoreBtn.classList.add("hidden");
            return;
        }

        recBody.innerHTML = filtered
            .map((r) => renderTableRow(REC_COLUMNS, r, ctx))
            .join("");

        // Show "show more" button when it's our turn and there are more to show
        // (or backend may still have more data for this filter)
        if (state.is_my_turn && (hasMore || !backendExhausted)) {
            showMoreBtn.classList.remove("hidden");
            showMoreBtn.textContent = `Show ${displayCount} more`;
        } else {
            showMoreBtn.classList.add("hidden");
        }
    }

    function renderRoster(state) {
        let roster = state.user_roster || [];
        const needs = state.team_needs || {};

        renderTableHeader(ROSTER_COLUMNS, $("#roster-head"));

        // Mark the active sort column and bind click handlers
        $$("#roster-head th.sortable").forEach((th) => {
            const key = th.dataset.sort;
            if (rosterSort && rosterSort.key === key) {
                th.classList.add("active");
                th.textContent += rosterSort.asc ? " ▲" : " ▼";
            }
            th.addEventListener("click", () => {
                if (rosterSort && rosterSort.key === key) {
                    rosterSort.asc = !rosterSort.asc;
                } else {
                    // Default: strings ascending, numbers descending
                    const numericKeys = new Set(["bye_week", "projected_total"]);
                    rosterSort = { key, asc: !numericKeys.has(key) };
                }
                if (currentState) renderRoster(currentState);
            });
        });

        rosterNeeds.innerHTML = Object.entries(needs)
            .map(([pos, count]) => {
                const urgent = count >= 2 ? "urgent" : "";
                return `<span class="need-chip ${urgent}">${pos}: ${count} needed</span>`;
            })
            .join("");

        const colSpan = visibleColCount(ROSTER_COLUMNS);

        if (!roster.length) {
            rosterBody.innerHTML =
                `<tr><td colspan="${colSpan}" class="empty-msg">No picks yet</td></tr>`;
            return;
        }

        // Sort roster if a sort column is active
        if (rosterSort) {
            const k = rosterSort.key;
            roster = roster.slice().sort((a, b) => {
                const av = a[k] ?? "", bv = b[k] ?? "";
                const cmp = typeof av === "number" && typeof bv === "number"
                    ? av - bv
                    : String(av).localeCompare(String(bv));
                return rosterSort.asc ? cmp : -cmp;
            });
        }

        rosterBody.innerHTML = roster
            .map((p) => renderTableRow(ROSTER_COLUMNS, p))
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
                <span>${p.player_name}${p.is_rookie ? ' <span class="rookie-badge" title="Rookie">R</span>' : ""}</span>
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

    function renderSimInsights(data) {
        if (!data || !data.sims_target) {
            simPanel.classList.add("hidden");
            return;
        }

        simPanel.classList.remove("hidden");

        // Progress bar
        const pct = Math.min(100, (data.sims_completed / data.sims_target) * 100);
        simProgressFill.style.width = pct + "%";
        const done = data.sims_completed >= data.sims_target;
        simProgressText.textContent = done
            ? `Complete (${data.sims_completed} sims)`
            : `Simulating... ${data.sims_completed}/${data.sims_target}`;

        // Strategy summaries
        const strats = data.strategies || {};
        let html = "";
        for (const [name, sr] of Object.entries(strats)) {
            html += `<div class="sim-strategy-row">
                <span class="sim-strat-name">${name.toUpperCase()}</span>
                <span class="sim-strat-mean">${sr.mean.toFixed(1)} pts</span>
                <span class="sim-strat-range">${sr.p10.toFixed(0)}&ndash;${sr.p90.toFixed(0)}</span>
            </div>`;
        }
        simStrategies.innerHTML = html;
    }

    function getSimValue(playerName) {
        if (!simData || !simData.strategies) return null;
        // Use the current strategy's pick_values if available
        const sr = simData.strategies[currentStrategy];
        if (sr && sr.pick_values && sr.pick_values[playerName] != null) {
            return sr.pick_values[playerName];
        }
        // Fallback: check all strategies
        for (const sr2 of Object.values(simData.strategies)) {
            if (sr2.pick_values && sr2.pick_values[playerName] != null) {
                return sr2.pick_values[playerName];
            }
        }
        return null;
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

    // Render initial empty table headers
    const initCtx = { showSim: false, valueMode: currentValueMode };
    renderTableHeader(REC_COLUMNS, $("#rec-head"), initCtx);
    recBody.innerHTML = `<tr><td colspan="${visibleColCount(REC_COLUMNS, initCtx)}" class="empty-msg">Connect to a draft to see recommendations</td></tr>`;
    renderTableHeader(ROSTER_COLUMNS, $("#roster-head"));
    rosterBody.innerHTML = `<tr><td colspan="${visibleColCount(ROSTER_COLUMNS)}" class="empty-msg">No picks yet</td></tr>`;

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
