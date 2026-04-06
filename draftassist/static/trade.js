// Trade Calculator – client-side logic
"use strict";

let sessionId = null;
let teamsData = [];       // [{team_id, name, roster, needs}, ...]
let sideAPlayers = [];    // player objects added to side A
let sideBPlayers = [];    // player objects added to side B
let freeAgents = {};      // {position: [players]}

// ---------------------------------------------------------------------------
// Platform tabs
// ---------------------------------------------------------------------------
document.querySelectorAll(".mode-tab[data-platform]").forEach(tab => {
    tab.addEventListener("click", () => {
        document.querySelectorAll(".mode-tab[data-platform]").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        const platform = tab.dataset.platform;
        document.getElementById("sleeper-connect").classList.toggle("hidden", platform !== "sleeper");
        document.getElementById("yahoo-connect").classList.toggle("hidden", platform !== "yahoo");
    });
});

// ---------------------------------------------------------------------------
// Connect
// ---------------------------------------------------------------------------
document.getElementById("connect-btn").addEventListener("click", connectSleeper);
document.getElementById("yahoo-connect-btn").addEventListener("click", connectYahoo);

async function connectSleeper() {
    const input = document.getElementById("league-id").value.trim();
    const status = document.getElementById("connect-status");
    const btn = document.getElementById("connect-btn");

    // Extract league ID from URL if needed
    let leagueId = input;
    const urlMatch = input.match(/sleeper\.com\/leagues\/(\w+)/i) || input.match(/sleeper\.com\/league\/\w+\/(\w+)/i);
    if (urlMatch) leagueId = urlMatch[1];

    if (!leagueId) {
        status.textContent = "Please enter a league ID";
        status.style.color = "#ff5252";
        return;
    }

    const sport = document.querySelector('input[name="sport"]:checked').value;

    btn.disabled = true;
    status.textContent = "Connecting...";
    status.style.color = "#4fc3f7";

    try {
        const resp = await fetch("/api/trade/connect-sleeper", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({league_id: leagueId, sport}),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${resp.status}`);
        }
        const data = await resp.json();
        sessionId = data.session_id;

        document.getElementById("trade-league-name").textContent = data.league_name;
        document.getElementById("trade-welcome").classList.add("hidden");
        document.getElementById("trade-main").classList.remove("hidden");

        // Load full team data
        await loadTeams();
        await loadFreeAgents();
    } catch (e) {
        status.textContent = `Error: ${e.message}`;
        status.style.color = "#ff5252";
    } finally {
        btn.disabled = false;
    }
}

async function connectYahoo() {
    const leagueId = document.getElementById("yahoo-league-id").value.trim();
    const key = document.getElementById("yahoo-key").value.trim();
    const secret = document.getElementById("yahoo-secret").value.trim();
    const status = document.getElementById("yahoo-status");
    const btn = document.getElementById("yahoo-connect-btn");
    const sport = document.querySelector('input[name="yahoo-sport"]:checked').value;

    if (!leagueId) { status.textContent = "Please enter a league ID"; status.style.color = "#ff5252"; return; }
    if (!key || !secret) { status.textContent = "Consumer key and secret are required"; status.style.color = "#ff5252"; return; }

    btn.disabled = true;
    status.textContent = "Connecting...";
    status.style.color = "#4fc3f7";

    try {
        const resp = await fetch("/api/trade/connect-yahoo", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({league_id: leagueId, sport, consumer_key: key, consumer_secret: secret}),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${resp.status}`);
        }
        const data = await resp.json();
        sessionId = data.session_id;

        document.getElementById("trade-league-name").textContent = data.league_name;
        document.getElementById("trade-welcome").classList.add("hidden");
        document.getElementById("trade-main").classList.remove("hidden");

        await loadTeams();
        await loadFreeAgents();
    } catch (e) {
        status.textContent = `Error: ${e.message}`;
        status.style.color = "#ff5252";
    } finally {
        btn.disabled = false;
    }
}

// ---------------------------------------------------------------------------
// Load teams
// ---------------------------------------------------------------------------
async function loadTeams() {
    const resp = await fetch(`/api/trade/teams?session_id=${sessionId}`);
    const data = await resp.json();
    teamsData = data.teams;

    // Populate team selectors
    const selA = document.getElementById("team-a-select");
    const selB = document.getElementById("team-b-select");
    [selA, selB].forEach(sel => {
        sel.innerHTML = '<option value="">Select Team</option>';
        teamsData.forEach(t => {
            const opt = document.createElement("option");
            opt.value = t.team_id;
            opt.textContent = `${t.name} (${t.roster.length} players)`;
            sel.appendChild(opt);
        });
    });

    selA.addEventListener("change", () => onTeamSelected("a"));
    selB.addEventListener("change", () => onTeamSelected("b"));
}

function onTeamSelected(side) {
    const sel = document.getElementById(`team-${side}-select`);
    const teamId = sel.value;
    const searchInput = document.getElementById(`search-${side}`);

    // Clear current players
    if (side === "a") sideAPlayers = [];
    else sideBPlayers = [];
    renderSidePlayers(side);

    if (teamId) {
        searchInput.disabled = false;
        searchInput.placeholder = "Type player name to add...";
        renderNeedsPanel();
    } else {
        searchInput.disabled = true;
        searchInput.placeholder = "Search players...";
    }
    updateEvaluateButton();
}

// ---------------------------------------------------------------------------
// Player search (autocomplete)
// ---------------------------------------------------------------------------
["a", "b"].forEach(side => {
    const input = document.getElementById(`search-${side}`);
    const dropdown = document.getElementById(`search-${side}-results`);
    let debounce = null;

    input.addEventListener("input", () => {
        clearTimeout(debounce);
        debounce = setTimeout(() => searchPlayers(side), 200);
    });

    input.addEventListener("focus", () => {
        if (input.value.length >= 1) searchPlayers(side);
    });

    // Close dropdown on outside click
    document.addEventListener("click", (e) => {
        if (!input.contains(e.target) && !dropdown.contains(e.target)) {
            dropdown.classList.add("hidden");
        }
    });
});

async function searchPlayers(side) {
    const input = document.getElementById(`search-${side}`);
    const dropdown = document.getElementById(`search-${side}-results`);
    const teamId = document.getElementById(`team-${side}-select`).value;
    const query = input.value.trim();

    if (!teamId || query.length < 1) {
        dropdown.classList.add("hidden");
        return;
    }

    // Filter from local team roster data
    const team = teamsData.find(t => t.team_id === teamId);
    if (!team) return;

    const alreadyAdded = new Set(
        (side === "a" ? sideAPlayers : sideBPlayers).map(p => p.sleeper_id)
    );

    const matches = team.roster.filter(p =>
        p.name.toLowerCase().includes(query.toLowerCase()) && !alreadyAdded.has(p.sleeper_id)
    ).slice(0, 8);

    if (matches.length === 0) {
        dropdown.classList.add("hidden");
        return;
    }

    dropdown.innerHTML = "";
    matches.forEach(p => {
        const div = document.createElement("div");
        div.className = "trade-search-item";
        div.innerHTML = `
            <span class="pos-badge pos-${p.position}">${p.position}</span>
            <span class="search-name">${p.name}</span>
            <span class="search-team">${p.team}</span>
            <span class="search-value">${p.trade_value}</span>
        `;
        div.addEventListener("click", () => {
            addPlayer(side, p);
            input.value = "";
            dropdown.classList.add("hidden");
        });
        dropdown.appendChild(div);
    });
    dropdown.classList.remove("hidden");
}

// ---------------------------------------------------------------------------
// Player management
// ---------------------------------------------------------------------------
function addPlayer(side, player) {
    if (side === "a") sideAPlayers.push(player);
    else sideBPlayers.push(player);
    renderSidePlayers(side);
    updateEvaluateButton();
}

function removePlayer(side, sleeperId) {
    if (side === "a") sideAPlayers = sideAPlayers.filter(p => p.sleeper_id !== sleeperId);
    else sideBPlayers = sideBPlayers.filter(p => p.sleeper_id !== sleeperId);
    renderSidePlayers(side);
    updateEvaluateButton();
    // Clear results if trade was evaluated
    document.getElementById("trade-result").classList.add("hidden");
}

function renderSidePlayers(side) {
    const container = document.getElementById(`side-${side}-players`);
    const players = side === "a" ? sideAPlayers : sideBPlayers;
    const valueEl = document.getElementById(`side-${side}-value`);

    if (players.length === 0) {
        container.innerHTML = '<div class="trade-empty-msg">Select a team and add players</div>';
        valueEl.textContent = "0";
        return;
    }

    let totalValue = 0;
    container.innerHTML = "";
    players.forEach(p => {
        totalValue += p.trade_value;
        const div = document.createElement("div");
        div.className = "trade-player-card";
        div.innerHTML = `
            <div class="trade-player-info">
                <span class="pos-badge pos-${p.position}">${p.position}</span>
                <span class="trade-player-name">${p.name}</span>
                <span class="trade-player-team">${p.team}</span>
            </div>
            <div class="trade-player-stats">
                <span class="trade-stat" title="Projected PPG">${p.projected_ppg} ppg</span>
                <span class="trade-stat" title="Season total">${p.projected_total} pts</span>
                <span class="trade-stat trade-value-badge" title="Trade value (VORP)">${p.trade_value}</span>
            </div>
            <button class="trade-remove-btn" title="Remove">&times;</button>
        `;
        div.querySelector(".trade-remove-btn").addEventListener("click", () => removePlayer(side, p.sleeper_id));
        container.appendChild(div);
    });

    valueEl.textContent = totalValue.toFixed(1);
}

function updateEvaluateButton() {
    const btn = document.getElementById("evaluate-btn");
    const teamA = document.getElementById("team-a-select").value;
    const teamB = document.getElementById("team-b-select").value;
    btn.disabled = !(teamA && teamB && (sideAPlayers.length > 0 || sideBPlayers.length > 0));
}

// ---------------------------------------------------------------------------
// Evaluate trade
// ---------------------------------------------------------------------------
document.getElementById("evaluate-btn").addEventListener("click", evaluateTrade);

async function evaluateTrade() {
    const btn = document.getElementById("evaluate-btn");
    btn.disabled = true;
    btn.textContent = "Evaluating...";

    try {
        const resp = await fetch("/api/trade/evaluate", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                session_id: sessionId,
                team_a: document.getElementById("team-a-select").value,
                team_b: document.getElementById("team-b-select").value,
                team_a_gives: sideAPlayers.map(p => p.sleeper_id),
                team_b_gives: sideBPlayers.map(p => p.sleeper_id),
            }),
        });
        const result = await resp.json();
        renderTradeResult(result);
    } catch (e) {
        console.error("Evaluate error:", e);
    } finally {
        btn.disabled = false;
        btn.textContent = "Evaluate Trade";
        updateEvaluateButton();
    }
}

function renderTradeResult(result) {
    const panel = document.getElementById("trade-result");
    panel.classList.remove("hidden");

    // Headline
    const headline = document.getElementById("result-headline");
    if (result.winner === "even") {
        headline.textContent = "Fair Trade";
        headline.className = "result-even";
    } else {
        const winnerName = result.winner === result.team_a.team_id ? result.team_a.name : result.team_b.name;
        headline.textContent = `${winnerName} wins this trade`;
        headline.className = "result-winner";
    }

    // Reasons
    const reasonsList = document.getElementById("result-reasons");
    reasonsList.innerHTML = "";
    result.reasons.forEach(r => {
        const li = document.createElement("li");
        li.textContent = r;
        reasonsList.appendChild(li);
    });

    // Fairness bar
    const delta = result.fairness_delta;
    const bar = document.getElementById("fairness-bar");
    const maxDelta = 100;
    const pct = Math.min(Math.abs(delta) / maxDelta, 1) * 50;
    if (delta >= 0) {
        bar.style.left = "50%";
        bar.style.right = "auto";
        bar.style.width = pct + "%";
        bar.className = "fairness-favors-a";
    } else {
        bar.style.right = "50%";
        bar.style.left = "auto";
        bar.style.width = pct + "%";
        bar.className = "fairness-favors-b";
    }
    document.getElementById("fairness-delta").textContent =
        Math.abs(delta) < 5 ? "Even" : `${delta > 0 ? "+" : ""}${delta.toFixed(1)} VORP`;
    document.getElementById("fairness-label").textContent = "";

    // Roster impact
    renderImpactSide("a", result.team_a);
    renderImpactSide("b", result.team_b);
}

function renderImpactSide(side, data) {
    document.getElementById(`impact-${side}-name`).textContent = data.name;
    document.getElementById(`impact-${side}-before`).textContent = data.starter_vorp_before.toFixed(0);
    document.getElementById(`impact-${side}-after`).textContent = data.starter_vorp_after.toFixed(0);

    const delta = data.roster_impact;
    const deltaEl = document.getElementById(`impact-${side}-delta`);
    deltaEl.textContent = `(${delta > 0 ? "+" : ""}${delta.toFixed(1)})`;
    deltaEl.className = delta > 0 ? "impact-delta delta-pos" : delta < 0 ? "impact-delta delta-neg" : "impact-delta";

    // Needs
    const needsEl = document.getElementById(`impact-${side}-needs`);
    needsEl.innerHTML = "";
    const needsAfter = data.needs_after;
    for (const [pos, info] of Object.entries(needsAfter)) {
        if (info.need === 0) continue;
        const div = document.createElement("div");
        div.className = `need-item need-${info.severity}`;
        div.innerHTML = `<span class="need-pos">${pos}</span> <span class="need-count">${info.have}/${info.need}</span>`;
        needsEl.appendChild(div);
    }
}

// ---------------------------------------------------------------------------
// Roster Needs Panel
// ---------------------------------------------------------------------------
function renderNeedsPanel() {
    const container = document.getElementById("needs-content");
    const teamAId = document.getElementById("team-a-select").value;
    const teamBId = document.getElementById("team-b-select").value;

    const selectedTeams = [teamAId, teamBId].filter(Boolean);
    if (selectedTeams.length === 0) {
        container.innerHTML = '<p class="trade-empty-msg">Select teams above to see needs</p>';
        return;
    }

    container.innerHTML = "";
    selectedTeams.forEach(tid => {
        const team = teamsData.find(t => t.team_id === tid);
        if (!team) return;

        const section = document.createElement("div");
        section.className = "needs-team-section";

        const header = document.createElement("h4");
        header.textContent = team.name;
        section.appendChild(header);

        const grid = document.createElement("div");
        grid.className = "needs-grid";

        for (const [pos, info] of Object.entries(team.needs)) {
            if (info.need === 0) continue;
            const cell = document.createElement("div");
            cell.className = `need-cell need-${info.severity}`;
            cell.innerHTML = `
                <span class="need-pos">${pos}</span>
                <span class="need-bar-label">${info.have}/${info.need}</span>
                <div class="need-bar">
                    <div class="need-bar-fill" style="width: ${Math.min(info.have / Math.max(info.need, 1), 1) * 100}%"></div>
                </div>
            `;
            grid.appendChild(cell);
        });

        section.appendChild(grid);
        container.appendChild(section);
    });
}

// ---------------------------------------------------------------------------
// Free Agents Panel
// ---------------------------------------------------------------------------
async function loadFreeAgents() {
    const resp = await fetch(`/api/trade/free-agents?session_id=${sessionId}&limit=5`);
    const data = await resp.json();
    freeAgents = data.free_agents;
    renderFreeAgents();
}

function renderFreeAgents() {
    const container = document.getElementById("fa-content");
    container.innerHTML = "";

    // Sort positions by highest VORP available
    const positions = Object.entries(freeAgents)
        .filter(([_, players]) => players.length > 0)
        .sort((a, b) => (b[1][0]?.vorp || 0) - (a[1][0]?.vorp || 0));

    if (positions.length === 0) {
        container.innerHTML = '<p class="trade-empty-msg">No free agent data available</p>';
        return;
    }

    positions.forEach(([pos, players]) => {
        const section = document.createElement("div");
        section.className = "fa-pos-section";

        const header = document.createElement("div");
        header.className = "fa-pos-header";
        header.innerHTML = `<span class="pos-badge pos-${pos}">${pos}</span> <span class="fa-pos-count">${players.length} available</span>`;
        section.appendChild(header);

        const list = document.createElement("div");
        list.className = "fa-player-list";

        players.slice(0, 5).forEach(p => {
            const row = document.createElement("div");
            row.className = "fa-player-row";
            row.innerHTML = `
                <span class="fa-name">${p.name}</span>
                <span class="fa-team">${p.team}</span>
                <span class="fa-ppg">${p.projected_ppg} ppg</span>
                <span class="fa-vorp">${p.vorp}</span>
            `;
            list.appendChild(row);
        });

        section.appendChild(list);
        container.appendChild(section);
    });
}
