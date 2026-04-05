// Projection Adjustments Editor
(function () {
    "use strict";

    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    // State
    let players = [];
    let adjustments = {};   // name -> {adjustment_ppg, adjustment_games, adjustment_volatility, note}
    let dirty = false;
    let posFilters = new Set(["QB", "RB", "WR", "TE"]);
    let searchQuery = "";
    let sortKey = "projected_total";
    let sortAsc = false;
    let displayCount = 50;
    let changedOnly = false;

    const POSITIONS = ["QB", "RB", "WR", "TE"];

    // --- Column definitions ---

    function escHtml(s) {
        const d = document.createElement("div");
        d.textContent = s;
        return d.innerHTML;
    }

    const COLUMNS = [
        {
            key: "pos_rank", label: "#", sortKey: "pos_rank",
            cls: "sortable",
            render: (p) => p.pos_rank,
        },
        {
            key: "name", label: "Player", sortKey: "name",
            cls: "sortable",
            render: (p) => `<strong>${escHtml(p.name)}</strong>`,
        },
        {
            key: "position", label: "Pos",
            render: (p) => `<span class="pos-badge pos-${p.position}">${p.position}</span>`,
        },
        {
            key: "team", label: "Team", sortKey: "team",
            cls: "sortable",
            render: (p) => escHtml(p.team),
        },
        {
            key: "projected_ppg", label: "PPG", sortKey: "projected_ppg",
            cls: "sortable",
            title: "Model projected points per game",
            render: (p) => p.projected_ppg.toFixed(1),
        },
        {
            key: "projected_games", label: "Games", sortKey: "projected_games",
            cls: "sortable",
            title: "Model projected games played",
            render: (p) => p.projected_games.toFixed(1),
        },
        {
            key: "projected_total", label: "Total", sortKey: "projected_total",
            cls: "sortable",
            title: "Model projected season total",
            render: (p) => p.projected_total,
        },
        {
            key: "upside", label: "Spread", sortKey: "upside",
            cls: "sortable",
            title: "Ceiling minus Floor (volatility)",
            render: (p) => p.upside,
        },
        {
            key: "adj_ppg", label: "Adj PPG",
            title: "Additive PPG adjustment (shifts all projections up/down)",
            cls: "adj-col-header",
            render: (p) => renderInput(p, "adjustment_ppg", 0.5, -10, 10),
        },
        {
            key: "adj_games", label: "Adj Games",
            title: "Additive games played adjustment",
            cls: "adj-col-header",
            render: (p) => renderInput(p, "adjustment_games", 1, -17, 17),
        },
        {
            key: "adj_vol", label: "Adj Vol",
            title: "Volatility adjustment: +0.1 = 10% wider floor/ceiling spread",
            cls: "adj-col-header",
            render: (p) => renderInput(p, "adjustment_volatility", 0.05, -1, 5),
        },
        {
            key: "note", label: "Note",
            cls: "adj-col-header",
            render: (p) => renderNoteInput(p),
        },
        {
            key: "preview_total", label: "Adj Total",
            title: "Adjusted projected total (PPG + adj) * (Games + adj)",
            cls: "sortable",
            sortKey: "preview_total",
            render: (p) => {
                const pv = computePreview(p);
                const delta = pv.total - p.projected_total;
                if (delta === 0) return pv.total;
                const cls = delta > 0 ? "delta-pos" : "delta-neg";
                const arrow = delta > 0 ? "\u2191" : "\u2193";
                return `<span class="${cls}">${pv.total} ${arrow}${Math.abs(delta)}</span>`;
            },
        },
        {
            key: "preview_floor", label: "Adj Floor",
            title: "Adjusted 10th percentile total",
            render: (p) => {
                const pv = computePreview(p);
                return `<span class="range-floor-text">${pv.floor}</span>`;
            },
        },
        {
            key: "preview_ceil", label: "Adj Ceil",
            title: "Adjusted 90th percentile total",
            render: (p) => {
                const pv = computePreview(p);
                return `<span class="range-ceil-text">${pv.ceil}</span>`;
            },
        },
    ];

    // --- Rendering helpers ---

    function renderInput(player, field, step, min, max) {
        const adj = adjustments[player.name] || {};
        const val = adj[field] || 0;
        const changed = val !== 0 ? ' class="adj-input adj-changed"' : ' class="adj-input"';
        return `<input type="number"${changed} data-player="${escHtml(player.name)}" data-field="${field}" value="${val}" step="${step}" min="${min}" max="${max}">`;
    }

    function renderNoteInput(player) {
        const adj = adjustments[player.name] || {};
        const val = adj.note || "";
        return `<input type="text" class="adj-note-input" data-player="${escHtml(player.name)}" data-field="note" value="${escHtml(val)}" placeholder="...">`;
    }

    function computePreview(player) {
        const adj = adjustments[player.name] || {};
        const adjPpg = adj.adjustment_ppg || 0;
        const adjGames = adj.adjustment_games || 0;
        const adjVol = adj.adjustment_volatility || 0;

        const newPpg = player.projected_ppg + adjPpg;
        const newGames = Math.max(0, Math.min(17, player.projected_games + adjGames));
        const total = Math.round(newPpg * newGames);

        // Volatility: scale floor/ceiling distance from median
        const medPpg = player.ppg_median + adjPpg;
        const floorPpg = medPpg + (player.ppg_floor - player.ppg_median) * (1 + adjVol);
        const ceilPpg = medPpg + (player.ppg_ceiling - player.ppg_median) * (1 + adjVol);
        const floor = Math.round(floorPpg * newGames);
        const ceil = Math.round(ceilPpg * newGames);

        return { total, floor, ceil };
    }

    // --- Table rendering ---

    function renderTableHeader() {
        const thead = $("#adj-head");
        const ths = COLUMNS.map((c) => {
            const attrs = [];
            const classes = c.cls ? c.cls.split(" ") : [];
            if (c.sortKey) {
                attrs.push(`data-sort="${c.sortKey}"`);
                if (!classes.includes("sortable")) classes.push("sortable");
                if (c.sortKey === sortKey) {
                    classes.push("active");
                }
            }
            if (c.title) attrs.push(`title="${c.title}"`);
            if (classes.length) attrs.push(`class="${classes.join(" ")}"`);
            let label = c.label;
            if (c.sortKey === sortKey) {
                label += ` <span class="sort-arrow">${sortAsc ? "\u25b2" : "\u25bc"}</span>`;
            }
            return `<th ${attrs.join(" ")}>${label}</th>`;
        }).join("");
        thead.innerHTML = `<tr>${ths}</tr>`;

        // Bind sort handlers
        thead.querySelectorAll("th[data-sort]").forEach((th) => {
            th.addEventListener("click", () => {
                const key = th.dataset.sort;
                if (sortKey === key) {
                    sortAsc = !sortAsc;
                } else {
                    sortKey = key;
                    sortAsc = key === "name" || key === "team";
                }
                renderTable();
            });
        });
    }

    function getFilteredPlayers() {
        let list = players;

        // Position filter
        list = list.filter((p) => posFilters.has(p.position));

        // Search filter
        if (searchQuery) {
            const q = searchQuery.toLowerCase();
            list = list.filter((p) => p.name.toLowerCase().includes(q));
        }

        // Changed only
        if (changedOnly) {
            list = list.filter((p) => {
                const adj = adjustments[p.name];
                if (!adj) return false;
                return adj.adjustment_ppg || adj.adjustment_games || adj.adjustment_volatility || adj.note;
            });
        }

        // Sort
        list = list.slice().sort((a, b) => {
            let va, vb;
            if (sortKey === "name" || sortKey === "team") {
                va = a[sortKey] || "";
                vb = b[sortKey] || "";
                return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
            } else if (sortKey === "preview_total") {
                va = computePreview(a).total;
                vb = computePreview(b).total;
            } else {
                va = a[sortKey] ?? 0;
                vb = b[sortKey] ?? 0;
            }
            return sortAsc ? va - vb : vb - va;
        });

        return list;
    }

    function renderTable() {
        renderTableHeader();
        const tbody = $("#adj-body");
        const filtered = getFilteredPlayers();
        const showing = filtered.slice(0, displayCount);

        if (showing.length === 0) {
            tbody.innerHTML = `<tr><td colspan="${COLUMNS.length}" class="empty-msg">No players match filters</td></tr>`;
        } else {
            tbody.innerHTML = showing.map((p) => {
                const tds = COLUMNS.map((c) => `<td>${c.render(p)}</td>`).join("");
                const adj = adjustments[p.name];
                const hasAdj = adj && (adj.adjustment_ppg || adj.adjustment_games || adj.adjustment_volatility);
                const cls = hasAdj ? ' class="adj-row-changed"' : "";
                return `<tr${cls}>${tds}</tr>`;
            }).join("");
        }

        // Show more button
        const btn = $("#adj-show-more");
        if (filtered.length > displayCount) {
            btn.classList.remove("hidden");
            btn.textContent = `Show more (${displayCount} of ${filtered.length})`;
        } else {
            btn.classList.add("hidden");
        }

        // Update count
        updateAdjCount();

        // Bind input handlers
        tbody.querySelectorAll(".adj-input, .adj-note-input").forEach((input) => {
            input.addEventListener("change", onInputChange);
            // For number inputs, also update on input for responsiveness
            if (input.type === "number") {
                input.addEventListener("input", onInputChange);
            }
        });
    }

    function onInputChange(e) {
        const input = e.target;
        const name = input.dataset.player;
        const field = input.dataset.field;

        if (!adjustments[name]) {
            adjustments[name] = { adjustment_ppg: 0, adjustment_games: 0, adjustment_volatility: 0, note: "" };
        }

        if (field === "note") {
            adjustments[name].note = input.value;
        } else {
            adjustments[name][field] = parseFloat(input.value) || 0;
        }

        // Update input styling
        if (field !== "note") {
            const val = adjustments[name][field];
            input.classList.toggle("adj-changed", val !== 0);
        }

        setDirty(true);

        // Re-render just the preview cells in this row
        const row = input.closest("tr");
        if (row) {
            const player = players.find((p) => p.name === name);
            if (player) {
                const pv = computePreview(player);
                const cells = row.querySelectorAll("td");
                // Preview total is column index 12, floor 13, ceil 14
                const previewTotalIdx = COLUMNS.findIndex((c) => c.key === "preview_total");
                const previewFloorIdx = COLUMNS.findIndex((c) => c.key === "preview_floor");
                const previewCeilIdx = COLUMNS.findIndex((c) => c.key === "preview_ceil");

                if (cells[previewTotalIdx]) {
                    const delta = pv.total - player.projected_total;
                    if (delta === 0) {
                        cells[previewTotalIdx].innerHTML = pv.total;
                    } else {
                        const cls = delta > 0 ? "delta-pos" : "delta-neg";
                        const arrow = delta > 0 ? "\u2191" : "\u2193";
                        cells[previewTotalIdx].innerHTML = `<span class="${cls}">${pv.total} ${arrow}${Math.abs(delta)}</span>`;
                    }
                }
                if (cells[previewFloorIdx]) {
                    cells[previewFloorIdx].innerHTML = `<span class="range-floor-text">${pv.floor}</span>`;
                }
                if (cells[previewCeilIdx]) {
                    cells[previewCeilIdx].innerHTML = `<span class="range-ceil-text">${pv.ceil}</span>`;
                }

                // Update row changed class
                const adj = adjustments[name];
                const hasAdj = adj && (adj.adjustment_ppg || adj.adjustment_games || adj.adjustment_volatility);
                row.classList.toggle("adj-row-changed", !!hasAdj);
            }
        }

        updateAdjCount();
    }

    function updateAdjCount() {
        const count = Object.values(adjustments).filter((a) =>
            a.adjustment_ppg || a.adjustment_games || a.adjustment_volatility
        ).length;
        const el = $("#adj-count");
        el.textContent = count > 0 ? `${count} adjustment${count !== 1 ? "s" : ""}` : "";
    }

    // --- Dirty state ---

    function setDirty(val) {
        dirty = val;
        const badge = $("#adj-dirty");
        badge.classList.toggle("hidden", !dirty);
        const saveBtn = $("#save-btn");
        saveBtn.classList.toggle("adj-btn-pulse", dirty);
        document.title = dirty ? "* Projection Adjustments" : "Projection Adjustments";
    }

    // --- Position filter ---

    function initPosFilter() {
        const container = $("#adj-pos-filter");

        function updateChipStyles() {
            container.querySelectorAll(".pos-chip[data-pos]").forEach((chip) => {
                chip.classList.toggle("inactive", !posFilters.has(chip.dataset.pos));
            });
        }

        POSITIONS.forEach((pos) => {
            const chip = document.createElement("span");
            chip.className = `pos-chip pos-${pos}`;
            chip.textContent = pos;
            chip.dataset.pos = pos;
            chip.addEventListener("click", (e) => {
                e.preventDefault();
                if (e.shiftKey || e.ctrlKey || e.metaKey) {
                    if (posFilters.has(pos)) {
                        if (posFilters.size > 1) posFilters.delete(pos);
                    } else {
                        posFilters.add(pos);
                    }
                } else {
                    if (posFilters.size === 1 && posFilters.has(pos)) {
                        posFilters = new Set(POSITIONS);
                    } else {
                        posFilters = new Set([pos]);
                    }
                }
                updateChipStyles();
                displayCount = 50;
                renderTable();
            });
            container.appendChild(chip);
        });

        const allChip = document.createElement("span");
        allChip.className = "pos-chip pos-all";
        allChip.textContent = "All";
        allChip.addEventListener("click", () => {
            posFilters = new Set(POSITIONS);
            updateChipStyles();
            displayCount = 50;
            renderTable();
        });
        container.appendChild(allChip);
    }

    // --- Search ---

    $("#adj-search").addEventListener("input", (e) => {
        searchQuery = e.target.value;
        displayCount = 50;
        renderTable();
    });

    // --- Changed only toggle ---

    $("#adj-changed-only").addEventListener("change", (e) => {
        changedOnly = e.target.checked;
        displayCount = 50;
        renderTable();
    });

    // --- Show more ---

    $("#adj-show-more").addEventListener("click", () => {
        displayCount += 50;
        renderTable();
    });

    // --- Save / Reset ---

    $("#save-btn").addEventListener("click", async () => {
        const btn = $("#save-btn");
        btn.disabled = true;
        btn.textContent = "Saving...";

        try {
            // Strip all-zero entries
            const cleaned = {};
            for (const [name, adj] of Object.entries(adjustments)) {
                if (adj.adjustment_ppg || adj.adjustment_games || adj.adjustment_volatility || adj.note) {
                    cleaned[name] = adj;
                }
            }

            const resp = await fetch("/api/nfl-adjustments", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ adjustments: cleaned }),
            });

            if (!resp.ok) throw new Error("Save failed");
            const data = await resp.json();
            adjustments = cleaned;
            setDirty(false);
            btn.textContent = `Saved (${data.count})`;
            setTimeout(() => { btn.textContent = "Save"; }, 2000);
        } catch (err) {
            console.error("Save failed:", err);
            btn.textContent = "Save failed!";
            btn.style.background = "#ff5252";
            setTimeout(() => {
                btn.textContent = "Save";
                btn.style.background = "";
            }, 2000);
        } finally {
            btn.disabled = false;
        }
    });

    $("#reset-btn").addEventListener("click", () => {
        if (!confirm("Reset all adjustments? This will clear all values (you still need to Save).")) return;
        adjustments = {};
        setDirty(true);
        renderTable();
    });

    // --- Unsaved changes warning ---

    window.addEventListener("beforeunload", (e) => {
        if (dirty) {
            e.preventDefault();
            e.returnValue = "";
        }
    });

    // --- Load data ---

    async function loadData() {
        try {
            const resp = await fetch("/api/nfl-adjustments");
            if (!resp.ok) throw new Error("Failed to load data");
            const data = await resp.json();

            players = data.players;
            // Compute derived fields
            players.forEach((p) => {
                p.upside = (p.total_ceiling || 0) - (p.total_floor || 0);
            });

            adjustments = data.adjustments || {};
            setDirty(false);
            renderTable();
            updateAdjCount();
        } catch (err) {
            console.error("Failed to load data:", err);
            $("#adj-body").innerHTML = `<tr><td colspan="${COLUMNS.length}" class="empty-msg">Failed to load player data</td></tr>`;
        }
    }

    // --- Init ---

    initPosFilter();
    loadData();
})();
