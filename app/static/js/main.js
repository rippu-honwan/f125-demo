(() => {
  "use strict";

  // ============================================================
  // API base configuration
  // ============================================================
  // The backend (FastAPI: /laps, /analyze) may live on a DIFFERENT origin than
  // this page. When the backend serves this file itself (local dev, or any
  // single-host deploy) the API is same-origin, so the base is "". When the
  // static frontend is hosted separately (e.g. GitHub Pages), point it at your
  // deployed backend by setting PROD_API_BASE below — that is the one line you
  // edit for a hosted deployment.
  //
  //   PROD_API_BASE = "https://your-backend.onrender.com";
  //
  // You can also override at runtime without editing the file:
  //   • add ?api=https://your-backend.onrender.com to the URL (it is remembered
  //     in localStorage for next time), or
  //   • run  localStorage.setItem("apiBase", "https://...")  in the console.
  const PROD_API_BASE = "https://f125-demo-api.onrender.com";   // hosted backend (used when served from GitHub Pages)

  function resolveApiBase() {
    const strip = (u) => String(u || "").trim().replace(/\/+$/, "");
    // 1) explicit ?api= override (persisted), then a stored override.
    try {
      const q = new URLSearchParams(location.search).get("api");
      if (q !== null) { const v = strip(q); localStorage.setItem("apiBase", v); return v; }
      const stored = strip(localStorage.getItem("apiBase"));
      if (stored) return stored;
    } catch (e) { /* localStorage may be unavailable; fall through */ }
    // 2) on a remote host (e.g. GitHub Pages) use the configured prod backend.
    const host = location.hostname;
    const isLocal = !host || host === "localhost" || host === "127.0.0.1" || host === "0.0.0.0";
    if (!isLocal && PROD_API_BASE) return strip(PROD_API_BASE);
    // 3) same-origin (local dev: the backend serves this page).
    return "";
  }

  const API_BASE = resolveApiBase();
  const apiUrl = (path) => API_BASE + path;
  // True when the page is hosted remotely but no backend URL is configured yet
  // — used to give a clearer error than a bare "network error".
  const API_UNCONFIGURED = !API_BASE &&
    !(["localhost", "127.0.0.1", "0.0.0.0", ""].includes(location.hostname));

  // A failed fetch (backend down, asleep, CORS-blocked, or unconfigured) surfaces
  // as a TypeError; turn that into a message that tells the user what to do.
  function isNetworkError(err) {
    return !!err && (err.name === "TypeError" ||
      /failed to fetch|networkerror|load failed/i.test(err.message || ""));
  }
  function apiErrorMessage(err, fallback) {
    if (isNetworkError(err)) {
      if (API_UNCONFIGURED) {
        return "Can’t reach the analysis backend. This page is hosted as a static site " +
          "(e.g. GitHub Pages), so it needs a separately deployed backend. Set " +
          "PROD_API_BASE in index.html (or append ?api=<your-backend-url> to this URL). " +
          "See DEPLOYMENT.md.";
      }
      return "Couldn’t reach the backend at " + (API_BASE || location.origin) +
        ". Is it running and awake? Free hosts can take ~30–60s to wake on the first request.";
    }
    return (err && err.message) || fallback || "Unknown error.";
  }

  // -------- Element refs --------
  const $ = (id) => document.getElementById(id);
  const dropzone   = $("dropzone");
  const fileInput  = $("fileInput");
  const fileChip   = $("fileChip");
  const fcName     = $("fcName");
  const fcSize     = $("fcSize");
  const fcClear    = $("fcClear");
  const analyzeBtn = $("analyzeBtn");
  const loadingState = $("loadingState");
  const ltStatus   = $("ltStatus");
  const errorBanner = $("errorBanner");
  const errorMsg   = $("errorMsg");
  const results    = $("results");

  // Lap selector refs
  const lapBlock   = $("lapBlock");
  const lapLoading = $("lapLoading");
  const lapTrigger = $("lapTrigger");
  const lapValue   = $("lapValue");
  const lapCombo   = $("lapCombo");
  const lapPanel   = $("lapPanel");
  const lapSearch  = $("lapSearch");
  const lapOptions = $("lapOptions");
  const lapEmpty   = $("lapEmpty");
  const lapStatus  = $("lapStatus");
  const lapNote    = $("lapNote");

  // Analysis-mode refs
  const modeGrid    = $("modeGrid");
  const modeDescText = $("modeDescText");
  const refOpt      = $("refOpt");
  const refHintText = $("refHintText");
  const configEl    = $("config");
  const analyzeLabel = $("analyzeLabel");

  // Track auto-detect refs
  const trackSelect    = $("track");
  const trackAuto      = $("trackAuto");
  const trackAutoName  = $("trackAutoName");
  const trackChange    = $("trackChange");
  const trackDetectMsg = $("trackDetectMsg");
  const trackDetectMsgText = $("trackDetectMsgText");

  let selectedFile = null;
  let statusTimer  = null;
  let kcData = [];          // per-corner trace data, for mini-chart hover tooltips
  let kcDriver = "Pro";     // reference driver label shown in those tooltips
  let lapChartData = { you: [], pro: [], delta: [], markers: [] };  // whole-lap hover

  // -------- Analysis modes --------
  // Each mode maps to one of the five scripts/ programs and declares which
  // result blocks it shows (in order) and whether it needs the reference
  // driver/year/session (the "vs Pro" comparison family).
  const MODES = {
    overview: {
      label: "Run Telemetry Overview",
      tag: "Single lap",
      needsReference: false,
      desc: "Telemetry Overview — a quick read-out of your uploaded lap: lap time, top speed, throttle/brake usage and a table of every lap in the file. No reference driver needed.",
      blocks: ["blkOverview"],
    },
    lap_analysis: {
      label: "Run Lap Analysis",
      tag: "Single lap",
      needsReference: false,
      desc: "Lap Analysis — corner-by-corner breakdown of your own lap: minimum, entry and exit speed, gear and time spent in each corner. Needs a track but no reference driver.",
      blocks: ["blkSolo"],
    },
    comparison: {
      label: "Run Lap Comparison",
      tag: "vs Pro",
      needsReference: true,
      desc: "Lap Comparison — a head-to-head against a real F1 driver: overall delta and a where-you-stand summary up top, key-corner cards with brake / throttle / gear comparison charts in the middle, and a full whole-lap pace & cumulative-delta chart at the bottom. No heatmap — that's Track Map.",
      blocks: ["blkTiming", "blkCompStrip", "blkKeyCorners", "blkLapChart"],
    },
    coaching: {
      label: "Run Coaching Report",
      tag: "vs Pro",
      needsReference: true,
      desc: "Coaching Report — actionable coaching: overall grade and consistency, braking & throttle tendencies, a step-by-step action plan and your highest-priority corner fixes.",
      blocks: ["blkTiming", "blkSummary", "blkActionPlan"],
    },
    track_map: {
      label: "Run Track Explorer",
      tag: "vs Pro",
      needsReference: true,
      desc: "Interactive Track Map — an SVG of the circuit you can hover point by point. The marker is the controller: it drives the telemetry panel below, reading out your speed, throttle, brake and gear at the hovered position, with every chart cursor moving together.",
      blocks: ["blkExplorer"],
    },
  };
  const ALL_BLOCKS = [
    "blkOverview", "blkSolo", "blkTiming", "blkLapChart", "blkCompStrip",
    "blkMap", "blkExplorer", "blkCorners", "blkKeyCorners", "blkSummary", "blkActionPlan", "blkSeverity",
  ];
  let currentMode = "overview";

  function applyModeUI(mode) {
    currentMode = mode;
    const meta = MODES[mode];

    // Segmented control active state
    modeGrid.querySelectorAll(".mode-btn").forEach((b) => {
      const on = b.dataset.mode === mode;
      b.classList.toggle("active", on);
      b.setAttribute("aria-selected", on ? "true" : "false");
    });

    // Description + analyze button label
    modeDescText.textContent = meta.desc;
    analyzeLabel.textContent = meta.label;

    // Reference fields: only the comparison family uses driver/year/session.
    configEl.classList.toggle("reference-muted", !meta.needsReference);
    if (meta.needsReference) {
      refOpt.textContent = "Used for comparison";
      refHintText.textContent =
        "The selected driver, year and session set the real F1 lap you are compared against.";
    } else {
      refOpt.textContent = "Track only";
      refHintText.textContent =
        "This mode analyses your lap on its own — only the track matters. Driver, year and session are ignored.";
    }
  }

  modeGrid.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      applyModeUI(btn.dataset.mode);
      results.classList.remove("show");   // a new mode means a fresh run
      hideError();
    });
  });

  // Lap selector state
  const AUTO_LABEL = "Auto / Fastest Lap";
  let lapData = [];             // [{lap_index, lap_number, lap_time, max_speed}]
  let selectedLapIndex = null;  // null = Auto / Fastest Lap
  let lapFetchToken = 0;        // guards against out-of-order /laps responses
  let comboActiveIdx = -1;      // keyboard highlight within visible rows

  // -------- File handling --------
  function setFile(file) {
    if (!file) return;
    selectedFile = file;
    fcName.textContent = file.name;
    fcSize.textContent = formatBytes(file.size);
    fileChip.classList.add("show");
    analyzeBtn.disabled = false;
    hideError();
    loadLaps(file);
  }
  function clearFile() {
    selectedFile = null;
    fileInput.value = "";
    fileChip.classList.remove("show");
    analyzeBtn.disabled = true;
    resetLapSelector();
    resetTrackDetection();
  }
  function formatBytes(b) {
    if (!b && b !== 0) return "";
    if (b < 1024) return b + " B";
    if (b < 1048576) return (b / 1024).toFixed(1) + " KB";
    return (b / 1048576).toFixed(1) + " MB";
  }

  // -------- Track auto-detection (Task 4) --------
  // Default path: the track is read from the CSV's trackId. Manual selection
  // only appears when detection fails or the user explicitly overrides it.
  function trackOptionExists(key) {
    return Array.from(trackSelect.options).some((o) => o.value === key);
  }
  function resetTrackDetection() {
    trackAuto.hidden = false;
    trackAuto.style.opacity = ".6";
    trackAutoName.textContent = "Detected on upload";
    trackChange.hidden = true;
    trackSelect.hidden = true;
    trackDetectMsg.hidden = true;
  }
  function showManualTrack(message) {
    trackAuto.hidden = true;
    trackChange.hidden = true;
    trackSelect.hidden = false;
    if (message) {
      trackDetectMsgText.textContent = message;
      trackDetectMsg.hidden = false;
    } else {
      trackDetectMsg.hidden = true;
    }
  }
  function applyDetectedTrack(key, name) {
    if (key && trackOptionExists(key)) {
      trackSelect.value = key;             // keep the form's source of truth in sync
      trackAuto.hidden = false;
      trackAuto.style.opacity = "1";
      trackAutoName.textContent = name || key;
      trackChange.hidden = false;
      trackSelect.hidden = true;
      trackDetectMsg.hidden = true;
    } else {
      showManualTrack("Couldn’t detect a track from this CSV — please choose one manually.");
    }
  }
  trackChange.addEventListener("click", () => {
    trackAuto.hidden = true;
    trackChange.hidden = true;
    trackSelect.hidden = false;
    trackDetectMsgText.textContent = "Manual override — auto-detection is off for this file.";
    trackDetectMsg.hidden = false;
  });

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) setFile(e.target.files[0]);
  });
  fcClear.addEventListener("click", (e) => { e.stopPropagation(); clearFile(); });

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("drag"); })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); })
  );
  dropzone.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) setFile(f);
  });

  // -------- Lap selector --------
  function resetLapSelector() {
    lapFetchToken++;            // cancel any in-flight /laps request
    lapData = [];
    selectedLapIndex = null;
    lapBlock.classList.remove("show");
    closeLapPanel();
    lapNote.hidden = true;
    lapLoading.hidden = true;
  }

  function lapLabel(li) {
    if (li == null) return AUTO_LABEL;
    const lap = lapData.find((l) => l.lap_index === li);
    if (!lap) return "Lap " + (li + 1);
    return "Lap " + lap.lap_number + (lap.lap_time != null ? " · " + fmtLap(lap.lap_time) : "");
  }

  function setSelectedLap(li) {
    selectedLapIndex = li;
    lapValue.textContent = lapLabel(li);
    lapStatus.innerHTML = "Analyzing <b></b>";
    lapStatus.querySelector("b").textContent = lapLabel(li);
    renderLapOptions(lapSearch.value);
  }

  function buildRows() {
    const rows = [{
      value: null, main: AUTO_LABEL, tag: "Default", meta: "",
      search: "auto fastest default", fastest: false,
    }];
    // Fastest = lowest recorded lap time (gets golden-yellow styling).
    let fastIdx = null, fastTime = Infinity;
    lapData.forEach((l) => {
      if (l.lap_time != null && l.lap_time < fastTime) {
        fastTime = l.lap_time; fastIdx = l.lap_index;
      }
    });
    lapData.forEach((l) => {
      const time = (l.lap_time != null) ? fmtLap(l.lap_time) : "";
      rows.push({
        value: l.lap_index,
        main: "Lap " + l.lap_number,
        tag: "",
        meta: time,                          // lap time only, on the right
        search: ("lap " + l.lap_number + " " + l.lap_index + " " + time).toLowerCase(),
        fastest: (l.lap_index === fastIdx),
      });
    });
    return rows;
  }

  function renderLapOptions(query) {
    const q = (query || "").trim().toLowerCase();
    // Auto is always offered; lap rows are filtered by the search query.
    const rows = buildRows().filter((r) => r.value == null || !q || r.search.includes(q));
    lapOptions.innerHTML = "";
    comboActiveIdx = -1;

    rows.forEach((r, i) => {
      const el = document.createElement("div");
      el.className = "combo-option"
        + (r.value === selectedLapIndex ? " selected" : "")
        + (r.fastest ? " fastest" : "");
      el.setAttribute("role", "option");
      el.setAttribute("aria-selected", r.value === selectedLapIndex ? "true" : "false");
      el.dataset.value = (r.value == null) ? "" : String(r.value);
      el.innerHTML =
        `<span class="co-main">${escapeHtml(r.main)}` +
        `${r.tag ? `<span class="co-tag">${escapeHtml(r.tag)}</span>` : ""}</span>` +
        `${r.meta ? `<span class="co-meta">${escapeHtml(r.meta)}</span>` : ""}`;
      el.addEventListener("click", () => {
        setSelectedLap(r.value);
        closeLapPanel();
        lapTrigger.focus();
      });
      el.addEventListener("mousemove", () => setActiveOption(i));
      lapOptions.appendChild(el);
    });

    const matchedLaps = rows.filter((r) => r.value != null).length;
    lapEmpty.hidden = !(q && matchedLaps === 0);
  }

  function setActiveOption(i) {
    const items = lapOptions.querySelectorAll(".combo-option");
    items.forEach((el, idx) => el.classList.toggle("active", idx === i));
    comboActiveIdx = i;
    if (items[i]) items[i].scrollIntoView({ block: "nearest" });
  }

  function openLapPanel() {
    if (lapTrigger.disabled || !lapPanel.hidden) return;
    lapPanel.hidden = false;
    lapTrigger.setAttribute("aria-expanded", "true");
    lapSearch.value = "";
    renderLapOptions("");
    setTimeout(() => lapSearch.focus(), 0);
    document.addEventListener("click", onDocClickCombo, true);
  }
  function closeLapPanel() {
    if (lapPanel.hidden) return;
    lapPanel.hidden = true;
    lapTrigger.setAttribute("aria-expanded", "false");
    document.removeEventListener("click", onDocClickCombo, true);
  }
  function onDocClickCombo(e) {
    if (!lapCombo.contains(e.target)) closeLapPanel();
  }

  lapTrigger.addEventListener("click", (e) => {
    e.stopPropagation();
    if (lapPanel.hidden) openLapPanel(); else closeLapPanel();
  });
  lapSearch.addEventListener("input", () => renderLapOptions(lapSearch.value));
  lapSearch.addEventListener("keydown", (e) => {
    const items = lapOptions.querySelectorAll(".combo-option");
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveOption(Math.min(comboActiveIdx + 1, items.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveOption(Math.max(comboActiveIdx - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      const cur = items[comboActiveIdx] || items[0];
      if (cur) {
        setSelectedLap(cur.dataset.value === "" ? null : Number(cur.dataset.value));
        closeLapPanel();
        lapTrigger.focus();
      }
    } else if (e.key === "Escape") {
      e.preventDefault();
      closeLapPanel();
      lapTrigger.focus();
    }
  });

  function showLapNote(safeHtml) {
    lapNote.innerHTML =
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" ' +
      'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
      '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/>' +
      '<line x1="12" y1="16" x2="12.01" y2="16"/></svg><span>' + safeHtml + '</span>';
    lapNote.hidden = false;
  }

  async function loadLaps(file) {
    const token = ++lapFetchToken;
    lapData = [];
    selectedLapIndex = null;
    lapBlock.classList.add("show");
    lapNote.hidden = true;
    lapLoading.hidden = false;
    lapTrigger.disabled = true;
    setSelectedLap(null);

    const fd = new FormData();
    fd.append("file", file);

    try {
      const res = await fetch(apiUrl("/laps"), { method: "POST", body: fd });
      if (token !== lapFetchToken) return;  // a newer upload superseded this one
      const ctype = res.headers.get("content-type") || "";
      const payload = ctype.includes("application/json") ? await res.json() : null;
      if (!res.ok) {
        const detail = (payload && payload.detail) || ("Could not read laps (" + res.status + ").");
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }

      // Auto-detect the track from the CSV's trackId (default path).
      applyDetectedTrack(payload.detected_track, payload.detected_track_name);

      lapData = Array.isArray(payload.laps) ? payload.laps : [];
      if (!payload.has_lap_index) {
        showLapNote("This CSV has no <b>lapIndex</b> column, so individual laps can’t be listed. The fastest lap will be analyzed automatically.");
        lapTrigger.disabled = true;
      } else if (lapData.length === 0) {
        showLapNote("No complete laps were found to list in this file. The fastest valid lap will be analyzed automatically.");
        lapTrigger.disabled = true;
      } else {
        lapTrigger.disabled = false;
      }
      setSelectedLap(null);
    } catch (err) {
      if (token !== lapFetchToken) return;
      const msg = isNetworkError(err)
        ? apiErrorMessage(err)
        : "Couldn’t read laps from this file: " + (err.message || "unknown error") +
          ". You can still run Auto / Fastest Lap.";
      showLapNote(escapeHtml(msg));
      lapTrigger.disabled = true;
      setSelectedLap(null);
    } finally {
      if (token === lapFetchToken) lapLoading.hidden = true;
    }
  }

  // -------- Loading status cycling --------
  const STATUS_COMPARISON = [
    "Loading your lap…",
    "Fetching real F1 telemetry…",
    "Aligning the two laps…",
    "Analyzing every corner…",
    "Building your coaching report…",
  ];
  const STATUS_SOLO = [
    "Loading your lap…",
    "Validating telemetry…",
    "Measuring each corner…",
    "Building your overview…",
  ];
  function startStatusCycle(mode) {
    const needsRef = MODES[mode] && MODES[mode].needsReference;
    const messages = needsRef ? STATUS_COMPARISON : STATUS_SOLO;
    $("ltTitle").textContent = needsRef
      ? "Running game-vs-real analysis" : "Running telemetry analysis";
    $("ltHint").textContent = needsRef
      ? "Fetching real F1 data can take a moment on the first run."
      : "Analysing your uploaded lap — no download required.";
    let i = 0;
    ltStatus.textContent = messages[0];
    statusTimer = setInterval(() => {
      i = Math.min(i + 1, messages.length - 1);
      ltStatus.textContent = messages[i];
    }, 2200);
  }
  function stopStatusCycle() { if (statusTimer) { clearInterval(statusTimer); statusTimer = null; } }

  // -------- Error handling --------
  function showError(msg) {
    errorMsg.textContent = msg || "Something went wrong.";
    errorBanner.classList.add("show");
    errorBanner.scrollIntoView({ behavior: "smooth", block: "center" });
  }
  function hideError() { errorBanner.classList.remove("show"); }

  // -------- Formatting helpers --------
  function fmtLap(s) {
    if (s == null || isNaN(s) || s <= 0) return "--";
    const m = Math.floor(s / 60);
    const sec = s - m * 60;
    return m + ":" + sec.toFixed(3).padStart(6, "0");
  }
  function fmtDelta(d) {
    if (d == null || isNaN(d)) return "--";
    return (d >= 0 ? "+" : "") + d.toFixed(3) + "s";
  }
  function fmtSigned(v, unit) {
    if (v == null || isNaN(v)) return null;
    return (v >= 0 ? "+" : "") + (Math.round(v * 10) / 10) + unit;
  }
  const SEV_VAR = { ok: "var(--ok)", minor: "var(--minor)", major: "var(--major)" };
  // Concrete hex equivalents of the severity tokens, for <canvas> drawing.
  const SEV_HEX = { ok: "#00cc66", minor: "#ffaa00", major: "#ff3355" };
  function gradeColor(g) {
    if (!g) return "var(--text-mute)";
    const L = g[0].toUpperCase();
    if (L === "A") return "var(--ok)";
    if (L === "B") return "var(--minor)";
    if (L === "C") return "#ff8800";
    if (L === "D") return "#ff5a2c";
    return "var(--major)"; // F
  }

  // -------- Render results --------
  // Show only the result blocks this mode uses, in the declared order.
  function applyResultLayout(mode) {
    const order = (MODES[mode] && MODES[mode].blocks) || [];
    ALL_BLOCKS.forEach((id) => { const el = $(id); if (el) el.style.display = "none"; });
    order.forEach((id) => {
      const el = $(id);
      if (!el) return;
      el.style.display = "";
      results.appendChild(el);   // physically reorder to match `order`
    });
  }

  function setFooter(data) {
    const bits = [];
    if (data.gp_name) bits.push(data.gp_name);
    if (data.year) bits.push(data.year);
    if (data.session) bits.push(data.session);
    if (data.consistency_score != null) bits.push("Consistency " + data.consistency_score + "/100");
    if (!bits.length) {                       // solo modes: no reference meta
      if (data.track_name) bits.push(data.track_name);
      if (data.lap_number != null) bits.push("Lap " + data.lap_number);
    }
    $("footMeta").textContent = bits.join(" · ");
  }

  // Mode dispatcher — each mode renders a genuinely distinct view.
  function renderResults(data) {
    const mode = data.mode || currentMode;
    applyResultLayout(mode);
    if (mode === "overview")          renderOverview(data);
    else if (mode === "lap_analysis") renderSolo(data);
    else if (mode === "comparison")   renderComparison(data);
    else if (mode === "coaching")     renderCoaching(data);
    else if (mode === "track_map")    renderTrackMap(data);
    setFooter(data);
    results.classList.add("show");
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // -------- Shared comparison-family pieces --------
  function renderTiming(data, label) {
    $("tYou").textContent = fmtLap(data.game_time);
    $("tReal").textContent = fmtLap(data.real_time);
    $("tRealLabel").textContent = data.driver || "Reference";
    const delta = data.overall_delta;
    $("tDelta").textContent = fmtDelta(delta);
    $("tDelta").parentElement.querySelector(".t-value").style.color =
      (delta != null && delta < 0) ? "var(--ok)" : "var(--accent-2)";
    const gradeEl = $("tGrade");
    gradeEl.textContent = data.overall_grade || "–";
    gradeEl.style.setProperty("--gc", gradeColor(data.overall_grade));
    $("timingLabel").textContent = label || "Lap Summary";
  }

  function renderMap(data) {
    clearHighlight();
    setCornerPositions(data.corner_positions || []);
    const mapImg = $("mapImg");
    if (data.track_map_base64) {
      mapImg.src = "data:image/png;base64," + data.track_map_base64;
      $("mapFrame").style.display = "";
    } else {
      $("mapFrame").style.display = "none";
    }
  }

  // -------- Lap Comparison (script 03): visual head-to-head, no heatmap --------
  // Hierarchy: TOP = overall delta + where-you-stand summary; MIDDLE = key-corner
  // cards with brake/throttle/gear mini charts; BOTTOM = the main whole-lap
  // pace & cumulative-delta comparison chart.
  function renderComparison(data) {
    clearHighlight();
    setCornerPositions([]);        // no map in this mode — clear any stale overlay
    renderTiming(data, "Lap Summary");

    // Top: where-you-stand summary strip.
    const worst = data.worst_corner, best = data.best_corner;
    const cells = [
      ["Corners Lost", (data.corners_lost != null ? String(data.corners_lost) : "—"), "lost"],
      ["Corners Gained", (data.corners_gained != null ? String(data.corners_gained) : "—"), "gain"],
      ["Biggest Loss", worst ? worst.short : "—", "lost", worst ? fmtDelta(worst.time_delta) : ""],
      ["Best Corner", best ? best.short : "—", "gain", best ? fmtDelta(best.time_delta) : ""],
    ];
    $("compStrip").innerHTML = cells.map((c) => {
      const sub = c[3] ? `<span class="u">${escapeHtml(c[3])}</span>` : "";
      return `<div class="comp-cell"><div class="cc-k">${escapeHtml(c[0])}</div>` +
             `<div class="cc-v ${c[2]}">${escapeHtml(String(c[1]))}${sub}</div></div>`;
    }).join("");

    // Middle: key-corner delta cards (brake / throttle / gear / speed charts).
    renderKeyCorners(data.key_corners || [], data.driver);

    // Bottom: the main whole-lap comparison chart.
    renderLapChart(data);
  }

  // -------- SVG comparison-chart builders (You vs Pro) --------
  function _svgPath(arr, X, Y, step) {
    let d = "", prevY = null, started = false;
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (v == null || isNaN(v)) continue;
      const x = X(i), y = Y(v);
      if (!started) { d = `M${x.toFixed(1)} ${y.toFixed(1)}`; started = true; }
      else {
        if (step && prevY != null) d += ` L${x.toFixed(1)} ${prevY.toFixed(1)}`;
        d += ` L${x.toFixed(1)} ${y.toFixed(1)}`;
      }
      prevY = y;
    }
    return d;
  }

  function cmpSvg(you, pro, opts) {
    opts = opts || {};
    const w = 100, h = opts.h || 40, pad = 3;
    const vals = [];
    (you || []).forEach((v) => { if (v != null && !isNaN(v)) vals.push(v); });
    (pro || []).forEach((v) => { if (v != null && !isNaN(v)) vals.push(v); });
    let lo = (opts.min != null) ? opts.min : (vals.length ? Math.min.apply(null, vals) : 0);
    let hi = (opts.max != null) ? opts.max : (vals.length ? Math.max.apply(null, vals) : 1);
    if (!isFinite(lo) || !isFinite(hi)) { lo = 0; hi = 1; }
    if (hi <= lo) hi = lo + 1;
    const n = Math.max((you || []).length, (pro || []).length, 2);
    const X = (i) => pad + (w - 2 * pad) * (i / (n - 1));
    const Y = (v) => h - pad - (h - 2 * pad) * ((v - lo) / (hi - lo));
    const proD = _svgPath(pro || [], X, Y, opts.step);
    const youD = _svgPath(you || [], X, Y, opts.step);
    let fills = "";
    if (opts.fill) {
      const base = (h - pad).toFixed(1), x0 = X(0).toFixed(1), xn = X(n - 1).toFixed(1);
      if (proD) fills += `<path d="${proD} L${xn} ${base} L${x0} ${base} Z" fill="var(--cmp-pro)" opacity=".10"/>`;
      if (youD) fills += `<path d="${youD} L${xn} ${base} L${x0} ${base} Z" fill="var(--cmp-you)" opacity=".13"/>`;
    }
    return `<svg class="${opts.cls || "cmp-svg"}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true">` +
      fills +
      `<path d="${proD}" fill="none" stroke="var(--cmp-pro)" stroke-width="1.4" vector-effect="non-scaling-stroke"/>` +
      `<path d="${youD}" fill="none" stroke="var(--cmp-you)" stroke-width="1.4" vector-effect="non-scaling-stroke"/></svg>`;
  }

  function cmpChart(label, you, pro, opts) {
    const type = String(label).toLowerCase();   // brake | throttle | gear | speed
    return `<div class="cmp-chart"><div class="cmp-h">${escapeHtml(label)}</div>` +
      `<div class="cmp-plot" data-ch="${type}">${cmpSvg(you, pro, opts)}` +
      `<span class="cmp-cursor" aria-hidden="true"></span></div></div>`;
  }

  // Dashed vertical guide lines at each corner apex, coloured by severity so the
  // problem corners stand out directly on the lap (shared by both lap plots).
  function lapGuides(markers, w, pad, h) {
    let g = "";
    (markers || []).forEach((m) => {
      const gx = (pad + (w - 2 * pad) * m.x).toFixed(2);
      const col = SEV_HEX[m.severity] || "#6f7494";
      const op = m.severity === "major" ? 0.6 : (m.severity === "minor" ? 0.5 : 0.3);
      g += `<line x1="${gx}" y1="${pad}" x2="${gx}" y2="${(h - pad).toFixed(1)}" ` +
           `stroke="${col}" stroke-width="0.6" stroke-dasharray="2.5 2.5" ` +
           `vector-effect="non-scaling-stroke" opacity="${op}"/>`;
    });
    return g;
  }

  // Whole-lap speed band: two traces + fill coloured by who is faster per segment.
  function lapSpeedBand(you, pro, h, markers) {
    you = you || []; pro = pro || [];
    const w = 100, pad = 4;
    const vals = [];
    you.concat(pro).forEach((v) => { if (v != null && !isNaN(v)) vals.push(v); });
    if (!vals.length) return "";
    let lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    if (hi <= lo) hi = lo + 1;
    const n = Math.max(you.length, pro.length, 2);
    const X = (i) => pad + (w - 2 * pad) * (i / (n - 1));
    const Y = (v) => h - pad - (h - 2 * pad) * ((v - lo) / (hi - lo));
    let band = "";
    for (let i = 0; i < n - 1; i++) {
      const y0 = you[i], y1 = you[i + 1], p0 = pro[i], p1 = pro[i + 1];
      if ([y0, y1, p0, p1].some((v) => v == null || isNaN(v))) continue;
      const col = ((y0 + y1) >= (p0 + p1)) ? "var(--cmp-you)" : "var(--cmp-pro)";
      band += `<path d="M${X(i).toFixed(1)} ${Y(y0).toFixed(1)} L${X(i + 1).toFixed(1)} ${Y(y1).toFixed(1)} ` +
              `L${X(i + 1).toFixed(1)} ${Y(p1).toFixed(1)} L${X(i).toFixed(1)} ${Y(p0).toFixed(1)} Z" fill="${col}" opacity=".16"/>`;
    }
    return `<svg class="lc-svg lc-speed" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true">` +
      lapGuides(markers, w, pad, h) + band +
      `<path d="${_svgPath(pro, X, Y, false)}" fill="none" stroke="var(--cmp-pro)" stroke-width="1.5" vector-effect="non-scaling-stroke"/>` +
      `<path d="${_svgPath(you, X, Y, false)}" fill="none" stroke="var(--cmp-you)" stroke-width="1.5" vector-effect="non-scaling-stroke"/></svg>`;
  }

  // Cumulative delta: segments coloured by local gain/loss (red losing, green gaining).
  function lapDeltaColored(delta, h, markers) {
    const arr = delta || [];
    const vals = arr.filter((v) => v != null && !isNaN(v));
    if (!vals.length) return "";
    const w = 100, pad = 4;
    let lo = Math.min(0, Math.min.apply(null, vals));
    let hi = Math.max(0, Math.max.apply(null, vals));
    if (hi <= lo) hi = lo + 0.05;
    const n = Math.max(arr.length, 2);
    const X = (i) => pad + (w - 2 * pad) * (i / (n - 1));
    const Y = (v) => h - pad - (h - 2 * pad) * ((v - lo) / (hi - lo));
    const zeroY = Y(0).toFixed(1);
    const line = _svgPath(arr, X, Y, false);
    const area = `${line} L${X(n - 1).toFixed(1)} ${zeroY} L${X(0).toFixed(1)} ${zeroY} Z`;
    let segs = "";
    for (let i = 0; i < n - 1; i++) {
      const a = arr[i], b = arr[i + 1];
      if (a == null || b == null || isNaN(a) || isNaN(b)) continue;
      const col = (b > a) ? "var(--cmp-pro)" : "var(--cmp-you)"; // rising delta = losing time
      segs += `<line x1="${X(i).toFixed(1)}" y1="${Y(a).toFixed(1)}" x2="${X(i + 1).toFixed(1)}" y2="${Y(b).toFixed(1)}" ` +
              `stroke="${col}" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
    }
    return `<svg class="lc-svg lc-delta" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true">` +
      `<path d="${area}" fill="var(--cmp-pro)" opacity=".08"/>` +
      lapGuides(markers, w, pad, h) +
      `<line x1="${pad}" y1="${zeroY}" x2="${w - pad}" y2="${zeroY}" stroke="var(--border-strong)" stroke-width="0.6"/>` +
      segs + `</svg>`;
  }

  // Bottom section: the main whole-lap comparison visualisation.
  function renderLapChart(data) {
    const lc = data.lap_chart || {};
    const markers = data.corner_markers || [];
    lapChartData = { you: lc.you || [], pro: lc.pro || [], delta: lc.delta || [], markers };
    const driver = escapeHtml(data.driver || "Reference");
    const finalDelta = (lc.delta && lc.delta.length)
      ? lc.delta[lc.delta.length - 1] : data.overall_delta;
    const legend =
      `<div class="lc-legend">` +
      `<span class="lg"><i class="sw you"></i>You</span>` +
      `<span class="lg"><i class="sw pro"></i>${driver}</span>` +
      `<span class="lg lg-sep"><i class="sw pro"></i>Losing time</span>` +
      `<span class="lg"><i class="sw you"></i>Gaining time</span></div>`;
    // Corner labels, coloured by severity and aligned to the dashed guide lines.
    const markerRow = markers.length
      ? `<div class="lc-markers">` + markers.map((m) => {
          const left = (4 + 92 * m.x).toFixed(2);
          const short = escapeHtml(m.short || "");
          return `<span class="lc-mk sev-${m.severity || "ok"}" style="left:${left}%" title="${short}">` +
            `<span class="mk-full">${short}</span>` +
            `<span class="mk-min">${m.corner_id}</span></span>`;
        }).join("") + `</div>`
      : "";
    const speedRow =
      `<div class="lc-row"><div class="lc-k">Speed<span class="lc-unit">km/h</span></div>` +
      `<div class="lc-plot">${lapSpeedBand(lc.you, lc.pro, 110, markers)}` +
      `<span class="lc-cursor" aria-hidden="true"></span></div></div>`;
    const deltaRow = (lc.delta && lc.delta.length)
      ? `<div class="lc-row"><div class="lc-k">Delta<span class="lc-unit">cumulative</span></div>` +
        `<div class="lc-plot">${lapDeltaColored(lc.delta, 92, markers)}` +
        `<span class="lc-cursor" aria-hidden="true"></span></div></div>`
      : "";
    const axis = `<div class="lc-axis"><span>Start</span><span>Lap distance →</span><span>Finish</span></div>`;
    const cap = (finalDelta != null)
      ? `<div class="lc-cap">Cumulative delta at the line: ` +
        `<b class="${finalDelta > 0 ? "slow" : "fast"}">${fmtDelta(finalDelta)}</b></div>`
      : "";
    $("lapChart").innerHTML = legend + markerRow + speedRow + deltaRow + axis + cap;
    // Hide any corner labels that would overlap once the real width is known.
    requestAnimationFrame(declutterLapMarkers);
  }

  // Greedily hide corner labels that would collide, keeping the chart readable
  // at any width. The dashed guide lines for every corner always remain.
  function declutterLapMarkers() {
    const wrap = document.querySelector("#lapChart .lc-markers");
    if (!wrap || wrap.offsetParent === null) return;   // not rendered / hidden
    const marks = Array.from(wrap.querySelectorAll(".lc-mk"));
    marks.forEach((m) => m.classList.remove("hide"));
    let lastRight = -Infinity;
    const GAP = 3;
    marks.forEach((m) => {
      const r = m.getBoundingClientRect();
      if (r.left < lastRight + GAP) m.classList.add("hide");
      else lastRight = r.right;
    });
  }

  function renderKeyCorners(list, driver) {
    const grid = $("keyCorners");
    kcData = list || [];
    kcDriver = driver || "Pro";
    if (!list.length) {
      grid.innerHTML = `<div class="empty-corners" style="grid-column:1/-1">No corner trace data available for this lap.</div>`;
      $("keyCornersLabel").textContent = "All Corners — Brake · Throttle · Gear · Speed";
      return;
    }
    $("keyCornersLabel").textContent =
      `All ${list.length} Corners — Brake · Throttle · Gear · Speed`;
    const drv = escapeHtml(driver || "Pro");
    const cards = list.map((c, idx) => {
      const sev = (c.severity in SEV_VAR) ? c.severity : "ok";
      const fast = c.time_delta != null && c.time_delta < 0;
      const ch = c.charts || {};
      const charts = [
        cmpChart("Brake", ch.brake && ch.brake.you, ch.brake && ch.brake.pro, { min: 0, max: 1, fill: true }),
        cmpChart("Throttle", ch.throttle && ch.throttle.you, ch.throttle && ch.throttle.pro, { min: 0, max: 1, fill: true }),
        cmpChart("Gear", ch.gear && ch.gear.you, ch.gear && ch.gear.pro, { step: true }),
        cmpChart("Speed", ch.speed && ch.speed.you, ch.speed && ch.speed.pro, {}),
      ].join("");
      // Glanceable numeric cues so the visuals (not prose) carry the meaning.
      const stats = [];
      const bd = fmtSigned(c.brake_diff_m, "m");
      const ad = fmtSigned(c.apex_speed_diff, " km/h");
      const xd = fmtSigned(c.exit_speed_diff, " km/h");
      if (bd) stats.push(`<span class="stat">Brake <b>${bd}</b></span>`);
      if (ad) stats.push(`<span class="stat">Apex <b>${ad}</b></span>`);
      if (xd) stats.push(`<span class="stat">Exit <b>${xd}</b></span>`);
      const statRow = stats.length ? `<div class="kc-stats">${stats.join("")}</div>` : "";
      const tip = c.tip ? `<span class="kc-tip">Fix: ${escapeHtml(c.tip)}</span>` : "";
      // Tiny speed sparkline in the (always-visible) header for at-a-glance pace.
      const spark = cmpSvg(ch.speed && ch.speed.you, ch.speed && ch.speed.pro, { h: 26, cls: "kc-spark-svg" });

      return `<div class="kc-card" style="--sev:${SEV_VAR[sev]}">` +
        `<button type="button" class="kc-toggle" aria-expanded="false">` +
          `<div class="kc-head"><div class="kc-id-wrap"><span class="kc-id">${escapeHtml(c.short || "")}</span>` +
          `<span class="kc-name">${escapeHtml(c.name || "")}</span></div>` +
          `<div class="kc-headright"><span class="kc-delta ${fast ? "fast" : "slow"}">${fmtDelta(c.time_delta)}</span>` +
          `<span class="kc-badge">${escapeHtml(c.grade || "–")}</span>` +
          `<span class="kc-chev" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg></span></div></div>` +
          `<div class="kc-spark">${spark}</div>` +
        `</button>` +
        `<div class="kc-detail"><div class="kc-detail-inner">` +
          `<div class="kc-legend"><span class="lg"><i class="sw you"></i>You</span>` +
          `<span class="lg"><i class="sw pro"></i>${drv}</span></div>` +
          statRow +
          `<div class="kc-charts" data-kc="${idx}">${charts}</div>` +
          `<div class="kc-explain"><span class="kc-why">${escapeHtml(c.explanation || "")}</span>${tip}</div>` +
        `</div></div>` +
        `</div>`;
    });
    // Split the cards across two independent columns. Splitting by half (rather
    // than interleaving) keeps the natural corner order when they stack on
    // mobile (column A then column B = corners in sequence).
    const half = Math.ceil(cards.length / 2);
    grid.innerHTML =
      `<div class="kc-col">${cards.slice(0, half).join("")}</div>` +
      `<div class="kc-col">${cards.slice(half).join("")}</div>`;
  }

  // -------- Coaching Report (script 04): actionable plan + fixes --------
  function renderCoaching(data) {
    renderTiming(data, "Overall Grade & Timing");

    $("brakingSummary").textContent = data.braking_summary || "Not enough corner data to assess braking tendency.";
    $("throttleSummary").textContent = data.throttle_summary || "Not enough corner data to assess throttle tendency.";

    // Headline banner: consistency, potential gain, target time.
    const banner = [
      ["Consistency", (data.consistency_score != null ? data.consistency_score + " /100" : "—"), false],
      ["Potential Gain", (data.potential_gain != null ? "-" + data.potential_gain + "s" : "—"), true],
      ["Target Time", fmtLap(data.target_time), false],
    ];
    $("apBanner").innerHTML = banner.map((b) =>
      `<div class="apb-item"><span class="apb-k">${escapeHtml(b[0])}</span>` +
      `<span class="apb-v ${b[2] ? "accent" : ""}">${escapeHtml(String(b[1]))}</span></div>`
    ).join("");

    // Step-by-step action plan.
    const plan = data.action_plan || [];
    $("apList").innerHTML = plan.length
      ? plan.map((step, i) => {
          let label = "Tip " + (i + 1), text = step;
          const idx = step.indexOf(":");
          if (idx > 0 && idx < 24) { label = step.slice(0, idx); text = step.slice(idx + 1); }
          return `<li class="ap-step"><span class="aps-n">${i + 1}</span>` +
                 `<div><div class="aps-label">${escapeHtml(label.trim())}</div>` +
                 `<div class="aps-text">${escapeHtml(text.trim())}</div></div></li>`;
        }).join("")
      : `<li class="ap-step"><div class="aps-text">No specific actions — your lap is already well matched to the reference.</div></li>`;

    // Priority corner fixes.
    const fixes = data.priority_fixes || [];
    if (!fixes.length) {
      $("fixGrid").innerHTML =
        `<div class="empty-corners" style="grid-column:1/-1">No high-priority corners — nicely balanced lap.</div>`;
    } else {
      $("fixGrid").innerHTML = fixes.map((f) => {
        const sev = (f.severity in SEV_VAR) ? f.severity : "ok";
        const issue = f.issue ? `<div class="fx-row issue"><span class="fx-k">Issue</span><span class="fx-t">${escapeHtml(f.issue)}</span></div>` : "";
        const tip = f.tip ? `<div class="fx-row tip"><span class="fx-k">Fix</span><span class="fx-t">${escapeHtml(f.tip)}</span></div>` : "";
        return `<div class="fix-card" style="--sev:${SEV_VAR[sev]}">` +
          `<div class="fx-head"><div><div class="fx-id">${escapeHtml(f.short || "")}</div>` +
          `<div class="fx-delta">${fmtDelta(f.time_delta)}</div></div>` +
          `<div class="fx-badge">${escapeHtml(f.grade || "–")}</div></div>` +
          issue + tip + `</div>`;
      }).join("");
    }
  }

  // -------- Track Map (script 05): the Interactive Track Explorer ----------
  // The SVG circuit is the controller. A single hovered index drives the moving
  // marker on the map AND every cursor / value in the telemetry panel below.
  // The old static-PNG renderer (renderMap + the <canvas> overlay) stays in the
  // file as a dormant fallback but is no longer wired into this mode.
  const CORNER_NEAR_M = 35;          // hover this close (m) to an apex -> label it
  let txState = {
    points: [], telemetry: {}, markers: [], n: 0,
    cursorEls: [], idx: 0, nearCid: undefined,
  };
  let txRaf = 0, txPendingEvt = null;

  function renderTrackMap(data) {
    txRender(data.track_explorer);
  }

  function txAt(arr, i) {
    const v = arr && arr[i];
    return (v == null || isNaN(v)) ? null : v;
  }

  // One trace path, normalised over index (x) and value (y) into a 0..100 × h
  // viewBox; the SVG itself stretches to the plot width (preserveAspectRatio
  // none) while vector-effect keeps the stroke crisp.
  function txTracePath(arr, n, cls, opts) {
    opts = opts || {};
    const h = opts.h || 92, w = 100, padX = 2, padY = 8;
    const vals = [];
    (arr || []).forEach((v) => { if (v != null && !isNaN(v)) vals.push(v); });
    let lo = (opts.min != null) ? opts.min : (vals.length ? Math.min.apply(null, vals) : 0);
    let hi = (opts.max != null) ? opts.max : (vals.length ? Math.max.apply(null, vals) : 1);
    if (!isFinite(lo) || !isFinite(hi)) { lo = 0; hi = 1; }
    if (hi <= lo) hi = lo + 1;
    const X = (i) => padX + (w - 2 * padX) * (n > 1 ? i / (n - 1) : 0);
    const Y = (v) => (h - padY) - (h - 2 * padY) * ((v - lo) / (hi - lo));
    return `<path class="${cls}" d="${_svgPath(arr, X, Y, opts.step)}"/>`;
  }

  function txChartRow(label, unit, inner, h) {
    const u = unit ? `<span class="tx-unit">${escapeHtml(unit)}</span>` : "";
    return `<div class="tx-row"><div class="tx-rk">${escapeHtml(label)}${u}</div>` +
      `<div class="tx-plot">` +
      `<svg class="tx-trace" viewBox="0 0 100 ${h}" preserveAspectRatio="none" aria-hidden="true">${inner}</svg>` +
      `<span class="tx-cursor" aria-hidden="true"></span></div></div>`;
  }

  function txBuildCharts(t) {
    const n = (t.speed && t.speed.length) || (t.dist && t.dist.length) || 0;
    const h = 92;
    const legend =
      `<div class="tx-legend">` +
      `<span class="lg"><i class="sw spd"></i>Speed</span>` +
      `<span class="lg"><i class="sw thr"></i>Throttle</span>` +
      `<span class="lg"><i class="sw brk"></i>Brake</span>` +
      `<span class="lg"><i class="sw gr"></i>Gear</span></div>`;
    const speed = txChartRow("Speed", "km/h",
      txTracePath(t.speed, n, "tx-line-spd", { h }), h);
    const tb = txChartRow("Thr / Brk", "0–100%",
      txTracePath(t.throttle, n, "tx-line-thr", { h, min: 0, max: 1 }) +
      txTracePath(t.brake, n, "tx-line-brk", { h, min: 0, max: 1 }), h);
    let gMax = 8;
    (t.gear || []).forEach((v) => { if (v != null && v > gMax) gMax = v; });
    const gear = txChartRow("Gear", "",
      txTracePath(t.gear, n, "tx-line-gr", { h, min: 0, max: gMax, step: true }), h);
    return legend + speed + tb + gear;
  }

  function txBuildSvg(ex) {
    const svg = $("txSvg");
    const tp = ex.track_path;
    const pts = tp.points;
    svg.setAttribute("viewBox", `0 0 ${tp.viewbox_w} ${tp.viewbox_h}`);
    let d = "";
    for (let i = 0; i < pts.length; i++) d += (i ? "L" : "M") + pts[i][0] + " " + pts[i][1];
    if (tp.closed) d += "Z";                    // close only a genuine lap loop
    const span = Math.min(tp.viewbox_w, tp.viewbox_h);
    const r = Math.max(6, span * 0.013);
    let corners = "";
    (ex.corner_markers || []).forEach((m) => {
      const t = escapeHtml((m.short || "") + (m.name ? " — " + m.name : ""));
      corners += `<circle class="tx-corner-dot" data-cid="${m.corner_id}" cx="${m.x}" cy="${m.y}" r="${(r * 0.72).toFixed(1)}"><title>${t}</title></circle>`;
    });
    svg.innerHTML =
      `<path class="tx-track" d="${d}"/>` +
      `<path class="tx-track-core" d="${d}"/>` +
      `<g id="txCorners">${corners}</g>` +
      `<circle id="txMarkerHalo" class="tx-marker-halo" r="${(r * 2.6).toFixed(1)}" cx="${pts[0][0]}" cy="${pts[0][1]}"/>` +
      `<circle id="txMarker" class="tx-marker" r="${(r * 1.25).toFixed(1)}" cx="${pts[0][0]}" cy="${pts[0][1]}"/>`;
  }

  // Pointer/touch client coords -> SVG viewBox coords (robust to letterboxing).
  function txClientToVb(evt) {
    const svg = $("txSvg");
    if (!svg.createSVGPoint) return null;
    const pt = svg.createSVGPoint();
    pt.x = evt.clientX; pt.y = evt.clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return null;
    const p = pt.matrixTransform(ctm.inverse());
    return { x: p.x, y: p.y };
  }

  function txNearestIndex(vx, vy) {
    const pts = txState.points;
    let best = 0, bd = Infinity;
    for (let i = 0; i < pts.length; i++) {
      const dx = pts[i][0] - vx, dy = pts[i][1] - vy;
      const dd = dx * dx + dy * dy;
      if (dd < bd) { bd = dd; best = i; }
    }
    return best;
  }

  function txNearestCorner(dist) {
    if (dist == null) return null;
    let best = null, bd = Infinity;
    (txState.markers || []).forEach((m) => {
      if (m.dist == null) return;
      const d = Math.abs(m.dist - dist);
      if (d < bd) { bd = d; best = m; }
    });
    return (best && bd <= CORNER_NEAR_M) ? best : null;
  }

  function txHighlightDot(cid) {
    if (txState.nearCid === cid) return;
    txState.nearCid = cid;
    const g = $("txCorners");
    if (!g) return;
    g.querySelectorAll(".tx-corner-dot").forEach((d) => {
      d.classList.toggle("near", cid != null && +d.getAttribute("data-cid") === cid);
    });
  }

  // Shared cursor state: move the marker, the readout and every chart cursor to
  // the same sampled index. `idle` is the resting state shown on mouse-leave.
  function txSetIndex(i, idle) {
    const t = txState.telemetry, pts = txState.points, n = txState.n;
    if (!pts.length) return;
    i = Math.max(0, Math.min(i, pts.length - 1));
    txState.idx = i;

    const mk = $("txMarker"), hl = $("txMarkerHalo");
    if (mk) { mk.setAttribute("cx", pts[i][0]); mk.setAttribute("cy", pts[i][1]); }
    if (hl) { hl.setAttribute("cx", pts[i][0]); hl.setAttribute("cy", pts[i][1]); }

    const spd = txAt(t.speed, i), thr = txAt(t.throttle, i),
          brk = txAt(t.brake, i), gr = txAt(t.gear, i);
    $("txSpeed").innerHTML = (spd == null ? "—" : Math.round(spd)) + "<i>km/h</i>";
    const thrEl = $("txThrottle");
    thrEl.innerHTML = (thr == null ? "—" : Math.round(thr * 100)) + "<i>%</i>";
    thrEl.classList.toggle("thr-on", thr != null && thr > 0.02);
    const brkEl = $("txBrake");
    brkEl.innerHTML = (brk == null ? "—" : Math.round(brk * 100)) + "<i>%</i>";
    brkEl.classList.toggle("brake-on", brk != null && brk > 0.02);
    $("txGear").innerHTML = (gr == null ? "—" : String(gr));

    const leftPct = (n > 1) ? (2 + 96 * (i / (n - 1))) : 2;
    txState.cursorEls.forEach((c) => { c.style.left = leftPct + "%"; });

    const dist = txAt(t.dist, i);
    const near = txNearestCorner(dist);
    txHighlightDot(near ? near.corner_id : null);
    const posEl = $("txPos");
    if (idle) {
      posEl.innerHTML = `Start / Idle<span class="txp-sub">Lap start · move cursor to explore</span>`;
    } else if (near) {
      const sub = (dist == null ? "" : Math.round(dist) + " m · ") + "apex";
      posEl.innerHTML = `${escapeHtml(near.short || "")}` +
        `${near.name ? " · " + escapeHtml(near.name) : ""}` +
        `<span class="txp-sub">${sub}</span>`;
    } else {
      posEl.innerHTML = (dist == null ? "On track" : Math.round(dist) + " m") +
        `<span class="txp-sub">Lap distance</span>`;
    }
  }

  // rAF-throttled so rapid mousemove stays smooth and visually stable.
  function txOnMove(evt) {
    txPendingEvt = evt;
    if (txRaf) return;
    txRaf = requestAnimationFrame(() => {
      txRaf = 0;
      const e = txPendingEvt; txPendingEvt = null;
      if (!e || !txState.points.length) return;
      const c = txClientToVb(e);
      if (!c) return;
      txSetIndex(txNearestIndex(c.x, c.y), false);
    });
  }

  function txRender(ex) {
    const svg = $("txSvg"), charts = $("txCharts"), stage = $("txStage"),
          readout = $("txReadout"), hint = $("txHint"), frame = $("txFrame");
    frame.classList.remove("is-active");
    if (!ex || !ex.track_path || !(ex.track_path.points || []).length ||
        ex.track_path.points.length < 2) {
      // GPS unavailable for this lap — graceful, non-distorting fallback.
      svg.removeAttribute("viewBox");
      svg.innerHTML = "";
      stage.style.display = "none";
      readout.style.display = "none";
      charts.innerHTML = "";
      hint.innerHTML = `<div class="tx-empty">No GPS telemetry was available for this lap, so the interactive map can't be drawn. Try a lap exported with position data.</div>`;
      txState = { points: [], telemetry: {}, markers: [], n: 0, cursorEls: [], idx: 0, nearCid: undefined };
      return;
    }
    stage.style.display = "";
    readout.style.display = "";
    hint.textContent = "Move your cursor along the track to inspect your telemetry · the chart below follows the marker";
    txBuildSvg(ex);
    charts.innerHTML = txBuildCharts(ex.telemetry);
    const t = ex.telemetry;
    txState = {
      points: ex.track_path.points,
      telemetry: t,
      markers: ex.corner_markers || [],
      n: (t.speed && t.speed.length) || (t.dist && t.dist.length) || ex.track_path.points.length,
      cursorEls: Array.from(charts.querySelectorAll(".tx-cursor")),
      idx: 0,
      nearCid: undefined,
    };
    txSetIndex(0, true);          // resting state: marker parked at the lap start
  }

  // -------- Solo: Telemetry Overview (script 01) --------
  function renderOverview(data) {
    $("ovLabel").textContent = "Telemetry Overview — " + (data.track_name || "Your Lap");

    const cards = [
      ["Lap Time", fmtLap(data.lap_time), ""],
      ["Top Speed", numOr(data.top_speed), "km/h"],
      ["Avg Speed", numOr(data.avg_speed), "km/h"],
      ["Track Length", numOr(data.track_length), "m"],
      ["Full Throttle", numOr(data.full_throttle_pct), "%"],
      ["Braking", numOr(data.braking_pct), "%"],
      ["Top Gear", data.max_gear != null ? String(data.max_gear) : "—", ""],
      ["Laps Found", data.n_laps != null ? String(data.n_laps) : "—", ""],
    ];
    $("ovStats").innerHTML = cards.map((c) => statCard(c)).join("");

    const laps = data.laps || [];
    if (!laps.length) {
      $("ovTableWrap").innerHTML =
        `<div class="empty-corners" style="border:none;background:none">` +
        `This file has no per-lap index, so individual laps can't be tabulated. ` +
        `The vitals above describe the fastest lap.</div>`;
      return;
    }
    const nSec = (laps[0].sector_times || []).length;
    const headers = ["Lap", "Index", "Lap Time"];
    for (let i = 0; i < nSec; i++) headers.push("S" + (i + 1));
    headers.push("Max Speed");

    const rows = laps.map((l) => {
      const cells = [
        `<span class="cell-strong">${l.lap_number}</span>`,
        String(l.lap_index),
        `<span class="cell-strong">${fmtLap(l.lap_time)}</span>`,
      ];
      for (let i = 0; i < nSec; i++) cells.push(fmtSec(l.sector_times[i]));
      cells.push(numOr(l.max_speed) + " km/h");
      const best = (l.lap_index === data.lap_index);
      if (best) cells[0] += `<span class="pill">Selected</span>`;
      return { cells, cls: best ? "is-best" : "" };
    });
    $("ovTableWrap").innerHTML = buildTable(headers, rows);
  }

  // -------- Solo: Lap Analysis (script 02) --------
  function renderSolo(data) {
    $("soloLabel").textContent = "Lap Analysis — " + (data.track_name || "Your Lap");

    const slow = data.slowest, fast = data.fastest;
    const cards = [
      ["Lap Time", fmtLap(data.lap_time), ""],
      ["Track Length", numOr(data.track_length), "m"],
      ["Corners", data.n_corners != null ? String(data.n_corners) : "—", ""],
      ["Slowest Corner", slow ? slow.short : "—", slow ? numOr(slow.min_speed) + " km/h" : ""],
      ["Fastest Corner", fast ? fast.short : "—", fast ? numOr(fast.min_speed) + " km/h" : ""],
    ];
    $("soloStats").innerHTML = cards.map((c) =>
      statCard([c[0], c[1], ""], c[2])).join("");

    const corners = data.corners_solo || [];
    if (!corners.length) {
      $("soloTableWrap").innerHTML =
        `<div class="empty-corners" style="border:none;background:none">` +
        `No corner definitions matched this track, so a per-corner breakdown isn't available.</div>`;
      return;
    }
    const headers = ["Corner", "Name", "Min", "Entry", "Exit", "Gear", "Time"];
    const slowShort = slow && slow.short, fastShort = fast && fast.short;
    const rows = corners.map((c) => {
      let label = `<span class="cell-strong">${escapeHtml(c.short)}</span>`;
      if (c.short === slowShort) label += `<span class="pill slow">Slowest</span>`;
      else if (c.short === fastShort) label += `<span class="pill fast">Fastest</span>`;
      return {
        cells: [
          label,
          escapeHtml(c.name || ""),
          `<span class="cell-strong">${numOr(c.min_speed)}</span>`,
          numOr(c.entry_speed),
          numOr(c.exit_speed),
          c.gear != null ? String(c.gear) : "—",
          c.corner_time != null ? c.corner_time.toFixed(3) + "s" : "—",
        ],
        cls: "",
      };
    });
    $("soloTableWrap").innerHTML = buildTable(headers, rows);
  }

  // -------- Small render helpers --------
  function numOr(v) { return (v == null || isNaN(v)) ? "—" : String(Math.round(v)); }
  function fmtSec(s) { return (s == null || isNaN(s)) ? "—" : Number(s).toFixed(3); }
  function statCard(c, sub) {
    const unit = c[2] ? `<span class="u">${c[2]}</span>` : "";
    const subLine = sub ? `<span class="u" style="display:block;margin-top:4px">${escapeHtml(sub)}</span>` : "";
    return `<div class="stat-card"><div class="sc-label">${escapeHtml(c[0])}</div>` +
           `<div class="sc-value">${escapeHtml(String(c[1]))}${unit}${subLine}</div></div>`;
  }
  function buildTable(headers, rows) {
    const thead = "<thead><tr>" + headers.map((h) => `<th>${escapeHtml(h)}</th>`).join("") + "</tr></thead>";
    const tbody = "<tbody>" + rows.map((r) =>
      `<tr class="${r.cls || ""}">` + r.cells.map((c) => `<td>${c}</td>`).join("") + "</tr>"
    ).join("") + "</tbody>";
    return `<table class="data-table">${thead}${tbody}</table>`;
  }

  function renderCorners(corners) {
    const grid = $("cornerGrid");
    grid.innerHTML = "";
    if (!corners.length) {
      const empty = document.createElement("div");
      empty.className = "empty-corners";
      empty.textContent = "No corner definitions were found for this track, so a per-corner breakdown isn't available. The lap summary and track map above still reflect your full lap.";
      grid.appendChild(empty);
      return;
    }

    corners.forEach((c, idx) => {
      const sev = c.severity in SEV_VAR ? c.severity : "ok";
      const card = document.createElement("div");
      card.className = "corner-card";
      card.style.setProperty("--sev", SEV_VAR[sev]);
      card.style.animationDelay = (idx * 45) + "ms";
      card.tabIndex = 0;
      card.setAttribute("role", "button");
      card.setAttribute("aria-expanded", "false");

      const fast = c.time_delta != null && c.time_delta < 0;
      const stats = [];
      const b = fmtSigned(c.brake_diff_m, "m");
      const a = fmtSigned(c.apex_speed_diff, " km/h");
      const ex = fmtSigned(c.exit_speed_diff, " km/h");
      if (b)  stats.push(`<span class="stat">Brake <b>${b}</b></span>`);
      if (a)  stats.push(`<span class="stat">Apex <b>${a}</b></span>`);
      if (ex) stats.push(`<span class="stat">Exit <b>${ex}</b></span>`);

      const issues = (c.issues || []).map((i) => `<li>${escapeHtml(i)}</li>`).join("");
      const tips   = (c.tips || []).map((t) => `<li>${escapeHtml(t)}</li>`).join("");

      card.innerHTML = `
        <div class="cc-head">
          <div>
            <div class="cc-id">${escapeHtml(c.short || ("C" + c.corner_id))}</div>
            <div class="cc-name">${escapeHtml(c.name || "")}</div>
          </div>
          <div class="cc-badge">${escapeHtml(c.grade || "–")}</div>
        </div>
        <div class="cc-delta ${fast ? "fast" : "slow"}">${fmtDelta(c.time_delta)}</div>
        ${stats.length ? `<div class="cc-stats">${stats.join("")}</div>` : ""}
        <div class="cc-detail">
          <div><div class="inner">
            ${issues ? `<div class="detail-block"><div class="detail-h">What happened</div><ul class="detail-list issues">${issues}</ul></div>` : ""}
            ${tips ? `<div class="detail-block"><div class="detail-h">How to fix it</div><ul class="detail-list tips">${tips}</ul></div>` : ""}
          </div></div>
        </div>
        <div class="cc-foot">
          <span>${sev === "ok" ? "On pace" : sev === "minor" ? "Minor loss" : "Major loss"}</span>
          <span class="chev"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg></span>
        </div>`;

      const toggle = () => {
        const open = card.classList.toggle("open");
        card.setAttribute("aria-expanded", open ? "true" : "false");
      };
      const glowColor = SEV_HEX[sev] || "#6f7494";

      // Hover / focus: spotlight this corner's apex on the track map overlay.
      card.addEventListener("mouseenter", () => highlightCorner(c.corner_id, glowColor, false));
      card.addEventListener("mouseleave", () => clearHighlight());
      card.addEventListener("focus", () => highlightCorner(c.corner_id, glowColor, false));
      card.addEventListener("blur", () => clearHighlight());

      // Click / Enter / Space: expand the card, pulse the apex, return to the map.
      card.addEventListener("click", () => {
        toggle();
        highlightCorner(c.corner_id, glowColor, true);
        scrollMapIntoView();
      });
      card.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          toggle();
          highlightCorner(c.corner_id, glowColor, true);
          scrollMapIntoView();
        }
      });

      grid.appendChild(card);
    });
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (m) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));
  }

  // -------- Submit --------
  async function analyze() {
    if (!selectedFile) { showError("Please choose a telemetry CSV first."); return; }
    hideError();
    results.classList.remove("show");
    analyzeBtn.classList.add("loading");
    analyzeBtn.disabled = true;
    loadingState.classList.add("show");
    startStatusCycle(currentMode);

    const fd = new FormData();
    fd.append("file", selectedFile);
    fd.append("mode", currentMode);
    fd.append("driver", $("driver").value || "VER");
    fd.append("year", $("year").value);
    fd.append("session", $("session").value);
    fd.append("track", $("track").value);
    fd.append("lap_index", selectedLapIndex == null ? "" : String(selectedLapIndex));

    try {
      const res = await fetch(apiUrl("/analyze"), { method: "POST", body: fd });
      const ctype = res.headers.get("content-type") || "";
      const payload = ctype.includes("application/json") ? await res.json() : null;
      if (!res.ok) {
        const detail = (payload && (payload.detail || payload.message)) || ("Server error (" + res.status + ").");
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      renderResults(payload);
    } catch (err) {
      showError(apiErrorMessage(err, "Network error. Is the server running?"));
    } finally {
      stopStatusCycle();
      loadingState.classList.remove("show");
      analyzeBtn.classList.remove("loading");
      analyzeBtn.disabled = false;
    }
  }
  analyzeBtn.addEventListener("click", analyze);

  // -------- Track-map corner overlay --------
  // A <canvas> sits exactly over the server-rendered PNG.  /analyze returns each
  // corner apex as a 0..1 fraction of that image (corner_positions), so we can
  // spotlight the true apex when a corner card is hovered, focused or clicked.
  const mapFrame  = $("mapFrame");
  const mapImg    = $("mapImg");
  const mapCanvas = $("mapCanvas");
  const mapCtx    = mapCanvas.getContext("2d");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let cornerPos = new Map();   // corner_id -> {x_norm, y_norm}
  let activeGlow = null;       // {x_norm, y_norm, color} currently drawn
  let pulseRAF = 0;
  let cssW = 0, cssH = 0;

  function setCornerPositions(list) {
    cornerPos = new Map();
    (list || []).forEach((p) => {
      if (p && p.corner_id != null) cornerPos.set(p.corner_id, p);
    });
  }

  function sizeMapCanvas() {
    if (!mapImg.clientWidth || !mapImg.clientHeight) return;
    const dpr = window.devicePixelRatio || 1;
    cssW = mapImg.clientWidth;
    cssH = mapImg.clientHeight;
    mapCanvas.style.width  = cssW + "px";
    mapCanvas.style.height = cssH + "px";
    mapCanvas.width  = Math.round(cssW * dpr);
    mapCanvas.height = Math.round(cssH * dpr);
    mapCtx.setTransform(dpr, 0, 0, dpr, 0, 0);   // draw using CSS pixels
    if (activeGlow) drawGlow(activeGlow); else clearGlowCanvas();
  }

  function clearGlowCanvas() { mapCtx.clearRect(0, 0, cssW, cssH); }

  function glowRadius() { return Math.max(7, Math.min(cssW, cssH) * 0.016); }

  function drawGlow(g, ringExtra = 0, ringAlpha = 0) {
    if (!g || !cssW) return;
    const x = g.x_norm * cssW;
    const y = g.y_norm * cssH;
    const r = glowRadius();
    clearGlowCanvas();

    // soft outer halo
    mapCtx.save();
    mapCtx.shadowColor = g.color;
    mapCtx.shadowBlur  = 26;
    mapCtx.globalAlpha = 0.95;
    mapCtx.fillStyle   = g.color;
    mapCtx.beginPath();
    mapCtx.arc(x, y, r, 0, Math.PI * 2);
    mapCtx.fill();
    mapCtx.restore();

    // white-cored centre for contrast against the dark map
    mapCtx.beginPath();
    mapCtx.arc(x, y, Math.max(2, r * 0.38), 0, Math.PI * 2);
    mapCtx.globalAlpha = 0.9;
    mapCtx.fillStyle = "#ffffff";
    mapCtx.fill();
    mapCtx.globalAlpha = 1;

    // steady ring
    mapCtx.beginPath();
    mapCtx.arc(x, y, r + 5, 0, Math.PI * 2);
    mapCtx.strokeStyle = g.color;
    mapCtx.globalAlpha = 0.85;
    mapCtx.lineWidth = 2;
    mapCtx.stroke();
    mapCtx.globalAlpha = 1;

    // expanding pulse ring (click animation)
    if (ringExtra > 0 && ringAlpha > 0) {
      mapCtx.beginPath();
      mapCtx.arc(x, y, r + 5 + ringExtra, 0, Math.PI * 2);
      mapCtx.strokeStyle = g.color;
      mapCtx.globalAlpha = ringAlpha;
      mapCtx.lineWidth = 3;
      mapCtx.stroke();
      mapCtx.globalAlpha = 1;
    }
  }

  function cancelPulse() { if (pulseRAF) { cancelAnimationFrame(pulseRAF); pulseRAF = 0; } }

  function startPulse() {
    cancelPulse();
    if (reduceMotion) { drawGlow(activeGlow); return; }
    const dur = 620, total = dur * 2;
    const t0 = performance.now();
    const step = (now) => {
      if (!activeGlow) { pulseRAF = 0; return; }
      const elapsed = now - t0;
      const phase = (elapsed % dur) / dur;            // 0..1 each cycle
      drawGlow(activeGlow, phase * 30, (1 - phase) * 0.55);
      if (elapsed < total) {
        pulseRAF = requestAnimationFrame(step);
      } else {
        drawGlow(activeGlow);                          // settle to steady glow
        pulseRAF = 0;
      }
    };
    pulseRAF = requestAnimationFrame(step);
  }

  function highlightCorner(cornerId, color, pulse) {
    const p = cornerPos.get(cornerId);
    if (!p) return;
    if (!cssW) sizeMapCanvas();
    activeGlow = { x_norm: p.x_norm, y_norm: p.y_norm, color: color };
    mapFrame.classList.add("is-linked");
    if (pulse) startPulse(); else { cancelPulse(); drawGlow(activeGlow); }
  }

  function clearHighlight() {
    cancelPulse();
    activeGlow = null;
    clearGlowCanvas();
    mapFrame.classList.remove("is-linked");
  }

  function scrollMapIntoView() {
    if (mapFrame.offsetParent === null) return;   // map hidden -> nothing to show
    mapFrame.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  mapImg.addEventListener("load", sizeMapCanvas);
  window.addEventListener("resize", sizeMapCanvas);

  // -------- Interactive Track Explorer wiring (attached once) --------
  (() => {
    const svg = $("txSvg"), frame = $("txFrame");
    if (!svg) return;
    svg.addEventListener("mousemove", txOnMove);
    svg.addEventListener("mouseenter", () => frame.classList.add("is-active"));
    svg.addEventListener("mouseleave", () => {
      frame.classList.remove("is-active");
      txSetIndex(0, true);                 // Design B: return to lap-start idle
    });
    svg.addEventListener("touchmove", (e) => {
      if (e.touches && e.touches[0]) { txOnMove(e.touches[0]); e.preventDefault(); }
    }, { passive: false });
    svg.addEventListener("touchend", () => {
      frame.classList.remove("is-active");
      txSetIndex(0, true);
    });
  })();

  // -------- Lightbox --------
  const lightbox = $("lightbox");
  const lightboxImg = $("lightboxImg");
  $("mapImg").addEventListener("click", () => {
    if (!$("mapImg").src) return;
    lightboxImg.src = $("mapImg").src;
    lightbox.classList.add("show");
  });
  lightbox.addEventListener("click", () => lightbox.classList.remove("show"));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") lightbox.classList.remove("show");
  });

  // -------- Collapsible corner cards (delegation survives re-renders) --------
  function syncExpandAll() {
    const cards = $("keyCorners").querySelectorAll(".kc-card");
    const allOpen = cards.length && Array.from(cards).every((c) => c.classList.contains("open"));
    $("kcExpandAll").textContent = allOpen ? "Collapse all" : "Expand all";
    $("kcExpandAll").setAttribute("aria-pressed", allOpen ? "true" : "false");
  }
  $("keyCorners").addEventListener("click", (e) => {
    const toggle = e.target.closest(".kc-toggle");
    if (!toggle || !$("keyCorners").contains(toggle)) return;
    const card = toggle.closest(".kc-card");
    const open = card.classList.toggle("open");
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    syncExpandAll();
  });
  $("kcExpandAll").addEventListener("click", () => {
    const cards = $("keyCorners").querySelectorAll(".kc-card");
    const anyClosed = Array.from(cards).some((c) => !c.classList.contains("open"));
    cards.forEach((c) => {
      c.classList.toggle("open", anyClosed);
      const t = c.querySelector(".kc-toggle");
      if (t) t.setAttribute("aria-expanded", anyClosed ? "true" : "false");
    });
    syncExpandAll();
  });

  // -------- Mini-chart hover tooltips (Lap Comparison) --------
  // A single floating tooltip follows the cursor across the Brake / Throttle /
  // Gear / Speed charts, reading the corner's downsampled traces so the visuals
  // become numerically precise. A thin guide line snaps to the sampled point.
  const kcTip = document.createElement("div");
  kcTip.className = "kc-tip-pop";
  kcTip.setAttribute("aria-hidden", "true");
  document.body.appendChild(kcTip);
  let kcActivePlot = null;

  function kcAt(arr, i) {
    const v = arr && arr[i];
    return (v == null || isNaN(v)) ? null : v;
  }
  const _pct = (v) => (v == null ? "—" : Math.round(v * 100) + "%");
  const _kmh = (v) => (v == null ? "—" : Math.round(v) + '<span class="uu"> km/h</span>');
  const _gr  = (v) => (v == null ? "—" : String(Math.round(v)));
  function _tipRow(label, you, pro) {
    return `<span class="kt-k">${label}</span><span class="kt-v">` +
      `<span class="you">${you}</span><span class="sep">·</span><span class="pro">${pro}</span></span>`;
  }
  // The tooltip content is scoped to the hovered chart type: each chart shows
  // its own channel first, plus only the context that helps read it.
  function kcShowTip(corner, i, type) {
    const ch = corner.charts || {};
    const sp = ch.speed || {}, th = ch.throttle || {}, br = ch.brake || {}, ge = ch.gear || {};
    const drv = escapeHtml(kcDriver || "Pro");
    const youSp = kcAt(sp.you, i), proSp = kcAt(sp.pro, i);
    const speedRow = _tipRow("Speed", _kmh(youSp), _kmh(proSp));

    let title, rows = "", extra = "";
    if (type === "throttle") {
      title = "Throttle";
      rows = _tipRow("Throttle", _pct(kcAt(th.you, i)), _pct(kcAt(th.pro, i))) + speedRow;
    } else if (type === "brake") {
      title = "Brake";
      rows = _tipRow("Brake", _pct(kcAt(br.you, i)), _pct(kcAt(br.pro, i))) +
             speedRow +
             _tipRow("Gear", _gr(kcAt(ge.you, i)), _gr(kcAt(ge.pro, i)));
    } else if (type === "gear") {
      title = "Gear";
      rows = _tipRow("Gear", _gr(kcAt(ge.you, i)), _gr(kcAt(ge.pro, i))) + speedRow;
    } else {  // speed
      title = "Speed";
      rows = speedRow;
      if (youSp != null && proSp != null) {
        const d = youSp - proSp, cls = d >= 0 ? "fast" : "slow";   // you faster = green
        extra = `<div class="kc-tip-delta">Δ speed ` +
          `<b class="${cls}">${d >= 0 ? "+" : ""}${Math.round(d)} km/h</b></div>`;
      }
    }
    kcTip.innerHTML =
      `<div class="kc-tip-head"><span>${escapeHtml(corner.short || "")} · ${title}</span>` +
      `<span class="kt-leg"><span class="you">You</span> · <span class="pro">${drv}</span></span></div>` +
      `<div class="kc-tip-grid">${rows}</div>${extra}`;
    kcTip.classList.add("show");
  }
  // Whole-lap chart tooltip: speed at the point + cumulative delta + nearest corner.
  function lapShowTip(i) {
    const d = lapChartData;
    const you = kcAt(d.you, i), pro = kcAt(d.pro, i), del = kcAt(d.delta, i);
    const drv = escapeHtml(kcDriver || "Pro");
    let near = "";
    if (d.markers && d.markers.length && d.you.length > 1) {
      const frac = i / (d.you.length - 1);
      let best = null, bd = Infinity;
      d.markers.forEach((m) => { const dd = Math.abs(m.x - frac); if (dd < bd) { bd = dd; best = m; } });
      if (best) near = ` · ${escapeHtml(best.short)}`;
    }
    let extra = "";
    if (del != null) {
      const cls = del > 0 ? "slow" : "fast";
      extra = `<div class="kc-tip-delta">Cumulative Δ <b class="${cls}">${fmtDelta(del)}</b></div>`;
    }
    kcTip.innerHTML =
      `<div class="kc-tip-head"><span>Lap${near}</span>` +
      `<span class="kt-leg"><span class="you">You</span> · <span class="pro">${drv}</span></span></div>` +
      `<div class="kc-tip-grid">${_tipRow("Speed", _kmh(you), _kmh(pro))}</div>${extra}`;
    kcTip.classList.add("show");
  }
  function hideTip() {
    kcTip.classList.remove("show");
    if (kcActivePlot) { kcActivePlot.classList.remove("kc-hover"); kcActivePlot = null; }
    $("lapChart").classList.remove("lc-hovering");
  }
  function kcMoveTip(clientX, clientY) {
    const tw = kcTip.offsetWidth, th = kcTip.offsetHeight;
    let x = clientX + 14, y = clientY + 14;
    if (x + tw > window.innerWidth - 8)  x = clientX - tw - 14;
    if (y + th > window.innerHeight - 8) y = clientY - th - 14;
    kcTip.style.left = Math.max(8, x) + "px";
    kcTip.style.top  = Math.max(8, y) + "px";
  }
  $("keyCorners").addEventListener("mousemove", (e) => {
    const chartsEl = e.target.closest(".kc-charts");
    const plot = e.target.closest(".cmp-plot");
    if (!chartsEl || !plot) { hideTip(); return; }
    const corner = kcData[Number(chartsEl.dataset.kc)];
    if (!corner || !corner.charts) { hideTip(); return; }
    const speedYou = (corner.charts.speed && corner.charts.speed.you) || [];
    const n = Math.max(speedYou.length, 2);
    const rect = plot.getBoundingClientRect();
    let f = (e.clientX - rect.left) / rect.width;
    f = Math.max(0, Math.min(1, f));
    // Invert the mini chart's 3% horizontal padding (cmpSvg uses pad=3, w=100).
    let i = Math.round(((f * 100 - 3) / 94) * (n - 1));
    i = Math.max(0, Math.min(n - 1, i));
    if (plot !== kcActivePlot) {
      if (kcActivePlot) kcActivePlot.classList.remove("kc-hover");
      kcActivePlot = plot;
      plot.classList.add("kc-hover");
    }
    const cursor = plot.querySelector(".cmp-cursor");
    if (cursor) cursor.style.left = (3 + 94 * (i / (n - 1))) + "%";  // snap to point
    kcShowTip(corner, i, plot.dataset.ch || "speed");   // scoped to chart type
    kcMoveTip(e.clientX, e.clientY);
  });
  $("keyCorners").addEventListener("mouseleave", hideTip);

  // Whole-lap chart hover: a guide spans both plots; the tooltip reads the point.
  $("lapChart").addEventListener("mousemove", (e) => {
    const host = $("lapChart");
    const plot = e.target.closest(".lc-plot");
    if (!plot || !host.contains(plot)) { hideTip(); return; }
    const n = (lapChartData.you && lapChartData.you.length) || 0;
    if (n < 2) { hideTip(); return; }
    const rect = plot.getBoundingClientRect();
    let f = (e.clientX - rect.left) / rect.width;
    f = Math.max(0, Math.min(1, f));
    let i = Math.round(((f * 100 - 4) / 92) * (n - 1));   // lap plots use pad=4
    i = Math.max(0, Math.min(n - 1, i));
    host.classList.add("lc-hovering");
    const gx = (4 + 92 * (i / (n - 1))) + "%";
    host.querySelectorAll(".lc-cursor").forEach((c) => { c.style.left = gx; });
    lapShowTip(i);
    kcMoveTip(e.clientX, e.clientY);
  });
  $("lapChart").addEventListener("mouseleave", hideTip);
  window.addEventListener("resize", declutterLapMarkers);

  // -------- Init --------
  applyModeUI(currentMode);   // set default mode description, label & muted state
  resetTrackDetection();      // track is auto-detected from the CSV on upload
})();
