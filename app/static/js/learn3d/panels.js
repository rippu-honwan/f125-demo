// DOM / SVG panels for the 3D Learn mode: HUD, minimap, telemetry traces, corner list.
import { cuePoint, fmtTime, lineBand, LINE_BIG_M } from "./core.js";

const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ESC[c]);

const ACTION_LABEL = { brake: "BRAKE", lift: "LIFT", flat: "FLAT" };
export const PRO_COLOR = "#ff8a00";   // index.js paints the cars from these too
export const YOU_COLOR = "#2e7dff";

const lastHtml = new WeakMap();
/** Skip DOM writes when nothing changed (the HUD is redrawn every animation frame). */
function setHtml(el, html) {
  if (lastHtml.get(el) === html) return;
  lastHtml.set(el, html);
  el.innerHTML = html;
}

/** Top-left: who/where + countdown to the next cue. Bottom-left: pedals, speed, gear. */
export function renderHud(top, bottom, meta, s, cue) {
  setHtml(top,
    `${esc(meta.driver)} · ${meta.year} ${esc(meta.session)} · ${esc(meta.track_name)} · ${fmtTime(meta.lap_time)}` +
    (cue ? `<br><span class="l3d-cue">${esc(cue.corner.short)} · ${ACTION_LABEL[cue.corner.action]} in ${Math.round(cue.dist)} m</span>` : ""));
  setHtml(bottom,
    `<span class="l3d-pedal" title="Throttle"><i style="height:${Math.round(s.throttle * 100)}%;background:var(--ok)"></i></span>` +
    `<span class="l3d-pedal" title="Brake"><i style="height:${Math.round(s.brake * 100)}%;background:var(--accent)"></i></span>` +
    `<span class="l3d-speed">${Math.round(s.speed)}</span>km/h · G${s.gear}`);
}

/** "+0.42" / "−0.13" and its colour class; the sign comes from the rounded value, so never "−0.00". */
function gapLabel(g) {
  const r = Math.round(g * 100) / 100;
  return [`${r < 0 ? "−" : "+"}${Math.abs(r).toFixed(2)}`, r > 0 ? "l3d-gap-slow" : r < 0 ? "l3d-gap-fast" : ""];
}

/**
 * Compare HUD. Top: who vs who + live gap (red = you're behind). Bottom: your pedals and speed, the
 * pro's speed. A car with `done` set has crossed the line: it shows "finished", not its last speed.
 */
export function renderCompareHud(top, bottom, meta, you, pro, gap) {
  const [text, cls] = gapLabel(gap);
  setHtml(top,
    `<span style="color:${YOU_COLOR}">You</span> vs <span style="color:${PRO_COLOR}">${esc(meta.driver)}</span>` +
    ` · ${esc(meta.track_name)}<br><span class="l3d-gap ${cls}">${text}</span>`);
  const pedal = (v) => (you.done ? 0 : Math.round(v * 100));
  setHtml(bottom,
    `<span class="l3d-pedal" title="Your throttle"><i style="height:${pedal(you.throttle)}%;background:var(--ok)"></i></span>` +
    `<span class="l3d-pedal" title="Your brake"><i style="height:${pedal(you.brake)}%;background:var(--accent)"></i></span>` +
    (you.done ? `<span class="l3d-speed">Finished</span>` : `<span class="l3d-speed">${Math.round(you.speed)}</span>km/h · G${you.gear}`) +
    `<span class="l3d-pro-speed">${esc(meta.driver)} ${pro.done ? "finished" : Math.round(pro.speed)}</span>`);
}

/** Drive HUD. Top: lap clock, live gap to the pro, off-track note. Bottom: your pedals, speed and gear. */
export function renderDriveHud(top, bottom, meta, v) {
  const [text, cls] = gapLabel(v.gap);
  setHtml(top,
    `You vs ${esc(meta.driver)} (ghost) · ${fmtTime(v.t)}<br><span class="l3d-gap ${cls}">${text}</span>` +
    (v.invalid ? `<br><span class="l3d-cue">${v.offTrack ? "Off track · " : ""}lap invalid</span>` : ""));
  setHtml(bottom,
    `<span class="l3d-pedal" title="Throttle"><i style="height:${v.throttle * 100}%;background:var(--ok)"></i></span>` +
    `<span class="l3d-pedal" title="Brake"><i style="height:${v.brake * 100}%;background:var(--accent)"></i></span>` +
    `<span class="l3d-speed">${Math.round(v.kmh)}</span>km/h · G${v.gear}`);
}

/** North-up outline of the circuit with one dot per car (`colors[k]` for `set(...states)[k]`). */
export function createMinimap(svg, points, colors = ["var(--accent)"]) {
  const xs = [], ys = [];
  for (let i = 0; i < points.x.length; i += 5) { xs.push(points.x[i]); ys.push(points.y[i]); }
  xs.push(points.x[0]); ys.push(points.y[0]);   // close the loop
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const w = maxX - minX, h = maxY - minY, span = Math.max(w, h) || 1, pad = span * 0.06;
  // SVG y grows downward, so plot -y to keep north up; centre the shorter axis.
  svg.setAttribute("viewBox",
    `${minX - (span - w) / 2 - pad} ${-maxY - (span - h) / 2 - pad} ${span + 2 * pad} ${span + 2 * pad}`);
  svg.innerHTML =
    `<polyline points="${xs.map((x, i) => `${x},${-ys[i]}`).join(" ")}" fill="none" stroke="rgba(255,255,255,.75)" stroke-width="${span / 60}"/>` +
    colors.map((c) => `<circle r="${span / 30}" fill="${c}"/>`).join("");
  const dots = [...svg.querySelectorAll("circle")];
  return {
    set(...states) {
      states.forEach((s, k) => { dots[k].setAttribute("cx", s.x); dots[k].setAttribute("cy", -s.y); });
    },
  };
}

const ROWS = [
  { key: "speed", label: "Speed", color: "#4aa3ff" },
  { key: "throttle", label: "Throttle", color: "var(--ok)", min: 0, max: 1 },
  { key: "brake", label: "Brake", color: "var(--accent)", min: 0, max: 1 },
  { key: "steer", label: "Steer (est.)", color: "#c9a0ff", min: -1, max: 1 },
];

/** Rows to draw: Learn = the four pro traces; Compare = pro vs you for speed/throttle/brake + the gap. */
function rowsFor(points, you) {
  if (!you) return ROWS.map((r) => ({ ...r, series: [{ vals: points[r.key], color: r.color }] }));
  const gap = points.t.map((t, i) => you.t[i] - t);
  const g = Math.max(0.1, ...gap.map(Math.abs));
  return [
    ...ROWS.slice(0, 3).map((r) => ({
      ...r, series: [{ vals: points[r.key], color: PRO_COLOR }, { vals: you[r.key], color: YOU_COLOR }],
    })),
    { label: "Gap (s)", min: -g, max: g, zero: true, series: [{ vals: gap, color: "#ffd400" }] },
  ];
}

/**
 * Stacked traces over lap distance, shaded brake zones, one shared cursor. Click = seek.
 * With `you` (a youPoints object) each row overlays you on the pro and a gap row is added.
 */
export function createTelemetry(el, points, onSeek, you = null, proName = "Pro") {
  const W = 1000, H = 48, n = points.d.length, L = points.d[n - 1];
  const step = Math.max(1, Math.floor(n / 600));
  const xOf = (d) => ((d / L) * W).toFixed(1);

  const zones = [];
  for (let i = 0, start = null; i < n; i++) {
    const on = points.brake[i] >= 0.5;
    if (on && start === null) start = points.d[i];
    if ((!on || i === n - 1) && start !== null) { zones.push([start, points.d[i]]); start = null; }
  }
  const shade = zones.map(([a, b]) =>
    `<rect x="${xOf(a)}" width="${Math.max(2, xOf(b) - xOf(a))}" y="0" height="${H}" fill="var(--accent)" opacity=".15"/>`).join("");

  const legend = you
    ? `<div class="l3d-legend"><span><i style="background:${YOU_COLOR}"></i>You</span>` +
      `<span><i style="background:${PRO_COLOR}"></i>${esc(proName)}</span></div>`
    : "";
  el.innerHTML = legend + rowsFor(points, you).map((r) => {
    const all = r.series.flatMap((s) => s.vals);
    const lo = r.min ?? Math.min(...all), hi = r.max ?? Math.max(...all), range = hi - lo || 1;
    const yOf = (v) => (H - ((v - lo) / range) * H).toFixed(1);
    const lines = r.series.map((s) => {
      let pts = "";
      for (let i = 0; i < n; i += step) pts += `${xOf(points.d[i])},${yOf(s.vals[i])} `;
      return `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="1.5" vector-effect="non-scaling-stroke"/>`;
    }).join("");
    const zero = r.zero
      ? `<line x1="0" x2="${W}" y1="${yOf(0)}" y2="${yOf(0)}" stroke="rgba(255,255,255,.3)" vector-effect="non-scaling-stroke"/>`
      : "";
    return `<div class="l3d-trace"><span class="l3d-trace-label">${r.label}</span>` +
      `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">${shade}${zero}${lines}` +
      `<line class="l3d-cursor" x1="0" x2="0" y1="0" y2="${H}" stroke="#fff" vector-effect="non-scaling-stroke"/></svg></div>`;
  }).join("");

  el.querySelectorAll("svg").forEach((svg) => svg.addEventListener("click", (e) => {
    const box = svg.getBoundingClientRect();
    onSeek(((e.clientX - box.left) / box.width) * L);
  }));
  const cursors = el.querySelectorAll(".l3d-cursor");
  return {
    set(d) {
      const x = xOf(d);
      cursors.forEach((c) => { c.setAttribute("x1", x); c.setAttribute("x2", x); });
    },
  };
}

/**
 * One row per corner. Learn: brake point → min speed, brake length.
 * Compare: pro → your apex speed, and the time you gained (−) or lost (+) there. Click = replay that corner.
 * `line` (3D Compare with your own line): {offsets: lineOffsets(...), driver}; setLine(true) then shows
 * how your line runs against the pro's at entry / apex / exit in place of the apex speeds.
 */
export function createCornerList(el, corners, onPick, compare = false, line = null) {
  const head = compare
    ? `<span>Turn</span><span>Apex speed: pro → you</span><span>Gap</span>`
    : `<span>Turn</span><span>Brake point → min speed</span><span>Brake</span>`;
  el.innerHTML =
    `<div class="l3d-crow l3d-chead" aria-hidden="true">${head}</div>` +
    corners.map((c, i) => {
      let mid, last;
      if (compare) {
        mid = `${esc(ACTION_LABEL[c.action])} · ${Math.round(c.min_speed_kmh)} → ${Math.round(c.your_min_speed_kmh)} km/h`;
        const [text, cls] = c.time_delta == null ? ["—", ""] : gapLabel(c.time_delta);
        last = `<span class="${cls}">${text}</span>`;
      } else {
        const where = c.action === "brake" ? `${Math.round(c.apex_m - c.brake_start_m)} m before apex` : esc(c.action);
        mid = `${where} → ${Math.round(c.min_speed_kmh)} km/h · G${c.apex_gear}`;
        last = c.brake_duration_s != null ? `${c.brake_duration_s.toFixed(1)} s` : "—";
      }
      return `<button type="button" class="l3d-crow" data-i="${i}"><span>${esc(c.short)}</span>` +
        `<span>${mid}</span><span>${last}</span></button>`;
    }).join("");
  const rows = [...el.querySelectorAll("button.l3d-crow")];
  rows.forEach((r) => r.addEventListener("click", () => onPick(corners[+r.dataset.i])));
  // Line mode swaps each row's middle cell; keep both texts.
  const headMid = el.querySelector(".l3d-chead > span:nth-child(2)");
  const speedHead = headMid.innerHTML, speedMid = rows.map((r) => r.children[1].innerHTML);
  const lineMid = line ? line.offsets.map((o) =>
    `Entry ${lineBand(o.entry)} · Apex ${lineBand(o.apex)} · Exit ${lineBand(o.exit)}`) : [];
  const big = line ? line.offsets.map((o) =>
    [o.entry, o.apex, o.exit].some((m) => m != null && Math.abs(m) >= LINE_BIG_M)) : [];
  let current = -2;
  return {
    set(d) {
      const k = corners.findIndex((c) => d >= cuePoint(c) && d <= c.exit_m);
      if (k === current) return;
      current = k;
      rows.forEach((r, i) => {
        r.classList.toggle("active", i === k);
        if (i === k) r.setAttribute("aria-current", "true"); else r.removeAttribute("aria-current");
      });
      // Keep the active row visible while the lap plays (no-op while the list is hidden).
      if (k >= 0) {
        const rb = rows[k].getBoundingClientRect(), eb = el.getBoundingClientRect();
        if (rb.top < eb.top || rb.bottom > eb.bottom) el.scrollTop += rb.top - eb.top - (eb.height - rb.height) / 2;
      }
    },
    setLine(on) {
      if (!line) return;
      headMid.innerHTML = on ? `Line vs ${esc(line.driver)}: entry · apex · exit` : speedHead;
      rows.forEach((r, i) => {
        r.children[1].innerHTML = on ? lineMid[i] : speedMid[i];
        r.classList.toggle("l3d-line-big", on && big[i]);
      });
    },
  };
}
