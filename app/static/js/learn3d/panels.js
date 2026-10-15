// DOM / SVG panels for the 3D Learn mode: HUD, minimap, telemetry traces, corner list.
import { cuePoint, fmtTime } from "./core.js";

const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ESC[c]);

const ACTION_LABEL = { brake: "BRAKE", lift: "LIFT", flat: "FLAT" };

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

/** North-up outline of the circuit with a dot for the car. */
export function createMinimap(svg, points) {
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
    `<circle r="${span / 30}" fill="var(--accent)"/>`;
  const dot = svg.querySelector("circle");
  return { set(s) { dot.setAttribute("cx", s.x); dot.setAttribute("cy", -s.y); } };
}

const ROWS = [
  { key: "speed", label: "Speed", color: "#4aa3ff" },
  { key: "throttle", label: "Throttle", color: "var(--ok)", min: 0, max: 1 },
  { key: "brake", label: "Brake", color: "var(--accent)", min: 0, max: 1 },
  { key: "steer", label: "Steer (est.)", color: "#c9a0ff", min: -1, max: 1 },
];

/** Four stacked traces over lap distance, shaded brake zones, one shared cursor. Click = seek. */
export function createTelemetry(el, points, onSeek) {
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

  el.innerHTML = ROWS.map((r) => {
    const vals = points[r.key];
    const lo = r.min ?? Math.min(...vals), hi = r.max ?? Math.max(...vals), range = hi - lo || 1;
    let pts = "";
    for (let i = 0; i < n; i += step) pts += `${xOf(points.d[i])},${(H - ((vals[i] - lo) / range) * H).toFixed(1)} `;
    return `<div class="l3d-trace"><span class="l3d-trace-label">${r.label}</span>` +
      `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">${shade}` +
      `<polyline points="${pts}" fill="none" stroke="${r.color}" stroke-width="1.5" vector-effect="non-scaling-stroke"/>` +
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

/** One row per corner (brake point → min speed, brake length). Click = replay that corner. */
export function createCornerList(el, corners, onPick) {
  el.innerHTML =
    `<div class="l3d-crow l3d-chead" aria-hidden="true"><span>Turn</span><span>Brake point → min speed</span><span>Brake</span></div>` +
    corners.map((c, i) => {
      const where = c.action === "brake" ? `${Math.round(c.apex_m - c.brake_start_m)} m before apex` : esc(c.action);
      const len = c.brake_duration_s != null ? `${c.brake_duration_s.toFixed(1)} s` : "—";
      return `<button type="button" class="l3d-crow" data-i="${i}"><span>${esc(c.short)}</span>` +
        `<span>${where} → ${Math.round(c.min_speed_kmh)} km/h · G${c.apex_gear}</span><span>${len}</span></button>`;
    }).join("");
  const rows = [...el.querySelectorAll("button.l3d-crow")];
  rows.forEach((r) => r.addEventListener("click", () => onPick(corners[+r.dataset.i])));
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
  };
}
