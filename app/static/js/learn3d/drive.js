// 3D Learn drive mode: drive the loaded lap yourself, with the pro as a ghost.
// drive_sim.js does the physics; this module owns the setup card, countdown, inputs and results.
// index.js owns the scene and the HUD: it calls attach()/detach() per Learn session,
// frame(dt) every animation frame while isOpen(), and draws from view().
import { KMH, buildModel, createCar, advance, pose, gapAt, cornerStats, gearAt } from "./drive_sim.js";
import { esc } from "./panels.js";

const COUNTDOWN_S = 3;
const GO_SHOWN_S = 0.6;                         // "GO" stays up this long into the lap
const KEYS_DRIVE = "↑/W throttle · ↓/S/Space brake · ←→/A D steer (Manual) · R restart · Esc quit · 1/2/3 camera";
const PEDAL_KEYS = {
  throttle: new Set(["ArrowUp", "KeyW"]),
  brake: new Set(["ArrowDown", "KeyS", "Space"]),
  left: new Set(["ArrowLeft", "KeyA"]),
  right: new Set(["ArrowRight", "KeyD"]),
};
const coarse = () => window.matchMedia("(pointer: coarse)").matches;   // phones: Assist + touch pedals

const $ = (id) => document.getElementById(id);
const pedalBtns = [...document.querySelectorAll("[data-pedal]")];
// host: {points, corners, meta, store, setPlaying, setLine, blocked}, set by attach().
let host = null, model = null;
let phase = "off";                              // off | setup | countdown | run | results
let car = null, countdown = 0, keysBefore = "";
let settings = { mode: "assist", line: true, ghost: true };
// Inputs held now, per control: key codes and "ptr<id>" for touch. A control is on while any is held.
const held = { throttle: new Set(), brake: new Set(), left: new Set(), right: new Set() };

const fmtLap = (s) => {
  const ms = Math.round(s * 1000);                              // round first: never "1:60.000"
  return `${Math.floor(ms / 60000)}:${((ms % 60000) / 1000).toFixed(3).padStart(6, "0")}`;
};

function markPressed(b, on) {
  b.classList.toggle("active", on);
  b.setAttribute("aria-pressed", String(on));
}

/** Start offering Drive for a Learn session. */
export function attach(h) {
  detach();
  host = h;
  model = null;                                 // built on first open
  $("l3dDriveBtn").hidden = false;
}

/** Leave Drive (if open) and stop offering it. */
export function detach() {
  if (phase !== "off") close(false);
  host = null;
  model = null;
  $("l3dDriveBtn").hidden = true;
}

/** Drive is open (setup, countdown, run or results): Learn's controls stay locked. */
export const isOpen = () => phase !== "off";

function input() {
  return {
    throttle: held.throttle.size ? 1 : 0,
    brake: held.brake.size ? 1 : 0,
    steer: (held.left.size ? 1 : 0) - (held.right.size ? 1 : 0),   // + = left
  };
}

function releaseAll() {
  for (const s of Object.values(held)) s.clear();
}

function open() {
  if (!host || phase !== "off" || host.blocked()) return;
  model = model || buildModel(host.points);
  settings = {
    mode: coarse() || host.store.get("l3d_drive_mode", "assist") !== "manual" ? "assist" : "manual",
    line: host.store.get("l3d_drive_line", "1") === "1",
    ghost: host.store.get("l3d_drive_ghost", "1") === "1",
  };
  syncSetup();
  host.setPlaying(false);
  keysBefore = $("l3dKeys").textContent;
  $("l3dKeys").textContent = KEYS_DRIVE;
  $("blkLearn3d").classList.add("l3d-practicing", "l3d-driving");
  $("l3dDriveBtn").hidden = true;
  $("l3dPracticeBtn").disabled = true;   // one mode at a time, also for keyboard users
  showPanel("setup");
  phase = "setup";
  $("l3dDriveStart").focus();
}

function syncSetup() {
  document.querySelectorAll("[data-dmode]").forEach((b) => {
    markPressed(b, b.dataset.dmode === settings.mode);
    if (b.dataset.dmode === "manual") b.hidden = coarse();
  });
  document.querySelectorAll("[data-dopt]").forEach((b) => markPressed(b, settings[b.dataset.dopt]));
}

function showPanel(which) {                     // "setup" | "results" | null
  $("l3dDrive").hidden = !which;
  $("l3dDriveSetup").hidden = which !== "setup";
  $("l3dDriveResults").hidden = which !== "results";
}

/** Start (or restart) a lap: countdown first, the car waiting on the line. */
function begin() {
  host.store.set("l3d_drive_mode", settings.mode);
  host.store.set("l3d_drive_line", settings.line ? "1" : "0");
  host.store.set("l3d_drive_ghost", settings.ghost ? "1" : "0");
  car = createCar(model, settings.mode);
  countdown = COUNTDOWN_S;
  releaseAll();
  showPanel(null);
  if (document.activeElement) document.activeElement.blur();   // Space must not click a focused button
  host.setLine(settings.line);
  $("l3dPedals").hidden = !coarse();
  $("l3dCountdown").hidden = false;
  $("l3dCountdown").textContent = String(COUNTDOWN_S);
  phase = "countdown";
}

/** Every animation frame while open: countdown, physics, lap end. dt in seconds. */
export function frame(dt) {
  if (phase === "countdown") {
    countdown -= dt;
    const text = countdown > 0 ? String(Math.ceil(countdown)) : "GO";
    if ($("l3dCountdown").textContent !== text) $("l3dCountdown").textContent = text;
    if (countdown <= 0) phase = "run";          // inputs held during the countdown act from here
  } else if (phase === "run") {
    advance(model, car, input(), dt);
    if (car.t > GO_SHOWN_S) $("l3dCountdown").hidden = true;
    if (car.finished) finish();
  }
  for (const b of pedalBtns) b.classList.toggle("active", held[b.dataset.pedal].size > 0);
}

/** What index.js draws while driving, or null when Learn draws as usual (Drive off or on its setup card). */
export function view() {
  if (phase === "off" || phase === "setup") return null;
  const i = phase === "results" ? { throttle: 0, brake: 0 } : input();   // a finger may still be down after the line
  return {
    you: pose(host.points, model, car), ghost: settings.ghost, t: car.t, d: car.s,
    gap: car.trace.d.length > 1 ? gapAt(car.trace, host.points, car.s) : 0,
    kmh: car.v * KMH, gear: gearAt(model, car.v), throttle: i.throttle, brake: i.brake,
    offTrack: car.offTrack, invalid: car.invalid,
  };
}

function finish() {
  phase = "results";
  $("l3dPedals").hidden = true;
  $("l3dCountdown").hidden = true;
  renderResults();
  showPanel("results");
  $("l3dDriveResults").querySelector("button").focus();
}

function brakeText(r) {
  if (r.brakeAt == null) return r.proBrakeAt == null ? "—" : "no brake";
  if (r.proBrakeAt == null) return "braked (pro didn't)";
  const m = Math.round(r.proBrakeAt - r.brakeAt);
  return m === 0 ? "on point" : m > 0 ? `${m} m early` : `${-m} m late`;
}

function renderResults() {
  const m = host.meta, pts = host.points, proT = pts.t[pts.t.length - 1], delta = car.t - proT;
  const key = `l3d_drive_best_${m.year}_${m.driver}_${m.session}_${m.track}_${car.mode}`;
  const prev = parseFloat(host.store.get(key, ""));
  const isBest = !car.invalid && !(prev <= car.t);          // no previous best (NaN) counts as a new best
  if (isBest) host.store.set(key, car.t.toFixed(3));
  const best = car.invalid ? `Invalid lap (off track) — not counted${prev > 0 ? ` · Best: ${fmtLap(prev)}` : ""}`
    : isBest ? "New best!" : `Best: ${fmtLap(prev)}`;
  const signed = (x) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(2)} s`;
  $("l3dDriveResults").innerHTML =
    `<h3>Lap ${fmtLap(car.t)} · ${esc(m.driver)} ${fmtLap(proT)} · ${signed(delta)}</h3>` +
    `<p>${best} · ${car.mode === "manual" ? "Manual" : "Assist"} · + = slower than the pro</p>` +
    `<table class="l3d-results"><thead><tr><th>Turn</th><th>Brake point</th><th>Min speed</th><th>Time lost</th></tr></thead><tbody>` +
    cornerStats(car.trace, pts, host.corners).map((r) =>
      `<tr><td>${esc(r.corner.short)}</td><td>${brakeText(r)}</td>` +
      `<td>${r.minKmh == null ? "—" : `${Math.round(r.minKmh)} vs ${Math.round(r.proMinKmh)} km/h`}</td>` +
      `<td class="${r.lost > 0.005 ? "l3d-gap-slow" : r.lost < -0.005 ? "l3d-gap-fast" : ""}">${signed(r.lost)}</td></tr>`).join("") +
    `</tbody></table>` +
    `<div class="l3d-practice-actions"><button type="button" class="l3d-primary" data-drive="retry">Retry</button>` +
    `<button type="button" data-drive="back">Back to Learn</button></div>`;
}

function close(restoreFocus = true) {
  phase = "off";
  car = null;
  releaseAll();
  showPanel(null);
  $("l3dCountdown").hidden = true;
  $("l3dPedals").hidden = true;
  $("blkLearn3d").classList.remove("l3d-practicing", "l3d-driving");
  $("l3dPracticeBtn").disabled = false;
  if (keysBefore) $("l3dKeys").textContent = keysBefore;
  if (host) {
    host.setLine(false);
    $("l3dDriveBtn").hidden = false;
    if (restoreFocus) $("l3dDriveBtn").focus();
  }
}

/** Keys while Drive is open. Returns true when index.js should ignore the key. */
export function keydown(e) {
  if (phase === "off") return false;
  if (e.key === "Escape") {
    close();
    return true;
  }
  // Typing elsewhere on the page. Not the scrub slider: it can keep focus, and Drive owns the arrows.
  if (e.target.matches?.("input:not([type=range]), select, textarea")) return false;
  if (/^[123]$/.test(e.key)) return false;      // cameras still switch
  if (phase !== "countdown" && phase !== "run") return true;   // a card is up: Learn shortcuts stay off
  if (e.key === "r" || e.key === "R") {
    if (!e.repeat) begin();
    return true;
  }
  for (const [control, keys] of Object.entries(PEDAL_KEYS)) {
    if (keys.has(e.code)) {
      e.preventDefault();
      held[control].add(e.code);
    }
  }
  return true;
}

export function keyup(e) {
  // Also after the lap ended: a Space held over the line mustn't click the focused Retry.
  let was = false;
  for (const s of Object.values(held)) was = s.delete(e.code) || was;
  if (was) e.preventDefault();
}

// ---- controls (wired once; modules run after the DOM is parsed) ----
$("l3dDriveBtn").addEventListener("click", open);
document.querySelectorAll("[data-dmode]").forEach((b) => b.addEventListener("click", () => {
  settings.mode = b.dataset.dmode;
  syncSetup();
}));
document.querySelectorAll("[data-dopt]").forEach((b) => b.addEventListener("click", () => {
  settings[b.dataset.dopt] = !settings[b.dataset.dopt];
  syncSetup();
}));
$("l3dDriveStart").addEventListener("click", () => phase === "setup" && begin());
$("l3dDriveCancel").addEventListener("click", () => close());
$("l3dDriveResults").addEventListener("click", (e) => {
  const act = e.target.closest("[data-drive]")?.dataset.drive;
  if (act === "retry") begin();
  else if (act === "back") close();
});
for (const b of pedalBtns) {
  const set = held[b.dataset.pedal];
  b.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;                 // primary button / touch / pen only
    e.preventDefault();
    b.setPointerCapture(e.pointerId);
    set.add(`ptr${e.pointerId}`);
  });
  for (const ev of ["pointerup", "pointercancel"]) b.addEventListener(ev, (e) => set.delete(`ptr${e.pointerId}`));
  b.addEventListener("contextmenu", (e) => e.preventDefault());
}
// Switching windows while holding = letting go of everything.
window.addEventListener("blur", releaseAll);
