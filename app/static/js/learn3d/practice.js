// 3D Learn brake practice: hold the brake from each brake point until the pro lets go.
// rhythm.js judges; this module owns the practice UI. index.js owns the clock and calls
// attach()/detach() per Learn session, frame(t) every animation frame and finish() at the lap end.
import { stateAt } from "./core.js";
import { buildNotes, createJudge, dtMetres } from "./rhythm.js";
import { esc } from "./panels.js";

const LANE_AHEAD_S = 3;                        // lap time shown ahead of the hit line
const HIT_X = 15;                              // hit line, % of the lane width (matches .l3d-hitline)
const GRADE_LABEL = { perfect: "PERFECT", good: "GOOD", miss: "MISS" };
const BRAKE_KEYS = new Set(["Space", "ArrowDown"]);
const KEYS_PRACTICE = "Hold Space or ↓ to brake · Esc quits practice";

const $ = (id) => document.getElementById(id);
// host: {points, corners, meta, store, getT, seek, setPlaying, setRate, setCues, blocked}, set by attach().
let host = null;
let phase = "off";                              // off | setup | run | results
let judge = null, bars = [], keysBefore = "", lastScore = "";
const inputs = new Set();                      // brake inputs held now (key codes, "ptr<id>"); brake is on while any is
let settings = { diff: "easy", rate: 1 };

function markPressed(b, on) {
  b.classList.toggle("active", on);
  b.setAttribute("aria-pressed", String(on));
}

/** Start offering practice for a Learn session. */
export function attach(h) {
  detach();
  host = h;
  $("l3dPracticeBtn").hidden = false;
}

/** Leave practice (if any) and stop offering it. */
export function detach() {
  if (phase !== "off") close(false);
  host = null;
  $("l3dPracticeBtn").hidden = true;
}

export const isRunning = () => phase === "run";
/** Practice is open (setup, run or results): the Learn controls stay locked. */
export const isOpen = () => phase !== "off";
/** False only while a Hard lap runs: index.js hides the HUD's "brake in N m" cue then. */
export const hintsOn = () => phase !== "run" || settings.diff === "easy";

function open() {
  if (!host || phase !== "off" || host.blocked()) return;
  settings = {
    diff: host.store.get("l3d_practice_diff", "easy") === "hard" ? "hard" : "easy",
    rate: host.store.get("l3d_practice_rate", "1") === "0.5" ? 0.5 : 1,
  };
  syncSetup();
  host.setPlaying(false);
  keysBefore = $("l3dKeys").textContent;
  $("l3dKeys").textContent = KEYS_PRACTICE;
  $("blkLearn3d").classList.add("l3d-practicing");
  $("l3dPracticeBtn").hidden = true;
  showPanel("setup");
  phase = "setup";
  $("l3dPracticeStart").focus();
}

function syncSetup() {
  document.querySelectorAll("[data-diff]").forEach((b) => markPressed(b, b.dataset.diff === settings.diff));
  document.querySelectorAll("[data-prate]").forEach((b) => markPressed(b, +b.dataset.prate === settings.rate));
}

function showPanel(which) {                     // "setup" | "results" | null
  $("l3dPractice").hidden = !which;
  $("l3dPracticeSetup").hidden = which !== "setup";
  $("l3dPracticeResults").hidden = which !== "results";
}

function begin() {
  host.store.set("l3d_practice_diff", settings.diff);
  host.store.set("l3d_practice_rate", String(settings.rate));
  judge = createJudge(buildNotes(host.points, host.corners));
  const lane = $("l3dLane");
  lane.querySelectorAll(".l3d-note").forEach((n) => n.remove());
  bars = judge.notes.map(() => {
    const b = document.createElement("i");
    b.className = "l3d-note";
    lane.appendChild(b);
    return b;
  });
  const hard = settings.diff === "hard";
  lane.hidden = hard;
  host.setCues(!hard);
  $("blkLearn3d").classList.toggle("l3d-hard", hard);
  $("l3dScore").hidden = false;
  $("l3dBrake").hidden = false;
  lastScore = "";
  inputs.clear();
  clearPop();
  showPanel(null);
  if (document.activeElement) document.activeElement.blur();   // Space must not click a focused button
  phase = "run";
  host.seek(0);
  host.setRate(settings.rate);
  host.setPlaying(true);
}

/** Every animation frame while practising: time-out misses, the lane and the score. */
export function frame(t) {
  if (phase !== "run") return;
  for (const j of judge.tick(t)) popup(j);
  drawLane(t);
  const s = judge.score();
  const text = `${s.points} pts · combo ${s.combo}`;
  if (text !== lastScore) {
    lastScore = text;
    $("l3dScore").textContent = text;
  }
}

function drawLane(t) {
  const span = 100 - HIT_X;
  judge.notes.forEach((n, i) => {
    const b = bars[i];
    const x0 = HIT_X + ((n.start - t) / LANE_AHEAD_S) * span;
    const x1 = HIT_X + ((n.end - t) / LANE_AHEAD_S) * span;
    const show = x1 > 0 && x0 < 100;
    b.style.display = show ? "" : "none";
    if (!show) return;
    b.style.left = `${x0}%`;
    b.style.width = `${Math.max(0.5, x1 - x0)}%`;
    const st = n.press?.grade === "miss" || n.release?.grade === "miss" ? "miss"
      : judge.holding === n ? "held" : n.press ? "ok" : "";
    if (b.dataset.st !== st) {
      b.dataset.st = st;
      b.className = `l3d-note ${st}`;
    }
  });
}

function popup(j) {
  let text = GRADE_LABEL[j.grade];
  if (j.dt == null) text += j.kind === "press" ? " · brake point" : " · release";
  else {
    if (Math.abs(j.dt) < 0.005) text += " · on time";
    else {
      const speed = stateAt(host.points, j.kind === "press" ? j.note.start : j.note.end).speed;
      const m = Math.round(Math.abs(dtMetres(j.dt, speed)));
      text += ` · ${j.dt > 0 ? "late" : "early"} ${Math.abs(j.dt).toFixed(2)} s ${m < 1 ? "<" : "≈"} ${Math.max(1, m)} m`;
    }
  }
  const pop = $("l3dPop");
  pop.textContent = text;
  pop.className = `l3d-pop ${j.grade}`;
  void pop.offsetWidth;                         // restart the animation
  pop.classList.add("show");
}

function clearPop() {
  const pop = $("l3dPop");
  pop.className = "l3d-pop";
  pop.textContent = "";
}

/** Brake input `id` goes down at event time `ts`; the first one held presses the brake. */
function brakeDown(id, ts) {
  if (phase !== "run" || inputs.has(id)) return;
  inputs.add(id);
  if (inputs.size > 1) return;
  $("l3dBrake").classList.add("active");
  const j = judge.press(host.getT(ts));
  if (j) popup(j);
}

/** Brake input `id` comes up; the brake is released when the last one does. Returns whether it was held. */
function brakeUp(id, ts) {
  if (!inputs.delete(id)) return false;
  if (inputs.size || phase !== "run") return true;
  $("l3dBrake").classList.remove("active");
  const j = judge.release(host.getT(ts));
  if (j) popup(j);
  return true;
}

/** The lap ended: judge what's left, show the results. */
export function finish() {
  if (phase !== "run") return;
  judge.tick(Infinity);
  $("l3dBrake").classList.remove("active");
  $("l3dBrake").hidden = true;
  $("l3dLane").hidden = true;
  host.setCues(true);
  $("blkLearn3d").classList.remove("l3d-hard");
  phase = "results";
  renderResults();
  showPanel("results");
  $("l3dPracticeResults").querySelector("button").focus();
}

function renderResults() {
  const s = judge.score(), pct = Math.round(s.accuracy * 100), m = host.meta;
  const key = `l3d_best_${m.year}_${m.driver}_${m.session}_${m.track}_${settings.diff}`;
  const prev = parseFloat(host.store.get(key, ""));
  const isBest = !(prev >= pct);                // no previous best (NaN) counts as a new best
  if (isBest) host.store.set(key, String(pct));
  const cell = (r) => (r
    ? `${GRADE_LABEL[r.grade]}${r.dt != null ? ` <span class="l3d-dt">${r.dt > 0 ? "+" : "−"}${Math.abs(r.dt).toFixed(2)} s</span>` : ""}`
    : "—");
  $("l3dPracticeResults").innerHTML =
    `<h3>Lap complete · ${settings.diff === "hard" ? "Hard" : "Easy"} · ${settings.rate}×</h3>` +
    `<p class="l3d-big">${pct} % · ${s.points}/${s.max} pts · max combo ${s.maxCombo}</p>` +
    `<p>${isBest ? "New best!" : `Best: ${prev} %`} · + = late, − = early</p>` +
    `<table class="l3d-results"><thead><tr><th>Turn</th><th>Brake on</th><th>Brake off</th></tr></thead><tbody>` +
    judge.notes.map((n) =>
      `<tr><td>${esc(n.corner.short)}</td><td class="${n.press ? n.press.grade : ""}">${cell(n.press)}</td>` +
      `<td class="${n.release ? n.release.grade : ""}">${cell(n.release)}</td></tr>`).join("") +
    `</tbody></table>` +
    `<div class="l3d-practice-actions"><button type="button" class="l3d-primary" data-practice="retry">Retry</button>` +
    `<button type="button" data-practice="back">Back to Learn</button></div>`;
}

function close(restoreFocus = true) {
  phase = "off";
  inputs.clear();
  clearPop();
  showPanel(null);
  $("l3dLane").hidden = true;
  $("l3dScore").hidden = true;
  $("l3dBrake").hidden = true;
  $("l3dBrake").classList.remove("active");
  $("blkLearn3d").classList.remove("l3d-practicing", "l3d-hard");
  if (keysBefore) $("l3dKeys").textContent = keysBefore;
  if (host) {
    host.setCues(true);
    host.setPlaying(false);
    host.setRate(1);
    $("l3dPracticeBtn").hidden = false;
    if (restoreFocus) $("l3dPracticeBtn").focus();
  }
}

/** Keys while practice is open. Returns true when index.js should ignore the key. */
export function keydown(e) {
  if (phase === "off") return false;
  if (e.key === "Escape") {
    close();
    return true;
  }
  if (phase === "run" && BRAKE_KEYS.has(e.code)) {
    e.preventDefault();
    if (!e.repeat) brakeDown(e.code, e.timeStamp);
  }
  return true;                                  // Learn shortcuts are off while practice is open
}

export function keyup(e) {
  // Also after the lap ended: a Space held over the line mustn't click the focused Retry.
  if (BRAKE_KEYS.has(e.code) && brakeUp(e.code, e.timeStamp)) e.preventDefault();
}

// ---- controls (wired once; modules run after the DOM is parsed) ----
$("l3dPracticeBtn").addEventListener("click", open);
document.querySelectorAll("[data-diff]").forEach((b) => b.addEventListener("click", () => {
  settings.diff = b.dataset.diff;
  syncSetup();
}));
document.querySelectorAll("[data-prate]").forEach((b) => b.addEventListener("click", () => {
  settings.rate = +b.dataset.prate;
  syncSetup();
}));
$("l3dPracticeStart").addEventListener("click", () => phase === "setup" && begin());
$("l3dPracticeCancel").addEventListener("click", () => close());
$("l3dPracticeResults").addEventListener("click", (e) => {
  const act = e.target.closest("[data-practice]")?.dataset.practice;
  if (act === "retry") begin();
  else if (act === "back") close();
});
const brakeBtn = $("l3dBrake");
brakeBtn.addEventListener("pointerdown", (e) => {
  if (e.button !== 0) return;                   // primary button / touch / pen only
  e.preventDefault();
  brakeBtn.setPointerCapture(e.pointerId);
  brakeDown(`ptr${e.pointerId}`, e.timeStamp);
});
for (const ev of ["pointerup", "pointercancel"]) {
  brakeBtn.addEventListener(ev, (e) => brakeUp(`ptr${e.pointerId}`, e.timeStamp));
}
brakeBtn.addEventListener("contextmenu", (e) => e.preventDefault());
// Switching windows while holding = letting go of everything.
window.addEventListener("blur", () => [...inputs].forEach((id) => brakeUp(id)));
