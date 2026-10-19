// Entry point for the 3D viewer. main.js (a classic script) drives it via
//   window.Learn3D.load(url, onPickAvailable, signal?) -> fetch /pro_lap and start (3D Learn)
//   window.Learn3D.show(payload)                  -> start from a /analyze compare_3d payload (3D Compare)
//   window.Learn3D.unload()                       -> stop and free WebGL
// load() resolves once the viewer is up (or a "no data" message is shown) and
// rejects only on network errors, which main.js reports in its error banner.
import { stateAt, timeAtDistance, nextCue, cornerSeekRange, fmtTime, youPoints, gapAt, lineHeadings } from "./core.js";
import {
  esc, renderHud, renderCompareHud, renderDriveHud, createMinimap, createTelemetry, createCornerList, PRO_COLOR, YOU_COLOR,
} from "./panels.js";
import * as practice from "./practice.js";
import * as drive from "./drive.js";

const $ = (id) => document.getElementById(id);
const store = {
  get(k, fallback) { try { return localStorage.getItem(k) ?? fallback; } catch { return fallback; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode: ignore */ } },
};
const CAM_KEYS = { 1: "tcam", 2: "chase", 3: "top" };
const LEARN_CAR_COLOR = 0xe8002d;
const hex = (css) => parseInt(css.slice(1), 16);
/** Practice or Drive is open: Learn's playback, scrub and seeks stay locked. */
const locked = () => practice.isOpen() || drive.isOpen();

let view = null;      // the running session, or null
let loadSeq = 0;      // guards against overlapping load() calls

async function load(url, onPickAvailable, signal) {
  unload();
  const seq = ++loadSeq;
  const res = await fetch(url, { signal });
  const body = await res.json().catch(() => ({}));
  if (seq !== loadSeq) return;
  if (!res.ok) { showMessage(body, res.status, onPickAvailable); return; }
  if (!body.points) throw new Error("Malformed /pro_lap response");
  await start(body, seq);
}

/** 3D Compare: `payload` is the /analyze compare_3d body ({pro, you, ...}). */
async function show(payload) {
  unload();
  const seq = ++loadSeq;
  if (!payload || !payload.pro || !payload.pro.points || !payload.you) throw new Error("Malformed 3D compare response");
  await start({ ...payload.pro, you: youPoints(payload.pro.points, payload.you) }, seq);
}

function showMessage(body, status, onPick) {
  $("l3dMain").hidden = true;
  $("l3dControls").hidden = true;
  const avail = Array.isArray(body.available) ? body.available : [];
  const text = body.detail === "unknown track" ? "This track isn't supported."
    : status === 404 ? "No lap data for this driver / track / session."
    : status === 422 ? "That driver, session or year isn't valid."
    : `Couldn't load the lap (HTTP ${status}).`;
  const msg = $("l3dMsg");
  msg.innerHTML = `<p>${esc(text)}</p>` + (avail.length
    ? `<p>Laps you can load right away:</p><div class="l3d-chips">${avail.map((a, i) =>
      `<button type="button" class="l3d-chip" data-i="${i}">${a.year} · ${esc(a.driver)} · ${esc(a.session)} · ${esc(a.track)}</button>`).join("")}</div>`
    : "");
  msg.querySelectorAll(".l3d-chip").forEach((b) => b.addEventListener("click", () => onPick(avail[+b.dataset.i])));
  msg.hidden = false;
}

const KEYS_LEARN = "Space play/pause · ← → ±2 s · 1/2/3 camera";
const KEYS_COMPARE = KEYS_LEARN + " · F switch car";
const NOTE_OWN_LINE = "Your line is matched to the track shape (about ±1 m). The pro line is FastF1's smoothed position data, not the exact line.";
const NOTE_PRO_LINE = "Your line isn't available for this lap; your car follows the pro's line.";

async function start(data, seq) {
  const P = data.points, Y = data.you || null;   // Y: your lap (youPoints) in 3D Compare
  P.heading = lineHeadings(P);                   // stateAt and the scene read it: smooth through stalled points
  const ownLine = !!Y && !!Y.ownLine;           // the server sent your own line (youPoints swapped x/y in)
  if (Y && !ownLine) Y.heading = P.heading;      // your lap runs on the pro's line
  $("l3dMsg").hidden = true;
  $("l3dMain").hidden = false;
  $("l3dControls").hidden = false;
  $("l3dFollow").hidden = !Y;
  $("l3dTitle").textContent = Y ? "3D Compare" : "3D Pro Lap";
  $("l3dKeys").textContent = Y ? KEYS_COMPARE : KEYS_LEARN;
  $("l3dLineNote").textContent = ownLine ? NOTE_OWN_LINE : NOTE_PRO_LINE;

  $("l3dStage").classList.remove("l3d-nogl");
  let scene = null;
  try {
    const { createScene } = await import("./scene.js");
    if (seq !== loadSeq) return;
    // Your own line: both cars on their real positions, the pro drawn see-through so the two can overlap.
    // Otherwise both share the pro's line, side by side.
    const cars = !Y ? [{ color: LEARN_CAR_COLOR, lane: 0 }, { color: hex(YOU_COLOR), lane: 0 }]   // car 1 = you, in Drive only
      : ownLine ? [{ color: hex(PRO_COLOR), lane: 0 }, { color: hex(YOU_COLOR), lane: 0 }]
      : [{ color: hex(PRO_COLOR), lane: -1.6 }, { color: hex(YOU_COLOR), lane: 1.6 }];
    scene = createScene($("l3dStage"), P, data.corners, cars, ownLine ? Y : null);
  } catch (err) {
    console.warn("3D view unavailable:", err);
    if (seq === loadSeq) $("l3dStage").classList.add("l3d-nogl");
  }
  if (seq !== loadSeq) { if (scene) scene.dispose(); return; }

  const end = Math.max(P.t[P.t.length - 1], Y ? Y.t[Y.t.length - 1] : 0);   // both cars finish
  const S = view = {
    data, scene, ghostPro: ownLine, t: 0, end, playing: false, rate: 1, stopAt: null, raf: 0, last: 0,
    cam: store.get("l3d_cam", "tcam"), follow: Y && store.get("l3d_follow", "you") !== "pro" ? "you" : "pro",
  };
  // Distance-based seeks (telemetry click, corner replay) use the followed car's clock.
  const timeAt = (d) => timeAtDistance(S.follow === "you" && Y ? Y : P, d);
  S.minimap = createMinimap($("l3dMinimap"), P, Y ? [PRO_COLOR, YOU_COLOR] : undefined);
  S.tele = createTelemetry($("l3dTelemetry"), P, (d) => locked() || seek(timeAt(d)), Y, data.meta.driver);
  S.list = createCornerList($("l3dCorners"), data.corners, (c) => {
    if (locked()) return;
    const [from, to] = cornerSeekRange(c);
    seek(timeAt(from));
    S.stopAt = timeAt(to);
    setRate(0.5);
    setPlaying(true);
  }, !!Y);
  $("l3dScrub").max = String(Math.round(end * 10));   // 0.1 s per step
  $("l3dLineBtn").hidden = !ownLine || !scene;
  $("l3dLineNote").hidden = !Y || !scene;         // no 3D view, no line to explain
  markPressed($("l3dLineBtn"), false);
  if (Y) {
    practice.detach();
    drive.detach();
  } else {
    practice.attach({
      points: P, corners: data.corners, meta: data.meta, store,
      // Judge an input at its own event time, not the last frame's (up to a frame early otherwise).
      getT: (ts) => (S.playing && ts && S.last
        ? Math.min(S.end, S.t + Math.min(0.1, Math.max(0, (ts - S.last) / 1000)) * S.rate) : S.t),
      seek, setPlaying, setRate,
      setCues: (on) => { if (S.scene) S.scene.setCues(on); },
      blocked: drive.isOpen,
    });
    if (S.scene) drive.attach({
      points: P, corners: data.corners, meta: data.meta, store, setPlaying,
      setLine: (on) => { if (S.scene) S.scene.setLine(on); },
      blocked: practice.isOpen,
    });
    else drive.detach();                                               // no WebGL: nothing to drive in
  }
  setView(store.get("l3d_view", "telemetry"));
  setCam(S.cam);
  if (Y) setFollow(S.follow);
  setRate(1);
  setPlaying(false);
  S.raf = requestAnimationFrame(frame);
}

function frame(now) {
  const S = view;
  if (!S) return;
  S.raf = requestAnimationFrame(frame);
  const dt = S.last ? Math.min(0.1, (now - S.last) / 1000) : 0;
  S.last = now;
  if (drive.isOpen()) drive.frame(dt);
  else if (S.playing) {
    S.t += dt * S.rate;
    if (practice.isRunning() && S.t >= S.end) { S.t = S.end; setPlaying(false); practice.finish(); }
    else if (S.stopAt != null && S.t >= S.stopAt) { S.t = S.stopAt; S.stopAt = null; setPlaying(false); }
    else if (S.t >= S.end) S.t -= S.end;                                // loop once both cars finish
  }
  practice.frame(S.t);
  draw();
}

function draw() {
  const S = view, P = S.data.points, Y = S.data.you;
  const dv = drive.view();
  if (S.scene) S.scene.setGhost(0, !!dv || S.ghostPro);
  if (dv) {                        // Drive: the pro is the ghost, the camera follows you
    const g = stateAt(P, dv.t);
    if (S.scene) S.scene.update([dv.ghost ? g : null, dv.you], S.cam, 1);
    renderDriveHud($("l3dHudTop"), $("l3dHudBottom"), S.data.meta, dv);
    S.minimap.set(dv.you);
    S.tele.set(dv.d);
    S.list.set(dv.d);
    return;
  }
  const s = stateAt(P, S.t);       // a finished car waits at the line (stateAt clamps t)
  let at = s;                      // the followed car: drives the cursor and corner highlight
  if (Y) {
    const y = stateAt(Y, S.t);
    // Crossed the line, waiting for the other car. 0.05 s slack: the scrub bar works in 0.1 s steps.
    y.done = S.t >= Y.t[Y.t.length - 1] - 0.05;
    s.done = S.t >= P.t[P.t.length - 1] - 0.05;
    const youFirst = S.follow === "you";
    at = youFirst ? y : s;
    if (S.scene) S.scene.update([s, y], S.cam, youFirst ? 1 : 0);
    renderCompareHud($("l3dHudTop"), $("l3dHudBottom"), S.data.meta, y, s, gapAt(P, Y, y.d));
    S.minimap.set(s, y);
  } else {
    if (S.scene) S.scene.update([s], S.cam);
    const cue = practice.hintsOn() ? nextCue(S.data.corners, s.d, P.d[P.d.length - 1]) : null;
    renderHud($("l3dHudTop"), $("l3dHudBottom"), S.data.meta, s, cue);
    S.minimap.set(s);
  }
  S.tele.set(at.d);
  S.list.set(at.d);
  const time = `${fmtTime(S.t)} / ${fmtTime(S.end)}`;
  const scrub = $("l3dScrub");
  if (!scrub.matches(":active")) scrub.value = String(Math.round(S.t * 10));   // don't fight a drag
  if (time !== S.lastTime) {                         // only when the shown tenths change
    S.lastTime = time;
    scrub.setAttribute("aria-valuetext", time);
    $("l3dTime").textContent = time;
  }
}

function seek(t) {
  view.t = Math.min(Math.max(0, t), view.end);
  view.stopAt = null;
}

/** Keep a toggle button's visual .active state and its aria-pressed in sync. */
function markPressed(b, on) {
  b.classList.toggle("active", on);
  b.setAttribute("aria-pressed", String(on));
}

function setPlaying(on) {
  view.playing = on;
  $("l3dPlay").textContent = on ? "⏸" : "▶";
  $("l3dPlay").setAttribute("aria-label", on ? "Pause" : "Play");
}

function setRate(r) {
  view.rate = r;
  document.querySelectorAll("[data-rate]").forEach((b) => markPressed(b, +b.dataset.rate === r));
}

function setCam(cam) {
  view.cam = cam;
  store.set("l3d_cam", cam);
  document.querySelectorAll("[data-cam]").forEach((b) => markPressed(b, b.dataset.cam === cam));
}

function setView(name) {
  $("l3dMain").dataset.view = name;
  store.set("l3d_view", name);
  document.querySelectorAll(".l3d-view").forEach((b) => markPressed(b, b.dataset.view === name));
}

function setFollow(who) {
  view.follow = who;
  view.stopAt = null;              // a corner replay's stop time was on the other car's clock
  store.set("l3d_follow", who);
  document.querySelectorAll("[data-follow]").forEach((b) => markPressed(b, b.dataset.follow === who));
}

function unload() {
  loadSeq++;
  if (!view) return;
  practice.detach();               // before the scene and view go away: they restore cues and controls
  drive.detach();
  cancelAnimationFrame(view.raf);
  if (view.scene) view.scene.dispose();
  $("l3dStage").classList.remove("l3d-nogl");
  view = null;
  // Back to the 3D Learn head, so an error screen after Compare doesn't keep Compare's labels.
  $("l3dFollow").hidden = true;
  $("l3dTitle").textContent = "3D Pro Lap";
  $("l3dKeys").textContent = KEYS_LEARN;
  $("l3dLineNote").hidden = true;
  $("l3dLineBtn").hidden = true;
}

// ---- controls (wired once; modules run after the DOM is parsed) ----
document.querySelectorAll(".l3d-view").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
document.querySelectorAll("[data-cam]").forEach((b) => b.addEventListener("click", () => view && setCam(b.dataset.cam)));
$("l3dLineBtn").addEventListener("click", () => {
  if (!view || !view.scene) return;
  const on = $("l3dLineBtn").getAttribute("aria-pressed") !== "true";
  markPressed($("l3dLineBtn"), on);
  view.scene.setLine(on);
});
document.querySelectorAll("[data-rate]").forEach((b) => b.addEventListener("click", () => view && !locked() && setRate(+b.dataset.rate)));
document.querySelectorAll("[data-follow]").forEach((b) => b.addEventListener("click", () => view && view.data.you && setFollow(b.dataset.follow)));
$("l3dPlay").addEventListener("click", () => view && !locked() && setPlaying(!view.playing));
$("l3dScrub").addEventListener("input", (e) => view && !locked() && seek(+e.target.value / 10));
document.addEventListener("keydown", (e) => {
  if (!view || $("blkLearn3d").offsetParent === null) return;          // block not visible
  if (e.ctrlKey || e.metaKey || e.altKey) return;                      // leave browser shortcuts alone
  if (practice.keydown(e) || drive.keydown(e)) return;                 // an open mode owns its keys
  const tag = e.target.tagName;
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(tag)) return;                   // typing / native range keys
  if (e.code === "Space") {
    if (tag === "BUTTON") return;                                      // a focused button handles Space itself
    e.preventDefault();
    if (!e.repeat) setPlaying(!view.playing);
  } else if (e.key === "ArrowLeft") seek(view.t - 2);
  else if (e.key === "ArrowRight") seek(view.t + 2);
  else if (CAM_KEYS[e.key]) setCam(CAM_KEYS[e.key]);
  else if ((e.key === "f" || e.key === "F") && view.data.you) setFollow(view.follow === "you" ? "pro" : "you");
});
document.addEventListener("keyup", (e) => {
  if (!view) return;
  practice.keyup(e);
  drive.keyup(e);
});

window.Learn3D = { load, show, unload };
