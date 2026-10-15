// Entry point for the 3D Learn mode. main.js (a classic script) drives it via
//   window.Learn3D.load(url, onPickAvailable, signal?) -> fetch /pro_lap and start
//   window.Learn3D.unload()                       -> stop and free WebGL
// load() resolves once the viewer is up (or a "no data" message is shown) and
// rejects only on network errors, which main.js reports in its error banner.
import { stateAt, timeAtDistance, nextCue, cornerSeekRange, fmtTime } from "./core.js";
import { esc, renderHud, createMinimap, createTelemetry, createCornerList } from "./panels.js";

const $ = (id) => document.getElementById(id);
const store = {
  get(k, fallback) { try { return localStorage.getItem(k) ?? fallback; } catch { return fallback; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode: ignore */ } },
};
const CAM_KEYS = { 1: "tcam", 2: "chase", 3: "top" };

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

async function start(data, seq) {
  const P = data.points;
  $("l3dMsg").hidden = true;
  $("l3dMain").hidden = false;
  $("l3dControls").hidden = false;

  $("l3dStage").classList.remove("l3d-nogl");
  let scene = null;
  try {
    const { createScene } = await import("./scene.js");
    if (seq !== loadSeq) return;
    scene = createScene($("l3dStage"), P, data.corners);
  } catch (err) {
    console.warn("3D view unavailable:", err);
    if (seq === loadSeq) $("l3dStage").classList.add("l3d-nogl");
  }
  if (seq !== loadSeq) { if (scene) scene.dispose(); return; }

  const S = view = {
    data, scene, t: 0, playing: false, rate: 1, stopAt: null, raf: 0, last: 0,
    cam: store.get("l3d_cam", "tcam"),
  };
  S.minimap = createMinimap($("l3dMinimap"), P);
  S.tele = createTelemetry($("l3dTelemetry"), P, (d) => seek(timeAtDistance(P, d)));
  S.list = createCornerList($("l3dCorners"), data.corners, (c) => {
    const [from, to] = cornerSeekRange(c);
    seek(timeAtDistance(P, from));
    S.stopAt = timeAtDistance(P, to);
    setRate(0.5);
    setPlaying(true);
  });
  $("l3dScrub").max = String(Math.round(P.t[P.t.length - 1] * 10));   // 0.1 s per step
  setView(store.get("l3d_view", "telemetry"));
  setCam(S.cam);
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
  const lapTime = S.data.points.t[S.data.points.t.length - 1];
  if (S.playing) {
    S.t += dt * S.rate;
    if (S.stopAt != null && S.t >= S.stopAt) { S.t = S.stopAt; S.stopAt = null; setPlaying(false); }
    else if (S.t >= lapTime) S.t -= lapTime;                            // loop the lap
  }
  draw();
}

function draw() {
  const S = view, P = S.data.points;
  const s = stateAt(P, S.t);
  if (S.scene) S.scene.update(s, S.cam);
  renderHud($("l3dHudTop"), $("l3dHudBottom"), S.data.meta, s, nextCue(S.data.corners, s.d, P.d[P.d.length - 1]));
  S.minimap.set(s);
  S.tele.set(s.d);
  S.list.set(s.d);
  const time = `${fmtTime(S.t)} / ${fmtTime(P.t[P.t.length - 1])}`;
  const scrub = $("l3dScrub");
  if (!scrub.matches(":active")) scrub.value = String(Math.round(S.t * 10));   // don't fight a drag
  if (time !== S.lastTime) {                         // only when the shown tenths change
    S.lastTime = time;
    scrub.setAttribute("aria-valuetext", time);
    $("l3dTime").textContent = time;
  }
}

function seek(t) {
  const P = view.data.points;
  view.t = Math.min(Math.max(0, t), P.t[P.t.length - 1]);
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

function unload() {
  loadSeq++;
  if (!view) return;
  cancelAnimationFrame(view.raf);
  if (view.scene) view.scene.dispose();
  $("l3dStage").classList.remove("l3d-nogl");
  view = null;
}

// ---- controls (wired once; modules run after the DOM is parsed) ----
document.querySelectorAll(".l3d-view").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
document.querySelectorAll("[data-cam]").forEach((b) => b.addEventListener("click", () => view && setCam(b.dataset.cam)));
document.querySelectorAll("[data-rate]").forEach((b) => b.addEventListener("click", () => view && setRate(+b.dataset.rate)));
$("l3dPlay").addEventListener("click", () => view && setPlaying(!view.playing));
$("l3dScrub").addEventListener("input", (e) => view && seek(+e.target.value / 10));
document.addEventListener("keydown", (e) => {
  if (!view || $("blkLearn3d").offsetParent === null) return;          // block not visible
  if (e.ctrlKey || e.metaKey || e.altKey) return;                      // leave browser shortcuts alone
  const tag = e.target.tagName;
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(tag)) return;                   // typing / native range keys
  if (e.code === "Space") {
    if (tag === "BUTTON") return;                                      // a focused button handles Space itself
    e.preventDefault();
    if (!e.repeat) setPlaying(!view.playing);
  } else if (e.key === "ArrowLeft") seek(view.t - 2);
  else if (e.key === "ArrowRight") seek(view.t + 2);
  else if (CAM_KEYS[e.key]) setCam(CAM_KEYS[e.key]);
});

window.Learn3D = { load, unload };
