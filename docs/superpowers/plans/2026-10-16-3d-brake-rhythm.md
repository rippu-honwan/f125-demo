# 3D Brake Rhythm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Practice mode inside 3D Learn. The pro's car drives by itself, and the player holds a brake key from each brake point until the pro releases. Each press and release is judged Perfect/Good/Miss, with early/late feedback.

**Architecture:** Frontend only, in `app/static/js/learn3d/`.
- `rhythm.js` is pure and tested with node. It builds the notes from the `/pro_lap` corners and judges press/release times.
- `practice.js` owns the practice UI: the setup card, note lane, pop-ups, score, BRAKE button and results. It talks to `index.js` through a small host API (clock, seek, play, rate, cue visibility).
- `index.js` keeps the clock. While practice runs it stops looping the lap and locks the Learn controls.
- `scene.js` gains `setCues(on)` to hide the brake/lift stripes in Hard.

**Tech Stack:** vanilla ES modules, Three.js 0.170.0, `node --test`, Playwright MCP for the browser check.

**Spec:** `docs/superpowers/specs/2026-10-16-3d-brake-rhythm-design.md`

**Conventions (read once):**
- Code, comments and commits are in English.
- Never modify `src/`, `tracks/` or `data/`. No backend changes.
- Run JS tests with `node --test tests/js/learn3d_core.test.mjs tests/js/learn3d_rhythm.test.mjs` (file paths, not the directory).
- Run Python tests with `python3 -m pytest -q`; they must stay green.
- End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Start the server with `python3 -m app.main` (port 8000) and stop it with `kill $(lsof -ti:8000)`.

---

## File map

| File | Change |
| --- | --- |
| `app/static/js/learn3d/rhythm.js` | new: `buildNotes`, `createJudge`, `dtMetres`, windows |
| `tests/js/learn3d_rhythm.test.mjs` | new: node tests |
| `app/static/js/learn3d/scene.js` | keep the cue stripes in a list; `setCues(on)` |
| `app/static/index.html` | importmap entries; Practice button; lane, pop-up, score, BRAKE button and practice card inside the stage |
| `app/static/css/main.css` | practice styles |
| `app/static/js/learn3d/practice.js` | new: the practice UI (attach/detach/frame/finish/keys) |
| `app/static/js/learn3d/index.js` | attach practice in Learn, finish at the lap end, lock controls, route keys, hide the HUD cue in Hard |

---

### Task 1: `rhythm.js` (pure notes and judging)

**Files:**
- Create: `app/static/js/learn3d/rhythm.js`
- Create: `tests/js/learn3d_rhythm.test.mjs`

- [ ] **Step 1: Write the failing tests**

Create `tests/js/learn3d_rhythm.test.mjs`:

```js
// Run: node --test tests/js/learn3d_rhythm.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildNotes, createJudge, dtMetres } from "../../app/static/js/learn3d/rhythm.js";

// 10 m per second, so lap distance d is reached at t = d / 10.
const P = { d: [0, 10, 20, 30], t: [0, 1, 2, 3] };
const CORNERS = [
  { short: "T2", action: "brake", brake_start_m: 20, brake_duration_s: 0.4 },
  { short: "K", action: "lift", brake_start_m: null, brake_duration_s: null, entry_m: 15 },
  { short: "T1", action: "brake", brake_start_m: 10, brake_duration_s: 0.5 },
];
const fresh = () => createJudge(buildNotes(P, CORNERS));   // notes: T1 1.0–1.5 s, T2 2.0–2.4 s

test("buildNotes keeps brake corners only, in lap order, with start and end times", () => {
  const notes = buildNotes(P, CORNERS);
  assert.deepEqual(notes.map((n) => n.corner.short), ["T1", "T2"]);
  assert.deepEqual(notes.map((n) => [n.start, n.end]), [[1, 1.5], [2, 2.4]]);
  assert.ok(notes.every((n) => n.press === null && n.release === null));
});

test("press: perfect within 0.05 s, good within 0.12 s, early or late", () => {
  let j = fresh();
  assert.equal(j.press(1.04).grade, "perfect");
  j = fresh();
  const early = j.press(0.94);
  assert.equal(early.grade, "good");
  assert.ok(early.dt < 0);
  j = fresh();
  assert.equal(j.press(1.11).grade, "good");
});

test("a press before the window is ignored and leaves the note open", () => {
  const j = fresh();
  assert.equal(j.press(0.8), null);
  assert.equal(j.notes[0].press, null);
  assert.equal(j.press(1.0).grade, "perfect");
});

test("tick misses a note nobody pressed, and the next press goes to the next note", () => {
  const j = fresh();
  const out = j.tick(1.2);
  assert.equal(out.length, 1);
  assert.equal(out[0].grade, "miss");
  assert.equal(j.notes[0].press.grade, "miss");
  assert.equal(j.press(1.3), null);              // too early for T2 (2.0)
  assert.equal(j.press(2.0).note.corner.short, "T2");
});

test("release: perfect within 0.10 s, good within 0.25 s, otherwise miss", () => {
  let j = fresh(); j.press(1.0);
  assert.equal(j.release(1.58).grade, "perfect");
  j = fresh(); j.press(1.0);
  assert.equal(j.release(1.7).grade, "good");
  j = fresh(); j.press(1.0);
  const early = j.release(1.2);
  assert.equal(early.grade, "miss");
  assert.ok(early.dt < 0);
});

test("no release judgement after a missed press", () => {
  const j = fresh();
  j.tick(1.2);
  assert.equal(j.release(1.5), null);
  assert.equal(j.notes[0].release, null);
});

test("holding past end + 0.25 s is a release miss", () => {
  const j = fresh();
  j.press(1.0);
  const out = j.tick(1.8);
  assert.deepEqual(out.map((o) => [o.kind, o.grade]), [["release", "miss"]]);
  assert.equal(j.holding, null);
});

test("score: points, max, accuracy and combo", () => {
  const j = fresh();
  j.press(1.0); j.release(1.5);                   // 2 + 2
  j.tick(2.2);                                    // T2 press miss
  assert.deepEqual(j.score(), { points: 4, max: 8, accuracy: 0.5, combo: 0, maxCombo: 2 });
  assert.equal(createJudge([]).score().accuracy, 1);
});

test("dtMetres converts a timing error to metres at a speed", () => {
  assert.equal(dtMetres(0.08, 270), 6);
});
```

- [ ] **Step 2: Run to verify they fail**

Run: `node --test tests/js/learn3d_rhythm.test.mjs`
Expected: FAIL with a module-not-found error for `rhythm.js`.

- [ ] **Step 3: Implement**

Create `app/static/js/learn3d/rhythm.js`:

```js
// Pure brake-rhythm logic for 3D Learn practice — no DOM, so it runs under
// `node --test tests/js/learn3d_rhythm.test.mjs`.
//
// A note is one brake corner: press at the pro's brake point, release when the pro
// lets go. Times are lap time (the playback clock), so slower playback is easier.
import { timeAtDistance } from "./core.js";

export const PRESS_WINDOWS = { perfect: 0.05, good: 0.12 };     // s, |t − note start|
export const RELEASE_WINDOWS = { perfect: 0.10, good: 0.25 };   // s, |t − note end|
const POINTS = { perfect: 2, good: 1, miss: 0 };

/** One note per brake corner, sorted by start: {corner, start, end, press: null, release: null}. */
export function buildNotes(points, corners) {
  return corners
    .filter((c) => c.action === "brake" && c.brake_start_m != null && c.brake_duration_s != null)
    .map((c) => {
      const start = timeAtDistance(points, c.brake_start_m);
      return { corner: c, start, end: start + c.brake_duration_s, press: null, release: null };
    })
    .sort((a, b) => a.start - b.start);
}

const gradeOf = (err, w) => (Math.abs(err) <= w.perfect ? "perfect" : Math.abs(err) <= w.good ? "good" : "miss");

/**
 * Judges presses and releases against `notes` (mutating their press/release fields).
 * Every method returns the new judgement(s) — {note, kind: "press"|"release", grade, dt} —
 * where dt > 0 means late and dt is null for a timeout miss.
 */
export function createJudge(notes) {
  let held = null, combo = 0, maxCombo = 0;
  const record = (note, kind, grade, dt) => {
    note[kind] = { grade, dt };
    if (grade === "miss") combo = 0; else maxCombo = Math.max(maxCombo, ++combo);
    return { note, kind, grade, dt };
  };
  return {
    notes,
    get holding() { return held; },
    /** Brake on at lap time t. Presses before a note's window are ignored (null). */
    press(t) {
      if (held) return null;
      const note = notes.find((n) => !n.press && n.start + PRESS_WINDOWS.good >= t);
      if (!note || t < note.start - PRESS_WINDOWS.good) return null;
      held = note;
      return record(note, "press", gradeOf(t - note.start, PRESS_WINDOWS), t - note.start);
    },
    /** Brake off at lap time t: judges the held note, if any. */
    release(t) {
      if (!held) return null;
      const note = held;
      held = null;
      return record(note, "release", gradeOf(t - note.end, RELEASE_WINDOWS), t - note.end);
    },
    /** Time passing: unpressed notes past their window and holds far past the end become misses. */
    tick(t) {
      const out = [];
      for (const n of notes) {
        if (!n.press && t > n.start + PRESS_WINDOWS.good) out.push(record(n, "press", "miss", null));
      }
      if (held && t > held.end + RELEASE_WINDOWS.good) {
        const n = held;
        held = null;
        out.push(record(n, "release", "miss", null));
      }
      return out;
    },
    score() {
      let points = 0;
      for (const n of notes) for (const k of ["press", "release"]) if (n[k]) points += POINTS[n[k].grade];
      const max = notes.length * 4;
      return { points, max, accuracy: max ? points / max : 1, combo, maxCombo };
    },
  };
}

/** A timing error (s) as metres at `speedKmh`. */
export const dtMetres = (dt, speedKmh) => (dt * speedKmh) / 3.6;
```

- [ ] **Step 4: Run to verify they pass**

Run: `node --test tests/js/learn3d_core.test.mjs tests/js/learn3d_rhythm.test.mjs`
Expected: `# pass 20`, `# fail 0` (11 core tests plus 9 rhythm tests).

- [ ] **Step 5: Commit**

```bash
git add app/static/js/learn3d/rhythm.js tests/js/learn3d_rhythm.test.mjs
git commit -m "feat(web): brake rhythm notes and judging for 3D practice

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `scene.setCues(on)`

**Files:**
- Modify: `app/static/js/learn3d/scene.js`

There is no automated test (WebGL); Task 6 checks it in the browser.

- [ ] **Step 1: Make `stripe` return its mesh and keep the cue stripes**

In `scene.js`, change the end of the `stripe` arrow function from:

```js
    m.rotation.y = Math.atan2(-(b.z - a.z), b.x - a.x);
    scene.add(m);
  };
  stripe(points.d[0], 0xffffff, 1.5);                         // start / finish
  for (const c of corners) {
    if (c.action === "brake") stripe(cuePoint(c), 0xff2e2e, 2);
    else if (c.action === "lift") stripe(cuePoint(c), 0xff9900, 1.2);
```

to:

```js
    m.rotation.y = Math.atan2(-(b.z - a.z), b.x - a.x);
    scene.add(m);
    return m;
  };
  stripe(points.d[0], 0xffffff, 1.5);                         // start / finish
  const cues = [];                                            // brake/lift stripes; hidden in Hard practice
  for (const c of corners) {
    if (c.action === "brake") cues.push(stripe(cuePoint(c), 0xff2e2e, 2));
    else if (c.action === "lift") cues.push(stripe(cuePoint(c), 0xff9900, 1.2));
```

- [ ] **Step 2: Expose `setCues`**

Replace `  return { update, dispose };` with:

```js
  /** Show or hide the brake/lift stripes (the start/finish line stays). */
  function setCues(on) {
    for (const m of cues) m.visible = on;
  }

  return { update, dispose, setCues };
```

- [ ] **Step 3: Syntax check and tests**

Run: `node --check app/static/js/learn3d/scene.js && node --test tests/js/learn3d_core.test.mjs tests/js/learn3d_rhythm.test.mjs`
Expected: no syntax error; `# pass 20`, `# fail 0`.

- [ ] **Step 4: Commit**

```bash
git add app/static/js/learn3d/scene.js
git commit -m "feat(web): scene.setCues hides brake and lift stripes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Practice markup and styles

**Files:**
- Modify: `app/static/index.html`
- Modify: `app/static/css/main.css`

- [ ] **Step 1: Version the new modules in the importmap**

In `app/static/index.html`, change the importmap line:

```html
  "./static/js/learn3d/scene.js": "./static/js/learn3d/scene.js?v=__BUILD__"
```

to:

```html
  "./static/js/learn3d/scene.js": "./static/js/learn3d/scene.js?v=__BUILD__",
  "./static/js/learn3d/rhythm.js": "./static/js/learn3d/rhythm.js?v=__BUILD__",
  "./static/js/learn3d/practice.js": "./static/js/learn3d/practice.js?v=__BUILD__"
```

- [ ] **Step 2: Practice layers inside the stage**

In the 3D block, right after the camera group's closing `</div>` (the `<div class="l3d-cams" …>` that holds the T-cam/Chase/Top buttons) and before the stage's closing `</div>`, add:

```html
            <div class="l3d-lane" id="l3dLane" hidden aria-hidden="true"><i class="l3d-hitline"></i></div>
            <div class="l3d-pop" id="l3dPop" role="status" aria-live="polite"></div>
            <div class="l3d-score" id="l3dScore" hidden></div>
            <button type="button" class="l3d-brake" id="l3dBrake" hidden>BRAKE</button>
            <div class="l3d-practice" id="l3dPractice" hidden>
              <div id="l3dPracticeSetup">
                <h3>Brake practice</h3>
                <p>Hold <kbd>Space</kbd> or <kbd>↓</kbd> (or the BRAKE button) from each brake point until the pro lets go. Both moments are judged.</p>
                <div class="l3d-views" role="group" aria-label="Difficulty">
                  <button type="button" data-diff="easy" aria-pressed="true">Easy · lane + stripes</button>
                  <button type="button" data-diff="hard" aria-pressed="false">Hard · road only</button>
                </div>
                <div class="l3d-views" role="group" aria-label="Practice speed">
                  <button type="button" data-prate="0.5" aria-pressed="false">0.5×</button>
                  <button type="button" data-prate="1" aria-pressed="true">1×</button>
                </div>
                <div class="l3d-practice-actions">
                  <button type="button" class="l3d-primary" id="l3dPracticeStart">Start</button>
                  <button type="button" id="l3dPracticeCancel">Cancel</button>
                </div>
              </div>
              <div id="l3dPracticeResults" hidden></div>
            </div>
```

- [ ] **Step 3: Practice button in the controls**

In `<div class="l3d-controls" id="l3dControls">`, after `<span class="l3d-time" id="l3dTime">0:00.0 / 0:00.0</span>`, add:

```html
          <button type="button" class="l3d-practice-btn" id="l3dPracticeBtn" hidden>🎯 Practice</button>
```

- [ ] **Step 4: Styles**

In `app/static/css/main.css`, add right before the line `@media (max-width: 900px) {` that follows the `.l3d-legend i` rule:

```css
.l3d-lane[hidden], .l3d-score[hidden], .l3d-brake[hidden], .l3d-practice[hidden], .l3d-practice-btn[hidden] { display: none; }
.l3d-lane { position: absolute; z-index: 1; left: 10px; right: 10px; bottom: 58px; height: 26px; background: rgba(0, 0, 0, 0.5); border-radius: 6px; overflow: hidden; }
.l3d-hitline { position: absolute; left: 15%; top: 0; bottom: 0; width: 3px; background: #fff; }
.l3d-note { position: absolute; top: 5px; height: 16px; border-radius: 4px; background: #ff2e2e; }
.l3d-note.held { background: #ffd400; }
.l3d-note.ok { background: #3ddc84; }
.l3d-note.miss { background: #666; }
.l3d-pop { position: absolute; z-index: 2; left: 15%; bottom: 92px; transform: translateX(-50%); font: 800 1.05rem var(--font-display); color: #fff; text-shadow: 0 2px 6px #000; pointer-events: none; opacity: 0; white-space: nowrap; }
.l3d-pop.perfect { color: #ffd400; }
.l3d-pop.good { color: #3ddc84; }
.l3d-pop.miss { color: #ff5a5a; }
.l3d-pop.show { animation: l3d-pop 0.9s ease-out; }
@keyframes l3d-pop { 0% { opacity: 0; transform: translate(-50%, 6px); } 15% { opacity: 1; transform: translate(-50%, 0); } 80% { opacity: 1; } 100% { opacity: 0; } }
@media (prefers-reduced-motion: reduce) { .l3d-pop.show { animation: none; opacity: 1; } }
.l3d-score { position: absolute; z-index: 1; left: 50%; top: 10px; transform: translateX(-50%); color: #fff; background: rgba(0, 0, 0, 0.55); padding: 6px 10px; border-radius: 8px; font: 700 0.85rem var(--font-body); font-variant-numeric: tabular-nums; }
.l3d-brake { position: absolute; z-index: 2; right: 10px; bottom: 96px; width: 92px; height: 92px; border-radius: 50%; border: 0; background: var(--accent); color: #fff; font: 800 0.95rem var(--font-display); cursor: pointer; touch-action: none; user-select: none; -webkit-user-select: none; }
.l3d-brake.active { transform: scale(0.94); filter: brightness(1.3); }
.l3d-practice { position: absolute; z-index: 3; inset: 0; display: grid; place-items: center; background: rgba(0, 0, 0, 0.55); }
.l3d-practice > div { background: var(--surface); color: var(--text); border-radius: 12px; padding: 16px 18px; width: min(92%, 460px); max-height: 92%; overflow-y: auto; display: grid; gap: 10px; }
.l3d-practice h3 { margin: 0; font: 700 1.05rem var(--font-display); }
.l3d-practice p { margin: 0; font-size: 0.85rem; color: var(--text-dim); }
.l3d-practice .l3d-big { font: 700 1.3rem var(--font-display); color: var(--text); }
.l3d-practice-actions { display: flex; gap: 8px; }
.l3d-practice-actions button, .l3d-practice-btn { border: 1px solid var(--border-strong); background: var(--surface-2); color: var(--text); border-radius: 8px; padding: 6px 12px; cursor: pointer; font: 600 0.85rem var(--font-body); }
.l3d-practice-actions .l3d-primary { background: var(--accent); border-color: var(--accent); color: #fff; }
.l3d-practice-actions button:focus-visible, .l3d-practice-btn:focus-visible, .l3d-brake:focus-visible { outline: none; box-shadow: 0 0 0 3px rgba(232, 0, 45, 0.45); }
.l3d-results { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.l3d-results th, .l3d-results td { padding: 3px 6px; text-align: left; border-bottom: 1px solid var(--border); }
.l3d-results .perfect { color: #ffd400; }
.l3d-results .good { color: #3ddc84; }
.l3d-results .miss { color: #ff5a5a; }
.l3d-dt { color: var(--text-dim); font-variant-numeric: tabular-nums; }
.l3d.l3d-practicing .l3d-controls, .l3d.l3d-practicing .l3d-corners, .l3d.l3d-practicing .l3d-telemetry { pointer-events: none; opacity: 0.45; }
.l3d.l3d-hard .l3d-corners, .l3d.l3d-hard .l3d-telemetry { visibility: hidden; }   /* brake zones there would be hints */
```

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest -q && node --test tests/js/learn3d_core.test.mjs tests/js/learn3d_rhythm.test.mjs`
Expected: all Python tests pass; `# pass 20`, `# fail 0`.

- [ ] **Step 6: Commit**

```bash
git add app/static/index.html app/static/css/main.css
git commit -m "feat(web): practice lane, BRAKE button, setup and results markup

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `practice.js` (practice UI)

**Files:**
- Create: `app/static/js/learn3d/practice.js`

There is no automated test (DOM). The judging logic it calls is covered by Task 1, and Task 6 checks the UI in the browser.

- [ ] **Step 1: Create the module**

Create `app/static/js/learn3d/practice.js`:

```js
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
// host: {points, corners, meta, store, getT, seek, setPlaying, setRate, setCues}, set by attach().
let host = null;
let phase = "off";                              // off | setup | run | results
let judge = null, bars = [], keysBefore = "", lastScore = "";
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
  if (phase !== "off") close();
  host = null;
  $("l3dPracticeBtn").hidden = true;
}

export const isRunning = () => phase === "run";
/** False only while a Hard lap runs: index.js hides the HUD's "brake in N m" cue then. */
export const hintsOn = () => phase !== "run" || settings.diff === "easy";

function open() {
  if (!host || phase !== "off") return;
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
  if (j.dt != null) {
    if (Math.abs(j.dt) < 0.005) text += " · on time";
    else {
      const speed = stateAt(host.points, j.kind === "press" ? j.note.start : j.note.end).speed;
      text += ` · ${j.dt > 0 ? "late" : "early"} ${Math.abs(j.dt).toFixed(2)} s ≈ ${Math.round(Math.abs(dtMetres(j.dt, speed)))} m`;
    }
  }
  const pop = $("l3dPop");
  pop.textContent = text;
  pop.className = `l3d-pop ${j.grade}`;
  void pop.offsetWidth;                         // restart the animation
  pop.classList.add("show");
}

function brakeDown() {
  if (phase !== "run") return;
  $("l3dBrake").classList.add("active");
  const j = judge.press(host.getT());
  if (j) popup(j);
}

function brakeUp() {
  if (phase !== "run") return;
  $("l3dBrake").classList.remove("active");
  const j = judge.release(host.getT());
  if (j) popup(j);
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

function close() {
  phase = "off";
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
    if (!e.repeat) brakeDown();
    return true;
  }
  return phase === "run";                       // Learn shortcuts are off while a lap runs
}

export function keyup(e) {
  if (phase === "run" && BRAKE_KEYS.has(e.code)) {
    e.preventDefault();
    brakeUp();
  }
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
$("l3dPracticeCancel").addEventListener("click", close);
$("l3dPracticeResults").addEventListener("click", (e) => {
  const act = e.target.closest("[data-practice]")?.dataset.practice;
  if (act === "retry") begin();
  else if (act === "back") close();
});
const brakeBtn = $("l3dBrake");
brakeBtn.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  brakeBtn.setPointerCapture(e.pointerId);
  brakeDown();
});
for (const ev of ["pointerup", "pointercancel"]) brakeBtn.addEventListener(ev, brakeUp);
window.addEventListener("blur", brakeUp);       // switching windows while holding = letting go
```

- [ ] **Step 2: Syntax check and tests**

Run: `node --check app/static/js/learn3d/practice.js && node --test tests/js/learn3d_core.test.mjs tests/js/learn3d_rhythm.test.mjs`
Expected: no syntax error; `# pass 20`, `# fail 0`.

- [ ] **Step 3: Commit**

```bash
git add app/static/js/learn3d/practice.js
git commit -m "feat(web): brake practice UI — setup, lane, pop-ups, results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Wire practice into `index.js`

**Files:**
- Modify: `app/static/js/learn3d/index.js`

- [ ] **Step 1: Import**

After the `import { … } from "./panels.js";` statement, add:

```js
import * as practice from "./practice.js";
```

- [ ] **Step 2: Attach in Learn, detach in Compare**

In `start()`, change:

```js
  $("l3dScrub").max = String(Math.round(end * 10));   // 0.1 s per step
```

to:

```js
  $("l3dScrub").max = String(Math.round(end * 10));   // 0.1 s per step
  if (Y) practice.detach();
  else practice.attach({
    points: P, corners: data.corners, meta: data.meta, store,
    getT: () => S.t, seek, setPlaying, setRate,
    setCues: (on) => { if (S.scene) S.scene.setCues(on); },
  });
```

In the same function, change the corner-list pick callback's first line:

```js
  S.list = createCornerList($("l3dCorners"), data.corners, (c) => {
    const [from, to] = cornerSeekRange(c);
```

to:

```js
  S.list = createCornerList($("l3dCorners"), data.corners, (c) => {
    if (practice.isRunning()) return;
    const [from, to] = cornerSeekRange(c);
```

and change the telemetry line:

```js
  S.tele = createTelemetry($("l3dTelemetry"), P, (d) => seek(timeAt(d)), Y, data.meta.driver);
```

to:

```js
  S.tele = createTelemetry($("l3dTelemetry"), P, (d) => practice.isRunning() || seek(timeAt(d)), Y, data.meta.driver);
```

- [ ] **Step 3: End the lap instead of looping while practising**

In `frame`, replace:

```js
  if (S.playing) {
    S.t += dt * S.rate;
    if (S.stopAt != null && S.t >= S.stopAt) { S.t = S.stopAt; S.stopAt = null; setPlaying(false); }
    else if (S.t >= S.end) S.t -= S.end;                                // loop once both cars finish
  }
  draw();
```

with:

```js
  if (S.playing) {
    S.t += dt * S.rate;
    if (practice.isRunning() && S.t >= S.end) { S.t = S.end; setPlaying(false); practice.finish(); }
    else if (S.stopAt != null && S.t >= S.stopAt) { S.t = S.stopAt; S.stopAt = null; setPlaying(false); }
    else if (S.t >= S.end) S.t -= S.end;                                // loop once both cars finish
  }
  practice.frame(S.t);
  draw();
```

- [ ] **Step 4: No "brake in N m" cue during a Hard lap**

In `draw()`, change:

```js
    renderHud($("l3dHudTop"), $("l3dHudBottom"), S.data.meta, s, nextCue(S.data.corners, s.d, P.d[P.d.length - 1]));
```

to:

```js
    const cue = practice.hintsOn() ? nextCue(S.data.corners, s.d, P.d[P.d.length - 1]) : null;
    renderHud($("l3dHudTop"), $("l3dHudBottom"), S.data.meta, s, cue);
```

- [ ] **Step 5: Leave practice on unload**

In `unload()`, change:

```js
  if (!view) return;
  cancelAnimationFrame(view.raf);
```

to:

```js
  if (!view) return;
  practice.detach();               // before the scene and view go away: it restores cues and controls
  cancelAnimationFrame(view.raf);
```

- [ ] **Step 6: Lock the Learn controls while a lap runs and route keys**

In the controls section at the bottom, replace:

```js
document.querySelectorAll("[data-rate]").forEach((b) => b.addEventListener("click", () => view && setRate(+b.dataset.rate)));
```

with:

```js
document.querySelectorAll("[data-rate]").forEach((b) => b.addEventListener("click", () => view && !practice.isRunning() && setRate(+b.dataset.rate)));
```

Replace:

```js
$("l3dPlay").addEventListener("click", () => view && setPlaying(!view.playing));
$("l3dScrub").addEventListener("input", (e) => view && seek(+e.target.value / 10));
```

with:

```js
$("l3dPlay").addEventListener("click", () => view && !practice.isRunning() && setPlaying(!view.playing));
$("l3dScrub").addEventListener("input", (e) => view && !practice.isRunning() && seek(+e.target.value / 10));
```

In the `keydown` handler, change:

```js
  if (e.ctrlKey || e.metaKey || e.altKey) return;                      // leave browser shortcuts alone
  const tag = e.target.tagName;
```

to:

```js
  if (e.ctrlKey || e.metaKey || e.altKey) return;                      // leave browser shortcuts alone
  if (practice.keydown(e)) return;                                     // practice owns Space/↓/Esc
  const tag = e.target.tagName;
```

Right after the `keydown` listener's closing `});`, add:

```js
document.addEventListener("keyup", (e) => { if (view) practice.keyup(e); });
```

- [ ] **Step 7: Syntax check and tests**

Run: `node --check app/static/js/learn3d/index.js && node --test tests/js/learn3d_core.test.mjs tests/js/learn3d_rhythm.test.mjs && python3 -m pytest -q`
Expected: no syntax error; `# pass 20`, `# fail 0`; all Python tests pass.

- [ ] **Step 8: Commit**

```bash
git add app/static/js/learn3d/index.js
git commit -m "feat(web): wire brake practice into 3D Learn

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Browser check

**Files:** fixes only, if any check fails.

- [ ] **Step 1: Start the server** in the background with `python3 -m app.main`. Wait until `curl -s localhost:8000/health` returns JSON.

- [ ] **Step 2: Walk through it with Playwright MCP**

1. Open `http://localhost:8000/`, click **3D Learn**, choose VER / 2025 / Q / suzuka and load. Confirm **🎯 Practice** is visible.
2. Click Practice. Confirm the setup card shows and Easy and 1× are pressed. Click **Start**. Confirm all of these:
   - the lane is visible with note bars;
   - `#l3dScore` shows "0 pts · combo 0";
   - the BRAKE button is visible;
   - the controls are dimmed;
   - the play button doesn't respond.
3. Hit the first note. Run `page.evaluate` with code that polls the viewer clock through the lane: when the first `.l3d-note` bar's `left` style first drops to ≤ 15 %, dispatch `keydown` with `{code: "Space", key: " "}`; 0.3 s later, dispatch `keyup`. Then confirm `#l3dPop` shows PERFECT or GOOD, with "early"/"late"/"on time".
4. Wait for the lap to end; at 1× Suzuka is about 87 s, so use several `browser_wait_for` calls of up to 30 s. Confirm the results card shows a percentage, a table with one row per brake note, and Retry / Back to Learn.
5. Click **Back to Learn**. Confirm:
   - the lane, score and BRAKE button are hidden;
   - the controls work again (Play toggles);
   - the keys hint reads the Learn text.
6. Open Practice again, choose **Hard**, and Start. Confirm:
   - the lane is hidden;
   - the HUD top has no "BRAKE in" cue;
   - telemetry and corners are hidden.

   Press Esc and confirm everything is restored.
7. Load **3D Compare** with `data/suzuka_sample.csv` (see the previous plan for the flow). Confirm the Practice button is hidden.
8. Resize to 390 px wide, open Practice and Start. Confirm:
   - the BRAKE button is visible;
   - the lane doesn't overlap the camera buttons badly;
   - there's no horizontal scroll (`document.documentElement.scrollWidth === 390`).
9. Check the console for errors. The favicon 404 is expected.

If a check fails, fix it in the relevant file, re-run the tests and commit with a `fix(web): …` message.

- [ ] **Step 3: Stop the server** with `kill $(lsof -ti:8000)`.
