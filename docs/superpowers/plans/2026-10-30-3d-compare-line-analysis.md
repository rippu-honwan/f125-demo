# 3D Compare Line Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When **Line** is on in 3D Compare, the Corners panel shows how your line compares with the pro's at each corner's entry, apex and exit: `≈`, `wide` / `tight`, or `wide+` / `tight+`. Picking a corner then switches to the Top camera.

**Architecture:**
- **No server or payload change.** Everything is computed in the browser from data the 3D Compare payload already carries.
- **`core.js`.** A geometric projection of your line onto the pro's (`projectOnLine`, extracted from the existing `roadHeights`) gives each of your points a lap distance and a signed sideways offset. `lineOffsets` reads those offsets at each corner, and `lineBand` turns metres into a label.
- **UI.** `panels.js` renders the labels in the corner list. `index.js` wires them to the Line button and switches the camera on a corner pick.

**Tech Stack:** Plain ES modules (no build step), Three.js scene (untouched), `node --test` for the pure helpers, Playwright for the browser check.

**Spec:** `docs/superpowers/specs/2026-10-30-3d-compare-line-analysis-design.md`

**Branch:** `feature/compare-line-analysis`. It is built on `fix/compare-3d-road-height`, which added `roadHeights`.

---

## Background the engineer needs

- **`pro` and `you` points.**
  - `pro` is the column-array object the 3D views use: `{ d, t, x, y, z, speed, throttle, brake, gear, steer, heading? }`. All columns have the same length, and `d` is the lap distance in metres on a 2 m grid.
  - The last point sits on, or next to, the first: the lap is a closed loop.
  - In 3D Compare, `youPoints(pro, you)` (`app/static/js/learn3d/core.js`) builds the same shape for your lap. When the server sent your own line, it has your `x`/`y` and `ownLine: true`.
- **Your x/y at index i is not the same place as the pro's x/y at index i.** It can be −30 to +26 m along the lap from it. Never compare the two lines index by index; always project.
- **Corners** (`data.corners` in `index.js`) are the pro's corners on the pro's lap: `{ short, entry_m, apex_m, exit_m, action, time_delta, your_min_speed_kmh, ... }`.
- **Headings.** `lineHeadings(points)` returns unwrapped headings in radians. They increase through a left-hand (counter-clockwise) turn.
- **Running tests.**
  - JS: `node --test tests/js/*.mjs`. Don't pass a directory: `node --test tests/js/` fails.
  - Python: `python3 -m pytest -q`.
- **Server for browser checks.**
  - Start it with `python3 -m app.main`, run in the background. It listens on port 8000.
  - Stop it with `kill $(lsof -ti:8000 -sTCP:LISTEN)`.
  - Playwright screenshots must be saved under `.playwright-mcp/`.

## File structure

| File | Change |
|---|---|
| `app/static/js/learn3d/core.js` | `projectOnLine` (extracted from `roadHeights`), `lineOffsets`, `lineBand`, `LINE_SAME_M`, `LINE_BIG_M` |
| `app/static/js/learn3d/panels.js` | `createCornerList` takes an optional `line` argument and returns `setLine(on)` |
| `app/static/js/learn3d/index.js` | Computes the offsets, wires the Line button and the corner pick, updates the note text |
| `app/static/css/main.css` | `.l3d-line-big` |
| `tests/js/learn3d_core.test.mjs` | Tests for `lineBand` and `lineOffsets` |

---

### Task 1: `lineBand`

**Files:**
- Modify: `app/static/js/learn3d/core.js` (append near the other exported helpers, after `gapAt`)
- Test: `tests/js/learn3d_core.test.mjs`

- [ ] **Step 1: Write the failing test**

  In `tests/js/learn3d_core.test.mjs`, add `lineBand` to the import list at the top. The list becomes:

  ```js
  import {
    bisect, stateAt, timeAtDistance, cuePoint, nextCue, cornerSeekRange, fmtTime, youPoints, gapAt, lineHeadings,
    lineBand, lineOffsets,
  } from "../../app/static/js/learn3d/core.js";
  ```

  Then append at the end of the file:

  ```js
  test("lineBand: ≈ under 2 m, wide/tight from 2 m, + from 4 m, — without a reading", () => {
    assert.equal(lineBand(null), "—");
    assert.equal(lineBand(0), "≈");
    assert.equal(lineBand(1.99), "≈");
    assert.equal(lineBand(-1.99), "≈");
    assert.equal(lineBand(2), "wide");
    assert.equal(lineBand(3.99), "wide");
    assert.equal(lineBand(4), "wide+");
    assert.equal(lineBand(-2), "tight");
    assert.equal(lineBand(-4.5), "tight+");
  });
  ```

- [ ] **Step 2: Run the test and check it fails**

  Run: `node --test tests/js/learn3d_core.test.mjs 2>&1 | grep -E "^not ok|^# (pass|fail)|SyntaxError"`

  Expected: the run fails. The import of `lineBand` / `lineOffsets` reports "does not provide an export named".

- [ ] **Step 3: Implement**

  In `app/static/js/learn3d/core.js`, append after `gapAt`:

  ```js
  // Your line against the pro's, per corner (3D Compare). Bands, not metres: the line fit and the game's
  // track model are only good to about ±2 m.
  export const LINE_SAME_M = 2;   // under this, either way: same line as the pro
  export const LINE_BIG_M = 4;    // from this: a big difference, marked "+"

  /** Metres wider (+) or tighter (−) than the pro -> "≈", "wide", "tight", "wide+", "tight+"; "—" for no reading. */
  export function lineBand(m) {
    if (m == null) return "—";
    const a = Math.abs(m);
    if (a < LINE_SAME_M) return "≈";
    return (m > 0 ? "wide" : "tight") + (a >= LINE_BIG_M ? "+" : "");
  }
  ```

  Step 2 also imported `lineOffsets`, so the test file won't load until it exists. Add a temporary stub directly below `lineBand`; Task 2 replaces it:

  ```js
  export function lineOffsets() { return []; }
  ```

- [ ] **Step 4: Run the tests and check they pass**

  Run: `node --test tests/js/learn3d_core.test.mjs 2>&1 | grep -E "^not ok|^# (pass|fail)"`

  Expected: `# fail 0`.

- [ ] **Step 5: Commit**

  ```bash
  git add app/static/js/learn3d/core.js tests/js/learn3d_core.test.mjs
  git commit -m "feat(web): lineBand labels a line difference in bands

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

---

### Task 2: `projectOnLine` and `lineOffsets`

**Files:**
- Modify: `app/static/js/learn3d/core.js`. This task:
  - replaces `roadHeights`, and the `youPoints` line that calls it;
  - replaces the `lineOffsets` stub from Task 1.
- Test: `tests/js/learn3d_core.test.mjs`

- [ ] **Step 1: Write the failing tests**

  Append to `tests/js/learn3d_core.test.mjs`:

  ```js
  /**
   * A closed circular lap of radius r, 2 m apart, turning left (counter-clockwise) or right, as a pro
   * lap; `shift` starts the points that many indices further round (same geometry, different index).
   */
  function circleLap(r, left = true, shift = 0) {
    const n = 315, idx = Array.from({ length: n }, (_, i) => i);
    const th = (i) => (2 * Math.PI * (i + shift)) / (n - 1);
    return {
      d: idx.map((i) => (2 * Math.PI * 100 * i) / (n - 1)), t: idx,
      x: idx.map((i) => r * Math.cos(th(i))), y: idx.map((i) => (left ? 1 : -1) * r * Math.sin(th(i))),
      z: idx.map(() => 0),
    };
  }
  const CORNER = { short: "T1", entry_m: 200, apex_m: 300, exit_m: 400 };

  test("lineOffsets: + where your line runs outside the pro's, − inside, in left and right turns", () => {
    for (const left of [true, false]) {
      const pro = circleLap(100, left);
      const wide = lineOffsets(pro, circleLap(103, left), [CORNER])[0];
      for (const k of ["entry", "apex", "exit"]) assert.ok(Math.abs(wide[k] - 3) < 0.05, `${left} wide ${k}: ${wide[k]}`);
      const tight = lineOffsets(pro, circleLap(97, left), [CORNER])[0];
      for (const k of ["entry", "apex", "exit"]) assert.ok(Math.abs(tight[k] + 3) < 0.05, `${left} tight ${k}: ${tight[k]}`);
      assert.equal(lineBand(lineOffsets(pro, circleLap(105, left), [CORNER])[0].apex), "wide+");
    }
  });

  test("lineOffsets reads the same place on the track whatever index your points sit at", () => {
    // The server's alignment puts your x/y up to ~30 m along from the pro's at the same index.
    const pro = circleLap(100);
    const at = lineOffsets(pro, circleLap(103), [CORNER])[0];
    const shifted = lineOffsets(pro, circleLap(103, true, 10), [CORNER])[0];   // 10 points = 20 m further round
    for (const k of ["entry", "apex", "exit"]) assert.ok(Math.abs(shifted[k] - at[k]) < 0.05, `${k}: ${shifted[k]} vs ${at[k]}`);
  });

  test("lineOffsets gives null where none of your points reach a corner", () => {
    const pro = circleLap(100);
    assert.deepEqual(lineOffsets(pro, { x: [], y: [] }, [CORNER]), [{ entry: null, apex: null, exit: null }]);
  });
  ```

- [ ] **Step 2: Run the tests and check they fail**

  Run: `node --test tests/js/learn3d_core.test.mjs 2>&1 | grep -E "^not ok|^# (pass|fail)"`

  Expected: the three new `lineOffsets` tests fail, because the stub returns `[]`. Every other test passes.

- [ ] **Step 3: Implement**

  In `app/static/js/learn3d/core.js`:

  **3a.** In `youPoints`, replace the line

  ```js
      out.z = roadHeights(pro, you.x, you.y);
  ```

  with

  ```js
      out.z = projectOnLine(pro, you.x, you.y).z;
  ```

  **3b.** Replace the whole `roadHeights` function, including its doc comment, with:

  ```js
  /**
   * Your x/y points projected onto the pro's line (which the road follows): for each, the lap distance `s`
   * on the pro's line, the road height `z` there, and `side`, metres to the left (+) or right (−) of the
   * pro's line. Searched only `span` m either side of your index. Not the pro's point at your index: the
   * server's distance alignment puts your x/y up to ~30 m along from the pro's at the same d (on a slope
   * that sank your car metres into the road). Not the nearest point on the whole lap either: Suzuka
   * crosses itself.
   */
  export function projectOnLine(pro, x, y, span = 60) {
    const { d, x: px, y: py, z: pz } = pro, np = d.length, loop = np - 1;
    const w = Math.max(1, Math.round(span / ((d[np - 1] - d[0]) / loop)));
    const s = new Array(x.length), z = new Array(x.length), side = new Array(x.length);
    for (let i = 0; i < x.length; i++) {
      let best = Infinity;
      for (let k = i - w; k <= i + w; k++) {
        const a = ((k % loop) + loop) % loop, b = a + 1;             // segment a -> b, wrapping round the lap
        const ex = px[b] - px[a], ey = py[b] - py[a], len2 = ex * ex + ey * ey;
        const rx = x[i] - px[a], ry = y[i] - py[a];
        const f = len2 > 0 ? Math.min(1, Math.max(0, (rx * ex + ry * ey) / len2)) : 0;
        const dx = px[a] + f * ex - x[i], dy = py[a] + f * ey - y[i], dist = dx * dx + dy * dy;
        if (dist < best) {
          best = dist;
          s[i] = d[a] + f * (d[b] - d[a]);
          z[i] = pz[a] + f * (pz[b] - pz[a]);
          side[i] = len2 > 0 ? (ex * ry - ey * rx) / Math.sqrt(len2) : 0;   // cross product: + = left of travel
        }
      }
    }
    return { s, z, side };
  }
  ```

  The search loop and the `z` result are unchanged from `roadHeights`, so the existing `youPoints` tests guard the refactor.

  **3c.** Replace the `lineOffsets` stub from Task 1 with:

  ```js
  const LINE_WINDOW_M = 5;    // a corner point's reading: your points within this of it, along the lap
  const TURN_SPAN_M = 20;     // the corner's direction: the pro's heading change across this either side of the apex

  /**
   * Per corner, how much wider (+) or tighter (−) your line runs than the pro's at entry, apex and exit
   * (metres; null where none of your points come within LINE_WINDOW_M of that point). `pro`: the pro's
   * points; `you`: {x, y} of your line; `corners`: the pro's corners ({entry_m, apex_m, exit_m}). The
   * corner's direction comes from the pro's line, not the track json.
   */
  export function lineOffsets(pro, you, corners) {
    const { s, side } = projectOnLine(pro, you.x, you.y);
    const h = pro.heading || lineHeadings(pro);
    const median = (v) => {
      const a = [...v].sort((p, q) => p - q), m = a.length >> 1;
      return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
    };
    const at = (m) => {
      const v = [];
      for (let i = 0; i < s.length; i++) if (Math.abs(s[i] - m) <= LINE_WINDOW_M) v.push(side[i]);
      return v.length ? median(v) : null;
    };
    return corners.map((c) => {
      const turn = h[bisect(pro.d, c.apex_m + TURN_SPAN_M)] - h[bisect(pro.d, c.apex_m - TURN_SPAN_M)];
      const outside = turn > 0 ? -1 : 1;     // a left turn's outside is to the right (side < 0)
      const read = (m) => { const v = at(m); return v == null ? null : outside * v; };
      return { entry: read(c.entry_m), apex: read(c.apex_m), exit: read(c.exit_m) };
    });
  }
  ```

- [ ] **Step 4: Run the tests and check they pass**

  Run: `node --test tests/js/*.mjs 2>&1 | grep -E "^not ok|^# (pass|fail)"`

  Expected: `# fail 0`. The two existing `youPoints` road-height tests still pass.

- [ ] **Step 5: Commit**

  ```bash
  git add app/static/js/learn3d/core.js tests/js/learn3d_core.test.mjs
  git commit -m "feat(web): lineOffsets reads your line against the pro's at each corner

  projectOnLine replaces roadHeights: one projection now gives the road height
  and the sideways offset.

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

---

### Task 3: Corner list shows the bands

**Files:**
- Modify: `app/static/js/learn3d/panels.js`:
  - the import on line 2;
  - `createCornerList` (from `export function createCornerList(el, corners, onPick, compare = false) {` to the end of its returned object).
- Modify: `app/static/css/main.css` (after the `.l3d-chead:hover` rule)

`panels.js` touches the DOM, so it has no node tests. Task 5 checks it in the browser.

- [ ] **Step 1: Update the import**

  In `app/static/js/learn3d/panels.js`, change line 2 from

  ```js
  import { cuePoint, fmtTime } from "./core.js";
  ```

  to

  ```js
  import { cuePoint, fmtTime, lineBand, LINE_BIG_M } from "./core.js";
  ```

- [ ] **Step 2: Extend `createCornerList`**

  Replace the doc comment and the signature

  ```js
  /**
   * One row per corner. Learn: brake point → min speed, brake length.
   * Compare: pro → your apex speed, and the time you gained (−) or lost (+) there. Click = replay that corner.
   */
  export function createCornerList(el, corners, onPick, compare = false) {
  ```

  with

  ```js
  /**
   * One row per corner. Learn: brake point → min speed, brake length.
   * Compare: pro → your apex speed, and the time you gained (−) or lost (+) there. Click = replay that corner.
   * `line` (3D Compare with your own line): {offsets: lineOffsets(...), driver}; setLine(true) then shows
   * how your line runs against the pro's at entry / apex / exit in place of the apex speeds.
   */
  export function createCornerList(el, corners, onPick, compare = false, line = null) {
  ```

  Then replace the block that starts at `const rows = [...el.querySelectorAll("button.l3d-crow")];` and ends at `let current = -2;` with:

  ```js
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
  ```

  Finally, in the returned object, add a `setLine` method after `set(d) { ... },`, so the object ends:

  ```js
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
  ```

  `lineBand` returns only fixed strings, so `lineMid` needs no escaping. `line.driver` comes from the payload, so it goes through `esc`.

- [ ] **Step 3: Add the CSS**

  In `app/static/css/main.css`, directly after the line

  ```css
  .l3d-chead:hover { background: none; }
  ```

  add:

  ```css
  .l3d-crow.l3d-line-big > span:nth-child(2) { font-weight: 600; }   /* 3D Compare Line: a 4 m+ difference */
  ```

- [ ] **Step 4: Run the tests**

  Run: `node --test tests/js/*.mjs 2>&1 | grep -E "^# (pass|fail)"`

  Expected: `# fail 0`. This only checks that nothing else broke; `panels.js` is checked in Task 5.

- [ ] **Step 5: Commit**

  ```bash
  git add app/static/js/learn3d/panels.js app/static/css/main.css
  git commit -m "feat(web): corner list can show your line's bands in place of apex speeds

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

---

### Task 4: Wire it up in `index.js`

**Files:**
- Modify: `app/static/js/learn3d/index.js`:
  - the import on line 7;
  - `NOTE_OWN_LINE` (around line 68);
  - the `createCornerList` call (around line 111);
  - the `#l3dLineBtn` click handler (around line 272).

- [ ] **Step 1: Import `lineOffsets`**

  Change line 7 from

  ```js
  import { stateAt, timeAtDistance, nextCue, cornerSeekRange, fmtTime, youPoints, gapAt, lineHeadings } from "./core.js";
  ```

  to

  ```js
  import { stateAt, timeAtDistance, nextCue, cornerSeekRange, fmtTime, youPoints, gapAt, lineHeadings, lineOffsets } from "./core.js";
  ```

- [ ] **Step 2: Update the note**

  Replace

  ```js
  const NOTE_OWN_LINE = "Your line is matched to the track shape (about ±1 m). The pro line is FastF1's smoothed position data, not the exact line.";
  ```

  with

  ```js
  const NOTE_OWN_LINE = "Your line is matched to the track shape. Wide / tight compare it with the pro's line, to about ±2 m; an offset between the game's track and the real one shows up as well. The pro line is FastF1's smoothed position data, not the exact line.";
  ```

- [ ] **Step 3: Pass the offsets and switch to Top on a pick**

  Replace the `createCornerList` call

  ```js
    S.list = createCornerList($("l3dCorners"), data.corners, (c) => {
      if (locked()) return;
      const [from, to] = cornerSeekRange(c);
  ```

  with

  ```js
    S.list = createCornerList($("l3dCorners"), data.corners, (c) => {
      if (locked()) return;
      if ($("l3dLineBtn").getAttribute("aria-pressed") === "true") setCam("top");   // see both lines through the corner
      const [from, to] = cornerSeekRange(c);
  ```

  Its closing line

  ```js
    }, !!Y);
  ```

  becomes

  ```js
    }, !!Y, ownLine ? { offsets: lineOffsets(P, Y, data.corners), driver: data.meta.driver } : null);
  ```

  `P.heading` is already set at the top of `start()`, so `lineOffsets` reuses it.

- [ ] **Step 4: The Line button also switches the list**

  Replace

  ```js
    markPressed($("l3dLineBtn"), on);
    view.scene.setLine(on);
  ```

  with

  ```js
    markPressed($("l3dLineBtn"), on);
    view.scene.setLine(on);
    view.list.setLine(on);
  ```

  `start()` creates a new list for every lap, in its speed state, and resets the button to off, so the two stay in step.

- [ ] **Step 5: Run the tests**

  Run: `node --test tests/js/*.mjs 2>&1 | grep -E "^# (pass|fail)"; python3 -m pytest -q 2>&1 | tail -1`

  Expected: node `# fail 0`; pytest reports all tests passed, with no failures.

- [ ] **Step 6: Commit**

  ```bash
  git add app/static/js/learn3d/index.js
  git commit -m "feat(web): 3D Compare Line shows wide / tight per corner and picks a corner in Top view

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

---

### Task 5: Browser check

**Files:** none. If this check finds a bug, fix it in the task that owns the code, then rerun this task.

- [ ] **Step 1: Start the server**

  Run: `python3 -m app.main` (in the background), then `curl -s localhost:8000/health`

  Expected: `{"status":"ok"}`

- [ ] **Step 2: Load Suzuka in 3D Compare**

  In Playwright:
  1. Open `http://localhost:8000`.
  2. Click `[data-mode="compare_3d"]`.
  3. Click `#fileSection` and upload `data/suzuka_sample.csv`.
  4. Click **Run analysis**.
  5. Wait until `#l3dStage canvas` exists and `#l3dLineBtn` is not hidden.
  6. Click `[data-view="corners"]`.

- [ ] **Step 3: Check the Line toggle**

  Evaluate:

  ```js
  () => {
    const mid = () => [...document.querySelectorAll('#l3dCorners button.l3d-crow')].map(r => r.children[1].textContent);
    const head = () => document.querySelector('#l3dCorners .l3d-chead > span:nth-child(2)').textContent;
    const before = [head(), mid()[0]];
    document.getElementById('l3dLineBtn').click();
    const on = [head(), mid(), document.querySelectorAll('#l3dCorners .l3d-line-big').length];
    document.getElementById('l3dLineBtn').click();
    const after = [head(), mid()[0]];
    document.getElementById('l3dLineBtn').click();
    return { before, on, after };
  }
  ```

  Expected:
  - `before` and `after` are the apex-speed header and text.
  - `on[0]` is `Line vs VER: entry · apex · exit`.
  - Every `on[1]` row matches `Entry … · Apex … · Exit …`, using only `≈ wide tight wide+ tight+ —`.
  - `on[2]` is ≥ 1.
  - On this lap, DUN reads wide at the apex (the feasibility study measured +6.5 m).

- [ ] **Step 4: Check a corner pick switches to Top**

  With Line on, click the `DUN` row. Then evaluate `document.querySelector('[data-cam="top"]').getAttribute('aria-pressed')`.

  Expected: `"true"`.

  Take a screenshot of `#l3dStage` to `.playwright-mcp/line-analysis-suzuka-dun-top.png`. Both ribbons should be visible through the corner.

- [ ] **Step 5: Check Zandvoort**

  Repeat Step 2 with `data/Zandvoort.csv`. The track is detected from the CSV; if not, choose Zandvoort.

  Turn Line on and read the `T2` row. Expected: `Exit wide+`. The feasibility study measured +5 to +9 m on laps 1–4; the default lap is the fastest one.

- [ ] **Step 6: Check the console**

  Read the console messages at level `warning`. Expected: no errors or warnings from `learn3d`. A 404 for `favicon.ico` is pre-existing and fine.

- [ ] **Step 7: Stop the server**

  Run: `kill $(lsof -ti:8000 -sTCP:LISTEN)`

  An exit code of 143 from the server's own shell is expected.

---

## Self-review notes

- **Spec coverage.**

  | Spec item | Task |
  |---|---|
  | `projectOnLine`, `lineOffsets`, `lineBand` and the constants | Tasks 1–2 |
  | Corner list text, bold rows, header and `setLine` | Task 3 |
  | Offsets, Line button, Top on pick and the note | Task 4 |
  | Browser checks, Suzuka and Zandvoort | Task 5 |
  | Error handling: no line → no button, so the list is unchanged; no reading → `—` | Tasks 2–3 |

- **Names used across tasks.**
  - `lineOffsets(pro, you, corners)` → `[{entry, apex, exit}]`
  - `lineBand(m)`
  - `LINE_BIG_M`
  - `createCornerList(el, corners, onPick, compare, line)` with `line = {offsets, driver}`
  - `list.setLine(on)`
