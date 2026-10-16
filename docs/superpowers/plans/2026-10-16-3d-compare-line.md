# 3D Compare: Your Own Line Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In 3D Compare your car drives your own line from the game CSV, and a Line toggle draws your line next to the pro's reference line.

**Architecture:**
- **Backend:** `app/main.py` maps the game's `game_world_x/y` path onto the pro lap's frame. It does a similarity fit and then ICP. It adds `you.x` / `you.y` on the pro's distance grid, unless the fit is poor.
- **Frontend:** `core.youPoints` uses those positions when present. `scene.js` gains a second ribbon. `index.js` sets lane 0, makes the pro car a ghost, and adds a Line button and a note.

**Tech Stack:** Python 3.11 / FastAPI / numpy / scipy (`cKDTree`, already a dependency); ES modules + Three.js; pytest; `node --test`; Playwright MCP for the browser check.

**Spec:** `docs/superpowers/specs/2026-10-16-3d-compare-line-design.md`

**Ground rules for every task:**
- Work on branch `feature/3d-compare-line`.
- Don't touch `src/`, `tracks/` or `data/raw/`.
- Leave the untracked files alone: `AGENTS.md`, `data/reference_laps/*.pkl`.
- End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

---

## File map

| File | Change |
| --- | --- |
| `app/main.py` | Add `LINE_FIT_MAX_M` and `_fit_line`. `_build_you_lap` takes `pro_xy` and adds `x`/`y`. `_compare_3d_payload` passes the pro's x/y. |
| `tests/test_compare_3d.py` | `TestFitLine`; line tests in `TestBuildYouLap`; update the endpoint key-set test; a line-on-track endpoint test. |
| `app/static/js/learn3d/core.js` | `youPoints` uses `you.x/y` and sets `heading`. |
| `tests/js/learn3d_core.test.mjs` | A `youPoints` own-line test. |
| `app/static/js/learn3d/scene.js` | `createScene(..., youLine)`; the ribbon helper takes a path; a second ribbon; `setLine` toggles both. |
| `app/static/js/learn3d/index.js` | Cars at lane 0 with the pro ghosted when your line is present; Line button; note; `unload` reset. |
| `app/static/index.html` | `#l3dLineBtn` in the camera group; `#l3dLineNote` under the keys line. |
| `app/static/css/main.css` | `.l3d-note`. |
| `docs/superpowers/specs/2026-10-15-3d-compare-design.md` | Mark the "your own line" non-goal as superseded. |

---

### Task 1: `_fit_line` (backend)

**Files:**
- Modify: `app/main.py`. Put the constant next to `PRO_LAP_STEP_M`'s block of constants and the function just above `def _build_you_lap`.
- Test: `tests/test_compare_3d.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_compare_3d.py`, add `from scipy.spatial import cKDTree` after `from fastapi.testclient import TestClient`, change the `app.main` import line to:

```python
from app.main import LINE_FIT_MAX_M, _build_you_lap, _fit_line, app
```

and add this below the `GRID = ...` line:

```python
def _loop(n, bump=0.0):
    """A closed, non-symmetric ~2.3 km loop of n points. `bump` m sideways over one stretch (a different line)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.c_[400 * np.cos(t) + 100 * np.cos(2 * t), 250 * np.sin(t)]
    tang = np.gradient(xy, axis=0)
    tang /= np.linalg.norm(tang, axis=1)[:, None]
    normal = np.c_[-tang[:, 1], tang[:, 0]]
    w = np.where((t > 1.0) & (t < 1.6), np.sin((t - 1.0) / 0.6 * np.pi) ** 2, 0.0)
    return xy + bump * w[:, None] * normal, w


def _to_game_frame(xy):
    """Mirror, rotate 30°, scale 1.05 and shift: a different frame, like the game's."""
    a = np.radians(30)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (1.05 * xy * [1, -1]) @ rot.T + [1000, -500]


class TestFitLine:
    def test_recovers_a_line_change_through_another_frame(self):
        pro, _ = _loop(1000)
        you, w = _loop(2400, bump=3.0)
        out, median = _fit_line(_to_game_frame(you), pro)
        dense, _ = _loop(20000)
        dist = cKDTree(dense).query(out)[0]
        assert median < 0.5
        assert dist[w > 0.95].max() == pytest.approx(3.0, abs=0.3)   # the 3 m wider stretch survives the fit
        assert np.median(dist[w == 0]) < 0.3                          # the rest lies on the pro's line

    def test_unrelated_path_fails_the_gate(self):
        pro, _ = _loop(1000)
        noise = np.random.default_rng(0).uniform(-500, 500, (2000, 2))
        assert _fit_line(noise, pro)[1] > LINE_FIT_MAX_M
```

- [ ] **Step 2: Run the tests to make sure they fail**

Run: `python3 -m pytest -q tests/test_compare_3d.py -k FitLine`
Expected: collection error `ImportError: cannot import name 'LINE_FIT_MAX_M'`.

- [ ] **Step 3: Implement**

In `app/main.py`, add next to the other 3D constants (near `PRO_LAP_STEP_M`):

```python
LINE_FIT_MAX_M = 4.0          # 3D Compare drops your line when it sits further than this (median) from the pro's
```

Add above `def _build_you_lap`:

```python
def _fit_line(game_xy, pro_xy, rounds: int = 30):
    """Map your game path onto the pro lap's frame: a similarity fit, then ICP onto the pro's line.

    Both are (n, 2) arrays in lap order, in any frame (the game's can be mirrored, rotated,
    scaled and shifted against FastF1's). The first fit pairs points at the same fraction of
    the lap; each ICP round refits against the nearest point on the pro's line, so distance
    drift between the two laps doesn't skew the shape. Returns (your path in the pro's
    frame, median distance in metres from it to the pro's line).
    """
    from scipy.spatial import cKDTree

    def similarity(a, b):        # least-squares rotation (mirror allowed), uniform scale and shift: a -> b
        ma, mb = a.mean(0), b.mean(0)
        u, s, vt = np.linalg.svd((a - ma).T @ (b - mb))
        r = vt.T @ u.T
        k = s.sum() / ((a - ma) ** 2).sum()
        return lambda p: k * (p - ma) @ r.T + mb

    def at_fraction(xy, n):
        i = np.linspace(0, len(xy) - 1, n)
        return np.c_[np.interp(i, np.arange(len(xy)), xy[:, 0]), np.interp(i, np.arange(len(xy)), xy[:, 1])]

    game_xy, pro_xy = np.asarray(game_xy, dtype=float), np.asarray(pro_xy, dtype=float)
    dense = at_fraction(pro_xy, 4 * len(pro_xy))   # nearest-vertex error well under a metre
    tree = cKDTree(dense)
    fit = similarity(at_fraction(game_xy, 2000), at_fraction(pro_xy, 2000))
    for _ in range(rounds):
        _, idx = tree.query(fit(game_xy))
        fit = similarity(game_xy, dense[idx])
    out = fit(game_xy)
    return out, float(np.median(tree.query(out)[0]))
```

- [ ] **Step 4: Run the tests to make sure they pass**

Run: `python3 -m pytest -q tests/test_compare_3d.py -k FitLine`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_compare_3d.py
git commit -m "feat(web): fit the game lap's path onto the pro's line (similarity + ICP)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `you.x` / `you.y` in the Compare payload

**Files:**
- Modify: `app/main.py`: `_build_you_lap` (signature, docstring, return value) and `_compare_3d_payload` (the `_build_you_lap(...)` call).
- Test: `tests/test_compare_3d.py`

- [ ] **Step 1: Write the failing tests**

Add to `class TestBuildYouLap` in `tests/test_compare_3d.py`:

```python
    def test_adds_your_line_when_the_game_path_fits(self):
        aligned = _aligned()
        pro, _ = _loop(GRID.size)
        game, _ = _loop(len(aligned))
        aligned["game_world_x"], aligned["game_world_y"] = _to_game_frame(game).T
        you = _build_you_lap(aligned, GRID, 20.0, pro_xy=pro)
        assert len(you["x"]) == len(you["y"]) == GRID.size
        assert np.median(np.hypot(np.array(you["x"]) - pro[:, 0], np.array(you["y"]) - pro[:, 1])) < 3.0

    def test_no_line_when_the_game_path_does_not_fit(self):
        aligned = _aligned()
        pro, _ = _loop(GRID.size)
        aligned["game_world_x"], aligned["game_world_y"] = np.random.default_rng(0).uniform(-500, 500, (2, len(aligned)))
        you = _build_you_lap(aligned, GRID, 20.0, pro_xy=pro)
        assert "x" not in you and "y" not in you

    def test_no_line_without_game_positions(self):
        pro, _ = _loop(GRID.size)
        assert "x" not in _build_you_lap(_aligned(), GRID, 20.0, pro_xy=pro)
```

In `class TestCompare3DEndpoint`, change the last line of `test_has_header_pro_and_you` to:

```python
        assert set(payload["you"]) == {"t", "speed", "throttle", "brake", "gear", "x", "y"}
```

and add:

```python
    def test_your_line_stays_on_the_track(self, payload):
        p = payload["pro"]["points"]
        pro = np.c_[p["x"], p["y"]]
        you = np.c_[payload["you"]["x"], payload["you"]["y"]]
        dist = cKDTree(pro).query(you)[0]
        assert np.median(dist) < 4.0       # measured 2.0 m on this lap
        assert dist.max() < 15.0           # never far off the road
```

- [ ] **Step 2: Run the tests to make sure they fail**

Run: `python3 -m pytest -q tests/test_compare_3d.py`
Expected:
- `test_adds_your_line_when_the_game_path_fits` and the two other new tests fail with `TypeError: _build_you_lap() got an unexpected keyword argument 'pro_xy'`;
- `test_has_header_pro_and_you` and `test_your_line_stays_on_the_track` fail with an assertion or `KeyError: 'x'`.

- [ ] **Step 3: Implement**

In `app/main.py`, change the `_build_you_lap` signature and docstring to:

```python
def _build_you_lap(aligned, d_grid, lap_time: float, pro_xy=None) -> dict:
    """Your game lap resampled onto the pro lap's distance grid, for the 3D compare view.

    ``aligned`` has one row per game metre; the pipeline's alignment anchors
    (``aligned.attrs``) map each game metre to real lap distance, the axis ``d_grid``
    (from ``_build_pro_lap``) uses. Time comes from integrating game speed, pinned to
    ``lap_time`` like the pro lap. With ``pro_xy`` (the pro's x/y on ``d_grid``), your game
    path is fitted onto it (``_fit_line``) and returned as ``x``/``y``, unless it sits more
    than LINE_FIT_MAX_M from the pro's line, or the game positions are missing.
    """
```

Replace its final `return {...}` with:

```python
    you = {
        "t": rnd(col(t), 3),
        "speed": rnd(col(aligned["game_speed_kmh"]), 1),
        "throttle": rnd(col(aligned["game_throttle"]), 3),
        "brake": rnd(col(aligned["game_brake"]), 3),
        "gear": [int(g) for g in np.rint(col(aligned["game_gear"]))],
    }
    if pro_xy is not None and {"game_world_x", "game_world_y"} <= set(aligned.columns):
        game_xy = aligned[["game_world_x", "game_world_y"]].to_numpy(dtype=float)
        if np.isfinite(game_xy).all():
            try:
                xy, median = _fit_line(game_xy, pro_xy)
            except (ValueError, np.linalg.LinAlgError):
                traceback.print_exc()
            else:
                if median <= LINE_FIT_MAX_M:
                    you["x"], you["y"] = rnd(col(xy[:, 0]), 2), rnd(col(xy[:, 1]), 2)
    return you
```

In `_compare_3d_payload`, change the call to:

```python
        you = _build_you_lap(aligned, pro["points"]["d"], float(report.game_time),
                             pro_xy=np.c_[pro["points"]["x"], pro["points"]["y"]])
```

- [ ] **Step 4: Run the tests to make sure they pass**

Run: `python3 -m pytest -q tests/test_compare_3d.py`
Expected: all pass.

Then run: `python3 -m pytest -q`
Expected: all pass (69 on main, plus the 7 new tests from Tasks 1–2, so 76).

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_compare_3d.py
git commit -m "feat(web): 3D Compare payload carries your own line when it fits the track

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `youPoints` uses your positions

**Files:**
- Modify: `app/static/js/learn3d/core.js` (`youPoints`, plus the header comment's mention of it)
- Test: `tests/js/learn3d_core.test.mjs`

- [ ] **Step 1: Write the failing test**

Add after the existing `youPoints keeps the pro's line...` test in `tests/js/learn3d_core.test.mjs`:

```js
test("youPoints drives your own line when you send x/y", () => {
  // 51 points / 100 m: lineHeadings treats the path as a closed lap, so only check away from the ends.
  const n = 51, idx = Array.from({ length: n }, (_, i) => i);
  const pro = {
    d: idx.map((i) => 2 * i), t: idx, x: idx.map((i) => 2 * i), y: idx.map(() => 0), z: idx.map(() => 1),
    speed: idx.map(() => 100), throttle: idx.map(() => 1), brake: idx.map(() => 0), steer: idx.map(() => 0),
    gear: idx.map(() => 5),
  };
  const you = {
    t: idx, speed: pro.speed, throttle: pro.throttle, brake: pro.brake, gear: pro.gear,
    x: idx.map((i) => i * Math.SQRT2), y: idx.map((i) => i * Math.SQRT2),   // 2 m steps, 45° off the pro's line
  };
  const Y2 = youPoints(pro, you);
  assert.deepEqual(Y2.x, you.x);
  assert.deepEqual(Y2.y, you.y);
  assert.deepEqual(Y2.z, pro.z);                      // road height stays the pro's
  assert.deepEqual(Y2.d, pro.d);
  assert.equal(Y2.heading.length, n);
  assert.ok(Math.abs(stateAt(Y2, 25.5).heading - Math.PI / 4) < 1e-6);
  assert.equal(Y.heading, undefined);                // without x/y nothing changes
});
```

- [ ] **Step 2: Run the test to make sure it fails**

Run: `node --test tests/js/learn3d_core.test.mjs`
Expected: the new test fails, because `Y2.x` still equals the pro's x.

- [ ] **Step 3: Implement**

In `app/static/js/learn3d/core.js`, replace `youPoints` with:

```js
/**
 * Your lap as a `points` object: your timing and inputs on the pro's distance grid. With `you.x/y`
 * (your line, fitted to the track by the server) the car drives your line, with the pro's road height;
 * without them it drives the pro's line.
 */
export function youPoints(pro, you) {
  const out = { ...pro, t: you.t, speed: you.speed, throttle: you.throttle, brake: you.brake, gear: you.gear };
  if (you.x && you.y) {
    out.x = you.x;
    out.y = you.y;
    out.heading = lineHeadings(out);
  }
  return out;
}
```

Also change the header comment line `// In 3D Compare, \`youPoints()\` builds the same shape for your lap.` to:

```js
// In 3D Compare, `youPoints()` builds the same shape for your lap (your own line when the server sends one).
```

- [ ] **Step 4: Run the tests to make sure they pass**

Run: `node --test tests/js/learn3d_core.test.mjs tests/js/learn3d_rhythm.test.mjs tests/js/learn3d_drive.test.mjs`
Expected: `# fail 0`, with one more pass than before (40).

- [ ] **Step 5: Commit**

```bash
git add app/static/js/learn3d/core.js tests/js/learn3d_core.test.mjs
git commit -m "feat(web): youPoints drives your own line when the server sends one

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: your line ribbon in the scene

**Files:**
- Modify: `app/static/js/learn3d/scene.js`

There are no unit tests for `scene.js` (Three.js/WebGL). It is checked in the browser in Task 6.

- [ ] **Step 1: Pass `youLine` through**

Change the doc comment and the two signatures:

```js
/**
 * Builds the scene into `container`. Throws if WebGL is unavailable or the data is unusable.
 * `cars`: one {color, lane} per car; lane = sideways offset (m) so cars at the same spot don't overlap.
 * `youLine`: your lap's points (3D Compare with your own line) for a second line ribbon, else null.
 */
export function createScene(container, points, corners, cars, youLine = null) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  try {
    return buildScene(renderer, container, points, corners, cars, youLine);
```

and `function buildScene(renderer, container, points, corners, cars, youLine) {`.

- [ ] **Step 2: Make the ribbon helper take a path**

Replace

```js
  const pos = Array.from({ length: n }, (_, i) => toThree(points.x[i], points.y[i], points.z[i]));
```

with

```js
  const path = (p) => Array.from({ length: p.x.length }, (_, i) => toThree(p.x[i], p.y[i], p.z[i]));
  const pos = path(points);
```

Replace the ribbon helper's first lines and its signature:

```js
  // A strip along a path (three.js points + headings), offset sideways by ±half; the last quad closes the loop.
  const ribbon = (pts, hd, half, lift) => {
    const m = pts.length;
    const verts = new Float32Array(m * 6);
    const side = new THREE.Vector3();
    for (let i = 0; i < m; i++) {
      side.set(Math.sin(hd[i]), 0, Math.cos(hd[i])).multiplyScalar(half);   // right of travel
      const p = pts[i];
      verts.set([p.x + side.x, p.y + lift, p.z + side.z, p.x - side.x, p.y + lift, p.z - side.z], i * 6);
    }
    const index = [];
    for (let i = 0; i < m; i++) {
      const a = 2 * i, b = a + 1, c = (2 * (i + 1)) % (2 * m), d = c + 1;
      index.push(a, c, b, b, c, d);
    }
```

(the three lines after it, from `const g = new THREE.BufferGeometry();` to `return g;`, stay the same).

Update the two existing callers:

```js
  const road = ribbon(pos, heading, TRACK_HALF_WIDTH, 0.05);
```

```js
  const lineGeo = ribbon(pos, heading, 0.35, 0.12);
```

- [ ] **Step 3: Add your ribbon**

Right after `scene.add(line);`, add:

```js
  // 3D Compare: your line in your car's colour, shown and hidden with the pro's.
  let youRibbon = null;
  if (youLine) {
    youRibbon = new THREE.Mesh(ribbon(path(youLine), youLine.heading || lineHeadings(youLine), 0.35, 0.13),
      new THREE.MeshBasicMaterial({ color: cars[1].color, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -3 }));
    youRibbon.visible = false;
    scene.add(youRibbon);
  }
```

and change `setLine` to:

```js
  /** Show or hide the racing line(s): the pro's, and yours in 3D Compare. */
  function setLine(on) {
    line.visible = on;
    if (youRibbon) youRibbon.visible = on;
  }
```

- [ ] **Step 4: Check that nothing else broke**

Run: `node --check app/static/js/learn3d/scene.js && node --test tests/js/learn3d_core.test.mjs tests/js/learn3d_rhythm.test.mjs tests/js/learn3d_drive.test.mjs`
Expected: no syntax error, `# fail 0`.

- [ ] **Step 5: Commit**

```bash
git add app/static/js/learn3d/scene.js
git commit -m "feat(web): scene draws your line as a second ribbon in 3D Compare

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Compare UI (cars, ghost, Line button, note)

**Files:**
- Modify: `app/static/index.html`, `app/static/css/main.css`, `app/static/js/learn3d/index.js`

- [ ] **Step 1: Markup**

In `app/static/index.html`, change the camera group to:

```html
            <div class="l3d-cams" role="group" aria-label="Camera and line">
              <button type="button" data-cam="tcam" class="active" aria-pressed="true">T-cam</button>
              <button type="button" data-cam="chase" aria-pressed="false">Chase</button>
              <button type="button" data-cam="top" aria-pressed="false">Top</button>
              <button type="button" id="l3dLineBtn" aria-pressed="false" hidden>Line</button>
            </div>
```

and right after `<div class="l3d-keys" id="l3dKeys">…</div>` add:

```html
        <div class="l3d-note" id="l3dLineNote" hidden></div>
```

- [ ] **Step 2: CSS**

In `app/static/css/main.css`, after the `.l3d-keys { … }` rule (the first one, outside media queries), add:

```css
.l3d-note { margin-top: 4px; font-size: 0.72rem; color: var(--text-dim); }
```

(`.l3d-note` stays visible on phones; `.l3d-keys` is hidden there.)

- [ ] **Step 3: `index.js`: notes, ghost flag, cars**

Below `const KEYS_COMPARE = …`, add:

```js
const NOTE_OWN_LINE = "Your line is matched to the track shape (about ±1 m). The pro line is FastF1's smoothed position data, not the exact line.";
const NOTE_PRO_LINE = "Your line isn't available for this lap; your car follows the pro's line.";
```

In `start()`, replace

```js
  if (Y) Y.heading = P.heading;                  // your lap runs on the pro's line
```

with

```js
  const ownLine = !!Y && Y.x !== P.x;            // the server sent your own line (youPoints swapped x/y in)
  if (Y && !ownLine) Y.heading = P.heading;      // your lap runs on the pro's line
```

After `$("l3dKeys").textContent = …;`, add:

```js
  $("l3dLineNote").hidden = !Y;
  $("l3dLineNote").textContent = ownLine ? NOTE_OWN_LINE : NOTE_PRO_LINE;
```

Replace the `cars` / `createScene` lines with:

```js
    // Your own line: both cars on their real positions, the pro a see-through ghost where they overlap.
    // Otherwise both share the pro's line, side by side.
    const cars = !Y ? [{ color: LEARN_CAR_COLOR, lane: 0 }, { color: hex(YOU_COLOR), lane: 0 }]   // car 1 = you, in Drive only
      : ownLine ? [{ color: hex(PRO_COLOR), lane: 0 }, { color: hex(YOU_COLOR), lane: 0 }]
      : [{ color: hex(PRO_COLOR), lane: -1.6 }, { color: hex(YOU_COLOR), lane: 1.6 }];
    scene = createScene($("l3dStage"), P, data.corners, cars, ownLine ? Y : null);
```

In the `const S = view = { … }` object, add `ghostPro: ownLine,` after `data, scene,`, so the line reads:

```js
    data, scene, ghostPro: ownLine, t: 0, end, playing: false, rate: 1, stopAt: null, raf: 0, last: 0,
```

After the `$("l3dScrub").max = …` line, add:

```js
  $("l3dLineBtn").hidden = !ownLine || !scene;
  markPressed($("l3dLineBtn"), false);
```

- [ ] **Step 4: `index.js`: draw, unload, button wiring**

In `draw()`, change

```js
  if (S.scene) S.scene.setGhost(0, !!dv);
```

to

```js
  if (S.scene) S.scene.setGhost(0, !!dv || S.ghostPro);
```

In `unload()`, after `$("l3dKeys").textContent = KEYS_LEARN;`, add:

```js
  $("l3dLineNote").hidden = true;
  $("l3dLineBtn").hidden = true;
```

In the "controls (wired once)" section, after the `[data-cam]` listener line, add:

```js
$("l3dLineBtn").addEventListener("click", () => {
  if (!view || !view.scene) return;
  const on = $("l3dLineBtn").getAttribute("aria-pressed") !== "true";
  markPressed($("l3dLineBtn"), on);
  view.scene.setLine(on);
});
```

- [ ] **Step 5: Syntax and tests**

Run: `node --check app/static/js/learn3d/index.js && node --test tests/js/learn3d_core.test.mjs tests/js/learn3d_rhythm.test.mjs tests/js/learn3d_drive.test.mjs && python3 -m pytest -q`
Expected: no syntax error; node `# fail 0`; pytest all pass.

- [ ] **Step 6: Commit**

```bash
git add app/static/index.html app/static/css/main.css app/static/js/learn3d/index.js
git commit -m "feat(web): 3D Compare drives your own line, with a Line toggle and an accuracy note

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Browser check and doc note

**Files:**
- Modify: `docs/superpowers/specs/2026-10-15-3d-compare-design.md`

- [ ] **Step 1: Restart the server on this commit**

The build id is the git HEAD at server start, so restart after committing:

```bash
kill $(lsof -ti:8000 -sTCP:LISTEN) 2>/dev/null; python3 -m app.main   # run in the background
```

- [ ] **Step 2: Playwright check**

With `browser_run_code_unsafe`, do everything in one call (background tabs throttle rAF between calls):
1. Go to `http://localhost:8000/` at 1280×900.
2. Click `[data-mode="compare_3d"]`. Select `#yearSel` 2025, `#driver` VER, `#session` Q and `#track` suzuka. Run `setInputFiles('#fileInput', '/Users/masashi/my project/f125-demo/data/suzuka_sample.csv')`, then click `#analyzeBtn`.
3. `waitForSelector('#l3dLineBtn:not([hidden])', { timeout: 120000 })`.
4. Read `#l3dLineNote` text. Expect it to start with "Your line is matched".
5. Click `#l3dLineBtn` and check `aria-pressed="true"`. Click the `Top` camera, seek to about 8 s with the scrub bar, then screenshot to `.playwright-mcp/compare-line-top.png`.
6. Click `Chase`, then screenshot `.playwright-mcp/compare-line-chase.png`.

Look at the screenshots. Expect:
- an orange/red/green pedal-coloured pro ribbon and a blue ribbon of yours;
- your (blue) car sitting on the blue ribbon;
- the orange pro car see-through.

Then:
- run `browser_console_messages` at level `error` and expect 0;
- switch to 3D Learn on Suzuka and check that `#l3dLineBtn` and `#l3dLineNote` are hidden.

- [ ] **Step 3: Mark the old non-goal as superseded**

In `docs/superpowers/specs/2026-10-15-3d-compare-design.md`, change the non-goal bullet that starts with `- Your own racing line.` to:

```markdown
- Your own racing line. Both cars drive the pro's line, because the game→real shape alignment can be off by up to 150 m, while real line differences are a few metres. *Superseded by `2026-10-16-3d-compare-line-design.md`: an ICP fit brings the game path to within ~2 m of the pro's line, so your car now drives your own line.*
```

- [ ] **Step 4: Full checks and commit**

Run: `python3 -m pytest -q && node --test tests/js/learn3d_core.test.mjs tests/js/learn3d_rhythm.test.mjs tests/js/learn3d_drive.test.mjs`
Expected: all pass.

```bash
git add docs/superpowers/specs/2026-10-15-3d-compare-design.md
git commit -m "docs: 3D Compare spec points to the own-line follow-up

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
