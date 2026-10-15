# 3D Pro Lap Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "3D Learn" mode to the web app that replays a real F1 driver's qualifying lap on a 3D track. The viewer has T-cam, Chase and Top cameras, a HUD, a minimap, and a switchable panel (telemetry traces or a corner list).

**Architecture:** A new `GET /pro_lap` endpoint in `app/main.py` turns a cached FastF1 reference lap (`data/reference_laps/*.pkl`) into compact column arrays: metres, seconds, inferred steering, and per-corner brake/lift/flat data. The frontend is four small ES modules under `app/static/js/learn3d/`:

- a pure `core.js`, unit-tested with `node --test`;
- a Three.js `scene.js` (loaded from a CDN through an importmap and imported dynamically, so a CDN failure only disables the 3D canvas);
- `panels.js` for DOM/SVG;
- an `index.js` entry that exposes `window.Learn3D`.

The existing classic-script `main.js` wires the new mode in.

**Tech Stack:** Python 3.11 · FastAPI · numpy/pandas · pytest + FastAPI TestClient · vanilla JS ES modules · Three.js 0.170.0 (jsDelivr CDN) · Node 22 built-in test runner.

**Spec:** `docs/superpowers/specs/2026-10-15-3d-pro-lap-viewer-design.md`

---

## Ground rules for the implementer

- **Never modify anything in `src/`.** Web changes live in `app/` only; `src/` is consumed read-only.
- **Never modify `data/`.** The reference `.pkl` files are inputs.
- Branch: `feature/3d-pro-lap-viewer` (already checked out).
- Python tests: `python3 -m pytest` (baseline: 20 passed).
- JS tests: `node --test tests/js/learn3d_core.test.mjs` (Node 22 detects ESM syntax in `.js`, so no `package.json` is needed).
- Run the app locally: `python3 -m app.main` → <http://127.0.0.1:8000>
- Every commit message ends with the trailer line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

### Verified data facts (don't re-derive)

- `pd.read_pickle(...)` returns a tuple `(df, meta)`. The `df` columns are: `lap_distance` (1 m steps), `speed_kmh`, `throttle` (0–1), `brake` (0–1, with ramps from resampling), `gear`, `rpm`, `drs`, `world_position_X/Y/Z`.
- Positions are in **1/10 m** (Bahrain path 53,312 vs `meta["track_length"]` 5,372). The scale is derived per lap (Bahrain: 0.1008). After smoothing and 2 m resampling, path/length = 1.0007.
- `meta` contains `lap_time` (Bahrain 90.267), `track_length`, `gp_name`, `driver`, `year`.
- Integrating `Σ ds / v` gives 90.37 s on Bahrain, so it is rescaled to `lap_time`.
- `_ref_steering_from_curvature(df, d_full, d)` from `app/main.py` works directly on the pkl `df`. It returns a list the same length as `d`, in the range −1…+1 (negative = left).
- All `tracks/*.json` corners have `id, short, name, direction, entry_m, apex_m, exit_m`.
- Brake rule (validated on Bahrain and Monza, where every onset maps to exactly one corner):
  - Count only onsets of `brake ≥ 0.5`.
  - The search window for a corner is `[max(entry − 250, prev_apex − 30), apex − 30)`.

---

## File map

| File | Status | Responsibility |
| --- | --- | --- |
| `app/main.py` | modify | `_corner_actions`, `_build_pro_lap`, `_available_pro_laps`, `GET /pro_lap` |
| `tests/test_pro_lap.py` | create | pytest for the helpers and the endpoint (offline; uses the Bahrain pkl) |
| `app/static/js/learn3d/core.js` | create | Pure helpers: `bisect`, `stateAt`, `timeAtDistance`, `cuePoint`, `nextCue`, `cornerSeekRange`, `fmtTime` |
| `tests/js/learn3d_core.test.mjs` | create | `node --test` for `core.js` |
| `app/static/js/learn3d/panels.js` | create | `esc`, `renderHud`, `createMinimap`, `createTelemetry`, `createCornerList` |
| `app/static/js/learn3d/scene.js` | create | `createScene(container, points, corners)` → `{ update(state, cam), dispose() }` |
| `app/static/js/learn3d/index.js` | create | Entry: fetch, playback loop, controls, keyboard, localStorage; `window.Learn3D = { load, unload }` |
| `app/static/index.html` | modify | importmap, `#fileSection` id, mode button, `#blkLearn3d` block, module script tag |
| `app/static/css/main.css` | modify | 6-column mode grid; `.l3d-*` styles |
| `app/static/js/main.js` | modify | i18n strings, `MODES.learn_3d`, `applyModeUI`, loading-overlay text, `runLearn3d()` |

---

### Task 1: Corner brake/lift/flat classification (`_corner_actions`)

**Files:**

- Modify: `app/main.py` (add a new section just above the `# Routes` banner, around line 1160)
- Test: `tests/test_pro_lap.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pro_lap.py`:

```python
"""Tests for the 3D Learn mode backend: GET /pro_lap and its helpers (app/main.py)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app, _corner_actions

client = TestClient(app)

T2 = {"id": 2, "short": "T2", "name": "Turn 2", "direction": "left",
      "entry_m": 620, "apex_m": 700, "exit_m": 780}


def _corners(*extra):
    t1 = {"id": 1, "short": "T1", "name": "Turn 1", "direction": "right",
          "entry_m": 400, "apex_m": 500, "exit_m": 600}
    return [t1, *extra]


def _lap(brake_from=None, lift_from=None):
    """Synthetic 1 km lap sampled every 2 m: 250 km/h, slowing to 120 around T1."""
    d = np.arange(0, 1000, 2.0)
    speed = np.full(d.size, 250.0)
    speed[(d >= 450) & (d <= 550)] = 120.0
    throttle = np.ones(d.size)
    brake = np.zeros(d.size)
    if brake_from is not None:
        zone = (d >= brake_from) & (d < 480)
        brake[zone] = 1.0
        throttle[zone] = 0.0
    if lift_from is not None:
        throttle[(d >= lift_from) & (d < 480)] = 0.5
    t = np.concatenate([[0.0], np.cumsum(np.diff(d) / (speed[1:] / 3.6))])
    gear = np.where(speed < 150, 3, 7)
    return d, t, speed, throttle, brake, gear


class TestCornerActions:
    def test_brake_corner(self):
        c = _corner_actions(_corners(), *_lap(brake_from=380))[0]
        assert c["action"] == "brake"
        assert c["brake_start_m"] == pytest.approx(380, abs=2)
        assert c["brake_duration_s"] > 0
        assert c["min_speed_kmh"] == 120
        assert c["apex_gear"] == 3

    def test_lift_corner(self):
        c = _corner_actions(_corners(), *_lap(lift_from=380))[0]
        assert c["action"] == "lift"
        assert c["brake_start_m"] is None
        assert c["brake_duration_s"] is None

    def test_flat_corner(self):
        c = _corner_actions(_corners(), *_lap())[0]
        assert c["action"] == "flat"

    def test_brake_far_before_entry_is_ignored(self):
        d, t, s, th, b, g = _lap()
        b[(d >= 100) & (d < 140)] = 1.0          # > 250 m before T1 entry (400)
        assert _corner_actions(_corners(), d, t, s, th, b, g)[0]["action"] == "flat"

    def test_does_not_inherit_previous_corner_braking(self):
        out = _corner_actions(_corners(T2), *_lap(brake_from=380))
        assert out[0]["action"] == "brake"
        assert out[1]["action"] != "brake"

    def test_onset_too_close_to_apex_belongs_to_next_corner(self):
        d, t, s, th, b, g = _lap()
        b[(d >= 485) & (d < 495)] = 1.0          # 15 m before T1 apex (500)
        out = _corner_actions(_corners(T2), d, t, s, th, b, g)
        assert out[0]["action"] != "brake"
        assert out[1]["action"] == "brake"
        assert out[1]["brake_start_m"] == pytest.approx(486, abs=2)

    def test_carries_corner_identity(self):
        c = _corner_actions(_corners(), *_lap())[0]
        assert (c["id"], c["short"], c["direction"]) == (1, "T1", "right")
        assert (c["entry_m"], c["apex_m"], c["exit_m"]) == (400, 500, 600)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_pro_lap.py -q`
Expected: collection ERROR with `ImportError: cannot import name '_corner_actions' from 'app.main'`

- [ ] **Step 3: Write the minimal implementation**

In `app/main.py`, insert this block immediately **above** the line `# Routes` (and its surrounding `# ----` banner):

```python
# ---------------------------------------------------------------------------
# 3D Learn mode: pro reference lap for the browser (GET /pro_lap)
# ---------------------------------------------------------------------------
PRO_LAP_STEP_M = 2.0          # sample spacing sent to the browser
BRAKE_SEARCH_LEAD_M = 250.0   # earliest brake point considered before a corner's entry
APEX_MARGIN_M = 30.0          # an onset this close to an apex belongs to the next corner
LIFT_THROTTLE = 0.9           # throttle below this (with no brake) counts as a lift
BRAKE_ON = 0.5                # reference brake is 0/1 but resampling leaves ramps


def _corner_actions(corners, d, t, speed, throttle, brake, gear) -> list:
    """Classify each corner of a pro lap as brake / lift / flat.

    All arrays are index-aligned along lap distance ``d`` (metres). Only brake
    *onsets* count, searched in ``[max(entry - 250, prev_apex - 30), apex - 30)``
    so a corner never inherits the previous corner's braking and an onset right
    at an apex is credited to the next corner (validated on Bahrain + Monza).
    """
    on = brake >= BRAKE_ON
    onsets = np.flatnonzero(on & ~np.r_[False, on[:-1]])    # off -> on transitions
    out, prev_apex = [], -np.inf
    for c in corners:
        entry, apex, exit_ = float(c["entry_m"]), float(c["apex_m"]), float(c["exit_m"])
        lo = max(entry - BRAKE_SEARCH_LEAD_M, prev_apex - APEX_MARGIN_M)
        hi = apex - APEX_MARGIN_M
        prev_apex = apex
        mine = onsets[(d[onsets] >= lo) & (d[onsets] < hi)]
        action, brake_start, brake_dur = "flat", None, None
        if mine.size:
            i0 = int(mine[0])
            released = np.flatnonzero(~on[i0:])
            i1 = i0 + (int(released[0]) - 1 if released.size else on.size - 1 - i0)
            action, brake_start = "brake", float(d[i0])
            brake_dur = round(float(t[i1] - t[i0]), 2)
        elif np.any((d >= lo) & (d < hi) & (throttle < LIFT_THROTTLE)):
            action = "lift"
        seg = np.flatnonzero((d >= entry) & (d <= exit_))
        k = int(seg[np.argmin(speed[seg])]) if seg.size else int(np.argmin(np.abs(d - apex)))
        out.append({
            "id": c.get("id"), "short": c.get("short"), "name": c.get("name"),
            "direction": c.get("direction"),
            "entry_m": entry, "apex_m": apex, "exit_m": exit_,
            "action": action, "brake_start_m": brake_start, "brake_duration_s": brake_dur,
            "min_speed_kmh": round(float(speed[k]), 1), "apex_gear": int(gear[k]),
        })
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_pro_lap.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_pro_lap.py
git commit -m "feat(api): classify pro-lap corners as brake/lift/flat

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `GET /pro_lap` endpoint

**Files:**

- Modify: `app/main.py` (imports near the top; new helpers after `_corner_actions`; a new route after the `/best_driver` route)
- Test: `tests/test_pro_lap.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pro_lap.py`:

```python
class TestProLapEndpoint:
    @pytest.fixture(scope="class")
    def payload(self):
        res = client.get("/pro_lap", params={"year": 2025, "driver": "NOR", "track": "bahrain"})
        assert res.status_code == 200, res.text
        return res.json()

    def test_meta(self, payload):
        m = payload["meta"]
        assert (m["driver"], m["year"], m["session"], m["track"]) == ("NOR", 2025, "Q", "bahrain")
        assert m["track_name"] == "Bahrain"
        assert m["lap_time"] == pytest.approx(90.267, abs=1e-3)

    def test_columns_same_length(self, payload):
        p = payload["points"]
        assert set(p) == {"d", "x", "y", "z", "t", "speed", "throttle", "brake", "gear", "steer"}
        n = len(p["d"])
        assert n > 2000
        assert all(len(v) == n for v in p.values())

    def test_units_are_metres(self, payload):
        x, y = np.array(payload["points"]["x"]), np.array(payload["points"]["y"])
        path = np.hypot(np.diff(x), np.diff(y)).sum()
        length = payload["meta"]["length_m"]
        assert abs(path - length) / length < 0.05

    def test_time_monotonic_and_matches_lap_time(self, payload):
        t = np.array(payload["points"]["t"])
        assert t[0] == 0
        assert np.all(np.diff(t) > 0)
        assert abs(t[-1] - payload["meta"]["lap_time"]) < 0.5

    def test_steer_normalised(self, payload):
        s = np.array(payload["points"]["steer"])
        assert s.min() >= -1 and s.max() <= 1
        assert np.abs(s).max() > 0.5                 # it actually turns

    def test_brake_points_before_apex(self, payload):
        brakes = [c for c in payload["corners"] if c["action"] == "brake"]
        assert len(brakes) >= 5
        for c in brakes:
            assert c["brake_start_m"] < c["apex_m"]

    def test_unknown_track_404(self):
        assert client.get("/pro_lap", params={"track": "atlantis"}).status_code == 404

    def test_path_traversal_rejected(self):
        assert client.get("/pro_lap", params={"track": "../pyproject"}).status_code == 404

    def test_no_data_lists_available(self, monkeypatch):
        import app.main as m

        def offline(*args, **kwargs):
            raise RuntimeError("offline")

        monkeypatch.setattr(m, "load_real_telemetry", offline)
        res = client.get("/pro_lap", params={"year": 2019, "driver": "XXX", "track": "bahrain"})
        assert res.status_code == 404
        body = res.json()
        assert body["detail"] == "no data"
        assert {"year": 2025, "driver": "NOR", "session": "Q", "track": "bahrain"} in body["available"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_pro_lap.py -q`
Expected:

- The 7 `TestCornerActions` tests pass.
- The `payload`-based tests ERROR with `assert 404 == 200` (the route doesn't exist yet).
- `test_no_data_lists_available` fails because `app.main` has no attribute `load_real_telemetry`.
- `test_unknown_track_404` and `test_path_traversal_rejected` already pass, because a missing route also returns 404. They become meaningful after Step 5.

- [ ] **Step 3: Update the imports**

In `app/main.py`, change:

```python
import os
import subprocess
```

to:

```python
import json
import os
import subprocess
```

Then change:

```python
from src.fastf1_loader import resolve_gp_name  # noqa: E402  (track -> official GP name)
```

to:

```python
from src.fastf1_loader import resolve_gp_name, load_real_telemetry  # noqa: E402
```

- [ ] **Step 4: Add the payload builder**

In `app/main.py`, add this immediately **after** `_corner_actions` (from Task 1):

```python
def _build_pro_lap(df, meta: dict, track_info: dict, step: float = PRO_LAP_STEP_M) -> dict:
    """A FastF1 reference lap -> compact column arrays (metres, seconds) for the 3D viewer."""
    d_full = df["lap_distance"].to_numpy(dtype=float)
    raw_x = df["world_position_X"].to_numpy(dtype=float)
    raw_y = df["world_position_Y"].to_numpy(dtype=float)
    raw_z = df["world_position_Z"].to_numpy(dtype=float)
    length = float(meta.get("track_length") or d_full[-1])

    # FastF1 positions are 1/10 m; derive the scale from the lap itself instead of
    # hardcoding 0.1, so any unit change upstream can't silently break the 3D scene.
    path = float(np.hypot(np.diff(raw_x), np.diff(raw_y)).sum())
    scale = length / path if path > 0 else 1.0
    xs, ys, _ = _smooth_loop_xy(raw_x * scale, raw_y * scale)
    zs = smooth(raw_z * scale, window=25)

    d = np.arange(0.0, d_full[-1] + 1e-9, step)

    def col(values):
        return np.interp(d, d_full, np.asarray(values, dtype=float))

    speed, throttle, brake = col(df["speed_kmh"]), col(df["throttle"]), col(df["brake"])
    gear = np.rint(col(df["gear"])).astype(int)
    v = np.maximum(speed / 3.6, 1.0)
    t = np.concatenate([[0.0], np.cumsum(np.diff(d) / ((v[1:] + v[:-1]) / 2))])
    lap_time = float(meta.get("lap_time") or t[-1])
    t *= lap_time / t[-1]                      # integration drifts ~0.1 s; pin to the real lap time

    steer = _ref_steering_from_curvature(df, d_full, d) or [0.0] * d.size
    steer = [0.0 if s is None else s for s in steer]

    def rnd(values, digits):
        return [round(float(x), digits) for x in values]

    return {
        "meta": {"gp_name": meta.get("gp_name"), "lap_time": round(lap_time, 3),
                 "length_m": round(length, 1)},
        "points": {
            "d": rnd(d, 1), "x": rnd(col(xs), 2), "y": rnd(col(ys), 2), "z": rnd(col(zs), 2),
            "t": rnd(t, 3), "speed": rnd(speed, 1), "throttle": rnd(throttle, 3),
            "brake": rnd(brake, 3), "gear": gear.tolist(), "steer": rnd(steer, 3),
        },
        "corners": _corner_actions(track_info.get("corners", []), d, t, speed, throttle, brake, gear),
    }


def _available_pro_laps() -> list:
    """Cached reference laps, parsed from data/reference_laps/<year>_<drv>_<session>_<track>.pkl."""
    out = []
    for p in sorted((PROJECT_ROOT / "data" / "reference_laps").glob("*.pkl")):
        parts = p.stem.split("_", 3)
        if len(parts) == 4 and parts[0].isdigit():
            out.append({"year": int(parts[0]), "driver": parts[1].upper(),
                        "session": parts[2].upper(), "track": parts[3]})
    return out
```

- [ ] **Step 5: Add the route**

In `app/main.py`, add this immediately **after** the `best_driver` route function (ends with `return JSONResponse({"driver": code, ...})`):

```python
@app.get("/pro_lap")
def pro_lap(year: int = DEFAULT_YEAR, driver: str = "NOR", track: str = "",
            session: str = "Q") -> JSONResponse:
    """A pro reference lap as compact column arrays for the 3D Learn mode.

    Served from data/reference_laps/*.pkl when cached, otherwise fetched via
    FastF1 (slow on first call). 404 with ``available`` combos when no data.
    """
    track = (track or "").strip().lower()
    if track not in {t["key"] for t in _discover_tracks()}:   # also blocks path tricks
        raise HTTPException(status_code=404, detail="unknown track")
    driver = (driver or "").strip().upper()
    session = (session or "Q").strip().upper()
    try:
        df, meta = load_real_telemetry(driver, int(year), session, track)
    except Exception:
        traceback.print_exc()
        return JSONResponse(status_code=404,
                            content={"detail": "no data", "available": _available_pro_laps()})
    track_info = json.loads((PROJECT_ROOT / "tracks" / f"{track}.json").read_text(encoding="utf-8"))
    payload = _build_pro_lap(df, meta, track_info)
    payload["meta"].update({
        "driver": driver, "year": int(year), "session": session, "track": track,
        "track_name": SHORT_NAMES.get(track, track_info.get("name", track)),
    })
    return JSONResponse(payload)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_pro_lap.py -q`
Expected: `16 passed`

- [ ] **Step 7: Run the full suite (nothing else broke)**

Run: `python3 -m pytest -q`
Expected: `36 passed`

- [ ] **Step 8: Check the payload size**

Run:

```bash
python3 -c "
from fastapi.testclient import TestClient; from app.main import app
r = TestClient(app).get('/pro_lap', params={'track': 'bahrain'}); print(r.status_code, len(r.content)//1024, 'KB')"
```

Expected: `200` and a size under ~300 KB (the spec targets ≈150 KB; the exact value depends on number formatting, and anything below 300 KB is acceptable).

- [ ] **Step 9: Commit**

```bash
git add app/main.py tests/test_pro_lap.py
git commit -m "feat(api): add GET /pro_lap for the 3D Learn mode

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Pure playback helpers (`core.js`)

**Files:**

- Create: `app/static/js/learn3d/core.js`
- Test: `tests/js/learn3d_core.test.mjs`

- [ ] **Step 1: Write the failing tests**

Create `tests/js/learn3d_core.test.mjs`:

```js
// Run: node --test tests/js/learn3d_core.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  bisect, stateAt, timeAtDistance, cuePoint, nextCue, cornerSeekRange, fmtTime,
} from "../../app/static/js/learn3d/core.js";

const P = {
  d: [0, 10, 20, 30], t: [0, 1, 2, 3],
  x: [0, 10, 20, 30], y: [0, 0, 0, 0], z: [0, 0, 0, 0],
  speed: [100, 200, 300, 400], throttle: [1, 1, 0, 0], brake: [0, 0, 1, 1],
  steer: [0, 0.5, 1, 0], gear: [3, 4, 5, 6],
};

const BRAKE = { action: "brake", brake_start_m: 100, entry_m: 150, exit_m: 200 };
const FLAT = { action: "flat", brake_start_m: null, entry_m: 5000, exit_m: 5100 };

test("bisect finds the interval and clamps at both ends", () => {
  assert.equal(bisect([0, 1, 2, 3], 1.5), 1);
  assert.equal(bisect([0, 1, 2, 3], -5), 0);
  assert.equal(bisect([0, 1, 2, 3], 99), 2);
  assert.equal(bisect([0, 1, 2, 3], 2), 2);
});

test("stateAt interpolates channels and picks the nearer gear", () => {
  const s = stateAt(P, 1.5);
  assert.equal(s.d, 15);
  assert.equal(s.speed, 250);
  assert.equal(s.steer, 0.75);
  assert.equal(s.gear, 5);
  assert.equal(s.heading, 0);
  assert.equal(stateAt(P, 1.2).gear, 4);
});

test("stateAt clamps past the end of the lap", () => {
  assert.equal(stateAt(P, 99).d, 30);
  assert.equal(stateAt(P, -1).d, 0);
});

test("timeAtDistance inverts distance to time", () => {
  assert.equal(timeAtDistance(P, 25), 2.5);
  assert.equal(timeAtDistance(P, 0), 0);
});

test("cuePoint uses the brake point, else the entry", () => {
  assert.equal(cuePoint(BRAKE), 100);
  assert.equal(cuePoint(FLAT), 5000);
});

test("nextCue returns the nearest cue ahead within the lookahead, wrapping the lap", () => {
  const cs = [BRAKE, FLAT];
  assert.deepEqual(nextCue(cs, 50, 5400), { corner: BRAKE, dist: 50 });
  assert.deepEqual(nextCue(cs, 4900, 5400), { corner: FLAT, dist: 100 });
  assert.deepEqual(nextCue(cs, 5390, 5400), { corner: BRAKE, dist: 110 });
  assert.equal(nextCue(cs, 400, 5400), null);
});

test("cornerSeekRange starts 150 m before the cue and ends at the exit", () => {
  assert.deepEqual(cornerSeekRange(BRAKE), [0, 200]);
  assert.deepEqual(cornerSeekRange(FLAT), [4850, 5100]);
});

test("fmtTime renders m:ss.t, truncating (never rounding up to :60.0)", () => {
  assert.equal(fmtTime(83.456), "1:23.4");
  assert.equal(fmtTime(5.04), "0:05.0");
  assert.equal(fmtTime(59.99), "0:59.9");
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `node --test tests/js/learn3d_core.test.mjs`
Expected: FAIL with `ERR_MODULE_NOT_FOUND` (`core.js` doesn't exist yet)

- [ ] **Step 3: Write the implementation**

Create `app/static/js/learn3d/core.js`:

```js
// Pure helpers for the 3D Learn mode — no DOM, no Three.js, so they run under
// `node --test tests/js/learn3d_core.test.mjs`.
//
// `points` is the column-array object from GET /pro_lap:
//   { d, t, x, y, z, speed, throttle, brake, gear, steer }  (all same length)

/** Index i of the interval arr[i]..arr[i+1] containing v (arr ascending), clamped to [0, n-2]. */
export function bisect(arr, v) {
  let lo = 0, hi = arr.length - 1;
  if (v <= arr[0]) return 0;
  if (v >= arr[hi]) return hi - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] <= v) lo = mid; else hi = mid;
  }
  return lo;
}

const LERP_KEYS = ["d", "x", "y", "z", "speed", "throttle", "brake", "steer"];

/** Car state at time t (s): interpolated channels, nearest gear, heading of the path in the XY plane. */
export function stateAt(points, t) {
  const i = bisect(points.t, t);
  const t0 = points.t[i], t1 = points.t[i + 1];
  const f = t1 > t0 ? Math.min(1, Math.max(0, (t - t0) / (t1 - t0))) : 0;
  const s = { t, i };
  for (const k of LERP_KEYS) s[k] = points[k][i] + (points[k][i + 1] - points[k][i]) * f;
  s.gear = f < 0.5 ? points.gear[i] : points.gear[i + 1];
  s.heading = Math.atan2(points.y[i + 1] - points.y[i], points.x[i + 1] - points.x[i]);
  return s;
}

/** Lap time (s) at lap distance d (m). */
export function timeAtDistance(points, d) {
  const i = bisect(points.d, d);
  const d0 = points.d[i], d1 = points.d[i + 1];
  const f = d1 > d0 ? Math.min(1, Math.max(0, (d - d0) / (d1 - d0))) : 0;
  return points.t[i] + (points.t[i + 1] - points.t[i]) * f;
}

/** Where the driver acts for a corner: the brake point, or the entry for lift/flat corners. */
export function cuePoint(c) {
  return c.action === "brake" ? c.brake_start_m : c.entry_m;
}

/** Nearest corner cue ahead of d within `lookahead` metres (wraps the lap), or null. */
export function nextCue(corners, d, lapLength, lookahead = 300) {
  let best = null;
  for (const c of corners) {
    let dist = cuePoint(c) - d;
    if (dist < 0) dist += lapLength;
    if (dist <= lookahead && (!best || dist < best.dist)) best = { corner: c, dist };
  }
  return best;
}

/** [from, to] lap distances for the corner-list replay: `lead` m before the cue to the exit. */
export function cornerSeekRange(c, lead = 150) {
  return [Math.max(0, cuePoint(c) - lead), c.exit_m];
}

/** 83.456 -> "1:23.4" (truncated to tenths). */
export function fmtTime(s) {
  const tenths = Math.floor(Math.max(0, s) * 10);
  const m = Math.floor(tenths / 600);
  return `${m}:${((tenths % 600) / 10).toFixed(1).padStart(4, "0")}`;
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `node --test tests/js/learn3d_core.test.mjs`
Expected: `# pass 8` and `# fail 0`

- [ ] **Step 5: Commit**

```bash
git add app/static/js/learn3d/core.js tests/js/learn3d_core.test.mjs
git commit -m "feat(web): add pure playback helpers for 3D Learn mode

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Page skeleton (HTML block and CSS)

**Files:**

- Modify: `app/static/index.html`
- Modify: `app/static/css/main.css`

No automated test (markup only). Verification happens in Task 8.

- [ ] **Step 1: Add the Three.js importmap**

In `app/static/index.html`, directly after:

```html
<link rel="stylesheet" href="static/css/main.css?v=__BUILD__" />
```

insert:

```html
<!-- Three.js for the 3D Learn mode (pinned; only loaded when that mode runs). -->
<script type="importmap">
{ "imports": { "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js" } }
</script>
```

- [ ] **Step 2: Give the upload section an id**

Replace:

```html
      <div class="form-section">
        <div class="section-heading"><span class="sh-num">1</span>
```

with:

```html
      <div class="form-section" id="fileSection">
        <div class="section-heading"><span class="sh-num">1</span>
```

- [ ] **Step 3: Add the mode button**

Replace:

```html
        </div>
        <div class="mode-desc" id="modeDesc">
```

with:

```html
          <button type="button" class="mode-btn" data-mode="learn_3d" role="tab" aria-selected="false">
            <span class="mb-ico" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2 3 7v10l9 5 9-5V7z"/><path d="M3 7l9 5 9-5M12 12v10"/></svg></span>
            <span class="mb-name">3D<br>Learn</span>
            <span class="mb-tag">Pro lap</span>
          </button>
        </div>
        <div class="mode-desc" id="modeDesc">
```

- [ ] **Step 4: Add the result block**

Directly **before** the line `      <!-- Corner cards -->`, insert:

```html
      <!-- 3D Learn: replay a pro lap on a 3D track (driven by static/js/learn3d/) -->
      <div class="result-block l3d" id="blkLearn3d">
        <div class="l3d-head">
          <div class="section-label">3D Pro Lap</div>
          <div class="l3d-views" role="tablist" aria-label="Side panel">
            <button type="button" class="l3d-view active" data-view="telemetry" role="tab">📈 Telemetry</button>
            <button type="button" class="l3d-view" data-view="corners" role="tab">🏁 Corners</button>
          </div>
        </div>
        <div class="l3d-msg" id="l3dMsg" role="status" hidden></div>
        <div class="l3d-main" id="l3dMain" data-view="telemetry">
          <div class="l3d-stage" id="l3dStage">
            <div class="l3d-hud l3d-hud-tl" id="l3dHudTop"></div>
            <div class="l3d-hud l3d-hud-bl" id="l3dHudBottom"></div>
            <svg class="l3d-minimap" id="l3dMinimap" aria-hidden="true"></svg>
            <div class="l3d-cams" role="group" aria-label="Camera">
              <button type="button" data-cam="tcam" class="active">T-cam</button>
              <button type="button" data-cam="chase">Chase</button>
              <button type="button" data-cam="top">Top</button>
            </div>
          </div>
          <div class="l3d-corners" id="l3dCorners"></div>
          <div class="l3d-telemetry" id="l3dTelemetry"></div>
        </div>
        <div class="l3d-controls" id="l3dControls">
          <button type="button" id="l3dPlay" aria-label="Play">▶</button>
          <span class="l3d-rates" role="group" aria-label="Playback speed">
            <button type="button" data-rate="0.25">0.25×</button>
            <button type="button" data-rate="0.5">0.5×</button>
            <button type="button" data-rate="1" class="active">1×</button>
            <button type="button" data-rate="2">2×</button>
          </span>
          <input type="range" id="l3dScrub" min="0" max="1000" value="0" step="1" aria-label="Lap position" />
          <span class="l3d-time" id="l3dTime">0:00.0 / 0:00.0</span>
        </div>
        <div class="l3d-keys">Space play/pause · ← → ±2 s · 1/2/3 camera</div>
      </div>
```

- [ ] **Step 5: Load the module**

Replace:

```html
<script src="static/js/main.js?v=__BUILD__"></script>
```

with:

```html
<script type="module" src="static/js/learn3d/index.js?v=__BUILD__"></script>
<script src="static/js/main.js?v=__BUILD__"></script>
```

- [ ] **Step 6: Widen the mode grid to 6 columns**

In `app/static/css/main.css`, inside the `.mode-grid {` rule (around line 767), replace:

```css
    grid-template-columns: repeat(5, 1fr);
```

with:

```css
    grid-template-columns: repeat(6, 1fr);
```

- [ ] **Step 7: Append the 3D styles**

Append to the **end** of `app/static/css/main.css`:

```css
/* ------------------------------------------------------------------
   3D Learn mode (#blkLearn3d) — driven by static/js/learn3d/
   ------------------------------------------------------------------ */
.l3d-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
.l3d-views { display: flex; gap: 4px; background: var(--surface-2); border-radius: 10px; padding: 3px; }
.l3d-views button, .l3d-cams button, .l3d-rates button {
  background: none; border: 0; color: var(--text-dim); font: 600 0.8rem var(--font-body);
  padding: 5px 10px; border-radius: 7px; cursor: pointer;
}
.l3d-views button.active, .l3d-cams button.active, .l3d-rates button.active { background: var(--accent); color: #fff; }
.l3d-main { display: grid; grid-template-columns: 1fr; gap: 10px; }
.l3d-main[data-view="corners"] { grid-template-columns: 1fr 300px; }
.l3d-main[data-view="telemetry"] .l3d-corners,
.l3d-main[data-view="corners"] .l3d-telemetry { display: none; }
.l3d-stage { position: relative; height: 440px; border-radius: var(--radius-sm); overflow: hidden; background: #9cc3dd; }
.l3d-stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
.l3d-stage.l3d-nogl { display: grid; place-items: center; background: var(--surface-2); }
.l3d-stage.l3d-nogl::after { content: "3D view needs WebGL — telemetry and corners still work."; color: var(--text-dim); font-size: 0.9rem; }
.l3d-hud { position: absolute; z-index: 1; color: #fff; background: rgba(0, 0, 0, 0.55); padding: 6px 9px; border-radius: 8px; font: 600 0.8rem var(--font-body); }
.l3d-hud-tl { left: 10px; top: 10px; }
.l3d-hud-bl { left: 10px; bottom: 10px; display: flex; align-items: flex-end; gap: 4px; }
.l3d-cue { color: #ffd400; }
.l3d-pedal { display: inline-block; width: 9px; height: 40px; background: #333; position: relative; }
.l3d-pedal i { position: absolute; left: 0; right: 0; bottom: 0; }
.l3d-speed { font: 700 1.6rem var(--font-display); margin: 0 4px 0 6px; line-height: 1; }
.l3d-minimap { position: absolute; z-index: 1; right: 10px; top: 10px; width: 96px; height: 96px; background: rgba(0, 0, 0, 0.4); border-radius: 8px; }
.l3d-cams { position: absolute; z-index: 1; right: 10px; bottom: 10px; display: flex; gap: 4px; background: rgba(0, 0, 0, 0.55); border-radius: 9px; padding: 3px; }
.l3d-cams button { color: #fff; }
.l3d-corners { max-height: 440px; overflow-y: auto; display: flex; flex-direction: column; gap: 2px; }
.l3d-crow {
  display: grid; grid-template-columns: 44px 1fr 48px; gap: 6px; text-align: left; padding: 7px 8px;
  background: var(--surface); border: 0; border-radius: 8px; color: var(--text); font: 0.8rem var(--font-body); cursor: pointer;
}
.l3d-crow:hover { background: var(--surface-hover); }
.l3d-crow.active { background: var(--minor-soft); color: #ffd27a; }
.l3d-chead { background: none; color: var(--text-mute); cursor: default; font-size: 0.72rem; }
.l3d-chead:hover { background: none; }
.l3d-telemetry { display: flex; flex-direction: column; gap: 6px; }
.l3d-trace { display: grid; grid-template-columns: 90px 1fr; align-items: center; gap: 8px; }
.l3d-trace-label { color: var(--text-dim); font-size: 0.75rem; }
.l3d-trace svg { width: 100%; height: 48px; cursor: crosshair; background: var(--surface); border-radius: 6px; }
.l3d-controls { display: flex; align-items: center; gap: 10px; margin-top: 10px; }
#l3dPlay { width: 38px; height: 34px; border: 0; border-radius: 8px; background: var(--accent); color: #fff; cursor: pointer; }
.l3d-rates { display: flex; gap: 2px; }
.l3d-controls input[type="range"] { flex: 1; accent-color: var(--accent); }
.l3d-time { font: 600 0.85rem var(--font-body); color: var(--text-dim); font-variant-numeric: tabular-nums; }
.l3d-keys { margin-top: 6px; font-size: 0.72rem; color: var(--text-mute); }
.l3d-msg { padding: 16px; color: var(--text-dim); }
.l3d-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.l3d-chip { background: var(--surface-2); border: 1px solid var(--border-strong); color: var(--text); border-radius: 999px; padding: 4px 10px; cursor: pointer; font: 0.8rem var(--font-body); }
@media (max-width: 768px) {
  .l3d-main[data-view="corners"] { grid-template-columns: 1fr; }
  .l3d-stage { height: 300px; }
  .l3d-corners { max-height: 260px; }
  .l3d-trace { grid-template-columns: 1fr; gap: 2px; }
  .l3d-controls { flex-wrap: wrap; }
  .l3d-keys { display: none; }
}
```

On phones, picking 🏁 Corners stacks the list under the 3D view. The view toggle is the "corners button" from the spec, so no extra overlay is needed.

- [ ] **Step 8: Smoke-check that the page still loads**

Run: `python3 -m app.main` in the background, then `curl -s http://127.0.0.1:8000/ | grep -c 'blkLearn3d\|importmap\|data-mode="learn_3d"'`
Expected: `3` or more. Stop the server afterwards.

- [ ] **Step 9: Commit**

```bash
git add app/static/index.html app/static/css/main.css
git commit -m "feat(web): add 3D Learn mode skeleton markup and styles

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Panels (HUD, minimap, telemetry traces, corner list)

**Files:**

- Create: `app/static/js/learn3d/panels.js`

These are DOM-only builders. The logic they rely on (`cuePoint`, `fmtTime`) is already tested in Task 3. They are verified visually in Task 8.

- [ ] **Step 1: Create the module**

Create `app/static/js/learn3d/panels.js`:

```js
// DOM / SVG panels for the 3D Learn mode: HUD, minimap, telemetry traces, corner list.
import { cuePoint, fmtTime } from "./core.js";

const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ESC[c]);

const ACTION_LABEL = { brake: "BRAKE", lift: "LIFT", flat: "FLAT" };

/** Top-left: who/where + countdown to the next cue. Bottom-left: pedals, speed, gear. */
export function renderHud(top, bottom, meta, s, cue) {
  top.innerHTML =
    `${esc(meta.driver)} · ${meta.year} ${esc(meta.session)} · ${esc(meta.track_name)} · ${fmtTime(meta.lap_time)}` +
    (cue ? `<br><span class="l3d-cue">${esc(cue.corner.short)} · ${ACTION_LABEL[cue.corner.action]} in ${Math.round(cue.dist)} m</span>` : "");
  bottom.innerHTML =
    `<span class="l3d-pedal" title="Throttle"><i style="height:${Math.round(s.throttle * 100)}%;background:var(--ok)"></i></span>` +
    `<span class="l3d-pedal" title="Brake"><i style="height:${Math.round(s.brake * 100)}%;background:var(--accent)"></i></span>` +
    `<span class="l3d-speed">${Math.round(s.speed)}</span>km/h · G${s.gear}`;
}

/** North-up outline of the circuit with a dot for the car. */
export function createMinimap(svg, points) {
  const xs = [], ys = [];
  for (let i = 0; i < points.x.length; i += 5) { xs.push(points.x[i]); ys.push(points.y[i]); }
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const span = Math.max(maxX - minX, maxY - minY) || 1, pad = span * 0.06;
  // SVG y grows downward, so plot -y to keep north up.
  svg.setAttribute("viewBox", `${minX - pad} ${-maxY - pad} ${span + 2 * pad} ${span + 2 * pad}`);
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
      `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${shade}` +
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
    `<div class="l3d-crow l3d-chead"><span>Turn</span><span>Brake point → min speed</span><span>Brake</span></div>` +
    corners.map((c, i) => {
      const where = c.action === "brake" ? `${Math.round(c.apex_m - c.brake_start_m)} m before apex` : c.action;
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
      rows.forEach((r, i) => r.classList.toggle("active", i === k));
    },
  };
}
```

- [ ] **Step 2: Syntax-check it**

Run: `node --check app/static/js/learn3d/panels.js && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/static/js/learn3d/panels.js
git commit -m "feat(web): add HUD, minimap, telemetry and corner panels for 3D Learn

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Three.js scene (track, car, markers, cameras)

**Files:**

- Create: `app/static/js/learn3d/scene.js`

Coordinates: world `(x, y, elevation)` maps to Three.js `(x, elevation, -y)`, centred on the track. That mapping is a proper rotation, so the track is not mirrored. A car built facing local `+x` and given `rotation.y = heading` points along the world heading `(cos h, sin h)`.

- [ ] **Step 1: Create the module**

Create `app/static/js/learn3d/scene.js`:

```js
// Three.js scene for the 3D Learn mode: track ribbon, car, corner markers, cameras.
// Imported dynamically by index.js, so a CDN/WebGL failure only disables the canvas.
import * as THREE from "three";
import { bisect, cuePoint } from "./core.js";

// ponytail: fixed 14 m track; real widths, kerbs and run-off aren't in the data.
const TRACK_HALF_WIDTH = 7;
const CAMS = {
  tcam: { back: 0.6, height: 1.25, ahead: 40, look: 0.8 },   // above the airbox
  chase: { back: 9, height: 3.2, ahead: 8, look: 0.5 },
};

/** Builds the scene into `container`. Throws if WebGL is unavailable. */
export function createScene(container, points, corners) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  container.prepend(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x9cc3dd);
  scene.fog = new THREE.Fog(0x9cc3dd, 400, 2500);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x556644, 1.2));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5);
  sun.position.set(300, 600, 200);
  scene.add(sun);

  // world (x, y, elev) -> three (x, elev, -y), centred so floats stay small
  const n = points.x.length;
  const cx = (Math.min(...points.x) + Math.max(...points.x)) / 2;
  const cy = (Math.min(...points.y) + Math.max(...points.y)) / 2;
  const zMin = Math.min(...points.z);
  const toThree = (x, y, z, out = new THREE.Vector3()) => out.set(x - cx, z - zMin, -(y - cy));
  const pos = Array.from({ length: n }, (_, i) => toThree(points.x[i], points.y[i], points.z[i]));
  const up = new THREE.Vector3(0, 1, 0);

  // Road ribbon: offset the centre line sideways by ±half width.
  const verts = new Float32Array(n * 6);
  const side = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    side.subVectors(pos[Math.min(n - 1, i + 1)], pos[Math.max(0, i - 1)]).cross(up).normalize()
      .multiplyScalar(TRACK_HALF_WIDTH);
    const p = pos[i];
    verts.set([p.x + side.x, p.y + 0.05, p.z + side.z, p.x - side.x, p.y + 0.05, p.z - side.z], i * 6);
  }
  const index = [];
  for (let i = 0; i < n; i++) {
    const a = 2 * i, b = a + 1, c = (2 * (i + 1)) % (2 * n), d = c + 1;   // last quad closes the loop
    index.push(a, c, b, b, c, d);
  }
  const road = new THREE.BufferGeometry();
  road.setAttribute("position", new THREE.BufferAttribute(verts, 3));
  road.setIndex(index);
  road.computeVertexNormals();
  scene.add(new THREE.Mesh(road, new THREE.MeshStandardMaterial({ color: 0x3a3d42, roughness: 0.9, side: THREE.DoubleSide })));

  // ponytail: flat ground at the lowest point; hilly tracks show the road raised above it.
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(6000, 6000),
    new THREE.MeshStandardMaterial({ color: 0x5d7a4a, roughness: 1 }));
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.5;
  scene.add(ground);

  // A thin box lying across the road at lap distance d.
  const stripe = (d, color, depth) => {
    const i = bisect(points.d, d);
    const a = pos[Math.max(0, i - 1)], b = pos[Math.min(n - 1, i + 1)];
    const m = new THREE.Mesh(new THREE.BoxGeometry(depth, 0.08, TRACK_HALF_WIDTH * 2),
      new THREE.MeshBasicMaterial({ color }));
    m.position.copy(pos[i]).y += 0.1;
    m.rotation.y = Math.atan2(-(b.z - a.z), b.x - a.x);
    scene.add(m);
  };
  stripe(points.d[0], 0xffffff, 1.5);                         // start / finish
  for (const c of corners) {
    if (c.action === "brake") stripe(cuePoint(c), 0xff2e2e, 2);
    else if (c.action === "lift") stripe(cuePoint(c), 0xff9900, 1.2);
    const apex = new THREE.Mesh(new THREE.SphereGeometry(1.2, 12, 8), new THREE.MeshBasicMaterial({ color: 0xffd400 }));
    apex.position.copy(pos[bisect(points.d, c.apex_m)]).y += 1.2;
    scene.add(apex);
  }

  const car = buildCar();
  scene.add(car);

  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 5000);
  const resize = () => {
    const w = container.clientWidth || 1, h = container.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  const ro = new ResizeObserver(resize);
  ro.observe(container);
  resize();

  const fwd = new THREE.Vector3(), target = new THREE.Vector3();

  /** Place the car at state `s` (from core.stateAt), aim camera `cam`, render one frame. */
  function update(s, cam) {
    toThree(s.x, s.y, s.z, car.position).y += 0.05;
    car.rotation.y = s.heading;
    fwd.set(Math.cos(s.heading), 0, -Math.sin(s.heading));
    const p = car.position;
    if (cam === "top") {
      camera.up.copy(fwd);                                     // heading-up overhead view
      camera.position.set(p.x, p.y + 140, p.z);
      camera.lookAt(p);
    } else {
      const c = CAMS[cam] || CAMS.tcam;
      camera.up.copy(up);
      camera.position.copy(p).addScaledVector(fwd, -c.back);
      camera.position.y += c.height;
      target.copy(p).addScaledVector(fwd, c.ahead);
      target.y += c.look;
      camera.lookAt(target);
    }
    renderer.render(scene, camera);
  }

  function dispose() {
    ro.disconnect();
    scene.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    });
    renderer.dispose();
    renderer.domElement.remove();
  }

  return { update, dispose };
}

/** A generic open-wheel silhouette from primitives — original, no real livery or branding. Nose = +x. */
function buildCar() {
  const g = new THREE.Group();
  const paint = new THREE.MeshStandardMaterial({ color: 0xe8002d, roughness: 0.4, metalness: 0.2 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x15161a, roughness: 0.8 });
  const add = (geo, mat, x, y, z) => {
    const m = new THREE.Mesh(geo, mat);
    m.position.set(x, y, z);
    g.add(m);
    return m;
  };
  add(new THREE.BoxGeometry(4.6, 0.45, 0.9), paint, 0, 0.45, 0);      // tub + nose
  add(new THREE.BoxGeometry(1.6, 0.35, 1.5), paint, -0.6, 0.4, 0);    // sidepods
  add(new THREE.BoxGeometry(0.9, 0.5, 0.5), paint, -0.9, 0.9, 0);     // engine cover
  add(new THREE.BoxGeometry(0.3, 0.06, 1.9), paint, 2.5, 0.25, 0);    // front wing
  add(new THREE.BoxGeometry(0.35, 0.08, 1.1), dark, -2.3, 1.0, 0);    // rear wing
  for (const z of [0.5, -0.5]) add(new THREE.BoxGeometry(0.4, 0.6, 0.05), dark, -2.3, 0.75, z);  // wing pillars
  for (const [x, z] of [[1.7, 0.85], [1.7, -0.85], [-1.6, 0.85], [-1.6, -0.85]]) {
    add(new THREE.CylinderGeometry(0.36, 0.36, 0.4, 16), dark, x, 0.36, z).rotation.x = Math.PI / 2;
  }
  return g;
}
```

- [ ] **Step 2: Syntax-check it**

Run: `node --check app/static/js/learn3d/scene.js && echo OK`
Expected: `OK`. This only parses the file; it doesn't resolve the bare `three` import, which happens in the browser.

- [ ] **Step 3: Commit**

```bash
git add app/static/js/learn3d/scene.js
git commit -m "feat(web): add Three.js track, car and cameras for 3D Learn

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Entry module (playback, controls, `window.Learn3D`)

**Files:**

- Create: `app/static/js/learn3d/index.js`

- [ ] **Step 1: Create the module**

Create `app/static/js/learn3d/index.js`:

```js
// Entry point for the 3D Learn mode. main.js (a classic script) drives it via
//   window.Learn3D.load(url, onPickAvailable)   -> fetch /pro_lap and start
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

async function load(url, onPickAvailable) {
  unload();
  const seq = ++loadSeq;
  const res = await fetch(url);
  const body = await res.json().catch(() => ({}));
  if (seq !== loadSeq) return;
  if (!res.ok) { showMessage(body, onPickAvailable); return; }
  await start(body, seq);
}

function showMessage(body, onPick) {
  $("l3dMain").hidden = true;
  $("l3dControls").hidden = true;
  const avail = Array.isArray(body.available) ? body.available : [];
  const text = body.detail === "unknown track"
    ? "This track isn't supported."
    : "No lap data for this driver / track / session.";
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

  let scene = null;
  try {
    const { createScene } = await import("./scene.js");
    if (seq !== loadSeq) return;
    scene = createScene($("l3dStage"), P, data.corners);
  } catch (err) {
    console.warn("3D view unavailable:", err);
    $("l3dStage").classList.add("l3d-nogl");
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
  const dt = S.last ? Math.min(0.1, (now - S.last) / 1000) : 0;
  S.last = now;
  const lapTime = S.data.points.t[S.data.points.t.length - 1];
  if (S.playing) {
    S.t += dt * S.rate;
    if (S.stopAt != null && S.t >= S.stopAt) { S.t = S.stopAt; S.stopAt = null; setPlaying(false); }
    else if (S.t >= lapTime) S.t -= lapTime;                            // loop the lap
  }
  draw();
  S.raf = requestAnimationFrame(frame);
}

function draw() {
  const S = view, P = S.data.points;
  const s = stateAt(P, S.t);
  if (S.scene) S.scene.update(s, S.cam);
  renderHud($("l3dHudTop"), $("l3dHudBottom"), S.data.meta, s, nextCue(S.data.corners, s.d, P.d[P.d.length - 1]));
  S.minimap.set(s);
  S.tele.set(s.d);
  S.list.set(s.d);
  $("l3dScrub").value = String(Math.round(S.t * 10));
  $("l3dTime").textContent = `${fmtTime(S.t)} / ${fmtTime(P.t[P.t.length - 1])}`;
}

function seek(t) {
  const P = view.data.points;
  view.t = Math.min(Math.max(0, t), P.t[P.t.length - 1]);
  view.stopAt = null;
}

function setPlaying(on) {
  view.playing = on;
  $("l3dPlay").textContent = on ? "⏸" : "▶";
  $("l3dPlay").setAttribute("aria-label", on ? "Pause" : "Play");
}

function setRate(r) {
  view.rate = r;
  document.querySelectorAll("[data-rate]").forEach((b) => b.classList.toggle("active", +b.dataset.rate === r));
}

function setCam(cam) {
  view.cam = cam;
  store.set("l3d_cam", cam);
  document.querySelectorAll("[data-cam]").forEach((b) => b.classList.toggle("active", b.dataset.cam === cam));
}

function setView(name) {
  $("l3dMain").dataset.view = name;
  store.set("l3d_view", name);
  document.querySelectorAll(".l3d-view").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
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
  if (/^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(e.target.tagName)) return;
  if (e.code === "Space") { e.preventDefault(); setPlaying(!view.playing); }
  else if (e.key === "ArrowLeft") seek(view.t - 2);
  else if (e.key === "ArrowRight") seek(view.t + 2);
  else if (CAM_KEYS[e.key]) setCam(CAM_KEYS[e.key]);
});

window.Learn3D = { load, unload };
```

- [ ] **Step 2: Syntax-check it**

Run: `node --check app/static/js/learn3d/index.js && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/static/js/learn3d/index.js
git commit -m "feat(web): add 3D Learn playback loop and controls

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Wire the mode into `main.js`

**Files:**

- Modify: `app/static/js/main.js`

- [ ] **Step 1: Add the English strings**

Replace:

```js
      "btn.runAnalysis.track_map":    "Run Track Explorer",
```

with:

```js
      "btn.runAnalysis.track_map":    "Run Track Explorer",
      "btn.runAnalysis.learn_3d":     "Load 3D Lap",
```

Replace:

```js
      "mode.trackMap.desc":    "Interactive circuit map: hover the track to see your speed, throttle, brake and gear at any point.",
```

with:

```js
      "mode.trackMap.desc":    "Interactive circuit map: hover the track to see your speed, throttle, brake and gear at any point.",
      "mode.learn3d.name":     "3D\nLearn",
      "mode.learn3d.tag":      "Pro lap",
      "mode.learn3d.desc":     "Watch a real F1 driver's lap on a 3D track — see where they brake, lift and get back on the throttle. No CSV needed.",
```

- [ ] **Step 2: Add the Japanese strings**

Replace:

```js
      "btn.runAnalysis.track_map":    "コースエクスプローラーを実行",
```

with:

```js
      "btn.runAnalysis.track_map":    "コースエクスプローラーを実行",
      "btn.runAnalysis.learn_3d":     "3Dラップを読み込む",
```

Replace:

```js
      "mode.trackMap.desc":    "インタラクティブなサーキットマップ：コース上でホバーしてどの地点でも速度・スロットル・ブレーキ・ギアを確認。",
```

with:

```js
      "mode.trackMap.desc":    "インタラクティブなサーキットマップ：コース上でホバーしてどの地点でも速度・スロットル・ブレーキ・ギアを確認。",
      "mode.learn3d.name":     "3D\n学習",
      "mode.learn3d.tag":      "プロのラップ",
      "mode.learn3d.desc":     "実際のF1ドライバーのラップを3Dコースで再生 — ブレーキ、リフト、再加速のポイントを確認。CSV不要。",
```

- [ ] **Step 3: Register the mode**

Replace:

```js
    coaching: "coaching", track_map: "trackMap",
  };
```

with:

```js
    coaching: "coaching", track_map: "trackMap", learn_3d: "learn3d",
  };
```

In `const MODES = {`, replace:

```js
      blocks: ["blkExplorer"],
    },
  };
  const ALL_BLOCKS = [
    "blkOverview", "blkSolo", "blkTiming", "blkLapChart", "blkCompStrip",
    "blkExplorer", "blkCorners", "blkKeyCorners", "blkSummary", "blkActionPlan", "blkSeverity",
  ];
```

with:

```js
      blocks: ["blkExplorer"],
    },
    learn_3d: {
      label: "Load 3D Lap",
      tag: "Pro lap",
      needsReference: true,
      desc: "3D Learn — replay a real F1 driver's lap on a 3D track. No CSV needed.",
      blocks: ["blkLearn3d"],
    },
  };
  const ALL_BLOCKS = [
    "blkOverview", "blkSolo", "blkTiming", "blkLapChart", "blkCompStrip",
    "blkExplorer", "blkCorners", "blkKeyCorners", "blkSummary", "blkActionPlan", "blkSeverity",
    "blkLearn3d",
  ];
```

- [ ] **Step 4: Switch the form for the 3D mode**

In `applyModeUI`, replace:

```js
      refOpt.textContent = "Track only";
      refHintText.textContent =
        "This mode analyses your lap on its own — only the track matters. Driver, year and session are ignored.";
    }
  }
```

with:

```js
      refOpt.textContent = "Track only";
      refHintText.textContent =
        "This mode analyses your lap on its own — only the track matters. Driver, year and session are ignored.";
    }

    // 3D Learn needs no CSV: hide the upload + lap picker, use the manual track
    // dropdown, and let the button run without a file.
    const is3d = mode === "learn_3d";
    $("fileSection").hidden = is3d;
    lapBlock.hidden = is3d;
    if (is3d) {
      refOpt.textContent = "Pro lap to watch";
      refHintText.textContent = "Pick the driver, year, session and track whose lap you want to study in 3D.";
      showManualTrack();
      analyzeBtn.disabled = false;
    } else {
      if (window.Learn3D) window.Learn3D.unload();
      if (!selectedFile) resetTrackDetection();
      analyzeBtn.disabled = !selectedFile;
    }
  }
```

- [ ] **Step 5: Fix the loading-overlay subtitle for this mode**

In `showLoadingOverlay`, replace:

```js
    $("loadingOverlaySub").textContent = needsRef
```

with:

```js
    $("loadingOverlaySub").textContent = currentMode === "learn_3d"
      ? (currentLang === "ja" ? `${driver}の${trackName}ラップを読み込み中` : `Loading ${driver}'s ${trackName} lap`)
      : needsRef
```

The rest of that ternary chain stays unchanged.

- [ ] **Step 6: Route the Run button**

Replace:

```js
  async function analyze() {
    if (!selectedFile) { showError("Please choose a telemetry CSV first."); return; }
```

with:

```js
  // 3D Learn: no upload — load the pro lap straight into the viewer.
  async function runLearn3d() {
    hideError();
    analyzeBtn.classList.add("loading");
    analyzeBtn.disabled = true;
    showLoadingOverlay();
    const qs = new URLSearchParams({
      year: $("yearSel").value, driver: $("driver").value || "NOR",
      session: $("session").value, track: $("track").value,
    });
    applyResultLayout("learn_3d");
    results.classList.add("show");
    try {
      if (!window.Learn3D) throw new Error("The 3D viewer didn't load — check your connection and reload.");
      await window.Learn3D.load(apiUrl("/pro_lap?" + qs), (c) => {
        $("yearSel").value = String(c.year);
        $("driver").value = c.driver;
        $("session").value = c.session;
        $("track").value = c.track;
        runLearn3d();
      });
      results.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      results.classList.remove("show");
      showError(apiErrorMessage(err, "Couldn't load the pro lap."));
    } finally {
      hideLoadingOverlay();
      analyzeBtn.classList.remove("loading");
      analyzeBtn.disabled = false;
    }
  }

  async function analyze() {
    if (currentMode === "learn_3d") return runLearn3d();
    if (!selectedFile) { showError("Please choose a telemetry CSV first."); return; }
```

- [ ] **Step 7: Syntax-check**

Run: `node --check app/static/js/main.js && echo OK`
Expected: `OK`

- [ ] **Step 8: Commit**

```bash
git add app/static/js/main.js
git commit -m "feat(web): wire 3D Learn mode into the analysis form

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: End-to-end verification in the browser

**Files:** none (verification only; fix forward in the relevant task's file if anything fails)

- [ ] **Step 1: Start the app**

Run (in the background): `python3 -m app.main`
Expected: Uvicorn is serving `http://127.0.0.1:8000`

- [ ] **Step 2: Drive it with Playwright (MCP browser tools)**

1. Navigate to `http://127.0.0.1:8000`.
2. Click the mode button `[data-mode="learn_3d"]`.
   - Expect `#fileSection` to be hidden, the track `<select>` to be visible, and the Run button to be enabled with the label "Load 3D Lap".
3. Set `#yearSel` = 2025, `#driver` = NOR, `#session` = Q, `#track` = bahrain, then click `#analyzeBtn`.
4. Wait for `#blkLearn3d canvas`.
   - Take a screenshot `l3d-tcam.png`. Expect the road, the red car from behind, the HUD top-left, the minimap top-right, and four traces below.
5. Click `▶`, wait 3 s, and screenshot. The HUD speed and the cursor position should have changed.
6. Click `[data-cam="chase"]` and screenshot, then click `[data-cam="top"]` and screenshot.
7. Click `[data-view="corners"]` and screenshot. Expect the corner list on the right.
   - Click the T4 row. Expect playback at `0.5×` that stops by itself after T4's exit.
8. Set `#driver` = HAM and click `#analyzeBtn` again.
   - If FastF1 can't fetch the lap (offline), expect the "No lap data…" message with chips. Click the "2025 · NOR · Q · bahrain" chip; the lap should load.
   - If you're online, HAM's lap loads after the download. Both outcomes are fine.
9. Resize to 390×844 and screenshot both views. Expect a stacked layout with no horizontal scroll.
10. Check the browser console. Expect no errors (the `3D view unavailable` warning only appears if WebGL is missing).
11. Click another mode (e.g. Lap Comparison). Expect `#fileSection` back and the run button disabled until a CSV is chosen.

- [ ] **Step 3: Check the CDN failure path**

Block `cdn.jsdelivr.net` (Playwright route abort), reload, and run step 2.3 again.
Expected: the stage shows "3D view needs WebGL — telemetry and corners still work." Traces, corner list and playback still work.

- [ ] **Step 4: Run the full test suites**

Run: `python3 -m pytest -q && node --test tests/js/learn3d_core.test.mjs`
Expected: `36 passed`, plus `# pass 8` / `# fail 0`

- [ ] **Step 5: Stop the server and commit any fixes**

```bash
git add -A app tests
git commit -m "fix(web): polish 3D Learn after browser verification

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Skip this commit if nothing changed.

---

## Out of scope (later sub-projects)

- **#2** You-vs-pro two-car comparison in 3D.
- **#3** Brake-timing rhythm game.
- **#4** Drivable car.

Scenery, real track width and kerbs are also out (see spec §1 / §8). The spec waives the example-notebook rule because this work adds no new analysis algorithm.
