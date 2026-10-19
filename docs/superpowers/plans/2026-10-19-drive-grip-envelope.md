# Drive: Lateral Grip by Location Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Drive car's lateral grip comes from what the pro pulled at each place on the track, instead of a per-speed-bin table.

**Architecture:** In `drive_sim.js`, `buildModel` returns `grip` (one value per point of the pro's line): the max of the pro's v²|κ| within ±15 m, × 1.02, floored at 8 m/s². `step` reads it by position. The speed-bin `aLat` table goes. Drive's best-lap key gets a `_v2` suffix, because old times aren't comparable.

**Tech Stack:** Plain ES modules; `node --test`.

**Spec:** `docs/superpowers/specs/2026-10-19-drive-grip-envelope-design.md`

**Ground rules:**
- Work on branch `feature/drive-grip-envelope`.
- Don't touch `src/`, `tracks/` or `data/raw/`.
- Leave the untracked files alone: `AGENTS.md`, `data/reference_laps/*.pkl`.
- End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

The code below was dry-run by the controller on a scratch copy: all 17 drive tests pass, and the Suzuka optimal lap is 4.4 % under the pro's.

---

### Task 1: grip by location

**Files:**
- Create: `tests/js/fixtures/spielberg_ver_pro.json`
- Modify: `app/static/js/learn3d/drive_sim.js` (constants block, `buildModel`, `step`)
- Test: `tests/js/learn3d_drive.test.mjs`

- [ ] **Step 1: Create the Spielberg fixture**

  The app server must be running (`python3 -m app.main`, port 8000). `data/reference_laps/2025_ver_q_spielberg.pkl` is cached locally, so this needs no network.

  ```bash
  curl -s "http://localhost:8000/pro_lap?year=2025&driver=VER&track=spielberg&session=Q" | python3 -c "
  import json, sys
  p = json.load(sys.stdin)['points']
  json.dump({'points': {k: p[k] for k in ['d', 't', 'x', 'y', 'z', 'speed', 'throttle', 'brake', 'gear']}},
            open('tests/js/fixtures/spielberg_ver_pro.json', 'w'), separators=(',', ':'))
  "
  python3 -c "import json; p = json.load(open('tests/js/fixtures/spielberg_ver_pro.json'))['points']; print(len(p['d']), p['t'][-1])"
  ```

  Expected: `2151 64.xxx` (about 110 KB).

- [ ] **Step 2: Write the failing tests**

  In `tests/js/learn3d_drive.test.mjs`, change the import to:

  ```js
  import {
    KMH, DT, HALF_WIDTH, WALL, GRIP_FLOOR, GRIP_MARGIN, buildModel, lookup, createCar, step, advance, pose, gapAt, cornerStats,
    gearAt,
  } from "../../app/static/js/learn3d/drive_sim.js";
  ```

  In the test `buildModel reads curvature, lateral grip, acceleration and braking from the lap`, replace the line

  ```js
    assert.ok(Math.abs(lookup(M.aLat, V_ARC) - V_ARC ** 2 / R) < 1.5, `aLat ${lookup(M.aLat, V_ARC)}`);
  ```

  with

  ```js
    const g = M.grip[P.d.findIndex((d) => d >= ARC_MID)];
    assert.ok(Math.abs(g - (V_ARC ** 2 / R) * GRIP_MARGIN) < 1.5, `grip ${g}`);   // what the pro pulled here
    assert.equal(M.grip[P.d.findIndex((d) => d >= 300)], GRIP_FLOOR);            // straight: just enough to steer
  ```

  Append to the end of the file:

  ```js
  /** Fastest lap the model allows on the line: corner speed from grip, then accel/brake passes (s). */
  function optimalLapTime(m) {
    const n = m.d.length, ds = m.d[1] - m.d[0];
    const v = m.d.map((_, i) => Math.min(m.vTop, Math.sqrt(m.grip[i] / Math.max(Math.abs(m.kappa[i]), 1e-9))));
    v[0] = Math.min(v[0], m.v0);
    for (let i = 0; i < n - 1; i++) v[i + 1] = Math.min(v[i + 1], Math.sqrt(v[i] ** 2 + 2 * lookup(m.accel, v[i]) * ds));
    for (let i = n - 2; i >= 0; i--) v[i] = Math.min(v[i], Math.sqrt(v[i + 1] ** 2 + 2 * lookup(m.brake, v[i + 1]) * ds));
    let t = 0;
    for (let i = 0; i < n - 1; i++) t += ds / ((v[i] + v[i + 1]) / 2);
    return t;
  }

  test("grip by place: the fastest lap the model allows on Suzuka stays within 7 % of the pro", () => {
    const S = JSON.parse(readFileSync(new URL("./fixtures/suzuka_pro.json", import.meta.url))).points;
    const pro = S.t[S.t.length - 1], best = optimalLapTime(buildModel(S));
    assert.ok(best / pro > 0.93, `optimal ${best.toFixed(2)} s vs pro ${pro} s`);   // speed bins allowed ~10 % here
  });

  test("grip by place: VER's Spielberg lap at its own speed stays on track (no speed-bin hole)", () => {
    const S = JSON.parse(readFileSync(new URL("./fixtures/spielberg_ver_pro.json", import.meta.url))).points;
    const m = buildModel(S);
    const car = createCar(m, "assist");
    drive(m, car, proSpeed(S), { maxT: 200 });
    assert.ok(car.finished);
    assert.equal(car.invalid, false);              // the 80-90 km/h bin allowed 15 m/s² where the pro pulls 25
  });
  ```

- [ ] **Step 3: Run the tests to make sure they fail**

  Run: `node --test tests/js/learn3d_drive.test.mjs`

  Expected: `SyntaxError: ... does not provide an export named 'GRIP_FLOOR'`.

  (With only the exports added, the Spielberg test fails because the car goes off at about 1381 m, and the other two fail with a `TypeError` on `grip`. The controller checked this.)

- [ ] **Step 4: Implement**

  In `app/static/js/learn3d/drive_sim.js`, replace

  ```js
  // ponytail: the one grip knob. 1.0 = the pro's 95th-percentile lateral G per speed; the bot test holds at 1.0.
  export const GRIP_MARGIN = 1.0;
  ```

  with

  ```js
  // ponytail: the one grip knob. 1.0 = the most lateral g the pro pulled at this place on the track;
  // 2 % on top is slack for braking a touch late (without it Jeddah and Lusail go off at the pro's pace).
  export const GRIP_MARGIN = 1.02;
  export const GRIP_FLOOR = 8;                   // m/s²: steering still works on straights, below every corner apex
  const GRIP_WIN_M = 15;                         // grip at a point = the pro's max lateral g within ± this (m)
  ```

  Replace

  ```js
  const A_LAT_CAP = 6.5 * 9.81;                  // no F1 car pulls more; caps noise in sparse bins
  ```

  with

  ```js
  const A_LAT_CAP = 6.5 * 9.81;                  // no F1 car pulls more; caps curvature spikes
  ```

  In `buildModel`, replace

  ```js
    const nb = Math.ceil(Math.max(...v) / BIN) + 2;
    const aLat = binTable(v.map((vi, i) => [vi, vi * vi * Math.abs(kappa[i])]), nb, PCT);
    for (let i = 1; i < nb; i++) aLat[i] = Math.max(aLat[i], aLat[i - 1]);   // downforce: grip never drops with speed
  ```

  with

  ```js
    // Lateral grip by place, not by speed: a corner allows what the pro pulled in it. Speed bins mixed corners,
    // so one corner's limit came from another (too loose), or from corner exits (VER Spielberg: too strict).
    const lat = v.map((vi, i) => Math.min(A_LAT_CAP, vi * vi * Math.abs(kappa[i])));
    const gw = Math.max(1, Math.round(GRIP_WIN_M / ds));
    const grip = lat.map((_, i) => {
      let g = 0;
      for (let j = i - gw; j <= i + gw; j++) g = Math.max(g, lat[at(j)]);
      return Math.max(GRIP_FLOOR, g * GRIP_MARGIN);
    });

    const nb = Math.ceil(Math.max(...v) / BIN) + 2;
  ```

  In the `return { … }` of `buildModel`, replace

  ```js
      aLat: aLat.map((a) => Math.min(A_LAT_CAP, a) * GRIP_MARGIN), accel, brake: binTable(brk, nb, PCT), gear,
  ```

  with

  ```js
      grip, accel, brake: binTable(brk, nb, PCT), gear,
  ```

  In `step`, replace

  ```js
    const kMax = lookup(m.aLat, car.v) * (off ? 0.5 : 1) / Math.max(car.v, 5) ** 2;
  ```

  with

  ```js
    const kMax = along(m, m.grip, car.s) * (off ? 0.5 : 1) / Math.max(car.v, 5) ** 2;
  ```

  (`at` and `ds` already exist in `buildModel`. `along` is defined below it and hoisted.)

  Then run `grep -n "aLat" app/static/js/learn3d/*.js tests/js/*.mjs`. Expected: no output.

- [ ] **Step 5: Run the tests to make sure they pass**

  Run: `node --test tests/js/learn3d_core.test.mjs tests/js/learn3d_rhythm.test.mjs tests/js/learn3d_drive.test.mjs`

  Expected: `# fail 0`, with 42 passing (40 before, plus the 2 new tests).

- [ ] **Step 6: Commit**

  ```bash
  git add app/static/js/learn3d/drive_sim.js tests/js/learn3d_drive.test.mjs tests/js/fixtures/spielberg_ver_pro.json
  git commit -m "feat(web): Drive grip comes from what the pro pulled at each place, not per speed bin

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

---

### Task 2: new best-lap key and doc pointers

**Files:**
- Modify: `app/static/js/learn3d/drive.js`, where the `l3d_drive_best_` key is built
- Modify: `docs/superpowers/specs/2026-10-16-3d-drive-design.md`, the `aLatMax(v)` bullet

- [ ] **Step 1: Version the best-lap key**

  In `app/static/js/learn3d/drive.js`, replace

  ```js
    const key = `l3d_drive_best_${m.year}_${m.driver}_${m.session}_${m.track}_${car.mode}`;
  ```

  with

  ```js
    // v2: grip by place (2026-10-19); times from the speed-bin model aren't comparable, so they stay unread.
    const key = `l3d_drive_best_${m.year}_${m.driver}_${m.session}_${m.track}_${car.mode}_v2`;
  ```

- [ ] **Step 2: Point the old spec at the new model**

  In `docs/superpowers/specs/2026-10-16-3d-drive-design.md`, find the bullet that starts with `- **`aLatMax(v)`**, the lateral grip available at speed v:`. Add this line directly above it:

  ```markdown
  > Superseded on 2026-10-19 by `2026-10-19-drive-grip-envelope-design.md`: lateral grip is now per place on the track (`grip[i]`), not per speed bin.
  ```

- [ ] **Step 3: Checks**

  Run: `node --check app/static/js/learn3d/drive.js && node --test tests/js/learn3d_core.test.mjs tests/js/learn3d_rhythm.test.mjs tests/js/learn3d_drive.test.mjs && python3 -m pytest -q`

  Expected: node `# fail 0` (42); pytest all pass (102).

- [ ] **Step 4: Commit**

  ```bash
  git add app/static/js/learn3d/drive.js docs/superpowers/specs/2026-10-16-3d-drive-design.md
  git commit -m "feat(web): Drive best laps get a v2 key for the new grip model

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

---

### Task 3: browser check (controller)

- [ ] Restart the server on the new HEAD.
- [ ] In Playwright, load 3D Learn → Suzuka → Drive (Assist), and let the lap run for a few seconds with throttle held. Check there are no console errors, and that the car stays on the road through T1 at a sane speed.
- [ ] Confirm `localStorage` keys written after a finished lap end in `_v2`. Finishing a lap takes about 90 s in real time; if that is too slow, check the key in code review instead and say so.
