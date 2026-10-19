# Drive: Braking Limit by Place Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Where the pro braked harder than the lap-wide `brake(v)` table allows, the Drive car may brake that hard too, so NOR's Shanghai lap stays on track at the pro's own speed.

**Architecture:** In `drive_sim.js`, `buildModel` also returns `brakeHere` (one value per point of the pro's line): the pro's hardest deceleration within ±30 m, measured over the ±15 m heading chord, 0 where the pro didn't brake. `step` brakes with `max(lookup(m.brake, v), brakeHere at s)`. The grip window's max loop becomes a shared `windowMax` helper, since braking needs the same one.

**Tech Stack:** Plain ES modules; `node --test`.

**Spec:** `docs/superpowers/specs/2026-10-19-drive-brake-by-place-design.md`

**Ground rules:**
- Work on branch `feature/drive-brake-by-place` (already created from `origin/main`; the spec is committed there).
- Don't touch `src/`, `tracks/` or `data/raw/`.
- Leave the untracked files alone: `AGENTS.md`, `data/reference_laps/*.pkl`.
- End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

The code below was dry-run by the controller on a scratch copy: all 44 JS tests pass, the Suzuka optimal lap stays 4.4 % under the pro's, and swapping the "larger of" for "the pro's value, else the table" makes the never-weaker test fail.

---

### Task 1: braking by place

**Files:**
- Create: `tests/js/fixtures/shanghai_nor_pro.json`
- Modify: `app/static/js/learn3d/drive_sim.js` (constants block, `buildModel`, `step`)
- Test: `tests/js/learn3d_drive.test.mjs`

- [ ] **Step 1: Create the Shanghai fixture**

  The app server must be running (`python3 -m app.main`, port 8000).

  ```bash
  curl -s "http://localhost:8000/pro_lap?year=2025&driver=NOR&track=shanghai&session=Q" | python3 -c "
  import json, sys
  p = json.load(sys.stdin)['points']
  json.dump({'points': {k: p[k] for k in ['d', 't', 'x', 'y', 'z', 'speed', 'throttle', 'brake', 'gear']}},
            open('tests/js/fixtures/shanghai_nor_pro.json', 'w'), separators=(',', ':'))
  "
  python3 -c "import json; p = json.load(open('tests/js/fixtures/shanghai_nor_pro.json'))['points']; print(len(p['d']), p['t'][-1])"
  ```

  Expected: `2683 90.787` (about 137 KB).

- [ ] **Step 2: Write the failing tests**

  In `tests/js/learn3d_drive.test.mjs`, inside `function optimalLapTime(m)`, replace

  ```js
    for (let i = n - 2; i >= 0; i--) v[i] = Math.min(v[i], Math.sqrt(v[i + 1] ** 2 + 2 * lookup(m.brake, v[i + 1]) * ds));
  ```

  with

  ```js
    for (let i = n - 2; i >= 0; i--) {
      const brk = Math.max(lookup(m.brake, v[i + 1]), m.brakeHere[i + 1]);   // as in step()
      v[i] = Math.min(v[i], Math.sqrt(v[i + 1] ** 2 + 2 * brk * ds));
    }
  ```

  Append to the end of the file:

  ```js
  test("brake by place: NOR's Shanghai lap at its own speed stays on track (T14 brakes harder than the table)", () => {
    const S = JSON.parse(readFileSync(new URL("./fixtures/shanghai_nor_pro.json", import.meta.url))).points;
    const m = buildModel(S);
    const car = createCar(m, "assist");
    drive(m, car, proSpeed(S), { maxT: 200 });
    assert.ok(car.finished);
    assert.equal(car.invalid, false);              // the lap-wide table alone went off at 4701 m
  });

  test("brake by place: braking is never weaker than the lap-wide table", () => {
    const S = JSON.parse(readFileSync(new URL("./fixtures/suzuka_pro.json", import.meta.url))).points;
    const m = buildModel(S);
    for (let i = 0; i < S.d.length - 1; i += 5) {
      const v = S.speed[i] / KMH, car = { ...createCar(m, "assist"), s: S.d[i], v };
      step(m, car, { throttle: 0, brake: 1 });
      assert.ok(v - car.v >= lookup(m.brake, v) * DT - 1e-9, `at ${S.d[i]} m: ${v - car.v}`);
    }
  });
  ```

- [ ] **Step 3: Run the tests to make sure they fail**

  Run: `node --test tests/js/learn3d_drive.test.mjs`

  Expected: `# fail 2`.
  - The Suzuka optimal-lap test fails with a `TypeError`, because `m.brakeHere` is undefined.
  - The Shanghai test fails on `car.invalid`.
  - The never-weaker test passes already. It guards the rule in Step 4 against a later change to "the pro's value alone".

- [ ] **Step 4: Implement**

  In `app/static/js/learn3d/drive_sim.js`, after the line

  ```js
  const GRIP_WIN_M = 15;                         // grip at a point = the pro's max lateral g within ± this (m)
  ```

  add

  ```js
  const BRAKE_WIN_M = 30;                        // braking at a point may be as hard as the pro's hardest within ± this (m)
  ```

  In `buildModel`, after the line

  ```js
    const kappa = heading.map((_, i) => (hAt(i + w) - hAt(i - w)) / (2 * w * ds));
  ```

  add

  ```js
    /** Max of arr over ± halfM metres around each point, wrapping round the lap. */
    const windowMax = (arr, halfM) => {
      const g = Math.max(1, Math.round(halfM / ds));
      return arr.map((_, i) => {
        let x = 0;
        for (let j = i - g; j <= i + g; j++) x = Math.max(x, arr[at(j)]);
        return x;
      });
    };
  ```

  Replace

  ```js
    const gw = Math.max(1, Math.round(GRIP_WIN_M / ds));
    const grip = lat.map((_, i) => {
      let g = 0;
      for (let j = i - gw; j <= i + gw; j++) g = Math.max(g, lat[at(j)]);
      return Math.max(GRIP_FLOOR, g * GRIP_MARGIN);
    });
  ```

  with

  ```js
    const grip = windowMax(lat, GRIP_WIN_M).map((g) => Math.max(GRIP_FLOOR, g * GRIP_MARGIN));

    // Braking by place, on top of the lap-wide table: where the pro braked harder than the table allows
    // (NOR Shanghai T14, ~34 m/s²), braking at the pro's point still makes the corner. Measured over the
    // heading chord, because sample-to-sample deceleration is too noisy to take a max of.
    const dec = v.map((_, i) => {
      const a = Math.max(0, i - w), b = Math.min(np - 1, i + w);
      return points.brake[i] >= 0.5 ? Math.max(0, (v[a] ** 2 - v[b] ** 2) / (2 * (d[b] - d[a]))) : 0;
    });
    const brakeHere = windowMax(dec, BRAKE_WIN_M);
  ```

  In the `return { … }` of `buildModel`, replace

  ```js
      grip, accel, brake: binTable(brk, nb, PCT), gear,
  ```

  with

  ```js
      grip, accel, brake: binTable(brk, nb, PCT), brakeHere, gear,
  ```

  In `step`, replace

  ```js
    let a = input.brake ? -lookup(m.brake, v0) : input.throttle ? lookup(m.accel, v0) : -COAST_K * v0 * v0;
  ```

  with

  ```js
    let a = input.brake ? -Math.max(lookup(m.brake, v0), along(m, m.brakeHere, car.s)) : input.throttle ? lookup(m.accel, v0) : -COAST_K * v0 * v0;
  ```

  (`w`, `at`, `ds`, `np`, `d` and `v` already exist in `buildModel`. `along` is defined below it and hoisted.)

- [ ] **Step 5: Run the tests to make sure they pass**

  Run: `node --test tests/js/learn3d_core.test.mjs tests/js/learn3d_rhythm.test.mjs tests/js/learn3d_drive.test.mjs && python3 -m pytest -q`

  Expected: node `# fail 0` with 44 passing (42 before, plus the 2 new tests); pytest all pass (102).

- [ ] **Step 6: Commit**

  ```bash
  git add app/static/js/learn3d/drive_sim.js tests/js/learn3d_drive.test.mjs tests/js/fixtures/shanghai_nor_pro.json
  git commit -m "feat(web): Drive may brake as hard as the pro did at each place, never less than the lap table

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

---

### Task 2: browser check (controller)

- [ ] Restart the server on the new HEAD.
- [ ] In Playwright, load 3D Learn → Shanghai → Drive (Assist). Hold throttle, then brake into T1, and check there are no console errors and the car slows down.
