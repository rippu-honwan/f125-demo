# 3D Drive: drive the lap yourself against the pro's ghost

Sub-project #4 of the 3D plan (see `2026-10-15-3d-pro-lap-viewer-design.md` §1). It builds on #1 (3D Learn), #2 (multi-car scene, `gapAt`) and #3 (the practice host-API pattern and results card).

## 1. Goal

In 3D Learn, the player drives the loaded lap with the keyboard (or touch buttons on a phone). The car is driven by a simple physics model whose limits come from the pro's own telemetry. The pro's car runs alongside as a ghost, and a racing line shows where the pro brakes and accelerates. After a 3-2-1 countdown the player does one flying lap. A live gap is shown while driving; at the end a results card compares every corner with the pro.

### Non-goals

- Gearbox, engine revs, tyre wear, fuel, DRS and ERS.
- More than one lap per run, and saved replays.
- Manual steering on touch screens.
- Real track widths, kerbs and walls: the data has none, so the road stays the fixed 14 m ribbon.
- Backend changes: everything needed is already in `GET /pro_lap`.
- Changes to `src/`.

## 2. Decisions

| Question | Decision |
| --- | --- |
| Controls | Two modes. **Assist**: throttle and brake only, and the steering follows the pro's line. **Manual**: the player steers as well. Both use the same physics. |
| Physics | Data-driven. Grip, acceleration and braking limits are taken from the pro's lap, so every track works without tuning. |
| Feedback | Lap time, a live gap to the pro, and a per-corner results table. |
| Touch | Phones get on-screen GAS and BRAKE buttons and Assist only. |
| Start | A 3-2-1-GO countdown, then a flying start from the pro's speed at d = 0. |
| Entry point | A **Drive** button in 3D Learn next to **Practice**. Compare has neither. Drive and Practice cannot be open together. |

## 3. Model and simulation: `learn3d/drive_sim.js` (pure, node-tested)

Units inside the simulation are SI: m, s and m/s. `points.speed` is km/h and is converted on the way in.

### 3.1 `buildModel(points)` (once per lap)

- **Curvature `kappa[i]`** at every point.
  - The heading at i is the direction of the chord from the point 15 m back to the point 15 m ahead. Indices wrap, because the lap is a closed loop.
  - The curvature is the change in that heading over the same ±15 m, unwrapped. Positive means a left turn.
  - Chords are used rather than neighbouring points because FastF1 positions stall and jitter by about a metre against `d`: Jeddah has 14 m of `d` where the position stands still or reverses. Neighbour headings there gave impossible 15 m radii at 200 km/h.
- **`aLatMax(v)`**, the lateral grip available at speed v:
  1. Take the pro's `v² · |kappa|` in 10 km/h speed bins.
  2. Keep the 95th percentile of each bin. A bin with fewer than 10 samples counts as empty.
  3. Fill empty bins from the nearest filled bin.
  4. Apply a running maximum over speed, because downforce means grip never falls as speed rises.
  5. Cap at 6.5 g, which no F1 car exceeds; the cap guards against sparse top-speed bins.
  6. Multiply by `GRIP_MARGIN = 1.0`.
- **`accelMax(v)`** and **`brakeMax(v)`**: build these the same way from the pro's `dv/dt = (v[i+1]² − v[i]²) / (2·Δd)`.
  - `accelMax` takes samples at full throttle (≥ 0.95) with `dv/dt > 0`.
  - `brakeMax` takes samples with brake ≥ 0.5 and `dv/dt < 0`.
  - Neither gets a running maximum.
  - `accelMax` is 0 above the pro's top speed.
- **Coasting drag**, with no throttle and no brake: `COAST_K · v²` with `COAST_K = 0.0008` per metre.
- **`gearAt(v)`**: the pro's most common gear in that speed bin. It is shown in the HUD only.
- **`lapLength`**: `points.d` at the last point.

### 3.2 Car state and `step(model, car, input, dt)`

Car state is `{ s, n, psi, v, t, offTrack, invalid }`:

- `s`: distance along the pro's line;
- `n`: sideways offset from the line, positive to the left;
- `psi`: heading relative to the line;
- `t`: time since GO.

`input` is `{ throttle: 0|1, brake: 0|1, steer: −1..1 }`.

The frame time is integrated in fixed steps of `DT = 1/120` s, with frame time capped at 0.1 s.

**Longitudinal.**
- Brake wins over throttle.
- `dv/dt` is `+accelMax(v)` on throttle, `−brakeMax(v)` on the brake, and `−COAST_K·v²` otherwise.
- `v` never goes below 0.

**Steering.**
- The steering curvature is `kSteer = steer · aLatMax(v) / max(v, 5)²`. Grip therefore caps how tight the car can turn at a given speed, and a car that enters a corner too fast runs wide on its own.
- **Manual**: the key axis eases toward its target at 4 per second and back to 0 at 6 per second.
- **Assist**: `steer` comes from a pure-pursuit controller that aims at the line `L = max(10, 0.5·v)` m ahead.
  - The target curvature is `kappa(s) − 2·(n + L·psi) / L²`.
  - Dividing it by the curvature cap gives `steer`, clamped to ±1.
  - ←/→ are ignored in this mode.

**Kinematics** (`kappa` is interpolated at `s`):

```
ds/dt   = v·cos(psi) / (1 − n·kappa)
dn/dt   = v·sin(psi)
dpsi/dt = v·kSteer − kappa·ds/dt
```

**Track limits.**
- When `|n| > HALF_WIDTH = 7` m the car is on grass:
  - an extra 8 m/s² of drag;
  - grip halved;
  - `invalid = true` for the rest of the lap.
- At `|n| ≥ WALL = 20` m, `n` is clamped and `psi` is set to 0. While the car touches the wall it loses an extra 15 m/s².
- When `|psi| > π/2` the car has spun. It is put back with `n = 0`, `psi = 0` and `v = 60 km/h`.

**Lap end.** When `s` passes `lapLength`, the exact finish time is interpolated within the step and `finished = true`.

**Trace.** Every step also appends `s`, `t`, `v` and `brake` to the trace that the HUD gap and the results read.

### 3.3 Helpers

- `pose(points, car)`: the world `{x, y, z, heading}`. It takes the point on the line at `s`, moves it `n` along the left normal, and adds `psi` to the line heading. The scene draws this pose as a car.
- `gapAt(trace, points, s)`: your time at `s` minus `timeAtDistance(points, s)`.
- `cornerStats(trace, points, corners)` gives one row per corner:
  - **Brake point**: the first `s` with the brake on in `[brake_start_m − 150, apex_m]`. It is compared with `brake_start_m` and is `null` if the player didn't brake. Lift and flat corners show "—" when the pro didn't brake either.
  - **Min speed**: the lowest `v` in `[entry_m, exit_m]`, compared with the pro's `min_speed_kmh`.
  - **Time lost**: `gapAt(exit_m) − gapAt(entry_m − 150)`.

## 4. UI: `learn3d/drive.js`

`drive.js` follows the host-API pattern of `practice.js` and exports `attach(host)`, `detach()`, `isOpen()`, `frame(ts)`, `keydown(e)` and `keyup(e)`. The host provides `{points, corners, meta, store, setPlaying, setCues, drawCars(youPose, ghostState), setCamTarget}`; the plan settles the exact names.

### 4.1 Setup card

A `role="dialog"` card, the same as the Practice setup card, with:

- **Assist / Manual**. Manual is hidden when `(pointer: coarse)` matches.
- **Racing line** on/off.
- **Ghost** on/off.
- **Start** and **Cancel**.

The choices are stored in `store`. Opening the card pauses Learn and locks its controls with the same `l3d-practicing` class.

### 4.2 Countdown and driving

- Start or Retry shows 3, 2, 1 and GO, one second each, in the centre of the stage.
  - The car and the ghost wait at d = 0.
  - Inputs are tracked during the countdown but only act after GO.
  - On GO the car starts at the pro's `speed[0]`.
- `drive.js` owns the clock while driving:
  - the ghost uses `stateAt(points, car.t)`;
  - the camera follows the player's car;
  - Learn playback, the scrub bar and corner seeks stay locked.
- **HUD**: lap time, gap (`+1.84 s`, red when behind and green when ahead), speed and gear. "Off track · lap invalid" appears from the first off-track moment.
- **Keys**:
  - ↑ or W: throttle;
  - ↓, S or Space: brake;
  - ← → or A D: steer, Manual only;
  - R: restart with a new countdown;
  - Esc: leave.
- Brake and throttle inputs use the same multi-input set as practice, so two keys held and one released keeps the pedal down. Window blur releases all inputs.
- **Touch**: large GAS (right) and BRAKE (left) buttons, with pointer capture and the primary button only, as in practice.

### 4.3 Racing line and ghost (`scene.js`)

- **Racing line**: a thin line on the road centre, which is the pro's line. Each segment is coloured by the pro's inputs there: red where braking, green at full throttle (≥ 95 %), yellow otherwise.
- **Cars**: the Learn scene is built with two cars.
  - Car 0 is the pro: normal outside Drive, and the ghost (about 35 % opacity, hideable) in Drive.
  - Car 1 is the player, shown only in Drive.
- The scene takes a world pose for a car directly, so the player's car can leave the line.

### 4.4 Results card

- **Header**: `Lap 1:33.912 · pro 1:28.197 · +5.715 s`.
- **Best line**:
  - "New best!" or "Best: 1:33.100";
  - "Invalid lap — not counted" when off track.
- The best lap is stored under `l3d_drive_best_${year}_${driver}_${session}_${track}_${mode}`.
- **Table**: Turn, Brake point (`12 m early` / `4 m late` / `—`), Min speed (`98 vs 104 km/h`) and Time lost (`+0.21 s`), one row per corner from `cornerStats`.
- **Buttons**: Retry and Back to Learn. Focus moves to Retry, and Esc goes back to Learn and focuses the Drive button.

## 5. Testing

### node, in `tests/js/learn3d_drive.test.mjs`

**Synthetic tracks** (a straight followed by a constant-radius arc):

- full throttle on the straight never beats `accelMax`, and braking never beats `brakeMax`;
- in Assist at the pro's speed, the car stays within ±3 m of the line (the synthetic arcs start with a step in curvature);
- entering it well above the limit ends off track;
- in Manual, holding steer at 0 into the arc runs off track;
- a spin resets the car.

**Pro-lap bot test.**

- The fixture is `tests/js/fixtures/suzuka_pro.json`: the Suzuka reference lap at its native 2 m spacing, with the columns the simulation reads, rounded, about 140 KB.
- A bot in Assist mode tracks the pro's speed, using throttle below it and brake more than 5 km/h above it.
- It must finish the lap without going off track, within 3 % of the pro's lap time.

**Results.**

- `cornerStats` and `gapAt` on a hand-made trace.

### Browser, by hand with Playwright

- Drive's setup, countdown and HUD work, and the gap updates.
- R restarts.
- Esc restores Learn.
- A full lap driven by a scripted throttle and brake shows the results card.
- At 390 px the GAS and BRAKE buttons show and Manual is hidden.
- Compare has no Drive button.
- There are no console errors.

## 6. Risks

- **Noisy positions.** FastF1 x/y is noisy, so `kappa` can spike and inflate `aLatMax`. The ±15 m smoothing and the 95th-percentile bins are the guards, and the bot test is the check. If it fails, widen the smoothing before adding any per-track tuning.
- **Grip margin.** The prototype was checked on all 24 cached laps.
  - With the 95th percentile and `GRIP_MARGIN = 1.0`, a bot tracking the pro's speed in Assist stays on track on every lap, 0.1 % to 1.4 % quicker than the pro.
  - The same bot at 1.1× the pro's speed runs off at Monza, Silverstone, Hungaroring and Suzuka.
  - With the 90th percentile, the bot leaves the track at 5 laps.
  - `GRIP_MARGIN` stays the single knob.
