# Drive: braking limit by place

Follow-up to `2026-10-19-drive-grip-envelope-design.md`, which left one lap where a bot at the pro's own speed goes off: NOR Shanghai T14. This spec fixes it.

## 1. Problem

Braking comes from one lap-wide table, `brake(v)`: the 95th percentile of the pro's deceleration per 10 km/h bin.

At Shanghai T14 (about 4.7 km), NOR brakes at about 34 m/s², harder than the table allows at that speed. A car that brakes at the pro's point can't slow down in time, so the simple bot (brakes once 5 km/h over the pro's speed) goes off at 4701 m.

## 2. Decision

The braking limit at a place is the larger of:
- the lap-wide `brake(v)` table, as today;
- the hardest the pro braked within ±30 m of that place.

This only raises the limit where the pro braked harder than the table. It never lowers it.

**Why not the pro's braking alone.** In 75–85 % of braking-zone points the pro braked lighter than the table, because they didn't need more. Using only the pro's value there would stop a player who arrives too fast from braking harder than the pro to save the corner.

**Measured (prototype, 28 cached 2025 Q laps):**
- the simple bot following the pro's speed stays on track on 28 of 28 laps (27 today);
- the model-optimal lap is a median 6.5 % faster than the pro (6.1 % today);
- the result is the same with a ±15 m or ±50 m window.

### Non-goals

- **Acceleration by place.** In the prototype it narrowed the optimal-lap slack from 6.1 % to 5.3 %. But it caps a player who is slower than the pro on a straight at the pro's lower acceleration there, so it was left out.
- **Closing the remaining ~6 % slack.** Most of it comes from the lateral grip window (±15 m), which spreads the apex grip over the corner's entry and exit. Removing the window (grip = the pro's lateral at that exact point) leaves only 3 of 28 laps where even a look-ahead bot at the pro's speed stays on track.
- **Best-lap key.** It stays `_v2`. This change only loosens a few braking zones, and `_v2` was introduced in the same week.

## 3. Model (`app/static/js/learn3d/drive_sim.js`)

### `buildModel`

- Deceleration at point i is measured over the same chord as heading (±`SMOOTH_M`, 15 m): `(v[i-w]² − v[i+w]²) / (2 · (d[i+w] − d[i-w]))`, clamped to the lap's ends. Sample-to-sample deceleration is too noisy for a max.
- It counts only where the pro is braking (`brake[i] ≥ 0.5`) and decelerating; elsewhere it is 0.
- `brakeHere[i]` is the max of that over ±`BRAKE_WIN_M` (30 m), wrapping round the lap like `grip`. It is 0 where the pro didn't brake within the window.
- The model returns `brakeHere` alongside `brake`.

### `step`

When braking: `a = −max(lookup(m.brake, v), along(m, m.brakeHere, car.s))`. Nothing else changes.

### Constants

One new constant: `BRAKE_WIN_M = 30`. There is no margin.

## 4. Tests (`tests/js/learn3d_drive.test.mjs`)

- **Shanghai regression.** New fixture `tests/js/fixtures/shanghai_nor_pro.json`: NOR's 2025 Q lap points from `/pro_lap`, in the same format as the Spielberg fixture. The simple bot following the pro's speed finishes and stays on track. With today's model it goes off at 4701 m.
- **Optimal-lap bound on Spielberg.** VER's Spielberg lap is where `brakeHere` loosens the model most (optimal lap 6.1 % → 6.7 % faster than the pro). A new test keeps it at most 8 % faster.
- **Never stricter.** On the Suzuka fixture, one braking step from the pro's speed at every point slows the car by at least `lookup(m.brake, v) · DT`. This guards the "larger of" rule.
- **Optimal-lap helper.** `optimalLapTime(m)` in the test file brakes with the same larger-of limit. The Suzuka bound (at most 6 % faster than the pro) still holds.
- The other existing tests pass unchanged.

## 5. Risks

- **Envelope picks up noise.** A spike in the pro's speed trace could raise the limit locally. The ±15 m chord damps this; the largest single-sample glitch in the 28 laps (3.7 km/h) moves it by at most about 3 m/s². The optimal-lap tests bound the effect on Suzuka and Spielberg.
- **Slightly looser.** The median optimal-lap slack rises from 6.1 % to 6.5 %. That is the price of keeping the table as a floor.
