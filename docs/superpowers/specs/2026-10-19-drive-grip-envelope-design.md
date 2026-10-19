# Drive: grip by location, with a friction circle

Follow-up to `2026-10-16-3d-drive-design.md`. The Drive car's grip comes from the pro's lap. The current model is too loose on most tracks and too strict in places, and this spec replaces it.

## 1. Problem (measured with bots on the 28 cached 2025 Q laps)

Today's lateral limit is a table per 10 km/h speed bin: the 95th percentile of the pro's v²|κ| in that bin, made non-decreasing with speed. Any lateral demand up to that value is free. Neither braking nor accelerating costs grip while you corner.

The bot behind these numbers runs Assist, targets k × the pro's speed, and brakes bang-bang.

- **Too loose.** 11 of the 28 laps stay on track at k = 1.10, and Monaco stays on at k = 1.15. The model rewards over-speeding, so its fastest valid lap is at k = 1.06 on 15 laps. At Monaco, k = 1.10 laps 8 % faster than the pro.
- **Too strict in places.** VER's Spielberg lap goes off at k = 1.00, at 1381 m. The pro needs 25 m/s² there at 88 km/h, but the 80–90 km/h bin allows 15. That bin is dominated by low-g corner exits.
- **Why the bins fail.** A corner's allowance is set by whichever corner in the same speed bin the pro took hardest. It is not set by this corner.
- **Not the cause: road width.** A narrower road barely helps. With a half-width of 4.5 m instead of 7, Monaco's break point moves from k = 1.10 to 1.08. A slow corner taken 10 % too fast only drifts 2–3 m wider. What makes over-speeding pay is the free acceleration at full lateral load.

## 2. Goal

Each place on the track allows about what the pro used there, no more. Cornering harder leaves less grip for braking and accelerating.

### Non-goals

- Per-track road widths: there is no width data, and they barely help (§1).
- Tyre, temperature or weight modelling.
- UI changes, apart from the best-lap storage key (§5).

## 3. Model (`app/static/js/learn3d/drive_sim.js`)

### Grip envelope G(s)

At each point of the pro's line, compute:
- **ay:** v²|κ|, with κ from the existing ±15 m chord;
- **ax:** the longitudinal acceleration over the same chord, (v[i+w]² − v[i−w]²) / (2 · chord length);
- **total:** hypot(ax, ay).

`G[i]` is the maximum total over ±`GRIP_WIN_M` (15 m) around i, wrapping round the lap. It is multiplied by `GRIP_MARGIN`, capped at 6.5 g, and floored at `GRIP_FLOOR` (8 m/s²). The floor keeps steering possible on straights.

`buildModel` returns `grip` (the per-point G) in place of `aLat`. The speed-bin `aLat` table is deleted. `accel`, `brake`, `gear` and `vTop` are unchanged.

### Step

1. **Steering first.** Use `kMax = G(s) · (off ? 0.5 : 1) / max(v, 5)²` (grass halves it, as today). Assist pure pursuit and Manual are unchanged, and `kSteer` is clamped to ±`kMax`.
2. **Longitudinal second.**
   - Lateral use is `ay = v² |kSteer|`, and what is left is `room = sqrt(max(0, G² − ay²))`.
   - Throttle gives `min(accel(v), room)`. Brake gives `−min(brake(v), room)`.
   - Coasting drag, grass and wall drag, and the `vTop` cap are unchanged. Drag isn't grip.
3. **Then the kinematics,** as today.

All the constants (`GRIP_MARGIN`, `GRIP_WIN_M`, `GRIP_FLOOR`) are exported or top-level. `GRIP_MARGIN` is the one tuning knob, set by §4.

## 4. Calibration and acceptance

There are two tools, written as test helpers and reused for tuning.

- **Model-optimal lap (`optimalLap(m)`).** This is a quasi-steady-state lap-time simulation on the line.
  - The corner speed limit is `sqrt(G/|κ|)`, capped at `vTop`.
  - A forward pass applies `min(accel(v), room)` and a backward pass applies `min(brake(v), room)`. Both use room from the lateral use at the limit speed.
  - It returns the lap time.
  - This is the fastest lap the model allows. It replaces "k × the pro" bots as the measure of slack.
- **Look-ahead bot.** Assist targets the optimal-lap speed profile computed with k × the pro's speed as an extra cap.
  - At k = 1.00 it must stay on track.

**Acceptance on all 28 cached laps.** The audit dumps of `/pro_lap` live in the scratchpad; the plan copies the ones it needs into `tests/js/fixtures`.
1. The optimal lap is between −2 % and +1 % of the pro's time on every lap. Today's bins are expected to allow about −5 % to −10 %. The plan measures both.
2. The look-ahead bot at k = 1.00 stays on track on every lap, including VER Spielberg.
3. The look-ahead bot at k = 1.05 goes off, or laps slower than at k = 1.00, on at least 24 of the 28 laps.

`GRIP_MARGIN` is the smallest value in [1.00, 1.10] (0.01 steps) that meets 2. Then 1 and 3 are checked.

**Fallback.** If no margin meets 1–3 together, the friction circle is dropped: steering uses G(s) built from ay only, and longitudinal is unchanged. That variant was measured in prototype: it fixes Spielberg and moves the median break point from k = 1.10 to 1.08. The plan stops and reports before shipping the fallback.

## 5. Best laps

Drive stores best laps under `l3d_drive_best_${year}_${driver}_${session}_${track}_${mode}`. Times under the new model aren't comparable with the old ones, so the key gains a `_v2` suffix. Old entries stay in localStorage but are no longer read.

## 6. Tests (`tests/js/learn3d_drive.test.mjs`)

- **Stadium fixture:**
  - `buildModel` gives `grip` ≈ the arc's lateral g on the arcs, and ≥ `GRIP_FLOOR` on the straights;
  - the existing Assist / too-fast tests still hold;
  - new: at full lateral use, throttle gives no acceleration;
  - new: braking at full lateral use gives no extra deceleration.
- **Suzuka fixture:**
  - `optimalLap` is within −2 % / +1 % of the pro;
  - the existing bot test still passes.
- **Speed-bin regression:** a synthetic lap where one tight corner sits in an otherwise gentle speed band. The pro's own speed through it must not run off.
- **Existing tests:** `cornerStats`, `gapAt`, `pose` and the gear tests are unchanged.

## 7. Risks

- **Noisy positions.** FastF1 positions jitter, and envelope maxima pick up spikes, which makes the limit a little loose locally. The 15 m chord smoothing already damps this. The acceptance numbers are the check.
- **Harder to drive.** Manual players will find it harder. That is the intent, but Assist must still finish the pro's pace (acceptance 2).
