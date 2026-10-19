# Drive: lateral grip by location

Follow-up to `2026-10-16-3d-drive-design.md`. The Drive car's lateral grip comes from the pro's lap. Today it is a per-speed-bin table, which is too loose on most tracks and has holes. This spec makes it a per-location envelope instead.

## 1. Problem (measured on the 28 cached 2025 Q laps)

Today's lateral limit `aLat(v)` is a table per 10 km/h bin: the 95th percentile of the pro's v²|κ| in that bin, made non-decreasing with speed.

- **A corner's allowance comes from the wrong place.** It is set by whichever corner in the same speed bin the pro took hardest, not by this corner.
- **Holes.** VER's Spielberg lap needs 25 m/s² at 88 km/h at 1381 m, but the 80–90 km/h bin allows 15. That bin is dominated by low-g corner exits, so a bot following the pro's own speed goes off there.
- **Too loose.** The model-optimal lap is a quasi-steady-state lap-time simulation on the line, with this model's limits. It is a median 10.6 % faster than the pro (range 6.3–20.2 %; Monaco about 20 %).

## 2. Decision

Lateral grip comes from the pro at this place on the track: `grip[i]` = the max of the pro's v²|κ| within ±15 m, times `GRIP_MARGIN = 1.02`.

**Measured with this model:**
- the model-optimal lap is a median 6.0 % faster than the pro (range 3.6–10.4 %), against 10.6 % today;
- a look-ahead bot following the pro's speed stays on track on all 28 laps;
- the simple bot (brakes only once 5 km/h over the pro's speed, like the existing Suzuka test) stays on track on 27 of 28 laps, Spielberg included.
  - The miss is NOR Shanghai T14 (4.7 km). The pro brakes there at about 34 m/s², harder than the lap-wide `brake(v)` table allows, so braking at the pro's point is too late.
  - Today's looser lateral limit hid this. Margins from 1.00 to 1.05 all keep it; 1.00 adds misses at Jeddah and Lusail.
  - This is the per-location braking follow-up below.

**What is left.** About 5 % of slack remains. It comes from braking and acceleration: `brake(v)` and `accel(v)` are lap-wide percentiles, so you can brake a little later or accelerate a little earlier than the pro almost anywhere. Over-speeding into a corner still pays, because there is no friction circle.

**Prototyped and not taken.** A per-location friction circle (total grip shared between steering and braking/throttle):
- it stops over-speeding from paying (27 of 28 laps);
- but its optimal lap is still about 6 % faster than the pro;
- and the bot at the pro's own speed goes off on 3 of 28 laps.

It needs more work. It is a possible follow-up, together with per-location braking, which would also fix Shanghai T14.

### Non-goals

- Friction circle and per-location braking/acceleration (see above).
- Per-track road widths: there is no data, and the prototype showed little effect (Monaco's break point moved from 1.10× to 1.08× at a 4.5 m half-width).
- UI changes, apart from the best-lap key (§4).

## 3. Model (`app/static/js/learn3d/drive_sim.js`)

### `buildModel`

- Compute `lat[i] = v[i]² · |kappa[i]|`.
- `grip[i]` is the max of `lat` over ±`GRIP_WIN_M` (15 m) around i, wrapping round the lap. It is capped at `A_LAT_CAP` (6.5 g), multiplied by `GRIP_MARGIN` (1.02), and floored at `GRIP_FLOOR` (8 m/s²).
  - The floor lets the car steer on straights and kinks, where the pro's lateral is near zero.
  - 8 m/s² is below every measured corner apex, so it doesn't loosen corners.
- The model returns `grip` (one value per point of `d`) in place of `aLat`. `aLat`, `binTable`'s use for it, and the "never drops with speed" pass are removed. `accel`, `brake`, `gear` and `vTop` are unchanged.

### `step`

`kMax = along(m, m.grip, car.s) · (off ? 0.5 : 1) / max(v, 5)²`. Nothing else changes.

### Constants

`GRIP_MARGIN = 1.02` stays the single knob. Its comment says what 1.0 would mean: the pro's own max lateral g at this place. The extra 2 % is slack for a driver who brakes a little late, and keeps Jeddah and Lusail on track for the simple bot.

## 4. Best laps

Lap times under the new model aren't comparable with the old ones. The Drive best-lap key gains a `_v2` suffix: `l3d_drive_best_${year}_${driver}_${session}_${track}_${mode}_v2`. Old entries stay in localStorage, unread.

## 5. Tests (`tests/js/learn3d_drive.test.mjs`)

- **Stadium:**
  - `grip` on the arcs ≈ V_ARC²/R (±1.5);
  - `grip` mid-straight equals `GRIP_FLOOR`;
  - the existing Assist and too-fast-runs-wide tests still pass.
- **Suzuka fixture:**
  - the existing bot test still passes (within 3 % of the pro, on track);
  - new: the model-optimal lap is at most 6 % faster than the pro. It is measured at 4.4 %; today's model is about 8 % on this lap. This guards against the model loosening again.
- **Spielberg regression.** New fixture `tests/js/fixtures/spielberg_ver_pro.json`: VER's 2025 Q lap points, from `/pro_lap`. The simple bot following the pro's speed stays on track. With today's model it goes off at 1381 m.
- **Test helper.** `optimalLapTime(m)`, the quasi-steady-state forward/backward pass on the line, lives in the test file. Nothing in the app needs it.

## 6. Risks

- **Envelope picks up noise.** The envelope max can pick up curvature spikes, which locally loosens the limit. The ±15 m chord smoothing damps this, and the optimal-lap test bounds it.
- **Harder than before** on tracks where the bins were generous (Monaco, Baku). That is the intent. Assist at the pro's pace finishes everywhere with a look-ahead bot (28/28), and with the simple bot everywhere except Shanghai T14 (§2).
