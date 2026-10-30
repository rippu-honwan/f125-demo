# 3D Compare: your own racing line

Follow-up to `2026-10-15-3d-compare-design.md`. That spec ruled out showing your line, because the pipeline's game→real alignment can be off by up to 150 m. This spec reverses that decision, based on the study in §1.

## 1. Feasibility (measured on Suzuka, VER 2025 Q, `data/suzuka_sample*.csv`)

- **Axes.** The game's ground plane is `world_position_X/Y`; Z is height. In the aligned frame these columns are `game_world_x/y`.
- **Global similarity fit, then ICP.** First a similarity fit (rotation, uniform scale, translation, mirror allowed) on matching fractions of the lap; then 30 ICP rounds against the nearest point on the pro's line. After that, your line sits a median of 2.0 m from the pro's (p90 3.8 m, max 8 m), which is within the track width. So the 150 m error came from the old alignment method, not from the track shapes.
- **Repeatable.** Four of your laps give the same lateral profile: they correlate at 0.87–0.96, and the RMS difference between any two is 0.7–1.3 m.
- **The pro line is not the driver's exact line.** VER's and NOR's FastF1 positions lie within 0.33 m of each other (median), even though their speeds differ. So FastF1 position data looks snapped or smoothed onto a reference path. This is inferred from the data; there is no official source.
- **Unresolved.** A fixed offset between the game's track model and the real track can't be separated from a genuine difference in your line.

**Conclusion.** Drawing your line on the track is reliable to about ±1 m. A number such as "2.4 m wider than VER at T1" would not be, so the view shows no such numbers.

## 2. Goal

In 3D Compare, your car drives your own line. A **Line** toggle draws your line next to the pro's reference line, so you can see where you run wide, cut in or take a different apex.

### Non-goals

- Per-corner or per-metre line-offset numbers.
- Comparing your laps with each other.
- Changes to `src/`. `aligned` already has `game_world_x/y`.

## 3. Backend (`app/main.py`)

- **`_fit_line(game_xy, pro_xy) -> (xy, median_m)`.** Maps your path onto the pro's frame.
  - Starts with an SVD similarity fit on 2000 points taken at the same fraction of each lap.
  - Then runs 30 ICP rounds, each refitting against the nearest pro point (`scipy.spatial.cKDTree`, which is already a dependency).
  - Returns your transformed points and the median distance to the pro's line.
- **`_build_you_lap`.** Fits `game_world_x/y` (one row per game metre) onto the pro lap's `points.x/y`, the smoothed path the scene draws. It resamples the result onto `d_grid` through the existing `real_d` mapping, and adds `x` and `y` (2 decimals) to `you`.
  - **Quality gate.** If the median exceeds `LINE_FIT_MAX_M = 4.0`, or the columns are missing, `x` and `y` are left out.
- Height isn't sent. The client takes the road's height under your `x`/`y` (the pro's line projected on, within 60 m of the same `d`). The pro's `z` at the same `d` sank your car: the distance alignment puts your `x`/`y` up to ~30 m along from the pro's.

## 4. Frontend (`app/static/js/learn3d/`)

- **`core.youPoints(pro, you)`.** If `you.x` exists, it uses your `x`/`y` (at the road's height under them) and sets `heading = lineHeadings(...)` from your path. Otherwise nothing changes.
- **`index.js`, Compare mode only.**
  - **Cars.** When your line is present, both cars use `lane: 0` and the pro car is drawn with `setGhost(0, true)`, so the two can overlap without z-fighting. Without your line, the lanes stay ±1.6 m and the pro car stays solid.
  - **Line button.** A **Line** button (`#l3dLineBtn`, `aria-pressed`) sits next to the camera buttons. It is shown only when your line is present and starts off. Pressing it shows:
    - the existing pedal-coloured pro ribbon (`setLine`);
    - a new ribbon for your line in `YOU_COLOR`.
  - **Note.** Under the stage, one line of text: "Your line is matched to the track shape (about ±1 m). The pro line is FastF1's smoothed position data, not the exact line."
  - **Fallback.** Without your line, the note reads "Your line isn't available for this lap; your car follows the pro's line." and there is no button.
- **`scene.js`.** `createScene` takes an optional `youLine` points object and builds a second, single-colour ribbon (width 0.35 m, lift 0.13 m). It is hidden by default. `setLine(on)` toggles both ribbons. The ribbon helper takes the points as an argument instead of closing over the pro's.

## 5. Errors

- **A failed fit throws** (degenerate data): it is caught inside `_build_you_lap`, which leaves out `x`/`y`. The Compare payload never fails because of the line.
- **A short or partial lap** fits fine. Coverage is whatever `aligned` covers, as today.

## 6. Tests

- **Python (`tests/test_compare_3d.py`):**
  - `_fit_line` recovers a 3 m lateral offset from a synthetic loop that was rotated, scaled by 1.05, shifted and mirrored. The median is within 0.3 m of 3.
  - The Compare payload for `suzuka_sample.csv` has `you.x` and `you.y` of grid length.
  - A garbage path (random points) is left out by the gate.
- **Node (`tests/js/learn3d_core.test.mjs`):** `youPoints` uses your `x/y` when given, and its headings follow your path.
- **Browser (Playwright):** in 3D Compare on Suzuka, the Line toggle shows two ribbons, your car sits on your ribbon, and the pro car is see-through. There are no console errors.

## 7. Known limits

- You don't see the pro's exact line; see §1.
- A fixed track-model offset shows up as a line difference.
- Laps from other tracks are untested until someone uploads one. The gate falls back to the old view.
