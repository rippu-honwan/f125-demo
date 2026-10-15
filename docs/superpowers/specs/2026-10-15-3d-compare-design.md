# 3D Compare: your lap vs the pro, on the 3D track

Sub-project #2 of the 3D plan (see `2026-10-15-3d-pro-lap-viewer-design.md` §1). Builds on #1 (Learn 3D, merged).

## 1. Goal

Upload a game lap (Sim Racing Telemetry CSV), pick the pro (driver / year / session), and watch both cars race on the 3D track from the same start, with a live time gap. See where you lose time, at what speed, and with what inputs.

### Non-goals

- Your own racing line. Both cars drive the pro's line, because the game→real shape alignment can be off by up to 150 m, while real line differences are a few metres.
- More than two cars.
- Coaching text. The Coaching mode already does this.
- Changes to `src/`. Web changes stay in `app/`; `src/` is only imported.

## 2. Decisions

| Question | Decision |
| --- | --- |
| Entry point | A new mode card, **3D Compare**. It needs a CSV, like the other comparison modes. Learn 3D stays CSV-free. |
| Car placement | Both cars run on the pro's 3D line, placed by aligned lap distance. |
| Sync | One clock, the same start (ghost race). The gap is visible on screen. Live delta = your time minus the pro's time at your car's position. |
| Backend shape | A new mode `compare_3d` on `POST /analyze`, reusing upload, validation and lap picking. There is no new endpoint. |

## 3. Data facts (verified)

- `run_pipeline` loads the real lap with `load_real_telemetry`. This is the same loader and the same `lap_distance` axis that `/pro_lap` uses.
- `aligned` has one row per game metre (`lap_distance` = game distance, 1 m step) with `game_speed_kmh`, `game_throttle`, `game_brake` and `game_gear`.
- `aligned.attrs["game_anch_d"]` / `["real_anch_d"]` are the alignment anchors. The real distance of each game metre is `np.interp(lap_distance, game_anch_d, real_anch_d)`, so `src/` needs no change.
- `report.game_time` / `report.real_time` are the lap times. `report.corner_insights[*].time_delta` is the per-corner gain or loss.
- Test data: `data/suzuka_sample.csv` plus the cached `data/reference_laps/2025_ver_q_suzuka.pkl`.

## 4. Backend: `POST /analyze` with `mode=compare_3d`

- Add `"compare_3d"` to `COMPARISON_MODES`.
- In `_run_comparison`, the `compare_3d` branch:
  1. Loads the pro lap with `load_real_telemetry(driver, year, session, track)`. It comes from the cache, so it is cheap. `run_pipeline` doesn't return its real data, and `src/` stays untouched.
  2. Calls `pro = _build_pro_lap(df, meta, track_info)`, the same payload as `/pro_lap`.
  3. Calls `you = _build_you_lap(aligned, pro["points"]["d"], report.game_time)` (new; see below).
  4. Adds to each pro corner `your_min_speed_kmh` (your speed at that corner's apex) and `time_delta` (from the `corner_insights` with the same `id`, or `null`).
  5. Returns the usual comparison header (`game_time`, `real_time`, `overall_delta`, …) plus `pro` and `you`.
- `_build_you_lap(aligned, d_grid, lap_time)` returns column arrays on the pro's 2 m grid `d_grid`:
  - `real_d = interp(game_d, game_anch_d, real_anch_d)`, where `game_d` is `aligned["lap_distance"]`.
  - Your time over game distance comes from integrating speed (trapezoid, `v ≥ 1 m/s`), then scaled so it ends at `lap_time` (same pin as the pro lap).
  - `t`, `speed`, `throttle`, `brake` and `gear` are each `np.interp(d_grid, real_d, …)`.
  - `real_d` is made strictly increasing first (`np.maximum.accumulate` plus a tiny epsilon), so the interpolation is valid.
  - The output keys are `t`, `speed`, `throttle`, `brake` and `gear`, all the same length as `d_grid`. `t` is strictly increasing and ends at `lap_time`.
- Errors use the existing paths: a bad CSV, a FastF1 failure or a failed alignment returns 422 with the existing messages. If the pro lap can't be built (no position data), the response is 422 "No 3D data for this lap".

## 5. Frontend

- A new mode card, **3D Compare**, with i18n in en and ja like the other cards. It uses the CSV upload and pro pickers like the comparison modes. After Analyze, it shows the 3D block instead of the report.
- It reuses the `learn3d/` modules:
  - `scene.js` takes a list of cars instead of one. The pro car is orange and your car is blue.
  - `core.js` adds `posAtTime(t, tArr, dArr)`, which looks up distance from time by binary search. It is pure and tested with node.
  - `index.js` gets one shared clock. Each car's position is `posAtTime(clock, car.t, d)`. A car that has finished waits at the line, and the clock ends when both are done.
- The camera follows your car by default; a toggle switches it to the pro. T-cam, Chase and Top work as in #1.
- **HUD**: a large live delta (`+0.42` in red when slower, `−0.13` in green when faster), read as `you.t[i] − pro.t[i]` at your car's index. Next to it are your speed and the pro's speed.
- **Panels** keep the B/C toggle from #1:
  - **Telemetry (B)** overlays speed, throttle and brake (you and the pro) and adds a delta trace, all with the playhead.
  - **Corners (C)** shows one row per corner: the pro's action and apex speed, your apex speed, and your time gained or lost there.
- Keyboard shortcuts, saved settings and the mobile layout follow #1.

## 6. Testing

- pytest:
  - `_build_you_lap` on synthetic data checks the column lengths, that `t` is strictly increasing, that it ends at the lap time, and that the anchors are honoured: a shifted anchor moves your samples.
  - An endpoint test posts `data/suzuka_sample.csv` with `mode=compare_3d`, `driver=VER` and `track=suzuka`. It checks a 200 response, that `pro` and `you` have equal lengths, and that `you.t[-1]` ≈ `game_time`.
- node: `posAtTime` returns the start, the end and points in between.
- Playwright: run the Suzuka sample in 3D Compare and check that both cars move, the delta updates and the B/C panels render.

## 7. Known limitations

- Your car follows the pro's line, so line differences aren't shown.
- The live delta depends on the pipeline alignment. Where the anchors are sparse, the gap can drift by a few tenths.
- The pro lap is loaded twice: once inside `run_pipeline` and once for the 3D payload. The second load comes from the pkl cache.
