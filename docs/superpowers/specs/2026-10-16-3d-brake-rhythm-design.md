# 3D Brake Rhythm: practice the pro's brake points like a music game

Sub-project #3 of the 3D plan (see `2026-10-15-3d-pro-lap-viewer-design.md` §1). It builds on #1 (3D Learn) and the corner fix that pairs braking zones with corners (`_corner_actions`).

## 1. Goal

While the pro's car drives the lap by itself in 3D Learn, the player holds a brake key from each brake point until the pro releases the brake. Both moments are judged **Perfect / Good / Miss** with early/late feedback, so the player learns *where* to brake and *for how long*.

### Non-goals

- Sound effects and music.
- Online leaderboards.
- Practice in any mode other than 3D Learn.
- Lift corners as notes.
- Backend changes: everything needed is already in `GET /pro_lap`.
- Changes to `src/`.

## 2. Decisions

| Question | Decision |
| --- | --- |
| What is timed | Long notes. Press at the brake point, release when the pro releases. Both moments are judged. |
| Entry point | A **Practice** button inside 3D Learn, after a lap has loaded. It needs no CSV and uses the same driver/track pickers. |
| Cues | A note lane plus two difficulties. **Easy**: the lane and the red brake stripes on the track. **Hard**: neither, so the player judges from the road. |

## 3. Notes and judging: `learn3d/rhythm.js` (pure, node-tested)

`buildNotes(points, corners)` returns one note per `action == "brake"` corner, sorted by start time:

- `start = timeAtDistance(points, brake_start_m)`
- `end = start + brake_duration_s`

Each note also carries `{ corner, start, end }`, plus judging state: `press` and `release` start as `null` and are filled with `{ grade, dt }`, where `dt` > 0 means late.

Judging works in lap time (the playback clock), not wall-clock time.

| | Perfect | Good | Otherwise |
| --- | --- | --- | --- |
| Press: \|t − start\| | ≤ 0.05 s | ≤ 0.12 s | Miss |
| Release: \|t − end\| | ≤ 0.10 s | ≤ 0.25 s | Miss |

`createJudge(notes)` returns an object with these methods:

- `press(t)`: judges the first note with no press yet whose `start + 0.12 > t`, but only if `t ≥ start − 0.12`. A press earlier than that window is ignored, so it can't break the combo. Returns the judged `{ note, grade, dt }` or `null`.
- `release(t)`: judges the note currently held, if any. If the press was a Miss, the release isn't judged (`release` stays `null`). Returns the judgement or `null`.
- `tick(t)`: marks a note's press as Miss once `t > start + 0.12` with no press. It also marks a held note's release as Miss once `t > end + 0.25` and releases it. Returns the new judgements so the UI can show them.
- `score()`: returns `{ points, max, accuracy, combo, maxCombo }`.
  - `points` gives 2 for Perfect, 1 for Good and 0 for Miss, summed over press and release.
  - `max` is 4 per note.
  - `accuracy` is `points / max`; with no notes it is 1.
  - `combo` counts consecutive non-Miss judgements and resets on a Miss.

`dtMetres(dt, speedKmh)` returns `dt × speed / 3.6`, for "late 0.08 s ≈ 6 m".

## 4. UI: 3D Learn

- **Practice button** in the 3D controls. It is enabled once a lap has loaded and hidden while practice runs.
- **Setup:** Easy/Hard plus 0.5×/1×, then **Start**. Start seeks to 0 and plays. In Hard, the scene hides the brake and lift stripes; `scene.setCues(false)` is a new scene API.
- **Note lane:** a horizontal strip over the bottom of the 3D stage. The hit line sits at 15 % from the left, and the lane shows the next 3 s of lap time.
  - Each note is a bar from `start` to `end`. It turns green or yellow when pressed and grey when missed.
  - The lane is hidden in Hard.
  - Hard also hides the HUD's "BRAKE in N m" cue and the telemetry/corner panels, whose brake zones would give the answer away.
- **Input:**
  - Hold **Space** or **↓** to brake.
  - **Esc** quits practice.
  - On touch screens, a large **BRAKE** button works the same way through `pointerdown`/`pointerup`/`pointercancel`.
  - During practice, Space is not play/pause, and the scrubber, ←/→ seeking and the corner list are disabled, so nobody can skip notes.
- **Feedback:** a pop-up near the hit line shows "PERFECT / GOOD / MISS" and "early/late 0.08 s ≈ 6 m". The top HUD shows the score and combo.
- **Results:** shown when the lap ends.
  - Points, accuracy %, max combo.
  - A per-corner table: corner, press grade + dt, release grade + dt.
  - The best score for `year_driver_session_track_difficulty` is kept in localStorage, wrapped in try/catch, and shown as "Best: 87 %".
  - Buttons: **Retry** and **Back to Learn**.
- **Leaving practice** (Esc, Back, a mode switch or unload) restores 3D Learn: cues on, lane hidden, Learn controls back.

## 5. Testing

- **node** (`tests/js/learn3d_rhythm.test.mjs`):
  - `buildNotes` keeps only brake corners and sorts by start time, with start and end correct.
  - Press boundaries at 0.05 s and 0.12 s, early and late.
  - A press before the window is ignored.
  - `tick` misses an unpressed note.
  - A release after a Miss press isn't judged.
  - Release boundaries at 0.10 s and 0.25 s.
  - A held note past `end + 0.25` gets a release Miss.
  - `score()`: points, max, accuracy and combo reset.
  - `dtMetres`.
- **Playwright:**
  - Load the Suzuka pro lap, open Practice (Easy, 1×), and use `page.evaluate` to send key events at the note times; expect PERFECT pop-ups and a results table.
  - Hard hides the lane and stripes.
  - Back to Learn restores normal playback.
  - At 390 px, the BRAKE button is visible and there is no horizontal scroll.

## 6. Known limitations

- The pro's brake signal is on/off, so a note's length is when the brake was released, not how hard they braked.
- Braking zones that the corner matcher merges, such as a brief release and re-application, become one long note.
- Judging uses the 2 m–sampled lap time. At 300 km/h one sample is about 0.024 s, well inside the 0.05 s Perfect window.
