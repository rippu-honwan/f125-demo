# 3D Compare: corner-by-corner line analysis

Follow-up to `2026-10-16-3d-compare-line-design.md`. That spec added the **Line** toggle, which draws your line beside the pro's. You can see the two lines, but nothing tells you where they differ. This spec puts a short wide / tight reading for each corner in the Corners panel.

Builds on `fix/compare-3d-road-height`, whose `roadHeights` projection this work generalises.

## 1. Evidence

Measured with `data/Zandvoort.csv` (laps 1–4), `data/suzuka_sample.csv` and `data/Austria.csv`, against VER's 2025 Q laps.

- **Index alignment can't be used.** Your x/y at a given index of the pro's distance grid is −30 to +26 m along the lap from the pro's x/y at that index (median −12 m). Comparing the two lines index by index would read the wrong stretch of track.
- **A geometric projection is repeatable.**
  - **Method.** Project each of your points onto the pro's line. Take the signed lateral distance at each corner's entry, apex and exit, as a median over ±5 m.
  - **Lap to lap.** On Zandvoort, laps 1–4 agree to within about 1–2 m.
  - **Real differences recur.** They show up every lap: T2 exit is 5–9 m wide, T8 4–8 m wide, and T14's apex 6–8 m tight.
- **The choice of pro doesn't matter.** Your offsets against VER and against NOR differ by well under 1 m, because FastF1's lines lie within 0.33 m of each other.
- **Numbers aren't reliable.** The line fit is good to about ±1–2 m. A fixed offset between the game's track model and the real circuit also can't be told apart from a real difference in your line. So the panel shows bands, not metres.

## 2. Scope

**In.** For each corner, a wide / tight band at entry, apex and exit in the 3D Compare Corners panel while **Line** is on. Picking a row with **Line** on switches to the Top camera.

**Out.**
- Metre values.
- Advice on which line to take.
- Line analysis in Drive, where your car follows the pro's line.
- Apex markers on the track, which need track edges the data doesn't have.
- Server or payload changes.

## 3. Design

### `core.js`

- **`projectOnLine(pro, x, y, span = 60)`** is extracted from `roadHeights`.
  - For each of your points, it finds the nearest point on the pro's line. It searches only `span` m either side of your index, because Suzuka crosses itself.
  - It returns three arrays:
    - `s`: lap distance on the pro's line;
    - `z`: road height;
    - `side`: signed lateral distance in metres, positive to the left of the pro's direction of travel.
  - `roadHeights` becomes `projectOnLine(...).z`, with behaviour unchanged.
- **`lineOffsets(pro, you, corners)`** returns one `{entry, apex, exit}` per corner, in metres, or `null` where the corner has no reading.
  - **Positive means wider than the pro**, towards the outside of the corner. Negative means tighter.
  - **Measure points.** Each value is the median `side` of your points whose `s` lies within ±5 m of the corner's `entry_m`, `apex_m` or `exit_m`. These are the corner positions on the pro's lap, as the payload already has them.
  - **Which side is the outside.** The corner's direction comes from the pro's line: the heading change across ±20 m of the apex (`lineHeadings`). The track json's `direction` is not used. If the turn is left, the outside is to the right, so the value is `-side`.
  - **Missing readings.** A measure point with no points of yours inside its ±5 m window gives `null`.
- **`lineBand(m)`** turns a value into a label:

  | Value | Label |
  |---|---|
  | `null` | `"—"` |
  | under 2 m either way | `"≈"` |
  | 2–4 m wide / tight | `"wide"` / `"tight"` |
  | 4 m or more wide / tight | `"wide+"` / `"tight+"` |

  The thresholds are constants: `LINE_SAME_M = 2` and `LINE_BIG_M = 4`.

### `panels.js` — `createCornerList`

- **Line text per row.** In compare mode with your own line, it takes an optional `{offsets, driver}` (the `lineOffsets` result and the pro's driver code) and builds a second middle-column text for each row: `Entry ≈ · Apex tight · Exit wide+`.
- **Highlighted rows.** A row with any `+` band gets the class `l3d-line-big`, which shows in bold.
- **Toggling.** The returned object gains `setLine(on)`, which swaps the middle column between the apex-speed text and the line text. The header reads `Line vs <driver>: entry · apex · exit` while the line text is showing.
- **Gap column.** It stays, so a line difference can be read against the time lost.

### `index.js`

- **On start.** With your own line, it computes `lineOffsets(P, Y, data.corners)` once and passes it, with `data.meta.driver`, to `createCornerList`.
- **Line button.** It also calls `S.list.setLine(on)`.
- **Corner pick.** Picking a row while **Line** is on calls `setCam("top")` before the existing seek-and-replay. Like any camera button, this is remembered.
- **Note.** `NOTE_OWN_LINE` becomes: "Your line is matched to the track shape. Wide / tight compare it with the pro's line, to about ±2 m; an offset between the game's track and the real one shows up as well. The pro line is FastF1's smoothed position data, not the exact line." (The last sentence is kept from the current note.)

### CSS

`.l3d-line-big` makes the middle column `font-weight: 600`.

## 4. Error handling

- Without your own line there is no Line button, so the Corners panel is unchanged.
- A corner with no reading shows `—`.
- `lineOffsets` never throws on short laps. It returns `null` readings.

## 5. Testing

**Node (`tests/js/learn3d_core.test.mjs`).**
- **Wide and tight.** On a synthetic left-hand arc, your line offset 3 m outwards reads about +3 (`wide`). At 5 m outwards it is `wide+`, and 3 m inwards is `tight`.
- **Right-hand turn.** The mirror-image right-hander gives the same signs.
- **No index dependence.** Shifting your points 20 m along the index (same geometry) gives the same readings.
- **Bands.** `lineBand` at its edges: 1.99, 2, 3.99, 4, −2 and `null`.
- **Unchanged behaviour.** The existing `roadHeights` / `youPoints` tests still pass.

**Browser (Playwright).** 3D Compare on Suzuka with `data/suzuka_sample.csv`:
- with Line on, rows show bands and the header changes; with Line off, rows go back to apex speeds;
- picking a row with Line on switches to Top and replays the corner;
- there are no console errors.

The same check is repeated on Zandvoort with `data/Zandvoort.csv`, where T2's exit should read `wide+`.

## 6. Known limits

- Bands compare your line with FastF1's smoothed pro line, not the exact line the pro drove.
- A constant offset between the game's track model and the real one reads as a line difference.
- "Wide" is not "wrong". A wide exit is often right, so the labels only say how your line differs from the pro's.
