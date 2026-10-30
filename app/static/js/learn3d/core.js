// Pure helpers for the 3D Learn mode — no DOM, no Three.js, so they run under
// `node --test tests/js/learn3d_core.test.mjs`.
//
// `points` is the column-array object from GET /pro_lap:
//   { d, t, x, y, z, speed, throttle, brake, gear, steer }  (all same length)
// In 3D Compare, `youPoints()` builds the same shape for your lap (your own line when the server sends one).

/** Index i of the interval arr[i]..arr[i+1] containing v (arr ascending), clamped to [0, n-2]. */
export function bisect(arr, v) {
  let lo = 0, hi = arr.length - 1;
  if (v <= arr[0]) return 0;
  if (v >= arr[hi]) return hi - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] <= v) lo = mid; else hi = mid;
  }
  return lo;
}

const LERP_KEYS = ["d", "x", "y", "z", "speed", "throttle", "brake", "steer"];

/**
 * Car state at time t (s): interpolated channels, nearest gear, heading of the path in the XY plane
 * (from `points.heading` when set, see lineHeadings; else from the two neighbouring points).
 */
export function stateAt(points, t) {
  const i = bisect(points.t, t);
  const t0 = points.t[i], t1 = points.t[i + 1];
  const f = t1 > t0 ? Math.min(1, Math.max(0, (t - t0) / (t1 - t0))) : 0;
  const s = { t, i };
  for (const k of LERP_KEYS) s[k] = points[k][i] + (points[k][i + 1] - points[k][i]) * f;
  s.gear = f < 0.5 ? points.gear[i] : points.gear[i + 1];
  const h = points.heading;
  s.heading = h ? h[i] + (h[i + 1] - h[i]) * f
    : Math.atan2(points.y[i + 1] - points.y[i], points.x[i + 1] - points.x[i]);
  return s;
}

/**
 * Heading (rad, XY plane, unwrapped) of the line at every point: the direction of the chord from
 * `span` m back to `span` m ahead, wrapping round the lap (the last point sits on or near the first).
 * Not neighbour to neighbour: FastF1 positions stall and even step back against d (Suzuka's first and
 * last metres, Jeddah at 4.2 km), which turns a neighbour heading round by up to 180°.
 */
export function lineHeadings(points, span = 15) {
  const { d, x, y } = points, np = d.length;
  const w = Math.max(1, Math.round(span / ((d[np - 1] - d[0]) / (np - 1))));
  const loop = np - 1;
  const at = (j) => ((j % loop) + loop) % loop;
  const h = new Array(np);
  for (let i = 0; i < np; i++) {
    h[i] = Math.atan2(y[at(i + w)] - y[at(i - w)], x[at(i + w)] - x[at(i - w)]);
    if (i) h[i] += 2 * Math.PI * Math.round((h[i - 1] - h[i]) / (2 * Math.PI));
  }
  return h;
}

/** Lap time (s) at lap distance d (m). */
export function timeAtDistance(points, d) {
  const i = bisect(points.d, d);
  const d0 = points.d[i], d1 = points.d[i + 1];
  const f = d1 > d0 ? Math.min(1, Math.max(0, (d - d0) / (d1 - d0))) : 0;
  return points.t[i] + (points.t[i + 1] - points.t[i]) * f;
}

/** Where the driver acts for a corner: the brake point, or the entry for lift/flat corners. */
export function cuePoint(c) {
  return c.action === "brake" ? c.brake_start_m : c.entry_m;
}

/** Nearest corner cue ahead of d within `lookahead` metres (wraps the lap), or null. */
export function nextCue(corners, d, lapLength, lookahead = 300) {
  let best = null;
  for (const c of corners) {
    let dist = cuePoint(c) - d;
    if (dist < 0) dist += lapLength;
    if (dist <= lookahead && (!best || dist < best.dist)) best = { corner: c, dist };
  }
  return best;
}

/** [from, to] lap distances for the corner-list replay: `lead` m before the cue to the exit. */
export function cornerSeekRange(c, lead = 150) {
  return [Math.max(0, cuePoint(c) - lead), c.exit_m];
}

/** 83.456 -> "1:23.4" (truncated to tenths). */
export function fmtTime(s) {
  const tenths = Math.floor(Math.max(0, s) * 10);
  const m = Math.floor(tenths / 600);
  return `${m}:${((tenths % 600) / 10).toFixed(1).padStart(4, "0")}`;
}

/**
 * Your lap as a `points` object: your timing and inputs on the pro's distance grid. With `you.x/y`
 * (your line, fitted to the track by the server) the car drives your line, at the road's height under
 * it; without them it drives the pro's line.
 */
export function youPoints(pro, you) {
  const out = { ...pro, t: you.t, speed: you.speed, throttle: you.throttle, brake: you.brake, gear: you.gear };
  if (you.x && you.y) {
    out.x = you.x;
    out.y = you.y;
    out.z = roadHeights(pro, you.x, you.y);
    out.heading = lineHeadings(out);
    out.ownLine = true;
  }
  return out;
}

/**
 * Height of the road (the pro's line, which the road follows) under each of your x/y points: the
 * pro's line projected on, searched only `span` m either side of your index. Not the pro's z at your
 * index: the server's distance alignment puts your x/y up to ~30 m along from the pro's at the same d,
 * which on a slope sank your car metres into the road. Not the nearest point on the whole lap
 * either: Suzuka crosses itself.
 */
function roadHeights(pro, x, y, span = 60) {
  const { d, x: px, y: py, z: pz } = pro, np = d.length, loop = np - 1;
  const w = Math.max(1, Math.round(span / ((d[np - 1] - d[0]) / loop)));
  return x.map((xi, i) => {
    let best = Infinity, z = pz[i];
    for (let k = i - w; k <= i + w; k++) {
      const a = ((k % loop) + loop) % loop, b = a + 1;             // segment a -> b, wrapping round the lap
      const ex = px[b] - px[a], ey = py[b] - py[a], len2 = ex * ex + ey * ey;
      const f = len2 > 0 ? Math.min(1, Math.max(0, ((xi - px[a]) * ex + (y[i] - py[a]) * ey) / len2)) : 0;
      const dx = px[a] + f * ex - xi, dy = py[a] + f * ey - y[i], dist = dx * dx + dy * dy;
      if (dist < best) { best = dist; z = pz[a] + f * (pz[b] - pz[a]); }
    }
    return z;
  });
}

/** Live gap (s) at lap distance d: your time minus the pro's. Positive = you're behind. */
export function gapAt(pro, you, d) {
  return timeAtDistance(you, d) - timeAtDistance(pro, d);
}

// Your line against the pro's, per corner (3D Compare). Bands, not metres: the line fit and the game's
// track model are only good to about ±2 m.
export const LINE_SAME_M = 2;   // under this, either way: same line as the pro
export const LINE_BIG_M = 4;    // from this: a big difference, marked "+"

/** Metres wider (+) or tighter (−) than the pro -> "≈", "wide", "tight", "wide+", "tight+"; "—" for no reading. */
export function lineBand(m) {
  if (m == null) return "—";
  const a = Math.abs(m);
  if (a < LINE_SAME_M) return "≈";
  return (m > 0 ? "wide" : "tight") + (a >= LINE_BIG_M ? "+" : "");
}

export function lineOffsets() { return []; }
