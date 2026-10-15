// Pure helpers for the 3D Learn mode — no DOM, no Three.js, so they run under
// `node --test tests/js/learn3d_core.test.mjs`.
//
// `points` is the column-array object from GET /pro_lap:
//   { d, t, x, y, z, speed, throttle, brake, gear, steer }  (all same length)

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

/** Car state at time t (s): interpolated channels, nearest gear, heading of the path in the XY plane. */
export function stateAt(points, t) {
  const i = bisect(points.t, t);
  const t0 = points.t[i], t1 = points.t[i + 1];
  const f = t1 > t0 ? Math.min(1, Math.max(0, (t - t0) / (t1 - t0))) : 0;
  const s = { t, i };
  for (const k of LERP_KEYS) s[k] = points[k][i] + (points[k][i + 1] - points[k][i]) * f;
  s.gear = f < 0.5 ? points.gear[i] : points.gear[i + 1];
  s.heading = Math.atan2(points.y[i + 1] - points.y[i], points.x[i + 1] - points.x[i]);
  return s;
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
