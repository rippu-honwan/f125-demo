// 3D Drive simulation: a car on the pro's line whose limits come from the pro's own lap.
// Pure (no DOM, no Three.js), so it runs under `node --test tests/js/learn3d_drive.test.mjs`.
//
// The car lives in track coordinates: s = distance along the pro's line (m), n = sideways
// offset (m, + = left), psi = heading relative to the line (rad, + = left). SI units inside;
// `points.speed` (km/h) is converted on the way in.
import { bisect, timeAtDistance } from "./core.js";

export const KMH = 3.6;
export const DT = 1 / 120;                     // fixed physics step (s)
export const HALF_WIDTH = 7;                   // matches scene.js TRACK_HALF_WIDTH
export const WALL = 20;                        // invisible wall: max |n| (m)
// ponytail: the one grip knob. 1.0 = the pro's 95th-percentile lateral G per speed; the bot test holds at 1.0.
export const GRIP_MARGIN = 1.0;
const COAST_K = 0.0008;                        // lift-off drag: COAST_K · v² (m/s²)
const BIN = 10 / KMH;                          // speed bin width for the limit tables (m/s)
const PCT = 0.95;                              // percentile taken per bin
const MIN_SAMPLES = 10;                        // fewer samples than this = empty bin
const A_LAT_CAP = 6.5 * 9.81;                  // no F1 car pulls more; caps noise in sparse bins
const SMOOTH_M = 15;                           // chord half-span for heading and curvature (m)
const GRASS_DRAG = 8, WALL_SCRUB = 15;         // extra decel (m/s²)
const SPIN_RESET_V = 60 / KMH;

const percentile = (a, q) => [...a].sort((x, y) => x - y)[Math.min(a.length - 1, Math.floor(q * a.length))];

/** Per-speed-bin q-percentile of [v, value] samples; empty bins copy the nearest filled bin. */
function binTable(samples, nb, q) {
  const bins = Array.from({ length: nb }, () => []);
  for (const [v, x] of samples) bins[Math.min(nb - 1, Math.round(v / BIN))].push(x);
  return fillNearest(bins.map((b) => (b.length >= MIN_SAMPLES ? percentile(b, q) : null)));
}

/** Replace each null with the nearest non-null entry (0 if there is none). */
function fillNearest(out) {
  const filled = out.flatMap((x, i) => (x == null ? [] : [i]));
  if (!filled.length) return out.map(() => 0);
  return out.map((x, i) => x ?? out[filled.reduce((a, j) => (Math.abs(j - i) < Math.abs(a - i) ? j : a))]);
}

/** Table value at speed v (m/s), linear between bin centres. */
export function lookup(table, v) {
  const f = Math.max(0, v / BIN), i = Math.min(table.length - 2, Math.floor(f));
  return table[i] + (table[i + 1] - table[i]) * Math.min(1, f - i);
}

/** Limits and line geometry from the pro's lap (`points` as from GET /pro_lap; the lap is a closed loop). */
export function buildModel(points) {
  const { d, x, y } = points, np = d.length;
  const v = points.speed.map((s) => s / KMH);
  const ds = (d[np - 1] - d[0]) / (np - 1);
  // Heading = direction of the chord p[i-w] -> p[i+w]; curvature = its change over the same span.
  // Chords, not neighbours: the positions stall and jitter by a metre or so against d.
  const w = Math.max(1, Math.round(SMOOTH_M / ds));
  const loop = np - 1;                                        // index np-1 sits on (or within a few m of) the start: wrap on np-1 points
  const at = (j) => ((j % loop) + loop) % loop;
  const heading = new Array(np);
  for (let i = 0; i < np; i++) {
    heading[i] = Math.atan2(y[at(i + w)] - y[at(i - w)], x[at(i + w)] - x[at(i - w)]);
    if (i) heading[i] += 2 * Math.PI * Math.round((heading[i - 1] - heading[i]) / (2 * Math.PI));
  }
  const turn = heading[loop] - heading[0];                    // ±2π for a lap
  const hAt = (j) => heading[at(j)] + turn * Math.floor(j / loop);
  const kappa = heading.map((_, i) => (hAt(i + w) - hAt(i - w)) / (2 * w * ds));

  const nb = Math.ceil(Math.max(...v) / BIN) + 2;
  const aLat = binTable(v.map((vi, i) => [vi, vi * vi * Math.abs(kappa[i])]), nb, PCT);
  for (let i = 1; i < nb; i++) aLat[i] = Math.max(aLat[i], aLat[i - 1]);   // downforce: grip never drops with speed
  const acc = [], brk = [];
  for (let i = 0; i < np - 1; i++) {
    const a = (v[i + 1] ** 2 - v[i] ** 2) / (2 * (d[i + 1] - d[i]));
    if (points.throttle[i] >= 0.95 && a > 0) acc.push([v[i], a]);
    if (points.brake[i] >= 0.5 && a < 0) brk.push([v[i], -a]);
  }
  const accel = binTable(acc, nb, PCT), vTop = Math.max(...v);
  const counts = Array.from({ length: nb }, () => new Map());
  v.forEach((vi, i) => {
    const c = counts[Math.min(nb - 1, Math.round(vi / BIN))];
    c.set(points.gear[i], (c.get(points.gear[i]) || 0) + 1);
  });
  const gear = fillNearest(counts.map((c) => (c.size ? [...c].sort((a, b) => b[1] - a[1])[0][0] : null)));
  return {
    d, heading, kappa, lapLength: d[np - 1], v0: v[0], vTop,
    aLat: aLat.map((a) => Math.min(A_LAT_CAP, a) * GRIP_MARGIN), accel, brake: binTable(brk, nb, PCT), gear,
  };
}

/** Linear interpolation of model array `arr` at lap distance s. */
function along(m, arr, s) {
  const i = bisect(m.d, s), f = Math.min(1, Math.max(0, (s - m.d[i]) / (m.d[i + 1] - m.d[i])));
  return arr[i] + (arr[i + 1] - arr[i]) * f;
}

/** A car on the start line at the pro's start speed. mode: "assist" | "manual". */
export function createCar(m, mode) {
  return {
    mode, s: 0, n: 0, psi: 0, v: m.v0, t: 0, axis: 0, acc: 0,
    offTrack: false, invalid: false, finished: false,
    trace: { d: [0], t: [0], v: [m.v0], brake: [0] },
  };
}

/** Shown gear at speed v: the pro's most common gear there. */
export const gearAt = (m, v) => m.gear[Math.min(m.gear.length - 1, Math.round(v / BIN))];

/**
 * One fixed physics step. input = {throttle: 0|1, brake: 0|1, steer: -1..1 (+ = left; Manual only)}.
 */
export function step(m, car, input, dt = DT) {
  if (car.finished) return;
  const off = Math.abs(car.n) > HALF_WIDTH;
  const v0 = car.v;
  let a = input.brake ? -lookup(m.brake, v0) : input.throttle ? lookup(m.accel, v0) : -COAST_K * v0 * v0;
  if (v0 >= m.vTop) a = Math.min(a, 0);                 // no faster than the pro's top speed
  // Grass and wall slow you to about 60 km/h but never stop you: below that, accel can be less than the drag.
  if (v0 > SPIN_RESET_V) a -= (off ? GRASS_DRAG : 0) + (Math.abs(car.n) >= WALL ? WALL_SCRUB : 0);
  car.v = Math.max(0, v0 + a * dt);

  const k = along(m, m.kappa, car.s);
  const kMax = lookup(m.aLat, car.v) * (off ? 0.5 : 1) / Math.max(car.v, 5) ** 2;
  let steer;
  if (car.mode === "assist") {                                // pure pursuit on the line, L m ahead
    const L = Math.max(10, 0.5 * car.v);
    steer = (k - 2 * (car.n + L * car.psi) / (L * L)) / kMax;
  } else {                                                    // keys ease in at 4/s, back to centre at 6/s
    const target = input.steer || 0, rate = (target ? 4 : 6) * dt;
    car.axis += Math.max(-rate, Math.min(rate, target - car.axis));
    steer = car.axis;
  }
  const kSteer = Math.max(-1, Math.min(1, steer)) * kMax;
  // ponytail: s advances with 1/(1 - n·kappa), so the inside of a hairpin is shorter, as on a 14 m-wide road.
  const sDot = car.v * Math.cos(car.psi) / Math.max(0.2, 1 - car.n * k);
  car.n += car.v * Math.sin(car.psi) * dt;
  car.psi += (car.v * kSteer - k * sDot) * dt;
  if (Math.abs(car.n) > WALL) { car.n = Math.sign(car.n) * WALL; car.psi = 0; }
  if (Math.abs(car.psi) > Math.PI / 2) { car.n = 0; car.psi = 0; car.v = SPIN_RESET_V; }   // spun: back on the line

  const s0 = car.s;
  car.s += sDot * dt;
  car.t += dt;
  car.offTrack = Math.abs(car.n) > HALF_WIDTH;
  if (car.offTrack) car.invalid = true;
  if (car.s >= m.lapLength) {                                 // finish time interpolated inside the step
    car.t -= dt * (car.s - m.lapLength) / (car.s - s0);
    car.s = m.lapLength;
    car.finished = true;
  }
  const tr = car.trace;
  tr.d.push(car.s); tr.t.push(car.t); tr.v.push(car.v); tr.brake.push(input.brake ? 1 : 0);
}

/** Run fixed steps for one animation frame of `frameDt` seconds (capped at 0.1 s). */
export function advance(m, car, input, frameDt) {
  car.acc += Math.min(0.1, frameDt);
  while (car.acc >= DT && !car.finished) {
    step(m, car, input);
    car.acc -= DT;
  }
}

/** World pose {x, y, z, heading} of the car, for scene.update(). */
export function pose(points, m, car) {
  const i = bisect(points.d, car.s);
  const f = Math.min(1, Math.max(0, (car.s - points.d[i]) / (points.d[i + 1] - points.d[i])));
  const lerp = (k) => points[k][i] + (points[k][i + 1] - points[k][i]) * f;
  const h = along(m, m.heading, car.s);
  return { x: lerp("x") - Math.sin(h) * car.n, y: lerp("y") + Math.cos(h) * car.n, z: lerp("z"), heading: h + car.psi };
}

/** Live gap (s) at lap distance s: your time minus the pro's. Positive = you're behind. */
export function gapAt(trace, points, s) {
  return timeAtDistance(trace, s) - timeAtDistance(points, s);
}

/**
 * One row per corner: where you first braked vs the pro, your min speed vs the pro's, and the time
 * lost from 150 m before the entry to the exit.
 */
export function cornerStats(trace, points, corners) {
  return corners.map((c) => {
    const from = (c.brake_start_m ?? c.entry_m) - 150;
    let brakeAt = null, minV = Infinity;
    for (let j = 0; j < trace.d.length; j++) {
      const s = trace.d[j];
      if (brakeAt == null && trace.brake[j] && s >= from && s <= c.apex_m) brakeAt = s;
      if (s >= c.entry_m && s <= c.exit_m) minV = Math.min(minV, trace.v[j]);
    }
    return {
      corner: c,
      brakeAt,
      proBrakeAt: c.brake_start_m ?? null,
      minKmh: Number.isFinite(minV) ? minV * KMH : null,
      proMinKmh: c.min_speed_kmh,
      lost: gapAt(trace, points, c.exit_m) - gapAt(trace, points, Math.max(0, c.entry_m - 150)),
    };
  });
}
