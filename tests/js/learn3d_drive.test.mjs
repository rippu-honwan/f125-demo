// Run: node --test tests/js/learn3d_drive.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  KMH, DT, HALF_WIDTH, WALL, GRIP_FLOOR, GRIP_MARGIN, buildModel, lookup, createCar, step, advance, pose, gapAt, cornerStats,
  gearAt,
} from "../../app/static/js/learn3d/drive_sim.js";

// Stadium loop, anticlockwise: 600 m straights joined by R = 60 m half circles, one point per ~1 m.
// The "pro" takes the arcs at 25 m/s (≈ 10.4 m/s² lateral), accelerates at 6 m/s² and brakes at 12 m/s².
const R = 60, STRAIGHT = 600, V_ARC = 25, ACC = 6, BRK = 12;
function stadium() {
  const arc = Math.PI * R, total = 2 * STRAIGHT + 2 * arc, n = Math.round(total);
  const P = { d: [], t: [], x: [], y: [], z: [], speed: [], throttle: [], brake: [], gear: [] };
  let t = 0;
  for (let i = 0; i <= n; i++) {
    const s = (i / n) * total;
    let x, y, v, thr = 0.5, brk = 0;
    const leg = s < STRAIGHT ? 0 : s < STRAIGHT + arc ? 1 : s < 2 * STRAIGHT + arc ? 2 : 3;
    const u = s - [0, STRAIGHT, STRAIGHT + arc, 2 * STRAIGHT + arc][leg];
    if (leg === 0 || leg === 2) {
      x = leg === 0 ? u : STRAIGHT - u;
      y = leg === 0 ? 0 : 2 * R;
      const up = Math.sqrt(V_ARC ** 2 + 2 * ACC * u), down = Math.sqrt(V_ARC ** 2 + 2 * BRK * (STRAIGHT - u));
      v = Math.min(up, down);
      if (up <= down) thr = 1; else { thr = 0; brk = 1; }
    } else {
      const a = u / R - Math.PI / 2;                       // leg 1 centre (600, 60), leg 3 centre (0, 60)
      x = leg === 1 ? STRAIGHT + R * Math.cos(a) : -R * Math.cos(a);
      y = leg === 1 ? R + R * Math.sin(a) : R - R * Math.sin(a);
      v = V_ARC;
    }
    if (i) t += (total / n) / ((v + P.speed[i - 1] / KMH) / 2);
    P.d.push(s); P.t.push(t); P.x.push(x); P.y.push(y); P.z.push(0);
    P.speed.push(v * KMH); P.throttle.push(thr); P.brake.push(brk); P.gear.push(Math.min(8, 1 + Math.floor(v / 10)));
  }
  return P;
}
const P = stadium();
const M = buildModel(P);
const ARC_MID = STRAIGHT + (Math.PI * R) / 2;

/** Drive until `stop(car)` or the lap ends, following `target(s)` (m/s) with throttle/brake. */
function drive(m, car, target, { steer = 0, stop = () => false, maxT = 300 } = {}) {
  let maxN = 0;
  while (!car.finished && car.t < maxT && !stop(car)) {
    const v = target(car.s);
    step(m, car, { throttle: car.v < v ? 1 : 0, brake: car.v > v + 5 / KMH ? 1 : 0, steer });
    maxN = Math.max(maxN, Math.abs(car.n));
  }
  return maxN;
}
const proSpeed = (pts) => (s) => pts.speed[Math.min(pts.d.length - 2, Math.round(s / (pts.d[1] - pts.d[0])))] / KMH;

test("buildModel reads curvature, lateral grip, acceleration and braking from the lap", () => {
  const i = P.d.findIndex((d) => d >= ARC_MID);
  assert.ok(Math.abs(M.kappa[i] - 1 / R) < 0.1 / R, `kappa ${M.kappa[i]}`);   // + = left turn
  assert.ok(Math.abs(M.kappa[P.d.findIndex((d) => d >= 300)]) < 1e-3);
  const g = M.grip[P.d.findIndex((d) => d >= ARC_MID)];
  assert.ok(Math.abs(g - (V_ARC ** 2 / R) * GRIP_MARGIN) < 1.5, `grip ${g}`);   // what the pro pulled here
  assert.equal(M.grip[P.d.findIndex((d) => d >= 300)], GRIP_FLOOR);            // straight: just enough to steer
  assert.ok(Math.abs(lookup(M.accel, 40) - ACC) < 0.6);
  assert.ok(Math.abs(lookup(M.brake, 50) - BRK) < 1.2);
  assert.equal(gearAt(M, V_ARC), 3);
});

test("Assist at the pro's speed stays on the line through the arcs", () => {
  const car = createCar(M, "assist");
  const maxN = drive(M, car, proSpeed(P));
  assert.ok(car.finished);
  assert.ok(maxN < 3, `max offset ${maxN}`);   // the arcs start with a step in curvature
  assert.equal(car.invalid, false);
});

test("Assist far too fast into an arc runs wide off the track", () => {
  const car = createCar(M, "assist");
  drive(M, car, (s) => proSpeed(P)(s) * 1.4, { stop: (c) => c.s > STRAIGHT + Math.PI * R });
  assert.equal(car.invalid, true);
});

/** A Manual car in the middle of the first straight, where the line is dead straight. */
function manualOnStraight() {
  const car = createCar(M, "manual");
  car.s = 300; car.v = proSpeed(P)(300);
  return car;
}

test("Manual without steering leaves the track at the first arc", () => {
  const car = manualOnStraight();
  drive(M, car, proSpeed(P), { stop: (c) => c.offTrack });
  assert.ok(car.offTrack);
  assert.ok(car.s > STRAIGHT - 20 && car.s < STRAIGHT + Math.PI * R, `left at s=${car.s}`);
});

test("Manual steering left through the arc keeps it on the road longer than none", () => {
  const run = (steer) => {
    const car = manualOnStraight();
    drive(M, car, () => V_ARC * 0.8, { steer: 0, stop: (c) => c.s >= STRAIGHT - 5 });
    drive(M, car, () => V_ARC * 0.8, { steer, stop: (c) => c.offTrack || c.s > ARC_MID });
    return car.s;
  };
  assert.ok(run(0.5) > run(0) + 20);
});

test("the wall holds the car within ±WALL and grass marks the lap invalid", () => {
  const car = createCar(M, "manual");
  car.psi = 0.5;
  for (let i = 0; i < 240; i++) step(M, car, { throttle: 1, brake: 0, steer: 0 });
  assert.ok(Math.abs(car.n) <= WALL);
  assert.ok(car.invalid);
  assert.ok(Math.abs(car.n) > HALF_WIDTH);
});

test("a spin puts the car back on the line at 60 km/h", () => {
  const car = createCar(M, "manual");
  car.n = 3; car.psi = 1.7;
  step(M, car, { throttle: 0, brake: 0, steer: 0 });
  assert.deepEqual([car.n, car.psi, Math.round(car.v * KMH)], [0, 0, 60]);
});

test("brake wins over throttle, grass adds drag, and throttle stops at the pro's top speed", () => {
  const v = 50;
  let car = createCar(M, "assist");
  car.s = 300; car.v = v;
  step(M, car, { throttle: 1, brake: 1 });
  assert.ok(Math.abs(car.v - (v - lookup(M.brake, v) * DT)) < 1e-9);
  car = createCar(M, "assist");
  car.s = 300; car.v = v; car.n = HALF_WIDTH + 1;
  step(M, car, { throttle: 0, brake: 0 });
  assert.ok(car.v < v - 7.9 * DT, `grass v ${car.v}`);
  car = createCar(M, "assist");
  car.s = 300; car.v = M.vTop;
  step(M, car, { throttle: 1, brake: 0 });
  assert.ok(car.v <= M.vTop);
});

test("a car stopped at the wall can drive away", () => {
  const car = createCar(M, "manual");
  car.s = 300; car.n = WALL; car.v = 0;
  for (let i = 0; i < 240; i++) step(M, car, { throttle: 1, brake: 0, steer: -1 });
  assert.ok(car.v > 5, `v ${car.v}`);
  assert.ok(Math.abs(car.n) < WALL, `n ${car.n}`);
});

test("advance runs fixed steps for the frame time, capped at 0.1 s", () => {
  const car = createCar(M, "assist");
  advance(M, car, { throttle: 1, brake: 0 }, 0.05 + DT / 2);
  assert.equal(car.trace.t.length - 1, 6);
  advance(M, car, { throttle: 1, brake: 0 }, 5);
  assert.ok(Math.abs(car.t - (0.05 + 0.1)) < 2 * DT);
});

test("the finish time is interpolated inside the last step", () => {
  const car = createCar(M, "assist");
  car.s = M.lapLength - 0.1; car.v = 60;
  step(M, car, { throttle: 0, brake: 0 });
  assert.ok(car.finished);
  assert.equal(car.s, M.lapLength);
  assert.ok(Math.abs(car.t - 0.1 / 60) < 1e-3, `t ${car.t}`);
});

test("pose moves the car to the left of the line for positive n", () => {
  const car = createCar(M, "assist");
  car.s = 300; car.n = 2; car.psi = 0.1;
  const p = pose(P, M, car);
  assert.ok(Math.abs(p.x - 300) < 0.5 && Math.abs(p.y - 2) < 0.1, JSON.stringify(p));
  assert.ok(Math.abs(p.heading - 0.1) < 0.01);
});

test("gapAt and cornerStats compare a trace with the pro", () => {
  const pro = { d: [0, 100, 200, 300, 400], t: [0, 1, 2, 3, 4] };
  const trace = { d: [0, 100, 200, 300, 400], t: [0, 1, 2.2, 3.5, 4.5], v: [50, 40, 20, 30, 50].map((k) => k / KMH), brake: [0, 1, 1, 0, 0] };
  assert.ok(Math.abs(gapAt(trace, pro, 300) - 0.5) < 1e-9);
  const corners = [
    { short: "T1", action: "brake", brake_start_m: 150, entry_m: 200, apex_m: 250, exit_m: 300, min_speed_kmh: 25 },
    { short: "T2", action: "flat", brake_start_m: null, entry_m: 360, apex_m: 380, exit_m: 400, min_speed_kmh: 45 },
  ];
  const [t1, t2] = cornerStats(trace, pro, corners);
  assert.deepEqual([t1.brakeAt, t1.proBrakeAt, Math.round(t1.minKmh)], [100, 150, 20]);
  assert.ok(Math.abs(t1.lost - 0.5) < 1e-9);                    // gap 0 at 0 m -> 0.5 at 300 m
  assert.deepEqual([t2.brakeAt, t2.proBrakeAt, Math.round(t2.minKmh)], [null, null, 46]);
});

test("cornerStats gives each brake press to one corner", () => {
  const pro = { d: [0, 1000], t: [0, 10] };
  const d = [], t = [], v = [], brake = [];
  for (let s = 0; s <= 1000; s += 10) {
    d.push(s);
    t.push(s / 100 + (s > 480 ? (s - 480) / 1000 : 0));          // you lose time only after 480 m
    v.push(50 / KMH);
    // a stray tap at 260, then A's brake at 400 and B's at 500
    brake.push((s >= 260 && s < 270) || (s >= 400 && s < 440) || (s >= 500 && s < 540) ? 1 : 0);
  }
  const corners = [
    { short: "A", action: "brake", brake_start_m: 400, entry_m: 420, apex_m: 450, exit_m: 480, min_speed_kmh: 50 },
    { short: "B", action: "brake", brake_start_m: 505, entry_m: 530, apex_m: 560, exit_m: 600, min_speed_kmh: 50 },
  ];
  const [a, b] = cornerStats({ d, t, v, brake }, pro, corners);
  assert.equal(a.brakeAt, 400);                                   // nearest the pro's, not the stray tap
  assert.equal(b.brakeAt, 500);                                   // not A's 400, though it's within 150 m
  assert.ok(Math.abs(a.lost) < 1e-9);
  assert.ok(Math.abs(b.lost - 0.12) < 1e-9);                      // you only lose time after 480 m
  // A flat corner whose stretch holds the next corner's brake point: the press is the brake corner's.
  const overlap = [
    { short: "F", action: "flat", brake_start_m: null, entry_m: 300, apex_m: 520, exit_m: 560, min_speed_kmh: 50 },
    { short: "G", action: "brake", brake_start_m: 505, entry_m: 530, apex_m: 560, exit_m: 600, min_speed_kmh: 50 },
  ];
  const [f, g] = cornerStats({ d, t, v, brake }, pro, overlap);
  assert.equal(g.brakeAt, 500);
  assert.equal(f.brakeAt, 260);                                   // F keeps only the press no brake corner wants
});

test("bot test: Assist following the pro's speed laps Suzuka on track, within 3 % of the pro", () => {
  const S = JSON.parse(readFileSync(new URL("./fixtures/suzuka_pro.json", import.meta.url))).points;
  const m = buildModel(S);
  const car = createCar(m, "assist");
  drive(m, car, proSpeed(S), { maxT: 200 });
  const pro = S.t[S.t.length - 1];
  assert.ok(car.finished);
  assert.equal(car.invalid, false);
  assert.ok(Math.abs(car.t / pro - 1) < 0.03, `lap ${car.t} vs ${pro}`);
});

/** Fastest lap the model allows on the line: corner speed from grip, then accel/brake passes (s). */
function optimalLapTime(m) {
  const n = m.d.length, ds = m.d[1] - m.d[0];
  const v = m.d.map((_, i) => Math.min(m.vTop, Math.sqrt(m.grip[i] / Math.max(Math.abs(m.kappa[i]), 1e-9))));
  v[0] = Math.min(v[0], m.v0);
  for (let i = 0; i < n - 1; i++) v[i + 1] = Math.min(v[i + 1], Math.sqrt(v[i] ** 2 + 2 * lookup(m.accel, v[i]) * ds));
  for (let i = n - 2; i >= 0; i--) {
    const brk = Math.max(lookup(m.brake, v[i + 1]), m.brakeHere[i + 1]);   // as in step()
    v[i] = Math.min(v[i], Math.sqrt(v[i + 1] ** 2 + 2 * brk * ds));
  }
  let t = 0;
  for (let i = 0; i < n - 1; i++) t += ds / ((v[i] + v[i + 1]) / 2);
  return t;
}

test("grip by place: the fastest lap the model allows on Suzuka stays within 7 % of the pro", () => {
  const S = JSON.parse(readFileSync(new URL("./fixtures/suzuka_pro.json", import.meta.url))).points;
  const pro = S.t[S.t.length - 1], best = optimalLapTime(buildModel(S));
  assert.ok(best / pro > 0.94, `optimal ${best.toFixed(2)} s vs pro ${pro} s`);   // 4.4 %; speed bins allowed ~8 % here
});

test("brake by place: the fastest lap the model allows on Spielberg stays within 8 % of the pro", () => {
  const S = JSON.parse(readFileSync(new URL("./fixtures/spielberg_ver_pro.json", import.meta.url))).points;
  const pro = S.t[S.t.length - 1], best = optimalLapTime(buildModel(S));
  assert.ok(best / pro > 0.92, `optimal ${best.toFixed(2)} s vs pro ${pro} s`);   // 6.7 %, the most brakeHere adds of 28 laps (6.1 % without)
});

test("grip by place: VER's Spielberg lap at its own speed stays on track (no speed-bin hole)", () => {
  const S = JSON.parse(readFileSync(new URL("./fixtures/spielberg_ver_pro.json", import.meta.url))).points;
  const m = buildModel(S);
  const car = createCar(m, "assist");
  drive(m, car, proSpeed(S), { maxT: 200 });
  assert.ok(car.finished);
  assert.equal(car.invalid, false);              // the 80-90 km/h bin allowed 15 m/s² where the pro pulls 25
});

test("brake by place: NOR's Shanghai lap at its own speed stays on track (T14 brakes harder than the table)", () => {
  const S = JSON.parse(readFileSync(new URL("./fixtures/shanghai_nor_pro.json", import.meta.url))).points;
  const m = buildModel(S);
  const car = createCar(m, "assist");
  drive(m, car, proSpeed(S), { maxT: 200 });
  assert.ok(car.finished);
  assert.equal(car.invalid, false);              // the lap-wide table alone went off at 4701 m
});

test("brake by place: braking is never weaker than the lap-wide table", () => {
  const S = JSON.parse(readFileSync(new URL("./fixtures/suzuka_pro.json", import.meta.url))).points;
  const m = buildModel(S);
  for (let i = 0; i < S.d.length - 1; i += 5) {
    const v = S.speed[i] / KMH, car = { ...createCar(m, "assist"), s: S.d[i], v };
    step(m, car, { throttle: 0, brake: 1 });
    assert.ok(v - car.v >= lookup(m.brake, v) * DT - 1e-9, `at ${S.d[i]} m: ${v - car.v}`);
  }
});
