// Run: node --test tests/js/learn3d_core.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  bisect, stateAt, timeAtDistance, cuePoint, nextCue, cornerSeekRange, fmtTime, youPoints, gapAt, lineHeadings,
} from "../../app/static/js/learn3d/core.js";

const P = {
  d: [0, 10, 20, 30], t: [0, 1, 2, 3],
  x: [0, 10, 20, 30], y: [0, 0, 0, 0], z: [0, 0, 0, 0],
  speed: [100, 200, 300, 400], throttle: [1, 1, 0, 0], brake: [0, 0, 1, 1],
  steer: [0, 0.5, 1, 0], gear: [3, 4, 5, 6],
};

const BRAKE = { action: "brake", brake_start_m: 100, entry_m: 150, exit_m: 200 };
const FLAT = { action: "flat", brake_start_m: null, entry_m: 5000, exit_m: 5100 };

test("bisect finds the interval and clamps at both ends", () => {
  assert.equal(bisect([0, 1, 2, 3], 1.5), 1);
  assert.equal(bisect([0, 1, 2, 3], -5), 0);
  assert.equal(bisect([0, 1, 2, 3], 99), 2);
  assert.equal(bisect([0, 1, 2, 3], 2), 2);
});

test("stateAt interpolates channels and picks the nearer gear", () => {
  const s = stateAt(P, 1.5);
  assert.equal(s.d, 15);
  assert.equal(s.speed, 250);
  assert.equal(s.steer, 0.75);
  assert.equal(s.gear, 5);
  assert.equal(s.heading, 0);
  assert.equal(stateAt(P, 1.2).gear, 4);
});

test("stateAt clamps past the end of the lap", () => {
  assert.equal(stateAt(P, 99).d, 30);
  assert.equal(stateAt(P, -1).d, 0);
});

test("stateAt interpolates every channel and follows the path direction", () => {
  const Q = {
    d: [0, 10, 20], t: [0, 1, 2],
    x: [0, 4, 4], y: [0, 8, 16], z: [0, 2, 6],
    speed: [100, 200, 300], throttle: [1, 0.5, 0], brake: [0, 0.5, 1],
    steer: [0, 1, -1], gear: [3, 4, 5],
  };
  const s = stateAt(Q, 0.5);
  assert.equal(s.x, 2);
  assert.equal(s.y, 4);
  assert.equal(s.z, 1);
  assert.equal(s.throttle, 0.75);
  assert.equal(s.brake, 0.25);
  assert.equal(s.heading, Math.atan2(8, 4));
  assert.equal(stateAt(Q, 1.5).heading, Math.atan2(8, 0));   // second segment points straight +y
});

// Suzuka's first points stand still and its last ones step back ~0.2 m: neighbour headings flip there.
const SUZUKA = JSON.parse(readFileSync(new URL("./fixtures/suzuka_pro.json", import.meta.url))).points;

test("lineHeadings turn smoothly round the whole lap, through stalled and backward points", () => {
  const h = lineHeadings(SUZUKA);
  assert.equal(h.length, SUZUKA.d.length);
  for (let i = 1; i < h.length; i++) assert.ok(Math.abs(h[i] - h[i - 1]) < 0.2, `jump at d=${SUZUKA.d[i]}`);
  const straight = Math.atan2(SUZUKA.y[15] - SUZUKA.y[5], SUZUKA.x[15] - SUZUKA.x[5]);
  assert.ok(Math.abs(h[0] - straight) < 0.05, `start ${h[0]} vs ${straight}`);
});

test("stateAt takes the heading from points.heading when it is there", () => {
  const pts = { ...SUZUKA, steer: SUZUKA.speed.map(() => 0), heading: lineHeadings(SUZUKA) };
  const straight = Math.atan2(SUZUKA.y[15] - SUZUKA.y[5], SUZUKA.x[15] - SUZUKA.x[5]);
  assert.ok(Math.abs(stateAt(pts, 0).heading - straight) < 0.05);
  const end = stateAt(pts, SUZUKA.t[SUZUKA.t.length - 1] - 0.01).heading;
  assert.ok(Math.abs(Math.cos(end - straight) - 1) < 0.01, `end ${end}`);   // same direction, maybe ±2π
});

test("timeAtDistance inverts distance to time", () => {
  assert.equal(timeAtDistance(P, 25), 2.5);
  assert.equal(timeAtDistance(P, 0), 0);
});

test("cuePoint uses the brake point, else the entry", () => {
  assert.equal(cuePoint(BRAKE), 100);
  assert.equal(cuePoint(FLAT), 5000);
});

test("nextCue returns the nearest cue ahead within the lookahead, wrapping the lap", () => {
  const cs = [BRAKE, FLAT];
  assert.deepEqual(nextCue(cs, 50, 5400), { corner: BRAKE, dist: 50 });
  assert.deepEqual(nextCue(cs, 4900, 5400), { corner: FLAT, dist: 100 });
  assert.deepEqual(nextCue(cs, 5390, 5400), { corner: BRAKE, dist: 110 });
  assert.equal(nextCue(cs, 400, 5400), null);
});

test("cornerSeekRange starts 150 m before the cue and ends at the exit", () => {
  assert.deepEqual(cornerSeekRange(BRAKE), [0, 200]);
  assert.deepEqual(cornerSeekRange(FLAT), [4850, 5100]);
});

test("fmtTime renders m:ss.t, truncating (never rounding up to :60.0)", () => {
  assert.equal(fmtTime(83.456), "1:23.4");
  assert.equal(fmtTime(5.04), "0:05.0");
  assert.equal(fmtTime(59.99), "0:59.9");
});

const Y = youPoints(P, {
  t: [0, 2, 4, 6], speed: [50, 50, 50, 50], throttle: [0, 0, 0, 0], brake: [1, 1, 1, 1], gear: [2, 2, 2, 2],
});

test("youPoints keeps the pro's line and swaps in your timing and inputs", () => {
  assert.deepEqual(Y.x, P.x);
  assert.deepEqual(Y.d, P.d);
  assert.deepEqual(Y.t, [0, 2, 4, 6]);
  const s = stateAt(Y, 3);          // halfway between your 10 m (2 s) and 20 m (4 s)
  assert.equal(s.d, 15);
  assert.equal(s.speed, 50);
  assert.equal(s.gear, 2);
});

test("youPoints drives your own line when you send x/y", () => {
  // 51 points / 100 m: lineHeadings treats the path as a closed lap, so only check away from the ends.
  const n = 51, idx = Array.from({ length: n }, (_, i) => i);
  const pro = {
    d: idx.map((i) => 2 * i), t: idx, x: idx.map((i) => 2 * i), y: idx.map(() => 0), z: idx.map(() => 1),
    speed: idx.map(() => 100), throttle: idx.map(() => 1), brake: idx.map(() => 0), steer: idx.map(() => 0),
    gear: idx.map(() => 5),
  };
  const you = {
    t: idx, speed: pro.speed, throttle: pro.throttle, brake: pro.brake, gear: pro.gear,
    x: idx.map((i) => i * Math.SQRT2), y: idx.map((i) => i * Math.SQRT2),   // 2 m steps, 45° off the pro's line
  };
  const Y2 = youPoints(pro, you);
  assert.deepEqual(Y2.x, you.x);
  assert.deepEqual(Y2.y, you.y);
  assert.deepEqual(Y2.z, pro.z);                      // flat road: the height under your line is the pro's
  assert.deepEqual(Y2.d, pro.d);
  assert.equal(Y2.heading.length, n);
  assert.ok(Math.abs(stateAt(Y2, 25.5).heading - Math.PI / 4) < 1e-6);
  assert.equal(Y2.ownLine, true);
  const Y3 = youPoints(pro, { ...you, x: undefined, y: undefined });   // without x/y nothing changes
  assert.equal(Y3.x, pro.x);
  assert.equal(Y3.heading, undefined);
  assert.equal(Y3.ownLine, undefined);
});

/** A pro lap of n points, 2 m apart, at (x(i), y(i), z(i)); flat inputs. */
function proLap(n, x, y, z) {
  const idx = Array.from({ length: n }, (_, i) => i);
  return {
    d: idx.map((i) => 2 * i), t: idx, x: idx.map(x), y: idx.map(y), z: idx.map(z),
    speed: idx.map(() => 100), throttle: idx.map(() => 1), brake: idx.map(() => 0), steer: idx.map(() => 0),
    gear: idx.map(() => 5),
  };
}
const yourLap = (pro, x, y) => ({ t: pro.t, speed: pro.speed, throttle: pro.throttle, brake: pro.brake, gear: pro.gear,
  x: pro.t.map(x), y: pro.t.map(y) });

test("youPoints puts your car on the road under your line, not at the pro's height for your index", () => {
  // A 10 % slope. Your x/y run 20 m ahead of the pro's at the same d (the server's distance alignment
  // is that far off on real laps) and 3 m to the side.
  const pro = proLap(301, (i) => 2 * i, () => 0, (i) => 0.2 * i);
  const Y2 = youPoints(pro, yourLap(pro, (i) => 2 * i + 20, () => 3));
  for (const i of [50, 150, 250]) assert.ok(Math.abs(Y2.z[i] - (0.2 * i + 2)) < 1e-9, `z at ${i}: ${Y2.z[i]}`);
});

test("youPoints takes the road height from the same part of the lap where the track crosses itself", () => {
  // Out along y = 0 at ground level, back over the same x/y on a bridge 10 m up; your line runs 1 m aside.
  const pro = proLap(400, (i) => 2 * (i < 200 ? i : 399 - i), () => 0, (i) => (i < 200 ? 0 : 10));
  const Y2 = youPoints(pro, yourLap(pro, (i) => pro.x[i], () => 1));
  for (const i of [50, 100, 150]) assert.equal(Y2.z[i], 0);
  for (const i of [250, 300, 350]) assert.equal(Y2.z[i], 10);
});

test("gapAt is your time minus the pro's at the same distance", () => {
  assert.equal(gapAt(P, Y, 0), 0);
  assert.equal(gapAt(P, Y, 15), 1.5);   // you 3.0 s, pro 1.5 s
  assert.equal(gapAt(P, Y, 30), 3);
});
