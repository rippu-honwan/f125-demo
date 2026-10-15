// Run: node --test tests/js/learn3d_core.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  bisect, stateAt, timeAtDistance, cuePoint, nextCue, cornerSeekRange, fmtTime, youPoints, gapAt,
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

test("gapAt is your time minus the pro's at the same distance", () => {
  assert.equal(gapAt(P, Y, 0), 0);
  assert.equal(gapAt(P, Y, 15), 1.5);   // you 3.0 s, pro 1.5 s
  assert.equal(gapAt(P, Y, 30), 3);
});
