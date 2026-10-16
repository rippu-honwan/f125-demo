// Run: node --test tests/js/learn3d_rhythm.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildNotes, createJudge, dtMetres } from "../../app/static/js/learn3d/rhythm.js";

// 10 m per second, so lap distance d is reached at t = d / 10.
const P = { d: [0, 10, 20, 30], t: [0, 1, 2, 3] };
const CORNERS = [
  { short: "T2", action: "brake", brake_start_m: 20, brake_duration_s: 0.4 },
  { short: "K", action: "lift", brake_start_m: null, brake_duration_s: null, entry_m: 15 },
  { short: "T1", action: "brake", brake_start_m: 10, brake_duration_s: 0.5 },
];
const fresh = () => createJudge(buildNotes(P, CORNERS));   // notes: T1 1.0–1.5 s, T2 2.0–2.4 s

test("buildNotes keeps brake corners only, in lap order, with start and end times", () => {
  const notes = buildNotes(P, CORNERS);
  assert.deepEqual(notes.map((n) => n.corner.short), ["T1", "T2"]);
  assert.deepEqual(notes.map((n) => [n.start, n.end]), [[1, 1.5], [2, 2.4]]);
  assert.ok(notes.every((n) => n.press === null && n.release === null));
});

test("press: perfect within 0.05 s, good within 0.12 s, early or late", () => {
  let j = fresh();
  assert.equal(j.press(1.04).grade, "perfect");
  j = fresh();
  const early = j.press(0.94);
  assert.equal(early.grade, "good");
  assert.ok(early.dt < 0);
  j = fresh();
  assert.equal(j.press(1.11).grade, "good");
});

test("a press before the window is ignored and leaves the note open", () => {
  const j = fresh();
  assert.equal(j.press(0.8), null);
  assert.equal(j.notes[0].press, null);
  assert.equal(j.press(1.0).grade, "perfect");
});

test("tick misses a note nobody pressed, and the next press goes to the next note", () => {
  const j = fresh();
  const out = j.tick(1.2);
  assert.equal(out.length, 1);
  assert.equal(out[0].grade, "miss");
  assert.equal(j.notes[0].press.grade, "miss");
  assert.equal(j.press(1.3), null);              // too early for T2 (2.0)
  assert.equal(j.press(2.0).note.corner.short, "T2");
});

test("release: perfect within 0.10 s, good within 0.25 s, otherwise miss", () => {
  let j = fresh(); j.press(1.0);
  assert.equal(j.release(1.58).grade, "perfect");
  j = fresh(); j.press(1.0);
  assert.equal(j.release(1.7).grade, "good");
  j = fresh(); j.press(1.0);
  const early = j.release(1.2);
  assert.equal(early.grade, "miss");
  assert.ok(early.dt < 0);
});

test("no release judgement after a missed press", () => {
  const j = fresh();
  j.tick(1.2);
  assert.equal(j.release(1.5), null);
  assert.equal(j.notes[0].release, null);
});

test("holding past end + 0.25 s is a release miss", () => {
  const j = fresh();
  j.press(1.0);
  const out = j.tick(1.8);
  assert.deepEqual(out.map((o) => [o.kind, o.grade]), [["release", "miss"]]);
  assert.equal(j.holding, null);
});

test("score: points, max, accuracy and combo", () => {
  const j = fresh();
  j.press(1.0); j.release(1.5);                   // 2 + 2
  j.tick(2.2);                                    // T2 press miss
  assert.deepEqual(j.score(), { points: 4, max: 8, accuracy: 0.5, combo: 0, maxCombo: 2 });
  assert.equal(createJudge([]).score().accuracy, 1);
});

test("dtMetres converts a timing error to metres at a speed", () => {
  assert.equal(dtMetres(0.08, 270), 6);
});
