// Pure brake-rhythm logic for 3D Learn practice — no DOM, so it runs under
// `node --test tests/js/learn3d_rhythm.test.mjs`.
//
// A note is one brake corner: press at the pro's brake point, release when the pro
// lets go. Times are lap time (the playback clock), so slower playback is easier.
import { timeAtDistance } from "./core.js";

export const PRESS_WINDOWS = { perfect: 0.05, good: 0.12 };     // s, |t − note start|
export const RELEASE_WINDOWS = { perfect: 0.10, good: 0.25 };   // s, |t − note end|
const POINTS = { perfect: 2, good: 1, miss: 0 };

/** One note per brake corner, sorted by start: {corner, start, end, press: null, release: null}. */
export function buildNotes(points, corners) {
  return corners
    .filter((c) => c.action === "brake" && c.brake_start_m != null && c.brake_duration_s != null)
    .map((c) => {
      const start = timeAtDistance(points, c.brake_start_m);
      return { corner: c, start, end: start + c.brake_duration_s, press: null, release: null };
    })
    .sort((a, b) => a.start - b.start);
}

const gradeOf = (err, w) => (Math.abs(err) <= w.perfect ? "perfect" : Math.abs(err) <= w.good ? "good" : "miss");

/**
 * Judges presses and releases against `notes` (mutating their press/release fields).
 * Every method returns the new judgement(s) — {note, kind: "press"|"release", grade, dt} —
 * where dt > 0 means late and dt is null for a timeout miss.
 */
export function createJudge(notes) {
  let held = null, combo = 0, maxCombo = 0;
  const record = (note, kind, grade, dt) => {
    note[kind] = { grade, dt };
    if (grade === "miss") combo = 0; else maxCombo = Math.max(maxCombo, ++combo);
    return { note, kind, grade, dt };
  };
  return {
    notes,
    get holding() { return held; },
    /** Brake on at lap time t. Presses before a note's window are ignored (null). */
    press(t) {
      if (held) return null;
      const note = notes.find((n) => !n.press && n.start + PRESS_WINDOWS.good >= t);
      if (!note || t < note.start - PRESS_WINDOWS.good) return null;
      held = note;
      return record(note, "press", gradeOf(t - note.start, PRESS_WINDOWS), t - note.start);
    },
    /** Brake off at lap time t: judges the held note, if any. */
    release(t) {
      if (!held) return null;
      const note = held;
      held = null;
      return record(note, "release", gradeOf(t - note.end, RELEASE_WINDOWS), t - note.end);
    },
    /** Time passing: unpressed notes past their window and holds far past the end become misses. */
    tick(t) {
      const out = [];
      for (const n of notes) {
        if (!n.press && t > n.start + PRESS_WINDOWS.good) out.push(record(n, "press", "miss", null));
      }
      if (held && t > held.end + RELEASE_WINDOWS.good) {
        const n = held;
        held = null;
        out.push(record(n, "release", "miss", null));
      }
      return out;
    },
    score() {
      let points = 0;
      for (const n of notes) for (const k of ["press", "release"]) if (n[k]) points += POINTS[n[k].grade];
      const max = notes.length * 4;
      return { points, max, accuracy: max ? points / max : 1, combo, maxCombo };
    },
  };
}

/** A timing error (s) as metres at `speedKmh`. */
export const dtMetres = (dt, speedKmh) => (dt * speedKmh) / 3.6;
