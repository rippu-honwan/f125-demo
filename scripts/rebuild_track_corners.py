"""Re-anchor the corners in tracks/*.json on FastF1's official corner list.

Most track files had hand-estimated corner distances (off by up to ~1.4 km).
For each track this takes the official corner markers (FastF1 circuit info),
projects them onto the cached pro lap in data/reference_laps/, and rewrites
the "corners" array:

  * apex_m  = official corner distance along the lap, scaled to the file's length_m
  * entry_m = apex - min(80 m, half the gap to the previous corner); exit_m likewise
  * direction and speed class (type) come from the pro lap; the hand values
    disagreed with the real geometry on about half the corners
  * same corner count as before: keep ids and names, and shift the other
    position fields (*_m, not the *_iqr_m spreads) by the apex change
  * different count (layout changes etc.): rebuild as "Turn N"

Everything else in the file (sectors, DRS zones, formatting) is left as is.

    python3 scripts/rebuild_track_corners.py            # dry run: print the changes
    python3 scripts/rebuild_track_corners.py --write    # rewrite tracks/*.json
"""
import glob
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)
import fastf1  # noqa: E402
import src.fastf1_loader as loader  # noqa: E402  (also enables the FastF1 cache)

HALF_SPAN_M = 80          # max entry/exit distance from the apex
CHORD_M = 30              # chord length for the turn direction
LOW_KMH, HIGH_KMH = 130, 220


def official_corners(track: str, df: pd.DataFrame) -> list:
    """[(label, distance_m, marker_offset_m)] in lap order."""
    gp = loader.TRACK_TO_GP.get(track) or loader.TRACK_TO_GP[track.split("_")[0]]
    s = fastf1.get_session(2025, gp, "Q")
    s.load(laps=True, telemetry=False, weather=False, messages=False)
    x, y = df["world_position_X"].to_numpy(float), df["world_position_Y"].to_numpy(float)
    d = df["lap_distance"].to_numpy(float)
    out = []
    for _, c in s.get_circuit_info().corners.iterrows():
        gap = np.hypot(x - c.X, y - c.Y)
        i = int(np.argmin(gap))
        out.append((f"{int(c.Number)}{c.Letter or ''}", float(d[i]), float(gap[i]) / 10))  # 1/10 m units
    return sorted(out, key=lambda r: r[1])


def lap_shape(df: pd.DataFrame, apex: float) -> tuple:
    """(direction, speed class) of the pro lap around lap distance `apex`."""
    d = df["lap_distance"].to_numpy(float)
    x, y = df["world_position_X"].to_numpy(float), df["world_position_Y"].to_numpy(float)
    px, py = (np.interp([apex - CHORD_M, apex, apex + CHORD_M], d, a) for a in (x, y))
    cross = (px[1] - px[0]) * (py[2] - py[1]) - (py[1] - py[0]) * (px[2] - px[1])
    near = np.abs(d - apex) <= 40
    v = float(df["speed_kmh"].to_numpy(float)[near].min())
    kind = "low_speed" if v < LOW_KMH else "medium_speed" if v < HIGH_KMH else "high_speed"
    return ("left" if cross > 0 else "right"), kind


def rebuild(track: str, data: dict, df: pd.DataFrame) -> tuple:
    """(new corners, report lines)."""
    off = official_corners(track, df)
    old = sorted(data["corners"], key=lambda c: float(c["apex_m"]))
    # FastF1 laps run ~0.5% short of the game's; store distances on the file's own length_m scale.
    scale = float(data["length_m"]) / float(df["lap_distance"].iloc[-1])
    apex = [round(r[1] * scale) for r in off]
    gaps = [int(g) for g in np.diff(apex)]
    before = [HALF_SPAN_M] + [min(HALF_SPAN_M, g // 2) for g in gaps]
    after = [min(HALF_SPAN_M, g // 2) for g in gaps] + [min(HALF_SPAN_M, int(data["length_m"] - apex[-1]))]
    keep = len(old) == len(off)
    report, new = [], []
    for k, (label, fastf1_d, marker) in enumerate(off):
        direction, kind = lap_shape(df, fastf1_d)
        if keep:
            c = dict(old[k])
            delta = apex[k] - float(c["apex_m"])
            for key, v in c.items():
                if key.endswith("_m") and not key.endswith("_iqr_m") and isinstance(v, (int, float)):
                    c[key] = round(v + delta, 1)
            fixed = [f for f, v in (("direction", direction), ("type", kind)) if c.get(f) != v]
            if fixed:
                report.append(f"  {c.get('short')}: " + ", ".join(f"{f} {c.get(f)} -> {v}" for f, v in
                                                            (("direction", direction), ("type", kind)) if f in fixed))
        else:
            c = {"id": k + 1, "name": f"Turn {label}", "short": f"T{label}", "complex": None}
        c.update(type=kind, direction=direction, entry_m=apex[k] - before[k], apex_m=apex[k], exit_m=apex[k] + after[k])
        if marker > 15:
            report.append(f"  T{label}: marker {marker:.0f} m from the lap")
        new.append(c)
    return new, keep, report


def replace_corners(text: str, corners: list) -> str:
    """Swap the "corners" array in the file text, one corner per line, rest untouched."""
    start = text.index('"corners"')
    open_i = text.index("[", start)
    depth, i = 0, open_i
    while True:
        depth += {"[": 1, "]": -1}.get(text[i], 0)
        if depth == 0:
            break
        i += 1
    body = ",\n".join("    " + json.dumps(c) for c in corners)
    return text[:open_i] + "[\n" + body + "\n  ]" + text[i + 1:]


def main(write: bool) -> None:
    for path in sorted(glob.glob(str(ROOT / "tracks" / "*.json"))):
        track = Path(path).stem
        laps = sorted(glob.glob(str(ROOT / "data" / "reference_laps" / f"2025_*_q_{track}.pkl")))
        if not laps:
            print(f"{track}: no cached pro lap, skipped")
            continue
        df = pd.read_pickle(laps[0])[0]
        text = Path(path).read_text()
        data = json.loads(text)
        new, keep, report = rebuild(track, data, df)
        moved = [abs(n["apex_m"] - o["apex_m"]) for n, o in
                 zip(new, sorted(data["corners"], key=lambda c: float(c["apex_m"])))] if keep else []
        print(f"{track}: {len(data['corners'])} -> {len(new)} corners"
              + (f", apex moved up to {max(moved):.0f} m" if keep else ", rebuilt as Turn N")
              + f"  [{Path(laps[0]).stem}]")
        for line in report:
            print(line)
        if write:
            out = replace_corners(text, new)
            assert json.loads(out)["corners"] == json.loads(json.dumps(new))
            Path(path).write_text(out)


if __name__ == "__main__":
    main("--write" in sys.argv)
