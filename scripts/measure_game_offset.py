"""Measure how far the game's lap distance runs ahead of FastF1's, and move tracks/*.json corners onto the game's.

The corners come from FastF1 (scripts/rebuild_track_corners.py), but Lap Analysis, Comparison and Coaching
apply them on the game's lap distance. FastF1's lap starts 13-44 m after the game's line, depending on the
track, so the corners sat that far early. This fits the game's driven path onto the cached pro lap (the one
the rebuild uses) and, at the same place, takes game distance minus pro distance, both on the file's
length_m scale. The median over the 3 fastest full laps is the file's game_offset_m. The 3D views use the
FastF1 lap and take it back off (app.main._real_corners).

    python3 scripts/measure_game_offset.py data/Austria.csv data/Texas.csv:austin           # print
    python3 scripts/measure_game_offset.py data/Austria.csv data/Texas.csv:austin --write   # set game_offset_m, shift the corners

The track comes from the CSV's trackId; name it after a colon when that isn't recognised.
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.main import _detect_track_from_csv, _fit_line, _is_position, detect_separator  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from rebuild_track_corners import replace_corners  # noqa: E402

LAPS = 3                  # fastest full laps measured
END_M = 150               # skip this much either side of the line, where one lap's distance wraps
MAX_OFF_LINE_M = 6        # skip places where your path is further than this from the pro's


def measure(csv: str, track: str = None) -> tuple:
    """(track, offset_m, per-lap medians) for one game CSV."""
    track = track or _detect_track_from_csv(csv)[0]
    if not track:
        sys.exit(f"{csv}: unknown trackId; name the track, e.g. {csv}:austin")
    data = json.loads((ROOT / "tracks" / f"{track}.json").read_text(encoding="utf-8"))
    pro, _ = pd.read_pickle(sorted(glob.glob(str(ROOT / "data" / "reference_laps" / f"2025_*_q_{track}.pkl")))[0])[:2]
    pro_xy = pro[["world_position_X", "world_position_Y"]].to_numpy(float)
    pro_d = pro["lap_distance"].to_numpy(float)
    tree = cKDTree(pro_xy)
    game = pd.read_csv(csv, sep=detect_separator(csv), usecols=[
        "lap_number", "lap_distance", "lap_time", "trackLength", "world_position_X", "world_position_Y"]).dropna()
    length = float(game["trackLength"].iloc[0])
    full = [(g["lap_time"].max(), g.sort_values("lap_distance")) for _, g in game.groupby("lap_number")
            if len(g) >= 0.98 * length and g["lap_distance"].min() < 5
            and g[["world_position_X", "world_position_Y"]].std().min() > 1]   # some exports log no positions
    per_lap = []
    for _, g in sorted(full, key=lambda x: x[0])[:LAPS]:
        d = g["lap_distance"].to_numpy(float)
        path, _ = _fit_line(g[["world_position_X", "world_position_Y"]].to_numpy(float), pro_xy)
        off_line, i = tree.query(path)
        ok = ((d > END_M) & (d < length - END_M) & (pro_d[i] > END_M) & (pro_d[i] < pro_d[-1] - END_M)
              & (off_line < MAX_OFF_LINE_M))
        file_m = data["length_m"]
        per_lap.append(float(np.median(d[ok] * file_m / length - pro_d[i][ok] * file_m / pro_d[-1])))
    return track, round(float(np.median(per_lap))), per_lap


def write(track: str, offset: int) -> None:
    """Set game_offset_m and move every corner position field by the change."""
    path = ROOT / "tracks" / f"{track}.json"
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    shift = offset - data.get("game_offset_m", 0)
    corners = [{f: (v + shift if isinstance(v, int) else round(v + shift, 2)) if _is_position(f, v) else v
                for f, v in c.items()} for c in data["corners"]]
    text = replace_corners(text, corners)
    if '"game_offset_m"' in text:
        text = re.sub(r'"game_offset_m": -?\d+', f'"game_offset_m": {offset}', text)
    else:
        text = re.sub(r'("length_m": [\d.]+,\n)', rf'\1  "game_offset_m": {offset},\n', text, count=1)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    for arg in [a for a in sys.argv[1:] if not a.startswith("--")]:
        csv, _, name = arg.partition(":")
        track, offset, per_lap = measure(csv, name or None)
        print(f"{track}: game_offset_m {offset}  (laps: {', '.join(f'{x:+.1f}' for x in per_lap)})")
        if "--write" in sys.argv:
            write(track, offset)
