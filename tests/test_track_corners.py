"""Sanity checks on the corner lists in tracks/*.json (see scripts/rebuild_track_corners.py)."""
import json
from pathlib import Path

import pytest

TRACKS = sorted((Path(__file__).resolve().parent.parent / "tracks").glob("*.json"))


@pytest.mark.parametrize("path", TRACKS, ids=lambda p: p.stem)
def test_corners_are_ordered_and_inside_the_lap(path):
    data = json.loads(path.read_text())
    corners = data["corners"]
    assert corners
    for c in corners:
        assert 0 <= c["entry_m"] < c["apex_m"] < c["exit_m"] <= data["length_m"], c["short"]
        assert c["direction"] in ("left", "right")
        assert c["type"] in ("low_speed", "medium_speed", "high_speed")
    for a, b in zip(corners, corners[1:]):
        assert a["exit_m"] <= b["entry_m"], (a["short"], b["short"])
    assert len({c["short"] for c in corners}) == len(corners)
