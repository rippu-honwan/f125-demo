"""Mexico: tracks/ distances follow the game, whose Mexico is ~5 % bigger than the real circuit."""
import json
from pathlib import Path

import pytest

from app.main import _real_corners
from src.track import auto_detect_track

ROOT = Path(__file__).resolve().parent.parent
GAME_LENGTH = 4525          # trackLength in the game's Mexico CSVs; the real circuit is 4304 m
# T1..T17 apex on the real circuit (the file before it moved to the game's scale): what the 3D pro lap uses.
# Rerunning scripts/rebuild_track_corners.py --write can move these by a metre; update them then.
REAL_APEX = [1201, 1280, 1339, 2032, 2108, 2244, 2586, 2692, 2780, 2966, 3070, 3635, 3814, 3880, 3927, 3991, 4122]


def _track(name):
    return json.loads((ROOT / "tracks" / f"{name}.json").read_text(encoding="utf-8"))


def test_a_game_lap_length_finds_mexico():
    assert auto_detect_track(GAME_LENGTH).short == "mexico_city"


def test_sectors_and_corners_cover_the_game_lap():
    mex = _track("mexico_city")
    assert mex["length_m"] == GAME_LENGTH
    assert mex["sectors"][-1]["end_m"] == GAME_LENGTH        # S3 used to stop 221 m short of the line
    assert mex["corners"][-1]["apex_m"] == pytest.approx(REAL_APEX[-1] * GAME_LENGTH / 4304, abs=1)


def test_3d_gets_the_real_circuit_positions_back():
    apex = [c["apex_m"] for c in _real_corners(_track("mexico_city"))]
    assert apex == REAL_APEX                                  # exactly: 3D Mexico is as before


def test_only_mexico_differs_from_the_real_circuit():
    for p in (ROOT / "tracks").glob("*.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        if p.stem != "mexico_city":
            assert "real_length_m" not in data, p.stem
            assert _real_corners(data) is data["corners"], p.stem
