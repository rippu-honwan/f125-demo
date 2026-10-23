"""tracks/*.json corners sit on the game's lap distance; 3D takes game_offset_m back off for the FastF1 lap."""
import importlib.util
import json
from pathlib import Path

import pytest

from app.main import _real_corners

ROOT = Path(__file__).resolve().parent.parent
# Measured with scripts/measure_game_offset.py on the game CSVs in data/; tracks without one stay at 0.
MEASURED = {"austin": 25, "bahrain": 43, "melbourne": 35, "mexico_city": 34, "montreal": 43,
            "monza": 30, "spa": 26, "spielberg": 35, "suzuka": 26, "zandvoort": 14}
# Suzuka T1..FIN apex on the FastF1 lap (the file before the offset): what the 3D pro lap uses.
SUZUKA_FASTF1_APEX = [701, 823, 1073, 1197, 1359, 1524, 1845, 2259, 2421, 2741, 2896, 3157, 3796, 3950, 5005, 5392, 5466, 5625]


def _track(name):
    return json.loads((ROOT / "tracks" / f"{name}.json").read_text(encoding="utf-8"))


def test_offsets_are_set_only_where_a_game_lap_was_measured():
    for p in (ROOT / "tracks").glob("*.json"):
        assert json.loads(p.read_text(encoding="utf-8")).get("game_offset_m", 0) == MEASURED.get(p.stem, 0), p.stem


def test_corners_move_onto_the_game_lap_and_3d_gets_the_fastf1_ones_back():
    suzuka = _track("suzuka")
    assert [c["apex_m"] for c in suzuka["corners"]] == [a + 26 for a in SUZUKA_FASTF1_APEX]
    assert [c["apex_m"] for c in _real_corners(suzuka)] == SUZUKA_FASTF1_APEX     # exactly: 3D is as before


def test_the_script_measures_the_stored_offset():
    spec = importlib.util.spec_from_file_location("measure_game_offset", ROOT / "scripts" / "measure_game_offset.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    track, offset, _ = script.measure(str(ROOT / "data" / "Zandvoort.csv"))
    assert (track, offset) == ("zandvoort", pytest.approx(MEASURED["zandvoort"], abs=2))
