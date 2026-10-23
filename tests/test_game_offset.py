"""tracks/*.json corners sit on the game's lap distance; 3D takes game_offset_m back off for the FastF1 lap."""
import importlib.util
import json
from pathlib import Path

import pytest

from app.main import _real_corners

ROOT = Path(__file__).resolve().parent.parent
# Measured with scripts/measure_game_offset.py on the game CSVs in data/; tracks without one stay at 0.
MEASURED = {"austin": 25, "bahrain": 42, "melbourne": 35, "mexico_city": 34, "montreal": 43,
            "monza": 30, "spa": 25, "spielberg": 35, "suzuka": 26, "zandvoort": 18}
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


def test_real_corners_takes_off_the_offset_then_the_scale_on_position_fields_only():
    corner = {"apex_m": 1100, "brake_start_m": 1000.5, "apex_iqr_m": 5, "turn_in_m": None, "flag_m": True, "type": "low_speed"}
    back = _real_corners({"length_m": 5000, "game_offset_m": 100, "corners": [corner]})[0]
    assert back == {**corner, "apex_m": 1000, "brake_start_m": 900}             # spreads, None and bools untouched
    back = _real_corners({"length_m": 5000, "real_length_m": 4000, "game_offset_m": 100, "corners": [corner]})[0]
    assert back["apex_m"] == 800                                                # (1100 - 100) * 4000 / 5000


def _script():
    spec = importlib.util.spec_from_file_location("measure_game_offset", ROOT / "scripts" / "measure_game_offset.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_the_script_measures_the_stored_offset():
    track, offset, _ = _script().measure(str(ROOT / "data" / "Zandvoort.csv"))
    assert (track, offset) == ("zandvoort", pytest.approx(MEASURED["zandvoort"], abs=2))


def test_write_sets_the_offset_and_moves_corners_by_the_change_only(tmp_path):
    path = tmp_path / "spielberg.json"
    path.write_text((ROOT / "tracks" / "spielberg.json").read_text(encoding="utf-8"), encoding="utf-8")
    apex = [c["apex_m"] for c in _track("spielberg")["corners"]]
    for offset in (40, 40, 30):                                                 # the second 40 is a rerun: no move
        _script().write(path, offset)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["game_offset_m"] == 30
    assert [c["apex_m"] for c in data["corners"]] == [a - 5 for a in apex]      # stored 35 -> 30


def test_write_adds_the_key_to_a_track_without_one(tmp_path):
    path = tmp_path / "imola.json"
    path.write_text((ROOT / "tracks" / "imola.json").read_text(encoding="utf-8"), encoding="utf-8")
    _script().write(path, 20)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["game_offset_m"] == 20
    assert [c["apex_m"] for c in data["corners"]] == [c["apex_m"] + 20 for c in _track("imola")["corners"]]
