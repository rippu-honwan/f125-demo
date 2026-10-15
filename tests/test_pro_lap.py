"""Tests for the 3D Learn mode backend: GET /pro_lap and its helpers (app/main.py)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app, _corner_actions

client = TestClient(app)

T2 = {"id": 2, "short": "T2", "name": "Turn 2", "direction": "left",
      "entry_m": 620, "apex_m": 700, "exit_m": 780}


def _corners(*extra):
    t1 = {"id": 1, "short": "T1", "name": "Turn 1", "direction": "right",
          "entry_m": 400, "apex_m": 500, "exit_m": 600}
    return [t1, *extra]


def _lap(brake_from=None, lift_from=None):
    """Synthetic 1 km lap sampled every 2 m: 250 km/h, slowing to 120 around T1."""
    d = np.arange(0, 1000, 2.0)
    speed = np.full(d.size, 250.0)
    speed[(d >= 450) & (d <= 550)] = 120.0
    throttle = np.ones(d.size)
    brake = np.zeros(d.size)
    if brake_from is not None:
        zone = (d >= brake_from) & (d < 480)
        brake[zone] = 1.0
        throttle[zone] = 0.0
    if lift_from is not None:
        throttle[(d >= lift_from) & (d < 480)] = 0.5
    t = np.concatenate([[0.0], np.cumsum(np.diff(d) / (speed[1:] / 3.6))])
    gear = np.where(speed < 150, 3, 7)
    return d, t, speed, throttle, brake, gear


class TestCornerActions:
    def test_brake_corner(self):
        c = _corner_actions(_corners(), *_lap(brake_from=380))[0]
        assert c["action"] == "brake"
        assert c["brake_start_m"] == pytest.approx(380, abs=2)
        assert c["brake_duration_s"] > 0
        assert c["min_speed_kmh"] == 120
        assert c["apex_gear"] == 3

    def test_lift_corner(self):
        c = _corner_actions(_corners(), *_lap(lift_from=380))[0]
        assert c["action"] == "lift"
        assert c["brake_start_m"] is None
        assert c["brake_duration_s"] is None

    def test_flat_corner(self):
        c = _corner_actions(_corners(), *_lap())[0]
        assert c["action"] == "flat"

    def test_brake_far_before_entry_is_ignored(self):
        d, t, s, th, b, g = _lap()
        b[(d >= 100) & (d < 140)] = 1.0          # > 250 m before T1 entry (400)
        assert _corner_actions(_corners(), d, t, s, th, b, g)[0]["action"] == "flat"

    def test_does_not_inherit_previous_corner_braking(self):
        out = _corner_actions(_corners(T2), *_lap(brake_from=380))
        assert out[0]["action"] == "brake"
        assert out[1]["action"] == "flat"

    def test_onset_too_close_to_apex_belongs_to_next_corner(self):
        d, t, s, th, b, g = _lap()
        b[(d >= 485) & (d < 495)] = 1.0          # 15 m before T1 apex (500)
        out = _corner_actions(_corners(T2), d, t, s, th, b, g)
        assert out[0]["action"] != "brake"
        assert out[1]["action"] == "brake"
        assert out[1]["brake_start_m"] == pytest.approx(486, abs=2)

    def test_carries_corner_identity(self):
        c = _corner_actions(_corners(), *_lap())[0]
        assert (c["id"], c["short"], c["direction"]) == (1, "T1", "right")
        assert (c["entry_m"], c["apex_m"], c["exit_m"]) == (400, 500, 600)

    def test_lift_ignores_previous_corner_exit(self):
        d, t, s, th, b, g = _lap()
        th[(d >= 600) & (d < 640)] = 0.6      # partial throttle after T1 exit (600), before T2 entry (620)
        out = _corner_actions(_corners(T2), d, t, s, th, b, g)
        assert out[1]["action"] == "flat"

    def test_lift_after_previous_exit_is_detected(self):
        d, t, s, th, b, g = _lap()
        th[(d >= 660) & (d < 690)] = 0.5      # drop starts at 660, within T2's lift window [620, 670)
        out = _corner_actions(_corners(T2), d, t, s, th, b, g)
        assert out[1]["action"] == "lift"

    def test_lift_continuing_from_previous_corner_is_ignored(self):
        d, t, s, th, b, g = _lap()
        th[(d >= 590) & (d < 640)] = 0.6      # drop starts at 590 (before T1 exit), continues past it
        out = _corner_actions(_corners(T2), d, t, s, th, b, g)
        assert out[1]["action"] == "flat"
