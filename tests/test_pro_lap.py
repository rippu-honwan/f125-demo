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

    def test_brake_that_does_not_slow_the_car_is_ignored(self):
        d, t, s, th, b, g = _lap()
        b[(d >= 100) & (d < 140)] = 1.0          # still 250 km/h afterwards
        assert _corner_actions(_corners(), d, t, s, th, b, g)[0]["action"] == "flat"

    def test_does_not_inherit_previous_corner_braking(self):
        out = _corner_actions(_corners(T2), *_lap(brake_from=380))
        assert out[0]["action"] == "brake"
        assert out[1]["action"] == "flat"

    def test_brake_goes_to_corner_nearest_its_slowest_point(self):
        d, t, s, th, b, g = _lap()
        b[(d >= 485) & (d < 560)] = 1.0          # starts at T1's apex (500)...
        s[(d >= 640) & (d < 660)] = 80.0         # ...but the car is slowest near T2 (700)
        out = _corner_actions(_corners(T2), d, t, s, th, b, g)
        assert out[0]["action"] != "brake"
        assert out[1]["action"] == "brake"
        assert out[1]["brake_start_m"] == pytest.approx(486, abs=2)
        assert out[1]["apex_m"] == pytest.approx(640, abs=2)

    def test_brake_far_from_json_corner_is_still_matched(self):
        d, t, s, th, b, g = _lap()                # json apex 500, but the pro brakes for it at 100
        b[(d >= 100) & (d < 180)] = 1.0
        s[(d >= 180) & (d < 200)] = 90.0
        c = _corner_actions(_corners(), d, t, s, th, b, g)[0]
        assert c["action"] == "brake"
        assert c["apex_m"] == pytest.approx(180, abs=2)

    def test_rebraking_without_speeding_up_is_one_zone(self):
        d, t, s, th, b, g = _lap(brake_from=380)
        b[(d >= 420) & (d < 430)] = 0.0          # brief release, car keeps slowing
        out = _corner_actions(_corners(T2), d, t, s, th, b, g)
        assert [c["action"] for c in out] == ["brake", "flat"]
        assert out[0]["brake_start_m"] == pytest.approx(380, abs=2)

    def test_zones_pair_one_to_one_in_order(self):
        d, t, s, th, b, g = _lap(brake_from=380)  # zone 1 slowest at 450
        b[(d >= 560) & (d < 600)] = 1.0          # zone 2 slowest at 600, still nearer T1 (500) than T2 (700)
        s[(d >= 600) & (d < 620)] = 80.0
        out = _corner_actions(_corners(T2), d, t, s, th, b, g)
        assert [c["action"] for c in out] == ["brake", "brake"]
        assert [c["apex_m"] for c in out] == pytest.approx([450, 600], abs=2)

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

    def test_brake_corner_snaps_to_slowest_point(self):
        c = _corner_actions(_corners(), *_lap(brake_from=380))[0]
        assert c["apex_m"] == pytest.approx(450, abs=2)
        assert c["entry_m"] == pytest.approx(350, abs=2)
        assert c["exit_m"] == pytest.approx(550, abs=2)

    def test_lift_corner_snaps_to_slowest_point(self):
        c = _corner_actions(_corners(), *_lap(lift_from=380))[0]
        assert c["action"] == "lift"
        assert c["apex_m"] == pytest.approx(450, abs=2)

    def test_flat_corner_reports_speed_at_json_apex(self):
        d, t, s, th, b, g = _lap()
        s[(d >= 560) & (d < 580)] = 90.0         # slower point inside T1 but away from its json apex
        c = _corner_actions(_corners(), d, t, s, th, b, g)[0]
        assert c["action"] == "flat"
        assert c["min_speed_kmh"] == 120
        assert c["apex_m"] == 500

    def test_refinement_stops_at_next_corners_onset(self):
        d, t, s, th, b, g = _lap(brake_from=380)
        b[(d >= 560) & (d < 600)] = 1.0
        th[(d >= 560) & (d < 600)] = 0.0
        s[(d >= 580) & (d < 640)] = 80.0
        out = _corner_actions(_corners(T2), d, t, s, th, b, g)
        assert [c["action"] for c in out] == ["brake", "brake"]
        assert out[0]["apex_m"] == pytest.approx(450, abs=2)
        assert out[1]["apex_m"] == pytest.approx(580, abs=2)

    def test_high_speed_kink_does_not_steal_braking(self):
        d, t, s, th, b, g = _lap(brake_from=380)
        b[(d >= 590) & (d < 610)] = 1.0
        th[(d >= 590) & (d < 610)] = 0.0
        s[(d >= 600) & (d < 620)] = 70.0
        kink = {"id": 2, "short": "K", "type": "high_speed", "entry_m": 540, "apex_m": 560, "exit_m": 580}
        t3 = {"id": 3, "short": "T3", "entry_m": 600, "apex_m": 650, "exit_m": 700}
        out = _corner_actions(_corners(kink, t3), d, t, s, th, b, g)
        assert [c["action"] for c in out] == ["brake", "flat", "brake"]
        # both neighbours sit 50 m early, so the flat kink moves with them
        assert [c["apex_m"] for c in out] == pytest.approx([450, 510, 600], abs=2)

    def test_corners_processed_in_lap_order(self):
        out = _corner_actions([T2, *_corners()], *_lap(brake_from=380))
        assert [c["short"] for c in out] == ["T1", "T2"]
        assert out[0]["action"] == "brake"


class TestProLapEndpoint:
    @pytest.fixture(scope="class")
    def payload(self):
        res = client.get("/pro_lap", params={"year": 2025, "driver": "NOR", "track": "bahrain"})
        assert res.status_code == 200, res.text
        return res.json()

    def test_meta(self, payload):
        m = payload["meta"]
        assert (m["driver"], m["year"], m["session"], m["track"]) == ("NOR", 2025, "Q", "bahrain")
        assert m["track_name"] == "Bahrain"
        assert m["lap_time"] == pytest.approx(90.267, abs=1e-3)

    def test_columns_same_length(self, payload):
        p = payload["points"]
        assert set(p) == {"d", "x", "y", "z", "t", "speed", "throttle", "brake", "gear", "steer"}
        n = len(p["d"])
        assert n > 2000
        assert all(len(v) == n for v in p.values())

    def test_units_are_metres(self, payload):
        x, y = np.array(payload["points"]["x"]), np.array(payload["points"]["y"])
        path = np.hypot(np.diff(x), np.diff(y)).sum()
        length = payload["meta"]["length_m"]
        assert abs(path - length) / length < 0.05

    def test_time_monotonic_and_matches_lap_time(self, payload):
        t = np.array(payload["points"]["t"])
        assert t[0] == 0
        assert np.all(np.diff(t) > 0)
        assert abs(t[-1] - payload["meta"]["lap_time"]) < 1e-3

    def test_steer_normalised(self, payload):
        s = np.array(payload["points"]["steer"])
        assert s.min() >= -1 and s.max() <= 1
        assert np.abs(s).max() > 0.5                 # it actually turns

    def test_brake_points_before_apex(self, payload):
        brakes = [c for c in payload["corners"] if c["action"] == "brake"]
        assert len(brakes) >= 5
        for c in brakes:
            assert c["brake_start_m"] < c["apex_m"]

    def test_apexes_are_distinct_and_in_lap_order(self, payload):
        ap = [c["apex_m"] for c in payload["corners"]]
        assert ap == sorted(set(ap))
        t4 = next(c for c in payload["corners"] if c["short"] == "T4")
        assert t4["apex_m"] == pytest.approx(1528, abs=10)   # json says 1680; the pro is slowest here

    def test_unknown_track_404(self):
        assert client.get("/pro_lap", params={"track": "atlantis"}).status_code == 404

    def test_path_traversal_rejected(self):
        assert client.get("/pro_lap", params={"track": "../pyproject"}).status_code == 404

    def test_no_data_lists_available(self, monkeypatch):
        import app.main as m

        def offline(*args, **kwargs):
            raise RuntimeError("offline")

        monkeypatch.setattr(m, "load_real_telemetry", offline)
        res = client.get("/pro_lap", params={"year": 2019, "driver": "XXX", "track": "bahrain"})
        assert res.status_code == 404
        body = res.json()
        assert body["detail"] == "no data"
        assert {"year": 2025, "driver": "NOR", "session": "Q", "track": "bahrain"} in body["available"]

    @pytest.mark.parametrize("params", [
        {"driver": "../../x"}, {"driver": "NORRIS"}, {"session": "../x"}, {"year": 1800},
    ])
    def test_rejects_bad_input_before_loading(self, monkeypatch, params):
        import app.main as m
        called = []
        monkeypatch.setattr(m, "load_real_telemetry", lambda *a, **k: called.append(a))
        res = client.get("/pro_lap", params={"track": "bahrain", **params})
        assert res.status_code == 422
        assert called == []
