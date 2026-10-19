"""Tests for the 3D Compare mode: POST /analyze mode=compare_3d and its helpers (app/main.py)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy.spatial import cKDTree

from app.main import LINE_FIT_MAX_M, _build_you_lap, _fit_line, app


def _aligned(game_anch=(0.0, 1000.0), real_anch=(0.0, 1000.0), speed=180.0):
    """A 1 km game lap at constant speed: full throttle for the first 500 game metres, then braking."""
    d = np.arange(1001, dtype=float)
    df = pd.DataFrame({
        "lap_distance": d,
        "game_speed_kmh": np.broadcast_to(np.asarray(speed, dtype=float), d.shape).copy(),
        "game_throttle": np.where(d < 500, 1.0, 0.0),
        "game_brake": np.where(d < 500, 0.0, 1.0),
        "game_gear": np.full(d.size, 6.0),
    })
    df.attrs["game_anch_d"] = np.array(game_anch)
    df.attrs["real_anch_d"] = np.array(real_anch)
    return df


GRID = np.arange(0.0, 1000.0 + 1e-9, 2.0)


def _loop(n, bump=0.0):
    """A closed, non-symmetric ~2.3 km loop of n points. `bump` m sideways over one stretch (a different line)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.c_[400 * np.cos(t) + 100 * np.cos(2 * t), 250 * np.sin(t)]
    tang = np.gradient(xy, axis=0)
    tang /= np.linalg.norm(tang, axis=1)[:, None]
    normal = np.c_[-tang[:, 1], tang[:, 0]]
    w = np.where((t > 1.0) & (t < 1.6), np.sin((t - 1.0) / 0.6 * np.pi) ** 2, 0.0)
    return xy + bump * w[:, None] * normal, w


def _to_game_frame(xy):
    """Mirror, rotate 30°, scale 1.05 and shift: a different frame, like the game's."""
    a = np.radians(30)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (1.05 * xy * [1, -1]) @ rot.T + [1000, -500]


class TestFitLine:
    def test_recovers_a_line_change_through_another_frame(self):
        pro, _ = _loop(1000)
        you, w = _loop(2400, bump=3.0)
        out, median = _fit_line(_to_game_frame(you), pro)
        dense, _ = _loop(20000)
        dist = cKDTree(dense).query(out)[0]
        assert median < 0.5
        assert dist[w > 0.95].max() == pytest.approx(3.0, abs=0.5)   # the 3 m wider stretch survives; the fit absorbs a little
        assert np.median(dist[w == 0]) < 0.3                          # the rest lies on the pro's line

    def test_unrelated_path_fails_the_gate(self):
        pro, _ = _loop(1000)
        noise = np.random.default_rng(0).uniform(-500, 500, (2000, 2))
        assert _fit_line(noise, pro)[1] > LINE_FIT_MAX_M


class TestBuildYouLap:
    def test_columns_match_the_grid(self):
        you = _build_you_lap(_aligned(), GRID, 20.0)
        assert set(you) == {"t", "speed", "throttle", "brake", "gear"}
        assert all(len(v) == GRID.size for v in you.values())

    def test_time_starts_at_zero_rises_and_ends_at_lap_time(self):
        # 25 s differs from the integrated 20 s, so this fails if the time pin is removed.
        t = np.array(_build_you_lap(_aligned(), GRID, 25.0)["t"])
        assert t[0] == 0
        assert np.all(np.diff(t) > 0)
        assert t[-1] == pytest.approx(25.0, abs=0.01)

    def test_alignment_anchors_move_your_samples(self):
        # game metre 400 is real metre 500: the first 400 game metres are stretched over 500 real ones.
        you = _build_you_lap(_aligned(game_anch=(0, 400, 1000), real_anch=(0, 500, 1000)), GRID, 20.0)
        def i(d):
            return int(np.argmin(np.abs(GRID - d)))
        assert you["t"][i(500)] == pytest.approx(8.0, abs=0.05)    # 400 of 1000 game metres at constant speed
        # game 500 -> real 500 + 100 * (500 / 600) = 583.3
        assert you["throttle"][i(570)] == 1.0
        assert you["throttle"][i(600)] == 0.0

    def test_gear_is_integer(self):
        you = _build_you_lap(_aligned(), GRID, 20.0)
        assert all(isinstance(g, int) for g in you["gear"])

    def test_channels_follow_the_game_lap(self):
        speed = np.linspace(5.0, 50.0, 1001)                 # km/h, slow linear ramp over 1000 game metres
        v = speed / 3.6
        t_trap = np.concatenate([[0.0], np.cumsum(1.0 / ((v[1:] + v[:-1]) / 2))])   # 1 m steps
        you = _build_you_lap(_aligned(speed=speed), GRID, float(t_trap[-1]))   # pin is then a no-op
        i = lambda d: int(np.argmin(np.abs(GRID - d)))
        assert you["speed"][i(200)] == pytest.approx(speed[200], abs=0.1)
        assert you["speed"][i(800)] == pytest.approx(speed[800], abs=0.1)
        assert you["brake"][i(400)] == 0.0
        assert you["brake"][i(600)] == 1.0
        assert you["t"][i(500)] == pytest.approx(t_trap[500], abs=0.01)
        t_left = np.concatenate([[0.0], np.cumsum(1.0 / v[:-1])])
        assert abs(t_left[500] * t_trap[-1] / t_left[-1] - t_trap[500]) > 0.05   # test can tell trapezoid from left sum

    def test_missing_end_anchor_keeps_time_rising(self):
        you = _build_you_lap(_aligned(game_anch=(0, 990), real_anch=(0, 990)), GRID, 20.0)
        t = np.array(you["t"])
        assert np.all(np.diff(t) > 0)
        assert t[-1] == pytest.approx(20.0, abs=0.01)


client = TestClient(app)
SAMPLE = Path(__file__).resolve().parents[1] / "data" / "suzuka_sample.csv"


class TestCompare3DEndpoint:
    # Uses the cached data/reference_laps/2025_ver_q_suzuka.pkl, so no network is needed.
    @pytest.fixture(scope="class")
    def payload(self):
        with SAMPLE.open("rb") as f:
            res = client.post("/analyze", files={"file": ("suzuka_sample.csv", f, "text/csv")},
                              data={"mode": "compare_3d", "driver": "VER", "year": "2025",
                                    "session": "Q", "track": "suzuka"})
        assert res.status_code == 200, res.text
        return res.json()

    def test_has_header_pro_and_you(self, payload):
        assert payload["mode"] == "compare_3d"
        assert payload["game_time"] > 0 and payload["real_time"] > 0
        assert set(payload["pro"]) >= {"meta", "points", "corners"}
        meta = payload["pro"]["meta"]
        assert (meta["driver"], meta["year"], meta["session"], meta["track"]) == ("VER", 2025, "Q", "suzuka")
        assert set(payload["you"]) == {"t", "speed", "throttle", "brake", "gear"}

    def test_you_shares_the_pro_grid(self, payload):
        n = len(payload["pro"]["points"]["d"])
        assert all(len(v) == n for v in payload["you"].values())

    def test_your_time_ends_at_your_lap_time(self, payload):
        t = np.array(payload["you"]["t"])
        assert t[0] == 0
        assert np.all(np.diff(t) > 0)
        assert t[-1] == pytest.approx(payload["game_time"], abs=0.1)

    def test_corners_carry_your_side(self, payload):
        corners = payload["pro"]["corners"]
        assert corners
        assert all("time_delta" in c and "your_min_speed_kmh" in c for c in corners)
        assert any(c["time_delta"] is not None for c in corners)

    def test_your_corner_speed_is_measured_like_the_pros(self, payload):
        # brake/lift: the pro's slowest point vs your slowest point in the corner window;
        # flat: the pro's speed at the apex vs yours at the same point.
        d = np.array(payload["pro"]["points"]["d"])
        s = np.array(payload["you"]["speed"])
        corners = payload["pro"]["corners"]
        assert any(c["action"] == "flat" for c in corners) and any(c["action"] != "flat" for c in corners)
        for c in corners:
            m = (d >= c["entry_m"]) & (d <= c["exit_m"])
            if c["action"] == "flat":
                assert c["your_min_speed_kmh"] == s[int(np.argmin(np.abs(d - c["apex_m"])))]
            elif m.any():
                assert c["your_min_speed_kmh"] == pytest.approx(s[m].min())


def _post_compare(**over):
    data = {"mode": "compare_3d", "driver": "VER", "year": "2025", "session": "Q", "track": "suzuka"}
    data.update(over)
    with SAMPLE.open("rb") as f:
        return client.post("/analyze", files={"file": ("suzuka_sample.csv", f, "text/csv")}, data=data)


class TestCompare3DErrors:
    def test_pro_lap_load_failure_is_422(self, monkeypatch):
        import app.main as m

        def flaky(*a, **k):     # app.main's own call only; src.pipeline uses its own import
            raise RuntimeError("boom")

        monkeypatch.setattr(m, "load_real_telemetry", flaky)
        res = _post_compare()
        assert res.status_code == 422
        assert res.json()["detail"] == "No 3D data for this lap"

    def test_unknown_track_fails_before_the_pipeline(self, monkeypatch):
        import app.main as m
        ran = []
        monkeypatch.setattr(m, "run_pipeline", lambda *a, **k: ran.append(1))
        res = _post_compare(track="suzuka_x")
        assert res.status_code == 422
        assert res.json()["detail"] == "unknown track"
        assert not ran


@pytest.mark.parametrize("over", [{"driver": "../x"}, {"session": "../x"}, {"year": "1800"}])
def test_analyze_rejects_bad_lap_params(monkeypatch, over):
    import app.main as m
    ran = []
    monkeypatch.setattr(m, "run_pipeline", lambda *a, **k: ran.append(1))
    res = _post_compare(mode="comparison", **over)
    assert res.status_code == 422
    assert res.json()["detail"] == "invalid driver, session or year"
    assert not ran
