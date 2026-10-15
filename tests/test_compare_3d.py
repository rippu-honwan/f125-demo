"""Tests for the 3D Compare mode: POST /analyze mode=compare_3d and its helpers (app/main.py)."""
import numpy as np
import pandas as pd
import pytest

from app.main import _build_you_lap


def _aligned(game_anch=(0.0, 1000.0), real_anch=(0.0, 1000.0), speed=180.0):
    """A 1 km game lap at constant speed: full throttle for the first 500 game metres, then braking."""
    d = np.arange(1001, dtype=float)
    df = pd.DataFrame({
        "lap_distance": d,
        "game_speed_kmh": np.full(d.size, speed),
        "game_throttle": np.where(d < 500, 1.0, 0.0),
        "game_brake": np.where(d < 500, 0.0, 1.0),
        "game_gear": np.full(d.size, 6.0),
    })
    df.attrs["game_anch_d"] = np.array(game_anch)
    df.attrs["real_anch_d"] = np.array(real_anch)
    return df


GRID = np.arange(0.0, 1000.0 + 1e-9, 2.0)


class TestBuildYouLap:
    def test_columns_match_the_grid(self):
        you = _build_you_lap(_aligned(), GRID, 20.0)
        assert set(you) == {"t", "speed", "throttle", "brake", "gear"}
        assert all(len(v) == GRID.size for v in you.values())

    def test_time_starts_at_zero_rises_and_ends_at_lap_time(self):
        t = np.array(_build_you_lap(_aligned(), GRID, 20.0)["t"])
        assert t[0] == 0
        assert np.all(np.diff(t) > 0)
        assert t[-1] == pytest.approx(20.0, abs=0.01)

    def test_alignment_anchors_move_your_samples(self):
        # game metre 400 is real metre 500: the first 400 game metres are stretched over 500 real ones.
        you = _build_you_lap(_aligned(game_anch=(0, 400, 1000), real_anch=(0, 500, 1000)), GRID, 20.0)
        i = lambda d: int(np.argmin(np.abs(GRID - d)))
        assert you["t"][i(500)] == pytest.approx(8.0, abs=0.05)    # 400 of 1000 game metres at constant speed
        # game 500 -> real 500 + 100 * (500 / 600) = 583.3
        assert you["throttle"][i(570)] == 1.0
        assert you["throttle"][i(600)] == 0.0

    def test_gear_is_integer(self):
        you = _build_you_lap(_aligned(), GRID, 20.0)
        assert all(isinstance(g, int) for g in you["gear"])
