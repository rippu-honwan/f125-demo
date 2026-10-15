"""Shared fixtures for tests."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def circle_xy():
    """Generate a simple circular trajectory (1000 points)."""
    t = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    return np.column_stack([np.cos(t), np.sin(t)])


@pytest.fixture
def oval_xy():
    """Generate an oval trajectory (different X/Y radii)."""
    t = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    return np.column_stack([2.0 * np.cos(t), 1.0 * np.sin(t)])


@pytest.fixture
def noisy_circle_xy(circle_xy):
    """Circle with small Gaussian noise."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.01, circle_xy.shape)
    return circle_xy + noise


@pytest.fixture
def sample_game_data():
    """Minimal game DataFrame for alignment tests."""
    n = 500
    dist = np.linspace(0, 5000, n)
    speed = 200 + 50 * np.sin(dist / 500)
    return pd.DataFrame({
        'lap_distance': dist,
        'speed_kmh': speed,
        'throttle': np.clip(speed / 300, 0, 1),
        'brake': np.clip(1 - speed / 250, 0, 1),
    })


@pytest.fixture
def sample_real_data():
    """Minimal real DataFrame for alignment tests."""
    n = 600
    dist = np.linspace(0, 5200, n)
    speed = 210 + 55 * np.sin(dist / 520)
    return pd.DataFrame({
        'lap_distance': dist,
        'speed_kmh': speed,
        'throttle': np.clip(speed / 310, 0, 1),
        'brake': np.clip(1 - speed / 260, 0, 1),
    })
