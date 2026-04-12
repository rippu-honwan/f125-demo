"""
Pure utility functions.
No business logic, no data loading.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d


def smooth(data, window=10):
    """Smooth 1D signal with uniform filter."""
    data = np.asarray(data, dtype=float)
    if len(data) < window:
        return data
    return uniform_filter1d(data, size=window)


def format_laptime(seconds):
    """Format seconds to M:SS.mmm."""
    if seconds is None or seconds <= 0:
        return "N/A"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{secs:06.3f}"


def format_delta(seconds):
    """Format time delta with sign."""
    if seconds is None:
        return "N/A"
    sign = "+" if seconds >= 0 else ""
    return f"{sign}{seconds:.3f}s"


def calculate_time_delta(aligned):
    """Add cumulative time delta column to aligned DataFrame."""
    game_ms = np.maximum(aligned['game_speed_kmh'].values / 3.6, 1.0)
    real_ms = np.maximum(aligned['real_speed_kmh'].values / 3.6, 1.0)

    delta_per_m = (1.0 / game_ms) - (1.0 / real_ms)
    aligned['time_delta'] = np.cumsum(delta_per_m)

    return aligned