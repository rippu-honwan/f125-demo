"""
Pure utility functions.
No business logic, no data loading.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


def smooth(data, window=10):
    """Smooth 1D signal with uniform filter."""
    data = np.asarray(data, dtype=float)
    if len(data) < window:
        return data
    return uniform_filter1d(data, size=window)


def normalize_pedal_array(arr, detect_boolean=False):
    """
    Normalize pedal (throttle/brake) input to 0-1 range.

    Returns (normalized_array, format_string) where format is:
      - "boolean": only {0, 1} values (when detect_boolean=True)
      - "percentage_0_100": max > 1.5, divided by 100 (NOT clamped)
      - "continuous_0_1": already in range, clamped to [0, 1]

    Caller may apply additional .clip() if needed.
    """
    arr = np.asarray(arr, dtype=float)

    if detect_boolean:
        unique_vals = np.unique(arr[~np.isnan(arr)])
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0.0, 1.0}):
            return arr, "boolean"

    if np.nanmax(arr) > 1.5:
        return arr / 100.0, "percentage_0_100"

    return arr.clip(0.0, 1.0), "continuous_0_1"


def resample_to_distance(df, track_length, columns=None, step_m=1.0,
                         distance_col='lap_distance'):
    """
    Resample DataFrame to uniform distance intervals via linear interpolation.

    If columns is None, resamples all columns except distance_col.
    """
    distances = np.arange(0, track_length, step_m)
    resampled = pd.DataFrame({distance_col: distances})

    src_dist = df[distance_col].values
    sort_idx = np.argsort(src_dist)
    src_sorted = src_dist[sort_idx]

    # Remove duplicate distances
    unique_mask = np.concatenate([[True], np.diff(src_sorted) > 0.01])
    src_unique = src_sorted[unique_mask]

    if columns is None:
        columns = [c for c in df.columns if c != distance_col]

    for col in columns:
        if col in df.columns:
            vals = df[col].values[sort_idx][unique_mask]
            resampled[col] = np.interp(distances, src_unique, vals)

    return resampled


def signed_curvature_from_smoothed(x_s, y_s, eps=1e-12):
    """
    Signed curvature from already-smoothed XY arrays.

    Formula: kappa = (dx * ddy - dy * ddx) / (dx^2 + dy^2)^1.5

    Positive = left turn, negative = right turn (assumes standard axes).
    Callers should smooth x and y first using their preferred filter.
    """
    dx = np.gradient(x_s)
    dy = np.gradient(y_s)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = np.maximum((dx**2 + dy**2)**1.5, eps)
    return (dx * ddy - dy * ddx) / denom


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