"""
Telemetry data loader.
Only responsible for reading CSV and basic validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any


# ============================================================
# Column mapping: game CSV → standard names
# ============================================================
COLUMN_MAP = {
    # Distance
    'lap_distance': 'lap_distance',
    'LapDist': 'lap_distance',
    'distance': 'lap_distance',

    # Speed (direct)
    'speed_kmh': 'speed_kmh',
    'Speed': 'speed_kmh',
    'speed': 'speed_kmh',

    # Velocity components (for calculating speed)
    'velocity_X': 'velocity_X',
    'velocity_Y': 'velocity_Y',
    'velocity_Z': 'velocity_Z',

    # Inputs
    'throttle': 'throttle',
    'Throttle': 'throttle',
    'brake': 'brake',
    'Brake': 'brake',
    'steering': 'steering',
    'Steering': 'steering',
    'gear': 'gear',
    'Gear': 'gear',
    'nGear': 'gear',

    # Position
    'world_position_X': 'world_position_X',
    'world_position_Y': 'world_position_Y',
    'world_position_Z': 'world_position_Z',
    'WorldPositionX': 'world_position_X',
    'WorldPositionY': 'world_position_Y',
    'WorldPositionZ': 'world_position_Z',

    # Lap info
    'lap_number': 'lap_number',
    'lap_num': 'lap_number',
    'lapNum': 'lap_number',
    'LapNumber': 'lap_number',

    # Time
    'lap_time': 'lap_time',
    'current_lap_time': 'lap_time',
    'CurrentLapTime': 'lap_time',
}

REQUIRED_STANDARD = ['lap_distance']


def detect_separator(csv_path: str) -> str:
    """Auto-detect CSV separator."""
    with open(csv_path, 'r') as f:
        first_line = f.readline()

    tab_count = first_line.count('\t')
    comma_count = first_line.count(',')
    semicolon_count = first_line.count(';')

    if tab_count > comma_count and tab_count > semicolon_count:
        return '\t'
    elif semicolon_count > comma_count:
        return ';'
    return ','


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw column names to standard names, avoiding duplicates."""
    rename = {}
    used_standard = set()

    # Prioritize original column names (exact match first)
    for raw_col in df.columns:
        raw_stripped = raw_col.strip()
        if raw_stripped in COLUMN_MAP:
            standard = COLUMN_MAP[raw_stripped]
            # Skip if this standard name is already taken
            if standard in used_standard:
                continue
            # Skip if raw name == standard name (already correct)
            if raw_stripped == standard:
                used_standard.add(standard)
                continue
            # Skip if standard name already exists as another column
            if standard in df.columns and raw_stripped != standard:
                used_standard.add(standard)
                continue
            rename[raw_col] = standard
            used_standard.add(standard)

    df = df.rename(columns=rename)

    # Remove any remaining duplicate columns
    df = df.loc[:, ~df.columns.duplicated(keep='first')]

    return df


def calculate_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate speed_kmh from velocity components if needed."""
    if 'speed_kmh' in df.columns:
        return df

    if all(c in df.columns for c in ['velocity_X', 'velocity_Y', 'velocity_Z']):
        vx = df['velocity_X'].values
        vy = df['velocity_Y'].values
        vz = df['velocity_Z'].values
        speed_ms = np.sqrt(vx**2 + vy**2 + vz**2)
        df['speed_kmh'] = speed_ms * 3.6
        print(f"  Calculated speed_kmh from velocity (max: {df['speed_kmh'].max():.0f} km/h)")
        return df

    if all(c in df.columns for c in ['velocity_X', 'velocity_Z']):
        vx = df['velocity_X'].values
        vz = df['velocity_Z'].values
        speed_ms = np.sqrt(vx**2 + vz**2)
        df['speed_kmh'] = speed_ms * 3.6
        print(f"  Calculated speed_kmh from velocity XZ (max: {df['speed_kmh'].max():.0f} km/h)")
        return df

    raise ValueError(
        "Cannot determine speed. Need 'speed_kmh' or "
        "'velocity_X/Y/Z' columns."
    )


def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Load raw telemetry CSV with auto-detection.

    Returns:
        Standardized DataFrame
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    # Auto-detect separator
    sep = detect_separator(str(path))
    sep_name = {'\\t': 'TAB', ',': 'COMMA', ';': 'SEMICOLON'}.get(
        sep if sep != '\t' else '\\t', sep
    )

    df = pd.read_csv(path, sep=sep)
    print(f"  Loaded: {path.name} ({len(df)} rows, {len(df.columns)} columns, sep={sep_name})")

    # Standardize column names
    df = standardize_columns(df)

    # Calculate speed if needed
    df = calculate_speed(df)

    # Validate required columns
    for col in REQUIRED_STANDARD:
        if col not in df.columns:
            print(f"  Available columns: {sorted(df.columns.tolist())}")
            raise ValueError(f"Missing required column: {col}")

    return df


def extract_laps(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Split raw data into individual laps."""
    if 'lap_number' not in df.columns:
        return {1: df.copy()}

    laps = {}
    for lap_num, group in df.groupby('lap_number'):
        lap_num = int(lap_num)
        if len(group) < 100:
            continue
        laps[lap_num] = group.reset_index(drop=True)

    print(f"  Found {len(laps)} laps: {sorted(laps.keys())}")
    return laps


def find_best_lap(laps: Dict[int, pd.DataFrame]) -> Tuple[int, pd.DataFrame]:
    """Find fastest complete lap."""
    best_num = None
    best_time = float('inf')
    best_data = None

    for lap_num, lap_data in laps.items():
        # Skip very short laps (pit laps, etc.)
        max_dist = lap_data['lap_distance'].max()
        if max_dist < 1000:
            continue

        if 'lap_time' in lap_data.columns:
            lap_time = float(lap_data['lap_time'].max())
        else:
            # Estimate from distance and speed
            dist = lap_data['lap_distance'].values
            speed_ms = np.maximum(lap_data['speed_kmh'].values / 3.6, 1.0)
            diffs = np.diff(dist)
            valid = diffs > 0
            if valid.sum() == 0:
                continue
            lap_time = float(np.sum(diffs[valid] / speed_ms[1:][valid]))

        if 30 < lap_time < best_time:
            best_time = lap_time
            best_num = lap_num
            best_data = lap_data

    if best_data is None:
        raise ValueError("No valid laps found")

    return best_num, best_data


def get_track_length(lap_data: pd.DataFrame) -> float:
    """Get track length from lap data."""
    return float(lap_data['lap_distance'].max())


def get_lap_time(lap_data: pd.DataFrame) -> float:
    """Get lap time from lap data."""
    if 'lap_time' in lap_data.columns:
        return float(lap_data['lap_time'].max())

    dist = lap_data['lap_distance'].values
    speed_ms = np.maximum(lap_data['speed_kmh'].values / 3.6, 1.0)
    diffs = np.diff(dist)
    valid = diffs > 0
    return float(np.sum(diffs[valid] / speed_ms[1:][valid]))


def resample_lap(lap_data: pd.DataFrame, track_length: float,
                 step_m: float = 1.0) -> pd.DataFrame:
    """Resample telemetry to uniform 1m intervals."""
    distances = np.arange(0, track_length, step_m)
    resampled = pd.DataFrame({'lap_distance': distances})

    src_dist = lap_data['lap_distance'].values

    # Sort by distance to ensure monotonic
    sort_idx = np.argsort(src_dist)
    src_dist_sorted = src_dist[sort_idx]

    for col in ['speed_kmh', 'throttle', 'brake', 'steering', 'gear',
                'world_position_X', 'world_position_Y', 'world_position_Z']:
        if col in lap_data.columns:
            src_vals = lap_data[col].values[sort_idx]
            resampled[col] = np.interp(distances, src_dist_sorted, src_vals)

    return resampled


def load_and_prepare(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Full pipeline: load CSV → find best lap → resample.

    Returns:
        (resampled_data, metadata_dict)
    """
    raw = load_csv(csv_path)
    laps = extract_laps(raw)
    lap_num, best_lap = find_best_lap(laps)
    track_length = get_track_length(best_lap)
    lap_time = get_lap_time(best_lap)
    resampled = resample_lap(best_lap, track_length)

    meta = {
        'csv_path': str(csv_path),
        'total_rows': len(raw),
        'total_laps': len(laps),
        'best_lap_number': lap_num,
        'best_time': lap_time,
        'track_length': track_length,
        'columns': list(raw.columns),
    }

    print(f"  Best lap: #{lap_num} ({lap_time:.3f}s)")
    print(f"  Track length: {track_length:.0f}m")
    print(f"  Resampled: {len(resampled)} points")

    return resampled, meta