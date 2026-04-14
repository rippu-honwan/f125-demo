"""
Telemetry data loader.
Reads SRT CSV, validates, filters, and resamples.

Changes from original:
  - [FIX] Filter by validBin=1 (was ignoring 33% invalid rows)
  - [FIX] Proper lap time extraction from SRT's running timer
  - [NEW] Data validation: NaN check, range check, outlier detection
  - [NEW] Normalize brake/throttle to consistent 0-1 range
  - [NEW] Extract extra SRT columns (tyre temp, wear, ERS) for future use
  - [REFACTOR] Cleaner separation of concerns
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import warnings


# ============================================================
# Column mapping: SRT CSV → standard names
# ============================================================
COLUMN_MAP = {
    # Distance
    'lap_distance': 'lap_distance',
    'LapDist': 'lap_distance',
    'distance': 'lap_distance',

    # Speed (direct — rare in SRT, usually calculated)
    'speed_kmh': 'speed_kmh',
    'Speed': 'speed_kmh',
    'speed': 'speed_kmh',

    # Velocity components (SRT uses m/s)
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

    # SRT system fields
    'validBin': 'valid_bin',
    'lapFlag': 'lap_flag',
    'binIndex': 'bin_index',
    'trackLength': 'track_length_meta',
    'trackId': 'track_id',
    'carId': 'car_id',
}

# Extra columns worth preserving for future analysis
EXTRA_COLUMNS = {
    'tyre_wear_0': 'tyre_wear_fl',
    'tyre_wear_1': 'tyre_wear_fr',
    'tyre_wear_2': 'tyre_wear_rl',
    'tyre_wear_3': 'tyre_wear_rr',
    'tyre_temp_0': 'tyre_temp_fl',
    'tyre_temp_1': 'tyre_temp_fr',
    'tyre_temp_2': 'tyre_temp_rl',
    'tyre_temp_3': 'tyre_temp_rr',
    'ers_store': 'ers_store',
    'ers_deployMode': 'ers_deploy_mode',
    'drs': 'drs',
    'rpm': 'rpm',
    'gforce_X': 'gforce_longitudinal',
    'gforce_Y': 'gforce_lateral',
}

REQUIRED_STANDARD = ['lap_distance']


def detect_separator(csv_path: str) -> str:
    """Auto-detect CSV separator (SRT uses TAB)."""
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
    all_maps = {**COLUMN_MAP, **EXTRA_COLUMNS}
    rename = {}
    used_standard = set()

    for raw_col in df.columns:
        raw_stripped = raw_col.strip()
        if raw_stripped in all_maps:
            standard = all_maps[raw_stripped]
            if standard in used_standard:
                continue
            if raw_stripped == standard:
                used_standard.add(standard)
                continue
            if standard in df.columns and raw_stripped != standard:
                used_standard.add(standard)
                continue
            rename[raw_col] = standard
            used_standard.add(standard)

    df = df.rename(columns=rename)
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    return df


def calculate_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate speed_kmh from velocity components if needed."""
    if 'speed_kmh' in df.columns:
        # Validate existing speed
        if df['speed_kmh'].max() < 10:
            warnings.warn("speed_kmh appears to be in m/s, converting to km/h")
            df['speed_kmh'] = df['speed_kmh'] * 3.6
        return df

    # SRT provides velocity in m/s as 3D components
    if all(c in df.columns for c in ['velocity_X', 'velocity_Y', 'velocity_Z']):
        vx = df['velocity_X'].values
        vy = df['velocity_Y'].values
        vz = df['velocity_Z'].values
        speed_ms = np.sqrt(vx**2 + vy**2 + vz**2)
        df = df.assign(speed_kmh=speed_ms * 3.6)
        print(f"  Calculated speed from velocity XYZ "
              f"(max: {df['speed_kmh'].max():.0f} km/h)")
        return df

    # Fallback: 2D (some games don't export Y)
    if all(c in df.columns for c in ['velocity_X', 'velocity_Z']):
        vx = df['velocity_X'].values
        vz = df['velocity_Z'].values
        speed_ms = np.sqrt(vx**2 + vz**2)
        df = df.assign(speed_kmh=speed_ms * 3.6)
        print(f"  Calculated speed from velocity XZ "
              f"(max: {df['speed_kmh'].max():.0f} km/h)")
        return df

    raise ValueError(
        "Cannot determine speed. Need 'speed_kmh' or "
        "'velocity_X/Y/Z' columns."
    )


def normalize_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize throttle and brake to consistent 0.0-1.0 range.

    SRT already uses 0-1, but some sources use 0-100 or boolean.
    This ensures consistency regardless of source.
    """
    for col in ['throttle', 'brake']:
        if col not in df.columns:
            continue

        vals = df[col].values.astype(float)

        # Detect 0-100 range
        if np.nanmax(vals) > 1.5:
            df[col] = vals / 100.0
            print(f"  Normalized {col}: 0-100 → 0-1")

        # Clamp to valid range
        df[col] = df[col].clip(0.0, 1.0)

    return df


def validate_data(df: pd.DataFrame, context: str = "data") -> pd.DataFrame:
    """
    Validate and clean telemetry data.
    Fixes common issues instead of just crashing.
    """
    issues = []

    # Check for NaN in critical columns
    for col in ['lap_distance', 'speed_kmh']:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                issues.append(f"{col}: {nan_count} NaN values (filled with interpolation)")
                df[col] = df[col].interpolate(method='linear').ffill().bfill()

    # Speed sanity check
    if 'speed_kmh' in df.columns:
        max_spd = df['speed_kmh'].max()
        if max_spd > 400:
            outliers = (df['speed_kmh'] > 400).sum()
            issues.append(f"speed_kmh: {outliers} values > 400 km/h (capped)")
            df['speed_kmh'] = df['speed_kmh'].clip(0, 400)
        if max_spd < 50:
            issues.append(f"speed_kmh max is only {max_spd:.0f} — data may be wrong")

    # Distance should be monotonically increasing within a lap
    if 'lap_distance' in df.columns:
        diffs = df['lap_distance'].diff()
        neg_count = (diffs < -10).sum()  # allow small noise
        if neg_count > 0:
            issues.append(f"lap_distance: {neg_count} backward jumps detected")

    if issues:
        print(f"  ⚠ Validation ({context}):")
        for issue in issues:
            print(f"    - {issue}")

    return df


# ============================================================
# CSV loading
# ============================================================

def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Load raw telemetry CSV with auto-detection.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    sep = detect_separator(str(path))
    sep_name = {'\t': 'TAB', ',': 'COMMA', ';': 'SEMICOLON'}.get(sep, sep)

    df = pd.read_csv(path, sep=sep)
    print(f"  Loaded: {path.name} "
          f"({len(df)} rows, {len(df.columns)} cols, sep={sep_name})")

    # Standardize
    df = standardize_columns(df)
    df = df.copy()  # defragment after rename (266 cols → many renames)
    df = calculate_speed(df)
    df = normalize_inputs(df)

    # Validate required
    for col in REQUIRED_STANDARD:
        if col not in df.columns:
            print(f"  Available: {sorted(df.columns.tolist())}")
            raise ValueError(f"Missing required column: {col}")

    return df


# ============================================================
# Lap extraction (with validBin filtering)
# ============================================================

def extract_laps(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """
    Split raw data into individual laps.

    KEY FIX: Filters by validBin=1 and lapFlag=0.
    SRT marks invalid bins and race-start segments that
    should be excluded from analysis.
    """
    # Filter valid bins if the column exists
    if 'valid_bin' in df.columns:
        before = len(df)
        df = df[df['valid_bin'] == 1].copy()
        dropped = before - len(df)
        if dropped > 0:
            print(f"  Filtered: {dropped} invalid bins removed "
                  f"({dropped/before*100:.0f}%)")

    # Filter lap flag (exclude race-start segments)
    if 'lap_flag' in df.columns:
        df = df[df['lap_flag'] == 0].copy()

    # Group by lap number
    lap_col = 'lap_number' if 'lap_number' in df.columns else None
    if lap_col is None:
        return {1: df.copy()}

    laps = {}
    for lap_num, group in df.groupby(lap_col):
        lap_num = int(lap_num)
        group = group.reset_index(drop=True)

        # Need meaningful data (at least 500m of valid distance)
        max_dist = group['lap_distance'].max()
        if max_dist < 500:
            print(f"  Lap {lap_num}: skipped (only {max_dist:.0f}m coverage)")
            continue

        # Need reasonable number of data points
        if len(group) < 200:
            print(f"  Lap {lap_num}: skipped (only {len(group)} bins)")
            continue

        laps[lap_num] = group

    print(f"  Valid laps: {sorted(laps.keys())}")
    return laps


def find_best_lap(laps: Dict[int, pd.DataFrame]) -> Tuple[int, pd.DataFrame]:
    """
    Find fastest complete lap.

    KEY FIX: Uses proper lap time extraction.
    SRT's lap_time is a running counter within each lap,
    so the actual lap time is the maximum value for that lap.
    We also cross-validate against distance/speed estimation.
    """
    best_num = None
    best_time = float('inf')
    best_data = None

    for lap_num, lap_data in laps.items():
        max_dist = lap_data['lap_distance'].max()

        # Must cover most of the track
        if max_dist < 1000:
            continue

        # Method 1: Use SRT lap_time field (running timer)
        if 'lap_time' in lap_data.columns:
            lap_time = float(lap_data['lap_time'].max())
        else:
            lap_time = None

        # Method 2: Estimate from distance/speed (cross-validation)
        dist = lap_data['lap_distance'].values
        speed_ms = np.maximum(lap_data['speed_kmh'].values / 3.6, 1.0)
        diffs = np.diff(dist)
        valid_mask = diffs > 0
        estimated_time = float(np.sum(diffs[valid_mask] / speed_ms[1:][valid_mask]))

        # Prefer SRT lap_time, fallback to estimation
        if lap_time and 30 < lap_time < 300:
            final_time = lap_time
            # Sanity check: estimation should be within 5% of recorded
            if estimated_time > 0:
                diff_pct = abs(final_time - estimated_time) / final_time * 100
                if diff_pct > 10:
                    print(f"  Lap {lap_num}: time mismatch "
                          f"(recorded={final_time:.1f}s, "
                          f"estimated={estimated_time:.1f}s, "
                          f"diff={diff_pct:.0f}%)")
        elif 30 < estimated_time < 300:
            final_time = estimated_time
        else:
            continue

        if final_time < best_time:
            best_time = final_time
            best_num = lap_num
            best_data = lap_data

    if best_data is None:
        raise ValueError("No valid laps found in data")

    return best_num, best_data


def get_track_length(lap_data: pd.DataFrame) -> float:
    """Get track length from lap data."""
    # Prefer SRT metadata if available
    if 'track_length_meta' in lap_data.columns:
        meta_length = float(lap_data['track_length_meta'].iloc[0])
        if meta_length > 1000:
            return meta_length

    return float(lap_data['lap_distance'].max())


def get_lap_time(lap_data: pd.DataFrame) -> float:
    """Get lap time from lap data."""
    if 'lap_time' in lap_data.columns:
        t = float(lap_data['lap_time'].max())
        if 30 < t < 300:
            return t

    # Fallback: distance/speed estimation
    dist = lap_data['lap_distance'].values
    speed_ms = np.maximum(lap_data['speed_kmh'].values / 3.6, 1.0)
    diffs = np.diff(dist)
    valid = diffs > 0
    return float(np.sum(diffs[valid] / speed_ms[1:][valid]))


def get_track_id(lap_data: pd.DataFrame) -> Optional[str]:
    """Extract track ID from SRT metadata."""
    if 'track_id' in lap_data.columns:
        return str(lap_data['track_id'].iloc[0]).strip()
    return None


def get_car_id(lap_data: pd.DataFrame) -> Optional[str]:
    """Extract car/team ID from SRT metadata."""
    if 'car_id' in lap_data.columns:
        return str(lap_data['car_id'].iloc[0]).strip()
    return None


def resample_lap(lap_data: pd.DataFrame, track_length: float,
                 step_m: float = 1.0) -> pd.DataFrame:
    """Resample telemetry to uniform distance intervals."""
    distances = np.arange(0, track_length, step_m)
    resampled = pd.DataFrame({'lap_distance': distances})

    src_dist = lap_data['lap_distance'].values

    # Sort by distance (mandatory for interpolation)
    sort_idx = np.argsort(src_dist)
    src_dist_sorted = src_dist[sort_idx]

    # Remove duplicate distances (can happen with SRT bins)
    unique_mask = np.concatenate([[True], np.diff(src_dist_sorted) > 0.01])
    src_dist_unique = src_dist_sorted[unique_mask]

    # Columns to resample
    resample_cols = [
        'speed_kmh', 'throttle', 'brake', 'steering', 'gear',
        'world_position_X', 'world_position_Y', 'world_position_Z',
        # Extra columns if available
        'tyre_wear_fl', 'tyre_wear_fr', 'tyre_wear_rl', 'tyre_wear_rr',
        'tyre_temp_fl', 'tyre_temp_fr', 'tyre_temp_rl', 'tyre_temp_rr',
        'gforce_longitudinal', 'gforce_lateral',
        'rpm', 'drs', 'ers_store',
    ]

    for col in resample_cols:
        if col in lap_data.columns:
            src_vals = lap_data[col].values[sort_idx][unique_mask]
            resampled[col] = np.interp(distances, src_dist_unique, src_vals)

    return resampled


# ============================================================
# Main pipeline
# ============================================================

def load_and_prepare(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Full pipeline: load CSV → filter valid → find best lap → resample.

    Returns:
        (resampled_data, metadata_dict)
    """
    raw = load_csv(csv_path)
    laps = extract_laps(raw)
    lap_num, best_lap = find_best_lap(laps)
    track_length = get_track_length(best_lap)
    lap_time = get_lap_time(best_lap)

    # Validate before resample
    best_lap = validate_data(best_lap, context=f"lap #{lap_num}")

    resampled = resample_lap(best_lap, track_length)

    # Validate after resample
    resampled = validate_data(resampled, context="resampled")

    meta = {
        'csv_path': str(csv_path),
        'total_rows': len(raw),
        'total_laps': len(laps),
        'best_lap_number': lap_num,
        'best_time': lap_time,
        'track_length': track_length,
        'track_id': get_track_id(best_lap),
        'car_id': get_car_id(best_lap),
        'columns': list(raw.columns),
    }

    print(f"  Best lap: #{lap_num} ({lap_time:.3f}s)")
    print(f"  Track: {meta['track_id'] or 'unknown'} ({track_length:.0f}m)")
    print(f"  Team: {meta['car_id'] or 'unknown'}")
    print(f"  Resampled: {len(resampled)} points")

    return resampled, meta