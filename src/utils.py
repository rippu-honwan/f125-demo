"""
F1 Lap Insight - Utility Functions
All shared data loading, processing, and formatting functions.
"""

import numpy as np
import pandas as pd


# ============================================================
# Data Loading
# ============================================================

def load_telemetry(csv_path):
    """
    Load F1 game telemetry CSV and extract best lap.
    Auto-detects delimiter (comma, tab, semicolon).
    """
    with open(csv_path, 'r') as f:
        first_line = f.readline()

    if '\t' in first_line:
        sep = '\t'
        print(f"  Detected: Tab-separated")
    elif ';' in first_line:
        sep = ';'
        print(f"  Detected: Semicolon-separated")
    else:
        sep = ','
        print(f"  Detected: Comma-separated")

    df = pd.read_csv(csv_path, sep=sep)

    print(f"  CSV columns: {len(df.columns)} columns")
    print(f"  Total rows: {len(df)}")

    # Speed: calculate from velocity if no speed column
    speed_col = None
    for candidate in ['speed_kmh', 'speed', 'Speed', 'speedKmh', 'carSpeed']:
        if candidate in df.columns:
            speed_col = candidate
            break

    if speed_col is None and 'velocity_X' in df.columns and 'velocity_Z' in df.columns:
        vx = df['velocity_X'].values
        vy = df.get('velocity_Y', pd.Series(np.zeros(len(df)))).values
        vz = df['velocity_Z'].values
        df['speed_kmh'] = np.sqrt(vx**2 + vy**2 + vz**2) * 3.6
        speed_col = 'speed_kmh'
        print(f"  Calculated speed from velocity components")

    # Detect lap column
    lap_col = None
    for candidate in ['lap_number', 'lap', 'Lap', 'LAP', 'LapNumber', 'lapNum']:
        if candidate in df.columns:
            lap_col = candidate
            break

    if lap_col is None:
        print(f"  WARNING: No lap column found, treating as single lap")
        df['lap'] = 1
        lap_col = 'lap'

    # Detect distance column
    dist_col = None
    for candidate in ['lap_distance', 'LapDistance', 'distance', 'Distance',
                       'lap_dist', 'trackPosition']:
        if candidate in df.columns:
            dist_col = candidate
            break

    if dist_col is None:
        raise ValueError(f"Cannot find distance column in: {list(df.columns)}")

    if speed_col is None:
        raise ValueError(f"Cannot find speed column in: {list(df.columns)}")

    # Standardize column names
    rename_map = {}
    if dist_col != 'lap_distance':
        rename_map[dist_col] = 'lap_distance'
    if speed_col != 'speed_kmh':
        rename_map[speed_col] = 'speed_kmh'
    if lap_col != 'lap':
        rename_map[lap_col] = 'lap'

    optional_cols = {
        'throttle': ['throttle', 'Throttle', 'throttleInput'],
        'brake': ['brake', 'Brake', 'brakeInput'],
        'steering': ['steering', 'Steering', 'steeringInput'],
        'gear': ['gear', 'Gear', 'nGear', 'currentGear'],
        'world_position_X': ['world_position_X', 'worldPositionX', 'posX', 'X'],
        'world_position_Y': ['world_position_Y', 'worldPositionY', 'posY', 'Y'],
        'world_position_Z': ['world_position_Z', 'worldPositionZ', 'posZ', 'Z'],
    }

    for standard_name, candidates in optional_cols.items():
        for c in candidates:
            if c in df.columns and c != standard_name:
                rename_map[c] = standard_name
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    key_cols = ['lap_distance', 'speed_kmh', 'throttle', 'brake',
                'steering', 'gear', 'world_position_X']
    found = [c for c in key_cols if c in df.columns]
    print(f"  Key columns found: {found}")

    # Calculate lap times
    laps = df['lap'].unique()
    print(f"  Found {len(laps)} laps")

    lap_times = {}
    for lap_num in laps:
        lap_data = df[df['lap'] == lap_num].copy()

        if len(lap_data) < 50:
            continue

        track_len = lap_data['lap_distance'].max()

        if track_len < 1000:
            continue

        if 'lap_time' in lap_data.columns:
            lt = lap_data['lap_time'].iloc[-1]
            if 30 < lt < 300:
                lap_times[lap_num] = float(lt)
                continue

        distances = lap_data['lap_distance'].values
        speeds_ms = np.maximum(lap_data['speed_kmh'].values / 3.6, 1.0)

        dist_deltas = np.diff(distances, prepend=distances[0])
        dist_deltas = np.maximum(dist_deltas, 0)

        time_est = np.sum(dist_deltas / speeds_ms)

        if 30 < time_est < 300:
            lap_times[lap_num] = time_est

    if not lap_times:
        raise ValueError("No valid laps found")

    best_lap_num = min(lap_times, key=lap_times.get)
    best_time = lap_times[best_lap_num]

    print(f"  Best lap: #{best_lap_num} ({format_laptime(best_time)})")

    for lap_num in sorted(lap_times.keys()):
        marker = " <- BEST" if lap_num == best_lap_num else ""
        print(f"    Lap {lap_num}: {format_laptime(lap_times[lap_num])}{marker}")

    best_lap = df[df['lap'] == best_lap_num].copy()
    best_lap = best_lap.sort_values('lap_distance').reset_index(drop=True)

    track_length = float(best_lap['lap_distance'].max())

    track_name = "Unknown"
    csv_str = str(csv_path).lower()
    track_names = {
        'suzuka': 'Suzuka', 'monza': 'Monza', 'spa': 'Spa',
        'silverstone': 'Silverstone', 'monaco': 'Monaco',
        'bahrain': 'Bahrain', 'jeddah': 'Jeddah',
        'melbourne': 'Melbourne', 'imola': 'Imola',
        'barcelona': 'Barcelona', 'montreal': 'Montreal',
        'spielberg': 'Spielberg', 'hungaroring': 'Hungaroring',
        'zandvoort': 'Zandvoort', 'singapore': 'Singapore',
        'austin': 'Austin', 'interlagos': 'Interlagos',
        'vegas': 'LasVegas', 'lusail': 'Lusail',
        'yas': 'YasMarina', 'shanghai': 'Shanghai',
    }
    for key, name in track_names.items():
        if key in csv_str:
            track_name = name
            break

    if track_name == "Unknown":
        if 5700 < track_length < 5900:
            track_name = "Suzuka"

    meta = {
        'track_name': track_name,
        'track_length': track_length,
        'best_lap': best_lap_num,
        'best_time': best_time,
        'total_laps': len(laps),
        'all_lap_times': lap_times,
        'columns': list(best_lap.columns),
    }

    print(f"  Track: {track_name} ({track_length:.0f}m)")

    return best_lap, meta


# ============================================================
# Data Processing
# ============================================================

def resample(lap_data, track_length, interval=1.0):
    """
    Resample telemetry to uniform distance intervals.
    Only keeps essential columns for performance.
    """
    keep_cols = [
        'speed_kmh', 'throttle', 'brake', 'steering', 'gear',
        'world_position_X', 'world_position_Y', 'world_position_Z',
        'velocity_X', 'velocity_Y', 'velocity_Z',
        'rpm', 'drs', 'lap_time',
    ]

    distances = np.arange(0, track_length, interval)
    src_dist = lap_data['lap_distance'].values

    result = {'lap_distance': distances}

    for col in keep_cols:
        if col in lap_data.columns:
            try:
                values = lap_data[col].values.astype(float)
                result[col] = np.interp(distances, src_dist, values)
            except (ValueError, TypeError):
                pass

    return pd.DataFrame(result)


def smooth(data, window=15):
    """
    Smooth data using a moving average.
    """
    if window <= 1:
        return data

    data = np.array(data, dtype=float)
    kernel = np.ones(window) / window

    pad = window // 2
    padded = np.pad(data, (pad, pad), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')

    if len(smoothed) > len(data):
        smoothed = smoothed[:len(data)]
    elif len(smoothed) < len(data):
        smoothed = np.append(smoothed, [smoothed[-1]] * (len(data) - len(smoothed)))

    return smoothed


# ============================================================
# Formatting
# ============================================================

def format_laptime(seconds):
    """Format seconds to M:SS.mmm"""
    if seconds is None or seconds <= 0:
        return "N/A"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{secs:06.3f}"


def format_delta(seconds):
    """Format time delta with sign."""
    if seconds is None:
        return "N/A"
    return f"{seconds:+.3f}s"


# ============================================================
# Alignment Utilities (for Step 3)
# ============================================================

def find_braking_points(speed, threshold_decel=5.0, min_gap=100):
    """
    Find braking initiation points.
    """
    speed_s = smooth(speed, 10)
    decel = -np.diff(speed_s, prepend=speed_s[0])

    braking = decel > threshold_decel

    starts = []
    in_brake = False
    last_start = -min_gap

    for i in range(len(braking)):
        if braking[i] and not in_brake and (i - last_start) > min_gap:
            starts.append(i)
            last_start = i
            in_brake = True
        elif not braking[i]:
            in_brake = False

    return np.array(starts)


def find_throttle_points(throttle, threshold=0.8, min_gap=100):
    """
    Find full-throttle initiation points (corner exit).
    """
    throttle_s = smooth(throttle, 10)
    full = throttle_s > threshold

    starts = []
    in_full = False
    last_start = -min_gap

    for i in range(len(full)):
        if full[i] and not in_full and (i - last_start) > min_gap:
            starts.append(i)
            last_start = i
            in_full = True
        elif not full[i]:
            in_full = False

    return np.array(starts)