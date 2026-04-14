"""
FastF1 data loader.
Downloads real F1 telemetry for comparison.

Changes from original:
  - [FIX] Normalize brake data (FastF1 boolean → continuous 0-1)
  - [NEW] Detect and document data format differences
  - [NEW] Better error messages with fallback strategies
  - [NEW] Metadata includes data format info for downstream use
"""

import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import warnings

# Cache directory
CACHE_DIR = Path(__file__).parent.parent / "cache" / "fastf1"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# ============================================================
# Track name → GP event mapping
# ============================================================
TRACK_TO_GP = {
    'suzuka': 'Japanese Grand Prix',
    'japan': 'Japanese Grand Prix',
    'japanese': 'Japanese Grand Prix',
    'bahrain': 'Bahrain Grand Prix',
    'jeddah': 'Saudi Arabian Grand Prix',
    'saudi': 'Saudi Arabian Grand Prix',
    'melbourne': 'Australian Grand Prix',
    'australia': 'Australian Grand Prix',
    'albert_park': 'Australian Grand Prix',
    'baku': 'Azerbaijan Grand Prix',
    'azerbaijan': 'Azerbaijan Grand Prix',
    'miami': 'Miami Grand Prix',
    'imola': 'Emilia Romagna Grand Prix',
    'monaco': 'Monaco Grand Prix',
    'montreal': 'Canadian Grand Prix',
    'canada': 'Canadian Grand Prix',
    'barcelona': 'Spanish Grand Prix',
    'spain': 'Spanish Grand Prix',
    'spielberg': 'Austrian Grand Prix',
    'austria': 'Austrian Grand Prix',
    'red_bull_ring': 'Austrian Grand Prix',
    'silverstone': 'British Grand Prix',
    'britain': 'British Grand Prix',
    'british': 'British Grand Prix',
    'hungaroring': 'Hungarian Grand Prix',
    'hungary': 'Hungarian Grand Prix',
    'spa': 'Belgian Grand Prix',
    'belgium': 'Belgian Grand Prix',
    'zandvoort': 'Dutch Grand Prix',
    'netherlands': 'Dutch Grand Prix',
    'monza': 'Italian Grand Prix',
    'italy': 'Italian Grand Prix',
    'singapore': 'Singapore Grand Prix',
    'marina_bay': 'Singapore Grand Prix',
    'cota': 'United States Grand Prix',
    'austin': 'United States Grand Prix',
    'usa': 'United States Grand Prix',
    'mexico': 'Mexico City Grand Prix',
    'hermanos_rodriguez': 'Mexico City Grand Prix',
    'interlagos': 'São Paulo Grand Prix',
    'brazil': 'São Paulo Grand Prix',
    'sao_paulo': 'São Paulo Grand Prix',
    'las_vegas': 'Las Vegas Grand Prix',
    'vegas': 'Las Vegas Grand Prix',
    'lusail': 'Qatar Grand Prix',
    'qatar': 'Qatar Grand Prix',
    'yas_marina': 'Abu Dhabi Grand Prix',
    'abu_dhabi': 'Abu Dhabi Grand Prix',
    'shanghai': 'Chinese Grand Prix',
    'china': 'Chinese Grand Prix',
}

SESSION_MAP = {
    'Q': 'Q', 'R': 'R',
    'FP1': 'FP1', 'FP2': 'FP2', 'FP3': 'FP3',
    'S': 'S', 'SQ': 'SQ', 'SS': 'SS',
}


def resolve_gp_name(track_hint: str, year: int) -> str:
    """Resolve track hint to official GP name."""
    if track_hint is None:
        raise ValueError(
            "Track/GP name is required. "
            "Use --gp 'Japanese Grand Prix' or --track suzuka"
        )

    hint_lower = track_hint.lower().strip().replace(' ', '_')

    if hint_lower in TRACK_TO_GP:
        gp_name = TRACK_TO_GP[hint_lower]
        print(f"  Track '{track_hint}' → {gp_name}")
        return gp_name

    if 'grand prix' in track_hint.lower():
        return track_hint

    for key, gp in TRACK_TO_GP.items():
        if hint_lower in key or key in hint_lower:
            print(f"  Track '{track_hint}' → {gp} (partial match)")
            return gp

    available = sorted(set(TRACK_TO_GP.values()))
    raise ValueError(
        f"Cannot resolve track '{track_hint}'.\n"
        f"Available: {available}"
    )


def _normalize_brake(brake_raw: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Normalize brake data to continuous 0.0-1.0.

    FastF1 brake data varies by source:
      - Some years: boolean (True/False → 0/1)
      - Some years: 0-100 percentage
      - Some years: 0-1 float

    Returns:
        (normalized_array, format_description)
    """
    brake = brake_raw.astype(float)
    unique_vals = np.unique(brake[~np.isnan(brake)])

    # Case 1: Boolean (only 0 and 1, or True/False)
    if len(unique_vals) <= 2 and set(unique_vals).issubset({0.0, 1.0}):
        return brake, "boolean"

    # Case 2: 0-100 range
    if np.nanmax(brake) > 1.5:
        return brake / 100.0, "percentage_0_100"

    # Case 3: Already 0-1 continuous
    return brake.clip(0.0, 1.0), "continuous_0_1"


def _normalize_throttle(throttle_raw: np.ndarray) -> np.ndarray:
    """Normalize throttle to 0-1 range."""
    throttle = throttle_raw.astype(float)
    if np.nanmax(throttle) > 1.5:
        return throttle / 100.0
    return throttle.clip(0.0, 1.0)


def load_real_telemetry(driver: str, year: int, session: str = 'Q',
                        track: str = None, gp: str = None
                        ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load real F1 telemetry from FastF1.

    Returns:
        (resampled_data, metadata)
        metadata includes 'brake_format' to inform downstream analysis.
    """
    # Resolve GP name
    if gp:
        gp_name = gp
        if 'grand prix' not in gp.lower():
            gp_name = resolve_gp_name(gp, year)
    elif track:
        gp_name = resolve_gp_name(track, year)
    else:
        raise ValueError("Must provide --track or --gp.")

    session_type = SESSION_MAP.get(session.upper(), session)

    print(f"  Loading: {year} {gp_name} - {session_type}")
    print(f"  Driver: {driver}")

    # Load session
    try:
        f1_session = fastf1.get_session(year, gp_name, session_type)
        f1_session.load()
    except Exception as e:
        raise RuntimeError(
            f"Failed to load FastF1 session: {e}\n"
            f"Check: year={year}, gp='{gp_name}', session='{session_type}'\n"
            f"FastF1 cache: {CACHE_DIR}"
        ) from e

    # Get driver's laps
    driver_laps = f1_session.laps.pick_drivers(driver)

    if driver_laps.empty:
        available = f1_session.laps['Driver'].unique().tolist()
        raise ValueError(
            f"Driver '{driver}' not found in {year} {gp_name} {session_type}.\n"
            f"Available: {available}"
        )

    # Find fastest lap (with fallback)
    fastest = driver_laps.pick_fastest()

    if fastest is None or pd.isna(fastest['LapTime']):
        valid = driver_laps.dropna(subset=['LapTime'])
        if valid.empty:
            raise ValueError(f"No valid laps for {driver}")
        fastest = valid.loc[valid['LapTime'].idxmin()]
        print(f"  ⚠ Using manual fastest lap selection (pick_fastest failed)")

    lap_time = fastest['LapTime'].total_seconds()
    print(f"  Lap time: {int(lap_time//60)}:{lap_time%60:06.3f}")

    # Get telemetry
    tel = fastest.get_telemetry()

    if tel is None or tel.empty:
        raise ValueError(f"No telemetry data for {driver}'s fastest lap")

    print(f"  Telemetry points: {len(tel)}")

    # Build standardized DataFrame
    data = pd.DataFrame()

    if 'Distance' not in tel.columns:
        raise ValueError("No Distance column in FastF1 telemetry")
    data['lap_distance'] = tel['Distance'].values

    track_length = float(data['lap_distance'].max())
    print(f"  Track length: {track_length:.0f}m")

    if 'Speed' not in tel.columns:
        raise ValueError("No Speed column in FastF1 telemetry")
    data['speed_kmh'] = tel['Speed'].values.astype(float)

    # Throttle (normalize to 0-1)
    if 'Throttle' in tel.columns:
        data['throttle'] = _normalize_throttle(tel['Throttle'].values)

    # Brake (normalize + detect format)
    brake_format = "unavailable"
    if 'Brake' in tel.columns:
        data['brake'], brake_format = _normalize_brake(tel['Brake'].values)
        print(f"  Brake data format: {brake_format}")

    # Gear
    if 'nGear' in tel.columns:
        data['gear'] = tel['nGear'].values.astype(int)

    # RPM
    if 'RPM' in tel.columns:
        data['rpm'] = tel['RPM'].values.astype(float)

    # DRS
    if 'DRS' in tel.columns:
        data['drs'] = tel['DRS'].values.astype(int)

    # Position
    if 'X' in tel.columns:
        data['world_position_X'] = tel['X'].values.astype(float)
        data['world_position_Y'] = tel['Y'].values.astype(float)
        if 'Z' in tel.columns:
            data['world_position_Z'] = tel['Z'].values.astype(float)

    # Resample to 1m intervals
    distances = np.arange(0, track_length, 1.0)
    resampled = pd.DataFrame({'lap_distance': distances})

    src_dist = data['lap_distance'].values
    sort_idx = np.argsort(src_dist)
    src_sorted = src_dist[sort_idx]

    # Remove duplicates
    unique_mask = np.concatenate([[True], np.diff(src_sorted) > 0.01])
    src_unique = src_sorted[unique_mask]

    for col in data.columns:
        if col == 'lap_distance':
            continue
        vals = data[col].values[sort_idx][unique_mask]
        resampled[col] = np.interp(distances, src_unique, vals)

    meta = {
        'driver': driver,
        'year': year,
        'session': session_type,
        'gp_name': gp_name,
        'lap_time': lap_time,
        'track_length': track_length,
        'telemetry_points': len(tel),
        'brake_format': brake_format,  # NEW: downstream needs this
    }

    return resampled, meta