"""
FastF1 data loader.
Downloads real F1 telemetry for comparison.
"""

import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# Cache directory
CACHE_DIR = Path(__file__).parent.parent / "cache" / "fastf1"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# ============================================================
# Track name → GP event mapping
# ============================================================
TRACK_TO_GP = {
    # 2024 calendar
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
    'Q': 'Q',
    'R': 'R',
    'FP1': 'FP1',
    'FP2': 'FP2',
    'FP3': 'FP3',
    'S': 'S',
    'SQ': 'SQ',
    'SS': 'SS',
}


def resolve_gp_name(track_hint: str, year: int) -> str:
    """
    Resolve track hint to official GP name.

    Args:
        track_hint: e.g. 'suzuka', 'japan', 'Japanese Grand Prix'
        year: e.g. 2024

    Returns:
        Official GP name for FastF1
    """
    if track_hint is None:
        raise ValueError(
            "Track/GP name is required. Use --gp 'Japanese Grand Prix' "
            "or --track suzuka"
        )

    hint_lower = track_hint.lower().strip().replace(' ', '_')

    # Direct match in our mapping
    if hint_lower in TRACK_TO_GP:
        gp_name = TRACK_TO_GP[hint_lower]
        print(f"  Track '{track_hint}' → {gp_name}")
        return gp_name

    # Already a full GP name? Pass through
    if 'grand prix' in track_hint.lower():
        return track_hint

    # Try partial match
    for key, gp in TRACK_TO_GP.items():
        if hint_lower in key or key in hint_lower:
            print(f"  Track '{track_hint}' → {gp} (partial match)")
            return gp

    raise ValueError(
        f"Cannot resolve track '{track_hint}' to a GP name.\n"
        f"Available: {sorted(set(TRACK_TO_GP.values()))}"
    )


def load_real_telemetry(driver: str, year: int, session: str = 'Q',
                        track: str = None, gp: str = None
                        ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load real F1 telemetry from FastF1.

    Args:
        driver: e.g. 'VER', 'HAM'
        year: e.g. 2024
        session: 'Q', 'R', 'FP1', etc.
        track: Track short name e.g. 'suzuka'
        gp: Full GP name e.g. 'Japanese Grand Prix'

    Returns:
        (resampled_data, metadata)
    """
    # Resolve GP name
    if gp:
        gp_name = gp
        if 'grand prix' not in gp.lower():
            gp_name = resolve_gp_name(gp, year)
    elif track:
        gp_name = resolve_gp_name(track, year)
    else:
        raise ValueError(
            "Must provide --track or --gp.\n"
            "Example: --track suzuka  or  --gp 'Japanese Grand Prix'"
        )

    session_type = SESSION_MAP.get(session.upper(), session)

    print(f"  Loading: {year} {gp_name} - {session_type}")
    print(f"  Driver: {driver}")

    # Load session
    f1_session = fastf1.get_session(year, gp_name, session_type)
    f1_session.load()

    # Get driver's fastest lap
    driver_laps = f1_session.laps.pick_drivers(driver)

    if driver_laps.empty:
        available = f1_session.laps['Driver'].unique().tolist()
        raise ValueError(
            f"Driver '{driver}' not found in {year} {gp_name} {session_type}.\n"
            f"Available: {available}"
        )

    fastest = driver_laps.pick_fastest()

    if fastest is None or pd.isna(fastest['LapTime']):
        # Fallback: pick quickest valid lap manually
        valid = driver_laps.dropna(subset=['LapTime'])
        if valid.empty:
            raise ValueError(f"No valid laps for {driver}")
        fastest = valid.loc[valid['LapTime'].idxmin()]

    lap_time = fastest['LapTime'].total_seconds()
    print(f"  Lap time: {int(lap_time//60)}:{lap_time%60:06.3f}")

    # Get telemetry
    tel = fastest.get_telemetry()

    if tel is None or tel.empty:
        raise ValueError(f"No telemetry data for {driver}'s fastest lap")

    print(f"  Telemetry points: {len(tel)}")

    # Build standardized DataFrame
    data = pd.DataFrame()

    # Distance
    if 'Distance' in tel.columns:
        data['lap_distance'] = tel['Distance'].values
    else:
        raise ValueError("No Distance column in FastF1 telemetry")

    track_length = float(data['lap_distance'].max())
    print(f"  Track length: {track_length:.0f}m")

    # Speed
    if 'Speed' in tel.columns:
        data['speed_kmh'] = tel['Speed'].values.astype(float)
    else:
        raise ValueError("No Speed column in FastF1 telemetry")

    # Throttle (0-100 → 0-1)
    if 'Throttle' in tel.columns:
        throttle_raw = tel['Throttle'].values.astype(float)
        if throttle_raw.max() > 1.5:
            data['throttle'] = throttle_raw / 100.0
        else:
            data['throttle'] = throttle_raw

    # Brake
    if 'Brake' in tel.columns:
        data['brake'] = tel['Brake'].values.astype(float)

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

    for col in data.columns:
        if col == 'lap_distance':
            continue
        vals = data[col].values[sort_idx]
        resampled[col] = np.interp(distances, src_sorted, vals)

    meta = {
        'driver': driver,
        'year': year,
        'session': session_type,
        'gp_name': gp_name,
        'lap_time': lap_time,
        'track_length': track_length,
        'telemetry_points': len(tel),
    }

    return resampled, meta