#!/usr/bin/env python3
"""
Step 1: Build multi-lap reference dataset from FastF1 2025 Suzuka Qualifying.

Loads all valid flying laps, filters by quality criteria, resamples to
uniform 2m distance axis, and outputs parquet + metadata + overview plot.

Output consumed by subsequent corner segmentation / phase detection scripts.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fastf1

from src.fastf1_loader import (
    CACHE_DIR, _normalize_brake, _normalize_throttle, resolve_gp_name
)
from src.plotting import COLORS, style_axis, save_figure
from config import OUTPUT_DIR

# ============================================================
# Configuration
# ============================================================
YEAR = 2025
GP_NAME = "Japanese Grand Prix"
SESSION_TYPE = "Q"
RESAMPLE_STEP_M = 2
LAP_TIME_TOLERANCE_PCT = 2.0  # within 2% of fastest
MIN_TELEMETRY_COVERAGE = 0.95  # 95% of track length
MAX_TELEMETRY_GAP_S = 2.0  # FastF1 car data has inherent ~0.8-1.2s max gaps at ~3.7Hz
MAX_XY_JUMP_M = 50.0
MIN_LAPS_ERROR = 3
MIN_LAPS_WARNING = 5


# ============================================================
# Step 2: Load session
# ============================================================
def load_session():
    """Load 2025 Suzuka Qualifying session via FastF1."""
    print(f"  Loading: {YEAR} {GP_NAME} - {SESSION_TYPE}")
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    session = fastf1.get_session(YEAR, GP_NAME, SESSION_TYPE)
    session.load()

    track_length = session.session_info.get('TrackLength', None)
    if track_length is None:
        # Estimate from fastest lap telemetry
        fastest = session.laps.pick_fastest()
        tel = fastest.get_telemetry()
        track_length = float(tel['Distance'].max()) if tel is not None else 5807.0

    print(f"  Track length: {track_length:.0f}m")
    print(f"  Total laps in session: {len(session.laps)}")
    return session, track_length


# ============================================================
# Step 3: Filter laps
# ============================================================
def filter_laps(session, track_length):
    """
    Filter laps by quality criteria. Returns list of (lap, status, reason) tuples.
    """
    all_laps = session.laps
    fastest_time = all_laps.pick_fastest()['LapTime'].total_seconds()
    max_allowed_time = fastest_time * (1 + LAP_TIME_TOLERANCE_PCT / 100)

    print(f"  Session fastest: {fastest_time:.3f}s")
    print(f"  Max allowed: {max_allowed_time:.3f}s ({LAP_TIME_TOLERANCE_PCT}% tolerance)")

    results = []

    for idx, lap in all_laps.iterrows():
        driver = lap['Driver']
        lap_num = int(lap['LapNumber'])
        lap_id = f"2025_Q_{driver}_{lap_num}"

        # Check: timed flying lap
        if pd.isna(lap['LapTime']):
            results.append((lap, "excluded", "no lap time (outlap/inlap)", lap_id))
            continue

        lap_time_s = lap['LapTime'].total_seconds()

        # Check: not pit lap (PitOutTime = this lap started from pit,
        # PitInTime = this lap ended in pit)
        try:
            if pd.notna(lap.get('PitOutTime')) or pd.notna(lap.get('PitInTime')):
                results.append((lap, "excluded", "pit lap", lap_id))
                continue
        except (TypeError, ValueError):
            pass

        # Check: lap time within tolerance
        pct_above = ((lap_time_s - fastest_time) / fastest_time) * 100
        if pct_above > LAP_TIME_TOLERANCE_PCT:
            results.append((lap, "excluded",
                           f"too slow ({pct_above:.2f}% above best)", lap_id))
            continue

        # Check: no track status issues (yellow/red flag)
        # FastF1 marks deleted laps; also check IsAccurate
        is_accurate = lap.get('IsAccurate', True)
        if is_accurate is False:
            results.append((lap, "excluded", "marked inaccurate by FastF1", lap_id))
            continue

        # Get telemetry for quality checks
        try:
            tel = lap.get_telemetry()
        except Exception as e:
            results.append((lap, "excluded", f"telemetry load failed: {e}", lap_id))
            continue

        if tel is None or tel.empty or len(tel) < 20:
            results.append((lap, "excluded", "insufficient telemetry points", lap_id))
            continue

        # Check: telemetry coverage
        tel_max_dist = float(tel['Distance'].max())
        coverage = tel_max_dist / track_length
        if coverage < MIN_TELEMETRY_COVERAGE:
            results.append((lap, "excluded",
                           f"low coverage ({coverage:.1%})", lap_id))
            continue

        # Check: no telemetry gaps > 0.5s
        if 'Time' in tel.columns:
            time_vals = tel['Time'].dt.total_seconds().values
            time_diffs = np.diff(time_vals)
            max_gap = float(time_diffs.max()) if len(time_diffs) > 0 else 0
            if max_gap > MAX_TELEMETRY_GAP_S:
                results.append((lap, "excluded",
                               f"telemetry gap ({max_gap:.2f}s)", lap_id))
                continue

        # Note: XY discontinuity check removed. At FastF1's ~7.6Hz sampling
        # and F1 speeds (300 km/h = 83 m/s), consecutive XY points are naturally
        # ~11m apart, and interpolation artifacts push many above 50m.
        # Coverage + time gap checks are sufficient quality gates.

        # Passed all checks
        results.append((lap, "included", None, lap_id))

    included = [(lap, s, r, lid) for lap, s, r, lid in results if s == "included"]
    excluded = [(lap, s, r, lid) for lap, s, r, lid in results if s == "excluded"]

    print(f"\n  Filtering results:")
    print(f"    Included: {len(included)}")
    print(f"    Excluded: {len(excluded)}")

    if len(included) < MIN_LAPS_ERROR:
        raise RuntimeError(
            f"Only {len(included)} laps passed filtering (minimum {MIN_LAPS_ERROR}). "
            f"Cannot build reference dataset."
        )
    if len(included) < MIN_LAPS_WARNING:
        print(f"  ⚠ WARNING: Only {len(included)} laps (< {MIN_LAPS_WARNING})")

    return results, fastest_time


# ============================================================
# Step 4: Resample
# ============================================================
def resample_lap(lap, track_length):
    """
    Resample a single lap's telemetry to uniform 2m distance axis.
    Returns DataFrame with standardized columns or None on failure.
    """
    try:
        tel = lap.get_telemetry()
    except Exception:
        return None

    if tel is None or tel.empty:
        return None

    src_dist = tel['Distance'].values.astype(float)
    sort_idx = np.argsort(src_dist)
    src_sorted = src_dist[sort_idx]

    # Remove duplicates
    unique_mask = np.concatenate([[True], np.diff(src_sorted) > 0.01])
    src_unique = src_sorted[unique_mask]

    if len(src_unique) < 20:
        return None

    # Target distance axis
    distances = np.arange(0, track_length, RESAMPLE_STEP_M)

    resampled = pd.DataFrame({'distance_m': distances})

    # Speed
    speed_vals = tel['Speed'].values.astype(float)[sort_idx][unique_mask]
    resampled['speed'] = np.interp(distances, src_unique, speed_vals)

    # Throttle
    if 'Throttle' in tel.columns:
        throttle_raw = tel['Throttle'].values.astype(float)[sort_idx][unique_mask]
        resampled['throttle'] = np.interp(distances, src_unique,
                                          _normalize_throttle(throttle_raw))
    else:
        resampled['throttle'] = np.nan

    # Brake
    if 'Brake' in tel.columns:
        brake_raw = tel['Brake'].values[sort_idx][unique_mask]
        brake_norm, _ = _normalize_brake(brake_raw)
        resampled['brake'] = np.interp(distances, src_unique, brake_norm)
    else:
        resampled['brake'] = np.nan

    # X, Y
    if 'X' in tel.columns and 'Y' in tel.columns:
        x_vals = tel['X'].values.astype(float)[sort_idx][unique_mask]
        y_vals = tel['Y'].values.astype(float)[sort_idx][unique_mask]
        resampled['x'] = np.interp(distances, src_unique, x_vals)
        resampled['y'] = np.interp(distances, src_unique, y_vals)
    else:
        resampled['x'] = np.nan
        resampled['y'] = np.nan

    return resampled


def build_parquet(results, track_length, fastest_time):
    """
    Resample all included laps and build combined parquet DataFrame.
    Returns (combined_df, metadata_laps_list).
    """
    included = [(lap, s, r, lid) for lap, s, r, lid in results if s == "included"]

    all_frames = []
    meta_laps = []

    for lap, status, reason, lap_id in results:
        driver = lap['Driver']
        lap_num = int(lap['LapNumber'])
        lap_time_s = (lap['LapTime'].total_seconds()
                      if pd.notna(lap['LapTime']) else None)
        pct_above = (((lap_time_s - fastest_time) / fastest_time) * 100
                     if lap_time_s else None)

        tel_points = 0
        if status == "included":
            try:
                tel = lap.get_telemetry()
                tel_points = len(tel) if tel is not None else 0
            except Exception:
                pass

        meta_laps.append({
            "lap_id": lap_id,
            "driver": driver,
            "lap_number": lap_num,
            "lap_time_s": round(lap_time_s, 3) if lap_time_s else None,
            "pct_above_best": round(pct_above, 3) if pct_above else None,
            "telemetry_points": tel_points,
            "status": status,
            "exclusion_reason": reason,
        })

        if status != "included":
            continue

        resampled = resample_lap(lap, track_length)
        if resampled is None:
            continue

        resampled['lap_id'] = lap_id
        resampled['driver'] = driver
        resampled['lap_number'] = lap_num
        all_frames.append(resampled)

    if not all_frames:
        raise RuntimeError("No laps could be resampled successfully.")

    combined = pd.concat(all_frames, ignore_index=True)

    # Reorder columns
    col_order = ['lap_id', 'driver', 'lap_number', 'distance_m',
                 'speed', 'throttle', 'brake', 'x', 'y']
    combined = combined[col_order]

    return combined, meta_laps


# ============================================================
# Step 5C: Overview plot
# ============================================================
def plot_overview(combined_df, output_path):
    """Plot all included laps overlaid with median speed trace."""
    lap_ids = combined_df['lap_id'].unique()
    n_laps = len(lap_ids)

    fig, ax = plt.subplots(1, 1, figsize=(28, 8))
    fig.set_facecolor(COLORS['bg'])
    style_axis(ax, ylabel='Speed (km/h)', xlabel='Distance (m)',
               title=f'Suzuka 2025 Q — Reference Lap Pool (N={n_laps})')

    # Individual laps (thin, semi-transparent)
    for lap_id in lap_ids:
        lap_data = combined_df[combined_df['lap_id'] == lap_id]
        driver = lap_data['driver'].iloc[0]
        ax.plot(lap_data['distance_m'], lap_data['speed'],
                lw=0.8, alpha=0.4, label=f"{driver}")

    # Median speed trace (bold)
    pivot = combined_df.pivot_table(
        index='distance_m', values='speed', aggfunc='median'
    )
    ax.plot(pivot.index, pivot['speed'], color='white', lw=2.5,
            alpha=0.9, label='Median', zorder=10)

    # Legend (deduplicate)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc='upper right', facecolor=COLORS['card_bg'],
              edgecolor='white', labelcolor='white', fontsize=8,
              ncol=2)

    plt.tight_layout()
    save_figure(fig, output_path, dpi=200)
    print(f"  ✓ Overview plot saved: {output_path}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  Step 1: Build Reference Lap Pool")
    print("  2025 Suzuka Qualifying — FastF1")
    print("=" * 60)

    # Step 2: Load session
    print("\n  [Step 2] Loading session...")
    session, track_length = load_session()

    # Step 3: Filter laps
    print("\n  [Step 3] Filtering laps...")
    results, fastest_time = filter_laps(session, track_length)

    # Step 4 + 5A: Resample and build parquet
    print("\n  [Step 4] Resampling included laps...")
    combined_df, meta_laps = build_parquet(results, track_length, fastest_time)

    included_laps = [m for m in meta_laps if m['status'] == 'included']
    print(f"  Resampled {len(included_laps)} laps to {RESAMPLE_STEP_M}m steps")

    # 5A: Save parquet
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = OUTPUT_DIR / "suzuka_reference_laps.parquet"
    combined_df.to_parquet(parquet_path, index=False)
    print(f"  ✓ Parquet saved: {parquet_path} ({len(combined_df)} rows)")

    # 5B: Save metadata JSON
    metadata = {
        "session": f"{YEAR} {GP_NAME} - Qualifying",
        "track_length_m": float(track_length),
        "resample_step_m": RESAMPLE_STEP_M,
        "total_laps": len(included_laps),
        "laps": meta_laps,
    }
    meta_path = OUTPUT_DIR / "suzuka_reference_laps_metadata.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Metadata saved: {meta_path}")

    # 5C: Overview plot
    print("\n  [Step 5C] Generating overview plot...")
    plot_path = OUTPUT_DIR / "suzuka_reference_laps_overview.png"
    plot_overview(combined_df, plot_path)

    # Final report
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    excluded_count = sum(1 for m in meta_laps if m['status'] == 'excluded')
    print(f"  Included: {len(included_laps)}")
    print(f"  Excluded: {excluded_count}")
    print(f"\n  Included laps:")
    for m in included_laps:
        if m['lap_time_s'] is not None and m['pct_above_best'] is not None:
            print(f"    {m['lap_id']:>20}  {m['lap_time_s']:.3f}s  "
                  f"(+{m['pct_above_best']:.2f}%)")
        else:
            print(f"    {m['lap_id']:>20}  (no time data)")

    # Verify parquet is readable
    verify_df = pd.read_parquet(parquet_path)
    assert len(verify_df) == len(combined_df), "Parquet verification failed"
    print(f"\n  ✓ Parquet verified: {len(verify_df)} rows, "
          f"{len(verify_df.columns)} columns")
    print(f"    Columns: {list(verify_df.columns)}")


if __name__ == "__main__":
    main()
