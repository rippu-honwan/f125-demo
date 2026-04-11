"""
F1 Lap Insight - Step 3: Real vs Game Lap Comparison (v5)
Compare your F1 25 telemetry against real F1 driver (FastF1).

v5: Two-pass alignment + local cross-correlation for sparse sections
    18 individual corners + English coaching

Usage:
    python scripts/03_lap_comparison.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.gridspec as gridspec
import json
import fastf1
from scipy.signal import argrelextrema
from scipy.ndimage import uniform_filter1d

from config import DEFAULT_CSV, OUTPUT_DIR, BG_COLOR, DPI_SAVE
from src.utils import (
    load_telemetry, resample, smooth,
    format_laptime, format_delta,
    find_braking_points, find_throttle_points
)

# ============================================================
# Settings
# ============================================================

FASTF1_CACHE = Path("data/fastf1_cache")
FASTF1_CACHE.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(FASTF1_CACHE))

F1_YEAR = 2025
F1_RACE = "Japanese Grand Prix"
F1_SESSION_TYPE = "Q"
F1_DRIVER = "VER"

# ============================================================
# All 18 Suzuka Corners
# ============================================================

SUZUKA_CORNERS = [
    (1,  "Turn 1",           "T1",   612,   700,   810),
    (2,  "Turn 2",           "T2",   810,   880,   976),
    (3,  "S Curve 1",        "S1",  1032,  1100,  1170),
    (4,  "S Curve 2",        "S2",  1170,  1240,  1304),
    (5,  "S Curve 3",        "S3",  1322,  1375,  1453),
    (6,  "S Curve 4",        "S4",  1486,  1575,  1664),
    (7,  "Dunlop",           "DUN",  1687,  1787,  1870),
    (8,  "Degner 1",         "DG1",  2268,  2304,  2355),
    (9,  "Degner 2",         "DG2",  2425,  2467,  2514),
    (10, "200R",             "200R", 2729,  2793,  2830),
    (11, "Hairpin",          "HAIR", 2850,  2938,  3020),
    (12, "Spoon Entry",      "SP-E", 3424,  3550,  3680),
    (13, "Spoon Exit",       "SP-X", 3680,  3820,  3964),
    (14, "Backstrtch Kink",  "KINK", 4450,  4500,  4560),
    (15, "130R",             "130R", 4920,  4992,  5100),
    (16, "Casio Triangle 1", "CS1",  5363,  5420,  5470),
    (17, "Casio Triangle 2", "CS2",  5470,  5520,  5550),
    (18, "Final Corner",     "FIN",  5550,  5600,  5741),
]

CORNER_COLORS = [
    "#ff4444", "#ff6633", "#ff9500", "#ffb300",
    "#ffd500", "#d4ed26", "#66bb6a", "#26a69a",
    "#29b6f6", "#42a5f5", "#5c6bc0", "#7e57c2",
    "#ab47bc", "#ec407a", "#ef5350", "#ff7043",
    "#8d6e63", "#78909c"
]


# ============================================================
# Load Real F1 Data
# ============================================================

def load_real_f1_lap():
    """Load fastest qualifying lap from FastF1."""
    print(f"\n  Loading real F1 data...")
    print(f"  {F1_YEAR} {F1_RACE} - Qualifying")
    print(f"  Driver: {F1_DRIVER}")

    session = fastf1.get_session(F1_YEAR, F1_RACE, F1_SESSION_TYPE)
    session.load()

    driver_laps = session.laps.pick_drivers(F1_DRIVER)
    fastest = driver_laps.pick_fastest()

    lap_time_sec = fastest['LapTime'].total_seconds()
    print(f"  Lap time: {format_laptime(lap_time_sec)}")

    tel = fastest.get_telemetry()
    print(f"  Telemetry points: {len(tel)}")

    real_data = pd.DataFrame()
    real_data['distance'] = tel['Distance'].values
    real_data['speed_kmh'] = tel['Speed'].values

    if 'Throttle' in tel.columns:
        real_data['throttle'] = tel['Throttle'].values / 100.0
    if 'Brake' in tel.columns:
        real_data['brake'] = tel['Brake'].astype(float).values
    if 'nGear' in tel.columns:
        real_data['gear'] = tel['nGear'].values
    if 'X' in tel.columns and 'Y' in tel.columns:
        real_data['world_x'] = tel['X'].values
        real_data['world_y'] = tel['Y'].values

    track_length = float(real_data['distance'].max())
    print(f"  Track length: {track_length:.0f}m")

    distances = np.arange(0, track_length, 1.0)
    resampled = pd.DataFrame({'lap_distance': distances})

    for col in ['speed_kmh', 'throttle', 'brake', 'gear']:
        if col in real_data.columns:
            resampled[col] = np.interp(
                distances, real_data['distance'].values, real_data[col].values
            )

    if 'world_x' in real_data.columns:
        resampled['world_position_X'] = np.interp(
            distances, real_data['distance'].values, real_data['world_x'].values
        )
        resampled['world_position_Y'] = np.interp(
            distances, real_data['distance'].values, real_data['world_y'].values
        )

    meta = {
        'driver': F1_DRIVER,
        'year': F1_YEAR,
        'race': F1_RACE,
        'session': 'Qualifying',
        'lap_time': lap_time_sec,
        'track_length': track_length,
        'driver_full': str(fastest['Driver']),
        'team': str(fastest['Team']),
    }

    return resampled, meta


# ============================================================
# Two-Pass Alignment System
# ============================================================

def find_multi_feature_anchors(speed, throttle=None, brake=None,
                               track_length=None, n_speed=15,
                               order_override=None):
    """
    Find anchor points using multiple signal features.
    order_override: smaller = finds more local features (for dense sections)
    """
    anchors = []
    speed_s = smooth(speed, 30)

    order = order_override if order_override else max(len(speed) // 40, 30)

    # Speed minima
    min_idx = argrelextrema(speed_s, np.less, order=order)[0]
    if len(min_idx) > 0:
        speeds_at_min = speed_s[min_idx]
        sorted_i = np.argsort(speeds_at_min)[:n_speed]
        for i in sorted_i:
            anchors.append((int(min_idx[i]), 'speed_min', float(speeds_at_min[i])))

    # Speed maxima
    max_idx = argrelextrema(speed_s, np.greater, order=order)[0]
    if len(max_idx) > 0:
        speeds_at_max = speed_s[max_idx]
        sorted_i = np.argsort(-speeds_at_max)[:10]
        for i in sorted_i:
            anchors.append((int(max_idx[i]), 'speed_max', float(speeds_at_max[i])))

    # Braking initiation
    if brake is not None:
        bp = find_braking_points(speed, threshold_decel=3.0, min_gap=80)
        for idx in bp:
            anchors.append((int(idx), 'brake_start', float(speed_s[idx])))

    # Full throttle start
    if throttle is not None:
        tp = find_throttle_points(throttle, threshold=0.8, min_gap=80)
        for idx in tp:
            anchors.append((int(idx), 'throttle_full', float(speed_s[idx])))

    anchors.sort(key=lambda a: a[0])
    return anchors


def match_multi_anchors(game_anchors, game_length,
                        real_anchors, real_length,
                        pos_tolerance=0.03, score_threshold=0.10):
    """Match anchor points between game and real data."""
    pairs = [(0.0, 0.0)]
    used_real = set()

    game_by_type = {}
    real_by_type = {}

    for idx, ftype, val in game_anchors:
        game_by_type.setdefault(ftype, []).append((idx / game_length, val))
    for idx, ftype, val in real_anchors:
        real_by_type.setdefault(ftype, []).append((idx / real_length, val))

    for ftype in ['speed_min', 'speed_max', 'brake_start', 'throttle_full']:
        g_list = game_by_type.get(ftype, [])
        r_list = real_by_type.get(ftype, [])

        for gn, gv in g_list:
            best_match = None
            best_score = float('inf')

            for ri, (rn, rv) in enumerate(r_list):
                if ri in used_real:
                    continue

                pos_diff = abs(gn - rn)
                if pos_diff > pos_tolerance:
                    continue

                if ftype == 'speed_min':
                    spd_diff = abs(gv - rv) / max(gv, rv, 1)
                    if spd_diff > 0.35:
                        continue
                    score = pos_diff * 1.5 + spd_diff * 2.0
                else:
                    spd_diff = abs(gv - rv) / 300.0
                    score = pos_diff * 2.0 + spd_diff * 1.0

                if score < best_score:
                    best_score = score
                    best_match = (ri, rn)

            if best_match is not None and best_score < score_threshold:
                ri, rn = best_match
                pairs.append((gn, rn))
                used_real.add(ri)

    pairs.append((1.0, 1.0))

    pairs.sort(key=lambda p: p[0])
    cleaned = [pairs[0]]
    for p in pairs[1:]:
        if p[0] > cleaned[-1][0] + 0.005 and p[1] > cleaned[-1][1] + 0.005:
            cleaned.append(p)

    return cleaned


def local_cross_correlation(game_speed, real_speed,
                            game_start, game_end,
                            real_center, search_range=100,
                            window=200):
    """
    Find best local alignment using cross-correlation.

    Like tuning a radio: slide the real signal left/right
    until it best matches the game signal.

    Returns: optimal shift in meters
    """
    g_start = max(0, int(game_start))
    g_end = min(len(game_speed), int(game_end))
    game_segment = game_speed[g_start:g_end]

    if len(game_segment) < 50:
        return 0

    # Normalize
    game_norm = (game_segment - np.mean(game_segment))
    g_std = np.std(game_norm)
    if g_std < 1.0:
        return 0
    game_norm = game_norm / g_std

    best_corr = -1
    best_shift = 0

    for shift in range(-search_range, search_range + 1, 2):
        r_start = max(0, int(real_center - len(game_segment) // 2 + shift))
        r_end = r_start + len(game_segment)

        if r_end > len(real_speed):
            continue

        real_segment = real_speed[r_start:r_end]
        if len(real_segment) != len(game_segment):
            continue

        real_norm = (real_segment - np.mean(real_segment))
        r_std = np.std(real_norm)
        if r_std < 1.0:
            continue
        real_norm = real_norm / r_std

        corr = float(np.mean(game_norm * real_norm))

        if corr > best_corr:
            best_corr = corr
            best_shift = shift

    return best_shift, best_corr


def find_anchor_gaps(anchor_pairs, game_length, min_gap=400):
    """
    Find sections where anchors are too sparse.
    Returns list of (start_m, end_m) gaps.
    """
    gaps = []
    anchor_distances = sorted([a[0] * game_length for a in anchor_pairs])

    for i in range(len(anchor_distances) - 1):
        gap = anchor_distances[i + 1] - anchor_distances[i]
        if gap > min_gap:
            gaps.append((anchor_distances[i], anchor_distances[i + 1], gap))

    return gaps


def align_two_pass(game_data, game_length, real_data, real_length):
    """
    Two-pass alignment:
    Pass 1: Strict global feature matching
    Pass 2: Local cross-correlation to fill gaps
    """
    print(f"\n  --- Two-Pass Alignment ---")
    print(f"  Game: {game_length:.0f}m  |  Real: {real_length:.0f}m")

    game_speed = game_data['speed_kmh'].values
    real_speed = real_data['speed_kmh'].values
    game_throttle = game_data['throttle'].values if 'throttle' in game_data else None
    game_brake = game_data['brake'].values if 'brake' in game_data else None
    real_throttle = real_data['throttle'].values if 'throttle' in real_data else None
    real_brake = real_data['brake'].values if 'brake' in real_data else None

    # ---- PASS 1: Global strict matching ----
    print(f"\n  PASS 1: Global feature matching (strict)")

    game_anchors = find_multi_feature_anchors(
        game_speed, game_throttle, game_brake, game_length
    )
    real_anchors = find_multi_feature_anchors(
        real_speed, real_throttle, real_brake, real_length
    )

    print(f"  Game features: {len(game_anchors)}")
    print(f"  Real features: {len(real_anchors)}")

    anchor_pairs = match_multi_anchors(
        game_anchors, game_length,
        real_anchors, real_length,
        pos_tolerance=0.03, score_threshold=0.10
    )

    # Drift filter
    max_drift = 80
    filtered = []
    for g, r in anchor_pairs:
        drift = abs(g * game_length - r * real_length)
        if drift < max_drift or g == 0.0 or g == 1.0:
            filtered.append((g, r))
        else:
            print(f"  REMOVED: game {g*game_length:.0f}m <-> "
                  f"real {r*real_length:.0f}m (drift {drift:.0f}m)")
    anchor_pairs = filtered

    print(f"  Pass 1 anchors: {len(anchor_pairs)}")
    for i, (gn, rn) in enumerate(anchor_pairs):
        gd = gn * game_length
        rd = rn * real_length
        print(f"    {i:2d}: game {gd:6.0f}m <-> real {rd:6.0f}m "
              f"(drift {abs(gd-rd):.0f}m)")

    # ---- PASS 2: Fill gaps with local cross-correlation ----
    print(f"\n  PASS 2: Local cross-correlation for sparse sections")

    gaps = find_anchor_gaps(anchor_pairs, game_length, min_gap=350)
    print(f"  Found {len(gaps)} gaps:")
    for start, end, gap_size in gaps:
        print(f"    {start:.0f}m - {end:.0f}m ({gap_size:.0f}m gap)")

    # For each gap, find local anchors with relaxed settings
    new_anchors = []

    for gap_start, gap_end, gap_size in gaps:
        # Find features with smaller order (more sensitive)
        g_start_idx = int(gap_start)
        g_end_idx = min(int(gap_end), len(game_speed))

        if g_end_idx - g_start_idx < 100:
            continue

        # Extract segment
        game_seg = game_speed[g_start_idx:g_end_idx]
        game_seg_s = smooth(game_seg, 15)

        # Find local minima with smaller order
        local_order = max(len(game_seg) // 8, 15)
        local_min = argrelextrema(game_seg_s, np.less, order=local_order)[0]

        print(f"    Gap {gap_start:.0f}-{gap_end:.0f}m: "
              f"found {len(local_min)} local minima")

        for lm in local_min:
            game_dist = gap_start + lm
            game_spd = float(game_seg_s[lm])

            # Estimate where this should be in real data
            # Use linear interpolation from surrounding anchors
            game_norm = game_dist / game_length
            real_est_norm = np.interp(
                game_norm,
                [a[0] for a in anchor_pairs],
                [a[1] for a in anchor_pairs]
            )
            real_est_dist = real_est_norm * real_length

            # Cross-correlation to refine
            result = local_cross_correlation(
                game_speed, real_speed,
                game_dist - 80, game_dist + 80,
                real_est_dist,
                search_range=60,
                window=160
            )

            if result is None:
                continue

            shift, corr = result

            if corr < 0.5:
                print(f"      Skip dist={game_dist:.0f}m: low correlation ({corr:.2f})")
                continue

            real_refined = real_est_dist + shift

            # Verify: speed at refined point should be similar
            real_idx = int(np.clip(real_refined, 0, len(real_speed) - 1))
            real_spd = float(smooth(real_speed, 15)[real_idx])
            spd_diff = abs(game_spd - real_spd) / max(game_spd, 1)

            if spd_diff > 0.4:
                print(f"      Skip dist={game_dist:.0f}m: speed mismatch "
                      f"({game_spd:.0f} vs {real_spd:.0f})")
                continue

            drift = abs(game_dist - real_refined)
            if drift > 100:
                print(f"      Skip dist={game_dist:.0f}m: excessive drift ({drift:.0f}m)")
                continue

            new_pair = (game_dist / game_length, real_refined / real_length)
            new_anchors.append(new_pair)
            print(f"      Added: game {game_dist:.0f}m <-> real {real_refined:.0f}m "
                  f"(corr={corr:.2f}, drift={drift:.0f}m)")

    # Merge pass 1 and pass 2 anchors
    all_pairs = anchor_pairs + new_anchors
    all_pairs.sort(key=lambda p: p[0])

    # Remove duplicates / conflicts (keep monotonically increasing)
    cleaned = [all_pairs[0]]
    for p in all_pairs[1:]:
        if p[0] > cleaned[-1][0] + 0.003 and p[1] > cleaned[-1][1] + 0.003:
            cleaned.append(p)

    # Final drift check
    final_pairs = []
    for g, r in cleaned:
        drift = abs(g * game_length - r * real_length)
        if drift < 100 or g == 0.0 or g == 1.0:
            final_pairs.append((g, r))

    print(f"\n  Final anchors: {len(final_pairs)} "
          f"(Pass1: {len(anchor_pairs)}, Pass2: +{len(new_anchors)})")

    for i, (gn, rn) in enumerate(final_pairs):
        gd = gn * game_length
        rd = rn * real_length
        source = "P1" if (gn, rn) in anchor_pairs else "P2"
        print(f"    {i:2d} [{source}]: game {gd:6.0f}m <-> real {rd:6.0f}m "
              f"(drift {abs(gd-rd):.0f}m)")

    # ---- Build aligned data ----
    game_anch_d = np.array([a[0] for a in final_pairs]) * game_length
    real_anch_d = np.array([a[1] for a in final_pairs]) * real_length

    distances = np.arange(0, game_length, 1.0)
    real_mapped = np.interp(distances, game_anch_d, real_anch_d)

    aligned = pd.DataFrame({'lap_distance': distances})

    game_norm = game_data['lap_distance'].values
    for col in ['speed_kmh', 'throttle', 'brake', 'steering', 'gear']:
        if col in game_data.columns:
            aligned[f'game_{col}'] = np.interp(
                distances, game_norm, game_data[col].values
            )

    real_norm = real_data['lap_distance'].values
    for col in ['speed_kmh', 'throttle', 'brake', 'gear']:
        if col in real_data.columns:
            aligned[f'real_{col}'] = np.interp(
                real_mapped, real_norm, real_data[col].values
            )

    aligned['speed_delta'] = aligned['game_speed_kmh'] - aligned['real_speed_kmh']

    if 'world_position_X' in game_data.columns:
        aligned['world_x'] = np.interp(
            distances, game_norm, game_data['world_position_X'].values
        )
        aligned['world_y'] = np.interp(
            distances, game_norm, game_data['world_position_Y'].values
        )

    aligned.attrs['anchor_pairs'] = final_pairs
    aligned.attrs['game_anch_d'] = game_anch_d
    aligned.attrs['real_anch_d'] = real_anch_d
    aligned.attrs['pass1_count'] = len(anchor_pairs)
    aligned.attrs['pass2_count'] = len(new_anchors)

    print(f"\n  Aligned: {len(aligned)} points")
    print(f"  Avg speed delta: {aligned['speed_delta'].mean():+.1f} km/h")

    return aligned


# ============================================================
# Time Delta
# ============================================================

def calculate_time_delta(aligned):
    """Cumulative time delta."""
    game_ms = np.maximum(aligned['game_speed_kmh'].values / 3.6, 1.0)
    real_ms = np.maximum(aligned['real_speed_kmh'].values / 3.6, 1.0)

    delta_per_m = (1.0 / game_ms) - (1.0 / real_ms)
    aligned['time_delta'] = np.cumsum(delta_per_m)

    print(f"  Calculated time delta: {aligned['time_delta'].iloc[-1]:+.3f}s")
    return aligned


# ============================================================
# Corner Analysis with Confidence (18 corners)
# ============================================================

def calculate_confidence(corner_entry, corner_exit, anchor_distances):
    """Calculate alignment confidence for a corner."""
    # Check if anchor is inside corner zone
    for ad in anchor_distances:
        if corner_entry <= ad <= corner_exit:
            return "HIGH"
    # Check if anchor is nearby (within 100m)
    for ad in anchor_distances:
        if (corner_entry - 100) <= ad <= (corner_exit + 100):
            return "MEDIUM"
    return "LOW"


def analyze_corners(aligned):
    """Analyze all 18 individual corners with confidence."""
    dist = aligned['lap_distance'].values
    game_anch_d = aligned.attrs.get('game_anch_d', np.array([]))

    results = []

    for cid, name, short, entry, apex, exit_ in SUZUKA_CORNERS:
        mask = (dist >= entry) & (dist <= exit_)
        zone = aligned[mask]
        if len(zone) == 0:
            continue

        game_min = float(zone['game_speed_kmh'].min())
        real_min = float(zone['real_speed_kmh'].min())

        game_min_idx = int(zone['game_speed_kmh'].idxmin())
        game_min_dist = float(aligned['lap_distance'].iloc[game_min_idx])

        apex_idx = int(np.argmin(np.abs(dist - apex)))
        game_apex = float(aligned['game_speed_kmh'].iloc[apex_idx])
        real_apex = float(aligned['real_speed_kmh'].iloc[apex_idx])

        entry_idx = int(np.argmin(np.abs(dist - entry)))
        exit_idx = int(np.argmin(np.abs(dist - exit_)))
        t_entry = float(aligned['time_delta'].iloc[entry_idx])
        t_exit = float(aligned['time_delta'].iloc[exit_idx])
        corner_delta = t_exit - t_entry

        # Entry speed (50m before)
        pre = max(0, entry - 50)
        pre_idx = int(np.argmin(np.abs(dist - pre)))
        game_entry_spd = float(aligned['game_speed_kmh'].iloc[pre_idx])
        real_entry_spd = float(aligned['real_speed_kmh'].iloc[pre_idx])

        # Exit speed (50m after)
        post = min(exit_ + 50, dist.max())
        post_idx = int(np.argmin(np.abs(dist - post)))
        game_exit_spd = float(aligned['game_speed_kmh'].iloc[post_idx])
        real_exit_spd = float(aligned['real_speed_kmh'].iloc[post_idx])

        # Brake point detection
        brake_threshold = game_entry_spd - 20
        game_brake_dist = None
        real_brake_dist = None

        for i in range(pre_idx, exit_idx):
            if game_brake_dist is None and aligned['game_speed_kmh'].iloc[i] < brake_threshold:
                game_brake_dist = float(dist[i])
            if 'real_speed_kmh' in aligned.columns:
                real_brake_t = real_entry_spd - 20
                if real_brake_dist is None and aligned['real_speed_kmh'].iloc[i] < real_brake_t:
                    real_brake_dist = float(dist[i])

        confidence = calculate_confidence(entry, exit_, game_anch_d)

        # Sanity: if speed delta is absurd, mark LOW
        spd_delta = game_min - real_min
        if abs(spd_delta) > 60:
            confidence = "LOW"

        results.append({
            'id': cid,
            'name': name,
            'short': short,
            'color': CORNER_COLORS[cid - 1],
            'entry_dist': float(entry),
            'apex_dist': float(apex),
            'exit_dist': float(exit_),
            'game_min_speed': game_min,
            'real_min_speed': real_min,
            'min_speed_delta': spd_delta,
            'game_min_dist': game_min_dist,
            'game_apex_speed': game_apex,
            'real_apex_speed': real_apex,
            'game_entry_speed': game_entry_spd,
            'real_entry_speed': real_entry_spd,
            'game_exit_speed': game_exit_spd,
            'real_exit_speed': real_exit_spd,
            'game_brake_dist': game_brake_dist,
            'real_brake_dist': real_brake_dist,
            'time_delta': corner_delta,
            'confidence': confidence,
        })

    return results


# ============================================================
# Coaching Tips (All English)
# ============================================================

def generate_coaching_tips(corners):
    """Generate actionable coaching tips based on corner data."""
    tips = []

    for c in corners:
        if c['confidence'] == 'LOW':
            continue

        name = c['name']
        td = c['time_delta']
        spd_d = c['min_speed_delta']
        entry_d = c['game_entry_speed'] - c['real_entry_speed']
        exit_d = c['game_exit_speed'] - c['real_exit_speed']

        corner_tips = []

        if td > 0.02:
            if entry_d < -15:
                corner_tips.append({
                    'type': 'BRAKING',
                    'severity': 'HIGH' if entry_d < -30 else 'MEDIUM',
                    'message': (
                        f"Entry speed {abs(entry_d):.0f} km/h slower. "
                        f"Try braking 10-15m later or reduce brake pressure 10%."
                    ),
                })

            if spd_d < -10:
                if c['game_min_speed'] < 120:
                    corner_tips.append({
                        'type': 'CORNER_SPEED',
                        'severity': 'HIGH' if spd_d < -25 else 'MEDIUM',
                        'message': (
                            f"Apex speed {abs(spd_d):.0f} km/h slower. "
                            f"Try wider entry for higher min speed. "
                            f"Pro carries {c['real_min_speed']:.0f} km/h."
                        ),
                    })
                else:
                    corner_tips.append({
                        'type': 'CONFIDENCE',
                        'severity': 'MEDIUM',
                        'message': (
                            f"High-speed corner: {abs(spd_d):.0f} km/h slower. "
                            f"Build up gradually: +5 km/h per attempt."
                        ),
                    })

            if exit_d < -15:
                corner_tips.append({
                    'type': 'EXIT',
                    'severity': 'HIGH' if exit_d < -30 else 'MEDIUM',
                    'message': (
                        f"Exit speed {abs(exit_d):.0f} km/h slower. "
                        f"Focus on progressive throttle. "
                        f"Prioritize exit over apex speed."
                    ),
                })

            if (c['game_brake_dist'] is not None and
                c['real_brake_dist'] is not None):
                brake_diff = c['game_brake_dist'] - c['real_brake_dist']
                if brake_diff < -10:
                    corner_tips.append({
                        'type': 'BRAKE_POINT',
                        'severity': 'MEDIUM',
                        'message': (
                            f"Braking {abs(brake_diff):.0f}m earlier than pro. "
                            f"Move brake point 5m later each attempt."
                        ),
                    })

        elif td < -0.02:
            corner_tips.append({
                'type': 'POSITIVE',
                'severity': 'INFO',
                'message': (
                    f"Excellent! {abs(td):.3f}s faster. "
                    f"Maintain this as your strength."
                ),
            })

        if corner_tips:
            tips.append({
                'corner': name,
                'short': c['short'],
                'time_delta': td,
                'confidence': c['confidence'],
                'tips': corner_tips,
            })

    # Overall
    reliable = [c for c in corners if c['confidence'] != 'LOW']
    losses = [c for c in reliable if c['time_delta'] > 0.02]
    gains = [c for c in reliable if c['time_delta'] < -0.02]
    overall = []

    if losses:
        worst = max(losses, key=lambda c: c['time_delta'])
        overall.append({
            'type': 'PRIORITY',
            'message': (
                f"Biggest opportunity: {worst['name']} "
                f"({worst['time_delta']:+.3f}s). Focus here first."
            ),
        })

    slow = [c for c in reliable
            if c['game_min_speed'] < 120 and c['min_speed_delta'] < -10]
    if len(slow) >= 2:
        names = ", ".join(c['short'] for c in slow)
        overall.append({
            'type': 'PATTERN',
            'message': (
                f"Pattern: slower in low-speed corners ({names}). "
                f"Practice trail braking for better turn-in grip."
            ),
        })

    fast = [c for c in reliable
            if c['game_min_speed'] > 200 and c['min_speed_delta'] < -10]
    if len(fast) >= 2:
        names = ", ".join(c['short'] for c in fast)
        overall.append({
            'type': 'PATTERN',
            'message': (
                f"Pattern: lacking confidence in high-speed corners ({names}). "
                f"Build trust gradually with lower difficulty."
            ),
        })

    if gains:
        best = min(gains, key=lambda c: c['time_delta'])
        overall.append({
            'type': 'STRENGTH',
            'message': (
                f"Your strongest corner: {best['name']} "
                f"({best['time_delta']:+.3f}s). Keep it up!"
            ),
        })

    # Confidence stats
    conf_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    for c in corners:
        conf_counts[c['confidence']] += 1
    overall.append({
        'type': 'ALIGNMENT',
        'message': (
            f"Data quality: {conf_counts['HIGH']} HIGH, "
            f"{conf_counts['MEDIUM']} MEDIUM, {conf_counts['LOW']} LOW confidence. "
            f"Focus on HIGH/MEDIUM corners for reliable insights."
        ),
    })

    return tips, overall


# ============================================================
# Plot 1: Comparison Map
# ============================================================

def plot_comparison_map(aligned, corners, game_meta, real_meta):
    x = aligned['world_x'].values
    y = aligned['world_y'].values
    delta = aligned['speed_delta'].values
    dist = aligned['lap_distance'].values

    fig, ax = plt.subplots(figsize=(26, 22))
    fig.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    ax.plot(x, y, color='#0f0f0f', lw=18, zorder=1, solid_capstyle='round')
    ax.plot(x, y, color='#2a2a2a', lw=12, zorder=2, solid_capstyle='round')

    cmap = plt.get_cmap('RdYlGn')
    delta_s = smooth(delta, 30)
    d_max = min(float(np.percentile(np.abs(delta_s), 95)), 50)
    norm = plt.Normalize(-d_max, d_max)

    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = mcoll.LineCollection(segs, colors=cmap(norm(delta_s))[:-1], linewidth=7)
    ax.add_collection(lc)

    cx_t, cy_t = float(np.mean(x)), float(np.mean(y))
    tr = max(float(x.max() - x.min()), float(y.max() - y.min()))

    # Plot anchor points on map
    game_anch_d = aligned.attrs.get('game_anch_d', np.array([]))
    for ad in game_anch_d:
        if ad == 0 or ad >= dist.max():
            continue
        ai = int(np.argmin(np.abs(dist - ad)))
        ax.scatter(x[ai], y[ai], color='yellow', s=30, zorder=7,
                   marker='D', alpha=0.6)

    used_pos = []
    for c in corners:
        ai = int(np.argmin(np.abs(dist - c['apex_dist'])))
        cx, cy = float(x[ai]), float(y[ai])
        color = c['color']
        conf = c['confidence']

        mask = (dist >= c['entry_dist']) & (dist <= c['exit_dist'])
        cidx = np.where(mask)[0]
        if len(cidx) > 2:
            ax.plot(x[cidx], y[cidx], color=color, lw=13,
                    alpha=0.35, zorder=3, solid_capstyle='round')

        edge_c = '#ffffff' if conf == 'HIGH' else '#ffaa00' if conf == 'MEDIUM' else '#ff4444'
        edge_w = 2.5 if conf == 'HIGH' else 2.0
        ax.scatter(cx, cy, color=color, s=400, zorder=8,
                   marker='o', edgecolors=edge_c, linewidths=edge_w)
        ax.text(cx, cy, str(c['id']), color='white', fontsize=9,
                fontweight='bold', ha='center', va='center', zorder=9)

        td = c['time_delta']
        td_str = f"{td:+.3f}s"
        conf_mark = "" if conf == "HIGH" else " ?" if conf == "MEDIUM" else " ??"

        dx, dy = cx - cx_t, cy - cy_t
        d = max(np.sqrt(dx**2 + dy**2), 1.0)
        nx, ny = dx / d, dy / d
        ld = tr * 0.12

        for _ in range(20):
            tx = cx + nx * ld
            ty = cy + ny * ld
            if not any(abs(tx - px) < tr * 0.045 and abs(ty - py) < tr * 0.032
                       for px, py in used_pos):
                break
            ang = 0.22
            nx2 = nx * np.cos(ang) - ny * np.sin(ang)
            ny2 = nx * np.sin(ang) + ny * np.cos(ang)
            nx, ny = nx2, ny2
            ld *= 1.05

        tx, ty = cx + nx * ld, cy + ny * ld
        used_pos.append((tx, ty))

        box_c = "#881111" if td > 0.02 else "#116611" if td < -0.02 else "#333333"
        edge = "#ff4444" if td > 0.02 else "#44ff44" if td < -0.02 else "#888888"
        if conf == "LOW":
            box_c = "#333333"
            edge = "#666666"

        label = (f"T{c['id']} {c['short']}{conf_mark}\n"
                 f"You:{c['game_min_speed']:.0f} Pro:{c['real_min_speed']:.0f}\n"
                 f"{td_str}")

        ax.annotate(
            label, xy=(cx, cy), xytext=(tx, ty),
            fontsize=6, color='white', fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.25', facecolor=box_c,
                      edgecolor=edge, alpha=0.9, lw=1.5),
            arrowprops=dict(arrowstyle='->', color='white', lw=1.2,
                            connectionstyle='arc3,rad=0.08'),
            zorder=15
        )

    sx, sy = float(x[0]), float(y[0])
    ax.scatter(sx, sy, color='white', s=350, zorder=12,
               marker='s', edgecolors='lime', lw=3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.018, pad=0.03, shrink=0.5)
    cbar.set_label('Speed Delta (km/h)\nGreen=You Faster | Red=Pro Faster',
                   color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    actual_gap = game_meta['best_time'] - real_meta['lap_time']
    p1_count = aligned.attrs.get('pass1_count', '?')
    p2_count = aligned.attrs.get('pass2_count', '?')

    ax.set_title(
        f'SUZUKA - You vs {real_meta["driver"]} '
        f'({real_meta["year"]} {real_meta["session"]})\n'
        f'You: {format_laptime(game_meta["best_time"])}  |  '
        f'Pro: {format_laptime(real_meta["lap_time"])}  |  '
        f'Gap: {actual_gap:+.3f}s\n'
        f'Two-Pass Alignment (P1:{p1_count} + P2:{p2_count} anchors)  |  '
        f'Diamonds = anchor points',
        color='white', fontsize=13, fontweight='bold', pad=25
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig


# ============================================================
# Plot 2: Speed Overlay
# ============================================================

def plot_speed_overlay(aligned, corners, game_meta, real_meta):
    fig, axes = plt.subplots(4, 1, figsize=(28, 18),
                             gridspec_kw={'height_ratios': [4, 1.5, 1.2, 1.2]},
                             sharex=True)
    fig.set_facecolor(BG_COLOR)
    dist = aligned['lap_distance'].values

    # Speed
    ax1 = axes[0]
    ax1.set_facecolor(BG_COLOR)
    gs_data = smooth(aligned['game_speed_kmh'].values, 15)
    rs_data = smooth(aligned['real_speed_kmh'].values, 15)

    ax1.plot(dist, gs_data, color='#00ff88', lw=2.5,
             label=f'You ({format_laptime(game_meta["best_time"])})')
    ax1.plot(dist, rs_data, color='#ff4488', lw=2.5,
             label=f'{real_meta["driver"]} ({format_laptime(real_meta["lap_time"])})')
    ax1.fill_between(dist, gs_data, rs_data, where=gs_data > rs_data,
                     color='#00ff88', alpha=0.12)
    ax1.fill_between(dist, gs_data, rs_data, where=gs_data < rs_data,
                     color='#ff4488', alpha=0.12)

    # Anchor markers on speed trace
    game_anch_d = aligned.attrs.get('game_anch_d', np.array([]))
    for ad in game_anch_d:
        if 0 < ad < dist.max():
            ai = int(np.argmin(np.abs(dist - ad)))
            ax1.axvline(x=ad, color='yellow', alpha=0.15, lw=0.8)

    for c in corners:
        alpha = 0.06 if c['confidence'] == 'HIGH' else 0.03
        ax1.axvspan(c['entry_dist'], c['exit_dist'],
                    alpha=alpha, color=c['color'])
        conf_mark = "" if c['confidence'] == "HIGH" else "?" if c['confidence'] == "MEDIUM" else "??"
        ax1.text(c['apex_dist'], 370,
                 f"T{c['id']}\n{conf_mark}",
                 color='white', fontsize=5, ha='center', va='bottom',
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.1', facecolor=c['color'],
                           alpha=0.5, edgecolor='none'))

    ax1.set_ylabel('Speed (km/h)', color='white', fontsize=11)
    ax1.set_ylim(0, 380)
    ax1.tick_params(colors='white')
    ax1.grid(alpha=0.08, color='white')
    ax1.legend(loc='upper right', facecolor='#1a1a2e',
               edgecolor='white', labelcolor='white', fontsize=9)
    ax1.set_title(
        f'SUZUKA Speed Comparison (Two-Pass Alignment)  |  '
        f'You vs {real_meta["driver"]} {real_meta["year"]} Q',
        color='white', fontsize=14, fontweight='bold'
    )

    # Time delta
    ax2 = axes[1]
    ax2.set_facecolor(BG_COLOR)
    td = aligned['time_delta'].values
    ax2.fill_between(dist, td, where=td > 0,
                     color='#ff4444', alpha=0.5, label='You losing')
    ax2.fill_between(dist, td, where=td < 0,
                     color='#44ff44', alpha=0.5, label='You gaining')
    ax2.plot(dist, td, color='white', lw=1.5)
    ax2.axhline(y=0, color='#666666', lw=1)
    ax2.set_ylabel('Cum. Delta (s)', color='white', fontsize=10)
    ax2.tick_params(colors='white')
    ax2.legend(loc='upper right', facecolor='#1a1a2e',
               edgecolor='white', labelcolor='white', fontsize=8)

    # Throttle
    ax3 = axes[2]
    ax3.set_facecolor(BG_COLOR)
    if 'game_throttle' in aligned.columns:
        ax3.plot(dist, aligned['game_throttle'].values,
                 color='#00ff88', lw=1.2, alpha=0.7, label='You')
    if 'real_throttle' in aligned.columns:
        ax3.plot(dist, aligned['real_throttle'].values,
                 color='#ff4488', lw=1.2, alpha=0.7, label='Pro')
    ax3.set_ylabel('Throttle', color='white', fontsize=10)
    ax3.set_ylim(-0.05, 1.1)
    ax3.tick_params(colors='white')
    ax3.legend(loc='upper right', facecolor='#1a1a2e',
               edgecolor='white', labelcolor='white', fontsize=8)

    # Brake
    ax4 = axes[3]
    ax4.set_facecolor(BG_COLOR)
    if 'game_brake' in aligned.columns:
        ax4.fill_between(dist, aligned['game_brake'].values,
                         color='#00ff88', alpha=0.4, label='You')
    if 'real_brake' in aligned.columns:
        ax4.fill_between(dist, aligned['real_brake'].values,
                         color='#ff4488', alpha=0.4, label='Pro')
    ax4.set_ylabel('Brake', color='white', fontsize=10)
    ax4.set_xlabel('Track Distance (m)', color='white', fontsize=12)
    ax4.set_ylim(-0.05, 1.1)
    ax4.tick_params(colors='white')
    ax4.legend(loc='upper right', facecolor='#1a1a2e',
               edgecolor='white', labelcolor='white', fontsize=8)

    plt.tight_layout()
    return fig


# ============================================================
# Plot 3: Corner Analysis (18 corners)
# ============================================================

def plot_corner_analysis(corners, game_meta, real_meta):
    n = len(corners)
    fig, axes = plt.subplots(2, 1, figsize=(30, 14), sharex=True)
    fig.set_facecolor(BG_COLOR)

    x_pos = np.arange(n)
    bw = 0.35

    ax1 = axes[0]
    ax1.set_facecolor(BG_COLOR)
    g_speeds = [c['game_min_speed'] for c in corners]
    r_speeds = [c['real_min_speed'] for c in corners]

    ax1.bar(x_pos - bw / 2, g_speeds, bw, color='#00ff88', alpha=0.8,
            label='You', edgecolor='white', lw=0.5)
    ax1.bar(x_pos + bw / 2, r_speeds, bw, color='#ff4488', alpha=0.8,
            label='Pro', edgecolor='white', lw=0.5)

    for i, (g, r) in enumerate(zip(g_speeds, r_speeds)):
        diff = g - r
        c_color = '#44ff44' if diff > 2 else '#ff4444' if diff < -2 else '#888888'
        conf = corners[i]['confidence']
        marker = "" if conf == "HIGH" else " ?" if conf == "MEDIUM" else " X"
        ax1.text(i, max(g, r) + 8, f"{diff:+.0f}{marker}",
                 color=c_color, fontsize=7, ha='center', fontweight='bold')

    ax1.set_ylabel('Min Speed (km/h)', color='white', fontsize=11)
    ax1.tick_params(colors='white')
    ax1.legend(loc='upper right', facecolor='#1a1a2e',
               edgecolor='white', labelcolor='white', fontsize=10)
    ax1.grid(axis='y', alpha=0.1, color='white')
    ax1.set_title(
        f'18-Corner Analysis: You vs {real_meta["driver"]} '
        f'({real_meta["year"]} Q)  |  ?=Medium  X=Low conf',
        color='white', fontsize=14, fontweight='bold'
    )

    ax2 = axes[1]
    ax2.set_facecolor(BG_COLOR)
    deltas = [c['time_delta'] for c in corners]
    confs = [c['confidence'] for c in corners]

    bar_colors = []
    for d, conf in zip(deltas, confs):
        if conf == 'LOW':
            bar_colors.append('#555555')
        elif d > 0.01:
            bar_colors.append('#ff4444')
        elif d < -0.01:
            bar_colors.append('#44ff44')
        else:
            bar_colors.append('#888888')

    ax2.bar(x_pos, deltas, 0.6, color=bar_colors, edgecolor='white',
            lw=0.5, alpha=0.85)

    for i, (d, conf) in enumerate(zip(deltas, confs)):
        y_off = 0.003 if d >= 0 else -0.003
        va = 'bottom' if d >= 0 else 'top'
        marker = "" if conf == "HIGH" else " ?" if conf == "MEDIUM" else " X"
        ax2.text(i, d + y_off, f"{d:+.3f}s{marker}",
                 color='white', fontsize=6, ha='center', va=va,
                 fontweight='bold')

    ax2.axhline(y=0, color='white', lw=1)
    ax2.set_ylabel('Time Delta (s)\nRed=Losing | Grey=Low Conf',
                   color='white', fontsize=10)
    ax2.set_xlabel('Corner', color='white', fontsize=12)
    ax2.tick_params(colors='white')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(
        [f"T{c['id']}\n{c['short']}" for c in corners],
        fontsize=6.5, color='white'
    )
    ax2.grid(axis='y', alpha=0.1, color='white')

    high_total = sum(d for d, conf in zip(deltas, confs) if conf == 'HIGH')
    med_total = sum(d for d, conf in zip(deltas, confs) if conf == 'MEDIUM')
    ax2.text(n - 1, max(deltas) * 0.8,
             f"HIGH: {high_total:+.3f}s | MED: {med_total:+.3f}s",
             color='white', fontsize=10, fontweight='bold', ha='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e',
                       edgecolor='white', alpha=0.9))

    plt.tight_layout()
    return fig


# ============================================================
# Plot 4: Coaching Report
# ============================================================

def plot_coaching_report(corners, tips, overall, game_meta, real_meta):
    fig = plt.figure(figsize=(24, 18))
    fig.set_facecolor(BG_COLOR)

    gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 3.5, 2],
                           hspace=0.3, wspace=0.3)

    # Header
    ax_head = fig.add_subplot(gs[0, :])
    ax_head.set_facecolor(BG_COLOR)
    ax_head.set_xlim(0, 10)
    ax_head.set_ylim(0, 3)
    ax_head.set_xticks([])
    ax_head.set_yticks([])

    actual_gap = game_meta['best_time'] - real_meta['lap_time']
    ax_head.text(5, 2.5, "COACHING REPORT",
                 color='white', fontsize=22, fontweight='bold',
                 ha='center', va='center')
    ax_head.text(5, 1.8,
                 f"You ({format_laptime(game_meta['best_time'])})  vs  "
                 f"{real_meta['driver']} ({format_laptime(real_meta['lap_time'])})"
                 f"  |  Gap: {actual_gap:+.3f}s",
                 color='#aaaaaa', fontsize=14, ha='center', va='center')
    ax_head.text(5, 1.1,
                 f"Suzuka  |  {real_meta['year']} {real_meta['session']}  |  "
                 f"Two-Pass Alignment  |  18 Corners",
                 color='#888888', fontsize=12, ha='center', va='center')

    if overall:
        priority = [o for o in overall if o['type'] == 'PRIORITY']
        if priority:
            ax_head.text(5, 0.3,
                         f">> {priority[0]['message']}",
                         color='#ffaa00', fontsize=10, ha='center', va='center',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e',
                                   edgecolor='#ffaa00', alpha=0.9))

    for spine in ax_head.spines.values():
        spine.set_visible(False)

    # Corner tips (left)
    ax_tips = fig.add_subplot(gs[1, 0])
    ax_tips.set_facecolor(BG_COLOR)
    ax_tips.set_xlim(0, 10)
    ax_tips.set_xticks([])
    ax_tips.set_yticks([])

    ax_tips.text(5, 0.97, "CORNER-BY-CORNER TIPS",
                 color='white', fontsize=14, fontweight='bold',
                 ha='center', va='top', transform=ax_tips.transAxes)

    y = 0.90
    for tip_group in tips[:10]:
        if y < 0.05:
            break

        corner = tip_group['corner']
        td = tip_group['time_delta']
        conf = tip_group['confidence']

        td_color = '#ff4444' if td > 0 else '#44ff44'
        td_icon = "[-]" if td > 0 else "[+]"

        ax_tips.text(0.02, y,
                     f"{td_icon} {corner} ({td:+.3f}s) [{conf}]",
                     color=td_color, fontsize=9, fontweight='bold',
                     transform=ax_tips.transAxes)
        y -= 0.04

        for tip in tip_group['tips'][:2]:
            if y < 0.05:
                break
            sev_icon = "!!!" if tip['severity'] == 'HIGH' else \
                       "!" if tip['severity'] == 'MEDIUM' else "OK"
            ax_tips.text(0.05, y,
                         f"[{sev_icon}] {tip['message']}",
                         color='#cccccc', fontsize=7,
                         transform=ax_tips.transAxes, wrap=True)
            y -= 0.065
        y -= 0.015

    for spine in ax_tips.spines.values():
        spine.set_visible(False)

    # Speed comparison (right)
    ax_sum = fig.add_subplot(gs[1, 1])
    ax_sum.set_facecolor(BG_COLOR)

    reliable = [c for c in corners if c['confidence'] != 'LOW']
    if reliable:
        names = [f"T{c['id']} {c['short']}" for c in reliable]
        game_spds = [c['game_min_speed'] for c in reliable]
        real_spds = [c['real_min_speed'] for c in reliable]

        y_pos = np.arange(len(reliable))

        ax_sum.barh(y_pos + 0.15, game_spds, 0.3, color='#00ff88',
                    alpha=0.8, label='You')
        ax_sum.barh(y_pos - 0.15, real_spds, 0.3, color='#ff4488',
                    alpha=0.8, label='Pro')

        for i, (g, r) in enumerate(zip(game_spds, real_spds)):
            diff = g - r
            c_color = '#44ff44' if diff > 0 else '#ff4444'
            ax_sum.text(max(g, r) + 5, i, f"{diff:+.0f}",
                        color=c_color, fontsize=7, va='center',
                        fontweight='bold')

        ax_sum.set_yticks(y_pos)
        ax_sum.set_yticklabels(names, fontsize=7, color='white')
        ax_sum.set_xlabel('Min Speed (km/h)', color='white', fontsize=10)
        ax_sum.tick_params(colors='white')
        ax_sum.legend(loc='lower right', facecolor='#1a1a2e',
                      edgecolor='white', labelcolor='white', fontsize=9)
        ax_sum.set_title('Reliable Corners (HIGH + MEDIUM)',
                         color='white', fontsize=12, fontweight='bold')
        ax_sum.invert_yaxis()

    # Overall (bottom)
    ax_pat = fig.add_subplot(gs[2, :])
    ax_pat.set_facecolor(BG_COLOR)
    ax_pat.set_xlim(0, 10)
    ax_pat.set_xticks([])
    ax_pat.set_yticks([])

    ax_pat.text(5, 0.95, "OVERALL PATTERNS & RECOMMENDATIONS",
                color='white', fontsize=14, fontweight='bold',
                ha='center', va='top', transform=ax_pat.transAxes)

    y = 0.78
    for tip in overall:
        if y < 0.1:
            break
        icons = {
            'PRIORITY': ('>>', '#ffcc00'),
            'STRENGTH': ('++', '#44ff44'),
            'PATTERN': ('--', '#aaaaaa'),
            'ALIGNMENT': ('**', '#888888'),
        }
        icon, color = icons.get(tip['type'], ('--', '#aaaaaa'))
        ax_pat.text(0.05, y, f"{icon} {tip['message']}",
                    color=color, fontsize=9.5,
                    transform=ax_pat.transAxes, wrap=True)
        y -= 0.14

    for spine in ax_pat.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    return fig


# ============================================================
# Save Report JSON
# ============================================================

def save_report(corners, tips, overall, aligned, game_meta, real_meta):
    total_delta = float(aligned['time_delta'].iloc[-1])
    actual_gap = game_meta['best_time'] - real_meta['lap_time']

    reliable = [c for c in corners if c['confidence'] != 'LOW']
    worst = max(reliable, key=lambda c: c['time_delta']) if reliable else None
    best = min(reliable, key=lambda c: c['time_delta']) if reliable else None

    conf_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    for c in corners:
        conf_counts[c['confidence']] += 1

    report = {
        "comparison": {
            "game": {
                "track": game_meta['track_name'],
                "lap_time": format_laptime(game_meta['best_time']),
                "lap_time_sec": round(game_meta['best_time'], 3),
            },
            "real": {
                "driver": real_meta['driver'],
                "team": real_meta.get('team', ''),
                "year": real_meta['year'],
                "session": real_meta['session'],
                "lap_time": format_laptime(real_meta['lap_time']),
                "lap_time_sec": round(real_meta['lap_time'], 3),
            },
            "actual_gap_sec": round(actual_gap, 3),
            "calculated_gap_sec": round(total_delta, 3),
            "alignment": "two_pass_v5",
            "anchors": {
                "pass1": aligned.attrs.get('pass1_count', 0),
                "pass2": aligned.attrs.get('pass2_count', 0),
                "total": len(aligned.attrs.get('anchor_pairs', [])),
            },
            "confidence": conf_counts,
            "total_corners": 18,
            "biggest_loss": {
                "corner": worst['name'] if worst else "N/A",
                "delta_sec": round(worst['time_delta'], 3) if worst else 0,
            },
            "biggest_gain": {
                "corner": best['name'] if best else "N/A",
                "delta_sec": round(best['time_delta'], 3) if best else 0,
            },
        },
        "corners": [
            {
                "id": c['id'],
                "name": c['name'],
                "game_min_speed_kmh": round(c['game_min_speed'], 1),
                "real_min_speed_kmh": round(c['real_min_speed'], 1),
                "speed_delta_kmh": round(c['min_speed_delta'], 1),
                "time_delta_sec": round(c['time_delta'], 3),
                "confidence": c['confidence'],
            }
            for c in corners
        ],
        "coaching": {
            "corner_tips": [
                {
                    "corner": t['corner'],
                    "time_delta": round(t['time_delta'], 3),
                    "confidence": t['confidence'],
                    "tips": [
                        {"type": tip['type'], "severity": tip['severity'],
                         "message": tip['message']}
                        for tip in t['tips']
                    ]
                }
                for t in tips
            ],
            "overall": [
                {"type": o['type'], "message": o['message']}
                for o in overall
            ],
        }
    }

    path = OUTPUT_DIR / f"{game_meta['track_name']}_comparison_report.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Report: {path}")
    return path


# ============================================================
# Print Summary
# ============================================================

def print_summary(corners, tips, overall, game_meta, real_meta, total_delta):
    actual_gap = game_meta['best_time'] - real_meta['lap_time']

    print(f"\n{'=' * 85}")
    print(f"  YOU vs {real_meta['driver']} "
          f"({real_meta['year']} {real_meta['race']} - {real_meta['session']})")
    print(f"  You: {format_laptime(game_meta['best_time'])}  |  "
          f"Pro: {format_laptime(real_meta['lap_time'])}  |  "
          f"Gap: {actual_gap:+.3f}s")
    print(f"  Calculated gap: {total_delta:+.3f}s  "
          f"(error: {abs(total_delta - actual_gap):.3f}s)")
    print(f"{'=' * 85}")

    conf_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    for c in corners:
        conf_counts[c['confidence']] += 1
    print(f"\n  Confidence: HIGH={conf_counts['HIGH']} "
          f"MEDIUM={conf_counts['MEDIUM']} LOW={conf_counts['LOW']}")

    print(f"\n  {'#':<4} {'Name':<18} {'You':>6} {'Pro':>6} "
          f"{'Spd':>6} {'Time':>8} {'Conf':<6}")
    print(f"  {'-'*4} {'-'*18} {'-'*6} {'-'*6} "
          f"{'-'*6} {'-'*8} {'-'*6}")

    for c in corners:
        conf_icon = "OK" if c['confidence'] == 'HIGH' else \
                    "? " if c['confidence'] == 'MEDIUM' else "X "
        print(f"  T{c['id']:<2} {c['name']:<18} "
              f"{c['game_min_speed']:>5.0f}  {c['real_min_speed']:>5.0f}  "
              f"{c['min_speed_delta']:>+5.0f}  {c['time_delta']:>+7.3f}s "
              f"[{conf_icon}]")

    reliable = [c for c in corners if c['confidence'] != 'LOW']
    if reliable:
        losses = sorted(reliable, key=lambda c: c['time_delta'], reverse=True)
        print(f"\n  TOP 3 LOSSES:")
        for i, c in enumerate(losses[:3]):
            if c['time_delta'] > 0:
                print(f"    {i+1}. T{c['id']} {c['name']}: "
                      f"{c['time_delta']:+.3f}s "
                      f"(You {c['game_min_speed']:.0f} vs "
                      f"Pro {c['real_min_speed']:.0f}) [{c['confidence']}]")

        gains = sorted(reliable, key=lambda c: c['time_delta'])
        print(f"\n  TOP 3 GAINS:")
        for i, c in enumerate(gains[:3]):
            if c['time_delta'] < 0:
                print(f"    {i+1}. T{c['id']} {c['name']}: "
                      f"{c['time_delta']:+.3f}s [{c['confidence']}]")

    print(f"\n  {'='*45}")
    print(f"  COACHING TIPS")
    print(f"  {'='*45}")
    for t in tips:
        td_icon = "[-]" if t['time_delta'] > 0 else "[+]"
        print(f"\n  {td_icon} {t['corner']} "
              f"({t['time_delta']:+.3f}s) [{t['confidence']}]")
        for tip in t['tips']:
            sev = "!!!" if tip['severity'] == 'HIGH' else \
                  "! " if tip['severity'] == 'MEDIUM' else "OK"
            print(f"    [{sev}] {tip['message']}")

    if overall:
        print(f"\n  OVERALL:")
        for o in overall:
            icon = ">>" if o['type'] == 'PRIORITY' else \
                   "++" if o['type'] == 'STRENGTH' else \
                   "**" if o['type'] == 'ALIGNMENT' else "--"
            print(f"    {icon} {o['message']}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("  F1 Lap Insight - Step 3: Real vs Game Comparison (v5)")
    print(f"  You vs {F1_DRIVER} ({F1_YEAR} {F1_RACE} - Qualifying)")
    print(f"  Two-Pass Alignment + 18 Individual Corners")
    print("=" * 70)

    # [1] Game
    print(f"\n  [1/7] Loading game data...")
    best_lap, game_meta = load_telemetry(DEFAULT_CSV)
    game_data = resample(best_lap, game_meta['track_length'])

    # [2] Real
    print(f"\n  [2/7] Loading real F1 data...")
    real_data, real_meta = load_real_f1_lap()

    # [3] Two-pass alignment
    print(f"\n  [3/7] Two-pass alignment...")
    aligned = align_two_pass(
        game_data, game_meta['track_length'],
        real_data, real_meta['track_length']
    )
    aligned = calculate_time_delta(aligned)

    # [4] 18-corner analysis
    print(f"\n  [4/7] Analyzing 18 corners...")
    corners = analyze_corners(aligned)

    # [5] Coaching
    print(f"\n  [5/7] Generating coaching tips...")
    tips, overall = generate_coaching_tips(corners)

    total_delta = float(aligned['time_delta'].iloc[-1])
    print_summary(corners, tips, overall, game_meta, real_meta, total_delta)

    # [6] Plots
    print(f"\n  [6/7] Drawing plots...")

    fig1 = plot_comparison_map(aligned, corners, game_meta, real_meta)
    p1 = OUTPUT_DIR / f"{game_meta['track_name']}_comparison_map_v5.png"
    fig1.savefig(p1, dpi=DPI_SAVE, bbox_inches='tight')
    print(f"  Map: {p1}")
    plt.close(fig1)

    fig2 = plot_speed_overlay(aligned, corners, game_meta, real_meta)
    p2 = OUTPUT_DIR / f"{game_meta['track_name']}_comparison_speed_v5.png"
    fig2.savefig(p2, dpi=DPI_SAVE, bbox_inches='tight')
    print(f"  Speed: {p2}")
    plt.close(fig2)

    fig3 = plot_corner_analysis(corners, game_meta, real_meta)
    p3 = OUTPUT_DIR / f"{game_meta['track_name']}_comparison_corners_v5.png"
    fig3.savefig(p3, dpi=DPI_SAVE, bbox_inches='tight')
    print(f"  Corners: {p3}")
    plt.close(fig3)

    fig4 = plot_coaching_report(corners, tips, overall, game_meta, real_meta)
    p4 = OUTPUT_DIR / f"{game_meta['track_name']}_coaching_report_v5.png"
    fig4.savefig(p4, dpi=DPI_SAVE, bbox_inches='tight')
    print(f"  Coaching: {p4}")
    plt.close(fig4)

    # [7] JSON
    print(f"\n  [7/7] Saving report...")
    save_report(corners, tips, overall, aligned, game_meta, real_meta)

    print(f"\n{'=' * 70}")
    print(f"  COMPLETE! v5 - Two-Pass Alignment")
    actual_gap = game_meta['best_time'] - real_meta['lap_time']
    print(f"  Gap: {actual_gap:+.3f}s vs {real_meta['driver']}")
    print(f"  Calculated: {total_delta:+.3f}s "
          f"(error: {abs(total_delta - actual_gap):.3f}s)")
    print(f"\n  Files:")
    print(f"    Map:      {p1}")
    print(f"    Speed:    {p2}")
    print(f"    Corners:  {p3}")
    print(f"    Coaching: {p4}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()