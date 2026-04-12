"""
Track alignment engine.
Aligns game telemetry with real F1 data using two-pass approach.
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import List, Tuple, Optional

from src.utils import smooth


def find_braking_points(speed, threshold_decel=3.0, min_gap=80):
    """Find indices where significant deceleration begins."""
    if len(speed) < 10:
        return []
    decel = -np.gradient(smooth(speed, 15))
    points = []
    last = -min_gap
    for i in range(1, len(decel)):
        if decel[i] > threshold_decel and i - last > min_gap:
            points.append(i)
            last = i
    return points


def find_throttle_points(throttle, threshold=0.8, min_gap=80):
    """Find indices where throttle exceeds threshold."""
    if throttle is None or len(throttle) < 10:
        return []
    points = []
    last = -min_gap
    below = True
    for i in range(len(throttle)):
        if throttle[i] < threshold * 0.5:
            below = True
        if below and throttle[i] > threshold and i - last > min_gap:
            points.append(i)
            last = i
            below = False
    return points


def find_feature_anchors(speed, throttle=None, brake=None,
                         n_speed=15, order=None):
    """
    Find anchor points from speed/throttle/brake signals.

    Returns:
        List of (index, feature_type, value)
    """
    anchors = []
    speed_s = smooth(speed, 30)

    if order is None:
        order = max(len(speed) // 40, 30)

    # Speed minima
    min_idx = argrelextrema(speed_s, np.less, order=order)[0]
    if len(min_idx) > 0:
        speeds = speed_s[min_idx]
        top = np.argsort(speeds)[:n_speed]
        for i in top:
            anchors.append((int(min_idx[i]), 'speed_min', float(speeds[i])))

    # Speed maxima
    max_idx = argrelextrema(speed_s, np.greater, order=order)[0]
    if len(max_idx) > 0:
        speeds = speed_s[max_idx]
        top = np.argsort(-speeds)[:10]
        for i in top:
            anchors.append((int(max_idx[i]), 'speed_max', float(speeds[i])))

    # Braking points
    bp = find_braking_points(speed, threshold_decel=3.0, min_gap=80)
    for idx in bp:
        anchors.append((int(idx), 'brake_start', float(speed_s[idx])))

    # Throttle application
    if throttle is not None and len(throttle) > 0:
        tp = find_throttle_points(throttle, threshold=0.8, min_gap=80)
        for idx in tp:
            anchors.append((int(idx), 'throttle_full', float(speed_s[idx])))

    anchors.sort(key=lambda a: a[0])
    return anchors


def match_anchors(game_anchors, game_length,
                  real_anchors, real_length,
                  pos_tolerance=0.03, score_threshold=0.10):
    """
    Match game anchors to real anchors.

    Returns:
        List of (game_normalized, real_normalized) pairs
    """
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

    # Remove duplicates / non-monotonic
    cleaned = [pairs[0]]
    for p in pairs[1:]:
        if p[0] > cleaned[-1][0] + 0.005 and p[1] > cleaned[-1][1] + 0.005:
            cleaned.append(p)

    return cleaned


def local_cross_correlation(game_speed, real_speed,
                            game_start, game_end,
                            real_center, search_range=60):
    """
    Find best local alignment using cross-correlation.

    Returns:
        (shift, correlation) or None
    """
    g_start = max(0, int(game_start))
    g_end = min(len(game_speed), int(game_end))
    game_seg = game_speed[g_start:g_end]

    if len(game_seg) < 50:
        return None

    game_norm = game_seg - np.mean(game_seg)
    g_std = np.std(game_norm)
    if g_std < 1.0:
        return None
    game_norm = game_norm / g_std

    best_corr = -1
    best_shift = 0

    for shift in range(-search_range, search_range + 1, 2):
        r_start = max(0, int(real_center - len(game_seg) // 2 + shift))
        r_end = r_start + len(game_seg)

        if r_end > len(real_speed):
            continue

        real_seg = real_speed[r_start:r_end]
        if len(real_seg) != len(game_seg):
            continue

        real_norm = real_seg - np.mean(real_seg)
        r_std = np.std(real_norm)
        if r_std < 1.0:
            continue
        real_norm = real_norm / r_std

        corr = float(np.mean(game_norm * real_norm))
        if corr > best_corr:
            best_corr = corr
            best_shift = shift

    return best_shift, best_corr


def find_anchor_gaps(anchor_pairs, game_length, min_gap=350):
    """Find sections where anchors are too sparse."""
    gaps = []
    dists = sorted([a[0] * game_length for a in anchor_pairs])

    for i in range(len(dists) - 1):
        gap = dists[i + 1] - dists[i]
        if gap > min_gap:
            gaps.append((dists[i], dists[i + 1], gap))

    return gaps


def align_two_pass(game_data, game_length,
                   real_data, real_length,
                   max_drift=80, corr_threshold=0.5,
                   verbose=True):
    """
    Two-pass alignment: global features + local cross-correlation.

    Returns:
        Aligned DataFrame with game_* and real_* columns
    """
    if verbose:
        print(f"\n  --- Two-Pass Alignment ---")
        print(f"  Game: {game_length:.0f}m  |  Real: {real_length:.0f}m")

    game_speed = game_data['speed_kmh'].values
    real_speed = real_data['speed_kmh'].values

    game_throttle = (game_data['throttle'].values
                     if 'throttle' in game_data.columns else None)
    real_throttle = (real_data['throttle'].values
                     if 'throttle' in real_data.columns else None)

    # ---- PASS 1: Global feature matching ----
    if verbose:
        print(f"\n  PASS 1: Global feature matching")

    game_anchors = find_feature_anchors(game_speed, game_throttle)
    real_anchors = find_feature_anchors(real_speed, real_throttle)

    if verbose:
        print(f"  Game features: {len(game_anchors)}")
        print(f"  Real features: {len(real_anchors)}")

    anchor_pairs = match_anchors(
        game_anchors, game_length,
        real_anchors, real_length
    )

    # Drift filter
    filtered = []
    for g, r in anchor_pairs:
        drift = abs(g * game_length - r * real_length)
        if drift < max_drift or g == 0.0 or g == 1.0:
            filtered.append((g, r))
        elif verbose:
            print(f"  REMOVED: game {g*game_length:.0f}m (drift {drift:.0f}m)")
    anchor_pairs = filtered

    if verbose:
        print(f"  Pass 1: {len(anchor_pairs)} anchors")

    # ---- PASS 2: Local cross-correlation ----
    if verbose:
        print(f"\n  PASS 2: Local cross-correlation")

    gaps = find_anchor_gaps(anchor_pairs, game_length)
    if verbose:
        print(f"  Found {len(gaps)} gaps")

    new_anchors = []

    for gap_start, gap_end, gap_size in gaps:
        g_start_idx = int(gap_start)
        g_end_idx = min(int(gap_end), len(game_speed))

        if g_end_idx - g_start_idx < 100:
            continue

        game_seg = game_speed[g_start_idx:g_end_idx]
        game_seg_s = smooth(game_seg, 15)

        local_order = max(len(game_seg) // 8, 15)
        local_min = argrelextrema(game_seg_s, np.less, order=local_order)[0]

        for lm in local_min:
            game_dist = gap_start + lm
            game_norm = game_dist / game_length

            real_est_norm = np.interp(
                game_norm,
                [a[0] for a in anchor_pairs],
                [a[1] for a in anchor_pairs]
            )
            real_est_dist = real_est_norm * real_length

            result = local_cross_correlation(
                game_speed, real_speed,
                game_dist - 80, game_dist + 80,
                real_est_dist, search_range=60
            )

            if result is None:
                continue

            shift, corr = result
            if corr < corr_threshold:
                continue

            real_refined = real_est_dist + shift
            real_idx = int(np.clip(real_refined, 0, len(real_speed) - 1))
            real_spd = float(smooth(real_speed, 15)[real_idx])
            game_spd = float(game_seg_s[lm])

            spd_diff = abs(game_spd - real_spd) / max(game_spd, 1)
            if spd_diff > 0.4:
                continue

            drift = abs(game_dist - real_refined)
            if drift > 100:
                continue

            new_anchors.append((game_dist / game_length,
                                real_refined / real_length))

            if verbose:
                print(f"    Added: game {game_dist:.0f}m <-> "
                      f"real {real_refined:.0f}m (corr={corr:.2f})")

    # Merge all anchors
    all_pairs = anchor_pairs + new_anchors
    all_pairs.sort(key=lambda p: p[0])

    cleaned = [all_pairs[0]]
    for p in all_pairs[1:]:
        if p[0] > cleaned[-1][0] + 0.003 and p[1] > cleaned[-1][1] + 0.003:
            cleaned.append(p)

    final_pairs = []
    for g, r in cleaned:
        drift = abs(g * game_length - r * real_length)
        if drift < 100 or g == 0.0 or g == 1.0:
            final_pairs.append((g, r))

    if verbose:
        print(f"\n  Final: {len(final_pairs)} anchors "
              f"(P1:{len(anchor_pairs)} + P2:+{len(new_anchors)})")

    # ---- Build aligned data ----
    game_anch_d = np.array([a[0] for a in final_pairs]) * game_length
    real_anch_d = np.array([a[1] for a in final_pairs]) * real_length

    distances = np.arange(0, game_length, 1.0)
    real_mapped = np.interp(distances, game_anch_d, real_anch_d)

    aligned = pd.DataFrame({'lap_distance': distances})

    game_norm_d = game_data['lap_distance'].values
    for col in ['speed_kmh', 'throttle', 'brake', 'steering', 'gear']:
        if col in game_data.columns:
            aligned[f'game_{col}'] = np.interp(
                distances, game_norm_d, game_data[col].values
            )

    real_norm_d = real_data['lap_distance'].values
    for col in ['speed_kmh', 'throttle', 'brake', 'gear']:
        if col in real_data.columns:
            aligned[f'real_{col}'] = np.interp(
                real_mapped, real_norm_d, real_data[col].values
            )

    aligned['speed_delta'] = (aligned['game_speed_kmh'] -
                               aligned['real_speed_kmh'])

    if 'world_position_X' in game_data.columns:
        aligned['world_x'] = np.interp(
            distances, game_norm_d,
            game_data['world_position_X'].values
        )
        aligned['world_y'] = np.interp(
            distances, game_norm_d,
            game_data['world_position_Y'].values
        )

    # Store metadata in attrs
    aligned.attrs['anchor_pairs'] = final_pairs
    aligned.attrs['game_anch_d'] = game_anch_d
    aligned.attrs['real_anch_d'] = real_anch_d
    aligned.attrs['pass1_count'] = len(anchor_pairs)
    aligned.attrs['pass2_count'] = len(new_anchors)

    return aligned