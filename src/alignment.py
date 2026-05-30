"""
Track alignment engine.
Aligns game telemetry with real F1 data using shape + distance approach.

Architecture:
  Layer 1: Shape alignment (coordinate system unification)
    - Resample both trajectories to equal arc-length
    - Compute curvature profiles (rotation/scale invariant)
    - Cross-correlate curvature to find start-point offset
    - SVD-based rigid transform (translate + rotate + optional reflect)

  Layer 2: Distance-based alignment (telemetry signal matching)
    - Feature anchors (speed extrema, decel peaks, throttle/brake)
    - Anchor matching with curvature-assisted scoring
    - Local cross-correlation gap filling

Changes from previous version:
  - [NEW] Shape alignment layer for coordinate system unification
  - [NEW] Curvature-based start-point detection
  - [NEW] SVD rigid transform with reflection handling
  - [NEW] Curvature anchors in feature detection
  - [NEW] Shape alignment error in quality metrics
  - [IMPROVED] Anchor matching uses curvature similarity
  - [KEPT] All existing distance-based alignment logic
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.ndimage import uniform_filter1d
from typing import List, Tuple, Optional, Dict, Any

from src.utils import smooth, signed_curvature_from_smoothed


# ============================================================
# Shape Alignment — Layer 1
# ============================================================

def resample_trajectory(xy: np.ndarray, n_points: int = 1000) -> np.ndarray:
    """
    Resample 2D trajectory to uniform arc-length spacing.

    Args:
        xy: (N, 2) array of XY coordinates
        n_points: number of output points

    Returns:
        (n_points, 2) resampled trajectory
    """
    if len(xy) < 10:
        return xy

    # Compute cumulative arc length
    diffs = np.diff(xy, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    cum_length = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cum_length[-1]

    if total_length < 1.0:
        return xy

    # Uniform arc-length targets
    targets = np.linspace(0, total_length, n_points)

    # Interpolate X and Y separately
    resampled = np.column_stack([
        np.interp(targets, cum_length, xy[:, 0]),
        np.interp(targets, cum_length, xy[:, 1]),
    ])

    return resampled


def compute_curvature(xy: np.ndarray, smooth_window: int = 15) -> np.ndarray:
    """
    Compute signed curvature from 2D trajectory.

    Curvature = (dx * ddy - dy * ddx) / (dx^2 + dy^2)^1.5

    This is rotation and translation invariant — ideal for matching
    trajectories in different coordinate systems.

    Args:
        xy: (N, 2) array
        smooth_window: smoothing before differentiation

    Returns:
        (N,) curvature array (positive = left turn)
    """
    if len(xy) < smooth_window + 5:
        return np.zeros(len(xy))

    x = uniform_filter1d(xy[:, 0], size=smooth_window)
    y = uniform_filter1d(xy[:, 1], size=smooth_window)

    return signed_curvature_from_smoothed(x, y, eps=1e-10)


def find_start_offset_by_curvature(curv_a: np.ndarray,
                                    curv_b: np.ndarray) -> int:
    """
    Find the start-point offset between two closed-loop trajectories
    using curvature profile cross-correlation.

    The curvature profile is invariant to rotation, translation, and scale,
    so it directly reveals how the two trajectories' start/finish lines differ.

    Args:
        curv_a: curvature of trajectory A (reference)
        curv_b: curvature of trajectory B (to be shifted)

    Returns:
        offset: number of points to shift B forward to align with A
    """
    n = len(curv_a)
    if n != len(curv_b):
        # Resample B to match A's length
        curv_b = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(curv_b)),
            curv_b
        )

    # Normalize
    a = curv_a - np.mean(curv_a)
    b = curv_b - np.mean(curv_b)
    a_std = np.std(a)
    b_std = np.std(b)

    if a_std < 1e-8 or b_std < 1e-8:
        return 0

    a = a / a_std
    b = b / b_std

    # Circular cross-correlation via FFT
    fa = np.fft.fft(a)
    fb = np.fft.fft(b)
    cross = np.fft.ifft(fa * np.conj(fb)).real

    # Best offset (and check flipped version)
    offset_normal = int(np.argmax(cross))

    # Also try sign-flipped curvature (handles axis reflection)
    cross_flip = np.fft.ifft(fa * np.conj(np.fft.fft(-b))).real
    offset_flip = int(np.argmax(cross_flip))

    if cross[offset_normal] >= cross_flip[offset_flip]:
        return offset_normal
    else:
        # Negative offset signals that B needs reflection
        return -(offset_flip + 1)


def estimate_rigid_transform(src: np.ndarray, dst: np.ndarray,
                              allow_reflection: bool = True
                              ) -> Dict[str, Any]:
    """
    Estimate optimal rigid transform (rotation + translation + optional reflection)
    from src to dst using SVD decomposition.

    This is a Procrustes-style alignment but handles:
      - Different scales (normalizes first, returns scale factor)
      - Reflections (game engine may mirror an axis)
      - Robust to noise via least-squares SVD

    Args:
        src: (N, 2) source points
        dst: (N, 2) destination points
        allow_reflection: if True, allows mirror transforms

    Returns:
        dict with keys:
          rotation: 2x2 rotation matrix
          translation: (2,) translation vector
          scale: float scale factor
          reflection: bool whether reflection was applied
          residual: float mean alignment error after transform
    """
    assert len(src) == len(dst), "Point sets must have same length"

    # Centroids
    src_c = np.mean(src, axis=0)
    dst_c = np.mean(dst, axis=0)

    # Center
    src_centered = src - src_c
    dst_centered = dst - dst_c

    # Scale (RMS distance from centroid)
    src_scale = np.sqrt(np.mean(np.sum(src_centered**2, axis=1)))
    dst_scale = np.sqrt(np.mean(np.sum(dst_centered**2, axis=1)))

    if src_scale < 1e-10 or dst_scale < 1e-10:
        return {
            'rotation': np.eye(2),
            'translation': dst_c - src_c,
            'scale': 1.0,
            'reflection': False,
            'residual': float('inf'),
        }

    scale = dst_scale / src_scale
    src_norm = src_centered / src_scale
    dst_norm = dst_centered / dst_scale

    # SVD of cross-covariance matrix
    H = src_norm.T @ dst_norm
    U, S, Vt = np.linalg.svd(H)

    # Rotation (handle reflection)
    # Compute both options and pick the one with lower residual

    # No-reflection
    R_no = Vt.T @ U.T
    t_no = dst_c - scale * (R_no @ src_c)
    res_no = float(np.mean(np.sqrt(np.sum(
        (scale * (src @ R_no.T) + t_no - dst)**2, axis=1))))

    # With-reflection
    sign_matrix = np.diag([1.0, -1.0])
    R_ref = Vt.T @ sign_matrix @ U.T
    t_ref = dst_c - scale * (R_ref @ src_c)
    res_ref = float(np.mean(np.sqrt(np.sum(
        (scale * (src @ R_ref.T) + t_ref - dst)**2, axis=1))))

    # Pick the one with lower residual
    if allow_reflection and res_ref < res_no:
        R, translation, residual, reflection = R_ref, t_ref, res_ref, True
    else:
        R, translation, residual, reflection = R_no, t_no, res_no, False

    return {
        'rotation': R,
        'translation': translation,
        'scale': scale,
        'reflection': reflection,
        'residual': residual,
    }


def apply_transform(xy: np.ndarray, transform: Dict[str, Any]) -> np.ndarray:
    """Apply rigid transform to XY points."""
    R = transform['rotation']
    t = transform['translation']
    s = transform['scale']
    return s * (xy @ R.T) + t


def align_track_shapes(game_xy: np.ndarray, real_xy: np.ndarray,
                       n_resample: int = 1000,
                       verbose: bool = True) -> Dict[str, Any]:
    """
    Full shape alignment pipeline.

    Aligns game world coordinates to real (FastF1) coordinate system.
    Handles: different origins, rotations, scales, axis reflections,
    and start/finish line offsets.

    Strategy:
      1. Resample both to equal arc-length (removes sampling density bias)
      2. Compute curvature profiles (rotation/scale/translation invariant)
      3. Cross-correlate curvature to find start-point offset
      4. Roll game trajectory to match start point
      5. SVD rigid transform to align coordinate systems
      6. If residual is high, try with reflection

    Args:
        game_xy: (N, 2) game world positions
        real_xy: (M, 2) real world positions
        n_resample: points for curvature comparison
        verbose: print diagnostics

    Returns:
        dict with:
          transform: rigid transform parameters
          game_xy_aligned: game XY in real coordinate system
          start_offset_frac: fractional start-point offset (0-1)
          curvature_correlation: quality of curvature match
          residual_m: mean alignment error in meters
    """
    # Step 1: Resample to uniform arc-length
    game_rs = resample_trajectory(game_xy, n_resample)
    real_rs = resample_trajectory(real_xy, n_resample)

    # Step 2: Curvature profiles
    curv_game = compute_curvature(game_rs, smooth_window=15)
    curv_real = compute_curvature(real_rs, smooth_window=15)

    # Step 3: Find start-point offset via curvature cross-correlation
    offset = find_start_offset_by_curvature(curv_real, curv_game)

    needs_reflection = offset < 0
    if needs_reflection:
        offset = abs(offset) - 1

    start_offset_frac = offset / n_resample

    if verbose:
        print(f"    Shape: start offset = {start_offset_frac*100:.1f}% "
              f"of track{'  (reflected)' if needs_reflection else ''}")

    # Step 4: Roll game trajectory to align start points
    game_rolled = np.roll(game_rs, -offset, axis=0)

    # If reflection detected, flip one axis before SVD
    if needs_reflection:
        game_rolled[:, 1] = -game_rolled[:, 1]

    # Step 5: SVD rigid transform
    transform = estimate_rigid_transform(
        game_rolled, real_rs, allow_reflection=True
    )

    # Step 6: If residual is too high, try alternative approaches
    # Try without reflection override
    if transform['residual'] > 50 and needs_reflection:
        game_alt = np.roll(game_rs, -offset, axis=0)
        transform_alt = estimate_rigid_transform(
            game_alt, real_rs, allow_reflection=True
        )
        if transform_alt['residual'] < transform['residual']:
            transform = transform_alt
            needs_reflection = False
            game_rolled = game_alt

    # Compute curvature correlation after alignment
    curv_game_aligned = compute_curvature(game_rolled, smooth_window=15)
    curv_corr = _curvature_correlation(curv_real, curv_game_aligned)

    # Apply transform to original game XY (not resampled)
    # First handle start offset and reflection on original points
    game_aligned = apply_transform(game_xy, transform)

    if verbose:
        print(f"    Shape: residual = {transform['residual']:.1f}m, "
              f"curvature corr = {curv_corr:.3f}")
        print(f"    Shape: scale = {transform['scale']:.4f}, "
              f"reflection = {transform['reflection'] or needs_reflection}")

    return {
        'transform': transform,
        'game_xy_aligned': game_aligned,
        'start_offset_frac': start_offset_frac,
        'needs_reflection': needs_reflection,
        'curvature_correlation': curv_corr,
        'residual_m': transform['residual'],
        'game_resampled': game_rs,
        'real_resampled': real_rs,
    }


def _curvature_correlation(curv_a: np.ndarray, curv_b: np.ndarray) -> float:
    """Compute normalized correlation between two curvature profiles."""
    if len(curv_a) != len(curv_b):
        curv_b = np.interp(
            np.linspace(0, 1, len(curv_a)),
            np.linspace(0, 1, len(curv_b)),
            curv_b
        )
    a = curv_a - np.mean(curv_a)
    b = curv_b - np.mean(curv_b)
    a_std = np.std(a)
    b_std = np.std(b)
    if a_std < 1e-8 or b_std < 1e-8:
        return 0.0
    return float(np.mean((a / a_std) * (b / b_std)))


# ============================================================
# Feature detection
# ============================================================

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
    """Find indices where throttle exceeds threshold after being off."""
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


def find_decel_peaks(speed, min_gap=60, smooth_window=20):
    """
    Find peaks of deceleration (= braking zones).

    Catches corners that speed extrema miss because:
      - The corner is shallow (small speed drop)
      - Two corners are close together (merged into one minimum)
      - The corner is high-speed (e.g. 130R: 310→298 km/h)
    """
    speed_s = smooth(speed, smooth_window)
    decel = -np.gradient(speed_s)
    decel_s = smooth(decel, smooth_window)

    order = max(len(decel_s) // 60, 20)
    peaks = argrelextrema(decel_s, np.greater, order=order)[0]

    # Filter: only significant decelerations
    filtered = []
    last = -min_gap
    for p in peaks:
        if decel_s[p] > 1.0 and p - last > min_gap:
            filtered.append(p)
            last = p

    return filtered


def find_curvature_peaks(xy: np.ndarray, min_gap: int = 60,
                         threshold: float = 0.001) -> List[int]:
    """
    Find peaks of absolute curvature from XY trajectory.

    These correspond to corner apexes and are more reliable than
    speed-based detection for high-speed corners where speed barely drops.

    Args:
        xy: (N, 2) trajectory
        min_gap: minimum distance between peaks (in samples)
        threshold: minimum curvature magnitude

    Returns:
        list of indices where curvature peaks occur
    """
    if xy is None or len(xy) < 50:
        return []

    curv = compute_curvature(xy, smooth_window=20)
    abs_curv = np.abs(curv)
    abs_curv_s = smooth(abs_curv, 15)

    order = max(len(abs_curv_s) // 40, 15)
    peaks = argrelextrema(abs_curv_s, np.greater, order=order)[0]

    filtered = []
    last = -min_gap
    for p in peaks:
        if abs_curv_s[p] > threshold and p - last > min_gap:
            filtered.append(int(p))
            last = p

    return filtered


def find_feature_anchors(speed, throttle=None, brake=None,
                         xy=None, n_speed=15, order=None):
    """
    Find anchor points from speed/throttle/brake/curvature signals.

    Adaptive order based on track length + deceleration peaks.
    Now also uses curvature peaks from XY trajectory when available.

    Args:
        speed: speed array (km/h)
        throttle: throttle array (0-1), optional
        brake: brake array (0-1), optional
        xy: (N, 2) world position array, optional (enables curvature anchors)
        n_speed: max number of speed minima to keep
        order: override for argrelextrema order parameter

    Returns:
        List of (index, feature_type, value)
    """
    anchors = []
    speed_s = smooth(speed, 30)

    # Adaptive order: target ~15-20 features for a typical track
    if order is None:
        order = max(len(speed) // 80, 20)

    # Speed minima (primary anchors)
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
        top = np.argsort(-speeds)[:12]
        for i in top:
            anchors.append((int(max_idx[i]), 'speed_max', float(speeds[i])))

    # Deceleration peaks (catches shallow corners)
    decel_peaks = find_decel_peaks(speed, min_gap=60)
    for idx in decel_peaks:
        anchors.append((int(idx), 'decel_peak', float(speed_s[idx])))

    # Braking onset points
    bp = find_braking_points(speed, threshold_decel=3.0, min_gap=80)
    for idx in bp:
        anchors.append((int(idx), 'brake_start', float(speed_s[idx])))

    # Throttle application points
    if throttle is not None and len(throttle) > 0:
        tp = find_throttle_points(throttle, threshold=0.8, min_gap=80)
        for idx in tp:
            anchors.append((int(idx), 'throttle_full', float(speed_s[idx])))

    # Curvature peaks (from XY trajectory — catches high-speed corners)
    if xy is not None and len(xy) == len(speed):
        curv_peaks = find_curvature_peaks(xy, min_gap=60, threshold=0.001)
        for idx in curv_peaks:
            anchors.append((int(idx), 'curvature_peak', float(speed_s[idx])))

    # Deduplicate: remove anchors too close to each other
    anchors.sort(key=lambda a: a[0])
    deduped = []
    for anchor in anchors:
        if deduped and abs(anchor[0] - deduped[-1][0]) < 15:
            # Keep the one with lower speed (more distinctive)
            if anchor[2] < deduped[-1][2]:
                deduped[-1] = anchor
        else:
            deduped.append(anchor)

    return deduped


# ============================================================
# Anchor matching
# ============================================================

def match_anchors(game_anchors, game_length,
                  real_anchors, real_length,
                  pos_tolerance=0.03, score_threshold=0.10):
    """
    Match game anchors to real anchors.

    Uses position proximity + speed similarity + type matching.
    Curvature peaks get extra weight since they're more reliable
    than speed-only features for high-speed corners.

    Returns list of (game_normalized, real_normalized) pairs.
    """
    pairs = [(0.0, 0.0)]
    used_real = set()

    game_by_type = {}
    real_by_type = {}

    for idx, ftype, val in game_anchors:
        game_by_type.setdefault(ftype, []).append((idx / game_length, val))
    for idx, ftype, val in real_anchors:
        real_by_type.setdefault(ftype, []).append((idx / real_length, val))

    # Match priority: curvature_peak first (most geometry-reliable),
    # then speed_min, decel_peak, then others
    type_order = ['curvature_peak', 'speed_min', 'decel_peak',
                  'speed_max', 'brake_start', 'throttle_full']

    for ftype in type_order:
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

                if ftype in ('speed_min', 'decel_peak', 'curvature_peak'):
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

    # Remove non-monotonic
    cleaned = [pairs[0]]
    for p in pairs[1:]:
        if p[0] > cleaned[-1][0] + 0.005 and p[1] > cleaned[-1][1] + 0.005:
            cleaned.append(p)

    return cleaned


# ============================================================
# Local cross-correlation
# ============================================================

def local_cross_correlation(game_speed, real_speed,
                            game_start, game_end,
                            real_center, search_range=60):
    """Find best local alignment using cross-correlation."""
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


def find_anchor_gaps(anchor_pairs, game_length, min_gap=300):
    """Find sections where anchors are too sparse."""
    gaps = []
    dists = sorted([a[0] * game_length for a in anchor_pairs])

    for i in range(len(dists) - 1):
        gap = dists[i + 1] - dists[i]
        if gap > min_gap:
            gaps.append((dists[i], dists[i + 1], gap))

    return gaps


# ============================================================
# Two-pass alignment (main entry point)
# ============================================================

def _layer0_shape_align(game_data, real_data, verbose=True):
    """
    LAYER 0: Shape alignment (coordinate unification).

    Returns:
        (has_game_xy, has_real_xy, game_xy, real_xy, shape_result)
        where game_xy/real_xy are (N,2) arrays or None,
        and shape_result is the dict from align_track_shapes or None.
    """
    has_game_xy = ('world_position_X' in game_data.columns and
                   'world_position_Y' in game_data.columns)
    has_real_xy = ('world_position_X' in real_data.columns and
                   'world_position_Y' in real_data.columns)

    game_xy = None
    real_xy = None
    shape_result = None

    if has_game_xy and has_real_xy:
        if verbose:
            print(f"\n  LAYER 0: Shape alignment (coordinate unification)")

        game_xy = np.column_stack([
            game_data['world_position_X'].values,
            game_data['world_position_Y'].values,
        ])
        real_xy = np.column_stack([
            real_data['world_position_X'].values,
            real_data['world_position_Y'].values,
        ])

        try:
            shape_result = align_track_shapes(
                game_xy, real_xy,
                n_resample=min(1000, min(len(game_xy), len(real_xy))),
                verbose=verbose,
            )
        except Exception as e:
            if verbose:
                print(f"    Shape alignment failed: {e}")
                print(f"    Falling back to distance-only alignment")
            shape_result = None
    elif verbose:
        if not has_game_xy:
            print(f"  ⚠ No game XY coordinates — skipping shape alignment")
        if not has_real_xy:
            print(f"  ⚠ No real XY coordinates — skipping shape alignment")

    return has_game_xy, has_real_xy, game_xy, real_xy, shape_result


def align_two_pass(game_data, game_length,
                   real_data, real_length,
                   max_drift=80, corr_threshold=0.5,
                   verbose=True):
    """
    Two-pass alignment: shape alignment + global features + local xcorr.

    Pipeline:
      0. Shape alignment (if XY available) — unifies coordinate systems
      1. Feature anchors (speed + decel + throttle + curvature)
      2. Anchor matching
      3. Local cross-correlation gap filling
      4. Build aligned DataFrame with all channels

    Returns:
        Aligned DataFrame with game_* and real_* columns,
        plus aligned world coordinates.
    """
    if verbose:
        print(f"\n  --- Two-Pass Alignment ---")
        print(f"  Game: {game_length:.0f}m  |  Real: {real_length:.0f}m")

    game_speed = game_data['speed_kmh'].values
    real_speed = real_data['speed_kmh'].values

    # Cache smoothed speeds — reused in gap fill loop and quality metrics
    game_speed_s = smooth(game_speed, 15)
    real_speed_s = smooth(real_speed, 15)

    game_throttle = (game_data['throttle'].values
                     if 'throttle' in game_data.columns else None)
    real_throttle = (real_data['throttle'].values
                     if 'throttle' in real_data.columns else None)

    # ---- LAYER 0: Shape alignment (coordinate unification) ----
    has_game_xy, has_real_xy, game_xy, real_xy, shape_result = (
        _layer0_shape_align(game_data, real_data, verbose=verbose)
    )

    # ---- PASS 1: Global feature matching ----
    if verbose:
        print(f"\n  PASS 1: Global feature matching")

    game_anchors = find_feature_anchors(
        game_speed, game_throttle, xy=game_xy
    )
    real_anchors = find_feature_anchors(
        real_speed, real_throttle, xy=real_xy
    )

    if verbose:
        g_types = {}
        for _, t, _ in game_anchors:
            g_types[t] = g_types.get(t, 0) + 1
        r_types = {}
        for _, t, _ in real_anchors:
            r_types[t] = r_types.get(t, 0) + 1
        print(f"  Game features: {len(game_anchors)} {dict(g_types)}")
        print(f"  Real features: {len(real_anchors)} {dict(r_types)}")

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
            print(f"  REMOVED: game {g*game_length:.0f}m "
                  f"(drift {drift:.0f}m)")
    anchor_pairs = filtered

    if verbose:
        print(f"  Pass 1: {len(anchor_pairs)} anchors")

    # ---- PASS 2: Local cross-correlation gap fill ----
    if verbose:
        print(f"\n  PASS 2: Local cross-correlation")

    gaps = find_anchor_gaps(anchor_pairs, game_length)
    if verbose:
        print(f"  Found {len(gaps)} gaps")

    new_anchors = []

    # Precompute anchor arrays for np.interp (avoids list comp per candidate)
    anchor_g_arr = np.array([a[0] for a in anchor_pairs])
    anchor_r_arr = np.array([a[1] for a in anchor_pairs])

    for gap_start, gap_end, _gap_size in gaps:
        g_start_idx = int(gap_start)
        g_end_idx = min(int(gap_end), len(game_speed))

        if g_end_idx - g_start_idx < 100:
            continue

        game_seg = game_speed[g_start_idx:g_end_idx]
        game_seg_s = smooth(game_seg, 15)

        local_order = max(len(game_seg) // 8, 15)
        local_min = argrelextrema(game_seg_s, np.less, order=local_order)[0]

        # Also try deceleration peaks in gaps (reuse cached game_seg_s)
        decel = -np.gradient(game_seg_s)
        decel_peaks = argrelextrema(smooth(decel, 10), np.greater,
                                     order=local_order)[0]
        candidates = list(local_min) + list(decel_peaks)
        candidates = sorted(set(candidates))

        for lm in candidates:
            game_dist = gap_start + lm
            game_norm = game_dist / game_length

            real_est_norm = np.interp(
                game_norm, anchor_g_arr, anchor_r_arr
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
            real_spd = float(real_speed_s[real_idx])
            game_spd = (float(game_seg_s[lm])
                        if lm < len(game_seg_s) else 0)

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

    # ---- World coordinates ----
    # Game XY: raw (original coordinate system)
    if has_game_xy:
        aligned['game_world_x'] = np.interp(
            distances, game_norm_d,
            game_data['world_position_X'].values
        )
        aligned['game_world_y'] = np.interp(
            distances, game_norm_d,
            game_data['world_position_Y'].values
        )
        # Legacy compatibility
        aligned['world_x'] = aligned['game_world_x']
        aligned['world_y'] = aligned['game_world_y']

    # Real XY: interpolated via real_mapped
    if has_real_xy:
        aligned['real_world_x'] = np.interp(
            real_mapped, real_norm_d,
            real_data['world_position_X'].values
        )
        aligned['real_world_y'] = np.interp(
            real_mapped, real_norm_d,
            real_data['world_position_Y'].values
        )

    # Aligned game XY: game coordinates transformed into real coord system
    # GATING: only use shape result if quality is acceptable
    shape_accepted = False
    if shape_result is not None:
        _curv_corr = shape_result.get('curvature_correlation', 0)
        _residual = shape_result.get('residual_m', float('inf'))
        # Thresholds: curvature correlation >= 0.4 AND residual < 150m
        shape_accepted = (_curv_corr >= 0.4 and _residual < 150)
        if not shape_accepted and verbose:
            print(f"    ⚠ Shape alignment REJECTED "
                  f"(curv_corr={_curv_corr:.3f}, residual={_residual:.1f}m)")
            print(f"    → Aligned coordinates will NOT be written. "
                  f"Distance-based alignment still active.")

    if shape_accepted and has_game_xy:
        game_xy_raw = np.column_stack([
            aligned['game_world_x'].values,
            aligned['game_world_y'].values,
        ])
        game_xy_transformed = apply_transform(
            game_xy_raw, shape_result['transform']
        )
        aligned['game_world_x_aligned'] = game_xy_transformed[:, 0]
        aligned['game_world_y_aligned'] = game_xy_transformed[:, 1]

        # Also provide real in its own system (identity, for completeness)
        if has_real_xy:
            aligned['real_world_x_aligned'] = aligned['real_world_x']
            aligned['real_world_y_aligned'] = aligned['real_world_y']

    # ---- Alignment quality metrics ----
    quality = _compute_alignment_quality(
        final_pairs, game_length, game_speed, real_speed,
        game_anch_d, real_anch_d, shape_result,
        game_speed_s=game_speed_s, real_speed_s=real_speed_s
    )

    # Recompute confidence with shape_accepted gating
    quality['confidence'] = _compute_confidence(
        quality['anchor_density'],
        quality['max_gap'],
        game_length,
        quality['coverage_pct'],
        quality['mean_correlation'],
        quality['shape_error_m'],
        quality['curvature_correlation'],
        shape_accepted=shape_accepted
    )

    aligned.attrs['anchor_pairs'] = final_pairs
    aligned.attrs['game_anch_d'] = game_anch_d
    aligned.attrs['real_anch_d'] = real_anch_d
    aligned.attrs['pass1_count'] = len(anchor_pairs)
    aligned.attrs['pass2_count'] = len(new_anchors)
    aligned.attrs['quality'] = quality
    aligned.attrs['shape_result'] = shape_result
    aligned.attrs['shape_accepted'] = shape_accepted

    if verbose:
        print(f"\n  Alignment quality:")
        print(f"    Anchor density: {quality['anchor_density']:.1f} "
              f"per 1000m")
        print(f"    Max gap: {quality['max_gap']:.0f}m")
        print(f"    Mean correlation: {quality['mean_correlation']:.2f}")
        print(f"    Coverage: {quality['coverage_pct']:.0f}%")
        if quality['shape_error_m'] is not None:
            print(f"    Shape error: {quality['shape_error_m']:.1f}m")
            print(f"    Curvature corr: "
                  f"{quality['curvature_correlation']:.3f}")
        print(f"    Confidence: {quality['confidence']:.2f}")

    return aligned


# ============================================================
# Quality metrics
# ============================================================

def _compute_alignment_quality(final_pairs, game_length,
                                game_speed, real_speed,
                                game_anch_d, real_anch_d,
                                shape_result=None,
                                game_speed_s=None, real_speed_s=None):
    """
    Compute alignment quality metrics.

    Includes:
      - anchor_density: anchors per 1000m
      - max_gap: largest gap between anchors
      - coverage_pct: % of track within 200m of an anchor
      - mean_correlation: average speed correlation at anchors
      - shape_error_m: mean residual from shape alignment (meters)
      - curvature_correlation: curvature profile match quality
      - confidence: composite 0-1 score
    """
    # Anchor density
    n_anchors = len(final_pairs)
    density = n_anchors / (game_length / 1000)

    # Max gap between anchors
    sorted_dists = sorted(a[0] * game_length for a in final_pairs)
    gaps = [sorted_dists[i+1] - sorted_dists[i]
            for i in range(len(sorted_dists) - 1)]
    max_gap = max(gaps) if gaps else game_length

    # Coverage: percentage of track within 200m of an anchor
    anchor_positions = np.array(sorted_dists)
    covered = 0
    for d in np.arange(0, game_length, 10):
        min_dist = np.min(np.abs(anchor_positions - d))
        if min_dist < 200:
            covered += 10
    coverage_pct = (covered / game_length) * 100

    # Local speed correlation at anchor points
    correlations = []
    game_spd_s = game_speed_s if game_speed_s is not None else smooth(game_speed, 15)
    real_spd_s = real_speed_s if real_speed_s is not None else smooth(real_speed, 15)

    for gd, rd in zip(game_anch_d[1:-1], real_anch_d[1:-1]):
        gi = int(np.clip(gd, 0, len(game_spd_s) - 1))
        ri = int(np.clip(rd, 0, len(real_spd_s) - 1))

        g_window = game_spd_s[max(0, gi-30):min(len(game_spd_s), gi+30)]
        r_window = real_spd_s[max(0, ri-30):min(len(real_spd_s), ri+30)]

        if len(g_window) > 10 and len(r_window) > 10:
            min_len = min(len(g_window), len(r_window))
            g_w = g_window[:min_len] - np.mean(g_window[:min_len])
            r_w = r_window[:min_len] - np.mean(r_window[:min_len])
            g_std = np.std(g_w)
            r_std = np.std(r_w)
            if g_std > 1 and r_std > 1:
                corr = float(np.mean(g_w / g_std * r_w / r_std))
                correlations.append(corr)

    mean_corr = np.mean(correlations) if correlations else 0.0

    # Shape alignment metrics
    shape_error_m = None
    curv_corr = None
    if shape_result is not None:
        shape_error_m = shape_result['residual_m']
        curv_corr = shape_result['curvature_correlation']

    # Composite confidence score (0-1)
    confidence = _compute_confidence(
        density, max_gap, game_length, coverage_pct,
        mean_corr, shape_error_m, curv_corr
    )

    return {
        'n_anchors': n_anchors,
        'anchor_density': density,
        'max_gap': max_gap,
        'coverage_pct': coverage_pct,
        'mean_correlation': mean_corr,
        'shape_error_m': shape_error_m,
        'curvature_correlation': curv_corr,
        'confidence': confidence,
    }


def _compute_confidence(density, max_gap, _game_length, coverage_pct,
                         mean_corr, shape_error_m, curv_corr,
                         shape_accepted=True):
    """
    Compute composite confidence score (0.0 - 1.0).

    Weights:
      - Anchor density: 20%
      - Gap coverage: 20%
      - Speed correlation: 30%
      - Shape alignment: 30% (if available, else redistributed)
    """
    # Density score: 5+ per 1000m = perfect
    s_density = min(density / 5.0, 1.0)

    # Gap score: max_gap < 200m = perfect, > 800m = 0
    s_gap = max(0, 1.0 - (max_gap - 200) / 600)

    # Coverage score
    s_coverage = coverage_pct / 100.0

    # Correlation score: 0.7+ = perfect
    s_corr = min(max(mean_corr, 0) / 0.7, 1.0)

    if shape_accepted and shape_error_m is not None and curv_corr is not None:
        # Shape error score: < 20m = perfect, > 100m = 0
        s_shape = max(0, 1.0 - (shape_error_m - 20) / 80)
        # Curvature correlation: 0.8+ = perfect
        s_curv = min(max(curv_corr, 0) / 0.8, 1.0)

        confidence = (0.10 * s_density + 0.10 * s_gap +
                      0.10 * s_coverage + 0.25 * s_corr +
                      0.20 * s_shape + 0.25 * s_curv)
    else:
        # No shape data — redistribute weight
        confidence = (0.20 * s_density + 0.15 * s_gap +
                      0.20 * s_coverage + 0.45 * s_corr)

    return float(np.clip(confidence, 0, 1))
