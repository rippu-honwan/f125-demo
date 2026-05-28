#!/usr/bin/env python3
"""
Step 4: Fix S1–S4 sub-apex positions and HP-A/200R/FIN missing corners.

Part A: Derive S1–S4 from curvature local maxima (not even division).
Part B: Derive HP-A, 200R, FIN from speed-profile + throttle transitions.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from src.track import load_track, TRACKS_DIR
from src.plotting import COLORS, style_axis, save_figure
from config import OUTPUT_DIR

# IDs to fix
ESSES_IDS = [3, 4, 5, 6]  # S1, S2, S3, S4
MISSING_IDS = [10, 12, 18]  # HP-A, 200R, FIN
ALL_FIX_IDS = set(ESSES_IDS + MISSING_IDS)

# Search windows for missing corners
MISSING_WINDOWS = {
    10: (2600, 2840),   # HP-A
    12: (3050, 3350),   # 200R
    18: (5500, 5760),   # FIN
}


def compute_kappa(x, y, sigma=5):
    """Compute signed curvature with Gaussian smoothing."""
    x_s = gaussian_filter1d(x, sigma=sigma)
    y_s = gaussian_filter1d(y, sigma=sigma)
    dx = np.gradient(x_s)
    dy = np.gradient(y_s)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = np.maximum((dx**2 + dy**2)**1.5, 1e-12)
    return (dx * ddy - dy * ddx) / denom


# ============================================================
# Part A: Fix Esses sub-apexes
# ============================================================

def fix_esses(df):
    """Detect S1–S4 sub-apex positions from curvature peaks."""
    print("\n  [Part A] Fixing Esses sub-apexes (S1–S4)...")

    # A1: Compute median curvature in esses window (950–1700m)
    lap_ids = df['lap_id'].unique()
    esses_window = (950, 1700)
    step = 2
    dist_axis = np.arange(esses_window[0], esses_window[1], step)

    all_kappa_mag = []
    all_speed = []
    for lap_id in lap_ids:
        lap = df[df['lap_id'] == lap_id].sort_values('distance_m')
        mask = (lap['distance_m'] >= esses_window[0]) & (lap['distance_m'] < esses_window[1])
        seg = lap[mask]
        if len(seg) < 50:
            continue
        x = seg['x'].values
        y = seg['y'].values
        kappa = compute_kappa(x, y)
        kappa_mag = np.abs(kappa)
        # Interpolate to common axis
        seg_dist = seg['distance_m'].values
        kappa_interp = np.interp(dist_axis, seg_dist, kappa_mag)
        speed_interp = np.interp(dist_axis, seg_dist, seg['speed'].values)
        all_kappa_mag.append(kappa_interp)
        all_speed.append(speed_interp)

    median_kappa = np.median(np.array(all_kappa_mag), axis=0)
    median_speed = np.median(np.array(all_speed), axis=0)

    # A2: Find peaks
    peaks, props = find_peaks(median_kappa, prominence=0.0003,
                              distance=50, height=0.0005)
    print(f"    Peaks found (prom>=0.0003): {len(peaks)}")

    if len(peaks) < 4:
        peaks, props = find_peaks(median_kappa, prominence=0.0002,
                                  distance=50, height=0.0005)
        print(f"    Retry (prom>=0.0002): {len(peaks)}")

    if len(peaks) > 4:
        # Keep 4 most prominent
        prom_order = np.argsort(props['prominences'])[::-1][:4]
        peaks = np.sort(peaks[prom_order])
        print(f"    Trimmed to 4 most prominent")

    n_found = len(peaks)
    print(f"    Final sub-apex peaks: {n_found}")

    # A3: Compute apex zones for each peak
    results = []
    for pi, peak_idx in enumerate(peaks):
        peak_dist = float(dist_axis[peak_idx])
        peak_val = float(median_kappa[peak_idx])
        threshold = peak_val * 0.8

        # Search left for zone start
        zone_start_idx = peak_idx
        for j in range(peak_idx - 1, -1, -1):
            if median_kappa[j] < threshold:
                zone_start_idx = j + 1
                break

        # Search right for zone end
        zone_end_idx = peak_idx
        for j in range(peak_idx + 1, len(median_kappa)):
            if median_kappa[j] < threshold:
                zone_end_idx = j - 1
                break

        zone_start_m = float(dist_axis[zone_start_idx])
        zone_end_m = float(dist_axis[zone_end_idx])
        zone_center_m = round((zone_start_m + zone_end_m) / 2, 1)

        # A4: Per-lap validation
        per_lap_centers = []
        for kappa_arr in all_kappa_mag:
            # Find nearest peak within ±60m of consensus
            search_start = max(0, peak_idx - 30)
            search_end = min(len(kappa_arr), peak_idx + 30)
            local_seg = kappa_arr[search_start:search_end]
            if len(local_seg) > 0:
                local_peak = search_start + int(np.argmax(local_seg))
                per_lap_centers.append(float(dist_axis[local_peak]))

        per_lap_arr = np.array(per_lap_centers)
        iqr = float(np.percentile(per_lap_arr, 75) - np.percentile(per_lap_arr, 25))
        support_ratio = len(per_lap_arr) / len(lap_ids)

        # Check if speed minimum is within ±30m
        speed_window = median_speed[max(0, peak_idx-15):min(len(median_speed), peak_idx+15)]
        speed_min_local = max(0, peak_idx-15) + int(np.argmin(speed_window))
        speed_min_m = float(dist_axis[speed_min_local])
        has_speed_min = abs(speed_min_m - zone_center_m) <= 30

        signals = ["curvature"]
        if has_speed_min:
            signals.append("speed_min")

        # Confidence scoring
        score = 0
        if support_ratio >= 0.80:
            score += 2
        elif support_ratio >= 0.65:
            score += 1
        if iqr <= 8:
            score += 2
        elif iqr <= 15:
            score += 1
        if len(signals) >= 2:
            score += 2
        elif len(signals) >= 1:
            score += 1
        # No brake for esses (high-speed, flat-out)

        conf = "high" if score >= 6 else "medium" if score >= 4 else "low"

        # Review flag
        requires_review = not (conf in ("medium", "high") and iqr <= 20)
        review_reason = None
        if requires_review:
            parts = []
            if iqr > 20:
                parts.append(f"apex IQR={iqr:.0f}m")
            if conf == "low":
                parts.append(f"confidence=low (score={score})")
            review_reason = "; ".join(parts) if parts else "sub-apex within esses_complex"

        results.append({
            'peak_dist_m': peak_dist,
            'apex_zone_start_m': round(zone_start_m, 1),
            'apex_zone_end_m': round(zone_end_m, 1),
            'apex_zone_center_m': zone_center_m,
            'apex_iqr_m': round(iqr, 1),
            'apex_support_ratio': round(support_ratio, 3),
            'speed_minimum_m': round(speed_min_m, 1),
            'confidence_score': score,
            'confidence': conf,
            'signals': signals,
            'requires_review': requires_review,
            'review_reason': review_reason,
        })

    return results, dist_axis, median_kappa, median_speed, all_kappa_mag, all_speed


# ============================================================
# Part B: Fix missing corners (HP-A, 200R, FIN)
# ============================================================

def fix_missing_corner(df, corner_id, window_start, window_end):
    """Detect corner from speed profile + throttle within given window."""
    lap_ids = df['lap_id'].unique()
    step = 2
    dist_axis = np.arange(window_start, window_end, step)

    all_speed = []
    all_throttle = []
    all_brake = []
    for lap_id in lap_ids:
        lap = df[df['lap_id'] == lap_id].sort_values('distance_m')
        mask = (lap['distance_m'] >= window_start) & (lap['distance_m'] < window_end)
        seg = lap[mask]
        if len(seg) < 10:
            continue
        seg_dist = seg['distance_m'].values
        all_speed.append(np.interp(dist_axis, seg_dist, seg['speed'].values))
        all_throttle.append(np.interp(dist_axis, seg_dist, seg['throttle'].values))
        all_brake.append(np.interp(dist_axis, seg_dist, seg['brake'].values))

    median_speed = np.median(np.array(all_speed), axis=0)
    median_throttle = np.median(np.array(all_throttle), axis=0)
    median_brake = np.median(np.array(all_brake), axis=0)

    # B1a: Speed minimum
    speed_min_idx = int(np.argmin(median_speed))
    speed_min_m = float(dist_axis[speed_min_idx])

    # Per-lap speed minimum IQR
    per_lap_mins = []
    for spd in all_speed:
        local_min_idx = int(np.argmin(spd))
        per_lap_mins.append(float(dist_axis[local_min_idx]))
    per_lap_arr = np.array(per_lap_mins)
    speed_iqr = float(np.percentile(per_lap_arr, 75) - np.percentile(per_lap_arr, 25))
    speed_support = len(per_lap_arr) / len(lap_ids)

    # B1b: Throttle transition (exit)
    exit_m = None
    for i in range(speed_min_idx, len(median_throttle)):
        if median_throttle[i] > 0.5:
            # Check sustained for 20 steps (40m)
            sustained = True
            for j in range(i, min(i + 20, len(median_throttle))):
                if median_throttle[j] < 0.5:
                    sustained = False
                    break
            if sustained:
                exit_m = float(dist_axis[i])
                break
    if exit_m is None:
        exit_m = float(dist_axis[-1])

    # B1c: Approach marker (where speed starts descending)
    approach_m = float(dist_axis[0])
    mid_speed = (median_speed[0] + median_speed[speed_min_idx]) / 2
    for i in range(speed_min_idx - 1, -1, -1):
        if median_speed[i] > mid_speed:
            approach_m = float(dist_axis[i])
            break

    # B1d: Apex zone
    if corner_id == 18:  # Final Chicane — check for double apex
        neg_speed = -median_speed
        peaks_fin, _ = find_peaks(neg_speed, prominence=5, distance=25)
        if len(peaks_fin) >= 2:
            # Double apex — use first minimum
            apex_center = float(dist_axis[peaks_fin[0]])
        else:
            apex_center = speed_min_m
        apex_start = apex_center - 15
        apex_end = apex_center + 15
    else:
        # HP-A and 200R: speed minimum ± 20m
        apex_center = speed_min_m
        apex_start = speed_min_m - 20
        apex_end = speed_min_m + 20

    # B1e: Brake detection
    brake_m = None
    brake_src = None
    brake_iqr = None
    brake_support = 0.0

    # Search in approach window
    search_start = max(0, int((approach_m - 200 - window_start) / step))
    search_end = max(0, int((approach_m - window_start) / step))
    if search_end > search_start:
        brake_window = median_brake[search_start:search_end]
        for i in range(1, len(brake_window)):
            if brake_window[i] > 0.5 and brake_window[i - 1] < 0.5:
                brake_m = float(dist_axis[search_start + i])
                brake_src = "boolean"
                break

    if brake_m is None:
        # Fallback: deceleration gradient
        speed_ms = median_speed / 3.6
        decel = -np.gradient(speed_ms, step)
        search_seg = decel[search_start:search_end] if search_end > search_start else decel
        for i in range(len(search_seg)):
            if search_seg[i] > 8.0:
                brake_m = float(dist_axis[search_start + i])
                brake_src = "decel_gradient"
                break

    # Per-lap brake detection for support ratio
    if brake_m is not None:
        brake_detections = []
        for brk in all_brake:
            detected = False
            for i in range(max(0, search_start), min(search_end, len(brk) - 1)):
                if brk[i + 1] > 0.5 and brk[i] < 0.5:
                    brake_detections.append(float(dist_axis[i]))
                    detected = True
                    break
            if not detected:
                pass
        brake_support = len(brake_detections) / len(lap_ids)
        if brake_detections:
            brake_arr = np.array(brake_detections)
            brake_iqr = float(np.percentile(brake_arr, 75) - np.percentile(brake_arr, 25))

    # Confidence scoring (max 5 for low-curvature corners)
    signals = []
    if brake_src:
        signals.append(brake_src)
    signals.append("speed_min")

    score = 0
    if speed_support >= 0.80:
        score += 2
    elif speed_support >= 0.65:
        score += 1
    if speed_iqr <= 8:
        score += 2
    elif speed_iqr <= 15:
        score += 1
    if len(signals) >= 2:
        score += 2
    elif len(signals) >= 1:
        score += 1
    if brake_support >= 0.70:
        score += 1

    # Adjusted thresholds for low-curvature corners
    conf = "medium" if score >= 3 else "low"

    requires_review = conf == "low" or speed_iqr > 15
    review_reason = None
    if requires_review:
        parts = []
        if conf == "low":
            parts.append(f"confidence=low (score={score})")
        if speed_iqr > 15:
            parts.append(f"speed IQR={speed_iqr:.0f}m")
        review_reason = "; ".join(parts) if parts else None

    return {
        'approach_start_m': round(approach_m, 1),
        'brake_start_m': round(brake_m, 1) if brake_m else None,
        'brake_start_iqr_m': round(brake_iqr, 1) if brake_iqr else None,
        'brake_start_support_ratio': round(brake_support, 3),
        'apex_zone_start_m': round(apex_start, 1),
        'apex_zone_end_m': round(apex_end, 1),
        'apex_zone_center_m': round(apex_center, 1),
        'apex_m': round(apex_center, 1),
        'apex_iqr_m': round(speed_iqr, 1),
        'apex_support_ratio': round(speed_support, 3),
        'speed_minimum_m': round(speed_min_m, 1),
        'exit_end_m': round(exit_m, 1),
        'confidence_score': score,
        'confidence': conf,
        'signals': signals,
        'requires_review': requires_review,
        'review_reason': review_reason,
        'detection_method': 'speed_profile_throttle_transition',
        # For plotting
        '_dist_axis': dist_axis,
        '_median_speed': median_speed,
        '_median_throttle': median_throttle,
        '_all_speed': all_speed,
    }


# ============================================================
# Part C: Validation plot
# ============================================================

def plot_validation(esses_results, esses_dist, esses_kappa, esses_speed,
                    esses_all_speed, missing_results, old_track, output_path):
    """Produce validation figure."""
    n_missing = len(missing_results)
    fig, axes = plt.subplots(1 + n_missing, 1, figsize=(28, 5 * (1 + n_missing)))
    fig.set_facecolor(COLORS['bg'])
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # Section 1: Esses
    ax = axes[0]
    style_axis(ax, ylabel='Speed / Curvature',
               title='Esses Sub-Apex Detection (S1–S4)')

    # All laps speed (thin)
    for spd in esses_all_speed[:30]:
        ax.plot(esses_dist, spd, lw=0.4, alpha=0.2, color=COLORS['game'])
    # Median speed
    ax.plot(esses_dist, esses_speed, lw=2, color='#4488ff', label='Median speed')

    # Curvature on secondary axis
    ax2 = ax.twinx()
    ax2.plot(esses_dist, esses_kappa, lw=1.5, color='#ffaa00', label='Median |κ|')
    ax2.set_ylabel('|Curvature| (m⁻¹)', color='#ffaa00', fontsize=9)
    ax2.tick_params(colors='#ffaa00')

    # New apex zones
    labels = ['S1', 'S2', 'S3', 'S4']
    colors_sub = ['#00ff88', '#00ccaa', '#00ff88', '#00ccaa']
    for i, res in enumerate(esses_results):
        ax.axvspan(res['apex_zone_start_m'], res['apex_zone_end_m'],
                   alpha=0.2, color=colors_sub[i % len(colors_sub)])
        ax.axvline(res['apex_zone_center_m'], color=colors_sub[i % len(colors_sub)],
                   ls='-', lw=1.5, alpha=0.8)
        ax.text(res['apex_zone_center_m'], ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] else 300,
                labels[i] if i < len(labels) else f'S{i+1}',
                ha='center', fontsize=10, color='white', fontweight='bold')

    # Old apex positions (red dashed)
    for cid in ESSES_IDS:
        old_c = old_track.get_corner(cid)
        if old_c:
            ax.axvline(old_c.apex_m, color=COLORS['real'], ls='--', lw=1, alpha=0.6)

    ax.legend(loc='upper left', facecolor=COLORS['card_bg'],
              labelcolor='white', edgecolor='white', fontsize=8)

    # Section 2: Missing corners
    corner_names = {10: "HP-A", 12: "200R", 18: "Final Chicane"}
    for idx, (cid, res) in enumerate(missing_results.items()):
        ax = axes[1 + idx]
        old_c = old_track.get_corner(cid)
        old_apex = old_c.apex_m if old_c else 0

        style_axis(ax, ylabel='Speed / Throttle',
                   title=f"{corner_names[cid]} — conf={res['confidence']}, "
                         f"IQR={res['apex_iqr_m']}m",
                   xlabel='Distance (m)')

        dist_ax = res['_dist_axis']
        # All laps speed
        for spd in res['_all_speed'][:30]:
            ax.plot(dist_ax, spd, lw=0.4, alpha=0.2, color=COLORS['game'])
        # Median speed
        ax.plot(dist_ax, res['_median_speed'], lw=2, color='#4488ff', label='Median speed')

        # Throttle on secondary
        ax_t = ax.twinx()
        ax_t.plot(dist_ax, res['_median_throttle'], lw=1.5, color='#ffaa00', label='Throttle')
        ax_t.set_ylabel('Throttle', color='#ffaa00', fontsize=9)
        ax_t.set_ylim(0, 1.1)
        ax_t.tick_params(colors='#ffaa00')

        # New apex zone (green)
        ax.axvspan(res['apex_zone_start_m'], res['apex_zone_end_m'],
                   alpha=0.25, color=COLORS['faster'])
        # Old apex (red dashed)
        ax.axvline(old_apex, color=COLORS['real'], ls='--', lw=1.5,
                   label=f'Old apex={old_apex:.0f}')
        # New center (blue dashed)
        ax.axvline(res['apex_zone_center_m'], color='#4488ff', ls='--', lw=1.5,
                   label=f'New center={res["apex_zone_center_m"]:.0f}')
        ax.legend(loc='upper right', facecolor=COLORS['card_bg'],
                  labelcolor='white', edgecolor='white', fontsize=8)

    plt.tight_layout()
    save_figure(fig, output_path, dpi=150)
    print(f"  ✓ Validation plot: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  Step 4: Fix Esses Sub-Apexes + Missing Corners")
    print("=" * 60)

    # Load data
    df = pd.read_parquet(OUTPUT_DIR / "suzuka_reference_laps.parquet")
    gt_path = TRACKS_DIR / "suzuka.ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)
    old_track = load_track("suzuka")

    # Hash unchanged corners for verification
    unchanged_ids = [c['id'] for c in gt_data['corners'] if c['id'] not in ALL_FIX_IDS]
    before_hashes = {}
    for c in gt_data['corners']:
        if c['id'] not in ALL_FIX_IDS:
            before_hashes[c['id']] = hashlib.sha256(
                json.dumps(c, sort_keys=True).encode()
            ).hexdigest()

    # Part A: Fix Esses
    esses_results, esses_dist, esses_kappa, esses_speed, \
        esses_all_kappa, esses_all_speed = fix_esses(df)

    # Map results to S1–S4
    esses_labels = ['S1', 'S2', 'S3', 'S4']
    print(f"\n    Sub-apex positions:")
    for i, res in enumerate(esses_results):
        label = esses_labels[i] if i < 4 else f"S{i+1}"
        old_c = old_track.get_corner(ESSES_IDS[i]) if i < len(ESSES_IDS) else None
        old_apex = old_c.apex_m if old_c else 0
        delta = abs(res['apex_zone_center_m'] - old_apex)
        print(f"      {label}: center={res['apex_zone_center_m']:.0f}m "
              f"(old={old_apex:.0f}, Δ={delta:.0f}m) "
              f"IQR={res['apex_iqr_m']:.0f}m conf={res['confidence']}")

    # Update S1–S4 in ground truth
    for i, cid in enumerate(ESSES_IDS):
        if i >= len(esses_results):
            break
        res = esses_results[i]
        for c in gt_data['corners']:
            if c['id'] == cid:
                c['apex_zone_start_m'] = res['apex_zone_start_m']
                c['apex_zone_end_m'] = res['apex_zone_end_m']
                c['apex_zone_center_m'] = res['apex_zone_center_m']
                c['apex_m'] = res['apex_zone_center_m']
                c['apex_iqr_m'] = res['apex_iqr_m']
                c['apex_support_ratio'] = res['apex_support_ratio']
                c['confidence_score'] = res['confidence_score']
                c['confidence'] = res['confidence']
                c['signals'] = res['signals']
                c['requires_review'] = res['requires_review']
                c['review_reason'] = res['review_reason']
                # Update entry_m/exit_m for compat
                c['entry_m'] = res['apex_zone_start_m']
                c['exit_m'] = res['apex_zone_end_m']
                break

    # Part B: Fix missing corners
    print("\n  [Part B] Fixing missing corners (HP-A, 200R, FIN)...")
    missing_results = {}
    for cid, (w_start, w_end) in MISSING_WINDOWS.items():
        res = fix_missing_corner(df, cid, w_start, w_end)
        missing_results[cid] = res
        old_c = old_track.get_corner(cid)
        old_apex = old_c.apex_m if old_c else 0
        delta = abs(res['apex_zone_center_m'] - old_apex)
        print(f"    id={cid} ({old_c.short if old_c else '?'}): "
              f"center={res['apex_zone_center_m']:.0f}m "
              f"(old={old_apex:.0f}, Δ={delta:.0f}m) "
              f"IQR={res['apex_iqr_m']:.0f}m conf={res['confidence']}")

    # Update HP-A, 200R, FIN in ground truth
    for cid, res in missing_results.items():
        for c in gt_data['corners']:
            if c['id'] == cid:
                c['approach_start_m'] = res['approach_start_m']
                c['brake_start_m'] = res['brake_start_m']
                c['brake_start_iqr_m'] = res['brake_start_iqr_m']
                c['brake_start_support_ratio'] = res['brake_start_support_ratio']
                c['apex_zone_start_m'] = res['apex_zone_start_m']
                c['apex_zone_end_m'] = res['apex_zone_end_m']
                c['apex_zone_center_m'] = res['apex_zone_center_m']
                c['apex_m'] = res['apex_m']
                c['apex_iqr_m'] = res['apex_iqr_m']
                c['apex_support_ratio'] = res['apex_support_ratio']
                c['speed_minimum_m'] = res['speed_minimum_m']
                c['exit_end_m'] = res['exit_end_m']
                c['confidence_score'] = res['confidence_score']
                c['confidence'] = res['confidence']
                c['signals'] = res['signals']
                c['requires_review'] = res['requires_review']
                c['review_reason'] = res['review_reason']
                c['detection_method'] = res['detection_method']
                c['source_laps'] = 71
                # Update entry_m/exit_m for compat
                c['entry_m'] = res['apex_zone_start_m']
                c['exit_m'] = res['apex_zone_end_m']
                break

    # Verify apex_m exists in all entries
    for c in gt_data['corners']:
        if 'apex_m' not in c or c['apex_m'] is None:
            raise RuntimeError(f"Corner {c['id']} missing apex_m — aborting")

    # Verify unchanged corners are unmodified
    after_hashes = {}
    for c in gt_data['corners']:
        if c['id'] not in ALL_FIX_IDS:
            after_hashes[c['id']] = hashlib.sha256(
                json.dumps(c, sort_keys=True).encode()
            ).hexdigest()

    for cid in unchanged_ids:
        if before_hashes[cid] != after_hashes[cid]:
            raise RuntimeError(f"Corner {cid} was modified unexpectedly — aborting")
    print(f"\n  ✓ All {len(unchanged_ids)} unchanged corners verified identical")

    # Write updated JSON
    with open(gt_path, 'w', encoding='utf-8') as f:
        json.dump(gt_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Updated: {gt_path}")

    # Validation plot
    print("\n  [Part C] Generating validation plot...")
    plot_path = OUTPUT_DIR / "suzuka_fix_validation.png"
    plot_validation(esses_results, esses_dist, esses_kappa, esses_speed,
                    esses_all_speed, missing_results, old_track, plot_path)

    # Final diff
    print("\n  === FINAL DIFF (5 fixed corners) ===")
    print(f"  {'Corner':<12} {'Old apex':>9} {'New apex':>9} {'Delta':>6} "
          f"{'IQR':>5} {'Conf':<8} {'Review'}")
    print(f"  {'-'*12} {'-'*9} {'-'*9} {'-'*6} {'-'*5} {'-'*8} {'-'*6}")

    all_fixed = list(ESSES_IDS) + MISSING_IDS
    for cid in all_fixed:
        for c in gt_data['corners']:
            if c['id'] == cid:
                old_c = old_track.get_corner(cid)
                old_apex = old_c.apex_m if old_c else 0
                delta = abs(c['apex_m'] - old_apex)
                iqr_str = f"{c['apex_iqr_m']:.0f}" if c['apex_iqr_m'] else "—"
                rev = "YES" if c['requires_review'] else "no"
                print(f"  {c['short']:<12} {old_apex:>9.0f} {c['apex_m']:>9.1f} "
                      f"{delta:>5.0f}m {iqr_str:>5} {c['confidence']:<8} {rev}")
                break

    # Overall confidence distribution
    high = sum(1 for c in gt_data['corners'] if c['confidence'] == 'high')
    med = sum(1 for c in gt_data['corners'] if c['confidence'] == 'medium')
    low = sum(1 for c in gt_data['corners'] if c['confidence'] == 'low')
    review_count = sum(1 for c in gt_data['corners'] if c['requires_review'])
    no_review = sum(1 for c in gt_data['corners'] if not c['requires_review'])

    print(f"\n  === OVERALL (all 18 corners) ===")
    print(f"  Confidence: {high} high, {med} medium, {low} low")
    print(f"  Requires review: {review_count} / No review needed: {no_review}")
    fixed_no_review = sum(1 for cid in all_fixed
                          for c in gt_data['corners']
                          if c['id'] == cid and not c['requires_review'])
    print(f"  Of 7 fixed corners: {fixed_no_review} now have requires_review=false")


if __name__ == "__main__":
    main()
