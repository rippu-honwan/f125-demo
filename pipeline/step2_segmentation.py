#!/usr/bin/env python3
"""
Step 2: Geometric track segmentation + per-lap phase detection.

Part A — Curvature-based segmentation with cross-lap consensus.
Part B — Per-lap, per-segment corner phase markers.

Config-driven: compound zones loaded from tracks/{track}.json.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from src.track_registry import get_gp_name, get_output_prefix, get_compound_zones
from src.plotting import COLORS, style_axis, save_figure
from config import OUTPUT_DIR

# ============================================================
# Configuration
# ============================================================
CURVATURE_SIGMA = 5          # Gaussian sigma in samples (= 10m effective)
CURVATURE_THRESHOLD = 0.0008  # m^-1 — radius < 1250m
MERGE_GAP_M = 30             # merge segments closer than this
MIN_SEGMENT_LENGTH_M = 20    # discard shorter segments
CLUSTER_TOLERANCE_M = 25     # group boundary points within this
SUPPORT_RATIO_MIN = 0.7      # minimum fraction of laps agreeing
IQR_MAX_M = 20               # maximum IQR for stable boundary


# ============================================================
# Argument parsing
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 2: Geometric segmentation + phase detection."
    )
    parser.add_argument("--track", required=True, type=str,
                        help="Track short name (e.g. suzuka, monza)")
    return parser.parse_args()


# ============================================================
# Part A1–A2: Per-lap curvature + segment detection
# ============================================================

def compute_curvature(x, y, sigma=CURVATURE_SIGMA):
    """Compute smoothed signed curvature from XY trajectory."""
    x_s = gaussian_filter1d(x, sigma=sigma)
    y_s = gaussian_filter1d(y, sigma=sigma)

    dx = np.gradient(x_s)
    dy = np.gradient(y_s)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    denom = (dx**2 + dy**2)**1.5
    denom = np.maximum(denom, 1e-12)
    kappa = (dx * ddy - dy * ddx) / denom
    return kappa


def find_segments_for_lap(distance, kappa_mag, threshold=CURVATURE_THRESHOLD,
                          merge_gap=MERGE_GAP_M, min_length=MIN_SEGMENT_LENGTH_M):
    """
    Find corner segments where curvature exceeds threshold.
    Returns list of (start_m, end_m, mean_kappa, direction).
    """
    above = kappa_mag > threshold
    segments = []

    # Find contiguous regions above threshold
    in_segment = False
    seg_start = 0
    for i in range(len(above)):
        if above[i] and not in_segment:
            seg_start = i
            in_segment = True
        elif not above[i] and in_segment:
            segments.append((seg_start, i - 1))
            in_segment = False
    if in_segment:
        segments.append((seg_start, len(above) - 1))

    # Convert to distance and compute stats
    raw_segs = []
    for s, e in segments:
        start_m = float(distance[s])
        end_m = float(distance[e])
        raw_segs.append((start_m, end_m))

    # Merge adjacent segments
    merged = []
    for seg in raw_segs:
        if merged and (seg[0] - merged[-1][1]) < merge_gap:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(list(seg))

    # Filter by minimum length and compute stats
    result = []
    for start_m, end_m in merged:
        length = end_m - start_m
        if length < min_length:
            continue
        mask = (distance >= start_m) & (distance <= end_m)
        mean_k = float(np.mean(kappa_mag[mask]))
        result.append({
            'start_m': start_m,
            'end_m': end_m,
            'length_m': length,
            'mean_kappa': mean_k,
        })

    return result


def compute_all_lap_segments(df):
    """Compute curvature and segments for every lap. Returns dict of results."""
    lap_ids = df['lap_id'].unique()
    all_segments = {}
    all_curvatures = {}

    for lap_id in lap_ids:
        lap = df[df['lap_id'] == lap_id].sort_values('distance_m')
        x = lap['x'].values
        y = lap['y'].values
        dist = lap['distance_m'].values

        kappa = compute_curvature(x, y)
        kappa_mag = np.abs(kappa)

        # Determine dominant direction per segment
        segs = find_segments_for_lap(dist, kappa_mag)
        for seg in segs:
            mask = (dist >= seg['start_m']) & (dist <= seg['end_m'])
            mean_signed = float(np.mean(kappa[mask]))
            seg['dominant_direction'] = 'left' if mean_signed > 0 else 'right'

        all_segments[lap_id] = segs
        all_curvatures[lap_id] = (dist, kappa_mag, kappa)

    return all_segments, all_curvatures


# ============================================================
# Part A3: Cross-lap consensus
# ============================================================

def cluster_boundaries(values, tolerance=CLUSTER_TOLERANCE_M):
    """Group boundary values within tolerance into clusters."""
    if not values:
        return []
    values = sorted(values)
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] <= tolerance:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return clusters


def build_consensus_segments(all_segments, n_laps):
    """
    Cluster segment boundaries across laps and build consensus list.
    """
    # Collect all start and end points
    starts = []
    ends = []
    for lap_id, segs in all_segments.items():
        for seg in segs:
            starts.append(seg['start_m'])
            ends.append(seg['end_m'])

    # Cluster starts and ends
    start_clusters = cluster_boundaries(starts)
    end_clusters = cluster_boundaries(ends)

    # Build stable start points
    stable_starts = []
    for cluster in start_clusters:
        support = len(cluster)
        ratio = support / n_laps
        iqr = float(np.percentile(cluster, 75) - np.percentile(cluster, 25)) if len(cluster) > 1 else 0
        median = float(np.median(cluster))
        if ratio >= SUPPORT_RATIO_MIN and iqr <= IQR_MAX_M:
            stable_starts.append({
                'median_m': median, 'iqr_m': iqr,
                'support_count': support, 'support_ratio': ratio
            })

    # Build stable end points
    stable_ends = []
    for cluster in end_clusters:
        support = len(cluster)
        ratio = support / n_laps
        iqr = float(np.percentile(cluster, 75) - np.percentile(cluster, 25)) if len(cluster) > 1 else 0
        median = float(np.median(cluster))
        if ratio >= SUPPORT_RATIO_MIN and iqr <= IQR_MAX_M:
            stable_ends.append({
                'median_m': median, 'iqr_m': iqr,
                'support_count': support, 'support_ratio': ratio
            })

    # Pair starts with nearest ends (greedy, monotonic)
    stable_starts.sort(key=lambda s: s['median_m'])
    stable_ends.sort(key=lambda s: s['median_m'])

    segments = []
    end_idx = 0
    for start in stable_starts:
        # Find first end that is after this start
        best_end = None
        for ei in range(end_idx, len(stable_ends)):
            if stable_ends[ei]['median_m'] > start['median_m'] + MIN_SEGMENT_LENGTH_M:
                best_end = stable_ends[ei]
                end_idx = ei + 1
                break
        if best_end is None:
            continue
        segments.append({
            'consensus_start_m': start['median_m'],
            'consensus_end_m': best_end['median_m'],
            'start_iqr_m': start['iqr_m'],
            'end_iqr_m': best_end['iqr_m'],
            'support_ratio': min(start['support_ratio'], best_end['support_ratio']),
        })

    return segments


# ============================================================
# Part A4: Merge compound corners + label assignment
# ============================================================

def merge_and_label_segments(segments, all_segments, all_curvatures, compound_zones):
    """Apply compound merging rules and assign labels."""

    # Merge compound zones (config-driven)
    for zone in compound_zones:
        zone_label = zone["label"]
        zone_start = zone["start_m"]
        zone_end = zone["end_m"]
        inside = [s for s in segments
                  if s['consensus_start_m'] >= zone_start - 50
                  and s['consensus_end_m'] <= zone_end + 50]
        if len(inside) >= 2:
            merged_start = min(s['consensus_start_m'] for s in inside)
            merged_end = max(s['consensus_end_m'] for s in inside)
            merged_start_iqr = min(s['start_iqr_m'] for s in inside)
            merged_end_iqr = min(s['end_iqr_m'] for s in inside)
            merged_support = min(s['support_ratio'] for s in inside)

            # Remove individual segments
            segments = [s for s in segments if s not in inside]
            # Add merged
            segments.append({
                'consensus_start_m': merged_start,
                'consensus_end_m': merged_end,
                'start_iqr_m': merged_start_iqr,
                'end_iqr_m': merged_end_iqr,
                'support_ratio': merged_support,
                'complex': zone_label,
            })

    # Sort by start distance
    segments.sort(key=lambda s: s['consensus_start_m'])

    # Assign labels and compute mean curvature / direction
    for i, seg in enumerate(segments):
        seg['segment_id'] = i + 1
        if 'complex' not in seg:
            seg['complex'] = None

        # If complex already set, use that as label; otherwise use segment_N
        if seg['complex']:
            seg['label'] = seg['complex']
        else:
            seg['label'] = f"segment_{i+1}"

        # Check if segment falls within a compound zone for labelling
        mid = (seg['consensus_start_m'] + seg['consensus_end_m']) / 2
        for zone in compound_zones:
            if zone["start_m"] <= mid <= zone["end_m"]:
                seg['label'] = zone["label"]
                break

        # Compute mean kappa and direction from all laps
        kappas = []
        signed_kappas = []
        for lap_id, (dist, kappa_mag, kappa_signed) in all_curvatures.items():
            mask = (dist >= seg['consensus_start_m']) & (dist <= seg['consensus_end_m'])
            if mask.sum() > 0:
                kappas.append(float(np.mean(kappa_mag[mask])))
                signed_kappas.append(float(np.mean(kappa_signed[mask])))

        seg['mean_kappa'] = float(np.median(kappas)) if kappas else 0.0
        mean_signed = float(np.median(signed_kappas)) if signed_kappas else 0.0

        if seg['complex']:
            seg['dominant_direction'] = 'complex'
        else:
            seg['dominant_direction'] = 'left' if mean_signed > 0 else 'right'

    return segments


# ============================================================
# Part B: Per-lap phase detection
# ============================================================

def detect_brake_start(lap_data, seg_start, approach_window=200):
    """
    Find brake application start point.
    Returns (distance_m, source) or (None, None).
    """
    dist = lap_data['distance_m'].values
    brake = lap_data['brake'].values
    speed = lap_data['speed'].values

    # Search window: seg_start - 200m to seg_start
    w_start = seg_start - approach_window
    w_end = seg_start
    mask = (dist >= w_start) & (dist <= w_end)
    idx_range = np.where(mask)[0]

    if len(idx_range) < 5:
        return None, None

    # Method 1: brake signal transition 0 -> 1
    brake_window = brake[idx_range]
    for i in range(1, len(brake_window)):
        if brake_window[i] > 0.5 and brake_window[i - 1] < 0.5:
            return float(dist[idx_range[i]]), "boolean"

    # Method 2: deceleration > 8 m/s^2
    speed_ms = speed[idx_range] / 3.6
    dist_window = dist[idx_range]
    if len(speed_ms) > 3:
        decel = -np.gradient(speed_ms, dist_window)
        for i in range(len(decel)):
            if decel[i] > 8.0:
                return float(dist_window[i]), "decel_gradient"

    return None, None


def detect_turn_in(dist, kappa_mag, seg_start, threshold=0.0015, window=50):
    """Find first point where curvature crosses threshold from below."""
    w_start = seg_start - window
    w_end = seg_start + window
    mask = (dist >= w_start) & (dist <= w_end)
    idx_range = np.where(mask)[0]

    if len(idx_range) < 3:
        return None

    k_window = kappa_mag[idx_range]
    for i in range(1, len(k_window)):
        if k_window[i] >= threshold and k_window[i - 1] < threshold:
            return float(dist[idx_range[i]])

    return None


def detect_apex_zone(dist, kappa_mag, speed, seg_start, seg_end):
    """
    Find apex zone where curvature is within 80% of peak.
    Also find speed minimum. Returns dict of markers.
    """
    mask = (dist >= seg_start) & (dist <= seg_end)
    idx_range = np.where(mask)[0]

    if len(idx_range) < 3:
        return None, None, None

    k_seg = kappa_mag[idx_range]
    peak_k = k_seg.max()
    threshold_k = peak_k * 0.8

    # Find contiguous region above 80% of peak
    above = k_seg >= threshold_k
    first_above = None
    last_above = None
    for i in range(len(above)):
        if above[i]:
            if first_above is None:
                first_above = i
            last_above = i

    if first_above is None:
        return None, None, None

    apex_start = float(dist[idx_range[first_above]])
    apex_end = float(dist[idx_range[last_above]])

    # Speed minimum within segment
    speed_seg = speed[idx_range]
    speed_min_local_idx = int(np.argmin(speed_seg))
    speed_min_m = float(dist[idx_range[speed_min_local_idx]])

    # Expand apex zone to include speed minimum if outside
    if speed_min_m < apex_start:
        apex_start = speed_min_m
    if speed_min_m > apex_end:
        apex_end = speed_min_m

    return apex_start, apex_end, speed_min_m


def detect_exit_end(dist, throttle, apex_end, seg_end, sustain_m=30):
    """Find first point after apex where throttle > 0.8 sustained for 30m."""
    search_start = apex_end
    search_end = seg_end + 100
    mask = (dist >= search_start) & (dist <= search_end)
    idx_range = np.where(mask)[0]

    if len(idx_range) < 5:
        return seg_end + 50

    thr_window = throttle[idx_range]
    dist_window = dist[idx_range]

    for i in range(len(thr_window)):
        if thr_window[i] > 0.8:
            start_d = dist_window[i]
            sustained = True
            for j in range(i, len(thr_window)):
                if dist_window[j] - start_d > sustain_m:
                    break
                if thr_window[j] < 0.8:
                    sustained = False
                    break
            if sustained:
                return float(dist_window[i])

    return float(seg_end + 50)


def run_phase_detection(df, segments, all_curvatures):
    """Run phase detection for all laps and segments."""
    records = []
    lap_ids = df['lap_id'].unique()

    for lap_id in lap_ids:
        lap_data = df[df['lap_id'] == lap_id].sort_values('distance_m')
        dist = lap_data['distance_m'].values
        speed = lap_data['speed'].values
        throttle = lap_data['throttle'].values

        _, kappa_mag, _ = all_curvatures[lap_id]

        for seg in segments:
            seg_start = seg['consensus_start_m']
            seg_end = seg['consensus_end_m']

            brake_m, brake_src = detect_brake_start(lap_data, seg_start)
            turn_in_m = detect_turn_in(dist, kappa_mag, seg_start)
            apex_start, apex_end, speed_min_m = detect_apex_zone(
                dist, kappa_mag, speed, seg_start, seg_end
            )

            exit_m = None
            if apex_end is not None:
                exit_m = detect_exit_end(dist, throttle, apex_end, seg_end)

            notes = ""
            if brake_src is None:
                notes = "no brake detected"
            if apex_start is None:
                notes += "; no apex zone" if notes else "no apex zone"

            records.append({
                'lap_id': lap_id,
                'segment_id': seg['segment_id'],
                'label': seg['label'],
                'brake_start_m': brake_m,
                'turn_in_m': turn_in_m,
                'apex_zone_start_m': apex_start,
                'apex_zone_end_m': apex_end,
                'speed_minimum_m': speed_min_m,
                'exit_end_m': exit_m,
                'brake_source': brake_src,
                'detection_notes': notes if notes else None,
            })

    return pd.DataFrame(records)


# ============================================================
# Visualization
# ============================================================

def plot_segmentation_overview(df, segments, all_curvatures, output_path,
                               gp_name, year, session_type):
    """Plot speed + curvature with segment shading."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(28, 12),
                                    gridspec_kw={'height_ratios': [2, 1]})
    fig.set_facecolor(COLORS['bg'])

    # Compute median speed and curvature across all laps
    pivot_speed = df.pivot_table(index='distance_m', values='speed', aggfunc='median')
    dist_axis = pivot_speed.index.values
    median_speed = pivot_speed['speed'].values

    # Median curvature magnitude
    all_kappa = []
    for lap_id, (dist, kappa_mag, _) in all_curvatures.items():
        all_kappa.append(kappa_mag)
    median_kappa = np.median(np.array(all_kappa), axis=0)
    kappa_dist = list(all_curvatures.values())[0][0]

    # Segment colors (alternating)
    seg_colors = plt.cm.Set3(np.linspace(0, 1, max(len(segments), 1)))

    # Top panel: speed
    style_axis(ax1, ylabel='Speed (km/h)',
               title=f'{gp_name} {year} {session_type} — Geometric Segmentation Overview')
    ax1.plot(dist_axis, median_speed, color=COLORS['game'], lw=1.5)

    for i, seg in enumerate(segments):
        ax1.axvspan(seg['consensus_start_m'], seg['consensus_end_m'],
                    alpha=0.2, color=seg_colors[i])
        mid = (seg['consensus_start_m'] + seg['consensus_end_m']) / 2
        ax1.text(mid, ax1.get_ylim()[1] if ax1.get_ylim()[1] else 350,
                 seg['label'], ha='center', va='bottom',
                 fontsize=7, color='white', rotation=45,
                 fontweight='bold')

    # Bottom panel: curvature
    style_axis(ax2, ylabel='|Curvature| (m^-1)', xlabel='Distance (m)')
    ax2.plot(kappa_dist, median_kappa, color='#ffaa00', lw=1.2)
    ax2.axhline(CURVATURE_THRESHOLD, color='#ff4444', ls='--', lw=0.8,
                alpha=0.6, label=f'threshold={CURVATURE_THRESHOLD}')

    for i, seg in enumerate(segments):
        ax2.axvline(seg['consensus_start_m'], color=seg_colors[i],
                    lw=1, ls='-', alpha=0.7)
        ax2.axvline(seg['consensus_end_m'], color=seg_colors[i],
                    lw=1, ls='--', alpha=0.5)

    ax2.legend(loc='upper right', facecolor=COLORS['card_bg'],
               edgecolor='white', labelcolor='white', fontsize=8)
    ax2.set_ylim(0, None)

    plt.tight_layout()
    save_figure(fig, output_path, dpi=200)
    print(f"  Overview plot saved: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    gp_name = get_gp_name(args.track)
    prefix = get_output_prefix(args.track)
    compound_zones = get_compound_zones(args.track)

    print("=" * 60)
    print("  Step 2: Geometric Segmentation + Phase Detection")
    print(f"  {gp_name} — Reference Lap Pool")
    print("=" * 60)

    # Load data
    print("\n  Loading reference laps...")
    df = pd.read_parquet(OUTPUT_DIR / f"{prefix}_reference_laps.parquet")
    with open(OUTPUT_DIR / f"{prefix}_reference_laps_metadata.json") as f:
        meta = json.load(f)

    n_laps = df['lap_id'].nunique()
    track_length = meta['track_length_m']
    print(f"  Laps: {n_laps}, Track: {track_length:.0f}m")

    if compound_zones:
        print(f"  Compound zones: {len(compound_zones)}")
        for z in compound_zones:
            print(f"    {z['label']}: {z['start_m']}–{z['end_m']}m")
    else:
        print("  Compound zones: none defined")

    # Part A1-A2: Per-lap curvature + segments
    print("\n  [A1-A2] Computing per-lap curvature and segments...")
    all_segments, all_curvatures = compute_all_lap_segments(df)

    total_segs = sum(len(s) for s in all_segments.values())
    print(f"  Total segments across all laps: {total_segs}")
    print(f"  Mean segments per lap: {total_segs / n_laps:.1f}")

    # Part A3: Consensus
    print("\n  [A3] Building cross-lap consensus...")
    consensus = build_consensus_segments(all_segments, n_laps)
    print(f"  Raw consensus segments: {len(consensus)}")

    # Part A4: Merge + label
    print("\n  [A4] Merging compound corners and assigning labels...")
    segments = merge_and_label_segments(consensus, all_segments, all_curvatures,
                                        compound_zones)
    print(f"  Final segments: {len(segments)}")

    for seg in segments:
        print(f"    {seg['segment_id']:>2}. {seg['label']:<20} "
              f"{seg['consensus_start_m']:>6.0f}–{seg['consensus_end_m']:<6.0f}m  "
              f"kappa={seg['mean_kappa']:.4f}  dir={seg['dominant_direction']}")

    # Save consensus JSON
    consensus_path = OUTPUT_DIR / f"{prefix}_segments_consensus.json"
    with open(consensus_path, 'w', encoding='utf-8') as f:
        json.dump({"segments": segments}, f, indent=2, ensure_ascii=False)
    print(f"\n  Consensus JSON: {consensus_path}")

    # Part B: Phase detection
    print("\n  [B] Running per-lap phase detection...")
    phase_df = run_phase_detection(df, segments, all_curvatures)
    phase_path = OUTPUT_DIR / f"{prefix}_phase_detection_per_lap.parquet"
    phase_df.to_parquet(phase_path, index=False)
    print(f"  Phase parquet: {phase_path} ({len(phase_df)} rows)")

    # Report: brake detection rate per segment
    print("\n  Brake detection rate per segment:")
    for seg in segments:
        seg_phase = phase_df[phase_df['segment_id'] == seg['segment_id']]
        brake_pct = seg_phase['brake_start_m'].notna().mean() * 100
        print(f"    {seg['label']:<20} {brake_pct:>5.1f}%")

    # Report: wide apex zones
    print("\n  Segments with apex_zone width > 30m:")
    for seg in segments:
        seg_phase = phase_df[phase_df['segment_id'] == seg['segment_id']]
        widths = (seg_phase['apex_zone_end_m'] - seg_phase['apex_zone_start_m']).dropna()
        if len(widths) > 0:
            median_w = widths.median()
            if median_w > 30:
                print(f"    {seg['label']:<20} median width={median_w:.0f}m")

    # Visualization
    print("\n  [Viz] Generating overview plot...")
    plot_path = OUTPUT_DIR / f"{prefix}_segmentation_overview.png"
    # Extract year from metadata session string
    session_str = meta.get('session', '')
    year = session_str.split()[0] if session_str else ''
    session_type = session_str.split('-')[-1].strip() if '-' in session_str else 'Q'
    plot_segmentation_overview(df, segments, all_curvatures, plot_path,
                               gp_name, year, session_type)

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
