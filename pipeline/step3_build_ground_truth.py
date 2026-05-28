#!/usr/bin/env python3
"""
Step 3: Build ground_truth.json from aggregated phase detection.

Aggregates per-lap phase markers, scores confidence, flags corners
requiring human review, and produces the final ground truth file.

Config-driven: works with any track via --track argument.
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

from src.track_registry import get_gp_name, get_output_prefix, get_track_corners
from src.track import load_track, TRACKS_DIR
from src.plotting import COLORS, style_axis, save_figure
from config import OUTPUT_DIR

# ============================================================
# Corner mapping is built dynamically from the track JSON.
# No hardcoded corner names or complex labels.
# ============================================================

def build_corner_map(segments, old_track):
    """
    Build segment_id -> corner list mapping from existing track corners.
    Assigns each corner to the segment whose range contains its apex.
    Returns (corner_map, fallback_corners).
    """
    corner_map = {}  # segment_id -> [(cid, name, short, type, dir, complex)]
    assigned_ids = set()

    for seg in segments:
        seg_id = seg['segment_id']
        seg_start = seg['consensus_start_m']
        seg_end = seg['consensus_end_m']
        corner_map[seg_id] = []

        for corner in old_track.corners:
            if seg_start - 50 <= corner.apex_m <= seg_end + 50:
                # Read complex from track JSON data
                complex_name = None
                track_json = TRACKS_DIR / f"{old_track.short}.json"
                if track_json.exists():
                    with open(track_json) as f:
                        tdata = json.load(f)
                    for cj in tdata.get('corners', []):
                        if cj['id'] == corner.id:
                            complex_name = cj.get('complex', None)
                            break

                corner_map[seg_id].append((
                    corner.id, corner.name, corner.short,
                    corner.type, corner.direction, complex_name
                ))
                assigned_ids.add(corner.id)

    # Fallback: corners not assigned to any segment
    fallback_corners = []
    for corner in old_track.corners:
        if corner.id not in assigned_ids:
            fallback_corners.append((
                corner.id, corner.name, corner.short,
                corner.type, corner.direction, None
            ))

    return corner_map, fallback_corners


# ============================================================
# Argument parsing
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 3: Build ground truth from aggregated phase detection."
    )
    parser.add_argument("--track", required=True, type=str,
                        help="Track short name (e.g. suzuka, monza)")
    return parser.parse_args()


# ============================================================
# Step 1: Aggregate phase markers
# ============================================================

def aggregate_marker(values, n_laps):
    """Aggregate a single marker across laps with outlier removal."""
    vals = np.array([v for v in values if v is not None and not np.isnan(v)])
    if len(vals) == 0:
        return {'median_m': None, 'iqr_m': None, 'support_count': 0, 'support_ratio': 0.0}

    median = float(np.median(vals))
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1

    # Remove outliers > 1.5 * IQR from median
    if iqr > 0:
        lower = median - 1.5 * iqr
        upper = median + 1.5 * iqr
        filtered = vals[(vals >= lower) & (vals <= upper)]
        if len(filtered) > 0:
            median = float(np.median(filtered))
            q1f, q3f = np.percentile(filtered, [25, 75])
            iqr = q3f - q1f

    return {
        'median_m': round(median, 1),
        'iqr_m': round(iqr, 1),
        'support_count': int(len(vals)),
        'support_ratio': round(len(vals) / n_laps, 3),
    }


def aggregate_segment_phases(phase_df, segment_id, n_laps):
    """Aggregate all phase markers for one segment."""
    seg_data = phase_df[phase_df['segment_id'] == segment_id]

    result = {}
    for col in ['brake_start_m', 'turn_in_m', 'apex_zone_start_m',
                'apex_zone_end_m', 'speed_minimum_m', 'exit_end_m']:
        result[col] = aggregate_marker(seg_data[col].tolist(), n_laps)

    # Collect signal types used
    signals = set()
    brake_sources = seg_data['brake_source'].dropna().unique()
    for src in brake_sources:
        signals.add(src)
    if result['speed_minimum_m']['support_count'] > 0:
        signals.add('speed_min')
    if result['apex_zone_start_m']['support_count'] > 0:
        signals.add('curvature')

    result['signals'] = sorted(signals)
    result['source_laps'] = int(seg_data['lap_id'].nunique())
    return result


# ============================================================
# Step 2: Confidence scoring
# ============================================================

def compute_confidence(apex_support_ratio, apex_iqr_m, signal_count,
                       brake_support_ratio, source_laps):
    """Compute confidence score per the rubric (0-7 points)."""
    score = 0

    # Apex support ratio
    if apex_support_ratio >= 0.80:
        score += 2
    elif apex_support_ratio >= 0.65:
        score += 1

    # Apex IQR
    if apex_iqr_m is not None:
        if apex_iqr_m <= 8:
            score += 2
        elif apex_iqr_m <= 15:
            score += 1

    # Signal count
    if signal_count >= 2:
        score += 2
    elif signal_count >= 1:
        score += 1

    # Brake support
    if brake_support_ratio >= 0.70:
        score += 1

    # Override: fewer than 3 source laps -> low
    if source_laps < 3:
        return 0, "low"

    if score >= 6:
        label = "high"
    elif score >= 4:
        label = "medium"
    else:
        label = "low"

    return score, label


# ============================================================
# Step 3: Review flags
# ============================================================

def check_review_flags(corner, old_track):
    """Determine if corner requires human review. Returns (bool, reason)."""
    reasons = []

    if corner['confidence'] == 'low':
        reasons.append(f"confidence=low (score={corner['confidence_score']})")

    if corner['apex_iqr_m'] is not None and corner['apex_iqr_m'] > 15:
        reasons.append(f"apex IQR={corner['apex_iqr_m']:.0f}m — zone may be too wide")

    # Compare with old apex
    old_corner = old_track.get_corner(corner['id'])
    if old_corner:
        delta = abs(corner['apex_zone_center_m'] - old_corner.apex_m)
        if delta > 30:
            reasons.append(f"delta={delta:.0f}m from old apex ({old_corner.apex_m}m)")

    # Sub-apex within complex
    if corner.get('complex'):
        reasons.append(f"sub-apex within {corner['complex']} — needs visual confirmation")

    # Brake support for non-high-speed
    if corner['type'] != 'high_speed':
        if corner['brake_start_support_ratio'] < 0.50:
            reasons.append(
                f"brake support={corner['brake_start_support_ratio']:.0%} "
                f"for {corner['type']} corner"
            )

    if reasons:
        return True, "; ".join(reasons)
    return False, None


# ============================================================
# Build corner entries
# ============================================================

def build_corner_from_segment(seg_agg, corner_def, seg_info, old_track,
                              sub_idx=None, sub_count=None):
    """
    Build a single corner entry from aggregated segment data.
    For complexes with sub-corners, divide the segment proportionally.
    """
    cid, name, short, ctype, direction, complex_name = corner_def

    # For sub-corners within a complex, divide the segment
    if sub_idx is not None and sub_count is not None:
        seg_start = seg_info['consensus_start_m']
        seg_end = seg_info['consensus_end_m']
        seg_len = seg_end - seg_start
        sub_len = seg_len / sub_count
        sub_start = seg_start + sub_idx * sub_len
        sub_end = sub_start + sub_len

        # Use old track apex as anchor within sub-range if available
        old_corner = old_track.get_corner(cid)
        if old_corner and sub_start <= old_corner.apex_m <= sub_end:
            apex_center = old_corner.apex_m
        else:
            apex_center = (sub_start + sub_end) / 2

        apex_start = sub_start
        apex_end = sub_end
        apex_iqr = sub_len / 2  # wide by definition
        apex_support = seg_agg['apex_zone_start_m']['support_ratio']
        speed_min = apex_center  # approximate
        brake_m = seg_agg['brake_start_m']['median_m']
        brake_iqr = seg_agg['brake_start_m']['iqr_m']
        brake_support = seg_agg['brake_start_m']['support_ratio']
        turn_in = sub_start
        exit_end = sub_end
    else:
        apex_start = seg_agg['apex_zone_start_m']['median_m']
        apex_end = seg_agg['apex_zone_end_m']['median_m']
        apex_center = round((apex_start + apex_end) / 2, 1) if apex_start and apex_end else None
        apex_iqr = seg_agg['apex_zone_start_m']['iqr_m']
        apex_support = seg_agg['apex_zone_start_m']['support_ratio']
        speed_min = seg_agg['speed_minimum_m']['median_m']
        brake_m = seg_agg['brake_start_m']['median_m']
        brake_iqr = seg_agg['brake_start_m']['iqr_m']
        brake_support = seg_agg['brake_start_m']['support_ratio']
        turn_in = seg_agg['turn_in_m']['median_m']
        exit_end = seg_agg['exit_end_m']['median_m']

    if apex_center is None:
        old_corner = old_track.get_corner(cid)
        apex_center = old_corner.apex_m if old_corner else 0

    signal_count = len(seg_agg['signals'])
    score, conf_label = compute_confidence(
        apex_support, apex_iqr, signal_count, brake_support,
        seg_agg['source_laps']
    )

    corner = {
        'id': cid,
        'name': name,
        'short': short,
        'complex': complex_name,
        'type': ctype,
        'direction': direction,
        'segment_start_m': round(seg_info['consensus_start_m'], 1),
        'approach_start_m': round(brake_m - 50, 1) if brake_m else None,
        'brake_start_m': brake_m,
        'brake_start_iqr_m': brake_iqr,
        'brake_start_support_ratio': round(brake_support, 3),
        'turn_in_m': turn_in,
        'apex_zone_start_m': round(apex_start, 1) if apex_start else apex_center,
        'apex_zone_end_m': round(apex_end, 1) if apex_end else apex_center,
        'apex_zone_center_m': round(apex_center, 1),
        'apex_iqr_m': round(apex_iqr, 1) if apex_iqr else None,
        'apex_support_ratio': round(apex_support, 3),
        'speed_minimum_m': speed_min,
        'exit_end_m': exit_end,
        'confidence_score': score,
        'confidence': conf_label,
        'signals': seg_agg['signals'],
        'requires_review': False,
        'review_reason': None,
        'source_laps': seg_agg['source_laps'],
        'apex_m': round(apex_center, 1),  # backward compat
    }

    # Apply review flags
    requires, reason = check_review_flags(corner, old_track)
    corner['requires_review'] = requires
    corner['review_reason'] = reason

    return corner


def build_fallback_corner(corner_def, old_track):
    """Build a corner entry from old track data (not detected by segmentation)."""
    cid, name, short, ctype, direction, complex_name = corner_def
    old_corner = old_track.get_corner(cid)

    corner = {
        'id': cid,
        'name': name,
        'short': short,
        'complex': complex_name,
        'type': ctype,
        'direction': direction,
        'segment_start_m': old_corner.entry_m if old_corner else 0,
        'approach_start_m': None,
        'brake_start_m': None,
        'brake_start_iqr_m': None,
        'brake_start_support_ratio': 0.0,
        'turn_in_m': None,
        'apex_zone_start_m': old_corner.entry_m if old_corner else 0,
        'apex_zone_end_m': old_corner.exit_m if old_corner else 0,
        'apex_zone_center_m': old_corner.apex_m if old_corner else 0,
        'apex_iqr_m': None,
        'apex_support_ratio': 0.0,
        'speed_minimum_m': None,
        'exit_end_m': old_corner.exit_m if old_corner else 0,
        'confidence_score': 0,
        'confidence': 'low',
        'signals': [],
        'requires_review': True,
        'review_reason': 'Not detected by geometric segmentation — kept old values',
        'source_laps': 0,
        'apex_m': old_corner.apex_m if old_corner else 0,
    }
    return corner


# ============================================================
# Report generation
# ============================================================

def generate_report(corners, old_track, output_path, gp_name):
    """Generate markdown ground truth report."""
    high = sum(1 for c in corners if c['confidence'] == 'high')
    med = sum(1 for c in corners if c['confidence'] == 'medium')
    low = sum(1 for c in corners if c['confidence'] == 'low')
    review = [c for c in corners if c['requires_review']]
    n_corners = len(corners)

    lines = [
        f"# {gp_name} Ground Truth Report",
        "",
        "## 1. Summary",
        "",
        f"- **Session**: {gp_name} — Qualifying",
        f"- **Corners**: {n_corners}",
        "- **Method**: Curvature-based geometric segmentation + multi-signal phase detection",
        "- **Signals**: brake (boolean), deceleration gradient, curvature peaks, speed minima",
        "",
        "## 2. Confidence Distribution",
        "",
        f"| Level | Count |",
        f"|-------|-------|",
        f"| High  | {high} |",
        f"| Medium | {med} |",
        f"| Low   | {low} |",
        f"| **Total** | **{n_corners}** |",
        "",
        "## 3. Corner Comparison",
        "",
        "| # | Corner | Old apex | New apex zone | Center | Delta | IQR | Confidence | Review |",
        "|---|--------|----------|--------------|--------|-------|-----|------------|--------|",
    ]

    for c in corners:
        old_c = old_track.get_corner(c['id'])
        old_apex = old_c.apex_m if old_c else "—"
        zone = f"{c['apex_zone_start_m']:.0f}–{c['apex_zone_end_m']:.0f}"
        center = f"{c['apex_zone_center_m']:.0f}"
        delta = abs(c['apex_zone_center_m'] - old_c.apex_m) if old_c else 0
        iqr_str = f"{c['apex_iqr_m']:.0f}" if c['apex_iqr_m'] else "—"
        rev = "YES" if c['requires_review'] else ""
        lines.append(
            f"| {c['id']} | {c['name']} ({c['short']}) | {old_apex} | "
            f"{zone} | {center} | {delta:.0f} | {iqr_str} | "
            f"{c['confidence']} | {rev} |"
        )

    lines.extend(["", "## 4. Corners Requiring Human Review", ""])
    if review:
        for c in review:
            lines.append(f"- **{c['name']}** (id={c['id']}): {c['review_reason']}")
    else:
        lines.append("None — all corners accepted.")

    lines.extend(["", "## 5. Recommended Actions", ""])
    for c in review:
        if 'not detected' in (c['review_reason'] or '').lower():
            lines.append(f"- **{c['short']}**: Add speed-based detection for low-curvature corners")
        elif 'sub-apex' in (c['review_reason'] or '').lower():
            lines.append(f"- **{c['short']}**: Visually confirm sub-apex position within complex")
        elif 'IQR' in (c['review_reason'] or '').lower():
            lines.append(f"- **{c['short']}**: Narrow apex zone using speed minimum as anchor")
        else:
            lines.append(f"- **{c['short']}**: Manual review recommended")

    lines.extend(["", "---", "Generated by step3_build_ground_truth.py", ""])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report: {output_path}")


# ============================================================
# Validation plot
# ============================================================

def plot_validation(corners, df, all_curvatures, old_track, output_path):
    """Plot validation panels for corners requiring review."""
    review_corners = [c for c in corners if c['requires_review']]
    if not review_corners:
        print("  No corners require review — skipping validation plot.")
        return

    n_panels = len(review_corners)
    fig, axes = plt.subplots(n_panels, 3, figsize=(28, 4 * n_panels))
    fig.set_facecolor(COLORS['bg'])

    if n_panels == 1:
        axes = [axes]

    lap_ids = df['lap_id'].unique()

    for row, corner in enumerate(review_corners):
        ax_spd, ax_curv, ax_brk = axes[row]
        cid = corner['id']
        center = corner['apex_zone_center_m']
        window = 300  # show +/-300m around apex

        old_c = old_track.get_corner(cid)
        old_apex = old_c.apex_m if old_c else center

        # Speed panel
        style_axis(ax_spd, ylabel='Speed', title=f"{corner['name']} — conf={corner['confidence']}, "
                   f"IQR={corner['apex_iqr_m']}m")
        for lap_id in lap_ids:
            lap = df[df['lap_id'] == lap_id]
            mask = (lap['distance_m'] >= center - window) & (lap['distance_m'] <= center + window)
            seg = lap[mask]
            if len(seg) > 0:
                ax_spd.plot(seg['distance_m'], seg['speed'], lw=0.5, alpha=0.3, color=COLORS['game'])

        # Median speed
        pivot = df[(df['distance_m'] >= center - window) & (df['distance_m'] <= center + window)]
        med_spd = pivot.groupby('distance_m')['speed'].median()
        ax_spd.plot(med_spd.index, med_spd.values, lw=2, color='white')

        # Old apex (red dashed)
        ax_spd.axvline(old_apex, color=COLORS['real'], ls='--', lw=1.5, label=f'Old apex={old_apex:.0f}')
        # New apex zone (green shaded)
        ax_spd.axvspan(corner['apex_zone_start_m'], corner['apex_zone_end_m'],
                       alpha=0.2, color=COLORS['faster'])
        ax_spd.legend(fontsize=7, loc='upper right', facecolor=COLORS['card_bg'],
                      labelcolor='white', edgecolor='white')

        # Curvature panel
        style_axis(ax_curv, ylabel='|kappa|')
        for lap_id in lap_ids[:20]:
            dist, kappa_mag, _ = all_curvatures[lap_id]
            mask = (dist >= center - window) & (dist <= center + window)
            ax_curv.plot(dist[mask], kappa_mag[mask], lw=0.5, alpha=0.3, color='#ffaa00')
        ax_curv.axvline(old_apex, color=COLORS['real'], ls='--', lw=1.5)
        ax_curv.axvspan(corner['apex_zone_start_m'], corner['apex_zone_end_m'],
                        alpha=0.2, color=COLORS['faster'])

        # Brake panel
        style_axis(ax_brk, ylabel='Brake', xlabel='Distance (m)')
        for lap_id in lap_ids[:20]:
            lap = df[df['lap_id'] == lap_id]
            mask = (lap['distance_m'] >= center - window) & (lap['distance_m'] <= center + window)
            seg = lap[mask]
            if len(seg) > 0:
                ax_brk.step(seg['distance_m'], seg['brake'], lw=0.5, alpha=0.3, color='#ff6666')
        ax_brk.axvline(old_apex, color=COLORS['real'], ls='--', lw=1.5)
        ax_brk.axvspan(corner['apex_zone_start_m'], corner['apex_zone_end_m'],
                       alpha=0.2, color=COLORS['faster'])

    plt.tight_layout()
    save_figure(fig, output_path, dpi=150)
    print(f"  Validation plot: {output_path}")


# ============================================================
# Backward compatibility check
# ============================================================

def verify_backward_compat(gt_path, n_corners):
    """Verify ground truth JSON is loadable by existing corner loader."""
    from src.track import Corner, CORNER_COLORS

    with open(gt_path) as f:
        data = json.load(f)

    # Check structure
    assert 'corners' in data, "Missing 'corners' key"
    assert len(data['corners']) == n_corners, (
        f"Corner count mismatch: {len(data['corners'])} vs expected {n_corners}"
    )

    # Check each corner has required fields for the loader
    for i, c in enumerate(data['corners']):
        assert 'id' in c, f"Corner {i}: missing 'id'"
        assert 'name' in c, f"Corner {i}: missing 'name'"
        assert 'short' in c, f"Corner {i}: missing 'short'"
        assert 'entry_m' in c or 'apex_zone_start_m' in c, f"Corner {i}: missing entry"
        assert 'apex_m' in c, f"Corner {i}: missing 'apex_m'"
        assert 'exit_m' in c or 'apex_zone_end_m' in c, f"Corner {i}: missing exit"

    # Dry-load test using the actual loader pattern
    corners = [Corner(
        id=c['id'], name=c['name'], short=c['short'],
        type=c.get('type', 'medium_speed'),
        direction=c.get('direction', 'right'),
        entry_m=c.get('entry_m', c.get('apex_zone_start_m', 0)),
        apex_m=c['apex_m'],
        exit_m=c.get('exit_m', c.get('apex_zone_end_m', 0)),
        color=CORNER_COLORS[i % len(CORNER_COLORS)],
    ) for i, c in enumerate(data['corners'])]

    assert len(corners) == n_corners, f"Expected {n_corners} corners, got {len(corners)}"
    print(f"  Backward compat: {len(corners)} corners loaded successfully")
    return True


# ============================================================
# Helper
# ============================================================

def _compute_kappa(x, y):
    """Helper: compute signed curvature from XY."""
    x_s = gaussian_filter1d(x, sigma=5)
    y_s = gaussian_filter1d(y, sigma=5)
    dx = np.gradient(x_s)
    dy = np.gradient(y_s)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = np.maximum((dx**2 + dy**2)**1.5, 1e-12)
    return (dx * ddy - dy * ddx) / denom


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    gp_name = get_gp_name(args.track)
    prefix = get_output_prefix(args.track)

    print("=" * 60)
    print("  Step 3: Build Ground Truth")
    print(f"  {gp_name} — Confidence Scoring + Review Flags")
    print("=" * 60)

    # Load prerequisites
    print("\n  Loading data...")
    df = pd.read_parquet(OUTPUT_DIR / f"{prefix}_reference_laps.parquet")
    with open(OUTPUT_DIR / f"{prefix}_segments_consensus.json") as f:
        seg_data = json.load(f)
    phase_df = pd.read_parquet(OUTPUT_DIR / f"{prefix}_phase_detection_per_lap.parquet")
    old_track = load_track(args.track)

    segments = seg_data['segments']
    n_laps = df['lap_id'].nunique()
    n_corners = old_track.n_corners
    print(f"  Laps: {n_laps}, Segments: {len(segments)}, Old corners: {n_corners}")

    # Recompute curvatures for validation plot
    print("  Computing curvatures for validation...")
    all_curvatures = {}
    for lap_id in df['lap_id'].unique():
        lap = df[df['lap_id'] == lap_id].sort_values('distance_m')
        x = lap['x'].values
        y = lap['y'].values
        dist = lap['distance_m'].values
        kappa = gaussian_filter1d(np.abs(
            _compute_kappa(x, y)
        ), sigma=5)
        all_curvatures[lap_id] = (dist, kappa, kappa)

    # Build dynamic corner map from track data
    corner_map, fallback_corners = build_corner_map(segments, old_track)

    # Build corners
    print("\n  [Step 1-2] Aggregating phases and scoring confidence...")
    corners = []

    for seg in segments:
        seg_id = seg['segment_id']
        seg_agg = aggregate_segment_phases(phase_df, seg_id, n_laps)

        if seg_id in corner_map and corner_map[seg_id]:
            corner_defs = corner_map[seg_id]
            n_sub = len(corner_defs)
            for sub_idx, cdef in enumerate(corner_defs):
                if n_sub > 1:
                    c = build_corner_from_segment(
                        seg_agg, cdef, seg, old_track,
                        sub_idx=sub_idx, sub_count=n_sub
                    )
                else:
                    c = build_corner_from_segment(seg_agg, cdef, seg, old_track)
                corners.append(c)

    # Add fallback corners (not assigned to any segment)
    for cdef in fallback_corners:
        corners.append(build_fallback_corner(cdef, old_track))

    # Sort by id
    corners.sort(key=lambda c: c['id'])
    print(f"  Total corners: {len(corners)}")

    # Step 4A: Write ground truth JSON
    print("\n  [Step 4] Writing ground truth...")
    gt_data = {
        "name": old_track.name,
        "country": old_track.country if hasattr(old_track, 'country') else "",
        "short": args.track,
        "length_m": old_track.length_m,
        "corners": corners,
        "sectors": [
            {"id": s.id, "name": s.name, "start_m": s.start_m, "end_m": s.end_m}
            for s in old_track.sectors
        ],
        "drs_zones": [
            {"start_m": d.start_m, "end_m": d.end_m}
            for d in old_track.drs_zones
        ],
    }

    # Add entry_m / exit_m for backward compat with loader
    for c in gt_data['corners']:
        c['entry_m'] = c['apex_zone_start_m']
        c['exit_m'] = c['apex_zone_end_m']

    gt_path = TRACKS_DIR / f"{args.track}.ground_truth.json"
    with open(gt_path, 'w', encoding='utf-8') as f:
        json.dump(gt_data, f, indent=2, ensure_ascii=False)
    print(f"  Ground truth: {gt_path}")

    # Step 4B: Report
    report_path = OUTPUT_DIR / f"{prefix}_ground_truth_report.md"
    generate_report(corners, old_track, report_path, gp_name)

    # Step 4C: Validation plot
    plot_path = OUTPUT_DIR / f"{prefix}_ground_truth_validation.png"
    plot_validation(corners, df, all_curvatures, old_track, plot_path)

    # Step 5: Backward compat
    print("\n  [Step 5] Backward compatibility check...")
    try:
        verify_backward_compat(gt_path, n_corners)
    except AssertionError as e:
        print(f"  FAILED: {e}")
        return

    # Step 6: Final diff
    print("\n  [Step 6] Final diff summary:")
    print(f"  {'Corner':<20} {'Old->New apex':<20} {'Delta':>6} {'Conf':<8} {'Review'}")
    print(f"  {'-'*20} {'-'*20} {'-'*6} {'-'*8} {'-'*6}")

    diffs = []
    for c in corners:
        old_c = old_track.get_corner(c['id'])
        old_apex = old_c.apex_m if old_c else 0
        delta = abs(c['apex_m'] - old_apex)
        diffs.append((c, old_apex, delta))

    diffs.sort(key=lambda x: x[2], reverse=True)
    for c, old_apex, delta in diffs:
        rev = "YES" if c['requires_review'] else ""
        print(f"  {c['name']:<20} {old_apex:>6.0f}->{c['apex_m']:<6.0f} "
              f"{delta:>5.0f}m {c['confidence']:<8} {rev}")

    # Summary
    high = sum(1 for c in corners if c['confidence'] == 'high')
    med = sum(1 for c in corners if c['confidence'] == 'medium')
    low = sum(1 for c in corners if c['confidence'] == 'low')
    review = sum(1 for c in corners if c['requires_review'])
    large_delta = [(c, d) for c, _, d in diffs if d > 30]

    print(f"\n  === SUMMARY ===")
    print(f"  Confidence: {high} high, {med} medium, {low} low")
    print(f"  Require review: {review}/{len(corners)}")
    if large_delta:
        print(f"  Delta > 30m: {len(large_delta)}")
        for c, d in large_delta:
            print(f"    - {c['name']}: {d:.0f}m")


if __name__ == "__main__":
    main()
