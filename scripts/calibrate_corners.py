#!/usr/bin/env python3
"""
Corner Calibration Tool
Auto-detect corner positions from telemetry data.

Usage:
    python scripts/calibrate_corners.py data/my_lap.csv
    python scripts/calibrate_corners.py data/my_lap.csv --order 60 --min-drop 20

Changes from original:
  - No hardcoded Suzuka corner names
  - Uses track JSON names if available (matching by count)
  - Better JSON output formatting
  - Uses shared plotting helpers
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

from src.loader import load_and_prepare
from src.track import list_tracks, load_track
from src.utils import smooth
from src.plotting import COLORS, style_axis, save_figure
from config import OUTPUT_DIR


def find_corners_from_speed(dist, speed, order=80, min_speed_drop=30):
    """
    Find corner locations from speed trace.
    Returns list of dicts with entry/apex/exit distances.
    """
    speed_s = smooth(speed, 20)

    min_idx = argrelextrema(speed_s, np.less, order=order)[0]
    max_idx = argrelextrema(speed_s, np.greater, order=order)[0]

    corners = []

    for mi in min_idx:
        apex_dist = float(dist[mi])
        apex_speed = float(speed_s[mi])

        # Entry: last speed max before this min
        prev_maxes = max_idx[max_idx < mi]
        if len(prev_maxes) > 0:
            entry_idx = prev_maxes[-1]
            entry_dist = float(dist[entry_idx])
            entry_speed = float(speed_s[entry_idx])
        else:
            entry_dist = max(0, apex_dist - 150)
            entry_speed = float(speed_s[max(0, mi - 150)])
            entry_idx = max(0, mi - 150)

        speed_drop = entry_speed - apex_speed
        if speed_drop < min_speed_drop:
            continue

        # Exit: next speed max after this min
        next_maxes = max_idx[max_idx > mi]
        if len(next_maxes) > 0:
            exit_idx = next_maxes[0]
            exit_dist = float(dist[exit_idx])
        else:
            exit_dist = min(dist.max(), apex_dist + 150)

        # Refine entry
        threshold = entry_speed * 0.95
        refined_entry = entry_idx
        for j in range(entry_idx, mi):
            if speed_s[j] < threshold:
                refined_entry = max(entry_idx, j - 20)
                break

        # Refine exit
        if len(next_maxes) > 0:
            next_max_speed = float(speed_s[next_maxes[0]])
            threshold_exit = apex_speed + \
                (next_max_speed - apex_speed) * 0.3
            refined_exit = exit_idx
            for j in range(mi, min(exit_idx + 1, len(speed_s))):
                if speed_s[j] > threshold_exit:
                    refined_exit = j
                    break
        else:
            refined_exit = min(len(dist) - 1, mi + 150)

        corners.append({
            'apex_m': apex_dist,
            'apex_speed': apex_speed,
            'entry_m': float(dist[refined_entry]),
            'exit_m': float(dist[refined_exit]),
            'speed_drop': speed_drop,
            'entry_speed': entry_speed,
        })

    corners.sort(key=lambda c: c['apex_m'])

    # Remove duplicates
    filtered = []
    for c in corners:
        if filtered and c['apex_m'] - filtered[-1]['apex_m'] < 60:
            if c['speed_drop'] > filtered[-1]['speed_drop']:
                filtered[-1] = c
        else:
            filtered.append(c)

    return filtered


def try_match_track(corners, track_length):
    """
    Try to match detected corners with a known track JSON.
    Returns corner name/type info if track matches.
    """
    for track_name in list_tracks():
        try:
            track = load_track(track_name)
            if abs(track.length_m - track_length) < 200:
                if len(track.corners) == len(corners):
                    print(f"  ✓ Matched {track.name} "
                          f"({len(corners)} corners)")
                    return [(c.name, c.short, c.type, c.direction)
                            for c in track.corners]
                else:
                    print(f"  ~ Partial match: {track.name} "
                          f"(expected {len(track.corners)}, "
                          f"found {len(corners)})")
        except Exception:
            continue
    return None


def plot_calibration(dist, speed, corners, track_length):
    """Plot speed trace with detected corners."""
    fig, axes = plt.subplots(2, 1, figsize=(30, 16),
                              gridspec_kw={'height_ratios': [3, 1]})
    fig.set_facecolor(COLORS['bg'])

    speed_s = smooth(speed, 15)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(corners), 1)))

    # Speed trace
    ax = axes[0]
    style_axis(ax, ylabel='Speed (km/h)',
               title=f'Corner Calibration - {len(corners)} corners  |  '
                     f'Track: {track_length:.0f}m')

    ax.plot(dist, speed_s, color=COLORS['game'], lw=2, label='Speed')

    for i, c in enumerate(corners):
        color = colors[i]
        ax.axvspan(c['entry_m'], c['exit_m'], alpha=0.15, color=color)
        ax.axvline(c['apex_m'], color=color, lw=1.5, ls='--', alpha=0.7)

        y_pos = c['apex_speed'] - 15
        ax.annotate(
            f"C{i+1}\n{c['apex_m']:.0f}m\n{c['apex_speed']:.0f}km/h",
            xy=(c['apex_m'], c['apex_speed']),
            xytext=(c['apex_m'], y_pos),
            fontsize=7, color='white', ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                     alpha=0.7, edgecolor='white', lw=0.5),
            arrowprops=dict(arrowstyle='->', color='white', lw=1),
            zorder=10)

    ax.legend(loc='upper right', facecolor=COLORS['card_bg'],
              edgecolor='white', labelcolor='white')

    # Deceleration
    ax2 = axes[1]
    style_axis(ax2, ylabel='Deceleration', xlabel='Distance (m)')
    decel = -np.gradient(speed_s)
    decel_s = smooth(decel, 20)

    ax2.fill_between(dist, decel_s, where=decel_s > 0,
                     color=COLORS['slower'], alpha=0.5, label='Braking')
    ax2.fill_between(dist, decel_s, where=decel_s < 0,
                     color='#44ff44', alpha=0.5, label='Accelerating')
    ax2.axhline(0, color='#666666', lw=0.5)

    for i, c in enumerate(corners):
        ax2.axvline(c['apex_m'], color=colors[i], lw=1,
                    ls='--', alpha=0.5)

    ax2.legend(loc='upper right', facecolor=COLORS['card_bg'],
               edgecolor='white', labelcolor='white', fontsize=8)

    plt.tight_layout()
    return fig


def generate_json(corners, track_names=None):
    """Generate JSON corner definitions."""
    print(f"\n{'='*70}")
    print(f"  DETECTED {len(corners)} CORNERS")
    print(f"{'='*70}")

    print(f"\n  {'#':<4} {'Apex':>7} {'Entry':>7} {'Exit':>7} "
          f"{'Speed':>6} {'Drop':>6}  Name")
    print(f"  {'-'*4} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*6}  {'-'*20}")

    for i, c in enumerate(corners):
        name = track_names[i][0] if track_names and i < len(track_names) \
            else f"Corner {i+1}"
        print(f"  C{i+1:<2} {c['apex_m']:>6.0f}  {c['entry_m']:>6.0f}  "
              f"{c['exit_m']:>6.0f}  {c['apex_speed']:>5.0f}  "
              f"{c['speed_drop']:>5.0f}  {name}")

    # Generate JSON
    print(f"\n\n  === JSON (copy to tracks/<name>.json) ===\n")
    json_corners = []

    for i, c in enumerate(corners):
        if track_names and i < len(track_names):
            name, short, ctype, direction = track_names[i]
        else:
            name = f"Corner {i+1}"
            short = f"C{i+1}"
            ctype = "medium_speed"
            direction = "right"

        json_corners.append({
            "id": i + 1,
            "name": name,
            "short": short,
            "type": ctype,
            "direction": direction,
            "entry_m": round(c['entry_m']),
            "apex_m": round(c['apex_m']),
            "exit_m": round(c['exit_m']),
        })

    print(json.dumps({"corners": json_corners}, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Corner Calibration Tool"
    )
    parser.add_argument('csv', nargs='?', default='data/my_lap.csv')
    parser.add_argument('--order', type=int, default=80,
                        help='Sensitivity (lower=more corners)')
    parser.add_argument('--min-drop', type=int, default=25,
                        help='Min speed drop to count as corner')
    args = parser.parse_args()

    print("=" * 60)
    print("  Corner Calibration Tool")
    print("=" * 60)

    data, meta = load_and_prepare(args.csv)

    dist = data['lap_distance'].values
    speed = data['speed_kmh'].values

    corners = find_corners_from_speed(
        dist, speed, order=args.order, min_speed_drop=args.min_drop
    )

    # Try to match with known track
    track_names = try_match_track(corners, meta['track_length'])

    generate_json(corners, track_names)

    fig = plot_calibration(dist, speed, corners, meta['track_length'])
    save_figure(fig, OUTPUT_DIR / "corner_calibration.png", dpi=200)

    # Guidance
    print(f"\n  Detected {len(corners)} corners.")
    if len(corners) < 10:
        print(f"  Too few? Try: --order 60 --min-drop 20")
    elif len(corners) > 25:
        print(f"  Too many? Try: --order 100 --min-drop 35")


if __name__ == "__main__":
    main()