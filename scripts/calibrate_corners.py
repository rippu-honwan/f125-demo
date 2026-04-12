"""
Auto-calibrate corner positions from telemetry data.
Finds speed minima and maps them to corners.

Usage:
    python scripts/calibrate_corners.py data/my_lap.csv
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

from src.loader import load_and_prepare
from src.utils import smooth
from config import OUTPUT_DIR, BG_COLOR


def find_corners_from_speed(dist, speed, order=80, min_speed_drop=30):
    """
    Find corner locations from speed trace.
    
    Returns list of dicts with entry/apex/exit distances.
    """
    speed_s = smooth(speed, 20)
    
    # Find speed minima (= apex points)
    min_idx = argrelextrema(speed_s, np.less, order=order)[0]
    # Find speed maxima (= straight ends / corner entries)
    max_idx = argrelextrema(speed_s, np.greater, order=order)[0]
    
    corners = []
    
    for mi in min_idx:
        apex_dist = float(dist[mi])
        apex_speed = float(speed_s[mi])
        
        # Find entry: last speed max before this min
        prev_maxes = max_idx[max_idx < mi]
        if len(prev_maxes) > 0:
            entry_idx = prev_maxes[-1]
            entry_dist = float(dist[entry_idx])
            entry_speed = float(speed_s[entry_idx])
        else:
            entry_dist = max(0, apex_dist - 150)
            entry_speed = float(speed_s[max(0, mi - 150)])
            entry_idx = max(0, mi - 150)
        
        # Check speed drop is significant
        speed_drop = entry_speed - apex_speed
        if speed_drop < min_speed_drop:
            continue
        
        # Find exit: next speed max after this min
        next_maxes = max_idx[max_idx > mi]
        if len(next_maxes) > 0:
            exit_idx = next_maxes[0]
            exit_dist = float(dist[exit_idx])
        else:
            exit_dist = min(dist.max(), apex_dist + 150)
        
        # Refine entry: where speed starts dropping (95% of entry speed)
        threshold = entry_speed * 0.95
        refined_entry = entry_idx
        for j in range(entry_idx, mi):
            if speed_s[j] < threshold:
                refined_entry = max(entry_idx, j - 20)
                break
        
        # Refine exit: where speed recovers to 90% of next max
        if len(next_maxes) > 0:
            next_max_speed = float(speed_s[next_maxes[0]])
            threshold_exit = apex_speed + (next_max_speed - apex_speed) * 0.3
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
    
    # Sort by distance
    corners.sort(key=lambda c: c['apex_m'])
    
    # Remove duplicates (corners too close together)
    filtered = []
    for c in corners:
        if filtered and c['apex_m'] - filtered[-1]['apex_m'] < 60:
            # Keep the one with bigger speed drop
            if c['speed_drop'] > filtered[-1]['speed_drop']:
                filtered[-1] = c
        else:
            filtered.append(c)
    
    return filtered


def plot_calibration(dist, speed, corners, track_length):
    """Plot speed trace with detected corners."""
    fig, axes = plt.subplots(2, 1, figsize=(30, 16),
                              gridspec_kw={'height_ratios': [3, 1]})
    fig.set_facecolor(BG_COLOR)
    
    speed_s = smooth(speed, 15)
    
    # ---- Top: Speed trace with corners ----
    ax = axes[0]
    ax.set_facecolor(BG_COLOR)
    ax.plot(dist, speed_s, color='#00ff88', lw=2, label='Speed')
    
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(corners), 1)))
    
    for i, c in enumerate(corners):
        color = colors[i]
        
        # Corner zone
        ax.axvspan(c['entry_m'], c['exit_m'], alpha=0.15, color=color)
        
        # Apex marker
        ax.axvline(c['apex_m'], color=color, lw=1.5, ls='--', alpha=0.7)
        
        # Label
        y_pos = c['apex_speed'] - 15
        ax.annotate(
            f"C{i+1}\n{c['apex_m']:.0f}m\n{c['apex_speed']:.0f}km/h\n"
            f"drop:{c['speed_drop']:.0f}",
            xy=(c['apex_m'], c['apex_speed']),
            xytext=(c['apex_m'], y_pos),
            fontsize=7, color='white', ha='center',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                     alpha=0.7, edgecolor='white', lw=0.5),
            arrowprops=dict(arrowstyle='->', color='white', lw=1),
            zorder=10
        )
    
    ax.set_ylabel('Speed (km/h)', color='white', fontsize=12)
    ax.set_title(
        f'Corner Calibration - {len(corners)} corners detected  |  '
        f'Track: {track_length:.0f}m',
        color='white', fontsize=14, fontweight='bold'
    )
    ax.tick_params(colors='white')
    ax.grid(alpha=0.1, color='white')
    ax.legend(loc='upper right', facecolor='#1a1a2e',
              edgecolor='white', labelcolor='white')
    
    # ---- Bottom: Speed gradient (deceleration) ----
    ax2 = axes[1]
    ax2.set_facecolor(BG_COLOR)
    
    decel = -np.gradient(speed_s)
    decel_s = smooth(decel, 20)
    
    ax2.fill_between(dist, decel_s, where=decel_s > 0,
                     color='#ff4444', alpha=0.5, label='Braking')
    ax2.fill_between(dist, decel_s, where=decel_s < 0,
                     color='#44ff44', alpha=0.5, label='Accelerating')
    ax2.axhline(0, color='#666666', lw=0.5)
    
    for i, c in enumerate(corners):
        ax2.axvline(c['apex_m'], color=colors[i], lw=1, ls='--', alpha=0.5)
    
    ax2.set_ylabel('Deceleration', color='white', fontsize=10)
    ax2.set_xlabel('Distance (m)', color='white', fontsize=12)
    ax2.tick_params(colors='white')
    ax2.legend(loc='upper right', facecolor='#1a1a2e',
               edgecolor='white', labelcolor='white', fontsize=8)
    
    plt.tight_layout()
    return fig


def generate_json_snippet(corners):
    """Generate JSON corner definitions."""
    
    # Known Suzuka corner names (in order)
    suzuka_names = [
        ("Turn 1", "T1", "medium_speed", "right"),
        ("Turn 2", "T2", "medium_speed", "left"),
        ("S Curve 1", "S1", "high_speed", "left"),
        ("S Curve 2", "S2", "high_speed", "right"),
        ("S Curve 3", "S3", "high_speed", "left"),
        ("S Curve 4", "S4", "high_speed", "right"),
        ("Dunlop", "DUN", "medium_speed", "right"),
        ("Degner 1", "DG1", "high_speed", "right"),
        ("Degner 2", "DG2", "medium_speed", "left"),
        ("200R", "200R", "high_speed", "left"),
        ("Hairpin", "HAIR", "low_speed", "left"),
        ("Spoon Entry", "SP-E", "medium_speed", "left"),
        ("Spoon Exit", "SP-X", "medium_speed", "left"),
        ("Backstretch Kink", "KINK", "flat_out", "right"),
        ("130R", "130R", "high_speed", "left"),
        ("Casio Triangle 1", "CS1", "low_speed", "left"),
        ("Casio Triangle 2", "CS2", "low_speed", "right"),
        ("Final Corner", "FIN", "low_speed", "right"),
    ]
    
    print(f"\n{'='*70}")
    print(f"  DETECTED {len(corners)} CORNERS")
    print(f"  Expected: {len(suzuka_names)} (Suzuka)")
    print(f"{'='*70}")
    
    print(f"\n  {'#':<4} {'Apex':>7} {'Entry':>7} {'Exit':>7} "
          f"{'Speed':>6} {'Drop':>6}")
    print(f"  {'-'*4} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*6}")
    
    for i, c in enumerate(corners):
        name = suzuka_names[i][0] if i < len(suzuka_names) else f"Corner {i+1}"
        print(f"  C{i+1:<2} {c['apex_m']:>6.0f}  {c['entry_m']:>6.0f}  "
              f"{c['exit_m']:>6.0f}  {c['apex_speed']:>5.0f}  "
              f"{c['speed_drop']:>5.0f}  {name}")
    
    # Generate JSON
    print(f"\n\n  === JSON SNIPPET (copy to tracks/suzuka.json) ===\n")
    print('  "corners": [')
    
    for i, c in enumerate(corners):
        if i < len(suzuka_names):
            name, short, ctype, direction = suzuka_names[i]
        else:
            name = f"Corner {i+1}"
            short = f"C{i+1}"
            ctype = "medium_speed"
            direction = "right"
        
        comma = "," if i < len(corners) - 1 else ""
        print(f'    {{"id": {i+1}, "name": "{name}", "short": "{short}", '
              f'"type": "{ctype}", "direction": "{direction}", '
              f'"entry_m": {c["entry_m"]:.0f}, '
              f'"apex_m": {c["apex_m"]:.0f}, '
              f'"exit_m": {c["exit_m"]:.0f}}}{comma}')
    
    print('  ]')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', nargs='?', default='data/my_lap.csv')
    parser.add_argument('--order', type=int, default=80,
                        help='Sensitivity (lower=more corners, default=80)')
    parser.add_argument('--min-drop', type=int, default=25,
                        help='Min speed drop to count as corner (default=25)')
    args = parser.parse_args()
    
    print("="*60)
    print("  Corner Calibration Tool")
    print("="*60)
    
    data, meta = load_and_prepare(args.csv)
    
    dist = data['lap_distance'].values
    speed = data['speed_kmh'].values
    
    corners = find_corners_from_speed(
        dist, speed,
        order=args.order,
        min_speed_drop=args.min_drop
    )
    
    generate_json_snippet(corners)
    
    fig = plot_calibration(dist, speed, corners, meta['track_length'])
    path = OUTPUT_DIR / "corner_calibration.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"\n  Saved: {path}")
    plt.close(fig)
    
    # Suggestions
    if len(corners) < 18:
        print(f"\n  ⚠️  Only {len(corners)} detected, expected 18.")
        print(f"  Try: --order 60 --min-drop 20")
    elif len(corners) > 18:
        print(f"\n  ⚠️  {len(corners)} detected, expected 18.")
        print(f"  Try: --order 100 --min-drop 35")
    else:
        print(f"\n  ✅ {len(corners)} corners = matches Suzuka!")


if __name__ == "__main__":
    main()