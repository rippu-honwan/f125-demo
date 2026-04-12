"""
F1 Lap Insight - Step 2: Lap Analysis
Detailed corner-by-corner analysis of your game lap.

Usage:
    python scripts/02_lap_analysis.py [csv_path] [--track suzuka]
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from config import DEFAULT_CSV, DEFAULT_TRACK, OUTPUT_DIR, BG_COLOR, DPI_SAVE
from src.loader import load_and_prepare
from src.track import load_track, auto_detect_track
from src.corners import analyze_solo, summarize_corners
from src.utils import smooth, format_laptime


def parse_args():
    parser = argparse.ArgumentParser(description="F1 Lap Insight - Lap Analysis")
    parser.add_argument('csv', nargs='?', default=DEFAULT_CSV)
    parser.add_argument('--track', '-t', default=None,
                        help='Track name (auto-detect if omitted)')
    return parser.parse_args()


def plot_corner_speeds(corners, track, meta):
    n = len(corners)
    fig, ax = plt.subplots(figsize=(max(20, n * 1.2), 8))
    fig.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    x = np.arange(n)
    colors = [c['color'] for c in corners]
    speeds = [c['min_speed'] for c in corners]

    bars = ax.bar(x, speeds, color=colors, alpha=0.85,
                  edgecolor='white', lw=0.5)

    for i, (spd, c) in enumerate(zip(speeds, corners)):
        ax.text(i, spd + 5, f"{spd:.0f}",
                color='white', ha='center', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"T{c['id']}\n{c['short']}" for c in corners],
        fontsize=7, color='white'
    )
    ax.set_ylabel('Min Speed (km/h)', color='white', fontsize=12)
    ax.set_title(
        f"{track.name} - Corner Speeds  |  "
        f"Lap: {format_laptime(meta['best_time'])}",
        color='white', fontsize=14, fontweight='bold'
    )
    ax.tick_params(colors='white')
    ax.grid(axis='y', alpha=0.1, color='white')

    plt.tight_layout()
    return fig


def plot_corner_map(data, corners, track, meta):
    if 'world_position_X' not in data.columns:
        return None

    fig, ax = plt.subplots(figsize=(22, 18))
    fig.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    x = data['world_position_X'].values
    y = data['world_position_Y'].values
    dist = data['lap_distance'].values

    ax.plot(x, y, color='#2a2a2a', lw=12, zorder=1, solid_capstyle='round')

    for c in corners:
        mask = (dist >= c['entry_dist']) & (dist <= c['exit_dist'])
        cidx = np.where(mask)[0]
        if len(cidx) > 2:
            ax.plot(x[cidx], y[cidx], color=c['color'], lw=10,
                    alpha=0.5, zorder=3, solid_capstyle='round')

        ai = int(np.argmin(np.abs(dist - c['apex_dist'])))
        ax.scatter(x[ai], y[ai], color=c['color'], s=300, zorder=8,
                   edgecolors='white', linewidths=2)
        ax.text(x[ai], y[ai], str(c['id']), color='white', fontsize=9,
                fontweight='bold', ha='center', va='center', zorder=9)

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{track.name} - Corner Map  |  {track.n_corners} Corners",
        color='white', fontsize=14, fontweight='bold'
    )

    plt.tight_layout()
    return fig


def print_summary(corners, track, meta):
    print(f"\n{'=' * 70}")
    print(f"  {track.name} - Lap Analysis")
    print(f"  Lap Time: {format_laptime(meta['best_time'])}")
    print(f"  {track.n_corners} Corners")
    print(f"{'=' * 70}")

    print(f"\n  {'#':<4} {'Name':<20} {'Min':>6} {'Entry':>6} "
          f"{'Exit':>6} {'Gear':>5} {'Time':>7}")
    print(f"  {'-'*4} {'-'*20} {'-'*6} {'-'*6} "
          f"{'-'*6} {'-'*5} {'-'*7}")

    for c in corners:
        gear = str(c['gear']) if c['gear'] else '-'
        print(f"  T{c['id']:<2} {c['name']:<20} "
              f"{c['min_speed']:>5.0f}  {c['entry_speed']:>5.0f}  "
              f"{c['exit_speed']:>5.0f}  {gear:>4}  "
              f"{c['corner_time']:>6.3f}s")

    summary = summarize_corners(corners, mode="solo")
    if summary['slowest']:
        s = summary['slowest']
        print(f"\n  Slowest: T{s['id']} {s['name']} ({s['min_speed']:.0f} km/h)")
    if summary['fastest']:
        f = summary['fastest']
        print(f"  Fastest: T{f['id']} {f['name']} ({f['min_speed']:.0f} km/h)")


def main():
    args = parse_args()

    print("=" * 60)
    print("  F1 Lap Insight - Step 2: Lap Analysis")
    print("=" * 60)

    data, meta = load_and_prepare(args.csv)

    if args.track:
        track = load_track(args.track)
    else:
        track = auto_detect_track(meta['track_length'])
        if track is None:
            print("  ERROR: Cannot auto-detect track. Use --track flag.")
            sys.exit(1)

    print(f"  Track: {track.name} ({track.n_corners} corners)")

    corners = analyze_solo(data, track)
    print_summary(corners, track, meta)

    # Plots
    fig1 = plot_corner_speeds(corners, track, meta)
    p1 = OUTPUT_DIR / f"{track.short}_corner_speeds.png"
    fig1.savefig(p1, dpi=DPI_SAVE, bbox_inches='tight')
    print(f"\n  Saved: {p1}")
    plt.close(fig1)

    fig2 = plot_corner_map(data, corners, track, meta)
    if fig2:
        p2 = OUTPUT_DIR / f"{track.short}_corner_map.png"
        fig2.savefig(p2, dpi=DPI_SAVE, bbox_inches='tight')
        print(f"  Saved: {p2}")
        plt.close(fig2)

    print(f"\n{'=' * 60}")
    print(f"  COMPLETE!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()