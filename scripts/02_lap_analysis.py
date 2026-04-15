#!/usr/bin/env python3
"""
F1 Lap Insight - Step 2: Lap Analysis
Corner-by-corner analysis of your game lap.

Usage:
    python scripts/02_lap_analysis.py [csv_path] [--track suzuka]

Changes from original:
  - Uses shared plotting helpers
  - Better auto-detection via SRT metadata
  - Cleaner print output
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from config import DEFAULT_CSV, OUTPUT_DIR, DPI_SAVE
from src.loader import load_and_prepare
from src.track import load_track, auto_detect_track
from src.corners import analyze_solo, summarize_corners
from src.utils import smooth, format_laptime
from src.plotting import COLORS, style_axis, save_figure


def parse_args():
    parser = argparse.ArgumentParser(
        description="F1 Lap Insight - Lap Analysis"
    )
    parser.add_argument('csv', nargs='?', default=DEFAULT_CSV)
    parser.add_argument('--track', '-t', default=None,
                        help='Track name (auto-detect if omitted)')
    return parser.parse_args()


def plot_corner_speeds(corners, track, meta):
    n = len(corners)
    fig, ax = plt.subplots(figsize=(max(20, n * 1.2), 8))
    fig.set_facecolor(COLORS['bg'])

    style_axis(ax, ylabel='Min Speed (km/h)',
               title=f"{track.name} - Corner Speeds  |  "
                     f"Lap: {format_laptime(meta['best_time'])}")

    x = np.arange(n)
    colors = [c['color'] for c in corners]
    speeds = [c['min_speed'] for c in corners]

    ax.bar(x, speeds, color=colors, alpha=0.85,
           edgecolor='white', lw=0.5)

    for i, spd in enumerate(speeds):
        ax.text(i, spd + 5, f"{spd:.0f}",
                color='white', ha='center', fontsize=8,
                fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"T{c['id']}\n{c['short']}" for c in corners],
        fontsize=7, color='white'
    )

    plt.tight_layout()
    return fig


def plot_corner_map(data, corners, track, meta):
    if 'world_position_X' not in data.columns:
        return None

    fig, ax = plt.subplots(figsize=(22, 18))
    fig.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    x = data['world_position_X'].values
    y = data['world_position_Y'].values
    dist = data['lap_distance'].values

    # Track outline
    ax.plot(x, y, color='#2a2a2a', lw=12, zorder=1,
            solid_capstyle='round')

    # Corner highlighting
    speed = data['speed_kmh'].values
    for c in corners:
        entry_m = c['entry_dist']
        exit_m = c['exit_dist']
        mask = (dist >= entry_m) & (dist <= exit_m)
        cidx = np.where(mask)[0]
        if len(cidx) < 2:
            continue

        # Highlight corner zone
        ax.plot(x[cidx], y[cidx], color=c['color'], lw=10,
                alpha=0.5, zorder=3, solid_capstyle='round')

        # Place marker at ACTUAL speed minimum within the zone,
        # not the JSON apex_m (which may differ from this lap)
        zone_speeds = speed[cidx]
        actual_apex_idx = cidx[np.argmin(zone_speeds)]

        ax.scatter(x[actual_apex_idx], y[actual_apex_idx],
                   color=c['color'], s=300, zorder=8,
                   edgecolors='white', linewidths=2)
        ax.text(x[actual_apex_idx], y[actual_apex_idx],
                str(c['id']), color='white', fontsize=9,
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
        print(f"\n  Slowest: T{s['id']} {s['name']} "
              f"({s['min_speed']:.0f} km/h)")
    if summary['fastest']:
        f = summary['fastest']
        print(f"  Fastest: T{f['id']} {f['name']} "
              f"({f['min_speed']:.0f} km/h)")


def main():
    args = parse_args()

    print("=" * 60)
    print("  F1 Lap Insight - Step 2: Lap Analysis")
    print("=" * 60)

    data, meta = load_and_prepare(args.csv)

    # Track detection: CLI flag → SRT metadata → auto-detect by length
    if args.track:
        track = load_track(args.track)
    elif meta.get('track_id'):
        try:
            track = load_track(meta['track_id'])
            print(f"  Track from CSV metadata: {track.name}")
        except FileNotFoundError:
            track = auto_detect_track(meta['track_length'])
    else:
        track = auto_detect_track(meta['track_length'])

    if track is None:
        print("  ERROR: Cannot detect track. Use --track flag.")
        sys.exit(1)

    print(f"  Track: {track.name} ({track.n_corners} corners)")

    corners = analyze_solo(data, track)
    print_summary(corners, track, meta)

    # Save plots
    fig1 = plot_corner_speeds(corners, track, meta)
    save_figure(fig1, OUTPUT_DIR / f"{track.short}_corner_speeds.png",
                dpi=DPI_SAVE)

    fig2 = plot_corner_map(data, corners, track, meta)
    if fig2:
        save_figure(fig2, OUTPUT_DIR / f"{track.short}_corner_map.png",
                    dpi=DPI_SAVE)

    print(f"\n{'=' * 60}")
    print(f"  COMPLETE!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()