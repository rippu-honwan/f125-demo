#!/usr/bin/env python3
"""
F1 Lap Insight - Step 1: Telemetry Overview
Quick visualization of your game telemetry data.

Usage:
    python scripts/01_telemetry_overview.py [csv_path]

Changes from original:
  - Uses shared plotting helpers
  - Better error handling
  - Cleaner layout
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import DEFAULT_CSV, OUTPUT_DIR, DPI_SAVE
from src.loader import load_and_prepare
from src.track import auto_detect_track
from src.utils import smooth, format_laptime
from src.plotting import COLORS, style_axis, save_figure


def parse_args():
    parser = argparse.ArgumentParser(
        description="F1 Lap Insight - Telemetry Overview"
    )
    parser.add_argument('csv', nargs='?', default=DEFAULT_CSV,
                        help='Path to telemetry CSV')
    return parser.parse_args()


def plot_overview(data, meta, track=None):
    fig = plt.figure(figsize=(24, 16))
    fig.set_facecolor(COLORS['bg'])
    gs = GridSpec(3, 2, hspace=0.35, wspace=0.25)

    dist = data['lap_distance'].values
    track_name = track.name if track else "Unknown Track"

    # ---- Speed ----
    ax1 = fig.add_subplot(gs[0, :])
    style_axis(ax1, ylabel='Speed (km/h)',
               title=f'{track_name} - Telemetry Overview  |  '
                     f'Lap: {format_laptime(meta["best_time"])}')

    ax1.plot(dist, smooth(data['speed_kmh'].values, 10),
             color=COLORS['game'], lw=2)

    if track:
        for c in track.corners:
            ax1.axvspan(c.entry_m, c.exit_m, alpha=0.05, color=c.color)
            ax1.text(c.apex_m, 5, f"T{c.id}", color='white',
                     fontsize=6, ha='center', alpha=0.5)

    # ---- Throttle ----
    ax2 = fig.add_subplot(gs[1, 0])
    style_axis(ax2, ylabel='Throttle', title='Throttle')
    if 'throttle' in data.columns:
        ax2.fill_between(dist, data['throttle'].values,
                         color=COLORS['game'], alpha=0.5)
    ax2.set_ylim(-0.05, 1.1)

    # ---- Brake ----
    ax3 = fig.add_subplot(gs[1, 1])
    style_axis(ax3, ylabel='Brake', title='Brake')
    if 'brake' in data.columns:
        ax3.fill_between(dist, data['brake'].values,
                         color=COLORS['slower'], alpha=0.5)
    ax3.set_ylim(-0.05, 1.1)

    # ---- Gear ----
    ax4 = fig.add_subplot(gs[2, 0])
    style_axis(ax4, ylabel='Gear', title='Gear')
    if 'gear' in data.columns:
        ax4.plot(dist, data['gear'].values, color='#42a5f5', lw=1.5)

    # ---- Track map ----
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(COLORS['bg'])
    if 'world_position_X' in data.columns:
        x = data['world_position_X'].values
        y = data['world_position_Y'].values
        speed = data['speed_kmh'].values

        cmap = plt.get_cmap('plasma')
        norm = plt.Normalize(speed.min(), speed.max())
        for i in range(len(x) - 1):
            ax5.plot(x[i:i+2], y[i:i+2],
                     color=cmap(norm(speed[i])), lw=3)

        ax5.set_aspect('equal')
        ax5.set_title('Track map (speed)', color='white',
                      fontsize=11, fontweight='bold')
    ax5.set_xticks([])
    ax5.set_yticks([])

    plt.tight_layout()
    return fig


def main():
    args = parse_args()

    print("=" * 60)
    print("  F1 Lap Insight - Step 1: Telemetry Overview")
    print("=" * 60)

    data, meta = load_and_prepare(args.csv)

    track = auto_detect_track(meta['track_length'])
    if track:
        print(f"  Track: {track.name} ({track.n_corners} corners)")
    else:
        print(f"  Track not detected (length: {meta['track_length']:.0f}m)")

    fig = plot_overview(data, meta, track)

    track_short = track.short if track else "unknown"
    path = OUTPUT_DIR / f"{track_short}_overview.png"
    save_figure(fig, path, dpi=DPI_SAVE)

    print(f"\n{'=' * 60}")
    print(f"  Lap: {format_laptime(meta['best_time'])}")
    print(f"  Track: {meta['track_id'] or 'unknown'} "
          f"({meta['track_length']:.0f}m)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()