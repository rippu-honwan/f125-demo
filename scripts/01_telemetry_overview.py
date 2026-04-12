"""
F1 Lap Insight - Step 1: Telemetry Overview
Quick visualization of your game telemetry data.

Usage:
    python scripts/01_telemetry_overview.py [csv_path]
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import DEFAULT_CSV, OUTPUT_DIR, BG_COLOR, DPI_SAVE
from src.loader import load_and_prepare
from src.track import auto_detect_track
from src.utils import smooth, format_laptime


def parse_args():
    parser = argparse.ArgumentParser(description="F1 Lap Insight - Telemetry Overview")
    parser.add_argument('csv', nargs='?', default=DEFAULT_CSV,
                        help='Path to telemetry CSV')
    return parser.parse_args()


def plot_overview(data, meta, track=None):
    fig = plt.figure(figsize=(24, 16))
    fig.set_facecolor(BG_COLOR)
    gs = GridSpec(3, 2, hspace=0.35, wspace=0.25)

    dist = data['lap_distance'].values
    track_name = track.name if track else "Unknown Track"

    # Speed
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(BG_COLOR)
    ax1.plot(dist, smooth(data['speed_kmh'].values, 10),
             color='#00ff88', lw=2)

    if track:
        for c in track.corners:
            ax1.axvspan(c.entry_m, c.exit_m, alpha=0.05, color=c.color)
            ax1.text(c.apex_m, 5, f"T{c.id}", color='white',
                     fontsize=6, ha='center', alpha=0.5)

    ax1.set_ylabel('Speed (km/h)', color='white', fontsize=11)
    ax1.set_title(
        f'{track_name} - Telemetry Overview  |  '
        f'Lap Time: {format_laptime(meta["best_time"])}',
        color='white', fontsize=14, fontweight='bold'
    )
    ax1.tick_params(colors='white')
    ax1.grid(alpha=0.08, color='white')

    # Throttle
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(BG_COLOR)
    if 'throttle' in data.columns:
        ax2.fill_between(dist, data['throttle'].values,
                         color='#00ff88', alpha=0.5)
        ax2.set_ylabel('Throttle', color='white')
    ax2.set_ylim(-0.05, 1.1)
    ax2.tick_params(colors='white')
    ax2.set_title('Throttle', color='white', fontsize=11)

    # Brake
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(BG_COLOR)
    if 'brake' in data.columns:
        ax3.fill_between(dist, data['brake'].values,
                         color='#ff4444', alpha=0.5)
        ax3.set_ylabel('Brake', color='white')
    ax3.set_ylim(-0.05, 1.1)
    ax3.tick_params(colors='white')
    ax3.set_title('Brake', color='white', fontsize=11)

    # Gear
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor(BG_COLOR)
    if 'gear' in data.columns:
        ax4.plot(dist, data['gear'].values, color='#42a5f5', lw=1.5)
        ax4.set_ylabel('Gear', color='white')
    ax4.tick_params(colors='white')
    ax4.set_title('Gear', color='white', fontsize=11)

    # Track map
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(BG_COLOR)
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
        ax5.set_title('Track Map (speed)', color='white', fontsize=11)
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
        print(f"  Auto-detected track: {track.name} ({track.n_corners} corners)")
    else:
        print(f"  Track not detected (length: {meta['track_length']:.0f}m)")

    fig = plot_overview(data, meta, track)

    track_short = track.short if track else "unknown"
    path = OUTPUT_DIR / f"{track_short}_overview.png"
    fig.savefig(path, dpi=DPI_SAVE, bbox_inches='tight')
    print(f"\n  Saved: {path}")
    plt.close(fig)

    print(f"\n{'=' * 60}")
    print(f"  COMPLETE!")
    print(f"  Lap: {format_laptime(meta['best_time'])}")
    print(f"  Track: {meta['track_length']:.0f}m")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()