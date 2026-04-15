#!/usr/bin/env python3
"""
F1 Lap Insight - Step 1: Telemetry Overview
Quick visualization of your game telemetry data + lap time table.

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
from matplotlib.patches import FancyBboxPatch

from config import DEFAULT_CSV, OUTPUT_DIR, DPI_SAVE
from src.loader import load_and_prepare, get_lap_summary
from src.track import auto_detect_track, load_track
from src.utils import smooth, format_laptime
from src.plotting import COLORS, style_axis, save_figure


def parse_args():
    parser = argparse.ArgumentParser(
        description="F1 Lap Insight - Telemetry Overview"
    )
    parser.add_argument('csv', nargs='?', default=DEFAULT_CSV,
                        help='Path to telemetry CSV')
    return parser.parse_args()


def get_sectors_from_track(track):
    """Extract sector boundaries from track object."""
    if track is None:
        return None
    return [{'start_m': s.start_m, 'end_m': s.end_m}
            for s in track.sectors]


def print_lap_table(summaries, best_idx=None):
    """Print a formatted lap time table to terminal."""
    if not summaries:
        print("  No complete laps found.")
        return

    n_sectors = len(summaries[0]['sector_times'])
    sec_headers = ''.join(f'{"S"+str(i+1):>9}' for i in range(n_sectors))

    print(f"\n  {'Lap':>4} {'lapIndex':>9} {'Lap Time':>12} "
          f"{sec_headers} {'Max Spd':>9}")
    print(f"  {'-'*4} {'-'*9} {'-'*12} "
          f"{''.join('-'*9 for _ in range(n_sectors))} {'-'*9}")

    for s in summaries:
        sec_strs = ''.join(f'{t:>9.3f}' for t in s['sector_times'])
        marker = ' *' if s['lap_index'] == best_idx else ''
        print(f"  {s['lap_number']:>4} {s['lap_index']:>9} "
              f"{format_laptime(s['lap_time']):>12} "
              f"{sec_strs} {s['max_speed']:>8.0f}{marker}")

    if best_idx is not None:
        print(f"\n  * = fastest lap (auto-selected for analysis)")
    print(f"\n  Tip: use --lap <lapIndex> in scripts 02-04 "
          f"to analyze a specific lap")


def plot_lap_table(summaries, best_idx=None):
    """Generate a visual lap time table as a matplotlib figure."""
    if not summaries:
        return None

    n_laps = len(summaries)
    n_sectors = len(summaries[0]['sector_times'])

    fig, ax = plt.subplots(figsize=(14, max(3, 1.5 + n_laps * 0.7)))
    fig.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    ax.axis('off')

    headers = ['Lap', 'lapIndex', 'Lap Time']
    headers += [f'S{i+1}' for i in range(n_sectors)]
    headers += ['Max Speed']
    n_cols = len(headers)

    col_x = np.linspace(0.02, 0.98, n_cols + 1)
    col_centers = (col_x[:-1] + col_x[1:]) / 2

    y_start = 0.92
    row_h = 0.75 / max(n_laps + 1, 2)

    for j, h in enumerate(headers):
        ax.text(col_centers[j], y_start, h,
                ha='center', va='center', fontsize=10,
                fontweight='bold', color='white')

    ax.plot([0.02, 0.98], [y_start - row_h * 0.5, y_start - row_h * 0.5],
            color=COLORS['grid'], lw=0.5)

    best_sectors = [999] * n_sectors
    for s in summaries:
        for i, t in enumerate(s['sector_times']):
            if 0 < t < best_sectors[i]:
                best_sectors[i] = t
    best_lap_time = min(s['lap_time'] for s in summaries)

    for row, s in enumerate(summaries):
        y = y_start - (row + 1.2) * row_h
        is_best = s['lap_index'] == best_idx

        if is_best:
            bg = FancyBboxPatch(
                (0.01, y - row_h * 0.35), 0.98, row_h * 0.7,
                boxstyle="round,pad=0.005",
                facecolor=COLORS['game'], alpha=0.08,
                edgecolor=COLORS['game'], linewidth=0.5)
            ax.add_patch(bg)

        values = [
            str(s['lap_number']),
            str(s['lap_index']),
            format_laptime(s['lap_time']),
        ]
        values += [f"{t:.3f}" for t in s['sector_times']]
        values += [f"{s['max_speed']:.0f}"]

        for j, v in enumerate(values):
            color = 'white'
            if j == 2 and s['lap_time'] == best_lap_time:
                color = COLORS['game']
            elif 3 <= j < 3 + n_sectors:
                sec_idx = j - 3
                if (s['sector_times'][sec_idx] > 0 and
                        abs(s['sector_times'][sec_idx] -
                            best_sectors[sec_idx]) < 0.001):
                    color = COLORS['game']

            weight = 'bold' if is_best else 'normal'
            ax.text(col_centers[j], y, v,
                    ha='center', va='center', fontsize=9,
                    color=color, fontweight=weight,
                    fontfamily='monospace')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('LAP TIMES', color='white', fontsize=13,
                 fontweight='bold', loc='left', pad=10)

    plt.tight_layout()
    return fig


def plot_overview(data, meta, track=None):
    fig = plt.figure(figsize=(24, 16))
    fig.set_facecolor(COLORS['bg'])
    gs = GridSpec(3, 2, hspace=0.35, wspace=0.25)

    dist = data['lap_distance'].values
    track_name = track.name if track else "Unknown Track"
    lap_label = f"lapIndex={meta.get('lap_index', '?')}"

    ax1 = fig.add_subplot(gs[0, :])
    style_axis(ax1, ylabel='Speed (km/h)',
               title=f'{track_name} - Telemetry Overview  |  '
                     f'Lap: {format_laptime(meta["best_time"])} '
                     f'({lap_label})')

    ax1.plot(dist, smooth(data['speed_kmh'].values, 10),
             color=COLORS['game'], lw=2)

    if track:
        for c in track.corners:
            ax1.axvspan(c.entry_m, c.exit_m, alpha=0.05, color=c.color)
            ax1.text(c.apex_m, 5, f"T{c.id}", color='white',
                     fontsize=6, ha='center', alpha=0.5)

    ax2 = fig.add_subplot(gs[1, 0])
    style_axis(ax2, ylabel='Throttle', title='Throttle')
    if 'throttle' in data.columns:
        ax2.fill_between(dist, data['throttle'].values,
                         color=COLORS['game'], alpha=0.5)
    ax2.set_ylim(-0.05, 1.1)

    ax3 = fig.add_subplot(gs[1, 1])
    style_axis(ax3, ylabel='Brake', title='Brake')
    if 'brake' in data.columns:
        ax3.fill_between(dist, data['brake'].values,
                         color=COLORS['slower'], alpha=0.5)
    ax3.set_ylim(-0.05, 1.1)

    ax4 = fig.add_subplot(gs[2, 0])
    style_axis(ax4, ylabel='Gear', title='Gear')
    if 'gear' in data.columns:
        ax4.plot(dist, data['gear'].values, color='#42a5f5', lw=1.5)

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

    track = None
    if meta.get('track_id'):
        try:
            track = load_track(meta['track_id'])
        except FileNotFoundError:
            track = auto_detect_track(meta['track_length'])
    else:
        track = auto_detect_track(meta['track_length'])

    if track:
        print(f"  Track: {track.name} ({track.n_corners} corners)")

    # Lap summary
    sectors = get_sectors_from_track(track)
    summaries = get_lap_summary(args.csv, sectors)
    best_idx = meta.get('lap_index')
    print_lap_table(summaries, best_idx)

    # Save plots
    track_short = track.short if track else "unknown"

    fig_table = plot_lap_table(summaries, best_idx)
    if fig_table:
        save_figure(fig_table,
                    OUTPUT_DIR / f"{track_short}_lap_times.png",
                    dpi=DPI_SAVE)

    fig = plot_overview(data, meta, track)
    save_figure(fig, OUTPUT_DIR / f"{track_short}_overview.png",
                dpi=DPI_SAVE)

    print(f"\n{'=' * 60}")
    print(f"  COMPLETE!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()