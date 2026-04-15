#!/usr/bin/env python3
"""
F1 Lap Insight - Step 3: Lap Comparison
Compare your game telemetry with real F1 driver data.

Usage:
    python scripts/03_lap_comparison.py data/my_lap.csv \
        --driver VER --year 2025 --session Q --track suzuka

Changes from original:
  - Uses pipeline.py (eliminates 60 lines of boilerplate)
  - Uses shared plotting.py (eliminates duplicated style helpers)
  - Removed coaching report generation (that's step 04's job)
  - Cleaner panel functions
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.pipeline import make_parser, run_pipeline
from src.utils import smooth, format_laptime, format_delta
from src.plotting import (
    COLORS, style_axis, add_corner_shading,
    save_figure, delta_color,
)
from config import OUTPUT_DIR


# ============================================================
# Panels
# ============================================================

def plot_speed_comparison(ax, aligned, corners, track_len):
    """Speed traces overlay."""
    dist = aligned['lap_distance'].values
    game_spd = smooth(aligned['game_speed_kmh'].values, 10)
    real_spd = smooth(aligned['real_speed_kmh'].values, 10)

    style_axis(ax, ylabel='Speed (km/h)', title='SPEED COMPARISON')
    add_corner_shading(ax, corners, track_len)

    ax.plot(dist, real_spd, color=COLORS['real'], lw=1.8, alpha=0.85,
            label='Real F1', zorder=5)
    ax.plot(dist, game_spd, color=COLORS['game'], lw=1.8, alpha=0.85,
            label='Game (You)', zorder=4)

    ax.fill_between(dist, game_spd, real_spd,
                    where=game_spd >= real_spd,
                    color=COLORS['faster'], alpha=0.08)
    ax.fill_between(dist, game_spd, real_spd,
                    where=game_spd < real_spd,
                    color=COLORS['slower'], alpha=0.08)

    ax.set_ylim(0, max(game_spd.max(), real_spd.max()) * 1.08)
    ax.legend(loc='upper right', fontsize=7, ncol=2,
              facecolor=COLORS['card_bg'], edgecolor=COLORS['grid'],
              labelcolor='white')


def plot_speed_delta(ax, aligned, corners, track_len):
    """Speed difference (game - real)."""
    dist = aligned['lap_distance'].values
    delta = smooth(aligned['speed_delta'].values, 15)

    style_axis(ax, ylabel='Δ Speed (km/h)',
               title='SPEED DELTA (You − Real)')
    add_corner_shading(ax, corners, track_len)

    ax.fill_between(dist, delta, where=delta >= 0,
                    color=COLORS['faster'], alpha=0.4)
    ax.fill_between(dist, delta, where=delta < 0,
                    color=COLORS['slower'], alpha=0.4)
    ax.axhline(0, color='#666666', lw=0.8)

    avg = np.mean(delta)
    ax.text(0.02, 0.92,
            f'Avg: {avg:+.1f} km/h  |  '
            f'Max faster: +{np.max(delta):.0f}  |  '
            f'Max slower: {np.min(delta):.0f}',
            transform=ax.transAxes, color='white', fontsize=7,
            bbox=dict(boxstyle='round,pad=0.3',
                     facecolor=COLORS['card_bg'],
                     edgecolor=COLORS['grid'], alpha=0.8))


def plot_time_delta(ax, aligned, corners, track_len):
    """Cumulative time delta."""
    dist = aligned['lap_distance'].values
    td = aligned['time_delta'].values

    style_axis(ax, ylabel='Δ Time (s)',
               title='CUMULATIVE TIME DELTA')
    add_corner_shading(ax, corners, track_len)

    color = delta_color(td[-1])
    ax.plot(dist, td, color=color, lw=2)
    ax.fill_between(dist, td, alpha=0.15, color=color)
    ax.axhline(0, color='#666666', lw=0.8)

    final = td[-1]
    sign = "FASTER" if final < 0 else "SLOWER"
    ax.annotate(
        f'{final:+.3f}s ({sign})',
        xy=(dist[-1], final), fontsize=9, fontweight='bold',
        color=color, ha='right',
        va='bottom' if final > 0 else 'top',
        bbox=dict(boxstyle='round,pad=0.3',
                 facecolor=COLORS['card_bg'],
                 edgecolor=color, alpha=0.9))


def plot_corner_delta(ax, aligned, corners, track_len):
    """Bar chart of time gain/loss per corner."""
    if not corners:
        ax.text(0.5, 0.5, 'No corner data',
                transform=ax.transAxes, ha='center',
                color='white', fontsize=12)
        return

    style_axis(ax, ylabel='Δ Time (s)', xlabel='Corner',
               title='CORNER-BY-CORNER DELTA')

    td = aligned['time_delta'].values
    dist = aligned['lap_distance'].values

    names, deltas, colors = [], [], []
    for c in corners:
        entry = c.get('entry_m', c['apex_m'] - 50)
        exit_m = c.get('exit_m', c['apex_m'] + 50)
        ei = min(np.searchsorted(dist, entry), len(td) - 1)
        xi = min(np.searchsorted(dist, exit_m), len(td) - 1)
        d = td[xi] - td[ei]
        names.append(c.get('short', f"C{c['id']}"))
        deltas.append(d)
        colors.append(delta_color(d))

    x = np.arange(len(names))
    bars = ax.bar(x, deltas, color=colors, alpha=0.7, width=0.6,
                  edgecolor='white', linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6, rotation=45, ha='right')
    ax.axhline(0, color='#666666', lw=0.8)

    for bar, d in zip(bars, deltas):
        va = 'bottom' if d >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{d:+.3f}', ha='center', va=va,
                fontsize=6, color='white', fontweight='bold')

    total = sum(deltas)
    ax.text(0.98, 0.92, f'Total: {total:+.3f}s',
            transform=ax.transAxes, ha='right', fontsize=8,
            color=delta_color(total), fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3',
                     facecolor=COLORS['card_bg'],
                     edgecolor=COLORS['grid'], alpha=0.8))


def plot_throttle(ax, aligned, corners, track_len):
    """Throttle traces overlay."""
    dist = aligned['lap_distance'].values
    style_axis(ax, ylabel='Throttle', title='THROTTLE')
    add_corner_shading(ax, corners, track_len)

    if 'real_throttle' in aligned.columns:
        ax.fill_between(dist, smooth(aligned['real_throttle'].values, 5),
                        alpha=0.2, color=COLORS['real'])
        ax.plot(dist, smooth(aligned['real_throttle'].values, 5),
                color=COLORS['real'], lw=1.2, alpha=0.8, label='Real')

    if 'game_throttle' in aligned.columns:
        ax.plot(dist, smooth(aligned['game_throttle'].values, 5),
                color=COLORS['game'], lw=1.2, alpha=0.8, label='Game')

    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc='lower right', fontsize=7,
              facecolor=COLORS['card_bg'], edgecolor=COLORS['grid'],
              labelcolor='white')


def plot_brake(ax, aligned, corners, track_len):
    """Brake traces overlay."""
    dist = aligned['lap_distance'].values
    style_axis(ax, ylabel='Brake', title='BRAKE')
    add_corner_shading(ax, corners, track_len)

    if 'real_brake' in aligned.columns:
        ax.fill_between(dist, aligned['real_brake'].values,
                        alpha=0.25, color=COLORS['real'], label='Real')

    if 'game_brake' in aligned.columns:
        ax.fill_between(dist, aligned['game_brake'].values,
                        alpha=0.25, color=COLORS['game'], label='Game')

    ax.set_ylim(-0.05, 1.3)
    ax.legend(loc='upper right', fontsize=7,
              facecolor=COLORS['card_bg'], edgecolor=COLORS['grid'],
              labelcolor='white')


# ============================================================
# Main
# ============================================================

def main():
    parser = make_parser('F1 Lap Insight - Step 3: Lap Comparison')
    args = parser.parse_args()

    print("=" * 60)
    print("  F1 Lap Insight - Step 3: Lap Comparison")
    print("=" * 60)

    # Single function call replaces 60 lines of boilerplate
    aligned, corners, report, game_meta, real_meta = run_pipeline(args)

    track_len = game_meta['track_length']

    # ---- Build figure ----
    print(f"\n  Generating comparison charts...")

    fig = plt.figure(figsize=(28, 24))
    fig.set_facecolor(COLORS['bg'])

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.32, wspace=0.15,
                           top=0.93, bottom=0.04,
                           left=0.05, right=0.97)

    plot_speed_comparison(fig.add_subplot(gs[0, :]),
                          aligned, corners, track_len)
    plot_speed_delta(fig.add_subplot(gs[1, :]),
                     aligned, corners, track_len)
    plot_time_delta(fig.add_subplot(gs[2, 0]),
                    aligned, corners, track_len)
    plot_corner_delta(fig.add_subplot(gs[2, 1]),
                      aligned, corners, track_len)
    plot_throttle(fig.add_subplot(gs[3, 0]),
                  aligned, corners, track_len)
    plot_brake(fig.add_subplot(gs[3, 1]),
               aligned, corners, track_len)

    # Header
    td = aligned['time_delta'].values[-1]
    fig.suptitle(
        f"YOUR LAP  {format_laptime(game_meta['best_time'])}    vs    "
        f"{real_meta['driver']}  {format_laptime(real_meta['lap_time'])}  "
        f"({real_meta['year']} {real_meta['gp_name']} "
        f"{real_meta['session']})",
        color='white', fontsize=14, fontweight='bold', y=0.98
    )
    fig.text(0.5, 0.955, f'Overall Delta: {format_delta(td)}',
             ha='center', fontsize=12, fontweight='bold',
             color=delta_color(td))

    # Save
    out = OUTPUT_DIR / (f"step3_comparison_{args.driver}_"
                        f"{args.year}_{args.session}.png")
    save_figure(fig, out, dpi=200)

    # ---- Print summary ----
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  Your lap:     {format_laptime(game_meta['best_time'])}")
    print(f"  {args.driver} lap:   "
          f"{format_laptime(real_meta['lap_time'])}")
    print(f"  Delta:        {format_delta(td)}")
    print(f"  Track length: Game {track_len:.0f}m / "
          f"Real {real_meta['track_length']:.0f}m")

    # Corner summary table
    if corners:
        dist = aligned['lap_distance'].values
        time_d = aligned['time_delta'].values
        print(f"\n  {'Corner':<12} {'Delta':>8}  Note")
        print(f"  {'-'*12} {'-'*8}  {'-'*20}")

        for c in corners:
            entry = c.get('entry_m', c['apex_m'] - 50)
            exit_m = c.get('exit_m', c['apex_m'] + 50)
            ei = min(np.searchsorted(dist, entry), len(time_d) - 1)
            xi = min(np.searchsorted(dist, exit_m), len(time_d) - 1)
            d = time_d[xi] - time_d[ei]
            name = c.get('short', f"C{c['id']}")
            if d < -0.02:
                note = "✅ Faster"
            elif d > 0.02:
                note = "❌ Slower"
            else:
                note = "≈ Equal"
            print(f"  {name:<12} {d:>+7.3f}s  {note}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()