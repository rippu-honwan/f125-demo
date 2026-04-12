#!/usr/bin/env python3
"""
F1 Lap Insight - Step 3: Lap Comparison
Compare your game telemetry with real F1 driver data.

Usage:
    python scripts/03_lap_comparison.py data/my_lap.csv \
        --driver VER --year 2024 --session Q --track suzuka
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

from src.loader import load_and_prepare
from src.fastf1_loader import load_real_telemetry
from src.alignment import align_two_pass
from src.utils import smooth, format_laptime, format_delta, calculate_time_delta
from config import OUTPUT_DIR, BG_COLOR


# ============================================================
# Plotting helpers
# ============================================================

def style_axis(ax, ylabel=None, xlabel=None, title=None):
    """Apply consistent dark styling."""
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors='white', labelsize=8)
    ax.grid(alpha=0.08, color='white')
    for spine in ax.spines.values():
        spine.set_color('#333333')
    if ylabel:
        ax.set_ylabel(ylabel, color='white', fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color='white', fontsize=9)
    if title:
        ax.set_title(title, color='white', fontsize=10, fontweight='bold',
                     loc='left', pad=8)


def add_corner_shading(ax, corners, track_length, alpha=0.06):
    """Add corner zones as background shading."""
    if corners is None:
        return
    for c in corners:
        entry = c.get('entry_m', c['apex_m'] - 50)
        exit_m = c.get('exit_m', c['apex_m'] + 50)
        if entry < track_length and exit_m > 0:
            ax.axvspan(entry, exit_m, alpha=alpha, color='#ffffff')


# ============================================================
# Panel: Speed comparison
# ============================================================

def plot_speed_comparison(ax, aligned, corners=None, track_length=None):
    """Speed traces overlay."""
    dist = aligned['lap_distance'].values
    game_spd = smooth(aligned['game_speed_kmh'].values, 10)
    real_spd = smooth(aligned['real_speed_kmh'].values, 10)

    style_axis(ax, ylabel='Speed (km/h)', title='🏎️  SPEED COMPARISON')

    if corners:
        add_corner_shading(ax, corners, track_length)

    ax.plot(dist, real_spd, color='#ff4444', lw=1.8, alpha=0.85,
            label='Real F1', zorder=5)
    ax.plot(dist, game_spd, color='#00ff88', lw=1.8, alpha=0.85,
            label='Game (You)', zorder=4)

    # Shade delta
    ax.fill_between(dist, game_spd, real_spd,
                    where=game_spd >= real_spd,
                    color='#00ff88', alpha=0.08, label='You faster')
    ax.fill_between(dist, game_spd, real_spd,
                    where=game_spd < real_spd,
                    color='#ff4444', alpha=0.08, label='Real faster')

    ax.set_ylim(0, max(game_spd.max(), real_spd.max()) * 1.08)

    leg = ax.legend(loc='upper right', fontsize=7, ncol=2,
                    facecolor='#1a1a2e', edgecolor='#333333',
                    labelcolor='white')
    leg.set_zorder(20)


# ============================================================
# Panel: Speed delta
# ============================================================

def plot_speed_delta(ax, aligned, corners=None, track_length=None):
    """Speed difference (game - real)."""
    dist = aligned['lap_distance'].values
    delta = smooth(aligned['speed_delta'].values, 15)

    style_axis(ax, ylabel='Δ Speed (km/h)', title='📊  SPEED DELTA (You − Real)')

    if corners:
        add_corner_shading(ax, corners, track_length)

    ax.fill_between(dist, delta, where=delta >= 0,
                    color='#00ff88', alpha=0.4)
    ax.fill_between(dist, delta, where=delta < 0,
                    color='#ff4444', alpha=0.4)
    ax.axhline(0, color='#666666', lw=0.8)

    # Stats text
    avg_delta = np.mean(delta)
    max_faster = np.max(delta)
    max_slower = np.min(delta)
    ax.text(0.02, 0.92,
            f'Avg: {avg_delta:+.1f} km/h  |  '
            f'Max faster: +{max_faster:.0f}  |  Max slower: {max_slower:.0f}',
            transform=ax.transAxes, color='white', fontsize=7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                     edgecolor='#333333', alpha=0.8))


# ============================================================
# Panel: Time delta
# ============================================================

def plot_time_delta(ax, aligned, corners=None, track_length=None):
    """Cumulative time delta."""
    dist = aligned['lap_distance'].values
    time_d = aligned['time_delta'].values

    style_axis(ax, ylabel='Δ Time (s)', title='⏱️  CUMULATIVE TIME DELTA')

    if corners:
        add_corner_shading(ax, corners, track_length)

    color = '#00ff88' if time_d[-1] <= 0 else '#ff4444'
    ax.plot(dist, time_d, color=color, lw=2)
    ax.fill_between(dist, time_d, alpha=0.15, color=color)
    ax.axhline(0, color='#666666', lw=0.8)

    # Annotate final delta
    final = time_d[-1]
    sign = "FASTER" if final < 0 else "SLOWER"
    ax.annotate(f'{final:+.3f}s ({sign})',
                xy=(dist[-1], final),
                fontsize=9, fontweight='bold',
                color=color,
                ha='right', va='bottom' if final > 0 else 'top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                         edgecolor=color, alpha=0.9))


# ============================================================
# Panel: Throttle comparison
# ============================================================

def plot_throttle_comparison(ax, aligned, corners=None, track_length=None):
    """Throttle traces overlay."""
    dist = aligned['lap_distance'].values

    style_axis(ax, ylabel='Throttle', title='🦶  THROTTLE')

    if corners:
        add_corner_shading(ax, corners, track_length)

    has_game = 'game_throttle' in aligned.columns
    has_real = 'real_throttle' in aligned.columns

    if has_real:
        real_thr = smooth(aligned['real_throttle'].values, 5)
        ax.fill_between(dist, real_thr, alpha=0.2, color='#ff4444')
        ax.plot(dist, real_thr, color='#ff4444', lw=1.2, alpha=0.8,
                label='Real')

    if has_game:
        game_thr = smooth(aligned['game_throttle'].values, 5)
        ax.plot(dist, game_thr, color='#00ff88', lw=1.2, alpha=0.8,
                label='Game')

    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc='lower right', fontsize=7,
              facecolor='#1a1a2e', edgecolor='#333333',
              labelcolor='white')


# ============================================================
# Panel: Brake comparison
# ============================================================

def plot_brake_comparison(ax, aligned, corners=None, track_length=None):
    """Brake traces overlay."""
    dist = aligned['lap_distance'].values

    style_axis(ax, ylabel='Brake', title='🛑  BRAKE')

    if corners:
        add_corner_shading(ax, corners, track_length)

    has_game = 'game_brake' in aligned.columns
    has_real = 'real_brake' in aligned.columns

    if has_real:
        real_brk = aligned['real_brake'].values
        ax.fill_between(dist, real_brk, alpha=0.25, color='#ff4444',
                        label='Real')

    if has_game:
        game_brk = aligned['game_brake'].values
        ax.fill_between(dist, game_brk, alpha=0.25, color='#00ff88',
                        label='Game')

    ax.set_ylim(-0.05, 1.3)
    ax.legend(loc='upper right', fontsize=7,
              facecolor='#1a1a2e', edgecolor='#333333',
              labelcolor='white')


# ============================================================
# Panel: Gear comparison
# ============================================================

def plot_gear_comparison(ax, aligned, corners=None, track_length=None):
    """Gear traces overlay."""
    dist = aligned['lap_distance'].values

    style_axis(ax, ylabel='Gear', xlabel='Distance (m)',
               title='⚙️  GEAR')

    if corners:
        add_corner_shading(ax, corners, track_length)

    has_game = 'game_gear' in aligned.columns
    has_real = 'real_gear' in aligned.columns

    if has_real:
        real_gear = aligned['real_gear'].values
        ax.step(dist, real_gear, color='#ff4444', lw=1.2, alpha=0.8,
                where='mid', label='Real')

    if has_game:
        game_gear = aligned['game_gear'].values
        ax.step(dist, game_gear, color='#00ff88', lw=1.2, alpha=0.8,
                where='mid', label='Game')

    ax.set_ylim(0, 9)
    ax.set_yticks(range(1, 9))
    ax.legend(loc='lower right', fontsize=7,
              facecolor='#1a1a2e', edgecolor='#333333',
              labelcolor='white')


# ============================================================
# Panel: Corner-by-corner analysis
# ============================================================

def plot_corner_analysis(ax, aligned, corners, track_length):
    """Bar chart of time gain/loss per corner."""
    if corners is None or len(corners) == 0:
        ax.text(0.5, 0.5, 'No corner data available',
                transform=ax.transAxes, ha='center', va='center',
                color='white', fontsize=12)
        return

    style_axis(ax, ylabel='Δ Time (s)', xlabel='Corner',
               title='🏁  CORNER-BY-CORNER DELTA')

    time_d = aligned['time_delta'].values
    dist = aligned['lap_distance'].values

    names = []
    deltas = []
    colors = []

    for c in corners:
        entry = c.get('entry_m', c['apex_m'] - 50)
        exit_m = c.get('exit_m', c['apex_m'] + 50)

        entry_idx = np.searchsorted(dist, entry)
        exit_idx = np.searchsorted(dist, exit_m)

        if entry_idx >= len(time_d) or exit_idx >= len(time_d):
            continue

        delta = time_d[min(exit_idx, len(time_d) - 1)] - \
                time_d[min(entry_idx, len(time_d) - 1)]

        name = c.get('short', c.get('name', f"C{c['id']}"))
        names.append(name)
        deltas.append(delta)
        colors.append('#00ff88' if delta <= 0 else '#ff4444')

    if not names:
        return

    x = np.arange(len(names))
    bars = ax.bar(x, deltas, color=colors, alpha=0.7, width=0.6,
                  edgecolor='white', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6, rotation=45, ha='right')
    ax.axhline(0, color='#666666', lw=0.8)

    # Value labels
    for bar, delta in zip(bars, deltas):
        y = bar.get_height()
        va = 'bottom' if y >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                f'{delta:+.3f}', ha='center', va=va,
                fontsize=6, color='white', fontweight='bold')

    # Total
    total = sum(deltas)
    ax.text(0.98, 0.92,
            f'Total corner delta: {total:+.3f}s',
            transform=ax.transAxes, ha='right', fontsize=8,
            color='#00ff88' if total <= 0 else '#ff4444',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                     edgecolor='#333333', alpha=0.8))


# ============================================================
# Summary header
# ============================================================

def add_summary_header(fig, game_meta, real_meta, aligned):
    """Add summary info at top of figure."""
    time_d = aligned['time_delta'].values
    final_delta = time_d[-1]

    game_time = game_meta.get('best_time', 0)
    real_time = real_meta.get('lap_time', 0)

    title = (
        f"YOUR LAP  {format_laptime(game_time)}    vs    "
        f"{real_meta['driver']}  {format_laptime(real_time)}  "
        f"({real_meta['year']} {real_meta['gp_name']} {real_meta['session']})"
    )

    delta_str = format_delta(final_delta)
    delta_color = '#00ff88' if final_delta <= 0 else '#ff4444'

    fig.suptitle(title, color='white', fontsize=14, fontweight='bold',
                 y=0.98)

    fig.text(0.5, 0.955,
             f'Overall Delta: {delta_str}',
             ha='center', fontsize=12, fontweight='bold',
             color=delta_color)


# ============================================================
# Load corners from track JSON
# ============================================================

def load_corners(track_name):
    """Load corner definitions from track JSON."""
    import json
    track_dir = PROJECT_ROOT / "tracks"

    if track_name is None:
        return None

    # Try exact match
    json_path = track_dir / f"{track_name.lower()}.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        return data.get('corners', None)

    # Try partial match
    for p in track_dir.glob("*.json"):
        if track_name.lower() in p.stem.lower():
            with open(p) as f:
                data = json.load(f)
            return data.get('corners', None)

    print(f"  ⚠️  No track JSON for '{track_name}'")
    return None


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='F1 Lap Insight - Compare with real F1 data'
    )
    parser.add_argument('csv', nargs='?', default='data/my_lap.csv',
                        help='Path to your telemetry CSV')
    parser.add_argument('--driver', type=str, default='VER',
                        help='F1 driver code (e.g. VER, HAM, LEC)')
    parser.add_argument('--year', type=int, default=2024,
                        help='Season year (default: 2024)')
    parser.add_argument('--session', type=str, default='Q',
                        help='Session type: Q, R, FP1, FP2, FP3')
    parser.add_argument('--track', type=str, default=None,
                        help='Track short name (e.g. suzuka)')
    parser.add_argument('--gp', type=str, default=None,
                        help='Full GP name (e.g. "Japanese Grand Prix")')
    parser.add_argument('--no-corners', action='store_true',
                        help='Skip corner analysis')
    args = parser.parse_args()

    print("=" * 60)
    print("  F1 Lap Insight - Step 3: Lap Comparison")
    print("=" * 60)

    # ---- Load game data ----
    print(f"\n  [1/4] Loading game data...")
    game_data, game_meta = load_and_prepare(args.csv)
    game_length = game_meta['track_length']

    # ---- Auto-detect track if not specified ----
    if args.track is None and args.gp is None:
        # Try to detect from CSV metadata
        raw = pd.read_csv(args.csv, sep='\t', nrows=1)
        if 'trackId' in raw.columns:
            track_id = str(raw['trackId'].iloc[0]).strip().lower()
            args.track = track_id
            print(f"  Auto-detected track: {track_id}")
        else:
            print("  ⚠️  Cannot auto-detect track.")
            print("  Use --track suzuka or --gp 'Japanese Grand Prix'")
            sys.exit(1)

    # ---- Load real F1 data ----
    print(f"\n  [2/4] Loading real F1 data...")
    try:
        real_data, real_meta = load_real_telemetry(
            driver=args.driver,
            year=args.year,
            session=args.session,
            track=args.track,
            gp=args.gp,
        )
    except Exception as e:
        print(f"\n  ❌ Failed to load real data: {e}")
        sys.exit(1)

    real_length = real_meta['track_length']

    # ---- Align data ----
    print(f"\n  [3/4] Aligning telemetry...")
    aligned = align_two_pass(
        game_data, game_length,
        real_data, real_length,
        verbose=True
    )

    # Add time delta
    aligned = calculate_time_delta(aligned)

    # ---- Load corners ----
    corners = None
    if not args.no_corners:
        track_name = args.track or args.gp
        corners = load_corners(track_name)
        if corners:
            print(f"  Loaded {len(corners)} corners for {track_name}")

    # ---- Plot ----
    print(f"\n  [4/4] Generating comparison charts...")

    fig = plt.figure(figsize=(28, 24))
    fig.set_facecolor(BG_COLOR)

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.32, wspace=0.15,
                           top=0.93, bottom=0.04,
                           left=0.05, right=0.97)

    track_len = game_length

    # Row 1: Speed comparison + Speed delta
    ax1 = fig.add_subplot(gs[0, :])
    plot_speed_comparison(ax1, aligned, corners, track_len)

    ax2 = fig.add_subplot(gs[1, :])
    plot_speed_delta(ax2, aligned, corners, track_len)

    # Row 2: Time delta (full width)
    ax3 = fig.add_subplot(gs[2, 0])
    plot_time_delta(ax3, aligned, corners, track_len)

    # Row 2 right: Corner analysis
    ax4 = fig.add_subplot(gs[2, 1])
    plot_corner_analysis(ax4, aligned, corners, track_len)

    # Row 3: Throttle + Brake
    ax5 = fig.add_subplot(gs[3, 0])
    plot_throttle_comparison(ax5, aligned, corners, track_len)

    ax6 = fig.add_subplot(gs[3, 1])
    plot_brake_comparison(ax6, aligned, corners, track_len)

    # Header
    add_summary_header(fig, game_meta, real_meta, aligned)

    # Save
    driver = args.driver
    year = args.year
    session = args.session
    out_path = OUTPUT_DIR / f"step3_comparison_{driver}_{year}_{session}.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor=BG_COLOR)
    print(f"\n  ✅ Saved: {out_path}")
    plt.close(fig)

    # ---- Print summary ----
    time_d = aligned['time_delta'].values
    final_delta = time_d[-1]

    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  Your lap:     {format_laptime(game_meta['best_time'])}")
    print(f"  {driver} lap:   {format_laptime(real_meta['lap_time'])}")
    print(f"  Delta:        {format_delta(final_delta)}")
    print(f"  Track length: Game {game_length:.0f}m / Real {real_length:.0f}m")

    # Corner summary
    if corners:
        dist = aligned['lap_distance'].values
        print(f"\n  {'Corner':<12} {'Delta':>8}  {'Note'}")
        print(f"  {'-'*12} {'-'*8}  {'-'*20}")

        for c in corners:
            entry = c.get('entry_m', c['apex_m'] - 50)
            exit_m = c.get('exit_m', c['apex_m'] + 50)
            ei = np.searchsorted(dist, entry)
            xi = np.searchsorted(dist, exit_m)
            if ei >= len(time_d) or xi >= len(time_d):
                continue
            d = time_d[min(xi, len(time_d)-1)] - time_d[min(ei, len(time_d)-1)]
            name = c.get('short', f"C{c['id']}")
            note = "✅ Faster" if d < -0.02 else ("❌ Slower" if d > 0.02 else "≈ Equal")
            color_code = '\033[92m' if d < -0.02 else ('\033[91m' if d > 0.02 else '\033[93m')
            print(f"  {name:<12} {d:>+7.3f}s  {note}")

    # Biggest gain / loss
    if corners:
        dists_vals = []
        for c in corners:
            entry = c.get('entry_m', c['apex_m'] - 50)
            exit_m = c.get('exit_m', c['apex_m'] + 50)
            ei = np.searchsorted(dist, entry)
            xi = np.searchsorted(dist, exit_m)
            if ei >= len(time_d) or xi >= len(time_d):
                continue
            d = time_d[min(xi, len(time_d)-1)] - time_d[min(ei, len(time_d)-1)]
            name = c.get('short', f"C{c['id']}")
            dists_vals.append((name, d))

        if dists_vals:
            dists_vals.sort(key=lambda x: x[1])
            best = dists_vals[0]
            worst = dists_vals[-1]
            print(f"\n  🏆 Best corner:  {best[0]} ({best[1]:+.3f}s)")
            print(f"  ⚠️  Worst corner: {worst[0]} ({worst[1]:+.3f}s)")

    print(f"\n{'='*60}")

    # ---- Coaching Report ----
    if corners:
        from src.coaching import generate_coaching_report, print_coaching_report
        print(f"\n  Generating coaching report...")
        report = generate_coaching_report(
            aligned, corners, game_meta, real_meta
        )
        print_coaching_report(report)

if __name__ == "__main__":
    main()