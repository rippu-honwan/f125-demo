"""
F1 Lap Insight - Step 2: Track Map with Official Corner Names
Suzuka International Racing Course - 18 Turns

Usage:
    python scripts/02_track_map.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import json

from config import (
    DEFAULT_CSV, OUTPUT_DIR, BG_COLOR, DPI_SAVE,
    CORNER_COLORS, FIGSIZE_MAP, FIGSIZE_SPEED
)
from src.utils import (
    load_telemetry, resample, smooth,
    format_laptime, print_session_summary
)

# ============================================================
# GROUND TRUTH: 18 Official Suzuka Corners
# ============================================================

SUZUKA_CORNERS = [
    # turn  name             short   entry    apex    exit
    (1,  "Turn 1",          "T1",    612,     707,    976),
    (2,  "Turn 2",          "T2",    655,     829,    912),
    (3,  "S Curve 1",       "S1",   1032,    1132,   1304),
    (4,  "S Curve 2",       "S2",   1033,    1239,   1304),
    (5,  "S Curve 3",       "S3",   1322,    1375,   1453),
    (6,  "S Curve 4",       "S4",   1486,    1575,   1664),
    (7,  "Dunlop",          "DUN",  1687,    1787,   1870),
    (8,  "Degner 1",        "DG1",  2268,    2304,   2355),
    (9,  "Degner 2",        "DG2",  2425,    2467,   2514),
    (10, "200R",            "200R", 2729,    2793,   2830),
    (11, "Hairpin",         "HAIR", 2896,    2938,   2972),
    (12, "Spoon Entry",     "SP1",  3424,    3560,   3688),
    (13, "Spoon Exit",      "SP2",  3756,    3816,   3964),
    (14, "Spoon Kink",      "SPK",  3964,    3964,   4120),
    (15, "130R",            "130R", 4920,    4992,   5191),
    (16, "Casio Triangle 1","CT1",  5363,    5410,   5431),
    (17, "Casio Triangle 2","CT2",  5431,    5479,   5500),
    (18, "Final Corner",    "FIN",  5500,    5553,   5741),
]


# ============================================================
# Build Corner Objects
# ============================================================

def build_corners(data):
    distance = data['lap_distance'].values
    corners = []

    for i, (turn, name, short, entry, apex, exit_) in enumerate(SUZUKA_CORNERS):
        apex_idx = int(np.argmin(np.abs(distance - apex)))
        entry_idx = int(np.argmin(np.abs(distance - entry)))
        exit_idx = int(np.argmin(np.abs(distance - exit_)))

        speed = float(data['speed_kmh'].iloc[apex_idx])
        wx = float(data['world_position_X'].iloc[apex_idx])
        wy = float(data['world_position_Y'].iloc[apex_idx])

        # Min speed in corner zone
        if exit_idx > entry_idx:
            zone = data['speed_kmh'].iloc[entry_idx:exit_idx + 1]
            min_speed = float(zone.min())
            min_idx = int(zone.idxmin())
        else:
            min_speed = speed
            min_idx = apex_idx

        # Direction from steering
        if 'steering' in data.columns:
            steer = data['steering'].iloc[
                max(0, apex_idx - 30):min(len(data), apex_idx + 30)
            ].mean()
            direction = "LEFT" if steer > 0.02 else "RIGHT" if steer < -0.02 else "STRAIGHT"
        else:
            direction = "UNKNOWN"

        # Intensity
        if min_speed < 100:
            intensity = "SLOW"
        elif min_speed < 160:
            intensity = "MEDIUM"
        elif min_speed < 250:
            intensity = "FAST"
        else:
            intensity = "FLAT-OUT"

        corners.append({
            'turn': turn, 'name': name, 'short': short,
            'color': CORNER_COLORS[i],
            'entry_dist': float(entry), 'apex_dist': float(apex),
            'exit_dist': float(exit_),
            'apex_idx': apex_idx,
            'apex_speed': speed, 'min_speed': min_speed,
            'min_speed_dist': float(distance[min_idx]),
            'world_x': wx, 'world_y': wy,
            'direction': direction, 'intensity': intensity,
        })

    return corners


# ============================================================
# Plot 1: Track Map
# ============================================================

def plot_track_map(data, corners, meta):
    x = data['world_position_X'].values
    y = data['world_position_Y'].values
    speed = data['speed_kmh'].values
    dist = data['lap_distance'].values

    fig, ax = plt.subplots(figsize=FIGSIZE_MAP)
    fig.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Track layers
    ax.plot(x, y, color='#0f0f0f', lw=18, zorder=1, solid_capstyle='round')
    ax.plot(x, y, color='#2a2a2a', lw=12, zorder=2, solid_capstyle='round')

    # Speed coloring
    cmap = plt.get_cmap('RdYlGn')
    s_min = max(float(speed.min()), 50)
    s_max = min(float(speed.max()), 340)
    norm = plt.Normalize(s_min, s_max)
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = mcoll.LineCollection(segs, colors=cmap(norm(speed))[:-1], linewidth=7)
    ax.add_collection(lc)

    cx_t = float(np.mean(x))
    cy_t = float(np.mean(y))
    tr = max(float(x.max() - x.min()), float(y.max() - y.min()))

    # Corner labels
    used_pos = []
    for c in corners:
        ci = c['apex_idx']
        cx, cy = float(x[ci]), float(y[ci])
        color = c['color']

        # Highlight section
        mask = (dist >= c['entry_dist']) & (dist <= c['exit_dist'])
        cidx = np.where(mask)[0]
        if len(cidx) > 2:
            ax.plot(x[cidx], y[cidx], color=color, lw=13,
                    alpha=0.45, zorder=3, solid_capstyle='round')

        # Circle
        ax.scatter(cx, cy, color=color, s=550, zorder=8,
                   marker='o', edgecolors='white', linewidths=2.5)
        ax.text(cx, cy, str(c['turn']), color='white', fontsize=12,
                fontweight='bold', ha='center', va='center', zorder=9)

        # Direction
        dir_sym = "<< L" if c['direction'] == "LEFT" else \
                  "R >>" if c['direction'] == "RIGHT" else "--"

        # Label offset
        dx = cx - cx_t
        dy = cy - cy_t
        d = max(np.sqrt(dx ** 2 + dy ** 2), 1.0)
        nx, ny = dx / d, dy / d
        ld = tr * 0.14

        for _ in range(15):
            tx = cx + nx * ld
            ty = cy + ny * ld
            overlap = any(
                abs(tx - px) < tr * 0.06 and abs(ty - py) < tr * 0.045
                for px, py in used_pos
            )
            if not overlap:
                break
            ang = 0.3
            nx2 = nx * np.cos(ang) - ny * np.sin(ang)
            ny2 = nx * np.sin(ang) + ny * np.cos(ang)
            nx, ny = nx2, ny2
            ld *= 1.08

        tx = cx + nx * ld
        ty = cy + ny * ld
        used_pos.append((tx, ty))

        label = (f"T{c['turn']}  {c['name']}\n"
                 f"{c['min_speed']:.0f} km/h  {dir_sym}\n"
                 f"{c['intensity']}")

        ax.annotate(
            label, xy=(cx, cy), xytext=(tx, ty),
            fontsize=7.5, color='white', fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=color,
                      edgecolor='white', alpha=0.88, lw=1.5),
            arrowprops=dict(arrowstyle='->', color='white', lw=1.8,
                            connectionstyle='arc3,rad=0.08'),
            zorder=15
        )

    # Start/Finish
    sx, sy = float(x[0]), float(y[0])
    ax.scatter(sx, sy, color='white', s=350, zorder=12,
               marker='s', edgecolors='lime', lw=3)
    sf_dx, sf_dy = sx - cx_t, sy - cy_t
    sf_d = max(np.sqrt(sf_dx ** 2 + sf_dy ** 2), 1.0)
    sf_off = tr * 0.09

    ax.annotate(
        'START / FINISH',
        xy=(sx, sy),
        xytext=(sx + sf_dx / sf_d * sf_off, sy + sf_dy / sf_d * sf_off),
        fontsize=14, color='lime', fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor=BG_COLOR,
                  edgecolor='lime', alpha=0.95, lw=2.5),
        arrowprops=dict(arrowstyle='->', color='lime', lw=3),
        zorder=20
    )

    # Direction arrow
    ai = int(np.argmin(np.abs(dist - 300)))
    step = 60
    if ai + step < len(x):
        ax.annotate('',
                    xy=(float(x[ai + step]), float(y[ai + step])),
                    xytext=(float(x[ai]), float(y[ai])),
                    arrowprops=dict(arrowstyle='->', color='yellow',
                                   lw=4, mutation_scale=25),
                    zorder=12)

    # Distance markers
    for d_val in range(500, int(dist.max()), 500):
        idx = int(np.argmin(np.abs(dist - d_val)))
        mx, my = float(x[idx]), float(y[idx])
        odx, ody = mx - cx_t, my - cy_t
        od = max(np.sqrt(odx ** 2 + ody ** 2), 1.0)
        off = tr * 0.02
        is_km = d_val % 1000 == 0
        fc = 'white' if is_km else '#555555'
        fs = 8 if is_km else 5
        label = f'{d_val // 1000}km' if is_km else f'{d_val}m'
        ax.text(mx + odx / od * off, my + ody / od * off,
                label, color=fc, fontsize=fs, ha='center', va='center',
                alpha=0.7)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.018, pad=0.03, shrink=0.5)
    cbar.set_label('Speed (km/h)', color='white', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    ax.set_title(
        f'SUZUKA International Racing Course\n'
        f'18-Turn Circuit Map  |  {meta["track_length"]:.0f}m  |  '
        f'{format_laptime(meta["best_time"])}',
        color='white', fontsize=18, fontweight='bold', pad=25
    )

    # Bottom legend
    parts = [f"T{c['turn']}: {c['name']} ({c['min_speed']:.0f})"
             for c in corners]
    line1 = "  |  ".join(parts[:9])
    line2 = "  |  ".join(parts[9:])
    fig.text(0.5, 0.012, f"{line1}\n{line2}",
             color='white', fontsize=7.5, ha='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e',
                       edgecolor='#444444', alpha=0.95))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


# ============================================================
# Plot 2: Speed Profile
# ============================================================

def plot_speed_profile(data, corners, meta):
    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE_SPEED,
                             gridspec_kw={'height_ratios': [4, 1.3, 1]},
                             sharex=True)
    fig.set_facecolor(BG_COLOR)

    dist = data['lap_distance'].values
    speed = data['speed_kmh'].values
    speed_s = smooth(speed, 25)

    ax1 = axes[0]
    ax1.set_facecolor(BG_COLOR)
    ax1.plot(dist, speed, color='#00ff88', lw=0.7, alpha=0.3)
    ax1.plot(dist, speed_s, color='#00ff88', lw=2.5)

    for c in corners:
        ax1.axvspan(c['entry_dist'], c['exit_dist'],
                    alpha=0.12, color=c['color'], zorder=1)
        ax1.axvline(x=c['apex_dist'], color=c['color'], lw=0.8, alpha=0.5)
        ax1.text(c['apex_dist'], 365, f"T{c['turn']}\n{c['short']}",
                 color='white', fontsize=6, ha='center', va='bottom',
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.12', facecolor=c['color'],
                           alpha=0.75, edgecolor='none'))
        ax1.annotate(f"{c['min_speed']:.0f}",
                     xy=(c['min_speed_dist'], c['min_speed']),
                     xytext=(c['min_speed_dist'], c['min_speed'] - 20),
                     fontsize=6, color='white', ha='center',
                     fontweight='bold', alpha=0.8,
                     arrowprops=dict(arrowstyle='-', color=c['color'],
                                    lw=0.5, alpha=0.5))

    for d_val in range(0, int(dist.max()) + 500, 500):
        ax1.axvline(x=d_val, color='#222222', lw=0.5)
    for d_val in range(0, int(dist.max()) + 1000, 1000):
        ax1.axvline(x=d_val, color='#444444', lw=1)

    ax1.set_ylabel('Speed (km/h)', color='white', fontsize=12)
    ax1.set_ylim(0, 400)
    ax1.tick_params(colors='white')
    ax1.grid(alpha=0.08, color='white')
    ax1.set_title(
        f'SUZUKA - Speed Profile  |  {format_laptime(meta["best_time"])}',
        color='white', fontsize=15, fontweight='bold'
    )

    ax2 = axes[1]
    ax2.set_facecolor(BG_COLOR)
    if 'steering' in data.columns:
        steer = data['steering'].values
        ax2.fill_between(dist, steer, where=steer > 0.02,
                         color='#4488ff', alpha=0.4, label='Left')
        ax2.fill_between(dist, steer, where=steer < -0.02,
                         color='#ff4444', alpha=0.4, label='Right')
        ax2.plot(dist, steer, color='#666666', lw=0.6)
    for c in corners:
        ax2.axvspan(c['entry_dist'], c['exit_dist'],
                    alpha=0.06, color=c['color'])
    ax2.axhline(y=0, color='#444444', lw=0.5)
    ax2.set_ylabel('Steering', color='white', fontsize=10)
    ax2.tick_params(colors='white')
    ax2.legend(loc='upper right', facecolor='#1a1a2e',
               edgecolor='white', labelcolor='white', fontsize=8)

    ax3 = axes[2]
    ax3.set_facecolor(BG_COLOR)
    if 'brake' in data.columns:
        ax3.fill_between(dist, data['brake'].values,
                         color='#ff4444', alpha=0.5, label='Brake')
    if 'throttle' in data.columns:
        ax3.fill_between(dist, data['throttle'].values,
                         color='#44ff44', alpha=0.3, label='Throttle')
    for c in corners:
        ax3.axvspan(c['entry_dist'], c['exit_dist'],
                    alpha=0.06, color=c['color'])
    ax3.set_ylabel('Pedals', color='white', fontsize=10)
    ax3.set_xlabel('Track Distance (m)', color='white', fontsize=12)
    ax3.set_ylim(0, 1.05)
    ax3.tick_params(colors='white')
    ax3.legend(loc='upper right', facecolor='#1a1a2e',
               edgecolor='white', labelcolor='white', fontsize=8)

    plt.tight_layout()
    return fig


# ============================================================
# Save JSON
# ============================================================

def save_json(corners, meta):
    out = {
        "track": meta['track_name'],
        "track_length_m": meta['track_length'],
        "lap_time": format_laptime(meta['best_time']),
        "total_corners": len(corners),
        "corners": [
            {
                "turn": c['turn'], "name": c['name'], "short": c['short'],
                "entry_dist_m": c['entry_dist'],
                "apex_dist_m": c['apex_dist'],
                "exit_dist_m": c['exit_dist'],
                "apex_speed_kmh": round(c['apex_speed'], 1),
                "min_speed_kmh": round(c['min_speed'], 1),
                "direction": c['direction'],
                "intensity": c['intensity'],
                "world_x": round(c['world_x'], 2),
                "world_y": round(c['world_y'], 2),
            }
            for c in corners
        ]
    }

    path = OUTPUT_DIR / f"{meta['track_name']}_corners_FINAL.json"
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {path}")
    return path


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  F1 Lap Insight - Step 2: Track Map")
    print("=" * 60)

    # ---- Load ----
    best_lap, meta = load_telemetry(DEFAULT_CSV)
    print_session_summary(meta)

    # ---- Resample ----
    data = resample(best_lap, meta['track_length'])
    print(f"\n  Resampled: {len(data)} points")

    # ---- Corners ----
    corners = build_corners(data)
    print(f"  Corners: {len(corners)}")

    # Sanity check
    hairpin = [c for c in corners if c['name'] == 'Hairpin']
    r130 = [c for c in corners if c['name'] == '130R']
    if hairpin and hairpin[0]['min_speed'] > 120:
        print(f"  !! Hairpin {hairpin[0]['min_speed']:.0f} km/h > 120 !!")
    if r130 and r130[0]['min_speed'] < 250:
        print(f"  !! 130R {r130[0]['min_speed']:.0f} km/h < 250 !!")
    else:
        print(f"  Sanity check: ALL PASSED")

    # Print table
    print(f"\n  {'Turn':<5} {'Name':<20} {'Min Spd':>8} {'Dir':<7} {'Type'}")
    print(f"  {'-' * 5} {'-' * 20} {'-' * 8} {'-' * 7} {'-' * 9}")
    for c in corners:
        print(f"  T{c['turn']:<3} {c['name']:<20} "
              f"{c['min_speed']:>7.0f}km/h "
              f"{c['direction']:<7} {c['intensity']}")

    # ---- Save JSON ----
    save_json(corners, meta)

    # ---- Plot 1: Map ----
    print(f"\n  Drawing map...")
    fig1 = plot_track_map(data, corners, meta)
    p1 = OUTPUT_DIR / f"{meta['track_name']}_FINAL_map.png"
    fig1.savefig(p1, dpi=DPI_SAVE, bbox_inches='tight')
    print(f"  Saved: {p1}")
    plt.close(fig1)

    # ---- Plot 2: Speed ----
    print(f"\n  Drawing speed profile...")
    fig2 = plot_speed_profile(data, corners, meta)
    p2 = OUTPUT_DIR / f"{meta['track_name']}_FINAL_speed.png"
    fig2.savefig(p2, dpi=DPI_SAVE, bbox_inches='tight')
    print(f"  Saved: {p2}")
    plt.close(fig2)

    print(f"\n{'=' * 60}")
    print(f"  Done! Map: {p1}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()