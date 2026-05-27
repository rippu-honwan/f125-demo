"""
Track map visualization module.
Draws a 2D track layout with corner-by-corner problem indicators.

Usage (standalone):
    from src.track_map import draw_track_map

    draw_track_map(
        aligned=aligned_df,        # output from pipeline.run_pipeline()
        corners=corners,           # list of corner dicts
        report=coaching_report,    # CoachingReport object
        output_path="output/track_map.png",
        title="Suzuka – You vs VER 2025 Q"
    )

What it draws:
  - Track outline from FastF1 GPS (world_position_X/Y of real data)
  - Each corner zone coloured by severity:
      🟢 Green  = OK (within tolerance)
      🟡 Yellow = Minor issue (1 problem detected)
      🔴 Red    = Major issue (2+ problems or large delta)
  - Corner number labels with small callout boxes
  - Legend explaining colours + icons for issue types
  - Optional: speed heatmap along track line (mode='heatmap')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Any, Optional
import warnings

# ============================================================
# Colour constants (matches plotting.py theme)
# ============================================================
BG       = "#0f0f1a"
TRACK_C  = "#2a2a3e"       # track surface fill
LINE_C   = "#4a4a6a"       # track outline
LABEL_BG = "#1a1a2e"

SEV_COLORS = {
    "ok":     "#00cc66",   # green  – no issues
    "minor":  "#ffaa00",   # amber  – 1 issue
    "major":  "#ff3355",   # red    – 2+ issues / big delta
    "nodata": "#555577",   # grey   – alignment failed
}

ISSUE_ICONS = {
    "brake_early":    "⬆B",   # brakes too early
    "brake_late":     "⬇B",   # brakes too late
    "throttle_late":  "⬇T",   # throttle application late
    "apex_slow":      "⬇V",   # apex speed too low
    "exit_slow":      "⬇E",   # exit speed too low
    "oversteer":      "OS",
    "understeer":     "US",
}


# ============================================================
# Severity helper
# ============================================================

def _severity(ci) -> str:
    """Map a CornerInsight to a severity string."""
    if ci is None:
        return "nodata"
    n_issues = len(getattr(ci, 'issues', []))
    if n_issues == 0:
        return "ok"
    elif n_issues == 1:
        return "minor"
    else:
        return "major"


def _issue_tags(ci) -> List[str]:
    """Return short icon strings for the corner's issues."""
    if ci is None:
        return []
    tags = []
    issues = getattr(ci, 'issues', [])
    for iss in issues:
        iss_lower = str(iss).lower()
        if "brake" in iss_lower and "early" in iss_lower:
            tags.append(ISSUE_ICONS["brake_early"])
        elif "brake" in iss_lower and "late" in iss_lower:
            tags.append(ISSUE_ICONS["brake_late"])
        elif "throttle" in iss_lower:
            tags.append(ISSUE_ICONS["throttle_late"])
        elif "apex" in iss_lower:
            tags.append(ISSUE_ICONS["apex_slow"])
        elif "exit" in iss_lower:
            tags.append(ISSUE_ICONS["exit_slow"])
    return tags[:2]   # max 2 icons per corner label


# ============================================================
# GPS track extraction
# ============================================================

def _extract_track_xy(aligned: pd.DataFrame):
    """
    Pull (X, Y) GPS points from aligned DataFrame.
    FastF1 telemetry stores GPS in world_position_X / world_position_Y.
    Falls back to game positions if real not present.
    """
    # Priority: real F1 GPS → game GPS → legacy names → None
    for xcol, ycol in [
        ("real_world_position_X", "real_world_position_Y"),
        ("game_world_position_X", "game_world_position_Y"),
        ("world_position_X",      "world_position_Y"),
        ("world_x",               "world_y"),           # legacy alignment.py names
    ]:
        if xcol in aligned.columns and ycol in aligned.columns:
            x = aligned[xcol].values.astype(float)
            y = aligned[ycol].values.astype(float)
            # Drop NaN
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() > 100:
                return x[mask], y[mask], aligned['lap_distance'].values[mask]
    return None, None, None


def _smooth_xy(x, y, window=15):
    """Simple moving-average smoothing for GPS noise."""
    from numpy.lib.stride_tricks import sliding_window_view
    if len(x) < window * 2:
        return x, y
    kernel = np.ones(window) / window
    xs = np.convolve(x, kernel, mode='same')
    ys = np.convolve(y, kernel, mode='same')
    return xs, ys


# ============================================================
# Speed-delta heatmap along track line
# ============================================================

def _speed_delta_segments(x, y, dist, aligned, smooth_w=20):
    """
    Build a LineCollection coloured by speed delta (game − real).
    Returns (segments, values) ready for LineCollection.
    """
    if 'game_speed_kmh' not in aligned.columns:
        return None, None

    g_spd = aligned['game_speed_kmh'].values.astype(float)
    r_spd = aligned['real_speed_kmh'].values.astype(float)

    # Resample delta onto the (x,y) grid (which may be subsampled)
    d_full = aligned['lap_distance'].values
    delta = g_spd - r_spd

    # Smooth delta
    kernel = np.ones(smooth_w) / smooth_w
    delta_s = np.convolve(delta, kernel, mode='same')

    # Interpolate delta to our x/y distance array
    delta_at_xy = np.interp(dist, d_full, delta_s)

    # Build segment list
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    vals = (delta_at_xy[:-1] + delta_at_xy[1:]) / 2

    return segs, vals


# ============================================================
# Corner annotation
# ============================================================

def _find_corner_xy(corner_dict, x, y, dist):
    """
    Find the (x, y) screen coordinate closest to the corner apex.
    corner_dict has apex_m key.
    """
    apex_m = corner_dict.get('apex_m', corner_dict.get('apex_dist', 0))
    idx = np.argmin(np.abs(dist - apex_m))
    return float(x[idx]), float(y[idx])


def _direction_offset(xi, yi, x, y, idx, base_offset=60):
    """
    Compute a label offset perpendicular to the track direction.
    Alternates side based on corner index so labels don't overlap.
    """
    n = len(x)
    i = int(np.argmin((x - xi)**2 + (y - yi)**2))
    i = max(2, min(i, n - 3))

    dx = x[i+2] - x[i-2]
    dy = y[i+2] - y[i-2]
    length = np.sqrt(dx**2 + dy**2) + 1e-9

    # Perpendicular
    px = -dy / length
    py =  dx / length

    # Alternate left/right
    sign = 1 if (idx % 2 == 0) else -1
    return px * base_offset * sign, py * base_offset * sign


# ============================================================
# Main drawing function
# ============================================================

def draw_track_map(
    aligned: pd.DataFrame,
    corners: List[Dict],
    report,                        # CoachingReport
    output_path: str = "output/track_map.png",
    title: str = "Track Map – Corner Analysis",
    mode: str = "issues",          # 'issues' | 'heatmap' | 'both'
    figsize: tuple = (16, 14),
    dpi: int = 200,
    show_sector_lines: bool = True,
    track_width: float = 18,       # line width for track surface
) -> plt.Figure:
    """
    Draw the 2D track map with per-corner problem indicators.

    Parameters
    ----------
    aligned      : DataFrame from pipeline (must contain position cols)
    corners      : list of corner dicts (from pipeline.load_corners)
    report       : CoachingReport with .corner_insights list
    output_path  : where to save PNG
    title        : figure title string
    mode         : 'issues'   → colour corners by problem severity
                   'heatmap'  → colour track line by speed delta
                   'both'     → heatmap line + corner labels on top
    figsize      : matplotlib figure size
    dpi          : output resolution
    show_sector_lines : draw sector markers on track

    Returns
    -------
    matplotlib Figure (also saved to output_path)
    """

    # ---- 1. Extract GPS ----
    x, y, dist = _extract_track_xy(aligned)

    if x is None:
        warnings.warn(
            "No GPS position data found in aligned DataFrame.\n"
            "Make sure FastF1 telemetry includes X/Y columns.\n"
            "Falling back to distance-based schematic layout."
        )
        return _draw_schematic_fallback(
            aligned, corners, report, output_path, title, dpi
        )

    x, y = _smooth_xy(x, y, window=12)

    # ---- 2. Build index: corner_id → CornerInsight ----
    ci_map: Dict[int, Any] = {}
    if report is not None and hasattr(report, 'corner_insights'):
        for ci in report.corner_insights:
            ci_map[ci.corner_id] = ci

    # ---- 3. Figure setup ----
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_aspect('equal')
    ax.axis('off')

    # ---- 4. Track surface (thick grey line = tarmac) ----
    ax.plot(x, y, color=TRACK_C, linewidth=track_width,
            solid_capstyle='round', solid_joinstyle='round', zorder=1)
    ax.plot(x, y, color=LINE_C, linewidth=track_width + 4,
            solid_capstyle='round', solid_joinstyle='round', zorder=0)

    # ---- 5. Speed-delta heatmap overlay (optional) ----
    if mode in ('heatmap', 'both'):
        segs, vals = _speed_delta_segments(x, y, dist, aligned)
        if segs is not None:
            vmax = np.percentile(np.abs(vals), 95)
            vmax = max(vmax, 10)
            # Green=faster than pro, Red=slower
            cmap = LinearSegmentedColormap.from_list(
                'delta', ['#ff3355', '#888888', '#00cc66']
            )
            lc = LineCollection(segs, array=vals, cmap=cmap,
                                norm=plt.Normalize(-vmax, vmax),
                                linewidth=track_width * 0.6, zorder=2,
                                capstyle='round')
            ax.add_collection(lc)

            # Colorbar
            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=plt.Normalize(-vmax, vmax)
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.35, pad=0.01,
                                orientation='vertical', aspect=20)
            cbar.set_label('Speed Δ (km/h)\n+ = You faster',
                           color='white', fontsize=9)
            cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

    # ---- 6. Corner annotations ----
    if mode in ('issues', 'both') and corners:
        annotated = []
        for idx, c in enumerate(corners):
            cid = c.get('id', idx + 1)
            ci = ci_map.get(cid)
            sev = _severity(ci)
            color = SEV_COLORS[sev]
            tags = _issue_tags(ci)

            cx, cy = _find_corner_xy(c, x, y, dist)
            ox, oy = _direction_offset(cx, cy, x, y, idx,
                                       base_offset=55)

            # Dot on track at apex
            ax.scatter(cx, cy, s=120, color=color,
                       zorder=5, linewidths=1.5,
                       edgecolors='white')

            # Label box
            short = c.get('short', f"T{cid}")
            icon_str = " ".join(tags) if tags else "✓"
            label = f"{short}\n{icon_str}"

            lx, ly = cx + ox, cy + oy

            # Connecting line
            ax.plot([cx, lx], [cy, ly],
                    color=color, linewidth=0.8, alpha=0.7, zorder=4)

            # Text box
            bbox_props = dict(
                boxstyle="round,pad=0.3",
                facecolor=LABEL_BG,
                edgecolor=color,
                linewidth=1.5,
                alpha=0.92,
            )
            ax.text(lx, ly, label,
                    ha='center', va='center',
                    fontsize=7.5, color=color,
                    fontweight='bold',
                    fontfamily='monospace',
                    bbox=bbox_props,
                    zorder=6)

            annotated.append((cid, sev))

    # ---- 7. Start/Finish line ----
    sf_idx = 0  # Start/finish is at distance ≈ 0
    ax.scatter(x[sf_idx], y[sf_idx], s=250, color='white',
               marker='D', zorder=7, linewidths=1)
    ax.text(x[sf_idx] + 20, y[sf_idx] + 20, 'S/F',
            color='white', fontsize=9, fontweight='bold', zorder=8)

    # ---- 8. Sector markers ----
    if show_sector_lines and 'lap_distance' in aligned.columns:
        sector_dists = []
        for col in aligned.columns:
            if 'sector' in col.lower():
                pass  # could extract from track JSON if needed
        # Use standard Suzuka sector splits as fallback
        for s_m, s_label, s_color in [
            (1920, "S1→S2", "#29b6f6"),
            (4090, "S2→S3", "#ab47bc"),
        ]:
            if s_m < dist.max():
                idx = np.argmin(np.abs(dist - s_m))
                ax.scatter(x[idx], y[idx], s=160, color=s_color,
                           marker='s', zorder=7, linewidths=1,
                           edgecolors='white')
                ax.text(x[idx] + 15, y[idx] + 15, s_label,
                        color=s_color, fontsize=8, zorder=8)

    # ---- 9. Legend ----
    legend_elements = [
        mpatches.Patch(color=SEV_COLORS['ok'],     label='✓  No issue'),
        mpatches.Patch(color=SEV_COLORS['minor'],  label='⚠  1 issue'),
        mpatches.Patch(color=SEV_COLORS['major'],  label='✗  2+ issues'),
        mpatches.Patch(color=SEV_COLORS['nodata'], label='?  No data'),
    ]
    if mode in ('issues', 'both'):
        icon_legend = [
            mpatches.Patch(color='white', label='⬆B = Brake too early'),
            mpatches.Patch(color='white', label='⬇B = Brake too late'),
            mpatches.Patch(color='white', label='⬇T = Throttle late'),
            mpatches.Patch(color='white', label='⬇V = Apex speed low'),
            mpatches.Patch(color='white', label='⬇E = Exit speed low'),
        ]
        legend_elements += icon_legend

    leg = ax.legend(
        handles=legend_elements,
        loc='lower left',
        fontsize=8,
        facecolor=LABEL_BG,
        edgecolor='#444466',
        labelcolor='white',
        title='Corner Status',
        title_fontsize=9,
        framealpha=0.9,
    )
    leg.get_title().set_color('white')

    # ---- 10. Title ----
    fig.text(0.5, 0.97, title,
             ha='center', va='top',
             color='white', fontsize=14, fontweight='bold')

    # Time delta summary (if available)
    if report is not None and hasattr(report, 'overall_delta'):
        delta = report.overall_delta
        sign = "+" if delta > 0 else ""
        delta_color = "#ff3355" if delta > 0 else "#00cc66"
        fig.text(0.5, 0.93,
                 f"Overall Δ: {sign}{delta:.3f}s  |  Grade: {report.overall_grade}",
                 ha='center', va='top',
                 color=delta_color, fontsize=11, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # ---- 11. Save ----
    try:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    facecolor=BG)
        print(f"  [track_map] Saved: {output_path}")
    except Exception as e:
        print(f"  [track_map] ⚠ Save failed: {e}")

    return fig


# ============================================================
# Schematic fallback (no GPS)
# ============================================================

def _draw_schematic_fallback(
    aligned, corners, report, output_path, title, dpi
):
    """
    If no GPS data is available, draw a simple oval schematic
    with corner labels around the outside.
    This is a degraded but still useful fallback.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw placeholder oval
    theta = np.linspace(0, 2 * np.pi, 300)
    rx, ry = 5.0, 3.0
    ox = rx * np.cos(theta)
    oy = ry * np.sin(theta)
    ax.plot(ox, oy, color=LINE_C, linewidth=14, zorder=0)
    ax.plot(ox, oy, color=TRACK_C, linewidth=10, zorder=1)

    ci_map: Dict[int, Any] = {}
    if report is not None and hasattr(report, 'corner_insights'):
        for ci in report.corner_insights:
            ci_map[ci.corner_id] = ci

    n = len(corners) if corners else 0
    for idx, c in enumerate(corners or []):
        cid = c.get('id', idx + 1)
        ci = ci_map.get(cid)
        sev = _severity(ci)
        color = SEV_COLORS[sev]
        tags = _issue_tags(ci)

        angle = 2 * np.pi * idx / max(n, 1)
        cx = rx * np.cos(angle)
        cy = ry * np.sin(angle)

        lx = (rx + 1.2) * np.cos(angle)
        ly = (ry + 0.8) * np.sin(angle)

        ax.scatter(cx, cy, s=100, color=color, zorder=5)
        ax.plot([cx, lx], [cy, ly], color=color, lw=0.8, alpha=0.6)

        short = c.get('short', f"T{cid}")
        icon_str = " ".join(tags) if tags else "✓"
        ax.text(lx, ly, f"{short}\n{icon_str}",
                ha='center', va='center', fontsize=7,
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor=LABEL_BG,
                          edgecolor=color, alpha=0.85))

    ax.text(0, 0,
            "⚠ No GPS data\nSchematic only",
            ha='center', va='center',
            color='#888888', fontsize=10,
            style='italic')

    fig.text(0.5, 0.97, title + " [Schematic]",
             ha='center', color='white', fontsize=13, fontweight='bold')

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor=BG)
    print(f"  [track_map] Saved schematic: {output_path}")
    return fig


# ============================================================
# Convenience: called from script 05 or standalone
# ============================================================

def draw_all_modes(aligned, corners, report,
                   output_dir="output", prefix="track",
                   driver="VER", year=2025, session="Q"):
    """
    Draw all three modes and save them.
    Useful for adding to the coaching report workflow.
    """
    from pathlib import Path
    Path(output_dir).mkdir(exist_ok=True)

    base_title = f"Suzuka – You vs {driver} {year} {session}"

    figs = {}
    for mode, suffix in [
        ("issues",  "corner_issues"),
        ("heatmap", "speed_heatmap"),
        ("both",    "combined"),
    ]:
        path = f"{output_dir}/{prefix}_{suffix}_{driver}_{year}_{session}.png"
        figs[mode] = draw_track_map(
            aligned, corners, report,
            output_path=path,
            title=base_title,
            mode=mode,
        )

    return figs
