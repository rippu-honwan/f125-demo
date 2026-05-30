"""
F1 Lap Insight - Step 3: Lap Comparison  (v2 - Graded Visual)
One-page dashboard: WHERE and HOW MUCH you lose time vs a real F1 driver.suzuka
python scripts/03_lap_comparison.py data/my_lap.csv --lap 1 \--driver VER --year 2025 --session Q --track suzuka


Layout:
  [HEADER]  Your time  vs  Pro time  |  Delta
  [ROW 0]   Speed traces + graded corner bands (wide)  |  Scorecard
  [ROW 1]   Cumulative time delta                      |  Corner bars
  [ROW 2]   Throttle trace  |  Brake trace             |  AI Tips

Grade per corner (time delta inside that corner):
  S  < -0.05s   (you are faster than pro)
  A  -0.05~+0.05s
  B  +0.05~+0.15s
  C  +0.15~+0.30s
  D  +0.30~+0.50s
  F  > +0.50s
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.collections import LineCollection

from src.pipeline import make_parser, run_pipeline
from src.utils import smooth, format_laptime, format_delta
from src.plotting import COLORS, style_axis, save_figure, delta_color
from config import OUTPUT_DIR, ensure_output_dir


# ============================================================
# Grade system
# ============================================================

GRADE_THRESHOLDS = [
    (-999,  -0.05, 'S',  '#00ffcc', 'FASTER than Pro'),
    (-0.05,  0.05, 'A',  '#00cc66', 'Excellent'),
    ( 0.05,  0.15, 'B',  '#88dd00', 'Good'),
    ( 0.15,  0.30, 'C',  '#ffaa00', 'Needs work'),
    ( 0.30,  0.50, 'D',  '#ff6600', 'Significant loss'),
    ( 0.50,  999,  'F',  '#ff2244', 'Major loss'),
]

def grade_corner(delta_s):
    for lo, hi, letter, color, label in GRADE_THRESHOLDS:
        if lo <= delta_s < hi:
            return letter, color, label
    return 'F', '#ff2244', 'Major loss'


def compute_corner_deltas(aligned, corners):
    """Return list of (corner_dict, delta_s, grade, color, label)."""
    if not corners:
        return []
    td   = aligned['time_delta'].values
    dist = aligned['lap_distance'].values
    out  = []
    for c in corners:
        entry  = c.get('entry_m', c['apex_m'] - 50)
        exit_m = c.get('exit_m',  c['apex_m'] + 50)
        ei = int(np.clip(np.searchsorted(dist, entry),  0, len(td)-1))
        xi = int(np.clip(np.searchsorted(dist, exit_m), 0, len(td)-1))
        d  = float(td[xi] - td[ei])
        g, gc, gl = grade_corner(d)
        out.append((c, d, g, gc, gl))
    return out


# ============================================================
# Panel A: Speed comparison with graded corner bands
# ============================================================

def plot_speed_panel(ax, aligned, corners, track_len, corner_data):
    dist     = aligned['lap_distance'].values
    game_spd = smooth(aligned['game_speed_kmh'].values, 10)
    real_spd = smooth(aligned['real_speed_kmh'].values, 10)
    ymax     = max(game_spd.max(), real_spd.max()) * 1.10

    style_axis(ax, ylabel='Speed (km/h)',
               title='SPEED  —  Green = You   Red = Pro')

    # Graded corner bands + corner labels
    for c, d, g, gc, gl in corner_data:
        entry  = c.get('entry_m', c['apex_m'] - 50)
        exit_m = c.get('exit_m',  c['apex_m'] + 50)
        mid    = (entry + exit_m) / 2
        ax.axvspan(entry, exit_m, color=gc, alpha=0.11, zorder=0)
        ax.text(mid, ymax * 0.985,
                f"{c.get('short','?')}", ha='center', va='top',
                fontsize=6, color=gc, fontweight='bold', zorder=6)
        ax.text(mid, ymax * 0.935,
                g, ha='center', va='top',
                fontsize=8, color=gc, fontweight='bold', zorder=6)

    # Fill between traces
    ax.fill_between(dist, game_spd, real_spd,
                    where=(game_spd >= real_spd),
                    color=COLORS['faster'], alpha=0.13, zorder=1)
    ax.fill_between(dist, game_spd, real_spd,
                    where=(game_spd < real_spd),
                    color=COLORS['slower'], alpha=0.13, zorder=1)

    ax.plot(dist, real_spd, color=COLORS['real'],
            lw=1.8, alpha=0.88, label='Pro', zorder=4)
    ax.plot(dist, game_spd, color=COLORS['game'],
            lw=1.8, alpha=0.88, label='You', zorder=5)

    ax.set_ylim(0, ymax)
    ax.set_xlim(0, track_len)
    ax.legend(loc='lower right', fontsize=8, ncol=2,
              facecolor=COLORS['card_bg'], edgecolor=COLORS['grid'],
              labelcolor='white')


# ============================================================
# Panel B: Cumulative time delta
# ============================================================

def plot_time_delta_panel(ax, aligned, corners, track_len, corner_data):
    dist = aligned['lap_distance'].values
    td   = aligned['time_delta'].values

    style_axis(ax, ylabel='Cumulative delta (s)',
               title='TIME DELTA  (above 0 = you are SLOWER)')
    ax.axhline(0, color='#555577', lw=1.0, ls='--')

    # Segment line colour by local delta rate
    seg_deltas = np.diff(td)
    points     = np.array([dist, td]).T.reshape(-1, 1, 2)
    segs       = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_colors = [grade_corner(float(sd) * 12)[1] for sd in seg_deltas]
    lc = LineCollection(segs, colors=seg_colors, linewidth=2.5, zorder=4)
    ax.add_collection(lc)

    ax.fill_between(dist, td, where=(td >= 0),
                    color=COLORS['slower'], alpha=0.08, zorder=1)
    ax.fill_between(dist, td, where=(td <  0),
                    color=COLORS['faster'], alpha=0.08, zorder=1)

    ylo = min(td.min() * 1.15, -0.05)
    yhi = max(td.max() * 1.15,  0.05)
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(0, track_len)

    # Grade dots at each corner apex
    for c, d, g, gc, gl in corner_data:
        apex = c['apex_m']
        i    = int(np.clip(np.searchsorted(dist, apex), 0, len(td)-1))
        ax.scatter(dist[i], td[i], s=60, color=gc,
                   edgecolors='white', linewidths=0.8, zorder=6)
        ax.text(dist[i], td[i] + (yhi - ylo)*0.045,
                g, ha='center', va='bottom',
                fontsize=7, color=gc, fontweight='bold', zorder=7)

    # Final annotation
    final = float(td[-1])
    sign  = '+' if final >= 0 else ''
    ax.annotate(
        f'{sign}{final:.3f}s',
        xy=(dist[-1], final),
        xytext=(-8, 14 if final >= 0 else -18),
        textcoords='offset points',
        fontsize=10, fontweight='bold',
        color=delta_color(final),
        bbox=dict(boxstyle='round,pad=0.35',
                  facecolor=COLORS['card_bg'],
                  edgecolor=delta_color(final), alpha=0.95))


# ============================================================
# Panel C: Corner grade bar chart
# ============================================================

def plot_corner_grade_bars(ax, corner_data):
    if not corner_data:
        ax.text(0.5, 0.5, 'No corner data', transform=ax.transAxes,
                ha='center', color='white', fontsize=12)
        return

    style_axis(ax, ylabel='Time lost (s)',
               title='CORNER GRADES  —  per corner time loss')

    names   = [c.get('short', f"C{c['id']}") for c, *_ in corner_data]
    deltas  = [d  for _, d, *_ in corner_data]
    grades  = [g  for _, _, g, *_ in corner_data]
    gcolors = [gc for _, _, _, gc, *_ in corner_data]
    x       = np.arange(len(names))

    bars = ax.bar(x, deltas, color=gcolors, alpha=0.82,
                  width=0.65, edgecolor='#ffffff22', linewidth=0.5, zorder=3)
    ax.axhline(0, color='#555577', lw=0.8)

    for bar, d, g, gc in zip(bars, deltas, grades, gcolors):
        xc = bar.get_x() + bar.get_width() / 2
        if d >= 0:
            ax.text(xc, d + 0.006, g,
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=gc, zorder=5)
            ax.text(xc, d + 0.024, f'{d:+.2f}',
                    ha='center', va='bottom',
                    fontsize=6, color='#bbbbbb', zorder=5)
        else:
            ax.text(xc, d - 0.006, g,
                    ha='center', va='top',
                    fontsize=9, fontweight='bold', color=gc, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=40, ha='right')

    total = sum(deltas)
    ax.text(0.98, 0.96, f'Total: {total:+.3f}s',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, fontweight='bold', color=delta_color(total),
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor=COLORS['card_bg'],
                      edgecolor=delta_color(total), alpha=0.9))


# ============================================================
# Panel D: Throttle / Brake input traces
# ============================================================

def plot_input_panel(ax, aligned, track_len,
                     corner_data, channel, title_str):
    dist = aligned['lap_distance'].values
    style_axis(ax, ylabel=title_str, title=title_str.upper())

    # Light grade shading
    for c, d, g, gc, gl in corner_data:
        entry  = c.get('entry_m', c['apex_m'] - 50)
        exit_m = c.get('exit_m',  c['apex_m'] + 50)
        ax.axvspan(entry, exit_m, color=gc, alpha=0.07, zorder=0)

    real_col = f'real_{channel}'
    game_col = f'game_{channel}'

    if real_col in aligned.columns:
        sig = smooth(aligned[real_col].values, 5)
        ax.fill_between(dist, sig, alpha=0.18,
                        color=COLORS['real'], zorder=1)
        ax.plot(dist, sig, color=COLORS['real'],
                lw=1.3, alpha=0.88, label='Pro', zorder=3)

    if game_col in aligned.columns:
        sig = smooth(aligned[game_col].values, 5)
        ax.plot(dist, sig, color=COLORS['game'],
                lw=1.3, alpha=0.88, label='You', zorder=4)

    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(0, track_len)
    ax.legend(loc='lower right', fontsize=7,
              facecolor=COLORS['card_bg'],
              edgecolor=COLORS['grid'], labelcolor='white')


# ============================================================
# Side tile: Scorecard
# ============================================================

def plot_scorecard(ax, corner_data, game_meta, real_meta, overall_td):
    ax.set_facecolor(COLORS['card_bg'])
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    border = FancyBboxPatch((0.03, 0.03), 0.94, 0.94,
                            boxstyle='round,pad=0.02',
                            facecolor=COLORS['card_bg'],
                            edgecolor='#445566', linewidth=1.5)
    ax.add_patch(border)

    oc   = delta_color(overall_td)
    sign = '+' if overall_td >= 0 else ''
    ax.text(0.5, 0.93, 'SCORECARD', ha='center',
            fontsize=9, color='#aaaacc', fontweight='bold')
    ax.text(0.5, 0.81, f'{sign}{overall_td:.3f}s',
            ha='center', fontsize=20, fontweight='bold', color=oc)
    ax.text(0.5, 0.71, format_laptime(game_meta['best_time']),
            ha='center', fontsize=9,  color='white')
    ax.text(0.5, 0.63, f"vs {format_laptime(real_meta['lap_time'])}",
            ha='center', fontsize=8, color='#aaaacc')

    # Grade tally bars
    ax.text(0.5, 0.55, 'GRADE TALLY', ha='center',
            fontsize=7, color='#778899')
    tally = {g: 0 for _, _, g, _, _ in GRADE_THRESHOLDS}
    for _, _, g, *_ in corner_data:
        tally[g] = tally.get(g, 0) + 1

    n_total = max(len(corner_data), 1)
    y = 0.49
    for _, _, g, gc, gl in GRADE_THRESHOLDS:
        count = tally.get(g, 0)
        if count == 0:
            continue
        bar_w = (count / n_total) * 0.68
        ax.add_patch(FancyBboxPatch(
            (0.14, y - 0.028), bar_w, 0.042,
            boxstyle='round,pad=0.004',
            facecolor=gc, alpha=0.75, edgecolor='none'))
        ax.text(0.11, y - 0.007, g, ha='right', va='center',
                fontsize=8, fontweight='bold', color=gc)
        ax.text(0.14 + bar_w + 0.03, y - 0.007,
                f'x{count}', va='center',
                fontsize=7, color='#cccccc')
        y -= 0.075

    # Top 3 worst corners
    if corner_data:
        worst = sorted(corner_data, key=lambda r: r[1], reverse=True)[:3]
        ax.text(0.5, y - 0.01, 'TOP 3 PRIORITY', ha='center',
                fontsize=7, color='#778899')
        y -= 0.065
        for rank, (c, d, g, gc, gl) in enumerate(worst, 1):
            name = c.get('name', c.get('short', '?'))
            ax.text(0.10, y, f'{rank}. {name}',
                    fontsize=6.5, color=gc, fontweight='bold')
            ax.text(0.90, y, f'{d:+.3f}s',
                    ha='right', fontsize=6.5, color=gc)
            y -= 0.062


# ============================================================
# Side tile: Grade scale reference
# ============================================================

def plot_grade_scale(ax):
    ax.set_facecolor(COLORS['card_bg'])
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.03, 0.03), 0.94, 0.94,
                                boxstyle='round,pad=0.02',
                                facecolor=COLORS['card_bg'],
                                edgecolor='#445566', linewidth=1.5))
    ax.text(0.5, 0.95, 'GRADE SCALE', ha='center',
            fontsize=9, color='#aaaacc', fontweight='bold')
    y = 0.84
    for _, lo, g, gc, gl in GRADE_THRESHOLDS:
        ax.add_patch(FancyBboxPatch(
            (0.06, y - 0.055), 0.88, 0.065,
            boxstyle='round,pad=0.008',
            facecolor=gc, alpha=0.16,
            edgecolor=gc, linewidth=1.3))
        ax.text(0.14, y - 0.022, g,
                fontsize=12, fontweight='bold',
                color=gc, va='center')
        ax.text(0.28, y - 0.022, gl,
                fontsize=7.5, color='#dddddd', va='center')
        y -= 0.13


# ============================================================
# Side tile: AI Coach tips
# ============================================================

def plot_ai_tips(ax, corner_data, report):
    ax.set_facecolor(COLORS['card_bg'])
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.03, 0.03), 0.94, 0.94,
                                boxstyle='round,pad=0.02',
                                facecolor=COLORS['card_bg'],
                                edgecolor='#445566', linewidth=1.5))
    ax.text(0.5, 0.95, 'AI COACH TIPS', ha='center',
            fontsize=9, color='#aaaacc', fontweight='bold')

    tips = []
    if report and hasattr(report, 'corner_insights'):
        ci_map = {ci.corner_id: ci for ci in report.corner_insights}
        worst4 = sorted(corner_data, key=lambda r: r[1], reverse=True)[:4]
        for c, d, g, gc, gl in worst4:
            ci = ci_map.get(c.get('id', 0))
            tip_text = (ci.tips[0] if (ci and ci.tips)
                        else f'Lost {d:+.3f}s — focus here')
            tips.append((c.get('short', '?'), gc, tip_text))
    elif corner_data:
        worst4 = sorted(corner_data, key=lambda r: r[1], reverse=True)[:4]
        for c, d, g, gc, gl in worst4:
            tips.append((c.get('short', '?'), gc,
                         f'Lost {d:+.3f}s — focus here'))

    y = 0.86
    for short, gc, tip in tips[:4]:
        ax.text(0.07, y, f'[{short}]',
                fontsize=8, color=gc, fontweight='bold', va='top')
        words = tip.split()
        line, lines_out = '', []
        for w in words:
            test = (line + ' ' + w).strip()
            if len(test) > 30:
                if line:
                    lines_out.append(line)
                line = w
            else:
                line = test
        if line:
            lines_out.append(line)
        for i, l in enumerate(lines_out[:2]):
            ax.text(0.07, y - 0.055 - i * 0.052,
                    l, fontsize=6.5, color='#cccccc', va='top')
        y -= 0.22


# ============================================================
# Main
# ============================================================

def main():
    parser = make_parser('F1 Lap Insight - Graded Lap Comparison')
    args = parser.parse_args()

    print('=' * 60)
    print('  F1 LAP INSIGHT  —  Graded Lap Comparison')
    print('=' * 60)

    aligned, corners, report, game_meta, real_meta = run_pipeline(args)

    track_len   = game_meta['track_length']
    corner_data = compute_corner_deltas(aligned, corners)
    overall_td  = float(aligned['time_delta'].values[-1])

    print(f'\n  Generating graded comparison chart...')

    # ---- Layout: 3 rows × 3 cols (last col narrower) ----
    fig = plt.figure(figsize=(30, 22))
    fig.set_facecolor(COLORS['bg'])

    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.38, wspace=0.22,
        top=0.91, bottom=0.04,
        left=0.05, right=0.97,
        width_ratios=[1.0, 1.0, 0.36],
    )

    # Row 0: speed (wide) | scorecard
    plot_speed_panel(fig.add_subplot(gs[0, :2]),
                     aligned, corners, track_len, corner_data)
    plot_scorecard(fig.add_subplot(gs[0, 2]),
                   corner_data, game_meta, real_meta, overall_td)

    # Row 1: time delta | corner bars | grade scale
    plot_time_delta_panel(fig.add_subplot(gs[1, 0]),
                          aligned, corners, track_len, corner_data)
    plot_corner_grade_bars(fig.add_subplot(gs[1, 1]), corner_data)
    plot_grade_scale(fig.add_subplot(gs[1, 2]))

    # Row 2: throttle | brake | AI tips
    plot_input_panel(fig.add_subplot(gs[2, 0]),
                     aligned, track_len, corner_data,
                     'throttle', 'Throttle')
    plot_input_panel(fig.add_subplot(gs[2, 1]),
                     aligned, track_len, corner_data,
                     'brake', 'Brake')
    plot_ai_tips(fig.add_subplot(gs[2, 2]), corner_data, report)

    # ---- Header ----
    driver = real_meta['driver']
    year   = real_meta['year']
    gp     = real_meta['gp_name']
    sess   = real_meta['session']

    fig.suptitle(
        f"YOU  {format_laptime(game_meta['best_time'])}    "
        f"vs    {driver}  {format_laptime(real_meta['lap_time'])}"
        f"   \u2014   {year} {gp} {sess}",
        color='white', fontsize=15, fontweight='bold', y=0.966
    )
    oc   = delta_color(overall_td)
    sign = '+' if overall_td >= 0 else ''
    fig.text(
        0.5, 0.946,
        f'Overall Delta: {sign}{overall_td:.3f}s   '
        f'{"▲ SLOWER" if overall_td > 0 else "▼ FASTER"}',
        ha='center', fontsize=11, fontweight='bold', color=oc,
    )

    # ---- Save ----
    out = OUTPUT_DIR / f'comparison_graded_{driver}_{year}_{sess}.png'
    save_figure(fig, out, dpi=200)
    print(f'\n  Saved: {out}')


if __name__ == '__main__':
    ensure_output_dir()
    main()