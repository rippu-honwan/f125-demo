#!/usr/bin/env python3
"""
F1 Lap Insight - Step 4: Visual Coaching Report
Generates multi-page visual coaching report.

Usage:
    python scripts/04_coaching_report.py data/my_lap.csv --lap 0 \
        --driver VER --year 2025 --session Q --track suzuka

Changes from original:
  - Uses pipeline.py (single function call for all data loading)
  - Uses shared plotting.py (no more duplicated helpers)
  - Cleaner page functions with less nesting
  - Skill calculation extracted properly
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
from matplotlib.patches import FancyBboxPatch

from src.pipeline import make_parser, run_pipeline
from src.utils import smooth, format_laptime, format_delta
from src.coaching import (
    print_coaching_report, _generate_action_plan, _estimate_potential,
)
from src.plotting import (
    COLORS, GRADE_COLORS, grade_color, delta_color,
    style_axis, style_card, save_figure, wrap_text,
)
from config import OUTPUT_DIR


# ============================================================
# Skill calculation
# ============================================================

def calculate_skills(report):
    """Calculate driving skill scores from corner insights."""
    buckets = {
        'Braking': [], 'Throttle': [], 'Apex\nSpeed': [],
        'Corner\nExit': [], 'Consistency': [], 'Overall': [],
    }

    for ci in report.corner_insights:
        if ci.brake_diff_m is not None:
            buckets['Braking'].append(
                max(0, 100 - abs(ci.brake_diff_m) * 1.5))
        if ci.throttle_diff_m is not None:
            buckets['Throttle'].append(
                max(0, 100 - abs(ci.throttle_diff_m) * 1.2))
        if ci.apex_speed_diff is not None:
            d = ci.apex_speed_diff
            s = min(100, 80 + d * 0.5) if d >= 0 else max(0, 80 + d * 1.5)
            buckets['Apex\nSpeed'].append(s)
        if ci.exit_speed_diff is not None:
            d = ci.exit_speed_diff
            s = min(100, 80 + d * 0.3) if d >= 0 else max(0, 80 + d * 1.2)
            buckets['Corner\nExit'].append(s)
        if ci.time_delta is not None:
            buckets['Consistency'].append(
                max(0, 100 - abs(ci.time_delta) * 200))

    buckets['Overall'] = [report.consistency_score]

    return {k: (np.mean(v) if v else 50) for k, v in buckets.items()}


# ============================================================
# Page 1: Overview Dashboard
# ============================================================

def page_overview(report, aligned, corners):
    fig = plt.figure(figsize=(32, 22))
    fig.set_facecolor(COLORS['bg'])
    fig.suptitle(
        f"COACHING REPORT  |  You vs {report.driver} "
        f"({report.year} {report.gp_name} {report.session})",
        color='white', fontsize=18, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.38, wspace=0.32,
                           top=0.94, bottom=0.04,
                           left=0.04, right=0.97)

    _card_score(fig.add_subplot(gs[0, 0]), report)
    _card_times(fig.add_subplot(gs[0, 1]), report)
    _card_consistency(fig.add_subplot(gs[0, 2]), report)
    _card_radar(fig.add_subplot(gs[0, 3], polar=True), report)
    _panel_speed_grades(fig.add_subplot(gs[1, :]), aligned, report, corners)
    _panel_corner_grades(fig.add_subplot(gs[2, :2]), report)
    _panel_priority_fixes(fig.add_subplot(gs[2, 2:]), report)

    return fig


def _card_score(ax, report):
    style_card(ax)
    gc = grade_color(report.overall_grade)

    border = FancyBboxPatch(
        (0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.05",
        facecolor=COLORS['card_bg'], edgecolor=gc, linewidth=3)
    ax.add_patch(border)

    ax.text(0.5, 0.70, report.overall_grade,
            ha='center', va='center', fontsize=55,
            fontweight='bold', color=gc, fontfamily='monospace')
    ax.text(0.5, 0.40, 'OVERALL', ha='center', fontsize=11,
            color=COLORS['text_dim'], fontweight='bold')
    ax.text(0.5, 0.18, format_delta(report.overall_delta),
            ha='center', fontsize=13, fontweight='bold',
            color=delta_color(report.overall_delta))


def _card_times(ax, report):
    style_card(ax)
    ax.text(0.5, 0.88, 'LAP TIMES', ha='center', fontsize=10,
            color=COLORS['text_dim'], fontweight='bold')
    ax.text(0.15, 0.68, 'YOU', ha='left', fontsize=9,
            color=COLORS['game'])
    ax.text(0.85, 0.68, format_laptime(report.game_time),
            ha='right', fontsize=14, fontweight='bold',
            color=COLORS['game'], fontfamily='monospace')
    ax.text(0.15, 0.45, report.driver, ha='left', fontsize=9,
            color=COLORS['real'])
    ax.text(0.85, 0.45, format_laptime(report.real_time),
            ha='right', fontsize=14, fontweight='bold',
            color=COLORS['real'], fontfamily='monospace')
    ax.plot([0.15, 0.85], [0.35, 0.35], color=COLORS['grid'], lw=0.5)
    ax.text(0.5, 0.18, format_delta(report.overall_delta),
            ha='center', fontsize=16, fontweight='bold',
            color=delta_color(report.overall_delta),
            fontfamily='monospace')


def _card_consistency(ax, report):
    style_card(ax)
    score = report.consistency_score

    if score >= 70:
        color, label = COLORS['grade_A'], 'SOLID'
    elif score >= 50:
        color, label = COLORS['grade_C'], 'IMPROVING'
    else:
        color, label = COLORS['grade_F'], 'NEEDS WORK'

    ax.text(0.5, 0.85, 'CONSISTENCY', ha='center', fontsize=11,
            color=COLORS['text_dim'], fontweight='bold')
    ax.text(0.5, 0.55, f'{score:.0f}', ha='center', fontsize=42,
            fontweight='bold', color=color, fontfamily='monospace')
    ax.text(0.5, 0.35, '/100', ha='center', fontsize=14,
            color=COLORS['text_dim'])
    ax.text(0.5, 0.22, label, ha='center', fontsize=10,
            color=color, fontweight='bold')

    # Progress bar
    bg = FancyBboxPatch((0.15, 0.12), 0.7, 0.05,
                         boxstyle="round,pad=0.01",
                         facecolor='#333333', edgecolor='none')
    ax.add_patch(bg)
    fill_w = max(0.7 * (score / 100), 0.01)
    fill = FancyBboxPatch((0.15, 0.12), fill_w, 0.05,
                           boxstyle="round,pad=0.01",
                           facecolor=color, edgecolor='none', alpha=0.7)
    ax.add_patch(fill)


def _card_radar(ax, report):
    ax.set_facecolor(COLORS['bg'])
    skills = calculate_skills(report)
    cats = list(skills.keys())
    vals = list(skills.values())
    N = len(cats)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    vals_plot = vals + [vals[0]]
    angles += [angles[0]]

    ref = [100] * (N + 1)
    ax.plot(angles, ref, color=COLORS['real'], lw=1, ls='--', alpha=0.3)
    ax.fill(angles, ref, color=COLORS['real'], alpha=0.03)
    ax.plot(angles, vals_plot, color=COLORS['game'], lw=2)
    ax.fill(angles, vals_plot, color=COLORS['game'], alpha=0.15)

    for a, v in zip(angles[:-1], vals):
        c = COLORS['grade_A'] if v >= 70 else (
            COLORS['grade_C'] if v >= 40 else COLORS['grade_F'])
        ax.scatter(a, v, color=c, s=40, zorder=10,
                   edgecolor='white', linewidth=0.5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=7, color='white')
    ax.set_ylim(0, 110)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], fontsize=6,
                        color=COLORS['text_dim'])
    ax.spines['polar'].set_color(COLORS['grid'])
    ax.grid(color=COLORS['grid'], alpha=0.3)
    ax.set_title('SKILL PROFILE', color='white', fontsize=10,
                 fontweight='bold', pad=15)


def _panel_speed_grades(ax, aligned, report, corners):
    dist = aligned['lap_distance'].values
    game_spd = smooth(aligned['game_speed_kmh'].values, 10)
    real_spd = smooth(aligned['real_speed_kmh'].values, 10)

    style_axis(ax, ylabel='Speed (km/h)',
               title='SPEED TRACE + CORNER GRADES')

    ax.plot(dist, real_spd, color=COLORS['real'], lw=1.5, alpha=0.7,
            label=f'{report.driver} (Real)')
    ax.plot(dist, game_spd, color=COLORS['game'], lw=1.5, alpha=0.7,
            label='You (Game)')
    ax.fill_between(dist, game_spd, real_spd,
                    where=game_spd >= real_spd,
                    color=COLORS['faster'], alpha=0.06)
    ax.fill_between(dist, game_spd, real_spd,
                    where=game_spd < real_spd,
                    color=COLORS['slower'], alpha=0.06)

    if corners and report.corner_insights:
        ci_map = {ci.corner_id: ci for ci in report.corner_insights}
        for corner in corners:
            ci = ci_map.get(corner['id'])
            if ci is None:
                continue
            apex_m = corner['apex_m']
            apex_idx = min(np.searchsorted(dist, apex_m), len(game_spd) - 1)
            y_pos = min(game_spd[apex_idx], real_spd[apex_idx]) - 20
            gc = grade_color(ci.grade)

            ax.annotate(
                f"{ci.short}\n[{ci.grade}]\n{ci.time_delta:+.2f}s",
                xy=(apex_m, y_pos), fontsize=6, color='white',
                ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=gc,
                         alpha=0.6, edgecolor='white', lw=0.5),
                zorder=15)
            ax.axvspan(corner.get('entry_m', apex_m - 50),
                       corner.get('exit_m', apex_m + 50),
                       alpha=0.04, color=gc)

    ax.set_ylim(0, max(game_spd.max(), real_spd.max()) * 1.1)
    ax.legend(loc='upper right', fontsize=8,
              facecolor=COLORS['card_bg'], edgecolor=COLORS['grid'],
              labelcolor='white')


def _panel_corner_grades(ax, report):
    style_axis(ax, xlabel='Time Delta (s)', title='CORNER PERFORMANCE')

    sorted_ci = sorted(report.corner_insights, key=lambda c: c.corner_id)
    names = [ci.short for ci in sorted_ci]
    deltas = [ci.time_delta for ci in sorted_ci]
    grades = [ci.grade for ci in sorted_ci]

    y_pos = np.arange(len(names))
    colors = [grade_color(g) for g in grades]

    bars = ax.barh(y_pos, deltas, color=colors, alpha=0.7,
                   height=0.55, edgecolor='white', linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color=COLORS['grid'], lw=0.8)
    ax.invert_yaxis()

    for i, (bar, d, g) in enumerate(zip(bars, deltas, grades)):
        w = bar.get_width()
        x = w + 0.015 if w >= 0 else w - 0.015
        ha = 'left' if w >= 0 else 'right'
        ax.text(x, i, f'[{g}] {d:+.3f}s', ha=ha, va='center',
                fontsize=8, color='white', fontweight='bold')


def _panel_priority_fixes(ax, report):
    style_card(ax)
    ax.text(0.5, 0.95, 'TOP PRIORITY FIXES',
            ha='center', fontsize=13, fontweight='bold',
            color=COLORS['neutral'])

    sorted_ci = sorted(report.corner_insights, key=lambda c: -c.priority)
    y = 0.86
    count = 0

    for ci in sorted_ci:
        if count >= 3 or y < 0.10 or ci.priority < 0.3:
            break
        count += 1
        gc = grade_color(ci.grade)

        ax.text(0.06, y,
                f"#{count}  {ci.short}  [{ci.grade}]  "
                f"{ci.time_delta:+.3f}s",
                fontsize=11, fontweight='bold', color=gc)
        y -= 0.055

        if ci.issues:
            issue = ci.issues[0][:38]
            ax.text(0.08, y, f">> {issue}",
                    fontsize=9, color=COLORS['slower'])
            y -= 0.05
        if ci.tips:
            tip = ci.tips[0][:38]
            ax.text(0.08, y, f"Fix: {tip}",
                    fontsize=9, color=COLORS['faster'])
            y -= 0.05
        y -= 0.03

    potential = _estimate_potential(report)
    ax.text(0.5, 0.04,
            f"Potential: -{potential:.1f}s   "
            f"Target: {format_laptime(report.game_time - potential)}",
            ha='center', fontsize=11, fontweight='bold',
            color=COLORS['neutral'], fontfamily='monospace')


# ============================================================
# Page 2: Corner Deep Dive
# ============================================================

def page_corner_dive(report, aligned, corners):
    sorted_ci = sorted(report.corner_insights, key=lambda c: -c.priority)

    # Pick worst 4 unique corners
    worst = []
    seen = set()
    for ci in sorted_ci:
        if ci.short not in seen:
            worst.append(ci)
            seen.add(ci.short)
        if len(worst) >= 4:
            break

    if not worst:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(28, 16))
    fig.set_facecolor(COLORS['bg'])
    fig.suptitle(
        f"CORNER DEEP DIVE  |  You vs {report.driver}  |  "
        f"Top {len(worst)} Priority Corners",
        color='white', fontsize=16, fontweight='bold', y=0.98
    )

    dist = aligned['lap_distance'].values
    game_spd = smooth(aligned['game_speed_kmh'].values, 8)
    real_spd = smooth(aligned['real_speed_kmh'].values, 8)
    corner_map = {c['id']: c for c in corners} if corners else {}

    for idx in range(4):
        ax = axes[idx // 2, idx % 2]
        if idx >= len(worst):
            ax.axis('off')
            continue

        ci = worst[idx]
        corner = corner_map.get(ci.corner_id, {})
        entry_m = corner.get('entry_m', 0)
        exit_m = corner.get('exit_m', 0)
        apex_m = corner.get('apex_m', (entry_m + exit_m) / 2)

        zoom_start = max(0, entry_m - 200)
        zoom_end = min(dist.max(), exit_m + 200)
        mask = (dist >= zoom_start) & (dist <= zoom_end)
        d = dist[mask]

        if len(d) == 0:
            ax.axis('off')
            continue

        gc = grade_color(ci.grade)
        style_axis(ax, xlabel='Distance (m)', ylabel='Speed (km/h)')
        ax.set_title(
            f"#{idx+1}  {ci.name} ({ci.short})  [{ci.grade}]  "
            f"{ci.time_delta:+.3f}s",
            color=gc, fontsize=13, fontweight='bold', loc='left')

        ax.axvspan(entry_m, exit_m, alpha=0.08, color=gc)
        ax.axvline(apex_m, color='yellow', lw=1, ls=':', alpha=0.5)
        ax.plot(d, game_spd[mask], color=COLORS['game'], lw=2.5,
                label='You')
        ax.plot(d, real_spd[mask], color=COLORS['real'], lw=2.5,
                label=report.driver)
        ax.fill_between(d, game_spd[mask], real_spd[mask],
                        where=game_spd[mask] >= real_spd[mask],
                        color=COLORS['faster'], alpha=0.1)
        ax.fill_between(d, game_spd[mask], real_spd[mask],
                        where=game_spd[mask] < real_spd[mask],
                        color=COLORS['slower'], alpha=0.1)

        # Pedal overlay on secondary axis
        ax2 = ax.twinx()
        ax2.set_ylim(-0.1, 1.5)
        ax2.tick_params(colors=COLORS['text_dim'], labelsize=6)
        ax2.set_ylabel('Pedal', color=COLORS['text_dim'], fontsize=7)

        for prefix, color in [('game', COLORS['game']),
                               ('real', COLORS['real'])]:
            bcol = f'{prefix}_brake'
            tcol = f'{prefix}_throttle'
            if bcol in aligned.columns:
                ax2.fill_between(d, aligned[bcol].values[mask],
                                alpha=0.12, color=color, step='mid')
            if tcol in aligned.columns:
                ax2.plot(d, aligned[tcol].values[mask],
                        color=color, lw=0.7, ls='--', alpha=0.3)

        # Brake point markers
        if (ci.game_brake_point is not None and
                ci.brake_diff_m is not None and abs(ci.brake_diff_m) > 5):
            ax.axvline(ci.game_brake_point, color=COLORS['game'],
                      lw=2, ls='--', alpha=0.7)
        if (ci.real_brake_point is not None and
                ci.brake_diff_m is not None and abs(ci.brake_diff_m) > 5):
            ax.axvline(ci.real_brake_point, color=COLORS['real'],
                      lw=2, ls='--', alpha=0.7)

        # Data box
        info = []
        if ci.brake_diff_m and abs(ci.brake_diff_m) > 5:
            info.append(f"Brk: {ci.brake_diff_m:+.0f}m")
        if ci.apex_speed_diff and abs(ci.apex_speed_diff) > 3:
            info.append(f"Apx: {ci.apex_speed_diff:+.0f}")
        if ci.exit_speed_diff and abs(ci.exit_speed_diff) > 3:
            info.append(f"Ext: {ci.exit_speed_diff:+.0f}")
        if info:
            ax.text(0.02, 0.04, '  |  '.join(info),
                    transform=ax.transAxes, fontsize=9,
                    color='white', fontfamily='monospace',
                    va='bottom', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4',
                             facecolor=COLORS['card_bg'],
                             edgecolor=gc, alpha=0.9, linewidth=2))

        # Coaching tip
        if ci.tips:
            tip = ci.tips[0][:50]
            ax.text(0.98, 0.96, tip, transform=ax.transAxes,
                    fontsize=7.5, color=COLORS['neutral'],
                    ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3',
                             facecolor=COLORS['card_bg'],
                             edgecolor=COLORS['neutral'],
                             alpha=0.85, linewidth=0.5))

        ax.legend(loc='upper left', fontsize=9,
                  facecolor=COLORS['card_bg'],
                  edgecolor=COLORS['grid'], labelcolor='white')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ============================================================
# Page 3: Braking & Throttle
# ============================================================

def page_brake_throttle(report, aligned, corners):
    fig = plt.figure(figsize=(28, 18))
    fig.set_facecolor(COLORS['bg'])
    fig.suptitle(
        f"BRAKING & THROTTLE ANALYSIS  |  You vs {report.driver}",
        color='white', fontsize=16, fontweight='bold', y=0.97)

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.40, wspace=0.35,
                           top=0.91, bottom=0.06,
                           left=0.07, right=0.95)

    _panel_brake_bars(fig.add_subplot(gs[0, :3]), report)
    _panel_tendency_pie(fig.add_subplot(gs[0, 3]), report, mode='brake')
    _panel_throttle_bars(fig.add_subplot(gs[1, :3]), report)
    _panel_tendency_pie(fig.add_subplot(gs[1, 3]), report, mode='throttle')

    return fig


def _panel_brake_bars(ax, report):
    style_axis(ax, xlabel='Brake Point Difference (m)',
               title='BRAKE POINT ANALYSIS  (- = you brake earlier)')

    sorted_ci = sorted(report.corner_insights, key=lambda c: c.corner_id)
    names, diffs, colors = [], [], []
    for ci in sorted_ci:
        if ci.brake_diff_m is not None:
            names.append(ci.short)
            diffs.append(ci.brake_diff_m)
            if ci.brake_severity == 'early':
                colors.append(COLORS['slower'])
            elif ci.brake_severity == 'late':
                colors.append(COLORS['neutral'])
            else:
                colors.append(COLORS['faster'])

    if not names:
        ax.text(0.5, 0.5, 'No brake data',
                transform=ax.transAxes, ha='center',
                color='white', fontsize=12)
        return

    y = np.arange(len(names))
    ax.barh(y, diffs, color=colors, alpha=0.7, height=0.55,
            edgecolor='white', linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color='white', lw=1)
    ax.invert_yaxis()

    for i, d in enumerate(diffs):
        x = d + 3 if d >= 0 else d - 3
        ha = 'left' if d >= 0 else 'right'
        ax.text(x, i, f'{d:+.0f}m', ha=ha, va='center',
                fontsize=9, color='white', fontweight='bold')


def _panel_throttle_bars(ax, report):
    style_axis(ax, xlabel='Throttle Application Difference (m)',
               title='THROTTLE APPLICATION  (+ = you apply later)')

    sorted_ci = sorted(report.corner_insights, key=lambda c: c.corner_id)
    names, diffs, colors = [], [], []
    for ci in sorted_ci:
        if ci.throttle_diff_m is not None:
            names.append(ci.short)
            diffs.append(ci.throttle_diff_m)
            if ci.throttle_severity == 'late':
                colors.append(COLORS['slower'])
            elif ci.throttle_severity == 'early':
                colors.append(COLORS['neutral'])
            else:
                colors.append(COLORS['faster'])

    if not names:
        ax.text(0.5, 0.5, 'No throttle data',
                transform=ax.transAxes, ha='center',
                color='white', fontsize=12)
        return

    y = np.arange(len(names))
    ax.barh(y, diffs, color=colors, alpha=0.7, height=0.55,
            edgecolor='white', linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color='white', lw=1)
    ax.invert_yaxis()

    for i, d in enumerate(diffs):
        x = d + 3 if d >= 0 else d - 3
        ha = 'left' if d >= 0 else 'right'
        ax.text(x, i, f'{d:+.0f}m', ha=ha, va='center',
                fontsize=9, color='white', fontweight='bold')


def _panel_tendency_pie(ax, report, mode='brake'):
    ax.set_facecolor(COLORS['card_bg'])

    if mode == 'brake':
        early = sum(1 for ci in report.corner_insights
                    if ci.brake_severity == 'early')
        late = sum(1 for ci in report.corner_insights
                   if ci.brake_severity == 'late')
        ok = sum(1 for ci in report.corner_insights
                 if ci.brake_severity == 'ok')
        title = 'BRAKE TENDENCY'
        labels_map = [('Early', early, COLORS['slower']),
                      ('OK', ok, COLORS['faster']),
                      ('Late', late, COLORS['neutral'])]
    else:
        late = sum(1 for ci in report.corner_insights
                   if ci.throttle_severity == 'late')
        early = sum(1 for ci in report.corner_insights
                    if ci.throttle_severity == 'early')
        ok = sum(1 for ci in report.corner_insights
                 if ci.throttle_severity == 'ok')
        title = 'THROTTLE TENDENCY'
        labels_map = [('Late', late, COLORS['slower']),
                      ('OK', ok, COLORS['faster']),
                      ('Early', early, COLORS['neutral'])]

    sizes, labels, pie_colors = [], [], []
    for lbl, count, color in labels_map:
        if count > 0:
            sizes.append(count)
            labels.append(f'{lbl} ({count})')
            pie_colors.append(color)

    if not sizes:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', color='white', fontsize=12)
        return

    wedges, _, autotexts = ax.pie(
        sizes, colors=pie_colors, autopct='%1.0f%%',
        startangle=90, pctdistance=0.55,
        textprops={'color': 'white', 'fontsize': 9, 'fontweight': 'bold'}
    )
    ax.legend(wedges, labels, loc='lower center', fontsize=8,
              facecolor=COLORS['card_bg'], edgecolor=COLORS['grid'],
              labelcolor='white', framealpha=0.9,
              bbox_to_anchor=(0.5, -0.05))
    ax.set_title(title, color='white', fontsize=11,
                 fontweight='bold', pad=10)


# ============================================================
# Page 4: Action Plan
# ============================================================

def page_action_plan(report):
    fig = plt.figure(figsize=(24, 16))
    fig.set_facecolor(COLORS['bg'])
    fig.suptitle('YOUR ACTION PLAN', color='white', fontsize=22,
                 fontweight='bold', y=0.96)

    ax = fig.add_axes([0.06, 0.04, 0.88, 0.86])
    ax.set_facecolor(COLORS['bg'])
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    actions = _generate_action_plan(report)
    potential = _estimate_potential(report)

    # Header card
    header = FancyBboxPatch(
        (0.02, 0.88), 0.96, 0.10, boxstyle="round,pad=0.02",
        facecolor=COLORS['card_bg'],
        edgecolor=COLORS['neutral'], linewidth=2)
    ax.add_patch(header)

    ax.text(0.5, 0.95,
            f"Current: {format_laptime(report.game_time)}   |   "
            f"Target: {format_laptime(report.game_time - potential)}   |   "
            f"Potential: -{potential:.1f}s",
            ha='center', fontsize=13, fontweight='bold',
            color=COLORS['neutral'], fontfamily='monospace')
    ax.text(0.5, 0.90,
            f"vs {report.driver} ({report.year}):  "
            f"{format_delta(report.overall_delta)}",
            ha='center', fontsize=11,
            color=delta_color(report.overall_delta),
            fontfamily='monospace')

    # Action cards
    card_colors = ['#ff4444', '#ff6600', '#ffaa00',
                   '#88ff00', '#00ff88', '#00ccff']
    y = 0.83

    for i, action in enumerate(actions):
        if y < 0.02:
            break
        color = card_colors[min(i, len(card_colors) - 1)]

        if ':' in action:
            label, content = action.split(':', 1)
        else:
            label, content = f"TIP {i+1}", action

        lines = wrap_text(content.strip(), max_chars=65)
        n_lines = min(len(lines), 3)
        card_h = 0.035 + n_lines * 0.035

        card = FancyBboxPatch(
            (0.02, y - card_h), 0.96, card_h,
            boxstyle="round,pad=0.012",
            facecolor=COLORS['card_bg'],
            edgecolor=color, linewidth=2, alpha=0.95)
        ax.add_patch(card)

        ax.text(0.05, y - 0.025,
                f"  {i+1}.  {label.strip()}",
                fontsize=12, fontweight='bold', color=color)

        text_y = y - 0.06
        for line in lines[:3]:
            ax.text(0.10, text_y, line, fontsize=10, color='white')
            text_y -= 0.033

        y -= card_h + 0.02

    return fig


# ============================================================
# Main
# ============================================================

def main():
    parser = make_parser('F1 Lap Insight - Step 4: Visual Coaching Report')
    args = parser.parse_args()

    print("=" * 60)
    print("  F1 Lap Insight - Step 4: Visual Coaching Report")
    print("=" * 60)

    aligned, corners, report, game_meta, real_meta = run_pipeline(args)

    driver = args.driver
    year = args.year
    session = args.session
    dpi = 150

    print(f"\n  Generating visual report...")

    print("  Page 1: Overview Dashboard...")
    save_figure(
        page_overview(report, aligned, corners),
        OUTPUT_DIR / f"coaching_1_overview_{driver}_{year}_{session}.png",
        dpi=dpi)

    print("  Page 2: Corner Deep Dive...")
    fig2 = page_corner_dive(report, aligned, corners)
    if fig2:
        save_figure(
            fig2,
            OUTPUT_DIR / f"coaching_2_corners_{driver}_{year}_{session}.png",
            dpi=dpi)

    print("  Page 3: Braking & Throttle...")
    save_figure(
        page_brake_throttle(report, aligned, corners),
        OUTPUT_DIR / f"coaching_3_braking_{driver}_{year}_{session}.png",
        dpi=dpi)

    print("  Page 4: Action Plan...")
    save_figure(
        page_action_plan(report),
        OUTPUT_DIR / f"coaching_4_action_{driver}_{year}_{session}.png",
        dpi=dpi)

    # Terminal report
    print_coaching_report(report)

    print(f"\n  All 4 pages saved to {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()