#!/usr/bin/env python3
"""
F1 Lap Insight - Step 4: Visual Coaching Report
Generates visual coaching charts from telemetry comparison.

Usage:
    python scripts/04_coaching_report.py data/my_lap.csv \
        --driver VER --year 2024 --session Q --track suzuka
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 10
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

from src.loader import load_and_prepare
from src.fastf1_loader import load_real_telemetry
from src.alignment import align_two_pass
from src.utils import smooth, format_laptime, format_delta, calculate_time_delta
from src.coaching import (
    generate_coaching_report, print_coaching_report,
    CornerInsight, CoachingReport,
    find_brake_point, find_throttle_on, find_min_speed_idx,
    _generate_action_plan, _estimate_potential
)
from config import OUTPUT_DIR, BG_COLOR


# ============================================================
# Color scheme
# ============================================================

COLORS = {
    'game':       '#00ff88',
    'real':       '#ff4444',
    'faster':     '#00ff88',
    'slower':     '#ff4444',
    'neutral':    '#ffaa00',
    'bg':         BG_COLOR,
    'card_bg':    '#1a1a2e',
    'text':       '#ffffff',
    'text_dim':   '#888888',
    'grid':       '#333333',
    'grade_A':    '#00ff88',
    'grade_B':    '#88ff00',
    'grade_C':    '#ffaa00',
    'grade_D':    '#ff6600',
    'grade_F':    '#ff0000',
}

GRADE_COLORS = {
    'A+': COLORS['grade_A'],
    'A':  COLORS['grade_A'],
    'B+': COLORS['grade_B'],
    'B':  COLORS['grade_B'],
    'C':  COLORS['grade_C'],
    'D':  COLORS['grade_D'],
    'F':  COLORS['grade_F'],
}


# ============================================================
# Helpers
# ============================================================

def style_axis(ax, ylabel=None, xlabel=None, title=None):
    """Apply dark theme."""
    ax.set_facecolor(COLORS['bg'])
    ax.tick_params(colors='white', labelsize=7)
    ax.grid(alpha=0.08, color='white')
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
    if ylabel:
        ax.set_ylabel(ylabel, color='white', fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color='white', fontsize=9)
    if title:
        ax.set_title(title, color='white', fontsize=11, fontweight='bold',
                     loc='left', pad=8)


def _wrap_text(text, max_chars=60):
    """Word wrap text into lines."""
    words = text.split()
    lines = []
    line = ""
    for word in words:
        test = (line + " " + word).strip()
        if len(test) > max_chars:
            if line:
                lines.append(line)
            line = word
        else:
            line = test
    if line:
        lines.append(line)
    return lines


# ============================================================
# Page 1: Overview Dashboard
# ============================================================

def plot_overview_dashboard(report, aligned, corners):
    """Main dashboard."""
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

    ax_score = fig.add_subplot(gs[0, 0])
    _plot_score_card(ax_score, report)

    ax_times = fig.add_subplot(gs[0, 1])
    _plot_time_comparison(ax_times, report)

    ax_consist = fig.add_subplot(gs[0, 2])
    _plot_consistency_gauge(ax_consist, report)

    ax_radar = fig.add_subplot(gs[0, 3], polar=True)
    _plot_skill_radar(ax_radar, report)

    ax_speed = fig.add_subplot(gs[1, :])
    _plot_speed_with_grades(ax_speed, aligned, report, corners)

    ax_grades = fig.add_subplot(gs[2, :2])
    _plot_corner_grades(ax_grades, report)

    ax_fixes = fig.add_subplot(gs[2, 2:])
    _plot_priority_fixes(ax_fixes, report)

    return fig


def _plot_score_card(ax, report):
    """Big grade display."""
    ax.set_facecolor(COLORS['card_bg'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    rect = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                          boxstyle="round,pad=0.05",
                          facecolor=COLORS['card_bg'],
                          edgecolor=GRADE_COLORS.get(report.overall_grade,
                                                      COLORS['neutral']),
                          linewidth=3)
    ax.add_patch(rect)

    grade_color = GRADE_COLORS.get(report.overall_grade, COLORS['neutral'])

    ax.text(0.5, 0.70, report.overall_grade,
            ha='center', va='center', fontsize=55,
            fontweight='bold', color=grade_color,
            fontfamily='monospace')

    ax.text(0.5, 0.40, 'OVERALL',
            ha='center', va='center', fontsize=11,
            color=COLORS['text_dim'], fontweight='bold')

    ax.text(0.5, 0.32, 'GRADE',
            ha='center', va='center', fontsize=9,
            color=COLORS['text_dim'])

    ax.text(0.5, 0.18, format_delta(report.overall_delta),
            ha='center', va='center', fontsize=13,
            fontweight='bold',
            color=COLORS['faster'] if report.overall_delta < 0
            else COLORS['slower'])


def _plot_time_comparison(ax, report):
    """Lap time bars."""
    ax.set_facecolor(COLORS['card_bg'])
    ax.axis('off')

    rect = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                          boxstyle="round,pad=0.05",
                          facecolor=COLORS['card_bg'],
                          edgecolor=COLORS['grid'], linewidth=1)
    ax.add_patch(rect)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

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
    delta_color = COLORS['faster'] if report.overall_delta < 0 \
        else COLORS['slower']
    ax.text(0.5, 0.18, format_delta(report.overall_delta),
            ha='center', fontsize=16, fontweight='bold',
            color=delta_color, fontfamily='monospace')


def _plot_consistency_gauge(ax, report):
    """Consistency score gauge."""
    ax.set_facecolor(COLORS['card_bg'])
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    rect = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                          boxstyle="round,pad=0.05",
                          facecolor=COLORS['card_bg'],
                          edgecolor=COLORS['grid'], linewidth=1)
    ax.add_patch(rect)

    score = report.consistency_score

    ax.text(0.5, 0.85, 'CONSISTENCY',
            ha='center', va='center', fontsize=11,
            color=COLORS['text_dim'], fontweight='bold')

    if score >= 70:
        color = COLORS['grade_A']
        label = 'SOLID'
    elif score >= 50:
        color = COLORS['grade_C']
        label = 'IMPROVING'
    else:
        color = COLORS['grade_F']
        label = 'NEEDS WORK'

    ax.text(0.5, 0.55, f'{score:.0f}',
            ha='center', va='center', fontsize=42,
            fontweight='bold', color=color, fontfamily='monospace')

    ax.text(0.5, 0.35, '/100',
            ha='center', va='center', fontsize=14,
            color=COLORS['text_dim'])

    ax.text(0.5, 0.22, label,
            ha='center', va='center', fontsize=10,
            color=color, fontweight='bold')

    bar_y = 0.12
    bar_h = 0.05
    bg_rect = FancyBboxPatch((0.15, bar_y), 0.7, bar_h,
                              boxstyle="round,pad=0.01",
                              facecolor='#333333', edgecolor='none')
    ax.add_patch(bg_rect)

    fill_w = 0.7 * (score / 100)
    fill_rect = FancyBboxPatch((0.15, bar_y), max(fill_w, 0.01), bar_h,
                                boxstyle="round,pad=0.01",
                                facecolor=color, edgecolor='none',
                                alpha=0.7)
    ax.add_patch(fill_rect)


def _plot_skill_radar(ax, report):
    """Radar chart of driving skills."""
    ax.set_facecolor(COLORS['bg'])

    skills = _calculate_skills(report)
    categories = list(skills.keys())
    values = list(skills.values())

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles += [angles[0]]

    ref = [100] * N + [100]

    ax.plot(angles, ref, color=COLORS['real'], lw=1, ls='--', alpha=0.3)
    ax.fill(angles, ref, color=COLORS['real'], alpha=0.03)

    ax.plot(angles, values_plot, color=COLORS['game'], lw=2)
    ax.fill(angles, values_plot, color=COLORS['game'], alpha=0.15)

    for a, v in zip(angles[:-1], values):
        color = COLORS['grade_A'] if v >= 70 else (
            COLORS['grade_C'] if v >= 40 else COLORS['grade_F'])
        ax.scatter(a, v, color=color, s=40, zorder=10, edgecolor='white',
                   linewidth=0.5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=7, color='white')
    ax.set_ylim(0, 110)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], fontsize=6,
                        color=COLORS['text_dim'])
    ax.spines['polar'].set_color(COLORS['grid'])
    ax.grid(color=COLORS['grid'], alpha=0.3)
    ax.set_title('SKILL PROFILE', color='white', fontsize=10,
                 fontweight='bold', pad=15)


def _calculate_skills(report):
    """Calculate skill scores from corner insights."""
    braking_scores = []
    throttle_scores = []
    apex_scores = []
    exit_scores = []
    consistency_scores = []

    for ci in report.corner_insights:
        if ci.brake_diff_m is not None:
            s = max(0, 100 - abs(ci.brake_diff_m) * 1.5)
            braking_scores.append(s)

        if ci.throttle_diff_m is not None:
            s = max(0, 100 - abs(ci.throttle_diff_m) * 1.2)
            throttle_scores.append(s)

        if ci.apex_speed_diff is not None:
            diff = ci.apex_speed_diff
            if diff >= 0:
                s = min(100, 80 + diff * 0.5)
            else:
                s = max(0, 80 + diff * 1.5)
            apex_scores.append(s)

        if ci.exit_speed_diff is not None:
            diff = ci.exit_speed_diff
            if diff >= 0:
                s = min(100, 80 + diff * 0.3)
            else:
                s = max(0, 80 + diff * 1.2)
            exit_scores.append(s)

        if ci.time_delta is not None:
            s = max(0, 100 - abs(ci.time_delta) * 200)
            consistency_scores.append(s)

    def avg_or(lst, default=50):
        return np.mean(lst) if lst else default

    return {
        'Braking': avg_or(braking_scores),
        'Throttle': avg_or(throttle_scores),
        'Apex\nSpeed': avg_or(apex_scores),
        'Corner\nExit': avg_or(exit_scores),
        'Consistency': avg_or(consistency_scores),
        'Overall': report.consistency_score,
    }


def _plot_speed_with_grades(ax, aligned, report, corners):
    """Speed trace with corner grade labels."""
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
            apex_idx = np.searchsorted(dist, apex_m)
            if apex_idx >= len(game_spd):
                continue

            y_pos = min(game_spd[apex_idx], real_spd[apex_idx]) - 20
            grade_color = GRADE_COLORS.get(ci.grade, COLORS['neutral'])

            ax.annotate(
                f"{ci.short}\n[{ci.grade}]\n{ci.time_delta:+.2f}s",
                xy=(apex_m, y_pos),
                fontsize=6, color='white', ha='center',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3',
                         facecolor=grade_color, alpha=0.6,
                         edgecolor='white', lw=0.5),
                zorder=15
            )

            entry = corner.get('entry_m', apex_m - 50)
            exit_m = corner.get('exit_m', apex_m + 50)
            ax.axvspan(entry, exit_m, alpha=0.04, color=grade_color)

    ax.set_ylim(0, max(game_spd.max(), real_spd.max()) * 1.1)
    ax.legend(loc='upper right', fontsize=8,
              facecolor=COLORS['card_bg'], edgecolor=COLORS['grid'],
              labelcolor='white')


def _plot_corner_grades(ax, report):
    """Horizontal bar chart of corner performance."""
    style_axis(ax, xlabel='Time Delta (s)',
               title='CORNER PERFORMANCE')

    sorted_ci = sorted(report.corner_insights,
                        key=lambda c: c.corner_id)

    names = [ci.short for ci in sorted_ci]
    deltas = [ci.time_delta for ci in sorted_ci]
    grades = [ci.grade for ci in sorted_ci]

    y_pos = np.arange(len(names))
    colors = [GRADE_COLORS.get(g, COLORS['neutral']) for g in grades]

    bars = ax.barh(y_pos, deltas, color=colors, alpha=0.7, height=0.55,
                   edgecolor='white', linewidth=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color=COLORS['grid'], lw=0.8)
    ax.invert_yaxis()

    for i, (bar, delta, grade) in enumerate(zip(bars, deltas, grades)):
        w = bar.get_width()
        x = w + 0.015 if w >= 0 else w - 0.015
        ha = 'left' if w >= 0 else 'right'
        ax.text(x, i,
                f'[{grade}] {delta:+.3f}s',
                ha=ha, va='center', fontsize=8,
                color='white', fontweight='bold')


def _plot_priority_fixes(ax, report):
    """Priority fixes panel."""
    ax.set_facecolor(COLORS['card_bg'])
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    rect = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                          boxstyle="round,pad=0.02",
                          facecolor=COLORS['card_bg'],
                          edgecolor=COLORS['grid'], linewidth=1)
    ax.add_patch(rect)

    ax.text(0.5, 0.95, 'TOP PRIORITY FIXES',
            ha='center', fontsize=13, fontweight='bold',
            color=COLORS['neutral'])

    sorted_ci = sorted(report.corner_insights,
                        key=lambda c: -c.priority)

    y = 0.86
    count = 0

    for ci in sorted_ci:
        if count >= 3 or y < 0.10:
            break
        if ci.priority < 0.3:
            continue

        count += 1
        grade_color = GRADE_COLORS.get(ci.grade, COLORS['neutral'])

        ax.text(0.06, y,
                f"#{count}  {ci.short}  [{ci.grade}]  "
                f"{ci.time_delta:+.3f}s",
                fontsize=11, fontweight='bold', color=grade_color)
        y -= 0.055

        if ci.issues:
            issue = ci.issues[0]
            if len(issue) > 38:
                issue = issue[:35] + '...'
            ax.text(0.08, y, f">> {issue}",
                    fontsize=9, color=COLORS['slower'])
            y -= 0.05

        if ci.tips:
            tip = ci.tips[0]
            if len(tip) > 38:
                tip = tip[:35] + '...'
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

def plot_corner_deep_dive(report, aligned, corners):
    """Detailed view of worst 4 corners."""
    sorted_ci = sorted(report.corner_insights,
                        key=lambda c: -c.priority)

    worst = []
    seen = set()
    for ci in sorted_ci:
        if ci.short not in seen and ci.priority > 0.2:
            worst.append(ci)
            seen.add(ci.short)
        if len(worst) >= 4:
            break

    if len(worst) < 4:
        for ci in sorted_ci:
            if ci.short not in seen:
                worst.append(ci)
                seen.add(ci.short)
            if len(worst) >= 4:
                break

    n_corners = len(worst)
    if n_corners == 0:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(28, 16))
    fig.set_facecolor(COLORS['bg'])
    fig.suptitle(
        f"CORNER DEEP DIVE  |  You vs {report.driver}  |  "
        f"Top {n_corners} Priority Corners",
        color='white', fontsize=16, fontweight='bold', y=0.98
    )

    dist = aligned['lap_distance'].values
    game_spd_s = smooth(aligned['game_speed_kmh'].values, 8)
    real_spd_s = smooth(aligned['real_speed_kmh'].values, 8)

    corner_map = {c['id']: c for c in corners} if corners else {}

    for idx in range(4):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        if idx >= n_corners:
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

        style_axis(ax)
        grade_color = GRADE_COLORS.get(ci.grade, COLORS['neutral'])

        ax.set_title(
            f"#{idx+1}  {ci.name} ({ci.short})  [{ci.grade}]  "
            f"{ci.time_delta:+.3f}s",
            color=grade_color, fontsize=13, fontweight='bold', loc='left'
        )

        ax.axvspan(entry_m, exit_m, alpha=0.08, color=grade_color)
        ax.axvline(apex_m, color='yellow', lw=1, ls=':', alpha=0.5)

        ax.plot(d, game_spd_s[mask], color=COLORS['game'], lw=2.5,
                label='You')
        ax.plot(d, real_spd_s[mask], color=COLORS['real'], lw=2.5,
                label=report.driver)

        ax.fill_between(d, game_spd_s[mask], real_spd_s[mask],
                        where=game_spd_s[mask] >= real_spd_s[mask],
                        color=COLORS['faster'], alpha=0.1)
        ax.fill_between(d, game_spd_s[mask], real_spd_s[mask],
                        where=game_spd_s[mask] < real_spd_s[mask],
                        color=COLORS['slower'], alpha=0.1)

        ax2 = ax.twinx()
        ax2.set_ylim(-0.1, 1.5)
        ax2.tick_params(colors=COLORS['text_dim'], labelsize=6)
        ax2.set_ylabel('Pedal', color=COLORS['text_dim'], fontsize=7)

        if 'game_brake' in aligned.columns:
            ax2.fill_between(d, aligned['game_brake'].values[mask],
                            alpha=0.12, color=COLORS['game'], step='mid')
        if 'real_brake' in aligned.columns:
            ax2.fill_between(d, aligned['real_brake'].values[mask],
                            alpha=0.12, color=COLORS['real'], step='mid')
        if 'game_throttle' in aligned.columns:
            ax2.plot(d, aligned['game_throttle'].values[mask],
                    color=COLORS['game'], lw=0.7, ls='--', alpha=0.3)
        if 'real_throttle' in aligned.columns:
            ax2.plot(d, aligned['real_throttle'].values[mask],
                    color=COLORS['real'], lw=0.7, ls='--', alpha=0.3)

        if ci.game_brake_point is not None and \
           ci.brake_diff_m is not None and abs(ci.brake_diff_m) > 5:
            ax.axvline(ci.game_brake_point, color=COLORS['game'],
                      lw=2, ls='--', alpha=0.7)
        if ci.real_brake_point is not None and \
           ci.brake_diff_m is not None and abs(ci.brake_diff_m) > 5:
            ax.axvline(ci.real_brake_point, color=COLORS['real'],
                      lw=2, ls='--', alpha=0.7)

        info = []
        if ci.brake_diff_m is not None and abs(ci.brake_diff_m) > 5:
            info.append(f"Brk: {ci.brake_diff_m:+.0f}m")
        if ci.apex_speed_diff is not None and abs(ci.apex_speed_diff) > 3:
            info.append(f"Apx: {ci.apex_speed_diff:+.0f}")
        if ci.exit_speed_diff is not None and abs(ci.exit_speed_diff) > 3:
            info.append(f"Ext: {ci.exit_speed_diff:+.0f}")

        if info:
            ax.text(0.02, 0.04, '  |  '.join(info),
                    transform=ax.transAxes, fontsize=9,
                    color='white', fontfamily='monospace',
                    va='bottom', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4',
                             facecolor=COLORS['card_bg'],
                             edgecolor=grade_color, alpha=0.9,
                             linewidth=2))

        if ci.tips:
            tip = ci.tips[0]
            if len(tip) > 50:
                tip = tip[:47] + '...'
            ax.text(0.98, 0.96, tip,
                    transform=ax.transAxes, fontsize=7.5,
                    color=COLORS['neutral'], ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3',
                             facecolor=COLORS['card_bg'],
                             edgecolor=COLORS['neutral'],
                             alpha=0.85, linewidth=0.5))

        ax.legend(loc='upper left', fontsize=9,
                  facecolor=COLORS['card_bg'],
                  edgecolor=COLORS['grid'],
                  labelcolor='white')

        ax.set_xlabel('Distance (m)', fontsize=9,
                      color=COLORS['text_dim'])
        ax.set_ylabel('Speed (km/h)', fontsize=9,
                      color=COLORS['text_dim'])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ============================================================
# Page 3: Braking & Throttle Analysis
# ============================================================

def plot_braking_analysis(report, aligned, corners):
    """Braking and throttle detailed analysis."""
    fig = plt.figure(figsize=(28, 18))
    fig.set_facecolor(COLORS['bg'])
    fig.suptitle(
        f"BRAKING & THROTTLE ANALYSIS  |  You vs {report.driver}",
        color='white', fontsize=16, fontweight='bold', y=0.97
    )

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.40, wspace=0.35,
                           top=0.91, bottom=0.06,
                           left=0.07, right=0.95)

    ax1 = fig.add_subplot(gs[0, :3])
    _plot_brake_points(ax1, report)

    ax2 = fig.add_subplot(gs[0, 3])
    _plot_brake_tendency(ax2, report)

    ax3 = fig.add_subplot(gs[1, :3])
    _plot_throttle_points(ax3, report)

    ax4 = fig.add_subplot(gs[1, 3])
    _plot_throttle_tendency(ax4, report)

    return fig


def _plot_brake_points(ax, report):
    """Brake point difference per corner."""
    style_axis(ax, xlabel='Brake Point Difference (m)',
               title='BRAKE POINT ANALYSIS  (- = you brake earlier)')

    sorted_ci = sorted(report.corner_insights,
                        key=lambda c: c.corner_id)

    names = []
    diffs = []
    colors = []

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
        ax.text(0.5, 0.5, 'No brake data available',
                transform=ax.transAxes, ha='center',
                color='white', fontsize=12)
        return

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, diffs, color=colors, alpha=0.7, height=0.55,
                   edgecolor='white', linewidth=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color='white', lw=1)
    ax.invert_yaxis()

    x_min = min(diffs) if diffs else -10
    x_max = max(diffs) if diffs else 10
    margin = max(abs(x_min), abs(x_max)) * 0.3
    ax.set_xlim(x_min - margin, x_max + margin)

    ax.axvline(-15, color=COLORS['slower'], lw=0.8, ls='--', alpha=0.5)
    ax.axvline(15, color=COLORS['neutral'], lw=0.8, ls='--', alpha=0.5)

    for i, (bar, diff) in enumerate(zip(bars, diffs)):
        w = bar.get_width()
        x = w + 3 if w >= 0 else w - 3
        ha = 'left' if w >= 0 else 'right'
        ax.text(x, i, f'{diff:+.0f}m', ha=ha, va='center',
                fontsize=9, color='white', fontweight='bold')


def _plot_brake_tendency(ax, report):
    """Pie chart of braking tendency."""
    ax.set_facecolor(COLORS['card_bg'])

    early = sum(1 for ci in report.corner_insights
                if ci.brake_severity == 'early')
    late = sum(1 for ci in report.corner_insights
               if ci.brake_severity == 'late')
    ok = sum(1 for ci in report.corner_insights
             if ci.brake_severity == 'ok')

    total = early + late + ok
    if total == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', color='white', fontsize=12)
        return

    sizes = []
    labels = []
    pie_colors = []

    if early > 0:
        sizes.append(early)
        labels.append(f'Early ({early})')
        pie_colors.append(COLORS['slower'])
    if ok > 0:
        sizes.append(ok)
        labels.append(f'OK ({ok})')
        pie_colors.append(COLORS['faster'])
    if late > 0:
        sizes.append(late)
        labels.append(f'Late ({late})')
        pie_colors.append(COLORS['neutral'])

    if not sizes:
        return

    wedges, texts, autotexts = ax.pie(
        sizes, colors=pie_colors,
        autopct='%1.0f%%', startangle=90,
        pctdistance=0.55,
        textprops={'color': 'white', 'fontsize': 9, 'fontweight': 'bold'}
    )

    ax.legend(wedges, labels,
              loc='lower center',
              fontsize=8,
              facecolor=COLORS['card_bg'],
              edgecolor=COLORS['grid'],
              labelcolor='white',
              framealpha=0.9,
              bbox_to_anchor=(0.5, -0.05))

    ax.set_title('BRAKE TENDENCY', color='white', fontsize=11,
                 fontweight='bold', pad=10)


def _plot_throttle_points(ax, report):
    """Throttle application difference per corner."""
    style_axis(ax, xlabel='Throttle Application Difference (m)',
               title='THROTTLE APPLICATION  (+ = you apply later)')

    sorted_ci = sorted(report.corner_insights,
                        key=lambda c: c.corner_id)

    names = []
    diffs = []
    colors = []

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
        ax.text(0.5, 0.5, 'No throttle data available',
                transform=ax.transAxes, ha='center',
                color='white', fontsize=12)
        return

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, diffs, color=colors, alpha=0.7, height=0.55,
                   edgecolor='white', linewidth=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color='white', lw=1)
    ax.invert_yaxis()

    x_min = min(diffs) if diffs else -10
    x_max = max(diffs) if diffs else 10
    margin = max(abs(x_min), abs(x_max)) * 0.3
    ax.set_xlim(x_min - margin, x_max + margin)

    for i, (bar, diff) in enumerate(zip(bars, diffs)):
        w = bar.get_width()
        x = w + 3 if w >= 0 else w - 3
        ha = 'left' if w >= 0 else 'right'
        ax.text(x, i, f'{diff:+.0f}m', ha=ha, va='center',
                fontsize=9, color='white', fontweight='bold')


def _plot_throttle_tendency(ax, report):
    """Pie chart of throttle tendency."""
    ax.set_facecolor(COLORS['card_bg'])

    late = sum(1 for ci in report.corner_insights
               if ci.throttle_severity == 'late')
    early = sum(1 for ci in report.corner_insights
                if ci.throttle_severity == 'early')
    ok = sum(1 for ci in report.corner_insights
             if ci.throttle_severity == 'ok')

    total = late + early + ok
    if total == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', color='white', fontsize=12)
        return

    sizes = []
    labels = []
    pie_colors = []

    if late > 0:
        sizes.append(late)
        labels.append(f'Late ({late})')
        pie_colors.append(COLORS['slower'])
    if ok > 0:
        sizes.append(ok)
        labels.append(f'OK ({ok})')
        pie_colors.append(COLORS['faster'])
    if early > 0:
        sizes.append(early)
        labels.append(f'Early ({early})')
        pie_colors.append(COLORS['neutral'])

    if not sizes:
        return

    wedges, texts, autotexts = ax.pie(
        sizes, colors=pie_colors,
        autopct='%1.0f%%', startangle=90,
        pctdistance=0.55,
        textprops={'color': 'white', 'fontsize': 9, 'fontweight': 'bold'}
    )

    ax.legend(wedges, labels,
              loc='lower center',
              fontsize=8,
              facecolor=COLORS['card_bg'],
              edgecolor=COLORS['grid'],
              labelcolor='white',
              framealpha=0.9,
              bbox_to_anchor=(0.5, -0.05))

    ax.set_title('THROTTLE TENDENCY', color='white', fontsize=11,
                 fontweight='bold', pad=10)


# ============================================================
# Page 4: Action Plan
# ============================================================

def plot_action_plan(report):
    """Visual action plan page."""
    fig = plt.figure(figsize=(24, 16))
    fig.set_facecolor(COLORS['bg'])

    fig.suptitle('YOUR ACTION PLAN',
                 color='white', fontsize=22, fontweight='bold', y=0.96)

    ax = fig.add_axes([0.06, 0.04, 0.88, 0.86])
    ax.set_facecolor(COLORS['bg'])
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    actions = _generate_action_plan(report)
    potential = _estimate_potential(report)

    header_rect = FancyBboxPatch(
        (0.02, 0.88), 0.96, 0.10,
        boxstyle="round,pad=0.02",
        facecolor=COLORS['card_bg'],
        edgecolor=COLORS['neutral'], linewidth=2
    )
    ax.add_patch(header_rect)

    delta_color = COLORS['faster'] if report.overall_delta < 0 \
        else COLORS['slower']

    ax.text(0.5, 0.95,
            f"Current: {format_laptime(report.game_time)}   |   "
            f"Target: {format_laptime(report.game_time - potential)}   |   "
            f"Potential: -{potential:.1f}s",
            ha='center', fontsize=13, fontweight='bold',
            color=COLORS['neutral'], fontfamily='monospace')

    ax.text(0.5, 0.90,
            f"vs {report.driver} ({report.year}):  "
            f"{format_delta(report.overall_delta)}",
            ha='center', fontsize=11, color=delta_color,
            fontfamily='monospace')

    y = 0.83
    card_colors = ['#ff4444', '#ff6600', '#ffaa00', '#88ff00', '#00ff88',
                   '#00ccff']

    for i, action in enumerate(actions):
        if y < 0.02:
            break

        color = card_colors[min(i, len(card_colors) - 1)]

        if ':' in action:
            label, content = action.split(':', 1)
            label = label.strip()
            content = content.strip()
        else:
            label = f"TIP {i+1}"
            content = action

        lines = _wrap_text(content, max_chars=65)
        n_lines = min(len(lines), 3)
        card_h = 0.035 + n_lines * 0.035

        card_rect = FancyBboxPatch(
            (0.02, y - card_h), 0.96, card_h,
            boxstyle="round,pad=0.012",
            facecolor=COLORS['card_bg'],
            edgecolor=color, linewidth=2, alpha=0.95
        )
        ax.add_patch(card_rect)

        ax.text(0.05, y - 0.025,
                f"  {i+1}.  {label}",
                fontsize=12, fontweight='bold', color=color,
                fontfamily='DejaVu Sans')

        text_y = y - 0.025 - 0.035
        for j, line in enumerate(lines[:3]):
            ax.text(0.10, text_y, line,
                    fontsize=10, color='white',
                    fontfamily='DejaVu Sans')
            text_y -= 0.033

        y -= card_h + 0.02

    return fig


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='F1 Lap Insight - Visual Coaching Report'
    )
    parser.add_argument('csv', nargs='?', default='data/my_lap.csv')
    parser.add_argument('--driver', type=str, default='VER')
    parser.add_argument('--year', type=int, default=2024)
    parser.add_argument('--session', type=str, default='Q')
    parser.add_argument('--track', type=str, default=None)
    parser.add_argument('--gp', type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  F1 Lap Insight - Step 4: Visual Coaching Report")
    print("=" * 60)

    print(f"\n  [1/5] Loading game data...")
    game_data, game_meta = load_and_prepare(args.csv)
    game_length = game_meta['track_length']

    if args.track is None and args.gp is None:
        raw = pd.read_csv(args.csv, sep='\t', nrows=1)
        if 'trackId' in raw.columns:
            args.track = str(raw['trackId'].iloc[0]).strip().lower()
            print(f"  Auto-detected track: {args.track}")

    print(f"\n  [2/5] Loading real F1 data...")
    try:
        real_data, real_meta = load_real_telemetry(
            driver=args.driver,
            year=args.year,
            session=args.session,
            track=args.track,
            gp=args.gp,
        )
    except Exception as e:
        print(f"\n  ERROR: Failed to load real data: {e}")
        sys.exit(1)

    real_length = real_meta['track_length']

    print(f"\n  [3/5] Aligning telemetry...")
    aligned = align_two_pass(
        game_data, game_length,
        real_data, real_length,
        verbose=True
    )
    aligned = calculate_time_delta(aligned)

    corners = None
    track_name = args.track or args.gp
    if track_name:
        track_dir = PROJECT_ROOT / "tracks"
        json_path = track_dir / f"{track_name.lower()}.json"
        if json_path.exists():
            with open(json_path) as f:
                track_data = json.load(f)
            corners = track_data.get('corners', None)
            if corners:
                print(f"  Loaded {len(corners)} corners")

    if not corners:
        print("  Warning: No corners data.")
        corners = []

    print(f"\n  [4/5] Analyzing driving...")
    report = generate_coaching_report(
        aligned, corners, game_meta, real_meta
    )

    print(f"\n  [5/5] Generating visual report...")

    driver = args.driver
    year = args.year
    session = args.session
    save_dpi = 150

    print("  Page 1: Overview Dashboard...")
    fig1 = plot_overview_dashboard(report, aligned, corners)
    p1 = OUTPUT_DIR / f"coaching_1_overview_{driver}_{year}_{session}.png"
    fig1.savefig(p1, dpi=save_dpi, bbox_inches='tight',
                 facecolor=COLORS['bg'])
    plt.close(fig1)
    print(f"  Saved: {p1}")

    print("  Page 2: Corner Deep Dive...")
    fig2 = plot_corner_deep_dive(report, aligned, corners)
    if fig2:
        p2 = OUTPUT_DIR / f"coaching_2_corners_{driver}_{year}_{session}.png"
        fig2.savefig(p2, dpi=save_dpi, bbox_inches='tight',
                     facecolor=COLORS['bg'])
        plt.close(fig2)
        print(f"  Saved: {p2}")

    print("  Page 3: Braking & Throttle...")
    fig3 = plot_braking_analysis(report, aligned, corners)
    p3 = OUTPUT_DIR / f"coaching_3_braking_{driver}_{year}_{session}.png"
    fig3.savefig(p3, dpi=save_dpi, bbox_inches='tight',
                 facecolor=COLORS['bg'])
    plt.close(fig3)
    print(f"  Saved: {p3}")

    print("  Page 4: Action Plan...")
    fig4 = plot_action_plan(report)
    p4 = OUTPUT_DIR / f"coaching_4_action_{driver}_{year}_{session}.png"
    fig4.savefig(p4, dpi=save_dpi, bbox_inches='tight',
                 facecolor=COLORS['bg'])
    plt.close(fig4)
    print(f"  Saved: {p4}")

    print_coaching_report(report)

    print(f"\n  All 4 pages saved to {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()