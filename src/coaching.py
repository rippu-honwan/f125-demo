"""
Automated coaching engine.
Analyzes aligned telemetry to generate actionable driving advice.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from src.utils import smooth, format_laptime, format_delta


# ============================================================
# Data extraction helpers
# ============================================================

def find_brake_point(speed, brake, start_idx, end_idx, threshold=0.1):
    """Find where braking begins in a section."""
    for i in range(start_idx, min(end_idx, len(brake))):
        if brake[i] > threshold:
            return i
    return None


def find_throttle_on(throttle, start_idx, end_idx, threshold=0.3):
    """Find where throttle application begins after apex."""
    for i in range(start_idx, min(end_idx, len(throttle))):
        if throttle[i] > threshold:
            return i
    return None


def find_full_throttle(throttle, start_idx, end_idx, threshold=0.95):
    """Find where full throttle is reached."""
    for i in range(start_idx, min(end_idx, len(throttle))):
        if throttle[i] > threshold:
            return i
    return None


def find_min_speed_idx(speed, start_idx, end_idx):
    """Find index of minimum speed in section."""
    section = speed[start_idx:end_idx]
    if len(section) == 0:
        return start_idx
    return start_idx + np.argmin(section)


def find_max_speed_idx(speed, start_idx, end_idx):
    """Find index of maximum speed in section."""
    section = speed[start_idx:end_idx]
    if len(section) == 0:
        return start_idx
    return start_idx + np.argmax(section)


# ============================================================
# Corner analysis
# ============================================================

class CornerInsight:
    """Analysis results for a single corner."""

    def __init__(self, corner_id, name, short):
        self.corner_id = corner_id
        self.name = name
        self.short = short

        # Time
        self.time_delta = 0.0

        # Braking
        self.game_brake_point = None   # distance (m)
        self.real_brake_point = None
        self.brake_diff_m = None       # positive = game brakes earlier
        self.brake_severity = None     # 'early', 'late', 'ok'

        # Entry speed
        self.game_entry_speed = None
        self.real_entry_speed = None
        self.entry_speed_diff = None

        # Apex speed
        self.game_apex_speed = None
        self.real_apex_speed = None
        self.apex_speed_diff = None

        # Exit speed
        self.game_exit_speed = None
        self.real_exit_speed = None
        self.exit_speed_diff = None

        # Throttle
        self.game_throttle_on = None   # distance (m)
        self.real_throttle_on = None
        self.throttle_diff_m = None    # positive = game applies later
        self.game_full_throttle = None
        self.real_full_throttle = None
        self.full_throttle_diff_m = None
        self.throttle_severity = None

        # Gear
        self.game_min_gear = None
        self.real_min_gear = None
        self.gear_diff = None

        # Overall
        self.issues = []       # list of issue strings
        self.tips = []         # list of tip strings
        self.priority = 0      # higher = more important to fix
        self.grade = 'B'       # A/B/C/D/F


def analyze_corner(aligned, corner, track_length):
    """
    Deep analysis of a single corner.

    Returns:
        CornerInsight object
    """
    ci = CornerInsight(
        corner['id'],
        corner.get('name', f"Corner {corner['id']}"),
        corner.get('short', f"C{corner['id']}")
    )

    dist = aligned['lap_distance'].values
    n = len(dist)

    entry_m = corner.get('entry_m', corner['apex_m'] - 80)
    apex_m = corner['apex_m']
    exit_m = corner.get('exit_m', corner['apex_m'] + 80)

    # Extend search range for braking
    pre_entry = max(0, entry_m - 200)
    post_exit = min(track_length, exit_m + 200)

    # Indices
    pre_idx = max(0, np.searchsorted(dist, pre_entry))
    entry_idx = np.searchsorted(dist, entry_m)
    apex_idx = np.searchsorted(dist, apex_m)
    exit_idx = min(n - 1, np.searchsorted(dist, exit_m))
    post_idx = min(n - 1, np.searchsorted(dist, post_exit))

    # Clamp
    entry_idx = min(entry_idx, n - 1)
    apex_idx = min(apex_idx, n - 1)

    # Speed arrays
    game_spd = smooth(aligned['game_speed_kmh'].values, 8)
    real_spd = smooth(aligned['real_speed_kmh'].values, 8)

    # Time delta
    if 'time_delta' in aligned.columns:
        td = aligned['time_delta'].values
        ci.time_delta = td[exit_idx] - td[entry_idx]

    # ---- Braking Analysis ----
    has_game_brake = 'game_brake' in aligned.columns
    has_real_brake = 'real_brake' in aligned.columns

    if has_game_brake:
        game_brake = aligned['game_brake'].values
        bp = find_brake_point(game_spd, game_brake, pre_idx, apex_idx + 30)
        if bp is not None:
            ci.game_brake_point = float(dist[bp])

    if has_real_brake:
        real_brake = aligned['real_brake'].values
        bp = find_brake_point(real_spd, real_brake, pre_idx, apex_idx + 30)
        if bp is not None:
            ci.real_brake_point = float(dist[bp])

    if ci.game_brake_point is not None and ci.real_brake_point is not None:
        ci.brake_diff_m = ci.game_brake_point - ci.real_brake_point
        if ci.brake_diff_m < -15:
            ci.brake_severity = 'early'
        elif ci.brake_diff_m > 15:
            ci.brake_severity = 'late'
        else:
            ci.brake_severity = 'ok'

    # ---- Entry Speed ----
    ci.game_entry_speed = float(game_spd[entry_idx])
    ci.real_entry_speed = float(real_spd[entry_idx])
    ci.entry_speed_diff = ci.game_entry_speed - ci.real_entry_speed

    # ---- Apex Speed ----
    game_apex_i = find_min_speed_idx(game_spd, entry_idx, exit_idx + 1)
    real_apex_i = find_min_speed_idx(real_spd, entry_idx, exit_idx + 1)

    ci.game_apex_speed = float(game_spd[game_apex_i])
    ci.real_apex_speed = float(real_spd[real_apex_i])
    ci.apex_speed_diff = ci.game_apex_speed - ci.real_apex_speed

    # ---- Exit Speed ----
    game_exit_i = find_max_speed_idx(game_spd, apex_idx, post_idx + 1)
    real_exit_i = find_max_speed_idx(real_spd, apex_idx, post_idx + 1)

    ci.game_exit_speed = float(game_spd[min(exit_idx, n - 1)])
    ci.real_exit_speed = float(real_spd[min(exit_idx, n - 1)])
    ci.exit_speed_diff = ci.game_exit_speed - ci.real_exit_speed

    # ---- Throttle Analysis ----
    has_game_thr = 'game_throttle' in aligned.columns
    has_real_thr = 'real_throttle' in aligned.columns

    if has_game_thr:
        game_thr = aligned['game_throttle'].values
        ton = find_throttle_on(game_thr, game_apex_i, post_idx)
        if ton is not None:
            ci.game_throttle_on = float(dist[ton])
        ft = find_full_throttle(game_thr, game_apex_i, post_idx)
        if ft is not None:
            ci.game_full_throttle = float(dist[ft])

    if has_real_thr:
        real_thr = aligned['real_throttle'].values
        ton = find_throttle_on(real_thr, real_apex_i, post_idx)
        if ton is not None:
            ci.real_throttle_on = float(dist[ton])
        ft = find_full_throttle(real_thr, real_apex_i, post_idx)
        if ft is not None:
            ci.real_full_throttle = float(dist[ft])

    if ci.game_throttle_on is not None and ci.real_throttle_on is not None:
        ci.throttle_diff_m = ci.game_throttle_on - ci.real_throttle_on
        if ci.throttle_diff_m > 20:
            ci.throttle_severity = 'late'
        elif ci.throttle_diff_m < -20:
            ci.throttle_severity = 'early'
        else:
            ci.throttle_severity = 'ok'

    if ci.game_full_throttle is not None and ci.real_full_throttle is not None:
        ci.full_throttle_diff_m = ci.game_full_throttle - ci.real_full_throttle

    # ---- Gear ----
    if 'game_gear' in aligned.columns:
        game_gear = aligned['game_gear'].values
        ci.game_min_gear = int(np.min(game_gear[entry_idx:exit_idx + 1]))

    if 'real_gear' in aligned.columns:
        real_gear = aligned['real_gear'].values
        ci.real_min_gear = int(np.min(real_gear[entry_idx:exit_idx + 1]))

    if ci.game_min_gear is not None and ci.real_min_gear is not None:
        ci.gear_diff = ci.game_min_gear - ci.real_min_gear

    # ---- Generate Issues & Tips ----
    _generate_feedback(ci)

    return ci


def _generate_feedback(ci: CornerInsight):
    """Generate human-readable issues and tips."""

    priority = 0
    abs_delta = abs(ci.time_delta)

    # ---- Braking ----
    if ci.brake_severity == 'early':
        diff = abs(ci.brake_diff_m)
        ci.issues.append(f"Braking {diff:.0f}m too early")
        ci.tips.append(
            f"Try braking {min(diff, 30):.0f}m later. "
            f"Use the {ci.real_brake_point:.0f}m marker as reference."
        )
        priority += diff * 0.05 + abs_delta * 2

    elif ci.brake_severity == 'late':
        diff = abs(ci.brake_diff_m)
        ci.issues.append(f"Braking {diff:.0f}m too late")
        if ci.apex_speed_diff is not None and ci.apex_speed_diff > 10:
            ci.tips.append(
                f"You brake late but carry too much speed through apex "
                f"(+{ci.apex_speed_diff:.0f} km/h). "
                f"This causes understeer and slow exit."
            )
        else:
            ci.tips.append(
                f"Braking late is good IF apex speed is maintained. "
                f"Focus on trail braking into the turn."
            )
        priority += diff * 0.03

    # ---- Entry Speed ----
    if ci.entry_speed_diff is not None:
        diff = ci.entry_speed_diff
        if diff < -15:
            ci.issues.append(f"Entry speed {abs(diff):.0f} km/h too slow")
            ci.tips.append(
                f"Carry more speed into the corner. "
                f"Real driver enters at {ci.real_entry_speed:.0f} km/h, "
                f"you enter at {ci.game_entry_speed:.0f} km/h."
            )
            priority += abs(diff) * 0.03 + abs_delta * 1.5
        elif diff > 15:
            ci.issues.append(f"Entry speed {diff:.0f} km/h too fast")
            ci.tips.append(
                f"You enter too fast, likely causing understeer. "
                f"This hurts your apex and exit speed."
            )
            priority += diff * 0.02

    # ---- Apex Speed ----
    if ci.apex_speed_diff is not None:
        diff = ci.apex_speed_diff
        if diff < -10:
            ci.issues.append(f"Apex speed {abs(diff):.0f} km/h too slow")
            if ci.brake_severity == 'early':
                ci.tips.append(
                    f"Early braking causes low apex speed. "
                    f"Brake later and use trail braking to maintain speed."
                )
            else:
                ci.tips.append(
                    f"Try a wider entry line to carry more speed to apex. "
                    f"Real: {ci.real_apex_speed:.0f} km/h, "
                    f"You: {ci.game_apex_speed:.0f} km/h."
                )
            priority += abs(diff) * 0.04 + abs_delta * 1.5

        elif diff > 15:
            ci.issues.append(f"Apex speed {diff:.0f} km/h too fast")
            ci.tips.append(
                f"Too much speed at apex = missing the apex or understeer. "
                f"This kills your exit speed and straight-line advantage."
            )
            priority += diff * 0.02

    # ---- Exit Speed ----
    if ci.exit_speed_diff is not None:
        diff = ci.exit_speed_diff
        if diff < -10:
            ci.issues.append(f"Exit speed {abs(diff):.0f} km/h too slow")
            priority += abs(diff) * 0.04 + abs_delta * 2

            if ci.throttle_severity == 'late':
                ci.tips.append(
                    f"You apply throttle {abs(ci.throttle_diff_m):.0f}m later "
                    f"than the real driver. Get on the gas earlier."
                )
            elif ci.apex_speed_diff is not None and ci.apex_speed_diff > 10:
                ci.tips.append(
                    f"Too fast at apex → sliding → late throttle → slow exit. "
                    f"Slow down 10 km/h at apex for a FASTER exit."
                )
            else:
                ci.tips.append(
                    f"Focus on smooth throttle application from apex. "
                    f"Gradually increase, don't stab."
                )

    # ---- Throttle ----
    if ci.throttle_severity == 'late' and ci.throttle_diff_m:
        diff = abs(ci.throttle_diff_m)
        if not any('throttle' in i.lower() for i in ci.issues):
            ci.issues.append(f"Throttle application {diff:.0f}m late")
            ci.tips.append(
                f"Start applying throttle at {ci.real_throttle_on:.0f}m "
                f"(you start at {ci.game_throttle_on:.0f}m). "
                f"Even 10% throttle at apex helps rotation."
            )
        priority += diff * 0.03

    if ci.full_throttle_diff_m is not None and ci.full_throttle_diff_m > 30:
        diff = ci.full_throttle_diff_m
        ci.issues.append(f"Full throttle {diff:.0f}m late")
        ci.tips.append(
            f"Real driver reaches 100% throttle at "
            f"{ci.real_full_throttle:.0f}m, you at "
            f"{ci.game_full_throttle:.0f}m. "
            f"Build confidence to go full throttle sooner."
        )
        priority += diff * 0.02

    # ---- Gear ----
    if ci.gear_diff is not None:
        if ci.gear_diff > 1:
            ci.issues.append(
                f"Using gear {ci.game_min_gear} "
                f"(real uses {ci.real_min_gear})"
            )
            ci.tips.append(
                f"Try downshifting to gear {ci.real_min_gear}. "
                f"Lower gear = more engine braking + better traction."
            )
            priority += 0.5
        elif ci.gear_diff < -1:
            ci.issues.append(
                f"Over-downshifting to gear {ci.game_min_gear} "
                f"(real uses {ci.real_min_gear})"
            )
            ci.tips.append(
                f"Stay in gear {ci.real_min_gear}. "
                f"Too low = wheel lock risk under braking."
            )
            priority += 0.3

    # ---- No issues ----
    if not ci.issues:
        if abs_delta < 0.03:
            ci.issues.append("Excellent! Matching real driver")
            ci.tips.append("Maintain this technique. Focus on consistency.")
        elif ci.time_delta < -0.03:
            ci.issues.append(f"Faster than real driver by {abs_delta:.3f}s")
            ci.tips.append(
                "Great corner! This could be game physics advantage "
                "or a genuinely good technique."
            )
        else:
            ci.issues.append(f"Slightly slower ({ci.time_delta:+.3f}s)")
            ci.tips.append("Small gains available. Fine-tune your line.")

    # ---- Grade ----
    ci.priority = priority

    if ci.time_delta < -0.05:
        ci.grade = 'A+'
    elif abs(ci.time_delta) < 0.05:
        ci.grade = 'A'
    elif ci.time_delta < 0.1:
        ci.grade = 'B+'
    elif ci.time_delta < 0.2:
        ci.grade = 'B'
    elif ci.time_delta < 0.35:
        ci.grade = 'C'
    elif ci.time_delta < 0.5:
        ci.grade = 'D'
    else:
        ci.grade = 'F'


# ============================================================
# Full lap coaching report
# ============================================================

class CoachingReport:
    """Full coaching report for a lap."""

    def __init__(self):
        self.corner_insights: List[CornerInsight] = []
        self.overall_delta = 0.0
        self.game_time = 0.0
        self.real_time = 0.0
        self.driver = ''
        self.year = 0
        self.session = ''
        self.gp_name = ''

        # Aggregated
        self.top_issues: List[Tuple[str, CornerInsight]] = []
        self.braking_summary = ''
        self.throttle_summary = ''
        self.overall_grade = 'B'
        self.consistency_score = 0.0


def generate_coaching_report(aligned, corners, game_meta, real_meta):
    """
    Generate full coaching report.

    Args:
        aligned: aligned DataFrame with time_delta
        corners: list of corner dicts from track JSON
        game_meta: dict with game lap metadata
        real_meta: dict with real lap metadata

    Returns:
        CoachingReport
    """
    report = CoachingReport()
    report.game_time = game_meta.get('best_time', 0)
    report.real_time = real_meta.get('lap_time', 0)
    report.driver = real_meta.get('driver', '???')
    report.year = real_meta.get('year', 0)
    report.session = real_meta.get('session', '?')
    report.gp_name = real_meta.get('gp_name', '')

    track_length = float(aligned['lap_distance'].max())

    if 'time_delta' in aligned.columns:
        report.overall_delta = float(aligned['time_delta'].values[-1])

    # Analyze each corner
    if corners:
        for corner in corners:
            ci = analyze_corner(aligned, corner, track_length)
            report.corner_insights.append(ci)

    # Sort by priority (worst first)
    report.corner_insights.sort(key=lambda c: -c.priority)

    # Top issues
    for ci in report.corner_insights[:5]:
        if ci.priority > 0.5:
            for issue in ci.issues:
                report.top_issues.append((issue, ci))

    # Braking summary
    early_count = sum(1 for ci in report.corner_insights
                      if ci.brake_severity == 'early')
    late_count = sum(1 for ci in report.corner_insights
                     if ci.brake_severity == 'late')
    ok_count = sum(1 for ci in report.corner_insights
                   if ci.brake_severity == 'ok')
    total_brake = early_count + late_count + ok_count

    if total_brake > 0:
        if early_count > late_count and early_count > ok_count:
            report.braking_summary = (
                f"TENDENCY: You brake too early ({early_count}/{total_brake} corners). "
                f"This is the most common amateur mistake. "
                f"Focus on using reference points and braking later."
            )
        elif late_count > early_count:
            report.braking_summary = (
                f"TENDENCY: You brake late ({late_count}/{total_brake} corners). "
                f"This can be good, but check if it hurts your apex/exit speed."
            )
        else:
            report.braking_summary = (
                f"Braking points are generally good "
                f"({ok_count}/{total_brake} corners OK)."
            )

    # Throttle summary
    late_thr = sum(1 for ci in report.corner_insights
                   if ci.throttle_severity == 'late')
    early_thr = sum(1 for ci in report.corner_insights
                    if ci.throttle_severity == 'early')
    ok_thr = sum(1 for ci in report.corner_insights
                 if ci.throttle_severity == 'ok')
    total_thr = late_thr + early_thr + ok_thr

    if total_thr > 0:
        if late_thr > ok_thr:
            report.throttle_summary = (
                f"TENDENCY: Late throttle application "
                f"({late_thr}/{total_thr} corners). "
                f"You're leaving time on the table on corner exit. "
                f"Practice progressive throttle from the apex."
            )
        elif early_thr > ok_thr:
            report.throttle_summary = (
                f"TENDENCY: Early throttle ({early_thr}/{total_thr} corners). "
                f"Good aggression, but watch for wheelspin/oversteer."
            )
        else:
            report.throttle_summary = (
                f"Throttle application is generally good "
                f"({ok_thr}/{total_thr} corners OK)."
            )

    # Consistency score (lower std = more consistent)
    deltas = [ci.time_delta for ci in report.corner_insights]
    if deltas:
        std = np.std(deltas)
        mean_abs = np.mean([abs(d) for d in deltas])
        # 0-100 scale, 100 = perfectly consistent
        report.consistency_score = max(0, min(100,
            100 - std * 200 - mean_abs * 50
        ))

    # Overall grade
    delta = abs(report.overall_delta)
    if delta < 0.5:
        report.overall_grade = 'A+'
    elif delta < 1.0:
        report.overall_grade = 'A'
    elif delta < 2.0:
        report.overall_grade = 'B+'
    elif delta < 3.0:
        report.overall_grade = 'B'
    elif delta < 5.0:
        report.overall_grade = 'C'
    elif delta < 8.0:
        report.overall_grade = 'D'
    else:
        report.overall_grade = 'F'

    return report


# ============================================================
# Text output
# ============================================================

def print_coaching_report(report: CoachingReport):
    """Print coaching report to terminal."""

    W = 70

    print(f"\n{'='*W}")
    print(f"{'COACHING REPORT':^{W}}")
    print(f"{'='*W}")

    # Header
    print(f"\n  You:   {format_laptime(report.game_time)}")
    print(f"  {report.driver}:  {format_laptime(report.real_time)}"
          f"  ({report.year} {report.gp_name} {report.session})")
    print(f"  Delta: {format_delta(report.overall_delta)}")
    print(f"  Grade: {report.overall_grade}")
    print(f"  Consistency: {report.consistency_score:.0f}/100")

    # Top priority fixes
    print(f"\n{'-'*W}")
    print(f"  TOP PRIORITY FIXES")
    print(f"{'-'*W}")

    shown = set()
    count = 0
    for ci in report.corner_insights:
        if count >= 5:
            break
        if ci.priority < 0.5:
            continue
        if ci.short in shown:
            continue
        shown.add(ci.short)
        count += 1

        grade_color = {
            'A+': '🟢', 'A': '🟢', 'B+': '🟡', 'B': '🟡',
            'C': '🟠', 'D': '🔴', 'F': '🔴'
        }.get(ci.grade, '⚪')

        print(f"\n  #{count}  {ci.name} ({ci.short}) "
              f"[{ci.grade}]  {ci.time_delta:+.3f}s")

        for issue in ci.issues:
            print(f"      Problem: {issue}")
        for tip in ci.tips[:2]:
            print(f"      Fix:     {tip}")

        # Key numbers
        nums = []
        if ci.brake_diff_m is not None and abs(ci.brake_diff_m) > 5:
            nums.append(f"Brake: {ci.brake_diff_m:+.0f}m")
        if ci.apex_speed_diff is not None and abs(ci.apex_speed_diff) > 5:
            nums.append(f"Apex: {ci.apex_speed_diff:+.0f}km/h")
        if ci.exit_speed_diff is not None and abs(ci.exit_speed_diff) > 5:
            nums.append(f"Exit: {ci.exit_speed_diff:+.0f}km/h")
        if ci.throttle_diff_m is not None and abs(ci.throttle_diff_m) > 10:
            nums.append(f"Throttle: {ci.throttle_diff_m:+.0f}m")
        if nums:
            print(f"      Data:    {' | '.join(nums)}")

    # Tendency analysis
    print(f"\n{'-'*W}")
    print(f"  DRIVING TENDENCIES")
    print(f"{'-'*W}")

    if report.braking_summary:
        print(f"\n  Braking:  {report.braking_summary}")
    if report.throttle_summary:
        print(f"\n  Throttle: {report.throttle_summary}")

    # All corners summary
    print(f"\n{'-'*W}")
    print(f"  ALL CORNERS")
    print(f"{'-'*W}")

    print(f"\n  {'Corner':<14} {'Grade':>5} {'Delta':>8} "
          f"{'Brake':>8} {'Apex':>10} {'Exit':>10} {'Thr':>8}")
    print(f"  {'-'*14} {'-'*5} {'-'*8} "
          f"{'-'*8} {'-'*10} {'-'*10} {'-'*8}")

    # Sort by track position for display
    sorted_ci = sorted(report.corner_insights,
                        key=lambda c: c.corner_id)

    for ci in sorted_ci:
        brake_str = f"{ci.brake_diff_m:+.0f}m" if ci.brake_diff_m else "  -"
        apex_str = (f"{ci.apex_speed_diff:+.0f}km/h"
                    if ci.apex_speed_diff else "  -")
        exit_str = (f"{ci.exit_speed_diff:+.0f}km/h"
                    if ci.exit_speed_diff else "  -")
        thr_str = (f"{ci.throttle_diff_m:+.0f}m"
                   if ci.throttle_diff_m else "  -")

        marker = " <--" if ci.priority > 2.0 else ""

        print(f"  {ci.short:<14} [{ci.grade:>2}]  {ci.time_delta:>+7.3f}s "
              f"{brake_str:>8} {apex_str:>10} {exit_str:>10} "
              f"{thr_str:>8}{marker}")

    # Action plan
    print(f"\n{'-'*W}")
    print(f"  ACTION PLAN (Next Session)")
    print(f"{'-'*W}")

    actions = _generate_action_plan(report)
    for i, action in enumerate(actions, 1):
        print(f"\n  {i}. {action}")

    print(f"\n{'='*W}")

    # Potential time save
    potential = _estimate_potential(report)
    print(f"\n  Estimated potential time save: {potential:.1f}s")
    print(f"  Target lap time: {format_laptime(report.game_time - potential)}")
    print(f"\n{'='*W}\n")


def _generate_action_plan(report: CoachingReport) -> List[str]:
    """Generate prioritized action plan."""
    actions = []

    # Sort by priority
    worst = sorted(report.corner_insights, key=lambda c: -c.priority)

    # Early braking fix
    early_brakers = [ci for ci in worst if ci.brake_severity == 'early']
    if early_brakers:
        names = ', '.join(ci.short for ci in early_brakers[:3])
        avg_diff = np.mean([abs(ci.brake_diff_m) for ci in early_brakers
                           if ci.brake_diff_m is not None])
        actions.append(
            f"BRAKING: Move brake points {avg_diff:.0f}m later at {names}. "
            f"Start with 5m adjustments and build up."
        )

    # Late throttle fix
    late_throttle = [ci for ci in worst if ci.throttle_severity == 'late']
    if late_throttle:
        names = ', '.join(ci.short for ci in late_throttle[:3])
        actions.append(
            f"THROTTLE: Apply gas earlier at {names}. "
            f"Start with 10% throttle at apex, build to 100% progressively."
        )

    # Slow apex fix
    slow_apex = [ci for ci in worst
                 if ci.apex_speed_diff is not None and ci.apex_speed_diff < -15]
    if slow_apex:
        names = ', '.join(ci.short for ci in slow_apex[:3])
        actions.append(
            f"APEX SPEED: Carry more speed through {names}. "
            f"Widen your entry line to maintain momentum."
        )

    # Slow exit fix
    slow_exit = [ci for ci in worst
                 if ci.exit_speed_diff is not None and ci.exit_speed_diff < -10]
    if slow_exit:
        names = ', '.join(ci.short for ci in slow_exit[:3])
        actions.append(
            f"EXIT SPEED: Focus on corner exit at {names}. "
            f"A good exit leads to faster straights. "
            f"Sacrifice entry speed for better exit if needed."
        )

    # General tip
    if report.consistency_score < 50:
        actions.append(
            f"CONSISTENCY: Your corner performance varies a lot "
            f"(score: {report.consistency_score:.0f}/100). "
            f"Focus on repeatable technique before pushing for speed."
        )

    # Focus limit
    if not actions:
        actions.append(
            "Your driving is solid! Focus on consistency and "
            "reducing small errors across all corners."
        )

    actions.append(
        f"FOCUS: Pick your worst 2-3 corners ({', '.join(ci.short for ci in worst[:3])}) "
        f"and practice only those. Don't try to fix everything at once."
    )

    return actions


def _estimate_potential(report: CoachingReport) -> float:
    """Estimate realistic time save potential."""
    potential = 0.0

    for ci in report.corner_insights:
        if ci.time_delta > 0.05:
            # Can realistically recover 40-60% of delta
            potential += ci.time_delta * 0.5
        elif ci.time_delta > 0.02:
            potential += ci.time_delta * 0.3

    # Cap at 80% of total delta
    return min(potential, abs(report.overall_delta) * 0.8)