"""
Automated coaching engine.
Analyzes aligned telemetry to generate actionable driving advice.

Changes from original:
  - [FIX] Brake point detection works with both boolean and continuous data
  - [FIX] Uses speed-based brake detection as primary method (more reliable)
  - [NEW] Accepts brake_format metadata to adjust analysis strategy
  - [REFACTOR] Cleaner separation of detection vs feedback generation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from src.utils import smooth, format_laptime, format_delta


# ============================================================
# Brake/throttle detection (format-aware)
# ============================================================

def find_brake_point_by_speed(speed, start_idx, end_idx,
                               decel_threshold=2.0, min_sustained=5):
    """
    Find braking point using speed deceleration.

    This is MORE RELIABLE than pedal input because:
      1. Works identically for both game (continuous) and real (boolean) data
      2. Not affected by brake data format differences
      3. Detects actual deceleration, not just pedal position

    Args:
        speed: smoothed speed array
        start_idx, end_idx: search range
        decel_threshold: minimum deceleration (km/h per meter) to count
        min_sustained: minimum consecutive points of deceleration

    Returns:
        index where braking begins, or None
    """
    if end_idx <= start_idx + min_sustained:
        return None

    # Calculate deceleration (positive = slowing down)
    decel = -np.gradient(speed[start_idx:end_idx])

    # Find first sustained deceleration
    count = 0
    for i in range(len(decel)):
        if decel[i] > decel_threshold:
            count += 1
            if count >= min_sustained:
                return start_idx + i - min_sustained + 1
        else:
            count = 0

    return None


def find_brake_point_by_pedal(brake, start_idx, end_idx, threshold=0.1):
    """
    Find braking point using pedal input.
    Only reliable when brake data is continuous (0-1 float).

    For boolean brake data, this triggers too late (only when fully on).
    """
    for i in range(start_idx, min(end_idx, len(brake))):
        if brake[i] > threshold:
            return i
    return None


def find_brake_point(speed, brake, start_idx, end_idx,
                     brake_format="continuous_0_1"):
    """
    Find braking point using the best available method.

    Strategy:
      - Always use speed-based detection as primary
      - Use pedal data to cross-validate if continuous
      - For boolean brake data, speed-based is the only reliable option
    """
    # Primary: speed-based (works for all data types)
    speed_bp = find_brake_point_by_speed(
        speed, start_idx, end_idx,
        decel_threshold=1.5, min_sustained=5
    )

    # Secondary: pedal-based (only for continuous data)
    pedal_bp = None
    if brake is not None and brake_format == "continuous_0_1":
        pedal_bp = find_brake_point_by_pedal(
            brake, start_idx, end_idx, threshold=0.1
        )

    # Combine: prefer pedal if both available and close
    if speed_bp is not None and pedal_bp is not None:
        if abs(speed_bp - pedal_bp) < 30:
            # They agree — use pedal (more precise for continuous data)
            return pedal_bp
        else:
            # They disagree — trust speed (more robust)
            return speed_bp

    return speed_bp or pedal_bp


def find_throttle_on(throttle, start_idx, end_idx, threshold=0.3):
    """Find where throttle application begins after apex."""
    if throttle is None or len(throttle) < 10:
        return None
    for i in range(start_idx, min(end_idx, len(throttle))):
        if throttle[i] > threshold:
            return i
    return None


def find_full_throttle(throttle, start_idx, end_idx, threshold=0.95):
    """Find where full throttle is reached."""
    if throttle is None or len(throttle) < 10:
        return None
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
        self.game_brake_point = None
        self.real_brake_point = None
        self.brake_diff_m = None
        self.brake_severity = None  # 'early', 'late', 'ok'
        self.brake_detection_method = None  # 'speed' or 'pedal'

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
        self.game_throttle_on = None
        self.real_throttle_on = None
        self.throttle_diff_m = None
        self.game_full_throttle = None
        self.real_full_throttle = None
        self.full_throttle_diff_m = None
        self.throttle_severity = None

        # Gear
        self.game_min_gear = None
        self.real_min_gear = None
        self.gear_diff = None

        # Overall
        self.issues = []
        self.tips = []
        self.priority = 0
        self.grade = 'B'


def analyze_corner(aligned, corner, track_length,
                   brake_format="continuous_0_1"):
    """
    Deep analysis of a single corner.
    brake_format controls how brake point detection works.
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

    pre_entry = max(0, entry_m - 200)
    post_exit = min(track_length, exit_m + 200)

    # Indices (with bounds checking)
    pre_idx = max(0, np.searchsorted(dist, pre_entry))
    entry_idx = min(np.searchsorted(dist, entry_m), n - 1)
    apex_idx = min(np.searchsorted(dist, apex_m), n - 1)
    exit_idx = min(np.searchsorted(dist, exit_m), n - 1)
    post_idx = min(np.searchsorted(dist, post_exit), n - 1)

    # Speed arrays (smoothed)
    game_spd = smooth(aligned['game_speed_kmh'].values, 8)
    real_spd = smooth(aligned['real_speed_kmh'].values, 8)

    # Time delta
    if 'time_delta' in aligned.columns:
        td = aligned['time_delta'].values
        ci.time_delta = td[exit_idx] - td[entry_idx]

    # ---- Braking Analysis (format-aware) ----
    game_brake = (aligned['game_brake'].values
                  if 'game_brake' in aligned.columns else None)
    real_brake = (aligned['real_brake'].values
                  if 'real_brake' in aligned.columns else None)

    # Game brake point (game data is always continuous from SRT)
    game_bp = find_brake_point(
        game_spd, game_brake, pre_idx, apex_idx + 30,
        brake_format="continuous_0_1"  # SRT is always continuous
    )
    if game_bp is not None:
        ci.game_brake_point = float(dist[game_bp])

    # Real brake point (format depends on FastF1 version)
    real_bp = find_brake_point(
        real_spd, real_brake, pre_idx, apex_idx + 30,
        brake_format=brake_format  # Use detected format
    )
    if real_bp is not None:
        ci.real_brake_point = float(dist[real_bp])

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
    ci.game_exit_speed = float(game_spd[min(exit_idx, n - 1)])
    ci.real_exit_speed = float(real_spd[min(exit_idx, n - 1)])
    ci.exit_speed_diff = ci.game_exit_speed - ci.real_exit_speed

    # ---- Throttle Analysis ----
    game_thr = (aligned['game_throttle'].values
                if 'game_throttle' in aligned.columns else None)
    real_thr = (aligned['real_throttle'].values
                if 'real_throttle' in aligned.columns else None)

    if game_thr is not None:
        ton = find_throttle_on(game_thr, game_apex_i, post_idx)
        if ton is not None:
            ci.game_throttle_on = float(dist[ton])
        ft = find_full_throttle(game_thr, game_apex_i, post_idx)
        if ft is not None:
            ci.game_full_throttle = float(dist[ft])

    if real_thr is not None:
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

    # Braking
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
                f"Late braking + high apex speed = understeer. "
                f"Brake earlier or trail brake more."
            )
        else:
            ci.tips.append(
                f"Late braking is good IF apex speed is right. "
                f"Focus on trail braking."
            )
        priority += diff * 0.03

    # Entry speed
    if ci.entry_speed_diff is not None:
        diff = ci.entry_speed_diff
        if diff < -15:
            ci.issues.append(f"Entry speed {abs(diff):.0f} km/h too slow")
            ci.tips.append(
                f"Carry more speed in. Real: {ci.real_entry_speed:.0f}, "
                f"You: {ci.game_entry_speed:.0f} km/h."
            )
            priority += abs(diff) * 0.03 + abs_delta * 1.5
        elif diff > 15:
            ci.issues.append(f"Entry speed {diff:.0f} km/h too fast")
            ci.tips.append(
                f"Too fast in = understeer. "
                f"This hurts apex and exit speed."
            )
            priority += diff * 0.02

    # Apex speed
    if ci.apex_speed_diff is not None:
        diff = ci.apex_speed_diff
        if diff < -10:
            ci.issues.append(f"Apex speed {abs(diff):.0f} km/h too slow")
            if ci.brake_severity == 'early':
                ci.tips.append(
                    f"Early braking → low apex speed. "
                    f"Brake later + trail brake."
                )
            else:
                ci.tips.append(
                    f"Widen entry line to carry more speed. "
                    f"Real: {ci.real_apex_speed:.0f}, "
                    f"You: {ci.game_apex_speed:.0f} km/h."
                )
            priority += abs(diff) * 0.04 + abs_delta * 1.5
        elif diff > 15:
            ci.issues.append(f"Apex speed {diff:.0f} km/h too fast")
            ci.tips.append(
                f"Too fast at apex = missing apex or understeer. "
                f"Kills exit speed."
            )
            priority += diff * 0.02

    # Exit speed
    if ci.exit_speed_diff is not None:
        diff = ci.exit_speed_diff
        if diff < -10:
            ci.issues.append(f"Exit speed {abs(diff):.0f} km/h too slow")
            priority += abs(diff) * 0.04 + abs_delta * 2

            if ci.throttle_severity == 'late':
                ci.tips.append(
                    f"Throttle {abs(ci.throttle_diff_m):.0f}m late. "
                    f"Get on gas earlier."
                )
            elif ci.apex_speed_diff is not None and ci.apex_speed_diff > 10:
                ci.tips.append(
                    f"Fast apex → slide → late throttle → slow exit. "
                    f"Slow apex by 10 km/h for FASTER exit."
                )
            else:
                ci.tips.append(
                    f"Smooth progressive throttle from apex. "
                    f"Don't stab it."
                )

    # Throttle
    if ci.throttle_severity == 'late' and ci.throttle_diff_m:
        diff = abs(ci.throttle_diff_m)
        if not any('throttle' in i.lower() for i in ci.issues):
            ci.issues.append(f"Throttle {diff:.0f}m late")
            ci.tips.append(
                f"Start throttle at {ci.real_throttle_on:.0f}m "
                f"(you: {ci.game_throttle_on:.0f}m). "
                f"Even 10% at apex helps."
            )
        priority += diff * 0.03

    if ci.full_throttle_diff_m is not None and ci.full_throttle_diff_m > 30:
        diff = ci.full_throttle_diff_m
        ci.issues.append(f"Full throttle {diff:.0f}m late")
        ci.tips.append(
            f"Real driver at 100% by {ci.real_full_throttle:.0f}m, "
            f"you at {ci.game_full_throttle:.0f}m."
        )
        priority += diff * 0.02

    # Gear
    if ci.gear_diff is not None:
        if ci.gear_diff > 1:
            ci.issues.append(
                f"Gear {ci.game_min_gear} (real: {ci.real_min_gear})"
            )
            ci.tips.append(
                f"Downshift to {ci.real_min_gear}: "
                f"more engine braking + traction."
            )
            priority += 0.5
        elif ci.gear_diff < -1:
            ci.issues.append(
                f"Over-downshifting to {ci.game_min_gear} "
                f"(real: {ci.real_min_gear})"
            )
            ci.tips.append(
                f"Stay in {ci.real_min_gear}: "
                f"too low = wheel lock risk."
            )
            priority += 0.3

    # No issues found
    if not ci.issues:
        if abs_delta < 0.03:
            ci.issues.append("Excellent — matching real driver")
            ci.tips.append("Maintain this technique.")
        elif ci.time_delta < -0.03:
            ci.issues.append(f"Faster by {abs_delta:.3f}s")
            ci.tips.append("Great corner! Keep it consistent.")
        else:
            ci.issues.append(f"Slightly slower ({ci.time_delta:+.3f}s)")
            ci.tips.append("Fine-tune your line for small gains.")

    # Set priority and grade
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
# Full coaching report
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
        self.brake_format = 'unknown'

        self.top_issues: List[Tuple[str, CornerInsight]] = []
        self.braking_summary = ''
        self.throttle_summary = ''
        self.overall_grade = 'B'
        self.consistency_score = 0.0


def generate_coaching_report(aligned, corners, game_meta, real_meta):
    """Generate full coaching report."""
    report = CoachingReport()
    report.game_time = game_meta.get('best_time', 0)
    report.real_time = real_meta.get('lap_time', 0)
    report.driver = real_meta.get('driver', '???')
    report.year = real_meta.get('year', 0)
    report.session = real_meta.get('session', '?')
    report.gp_name = real_meta.get('gp_name', '')
    report.brake_format = real_meta.get('brake_format', 'unknown')

    track_length = float(aligned['lap_distance'].max())

    if 'time_delta' in aligned.columns:
        report.overall_delta = float(aligned['time_delta'].values[-1])

    # Analyze each corner (pass brake_format through)
    if corners:
        for corner in corners:
            ci = analyze_corner(
                aligned, corner, track_length,
                brake_format=report.brake_format
            )
            report.corner_insights.append(ci)

    # Sort by priority
    report.corner_insights.sort(key=lambda c: -c.priority)

    # Top issues
    for ci in report.corner_insights[:5]:
        if ci.priority > 0.5:
            for issue in ci.issues:
                report.top_issues.append((issue, ci))

    # Summaries
    _build_braking_summary(report)
    _build_throttle_summary(report)
    _calculate_consistency(report)
    _assign_overall_grade(report)

    return report


def _build_braking_summary(report):
    early = sum(1 for ci in report.corner_insights
                if ci.brake_severity == 'early')
    late = sum(1 for ci in report.corner_insights
               if ci.brake_severity == 'late')
    ok = sum(1 for ci in report.corner_insights
             if ci.brake_severity == 'ok')
    total = early + late + ok

    if total == 0:
        return

    if early > late and early > ok:
        report.braking_summary = (
            f"TENDENCY: Braking too early ({early}/{total} corners). "
            f"Most common amateur mistake. Use reference points."
        )
    elif late > early:
        report.braking_summary = (
            f"TENDENCY: Late braking ({late}/{total} corners). "
            f"Check if it hurts apex/exit speed."
        )
    else:
        report.braking_summary = (
            f"Braking generally good ({ok}/{total} OK)."
        )


def _build_throttle_summary(report):
    late = sum(1 for ci in report.corner_insights
               if ci.throttle_severity == 'late')
    early = sum(1 for ci in report.corner_insights
                if ci.throttle_severity == 'early')
    ok = sum(1 for ci in report.corner_insights
             if ci.throttle_severity == 'ok')
    total = late + early + ok

    if total == 0:
        return

    if late > ok:
        report.throttle_summary = (
            f"TENDENCY: Late throttle ({late}/{total} corners). "
            f"Practice progressive throttle from apex."
        )
    elif early > ok:
        report.throttle_summary = (
            f"TENDENCY: Early throttle ({early}/{total} corners). "
            f"Watch for wheelspin/oversteer."
        )
    else:
        report.throttle_summary = (
            f"Throttle application generally good ({ok}/{total} OK)."
        )


def _calculate_consistency(report):
    deltas = [ci.time_delta for ci in report.corner_insights]
    if deltas:
        std = np.std(deltas)
        mean_abs = np.mean([abs(d) for d in deltas])
        report.consistency_score = max(0, min(100,
            100 - std * 200 - mean_abs * 50
        ))


def _assign_overall_grade(report):
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


# ============================================================
# Text output
# ============================================================

def print_coaching_report(report: CoachingReport):
    """Print coaching report to terminal."""
    W = 70

    print(f"\n{'='*W}")
    print(f"{'COACHING REPORT':^{W}}")
    print(f"{'='*W}")

    print(f"\n  You:   {format_laptime(report.game_time)}")
    print(f"  {report.driver}:  {format_laptime(report.real_time)}"
          f"  ({report.year} {report.gp_name} {report.session})")
    print(f"  Delta: {format_delta(report.overall_delta)}")
    print(f"  Grade: {report.overall_grade}")
    print(f"  Consistency: {report.consistency_score:.0f}/100")

    if report.brake_format == 'boolean':
        print(f"  ⚠ Real brake data is boolean — "
              f"brake points detected via speed deceleration")

    # Priority fixes
    print(f"\n{'-'*W}")
    print(f"  TOP PRIORITY FIXES")
    print(f"{'-'*W}")

    shown = set()
    count = 0
    for ci in report.corner_insights:
        if count >= 5 or ci.priority < 0.5:
            break
        if ci.short in shown:
            continue
        shown.add(ci.short)
        count += 1

        print(f"\n  #{count}  {ci.name} ({ci.short}) "
              f"[{ci.grade}]  {ci.time_delta:+.3f}s")
        for issue in ci.issues:
            print(f"      Problem: {issue}")
        for tip in ci.tips[:2]:
            print(f"      Fix:     {tip}")

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

    # Tendencies
    print(f"\n{'-'*W}")
    print(f"  DRIVING TENDENCIES")
    print(f"{'-'*W}")
    if report.braking_summary:
        print(f"\n  Braking:  {report.braking_summary}")
    if report.throttle_summary:
        print(f"\n  Throttle: {report.throttle_summary}")

    # All corners
    print(f"\n{'-'*W}")
    print(f"  ALL CORNERS")
    print(f"{'-'*W}")

    print(f"\n  {'Corner':<14} {'Grade':>5} {'Delta':>8} "
          f"{'Brake':>8} {'Apex':>10} {'Exit':>10} {'Thr':>8}")
    print(f"  {'-'*14} {'-'*5} {'-'*8} "
          f"{'-'*8} {'-'*10} {'-'*10} {'-'*8}")

    sorted_ci = sorted(report.corner_insights, key=lambda c: c.corner_id)
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
    print(f"  ACTION PLAN")
    print(f"{'-'*W}")

    actions = _generate_action_plan(report)
    for i, action in enumerate(actions, 1):
        print(f"\n  {i}. {action}")

    potential = _estimate_potential(report)
    print(f"\n{'='*W}")
    print(f"  Potential save: {potential:.1f}s")
    print(f"  Target: {format_laptime(report.game_time - potential)}")
    print(f"{'='*W}\n")


def _generate_action_plan(report: CoachingReport) -> List[str]:
    """Generate prioritized action plan."""
    actions = []
    worst = sorted(report.corner_insights, key=lambda c: -c.priority)

    early_brakers = [ci for ci in worst if ci.brake_severity == 'early']
    if early_brakers:
        names = ', '.join(ci.short for ci in early_brakers[:3])
        avg_diff = np.mean([abs(ci.brake_diff_m) for ci in early_brakers
                           if ci.brake_diff_m is not None])
        actions.append(
            f"BRAKING: Move brake points {avg_diff:.0f}m later at {names}. "
            f"Start with 5m adjustments."
        )

    late_throttle = [ci for ci in worst if ci.throttle_severity == 'late']
    if late_throttle:
        names = ', '.join(ci.short for ci in late_throttle[:3])
        actions.append(
            f"THROTTLE: Apply gas earlier at {names}. "
            f"10% throttle at apex, build progressively."
        )

    slow_apex = [ci for ci in worst
                 if ci.apex_speed_diff is not None and ci.apex_speed_diff < -15]
    if slow_apex:
        names = ', '.join(ci.short for ci in slow_apex[:3])
        actions.append(
            f"APEX SPEED: Carry more speed through {names}. "
            f"Widen entry to maintain momentum."
        )

    slow_exit = [ci for ci in worst
                 if ci.exit_speed_diff is not None and ci.exit_speed_diff < -10]
    if slow_exit:
        names = ', '.join(ci.short for ci in slow_exit[:3])
        actions.append(
            f"EXIT SPEED: Focus on corner exit at {names}. "
            f"Good exit → faster straights."
        )

    if report.consistency_score < 50:
        actions.append(
            f"CONSISTENCY: Score {report.consistency_score:.0f}/100. "
            f"Focus on repeatable technique first."
        )

    if not actions:
        actions.append("Solid driving! Focus on consistency.")

    actions.append(
        f"FOCUS: Pick worst 2-3 corners "
        f"({', '.join(ci.short for ci in worst[:3])}) "
        f"and practice only those."
    )

    return actions


def _estimate_potential(report: CoachingReport) -> float:
    """Estimate realistic time save potential."""
    potential = 0.0
    for ci in report.corner_insights:
        if ci.time_delta > 0.05:
            potential += ci.time_delta * 0.5
        elif ci.time_delta > 0.02:
            potential += ci.time_delta * 0.3
    return min(potential, abs(report.overall_delta) * 0.8)