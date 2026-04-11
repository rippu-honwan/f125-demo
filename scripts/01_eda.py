"""
F1 Lap Insight - Step 1: Data Validation & Session Summary
Quick check that data is valid before detailed analysis.

Usage:
    python scripts/01_eda.py

Output:
    Terminal summary only (no images)
    Step 2 produces all visualizations
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from config import DEFAULT_CSV
from src.utils import load_telemetry, resample, smooth, format_laptime


def main():
    print("=" * 60)
    print("  F1 Lap Insight - Step 1: Data Validation")
    print("=" * 60)

    # ---- Load ----
    best_lap, meta = load_telemetry(DEFAULT_CSV)

    # ---- Session Info ----
    print(f"\n  Track:        {meta['track_name']}")
    print(f"  Length:        {meta['track_length']:.0f}m")
    print(f"  Valid laps:    {len(meta['lap_times'])}")

    for lap, time in sorted(meta['lap_times'].items()):
        delta = time - meta['best_time']
        marker = " <-- BEST" if delta == 0 else f" (+{delta:.3f}s)"
        print(f"    Lap {lap}: {format_laptime(time)}{marker}")

    # ---- Resample ----
    data = resample(best_lap, meta['track_length'])

    # ---- Data Quality Checks ----
    print(f"\n  --- DATA QUALITY ---")
    speed = data['speed_kmh'].values

    checks = {
        'Data points': (len(data), len(data) > 1000),
        'Max speed':   (f"{speed.max():.0f} km/h", speed.max() < 400),
        'Min speed':   (f"{speed.min():.0f} km/h", speed.min() >= 0),
        'Avg speed':   (f"{speed.mean():.0f} km/h", 100 < speed.mean() < 300),
    }

    if 'throttle' in data.columns:
        t = data['throttle'].values
        checks['Throttle range'] = (
            f"{t.min():.2f} - {t.max():.2f}",
            0 <= t.min() and t.max() <= 1.05
        )

    if 'brake' in data.columns:
        b = data['brake'].values
        checks['Brake range'] = (
            f"{b.min():.2f} - {b.max():.2f}",
            0 <= b.min() and b.max() <= 1.05
        )

    if 'steering' in data.columns:
        s = data['steering'].values
        checks['Steering range'] = (
            f"{s.min():.2f} to {s.max():.2f}",
            -1.5 <= s.min() and s.max() <= 1.5
        )

    if 'gear' in data.columns:
        g = data['gear'].values
        checks['Gear range'] = (
            f"{int(g.min())} - {int(g.max())}",
            0 <= g.min() and g.max() <= 8
        )

    if 'world_position_X' in data.columns:
        x = data['world_position_X'].values
        y = data['world_position_Y'].values
        track_w = float(x.max() - x.min())
        track_h = float(y.max() - y.min())
        checks['Track extent'] = (
            f"{track_w:.0f}m x {track_h:.0f}m",
            track_w > 100 and track_h > 100
        )

    all_pass = True
    for name, (value, passed) in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"    {status}  {name}: {value}")

    # ---- Speed Zones ----
    print(f"\n  --- SPEED DISTRIBUTION ---")
    zones = [
        ("Full braking (< 100 km/h)", speed < 100),
        ("Low speed (100-150 km/h)",  (speed >= 100) & (speed < 150)),
        ("Mid speed (150-250 km/h)",  (speed >= 150) & (speed < 250)),
        ("High speed (250-300 km/h)", (speed >= 250) & (speed < 300)),
        ("Top speed (> 300 km/h)",    speed >= 300),
    ]

    for label, mask in zones:
        pct = mask.sum() / len(speed) * 100
        meters = mask.sum()
        bar = "#" * int(pct / 2)
        print(f"    {label:<30} {meters:>5}m ({pct:>5.1f}%) {bar}")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    if all_pass:
        print(f"  ALL CHECKS PASSED")
        print(f"  Data is ready for Step 2 (Track Map)")
        print(f"  Run: python scripts/02_track_map.py")
    else:
        print(f"  SOME CHECKS FAILED - review data before continuing")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()