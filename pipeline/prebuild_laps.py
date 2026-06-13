#!/usr/bin/env python3
"""
Prebuild reference laps.

Standalone script that warms the pre-built reference lap cache by calling
the FastF1 loader for a hardcoded matrix of (year, driver, session, track)
combinations. Each successful load is persisted as a pickle under
data/reference_laps/ by the loader itself.

Run once locally (where FastF1 downloads are allowed):

    python pipeline/prebuild_laps.py

Then commit the generated data/reference_laps/*.pkl files. At runtime
(e.g. on Render) the loader serves these directly — no FastF1 download.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fastf1_loader import load_real_telemetry

# ============================================================
# Build matrix — targeted (track, year, driver) combos
# ============================================================
COMBOS = [
    ("bahrain",     2025, "NOR"),
    ("jeddah",      2025, "SAI"),
    ("melbourne",   2025, "NOR"),
    ("shanghai",    2025, "NOR"),
    ("miami",       2025, "NOR"),
    ("imola",       2025, "NOR"),
    ("monaco",      2025, "LEC"),
    ("montreal",    2025, "VER"),
    ("barcelona",   2025, "NOR"),
    ("spielberg",   2025, "NOR"),
    ("silverstone", 2025, "NOR"),
    ("hungaroring", 2025, "NOR"),
    ("spa",         2025, "VER"),
    ("zandvoort",   2025, "NOR"),
    ("monza",       2025, "NOR"),
    ("baku",        2025, "NOR"),
    ("singapore",   2025, "NOR"),
    ("suzuka",      2025, "VER"),
    ("austin",      2025, "NOR"),
    ("mexico_city", 2025, "NOR"),
    ("interlagos",  2025, "NOR"),
    ("las_vegas",   2025, "NOR"),
    ("lusail",      2025, "NOR"),
    ("yas_marina",  2025, "NOR"),
]


def main():
    succeeded = 0
    failed = 0

    for track, year, driver in COMBOS:
        print(f"Building {year} {driver} Q {track} ...")
        try:
            load_real_telemetry(
                driver=driver,
                year=year,
                session="Q",
                track=track,
            )
            succeeded += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    total = succeeded + failed
    print("\n" + "=" * 50)
    print(f"Done: {succeeded}/{total} succeeded, {failed}/{total} failed")
    print(f"Reference laps written to: {PROJECT_ROOT / 'data' / 'reference_laps'}")


if __name__ == "__main__":
    main()
