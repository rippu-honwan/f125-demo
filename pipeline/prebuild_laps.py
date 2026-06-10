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
# Build matrix
# ============================================================
YEARS = [2024, 2025]
DRIVERS = ["VER", "HAM", "LEC", "NOR", "SAI"]
SESSIONS = ["Q", "R"]
TRACKS = sorted(p.stem for p in (PROJECT_ROOT / "tracks").glob("*.json"))


def main():
    succeeded = 0
    failed = 0

    for year in YEARS:
        for driver in DRIVERS:
            for session in SESSIONS:
                for track in TRACKS:
                    print(f"Building {year} {driver} {session} {track} ...")
                    try:
                        load_real_telemetry(
                            driver=driver,
                            year=year,
                            session=session,
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
