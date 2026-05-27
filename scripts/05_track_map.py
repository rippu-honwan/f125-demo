#!/usr/bin/env python3
"""
F1 Lap Insight – Step 5: Track Map Visualization
Generates a 2D track layout with per-corner problem indicators.

Usage:
    python scripts/05_track_map.py data/my_lap.csv \
        --driver VER --year 2025 --session Q --track suzuka

    # Also draw speed-delta heatmap:
    python scripts/05_track_map.py data/my_lap.csv \
        --driver VER --year 2025 --session Q --track suzuka --mode both

Modes:
    issues   (default) – colour each corner by problem severity
    heatmap             – colour the track LINE by speed delta vs pro
    both                – heatmap line + corner labels on top
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

from src.pipeline import make_parser, run_pipeline
from src.track_map import draw_all_modes, draw_track_map
from config import OUTPUT_DIR


def main():
    parser = make_parser("F1 Track Map Visualization")
    parser.add_argument(
        '--mode',
        choices=['issues', 'heatmap', 'both', 'all'],
        default='both',
        help=(
            'issues  = corner problem labels only\n'
            'heatmap = speed delta colour on track line\n'
            'both    = combined\n'
            'all     = generate all 3 variants'
        )
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  F1 LAP INSIGHT – Track Map")
    print("=" * 60)

    # Run shared pipeline
    aligned, corners, report, game_meta, real_meta = run_pipeline(args)

    driver  = args.driver
    year    = args.year
    session = args.session
    track   = args.track or "track"

    base_title = (
        f"{track.upper()} – You vs {driver}  "
        f"({year} {session})"
    )

    print(f"\n  Generating track map (mode={args.mode})...")

    if args.mode == 'all':
        draw_all_modes(
            aligned, corners, report,
            output_dir=str(OUTPUT_DIR),
            prefix=track,
            driver=driver,
            year=year,
            session=session,
        )
    else:
        out_path = (
            OUTPUT_DIR /
            f"{track}_trackmap_{args.mode}_{driver}_{year}_{session}.png"
        )
        draw_track_map(
            aligned=aligned,
            corners=corners,
            report=report,
            output_path=str(out_path),
            title=base_title,
            mode=args.mode,
        )

    print("\n  ✅ Track map done.")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
