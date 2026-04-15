"""
Shared pipeline for loading, aligning, and analyzing telemetry.

Eliminates duplicated boilerplate across scripts 03 and 04.
Single source of truth for the game→real→align→coach workflow.
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

from src.loader import load_and_prepare
from src.fastf1_loader import load_real_telemetry
from src.alignment import align_two_pass
from src.utils import calculate_time_delta
from src.coaching import generate_coaching_report, CoachingReport


PROJECT_ROOT = Path(__file__).parent.parent


def make_parser(description: str) -> argparse.ArgumentParser:
    """Create standard argument parser for comparison scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('csv', nargs='?', default='data/my_lap.csv',
                        help='Path to your telemetry CSV')
    parser.add_argument('--lap', type=int, default=None,
                        help='lapIndex to use (0-based). '
                             'If omitted, auto-selects fastest lap.')
    parser.add_argument('--driver', type=str, default='VER',
                        help='F1 driver code (e.g. VER, HAM, LEC)')
    parser.add_argument('--year', type=int, default=2024,
                        help='Season year')
    parser.add_argument('--session', type=str, default='Q',
                        help='Session type: Q, R, FP1, FP2, FP3')
    parser.add_argument('--track', type=str, default=None,
                        help='Track short name (e.g. suzuka)')
    parser.add_argument('--gp', type=str, default=None,
                        help='Full GP name (e.g. "Japanese Grand Prix")')
    parser.add_argument('--no-corners', action='store_true',
                        help='Skip corner analysis')
    return parser


def auto_detect_track(csv_path: str) -> Optional[str]:
    """Try to detect track name from CSV metadata."""
    try:
        raw = pd.read_csv(csv_path, sep='\t', nrows=1)
        if 'trackId' in raw.columns:
            return str(raw['trackId'].iloc[0]).strip().lower()
    except Exception:
        pass
    return None


def load_corners(track_name: Optional[str]) -> Optional[List[Dict]]:
    """Load corner definitions from track JSON."""
    if track_name is None:
        return None

    track_dir = PROJECT_ROOT / "tracks"

    # Exact match
    json_path = track_dir / f"{track_name.lower()}.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        corners = data.get('corners', None)
        if corners:
            print(f"  Loaded {len(corners)} corners from {json_path.name}")
        return corners

    # Partial match
    for p in track_dir.glob("*.json"):
        if track_name.lower() in p.stem.lower():
            with open(p) as f:
                data = json.load(f)
            corners = data.get('corners', None)
            if corners:
                print(f"  Loaded {len(corners)} corners from {p.name}")
            return corners

    print(f"  ⚠ No track JSON for '{track_name}'")
    return None


def run_pipeline(args) -> Tuple[pd.DataFrame, List[Dict], CoachingReport,
                                 Dict, Dict]:
    """
    Run the full comparison pipeline.

    Returns:
        (aligned_data, corners, coaching_report, game_meta, real_meta)
    """
    # ---- Step 1: Load game data ----
    print(f"\n  [1/5] Loading game data...")
    lap_index = getattr(args, 'lap', None)
    game_data, game_meta = load_and_prepare(args.csv, lap_index=lap_index)
    game_length = game_meta['track_length']

    # ---- Step 2: Auto-detect track ----
    if args.track is None and args.gp is None:
        detected = auto_detect_track(args.csv)
        if detected:
            args.track = detected
            print(f"  Auto-detected track: {detected}")
        else:
            print("  ⚠ Cannot auto-detect track.")
            print("  Use --track suzuka or --gp 'Japanese Grand Prix'")
            sys.exit(1)

    # ---- Step 3: Load real F1 data ----
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
        print(f"\n  ❌ Failed to load real data: {e}")
        sys.exit(1)

    real_length = real_meta['track_length']

    # ---- Step 4: Align ----
    print(f"\n  [3/5] Aligning telemetry...")
    aligned = align_two_pass(
        game_data, game_length,
        real_data, real_length,
        verbose=True
    )
    aligned = calculate_time_delta(aligned)

    # ---- Step 5: Load corners & generate report ----
    print(f"\n  [4/5] Analyzing corners...")
    corners = []
    if not args.no_corners:
        track_name = args.track or args.gp
        corners = load_corners(track_name) or []

    print(f"\n  [5/5] Generating coaching report...")
    report = generate_coaching_report(
        aligned, corners, game_meta, real_meta
    )

    return aligned, corners, report, game_meta, real_meta