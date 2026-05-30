#!/usr/bin/env python3
"""
Step 5: Finalize ground truth — apply review decisions, fix brake detection,
promote as primary, and clean up intermediate files.

Config-driven: works with any track via --track argument.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import shutil
import numpy as np
import pandas as pd

from src.track_registry import get_gp_name, get_output_prefix
from src.track import load_track, TRACKS_DIR
from config import OUTPUT_DIR, ensure_output_dir


# ============================================================
# Argument parsing
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 5: Finalize ground truth and promote as primary."
    )
    parser.add_argument("--track", required=True, type=str,
                        help="Track short name (e.g. suzuka, monza)")
    return parser.parse_args()


def task1_apply_review_decisions(gt_data):
    """Clear requires_review for corners that pass confidence threshold."""
    print("\n  === TASK 1: Apply Review Decisions ===")
    changed = []
    for c in gt_data['corners']:
        if c.get('requires_review', False):
            # Auto-approve if confidence >= medium and IQR reasonable
            conf = c.get('confidence', 'low')
            iqr = c.get('apex_iqr_m')
            if conf in ('medium', 'high') and (iqr is None or iqr <= 20):
                c['requires_review'] = False
                c['review_reason'] = None
                changed.append(c['id'])
    print(f"  Updated {len(changed)} corners: {changed}")
    return changed


def task2_fix_brake_detection(gt_data, prefix):
    """Fix brake_start_m for corners with low brake support using wider search."""
    print("\n  === TASK 2: Fix Brake Detection ===")

    parquet_path = OUTPUT_DIR / f"{prefix}_reference_laps.parquet"
    if not parquet_path.exists():
        print("  Reference laps not found — skipping brake fix")
        return None

    df = pd.read_parquet(parquet_path)
    lap_ids = df['lap_id'].unique()
    n_laps = len(lap_ids)

    # Find corners with low brake support that aren't high_speed
    fixed = []
    for c in gt_data['corners']:
        if c.get('type') == 'high_speed':
            continue
        brake_support = c.get('brake_start_support_ratio', 0)
        if brake_support >= 0.5:
            continue

        # Try wider search window
        apex_m = c.get('apex_m', 0)
        if apex_m == 0:
            continue

        w_start = apex_m - 300
        w_end = apex_m - 50

        per_lap_brake = []
        for lap_id in lap_ids:
            lap = df[df['lap_id'] == lap_id].sort_values('distance_m')
            mask = (lap['distance_m'] >= w_start) & (lap['distance_m'] <= w_end)
            seg = lap[mask]
            if len(seg) < 5:
                continue
            dist = seg['distance_m'].values
            brake = seg['brake'].values

            for i in range(1, len(brake)):
                if brake[i] > 0.5 and brake[i - 1] < 0.5:
                    per_lap_brake.append(float(dist[i]))
                    break

        if not per_lap_brake:
            continue

        brake_arr = np.array(per_lap_brake)
        median_brake = float(np.median(brake_arr))
        q1, q3 = np.percentile(brake_arr, [25, 75])
        iqr = float(q3 - q1)
        support_ratio = len(brake_arr) / n_laps

        if support_ratio > brake_support:
            c['brake_start_m'] = round(median_brake, 1)
            c['brake_start_iqr_m'] = round(iqr, 1)
            c['brake_start_support_ratio'] = round(support_ratio, 3)
            if 'boolean' not in c.get('signals', []):
                c.setdefault('signals', []).append('boolean')

            # Recompute confidence
            apex_support = c.get('apex_support_ratio', 0)
            apex_iqr = c.get('apex_iqr_m')
            signal_count = len(c.get('signals', []))

            score = 0
            if apex_support >= 0.80:
                score += 2
            elif apex_support >= 0.65:
                score += 1
            if apex_iqr is not None:
                if apex_iqr <= 8:
                    score += 2
                elif apex_iqr <= 15:
                    score += 1
            if signal_count >= 2:
                score += 2
            elif signal_count >= 1:
                score += 1
            if support_ratio >= 0.70:
                score += 1

            if c.get('type') == 'high_speed':
                conf = "high" if score >= 6 else "medium" if score >= 4 else "low"
            else:
                conf = "medium" if score >= 3 else "low"

            c['confidence_score'] = score
            c['confidence'] = conf

            if support_ratio >= 0.70:
                c['requires_review'] = False
                c['review_reason'] = None
            else:
                c['requires_review'] = True
                c['review_reason'] = (
                    "brake_start support_ratio below threshold after window correction"
                )

            fixed.append(c['id'])

    print(f"  Fixed brake for {len(fixed)} corners: {fixed}")
    return fixed


def task3_promote(gt_data, track_short):
    """Promote ground_truth.json as primary track JSON."""
    print("\n  === TASK 3: Promote ground_truth.json ===")

    gt_path = TRACKS_DIR / f"{track_short}.ground_truth.json"
    orig_path = TRACKS_DIR / f"{track_short}.json"
    backup_path = TRACKS_DIR / f"{track_short}.original.json"

    # 3a: Verify valid JSON + loader compat
    for c in gt_data['corners']:
        assert 'apex_m' in c and c['apex_m'] is not None, (
            f"Corner {c['id']} missing apex_m"
        )
    n_corners = len(gt_data['corners'])
    print(f"  All {n_corners} corners have apex_m")

    # 3b: Verify loader can read it (dry test before rename)
    from src.track import Corner, CORNER_COLORS
    corners = [Corner(
        id=c['id'], name=c['name'], short=c['short'],
        type=c.get('type', 'medium_speed'),
        direction=c.get('direction', 'right'),
        entry_m=c.get('entry_m', c.get('apex_zone_start_m', 0)),
        apex_m=c['apex_m'],
        exit_m=c.get('exit_m', c.get('apex_zone_end_m', 0)),
        color=CORNER_COLORS[i % len(CORNER_COLORS)],
    ) for i, c in enumerate(gt_data['corners'])]
    assert len(corners) == n_corners, f"Expected {n_corners}, got {len(corners)}"
    print(f"  Dry-load test passed ({len(corners)} corners)")

    # 3c: Atomic rename
    try:
        # Save updated ground truth first
        with open(gt_path, 'w', encoding='utf-8') as f:
            json.dump(gt_data, f, indent=2, ensure_ascii=False)

        # Rename original -> backup (only if not already backed up)
        if orig_path.exists() and not backup_path.exists():
            shutil.move(str(orig_path), str(backup_path))
            print(f"  Renamed: {track_short}.json -> {track_short}.original.json")
        elif orig_path.exists() and backup_path.exists():
            orig_path.unlink()
            print(f"  Removed old {track_short}.json (backup already exists)")

        # Rename ground_truth -> primary
        shutil.move(str(gt_path), str(orig_path))
        print(f"  Renamed: {track_short}.ground_truth.json -> {track_short}.json")
    except Exception as e:
        print(f"  Rename FAILED: {e}")
        if backup_path.exists() and not orig_path.exists():
            shutil.move(str(backup_path), str(orig_path))
            print("  Reverted backup -> original")
        return False

    # 3d: Dry-load the new track JSON via actual loader
    try:
        track = load_track(track_short)
        assert track.n_corners == n_corners
        print(f"  Post-rename load test: {track.name} ({track.n_corners} corners)")
        return True
    except Exception as e:
        print(f"  Post-rename load FAILED: {e}")
        shutil.move(str(orig_path), str(gt_path))
        if backup_path.exists():
            shutil.move(str(backup_path), str(orig_path))
        print("  Reverted all renames")
        return False


def task4_cleanup(prefix):
    """Delete intermediate output files."""
    print("\n  === TASK 4: Delete Intermediate Files ===")

    delete_files = [
        OUTPUT_DIR / f"{prefix}_reference_laps.parquet",
        OUTPUT_DIR / f"{prefix}_reference_laps_metadata.json",
        OUTPUT_DIR / f"{prefix}_reference_laps_overview.png",
        OUTPUT_DIR / f"{prefix}_segments_consensus.json",
        OUTPUT_DIR / f"{prefix}_segmentation_overview.png",
        OUTPUT_DIR / f"{prefix}_phase_detection_per_lap.parquet",
        OUTPUT_DIR / f"{prefix}_ground_truth_validation.png",
        OUTPUT_DIR / f"{prefix}_fix_validation.png",
        OUTPUT_DIR / f"{prefix}_ground_truth_report.md",
    ]

    deleted = 0
    skipped = 0
    for path in delete_files:
        if path.exists():
            path.unlink()
            deleted += 1
        else:
            skipped += 1
    print(f"  Deleted: {deleted}, Skipped (not found): {skipped}")
    return deleted, skipped


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    gp_name = get_gp_name(args.track)
    prefix = get_output_prefix(args.track)

    gt_path = TRACKS_DIR / f"{args.track}.ground_truth.json"

    print("=" * 60)
    print("  Step 5: Finalize Ground Truth")
    print(f"  {gp_name}")
    print("=" * 60)

    with open(gt_path) as f:
        gt_data = json.load(f)

    # Task 1
    changed_ids = task1_apply_review_decisions(gt_data)

    # Task 2
    fixed_ids = task2_fix_brake_detection(gt_data, prefix)

    # Task 3
    success = task3_promote(gt_data, args.track)
    if not success:
        print("\n  ABORTING — Task 3 failed. No cleanup performed.")
        return

    # Task 4
    deleted, skipped = task4_cleanup(prefix)

    # Final report
    print("\n" + "=" * 60)
    print("  FINAL REPORT")
    print("=" * 60)
    print(f"\n  TASK 1: {len(changed_ids)} corners auto-approved")
    if fixed_ids:
        print(f"  TASK 2: Fixed brake for {len(fixed_ids)} corners")
    else:
        print("  TASK 2: No brake fixes needed")
    print(f"  TASK 3: Rename successful")
    print(f"  TASK 4: {deleted} files deleted, {skipped} skipped (not found)")

    # Final state
    track_json_path = TRACKS_DIR / f"{args.track}.json"
    backup_path = TRACKS_DIR / f"{args.track}.original.json"

    new_track = load_track(args.track)
    track_data = json.loads(track_json_path.read_text())
    review_corners = [c['short'] for c in track_data['corners']
                      if c.get('requires_review', False)]

    print(f"\n  FINAL STATE:")
    print(f"  - tracks/{args.track}.json          -> {new_track.n_corners} corners")
    print(f"  - tracks/{args.track}.original.json -> exists: "
          f"{'YES' if backup_path.exists() else 'NO'}")

    if review_corners:
        print(f"\n  Corners still requiring review: {review_corners}")
    else:
        print(f"\n  Corners still requiring review: none")


if __name__ == "__main__":
    ensure_output_dir()
    main()
