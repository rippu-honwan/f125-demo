#!/usr/bin/env python3
"""
Step 5: Finalize ground truth — apply review decisions, fix FIN brake,
promote as primary, and clean up intermediate files.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import hashlib
import shutil
import numpy as np
import pandas as pd

from src.track import load_track, TRACKS_DIR
from config import OUTPUT_DIR

GT_PATH = TRACKS_DIR / "suzuka.ground_truth.json"
ORIG_PATH = TRACKS_DIR / "suzuka.json"
BACKUP_PATH = TRACKS_DIR / "suzuka.original.json"

# IDs to clear review flag (Task 1)
CLEAR_REVIEW_IDS = [1, 3, 4, 5, 7, 13, 14, 15, 16, 17]

# Intermediate files to delete (Task 4)
DELETE_FILES = [
    OUTPUT_DIR / "suzuka_reference_laps.parquet",
    OUTPUT_DIR / "suzuka_reference_laps_metadata.json",
    OUTPUT_DIR / "suzuka_reference_laps_overview.png",
    OUTPUT_DIR / "suzuka_segments_consensus.json",
    OUTPUT_DIR / "suzuka_phase_detection_per_lap.parquet",
    OUTPUT_DIR / "suzuka_segmentation_overview.png",
    OUTPUT_DIR / "suzuka_ground_truth_validation.png",
    OUTPUT_DIR / "suzuka_fix_validation.png",
    OUTPUT_DIR / "suzuka_ground_truth_report.md",
]


def task1_apply_review_decisions(gt_data):
    """Clear requires_review for approved corners."""
    print("\n  === TASK 1: Apply Review Decisions ===")
    changed = []
    for c in gt_data['corners']:
        if c['id'] in CLEAR_REVIEW_IDS:
            if c.get('requires_review', False):
                c['requires_review'] = False
                c['review_reason'] = None
                changed.append(c['id'])
    print(f"  Updated {len(changed)} corners: {changed}")
    return changed


def task2_fix_fin_brake(gt_data):
    """Fix Final Chicane brake_start_m using wider search window."""
    print("\n  === TASK 2: Fix FIN brake_start_m ===")

    df = pd.read_parquet(OUTPUT_DIR / "suzuka_reference_laps.parquet")
    lap_ids = df['lap_id'].unique()
    n_laps = len(lap_ids)

    # Search window: 5300–5520m
    w_start, w_end = 5300, 5520

    per_lap_brake = []
    for lap_id in lap_ids:
        lap = df[df['lap_id'] == lap_id].sort_values('distance_m')
        mask = (lap['distance_m'] >= w_start) & (lap['distance_m'] <= w_end)
        seg = lap[mask]
        if len(seg) < 5:
            continue
        dist = seg['distance_m'].values
        brake = seg['brake'].values

        # Method 1: boolean transition
        found = False
        for i in range(1, len(brake)):
            if brake[i] > 0.5 and brake[i - 1] < 0.5:
                per_lap_brake.append(float(dist[i]))
                found = True
                break

        if not found:
            # Method 2: decel gradient > 8 m/s²
            speed = seg['speed'].values / 3.6
            decel = -np.gradient(speed, dist)
            for i in range(len(decel)):
                if decel[i] > 8.0:
                    per_lap_brake.append(float(dist[i]))
                    found = True
                    break

    brake_arr = np.array(per_lap_brake)
    if len(brake_arr) == 0:
        print("  ✗ No brake detections found — skipping")
        return None

    median_brake = float(np.median(brake_arr))
    q1, q3 = np.percentile(brake_arr, [25, 75])
    iqr = float(q3 - q1)
    support_ratio = len(brake_arr) / n_laps

    print(f"  brake_start_m = {median_brake:.1f}")
    print(f"  IQR = {iqr:.1f}m, support = {support_ratio:.3f} ({len(brake_arr)}/{n_laps})")

    # Update id=18
    for c in gt_data['corners']:
        if c['id'] == 18:
            c['brake_start_m'] = round(median_brake, 1)
            c['brake_start_iqr_m'] = round(iqr, 1)
            c['brake_start_support_ratio'] = round(support_ratio, 3)

            # Add boolean to signals if not present
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

            # Low-curvature corner: adjusted thresholds
            conf = "medium" if score >= 3 else "low"
            c['confidence_score'] = score
            c['confidence'] = conf

            # Review decision
            if support_ratio >= 0.70:
                c['requires_review'] = False
                c['review_reason'] = None
            else:
                c['requires_review'] = True
                c['review_reason'] = (
                    "brake_start support_ratio below threshold after window correction"
                )

            print(f"  confidence_score={score}, confidence={conf}")
            print(f"  requires_review={c['requires_review']}")
            return {
                'brake_start_m': c['brake_start_m'],
                'support_ratio': support_ratio,
                'confidence': conf,
            }
    return None


def task3_promote(gt_data):
    """Promote ground_truth.json as primary suzuka.json."""
    print("\n  === TASK 3: Promote ground_truth.json ===")

    # 3a: Verify valid JSON + loader compat
    for c in gt_data['corners']:
        assert 'apex_m' in c and c['apex_m'] is not None, (
            f"Corner {c['id']} missing apex_m"
        )
    print(f"  ✓ All {len(gt_data['corners'])} corners have apex_m")

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
    assert len(corners) == 18, f"Expected 18, got {len(corners)}"
    print(f"  ✓ Dry-load test passed ({len(corners)} corners)")

    # 3c: Atomic rename
    try:
        # Save updated ground truth first
        with open(GT_PATH, 'w', encoding='utf-8') as f:
            json.dump(gt_data, f, indent=2, ensure_ascii=False)

        # Rename original → backup
        shutil.move(str(ORIG_PATH), str(BACKUP_PATH))
        # Rename ground_truth → primary
        shutil.move(str(GT_PATH), str(ORIG_PATH))
        print(f"  ✓ Renamed: suzuka.json → suzuka.original.json")
        print(f"  ✓ Renamed: suzuka.ground_truth.json → suzuka.json")
    except Exception as e:
        # Revert
        print(f"  ✗ Rename FAILED: {e}")
        if BACKUP_PATH.exists() and not ORIG_PATH.exists():
            shutil.move(str(BACKUP_PATH), str(ORIG_PATH))
            print("  ✓ Reverted backup → original")
        return False

    # 3d: Dry-load the new suzuka.json via actual loader
    try:
        track = load_track("suzuka")
        assert track.n_corners == 18
        print(f"  ✓ Post-rename load test: {track.name} ({track.n_corners} corners)")
        return True
    except Exception as e:
        print(f"  ✗ Post-rename load FAILED: {e}")
        # Revert
        shutil.move(str(ORIG_PATH), str(GT_PATH))
        shutil.move(str(BACKUP_PATH), str(ORIG_PATH))
        print("  ✓ Reverted all renames")
        return False


def task4_cleanup():
    """Delete intermediate output files."""
    print("\n  === TASK 4: Delete Intermediate Files ===")
    deleted = 0
    skipped = 0
    for path in DELETE_FILES:
        if path.exists():
            path.unlink()
            deleted += 1
        else:
            skipped += 1
    print(f"  Deleted: {deleted}, Skipped (not found): {skipped}")
    return deleted, skipped


def main():
    print("=" * 60)
    print("  Step 5: Finalize Ground Truth")
    print("=" * 60)

    with open(GT_PATH) as f:
        gt_data = json.load(f)

    # Task 1
    changed_ids = task1_apply_review_decisions(gt_data)

    # Task 2
    fin_result = task2_fix_fin_brake(gt_data)

    # Task 3
    success = task3_promote(gt_data)
    if not success:
        print("\n  ✗ ABORTING — Task 3 failed. No cleanup performed.")
        return

    # Task 4
    deleted, skipped = task4_cleanup()

    # Final report
    print("\n" + "=" * 60)
    print("  FINAL REPORT")
    print("=" * 60)
    print(f"\n  TASK 1: {len(changed_ids)} corners updated — ids: {changed_ids}")
    if fin_result:
        print(f"  TASK 2: FIN brake_start_m = {fin_result['brake_start_m']}, "
              f"support_ratio = {fin_result['support_ratio']:.3f}, "
              f"confidence = {fin_result['confidence']}")
    else:
        print("  TASK 2: FAILED — no brake detections")
    print(f"  TASK 3: Rename successful")
    print(f"          Dry-load test: PASSED")
    print(f"  TASK 4: {deleted} files deleted, {skipped} skipped (not found)")

    # Final state
    new_track = load_track("suzuka")
    all_no_review = all(
        not c.get('requires_review', False)
        for c in json.loads((TRACKS_DIR / "suzuka.json").read_text())['corners']
    )
    review_corners = [
        c['short'] for c in json.loads((TRACKS_DIR / "suzuka.json").read_text())['corners']
        if c.get('requires_review', False)
    ]

    print(f"\n  FINAL STATE:")
    print(f"  - tracks/suzuka.json          → {new_track.n_corners} corners, "
          f"all requires_review=false: {'YES' if all_no_review else 'NO'}")
    print(f"  - tracks/suzuka.original.json → exists: "
          f"{'YES' if BACKUP_PATH.exists() else 'NO'}")

    scripts = [
        "scripts/step1_build_reference_laps.py",
        "scripts/step2_segmentation.py",
        "scripts/step3_build_ground_truth.py",
        "scripts/step4_fix_subapex_and_missing.py",
    ]
    all_scripts = all((PROJECT_ROOT / s).exists() for s in scripts)
    print(f"  - Pipeline scripts             → all present: "
          f"{'YES' if all_scripts else 'NO'}")

    intermediates_gone = all(not p.exists() for p in DELETE_FILES)
    print(f"  - Intermediate files           → all deleted: "
          f"{'YES' if intermediates_gone else 'NO'}")

    if review_corners:
        print(f"\n  corners still requiring review: {review_corners}")
    else:
        print(f"\n  corners still requiring review: none")


if __name__ == "__main__":
    main()
