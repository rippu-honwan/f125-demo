#!/usr/bin/env python3
"""
F1 Ground Truth Pipeline — Orchestrator.

Runs step1 through step5 sequentially for a given track.
python pipeline/run_pipeline.py \
  --track silverstone --year 2025 --session Q
"""

import sys
import argparse
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PIPELINE_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"

STEPS = [
    (1, "step1_build_reference_laps.py"),
    (2, "step2_segmentation.py"),
    (3, "step3_build_ground_truth.py"),
    (4, "step4_fix_subapex_and_missing.py"),
    (5, "step5_finalize.py"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full ground truth pipeline for an F1 track."
    )
    parser.add_argument("--track", required=True, type=str,
                        help="Track short name (e.g. suzuka, monza)")
    parser.add_argument("--year", required=True, type=int,
                        help="Season year (e.g. 2025)")
    parser.add_argument("--session", default="Q", type=str,
                        help="Session type (default: Q)")
    parser.add_argument("--driver", default="ALL", type=str,
                        help="Driver code or ALL (default: ALL)")
    parser.add_argument("--skip-review", action="store_true", default=False,
                        help="Skip the review gate after step3")
    parser.add_argument("--from-step", default=1, type=int, choices=[1, 2, 3, 4, 5],
                        help="Resume from a specific step (default: 1)")
    return parser.parse_args()


def print_box(lines):
    """Print a unicode box around lines of text."""
    width = 42
    print(f"╔{'═' * width}╗")
    for line in lines:
        print(f"║  {line:<{width - 2}}║")
    print(f"╚{'═' * width}╝")


def build_step_args(step_num, args):
    """Build CLI arguments for a given step."""
    if step_num == 1:
        return ["--track", args.track, "--year", str(args.year),
                "--session", args.session, "--driver", args.driver]
    else:
        return ["--track", args.track]


def parse_review_report(prefix):
    """Parse ground_truth_report.md for review gate info."""
    report_path = OUTPUT_DIR / f"{prefix}_ground_truth_report.md"
    if not report_path.exists():
        return None

    text = report_path.read_text(encoding="utf-8")

    # Parse confidence distribution from table
    high = med = low = total = 0
    for line in text.splitlines():
        if "| High" in line:
            parts = line.split("|")
            high = int(parts[2].strip())
        elif "| Medium" in line:
            parts = line.split("|")
            med = int(parts[2].strip())
        elif "| Low" in line:
            parts = line.split("|")
            low = int(parts[2].strip())
        elif "**Total**" in line:
            parts = line.split("|")
            for p in parts:
                p = p.strip().strip("*")
                if p.isdigit():
                    total = int(p)
                    break

    if total == 0:
        total = high + med + low

    # Count requires_review items (section 4 bullet points)
    review_count = 0
    in_section4 = False
    for line in text.splitlines():
        if "## 4." in line or "Requiring Human Review" in line:
            in_section4 = True
            continue
        if in_section4:
            if line.startswith("## "):
                break
            if line.startswith("- **"):
                review_count += 1
            if "None" in line and "all corners accepted" in line:
                review_count = 0
                break

    return {
        "total": total,
        "high": high,
        "medium": med,
        "low": low,
        "requires_review": review_count,
    }


def main():
    args = parse_args()
    prefix = args.track
    total_start = time.perf_counter()

    # Header
    print_box([
        "F1 Ground Truth Pipeline",
        f"Track  : {args.track}",
        f"Year   : {args.year}  Session: {args.session}",
    ])

    # Resume hint
    if args.from_step > 1:
        print(f"\n  ℹ️  Resuming from step {args.from_step} "
              f"(steps 1–{args.from_step - 1} skipped)\n")

    for step_num, script_name in STEPS:
        if step_num < args.from_step:
            continue

        # Review gate — after step3, before step4
        if step_num == 4 and args.from_step <= 3:
            info = parse_review_report(prefix)
            if info:
                print("┌─────────────────────────────────────┐")
                print("│  Step 3 Review Summary              │")
                print(f"│  Corners total    : {info['total']:<15}│")
                print(f"│  High confidence  : {info['high']:<15}│")
                print(f"│  Medium confidence: {info['medium']:<15}│")
                print(f"│  Low confidence   : {info['low']:<15}│")
                print(f"│  Requires review  : {info['requires_review']:<15}│")
                print("└─────────────────────────────────────┘")

                if info["requires_review"] > 0 and not args.skip_review:
                    r = info["requires_review"]
                    print(f"\n  ⚠️  {r} corner(s) require human review before finalising.\n")
                    print("  Please check:")
                    print(f"    open output/{prefix}_ground_truth_validation.png")
                    print(f"    open output/{prefix}_ground_truth_report.md")
                    print(f"\n  When ready, resume with:")
                    print(f"    python pipeline/run_pipeline.py \\")
                    print(f"      --track {args.track} --year {args.year} \\")
                    print(f"      --from-step 4 --skip-review\n")
                    sys.exit(0)
                else:
                    print("  ✓ Review gate passed — continuing to step4\n")

        # Run step
        script_path = PIPELINE_DIR / script_name
        step_args = build_step_args(step_num, args)

        print_box([f"Step {step_num}/5 — {script_name}"])

        step_start = time.perf_counter()
        result = subprocess.run(
            [sys.executable, str(script_path)] + step_args,
            check=False,
        )
        elapsed = time.perf_counter() - step_start

        if result.returncode != 0:
            print(f"\n  ✗ Step {step_num} FAILED (exit code {result.returncode})")
            print(f"  Pipeline stopped. Fix the error and re-run with:")
            print(f"    python pipeline/run_pipeline.py --track {args.track} \\")
            print(f"      --year {args.year} --session {args.session} "
                  f"--from-step {step_num}\n")
            sys.exit(1)

        print(f"\n  ✓ Step {step_num} complete — {elapsed:.1f}s\n")

    # Final summary
    total_elapsed = time.perf_counter() - total_start
    print_box([
        "Pipeline Complete",
        f"Track  : {args.track}",
        f"Time   : {total_elapsed:.1f}s",
        f"Output : tracks/{args.track}.json",
    ])
    print(f"\n  Next step: use tracks/{args.track}.json in scripts/03_lap_comparison.py")
    print(f"    python scripts/03_lap_comparison.py data/my_lap.csv \\")
    print(f"      --driver VER --year {args.year} --session {args.session} \\")
    print(f"      --track {args.track}\n")


if __name__ == "__main__":
    main()
