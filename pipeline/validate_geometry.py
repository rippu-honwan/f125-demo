#!/usr/bin/env python3
"""
Quick corner-geometry validation for tracks/{track}.json.

Asserts entry_m < apex_m < exit_m for every corner. Prints any violations.
Fixes violations by clamping entry_m/exit_m to ±PAD m around apex_m, then
writes the file back. Intended to run before step4 of the pipeline.

  python pipeline/validate_geometry.py --track spa
"""

import sys
import json
import argparse
from pathlib import Path

TRACKS_DIR = Path(__file__).parent.parent / "tracks"
PAD = 80  # metres of padding to enforce around apex when fixing a violation


def validate_and_fix(track):
    path = TRACKS_DIR / f"{track}.json"
    with open(path) as f:
        data = json.load(f)

    violations = []
    for c in data["corners"]:
        entry, apex, exit_ = c["entry_m"], c["apex_m"], c["exit_m"]
        if not (entry < apex < exit_):
            violations.append((c, entry, apex, exit_))

    if not violations:
        print(f"  ✓ {len(data['corners'])} corners valid — entry_m < apex_m < exit_m holds for all")
        return 0

    print(f"  ⚠️  {len(violations)} violation(s) of entry_m < apex_m < exit_m:")
    for c, entry, apex, exit_ in violations:
        new_entry = apex - PAD
        new_exit = apex + PAD
        print(f"    {c['short']} {c['name']}: "
              f"entry={entry} apex={apex} exit={exit_} "
              f"-> entry={new_entry} exit={new_exit} (±{PAD}m padding)")
        c["entry_m"] = new_entry
        c["exit_m"] = new_exit

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Fixed {len(violations)} corner(s) and wrote {path.name}")
    return len(violations)


def main():
    parser = argparse.ArgumentParser(description="Validate/fix corner geometry.")
    parser.add_argument("--track", required=True, type=str)
    args = parser.parse_args()
    validate_and_fix(args.track)


if __name__ == "__main__":
    main()
