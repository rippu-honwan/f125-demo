# Ground Truth Pipeline

Builds telemetry-backed corner apex zones for any F1 track
using FastF1 data (71+ laps per session).

## Quick Start

```bash
python pipeline/step1_build_reference_laps.py \
  --track <short> --year <year> --session Q
python pipeline/step2_segmentation.py         --track <short>
python pipeline/step3_build_ground_truth.py   --track <short>
python pipeline/step4_fix_subapex_and_missing.py --track <short>
python pipeline/step5_finalize.py             --track <short>
```

## All Supported Tracks (24 tracks)

| Short name   | Grand Prix name              | JSON exists | Status    |
|--------------|------------------------------|-------------|-----------|
| bahrain      | Bahrain Grand Prix           | No          | Pending   |
| jeddah       | Saudi Arabian Grand Prix     | No          | Pending   |
| lusail       | Qatar Grand Prix             | No          | Pending   |
| yas_marina   | Abu Dhabi Grand Prix         | No          | Pending   |
| baku         | Azerbaijan Grand Prix        | No          | Pending   |
| suzuka       | (see registry)               | Yes         | Complete  |
| shanghai     | Chinese Grand Prix           | Yes         | Skeleton  |
| singapore    | Singapore Grand Prix         | Yes         | Skeleton  |
| melbourne    | Australian Grand Prix        | Yes         | Skeleton  |
| monaco       | Monaco Grand Prix            | No          | Pending   |
| barcelona    | Spanish Grand Prix           | Yes         | Skeleton  |
| silverstone  | British Grand Prix           | Yes         | Skeleton  |
| spielberg    | Austrian Grand Prix          | No          | Pending   |
| hungaroring  | Hungarian Grand Prix         | No          | Pending   |
| spa          | Belgian Grand Prix           | Yes         | Skeleton  |
| zandvoort    | Dutch Grand Prix             | No          | Pending   |
| monza        | Italian Grand Prix           | Yes         | Skeleton  |
| imola        | Emilia Romagna Grand Prix    | No          | Pending   |
| miami        | Miami Grand Prix             | Yes         | Skeleton  |
| canada       | Canadian Grand Prix          | No          | Pending   |
| austin       | United States Grand Prix     | Yes         | Skeleton  |
| mexico       | Mexico City Grand Prix       | No          | Pending   |
| interlagos   | São Paulo Grand Prix         | Yes         | Skeleton  |
| las_vegas    | Las Vegas Grand Prix         | No          | Pending   |

## Adding a New Track

1. Add to `TRACK_MAP` in `src/track_registry.py`
2. Create `tracks/{short}.json` with corner skeleton
3. Add `compound_zones` if track has complex multi-apex sections
4. Run step1–5

## Output Files

| Filename pattern                          | Created by | Deleted by |
|-------------------------------------------|------------|------------|
| `{prefix}_reference_laps.parquet`         | step1      | step5      |
| `{prefix}_reference_laps_metadata.json`   | step1      | step5      |
| `{prefix}_reference_laps_overview.png`    | step1      | step5      |
| `{prefix}_segments_consensus.json`        | step2      | step5      |
| `{prefix}_segmentation_overview.png`      | step2      | step5      |
| `{prefix}_phase_detection_per_lap.parquet` | step2     | step5      |
| `{prefix}_ground_truth_validation.png`    | step3      | step5      |
| `{prefix}_ground_truth_report.md`         | step3      | step5      |
| `{prefix}_fix_validation.png`             | step4      | step5      |
| `tracks/{short}.ground_truth.json`        | step3      | step5 (renamed) |
| `tracks/{short}.json`                     | step5      | —          |
| `tracks/{short}.original.json`            | step5      | —          |

## Compound Zones

Compound zones define multi-apex corner complexes where the pipeline should
detect individual sub-apexes rather than treating the section as one corner.

Configuration in `tracks/{short}.json`:

```json
"compound_zones": [
  {
    "label": "s_curves",
    "start_m": 950,
    "end_m": 1650,
    "subapex_method": "curvature_peaks",
    "prominence_threshold": 0.0003,
    "expected_peaks": 4
  }
]
```

Fields:
- `label`: identifier matching the `complex` field in corner entries
- `start_m` / `end_m`: distance window for the complex
- `subapex_method`: `"curvature_peaks"` or `"speed_minima"`
- `prominence_threshold`: peak detection sensitivity (null = default)
- `expected_peaks`: number of sub-apexes to detect

If `compound_zones` is empty or absent, step4 skips sub-apex detection
and exits cleanly.
