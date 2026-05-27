# Graph Report - .  (2026-05-28)

## Corpus Check
- Corpus is ~20,820 words - fits in a single context window. You may not need a graph.

## Summary
- 321 nodes · 642 edges · 14 communities (12 shown, 2 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 16 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Track Shape Alignment|Track Shape Alignment]]
- [[_COMMUNITY_Coaching Report Visuals|Coaching Report Visuals]]
- [[_COMMUNITY_Telemetry & Track Loading|Telemetry & Track Loading]]
- [[_COMMUNITY_CSV Data Ingestion|CSV Data Ingestion]]
- [[_COMMUNITY_Pipeline & FastF1|Pipeline & FastF1]]
- [[_COMMUNITY_Track Map Rendering|Track Map Rendering]]
- [[_COMMUNITY_Corner Coaching Analysis|Corner Coaching Analysis]]
- [[_COMMUNITY_Corner & Track Models|Corner & Track Models]]
- [[_COMMUNITY_Lap Comparison Script|Lap Comparison Script]]
- [[_COMMUNITY_Monza Track Data|Monza Track Data]]
- [[_COMMUNITY_Suzuka Track Data|Suzuka Track Data]]
- [[_COMMUNITY_Claude Settings|Claude Settings]]
- [[_COMMUNITY_Global Config|Global Config]]

## God Nodes (most connected - your core abstractions)
1. `smooth()` - 23 edges
2. `load_and_prepare()` - 23 edges
3. `style_axis()` - 19 edges
4. `draw_track_map()` - 19 edges
5. `format_laptime()` - 18 edges
6. `run_pipeline()` - 17 edges
7. `main()` - 14 edges
8. `align_track_shapes()` - 14 edges
9. `delta_color()` - 13 edges
10. `DataFrame` - 13 edges

## Surprising Connections (you probably didn't know these)
- `_card_score()` --calls--> `delta_color()`  [EXTRACTED]
  scripts/04_coaching_report.py → src/plotting.py
- `_card_times()` --calls--> `delta_color()`  [EXTRACTED]
  scripts/04_coaching_report.py → src/plotting.py
- `_card_times()` --calls--> `format_laptime()`  [EXTRACTED]
  scripts/04_coaching_report.py → src/utils.py
- `_panel_speed_grades()` --calls--> `smooth()`  [EXTRACTED]
  scripts/04_coaching_report.py → src/utils.py
- `_panel_priority_fixes()` --calls--> `format_laptime()`  [EXTRACTED]
  scripts/04_coaching_report.py → src/utils.py

## Communities (14 total, 2 thin omitted)

### Community 0 - "Track Shape Alignment"
Cohesion: 0.08
Nodes (45): align_track_shapes(), align_two_pass(), apply_transform(), _compute_alignment_quality(), _compute_confidence(), compute_curvature(), _curvature_correlation(), estimate_rigid_transform() (+37 more)

### Community 1 - "Coaching Report Visuals"
Cohesion: 0.10
Nodes (40): calculate_skills(), _card_consistency(), _card_radar(), _card_score(), _card_times(), main(), page_action_plan(), page_brake_throttle() (+32 more)

### Community 2 - "Telemetry & Track Loading"
Cohesion: 0.09
Nodes (33): get_sectors_from_track(), main(), parse_args(), plot_lap_table(), plot_overview(), print_lap_table(), Extract sector boundaries from track object., Print a formatted lap time table to terminal. (+25 more)

### Community 3 - "CSV Data Ingestion"
Cohesion: 0.11
Nodes (36): calculate_speed(), detect_separator(), extract_laps(), find_best_lap(), get_car_id(), get_lap_summary(), get_lap_time(), get_track_id() (+28 more)

### Community 4 - "Pipeline & FastF1"
Cohesion: 0.08
Nodes (32): ArgumentParser, CoachingReport, CoachingReport, Full coaching report for a lap., load_real_telemetry(), _normalize_brake(), _normalize_throttle(), Any (+24 more)

### Community 5 - "Track Map Rendering"
Cohesion: 0.11
Nodes (28): Figure, main(), _direction_offset(), draw_all_modes(), _draw_schematic_fallback(), draw_track_map(), _extract_track_xy(), _find_corner_xy() (+20 more)

### Community 6 - "Corner Coaching Analysis"
Cohesion: 0.10
Nodes (27): analyze_corner(), _assign_overall_grade(), _build_braking_summary(), _build_throttle_summary(), _calculate_consistency(), CornerInsight, find_brake_point(), find_brake_point_by_pedal() (+19 more)

### Community 7 - "Corner & Track Models"
Cohesion: 0.19
Nodes (17): Corner, analyze_comparison(), analyze_solo(), calculate_confidence(), Any, DataFrame, float, str (+9 more)

### Community 8 - "Lap Comparison Script"
Cohesion: 0.22
Nodes (16): compute_corner_deltas(), grade_corner(), main(), plot_ai_tips(), plot_corner_grade_bars(), plot_grade_scale(), plot_input_panel(), plot_scorecard() (+8 more)

### Community 9 - "Monza Track Data"
Cohesion: 0.25
Nodes (7): corners, country, drs_zones, length_m, name, sectors, short

### Community 10 - "Suzuka Track Data"
Cohesion: 0.25
Nodes (7): corners, country, drs_zones, length_m, name, sectors, short

## Knowledge Gaps
- **24 isolated node(s):** `allow`, `name`, `country`, `short`, `length_m` (+19 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_pipeline()` connect `Pipeline & FastF1` to `Track Shape Alignment`, `Coaching Report Visuals`, `CSV Data Ingestion`, `Track Map Rendering`, `Corner Coaching Analysis`, `Lap Comparison Script`?**
  _High betweenness centrality (0.252) - this node is a cross-community bridge._
- **Why does `load_and_prepare()` connect `CSV Data Ingestion` to `Telemetry & Track Loading`, `Pipeline & FastF1`?**
  _High betweenness centrality (0.221) - this node is a cross-community bridge._
- **Why does `smooth()` connect `Track Shape Alignment` to `Coaching Report Visuals`, `Telemetry & Track Loading`, `Pipeline & FastF1`, `Corner Coaching Analysis`, `Lap Comparison Script`?**
  _High betweenness centrality (0.214) - this node is a cross-community bridge._
- **What connects `F1 Lap Insight - Global Configuration`, `allow`, `name` to the rest of the system?**
  _127 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Track Shape Alignment` be split into smaller, more focused modules?**
  _Cohesion score 0.08405797101449275 - nodes in this community are weakly interconnected._
- **Should `Coaching Report Visuals` be split into smaller, more focused modules?**
  _Cohesion score 0.0975609756097561 - nodes in this community are weakly interconnected._
- **Should `Telemetry & Track Loading` be split into smaller, more focused modules?**
  _Cohesion score 0.09390243902439024 - nodes in this community are weakly interconnected._