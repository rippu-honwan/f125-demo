"""
Corner analysis engine.
Analyzes telemetry at each corner with confidence scoring.
Works for both solo analysis (Step 2) and comparison (Step 3).
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from src.track import Track, Corner


def calculate_confidence(corner: Corner,
                         anchor_distances: np.ndarray,
                         proximity: float = 100) -> str:
    """Calculate alignment confidence for a corner."""
    for ad in anchor_distances:
        if corner.entry_m <= ad <= corner.exit_m:
            return "HIGH"
    for ad in anchor_distances:
        if (corner.entry_m - proximity) <= ad <= (corner.exit_m + proximity):
            return "MEDIUM"
    return "LOW"


def analyze_solo(data: pd.DataFrame, track: Track) -> List[Dict[str, Any]]:
    """
    Analyze corners for a single lap (Step 2).

    Args:
        data: Resampled telemetry (lap_distance, speed_kmh, etc.)
        track: Track definition

    Returns:
        List of corner analysis dicts
    """
    dist = data['lap_distance'].values
    speed = data['speed_kmh'].values
    results = []

    for corner in track.corners:
        mask = (dist >= corner.entry_m) & (dist <= corner.exit_m)
        zone = data[mask]
        if len(zone) == 0:
            continue

        min_speed = float(zone['speed_kmh'].min())
        max_speed = float(zone['speed_kmh'].max())
        avg_speed = float(zone['speed_kmh'].mean())

        min_idx = int(zone['speed_kmh'].idxmin())
        min_dist = float(data['lap_distance'].iloc[min_idx])

        # Entry speed (30m before entry)
        pre = max(0, corner.entry_m - 30)
        pre_idx = int(np.argmin(np.abs(dist - pre)))
        entry_speed = float(speed[pre_idx])

        # Exit speed (30m after exit)
        post = min(corner.exit_m + 30, dist.max())
        post_idx = int(np.argmin(np.abs(dist - post)))
        exit_speed = float(speed[post_idx])

        # Time spent in corner
        zone_dist = dist[mask]
        zone_speed_ms = np.maximum(speed[mask] / 3.6, 1.0)
        corner_time = float(np.sum(np.diff(zone_dist) / zone_speed_ms[1:]))

        # Gear at apex
        apex_idx = int(np.argmin(np.abs(dist - corner.apex_m)))
        gear = int(data['gear'].iloc[apex_idx]) if 'gear' in data.columns else None

        # Throttle at exit
        throttle_exit = None
        if 'throttle' in data.columns:
            throttle_exit = float(data['throttle'].iloc[post_idx])

        results.append({
            'id': corner.id,
            'name': corner.name,
            'short': corner.short,
            'type': corner.type,
            'direction': corner.direction,
            'color': corner.color,
            'entry_dist': corner.entry_m,
            'apex_dist': corner.apex_m,
            'exit_dist': corner.exit_m,
            'min_speed': min_speed,
            'max_speed': max_speed,
            'avg_speed': avg_speed,
            'min_speed_dist': min_dist,
            'entry_speed': entry_speed,
            'exit_speed': exit_speed,
            'corner_time': corner_time,
            'gear': gear,
            'throttle_exit': throttle_exit,
        })

    return results


def analyze_comparison(aligned: pd.DataFrame, track: Track) -> List[Dict[str, Any]]:
    """
    Analyze corners for comparison between game and real (Step 3).

    Args:
        aligned: Aligned DataFrame with game_* and real_* columns
        track: Track definition

    Returns:
        List of corner comparison dicts
    """
    dist = aligned['lap_distance'].values
    game_anch_d = aligned.attrs.get('game_anch_d', np.array([]))

    results = []

    for corner in track.corners:
        mask = (dist >= corner.entry_m) & (dist <= corner.exit_m)
        zone = aligned[mask]
        if len(zone) == 0:
            continue

        game_min = float(zone['game_speed_kmh'].min())
        real_min = float(zone['real_speed_kmh'].min())

        game_min_idx = int(zone['game_speed_kmh'].idxmin())
        game_min_dist = float(aligned['lap_distance'].iloc[game_min_idx])

        # Apex
        apex_idx = int(np.argmin(np.abs(dist - corner.apex_m)))
        game_apex = float(aligned['game_speed_kmh'].iloc[apex_idx])
        real_apex = float(aligned['real_speed_kmh'].iloc[apex_idx])

        # Time delta
        entry_idx = int(np.argmin(np.abs(dist - corner.entry_m)))
        exit_idx = int(np.argmin(np.abs(dist - corner.exit_m)))
        t_entry = float(aligned['time_delta'].iloc[entry_idx])
        t_exit = float(aligned['time_delta'].iloc[exit_idx])
        corner_delta = t_exit - t_entry

        # Entry speed (50m before)
        pre = max(0, corner.entry_m - 50)
        pre_idx = int(np.argmin(np.abs(dist - pre)))
        game_entry_spd = float(aligned['game_speed_kmh'].iloc[pre_idx])
        real_entry_spd = float(aligned['real_speed_kmh'].iloc[pre_idx])

        # Exit speed (50m after)
        post = min(corner.exit_m + 50, dist.max())
        post_idx = int(np.argmin(np.abs(dist - post)))
        game_exit_spd = float(aligned['game_speed_kmh'].iloc[post_idx])
        real_exit_spd = float(aligned['real_speed_kmh'].iloc[post_idx])

        # Brake point
        brake_threshold = game_entry_spd - 20
        game_brake_dist = None
        real_brake_dist = None
        for i in range(pre_idx, exit_idx):
            if (game_brake_dist is None and
                aligned['game_speed_kmh'].iloc[i] < brake_threshold):
                game_brake_dist = float(dist[i])
            if 'real_speed_kmh' in aligned.columns:
                real_t = real_entry_spd - 20
                if (real_brake_dist is None and
                    aligned['real_speed_kmh'].iloc[i] < real_t):
                    real_brake_dist = float(dist[i])

        confidence = calculate_confidence(corner, game_anch_d)

        spd_delta = game_min - real_min
        if abs(spd_delta) > 60:
            confidence = "LOW"

        results.append({
            'id': corner.id,
            'name': corner.name,
            'short': corner.short,
            'type': corner.type,
            'direction': corner.direction,
            'color': corner.color,
            'entry_dist': corner.entry_m,
            'apex_dist': corner.apex_m,
            'exit_dist': corner.exit_m,
            'game_min_speed': game_min,
            'real_min_speed': real_min,
            'min_speed_delta': spd_delta,
            'game_min_dist': game_min_dist,
            'game_apex_speed': game_apex,
            'real_apex_speed': real_apex,
            'game_entry_speed': game_entry_spd,
            'real_entry_speed': real_entry_spd,
            'game_exit_speed': game_exit_spd,
            'real_exit_speed': real_exit_spd,
            'game_brake_dist': game_brake_dist,
            'real_brake_dist': real_brake_dist,
            'time_delta': corner_delta,
            'confidence': confidence,
        })

    return results


def summarize_corners(corners: List[Dict], mode: str = "comparison") -> Dict:
    """Generate summary statistics."""
    if mode == "comparison":
        reliable = [c for c in corners if c.get('confidence', '') != 'LOW']
        losses = [c for c in reliable if c.get('time_delta', 0) > 0.02]
        gains = [c for c in reliable if c.get('time_delta', 0) < -0.02]

        worst = max(reliable, key=lambda c: c.get('time_delta', 0)) if reliable else None
        best = min(reliable, key=lambda c: c.get('time_delta', 0)) if reliable else None

        conf_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for c in corners:
            conf_counts[c.get('confidence', 'LOW')] += 1

        return {
            'total': len(corners),
            'reliable': len(reliable),
            'losses': len(losses),
            'gains': len(gains),
            'worst': worst,
            'best': best,
            'confidence': conf_counts,
        }

    # Solo mode
    slowest = min(corners, key=lambda c: c['min_speed']) if corners else None
    fastest = max(corners, key=lambda c: c['min_speed']) if corners else None

    return {
        'total': len(corners),
        'slowest': slowest,
        'fastest': fastest,
    }