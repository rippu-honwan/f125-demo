"""
Track definition loader.
Reads track JSON files and provides corner/sector data.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


TRACKS_DIR = Path(__file__).parent.parent / "tracks"

CORNER_COLORS = [
    "#ff4444", "#ff6633", "#ff9500", "#ffb300",
    "#ffd500", "#d4ed26", "#66bb6a", "#26a69a",
    "#29b6f6", "#42a5f5", "#5c6bc0", "#7e57c2",
    "#ab47bc", "#ec407a", "#ef5350", "#ff7043",
    "#8d6e63", "#78909c", "#90a4ae", "#a1887f",
]


@dataclass
class Corner:
    id: int
    name: str
    short: str
    type: str           # low_speed, medium_speed, high_speed, flat_out
    direction: str      # left, right
    entry_m: float
    apex_m: float
    exit_m: float
    color: str = ""

    @property
    def length(self) -> float:
        return self.exit_m - self.entry_m

    @property
    def is_slow(self) -> bool:
        return self.type == "low_speed"

    @property
    def is_fast(self) -> bool:
        return self.type in ("high_speed", "flat_out")


@dataclass
class Sector:
    id: int
    name: str
    start_m: float
    end_m: float

    @property
    def length(self) -> float:
        return self.end_m - self.start_m


@dataclass
class DRSZone:
    start_m: float
    end_m: float
    detection_m: float = 0.0


@dataclass
class Track:
    name: str
    short: str
    country: str
    length_m: float
    corners: List[Corner]
    sectors: List[Sector]
    drs_zones: List[DRSZone]
    fastf1_race_name: str = ""

    @property
    def n_corners(self) -> int:
        return len(self.corners)

    def get_corner(self, corner_id: int) -> Optional[Corner]:
        for c in self.corners:
            if c.id == corner_id:
                return c
        return None

    def get_sector(self, sector_id: int) -> Optional[Sector]:
        for s in self.sectors:
            if s.id == sector_id:
                return s
        return None

    def corners_in_sector(self, sector_id: int) -> List[Corner]:
        sector = self.get_sector(sector_id)
        if sector is None:
            return []
        return [c for c in self.corners
                if sector.start_m <= c.apex_m <= sector.end_m]

    def corner_tuples(self):
        """Legacy format: list of (id, name, short, entry, apex, exit)."""
        return [
            (c.id, c.name, c.short, c.entry_m, c.apex_m, c.exit_m)
            for c in self.corners
        ]


def load_track(track_name: str) -> Track:
    """
    Load track definition from JSON file.

    Args:
        track_name: e.g. "suzuka", "monza", "spa"

    Returns:
        Track dataclass
    """
    path = TRACKS_DIR / f"{track_name.lower()}.json"

    if not path.exists():
        available = [f.stem for f in TRACKS_DIR.glob("*.json")]
        raise FileNotFoundError(
            f"Track '{track_name}' not found at {path}\n"
            f"Available: {available}"
        )

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    corners = []
    for i, c in enumerate(data.get('corners', [])):
        corners.append(Corner(
            id=c['id'],
            name=c['name'],
            short=c['short'],
            type=c.get('type', 'medium_speed'),
            direction=c.get('direction', 'right'),
            entry_m=c['entry_m'],
            apex_m=c['apex_m'],
            exit_m=c['exit_m'],
            color=CORNER_COLORS[i % len(CORNER_COLORS)],
        ))

    sectors = []
    for s in data.get('sectors', []):
        sectors.append(Sector(
            id=s['id'],
            name=s.get('name', f"Sector {s['id']}"),
            start_m=s['start_m'],
            end_m=s['end_m'],
        ))

    drs_zones = []
    for d in data.get('drs_zones', []):
        drs_zones.append(DRSZone(
            start_m=d['start_m'],
            end_m=d['end_m'],
            detection_m=d.get('detection_m', 0),
        ))

    fastf1_data = data.get('fastf1', {})

    track = Track(
        name=data['name'],
        short=data.get('short', track_name.lower()),
        country=data.get('country', ''),
        length_m=data['length_m'],
        corners=corners,
        sectors=sectors,
        drs_zones=drs_zones,
        fastf1_race_name=fastf1_data.get('race_name', ''),
    )

    return track


def list_tracks() -> List[str]:
    """List all available track names."""
    return sorted([f.stem for f in TRACKS_DIR.glob("*.json")])


def auto_detect_track(track_length: float, tolerance: float = 200) -> Optional[Track]:
    """
    Auto-detect track from telemetry track length.

    Args:
        track_length: measured track length in meters
        tolerance: matching tolerance in meters

    Returns:
        Track if matched, None otherwise
    """
    best_match = None
    best_diff = float('inf')

    for name in list_tracks():
        try:
            track = load_track(name)
            diff = abs(track.length_m - track_length)
            if diff < tolerance and diff < best_diff:
                best_diff = diff
                best_match = track
        except Exception:
            continue

    return best_match