"""
Track Registry — config-driven track lookup for the ground truth pipeline.

Provides GP name resolution, output prefix generation, and track JSON loading
for all current F1 calendar tracks (2024–2025).
"""

import json
from pathlib import Path

TRACKS_DIR = Path(__file__).parent.parent / "tracks"

# ============================================================
# Track Map: short_name → Grand Prix name
# ============================================================
TRACK_MAP = {
    # Middle East
    "bahrain": "Bahrain Grand Prix",
    "jeddah": "Saudi Arabian Grand Prix",
    "lusail": "Qatar Grand Prix",
    "yas_marina": "Abu Dhabi Grand Prix",
    "baku": "Azerbaijan Grand Prix",

    # Asia-Pacific
    "suzuka": "Japanese Grand Prix",
    "shanghai": "Chinese Grand Prix",
    "singapore": "Singapore Grand Prix",
    "melbourne": "Australian Grand Prix",

    # Europe
    "monaco": "Monaco Grand Prix",
    "barcelona": "Spanish Grand Prix",
    "silverstone": "British Grand Prix",
    "spielberg": "Austrian Grand Prix",
    "hungaroring": "Hungarian Grand Prix",
    "spa": "Belgian Grand Prix",
    "zandvoort": "Dutch Grand Prix",
    "monza": "Italian Grand Prix",
    "imola": "Emilia Romagna Grand Prix",

    # Americas
    "miami": "Miami Grand Prix",
    "canada": "Canadian Grand Prix",
    "austin": "United States Grand Prix",
    "mexico": "Mexico City Grand Prix",
    "interlagos": "São Paulo Grand Prix",

    # Africa
    "las_vegas": "Las Vegas Grand Prix",
}


def get_gp_name(track_short: str) -> str:
    """Return the Grand Prix name for a track short name.

    Raises ValueError with valid key list if not found.
    """
    if track_short not in TRACK_MAP:
        valid = sorted(TRACK_MAP.keys())
        raise ValueError(
            f"Unknown track '{track_short}'. Valid keys: {valid}"
        )
    return TRACK_MAP[track_short]


def get_output_prefix(track_short: str) -> str:
    """Return track_short as-is for use in filenames."""
    # Validate the key exists
    get_gp_name(track_short)
    return track_short


def get_compound_zones(track_short: str) -> list[dict]:
    """Load compound_zones from tracks/{track_short}.json.

    Returns [] if file not found or key absent.
    Each zone dict has keys: label, start_m, end_m,
    and optionally: subapex_method, prominence_threshold, expected_peaks.
    """
    json_path = TRACKS_DIR / f"{track_short}.json"
    if not json_path.exists():
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("compound_zones", [])
    except (json.JSONDecodeError, OSError):
        return []


def get_track_corners(track_short: str) -> list[dict]:
    """Load corners list from tracks/{track_short}.json.

    Returns [] if file missing.
    """
    json_path = TRACKS_DIR / f"{track_short}.json"
    if not json_path.exists():
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("corners", [])
    except (json.JSONDecodeError, OSError):
        return []
