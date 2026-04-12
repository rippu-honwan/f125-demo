"""
F1 Lap Insight - Global Configuration
"""

from pathlib import Path

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
TRACKS_DIR = PROJECT_ROOT / "tracks"

OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# Default Settings
# ============================================================
DEFAULT_CSV = str(DATA_DIR / "f1_telemetry.csv")
DEFAULT_TRACK = "suzuka"

# ============================================================
# Comparison Defaults
# ============================================================
DEFAULT_YEAR = 2024
DEFAULT_DRIVER = "VER"
DEFAULT_SESSION = "Q"

# ============================================================
# Visual Theme
# ============================================================
BG_COLOR = "#0f0f1a"
GAME_COLOR = "#00ff88"
REAL_COLOR = "#ff4488"
DPI_SAVE = 200