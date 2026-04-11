"""
F1 Lap Insight - Central Configuration
All settings in one place. Change here, affects everywhere.
"""

from pathlib import Path

# ============================================================
# Paths
# ============================================================
DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Default session file
DEFAULT_CSV = DATA_DIR / "SRT_4-11_2lap-suzuka.csv"

# ============================================================
# Data Processing
# ============================================================
MIN_LAP_POINTS = 1000       # Minimum data points for a valid lap
RESAMPLE_RESOLUTION = 1.0   # meters per data point
SMOOTH_WINDOW = 15          # smoothing window for curvature etc.

# ============================================================
# Visualization
# ============================================================
BG_COLOR = '#0a0e1a'
TRACK_COLOR_OUTER = '#0f0f0f'
TRACK_COLOR_INNER = '#2a2a2a'
SPEED_LINE_COLOR = '#00ff88'
BRAKE_COLOR = '#ff4444'
THROTTLE_COLOR = '#44ff44'

DPI_SAVE = 200
FIGSIZE_MAP = (26, 22)
FIGSIZE_SPEED = (28, 13)
FIGSIZE_EDA = (28, 16)

# ============================================================
# Corner Colors (18 turns)
# ============================================================
CORNER_COLORS = [
    "#ff4444", "#ff6b35", "#ff9500", "#ffb700",
    "#ffd500", "#d4ed26", "#66bb6a", "#26a69a",
    "#29b6f6", "#42a5f5", "#5c6bc0", "#7e57c2",
    "#ab47bc", "#ec407a", "#ef5350", "#ff7043",
    "#8d6e63", "#78909c"
]