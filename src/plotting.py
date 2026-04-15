"""
Shared plot styling and helpers.
Used by scripts 03, 04 and any future visualization.

Eliminates duplicated style_axis(), add_corner_shading(),
color constants, etc. across scripts.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from config import BG_COLOR


# ============================================================
# Color scheme (single source of truth)
# ============================================================

COLORS = {
    'game':       '#00ff88',
    'real':       '#ff4444',
    'faster':     '#00ff88',
    'slower':     '#ff4444',
    'neutral':    '#ffaa00',
    'bg':         BG_COLOR,
    'card_bg':    '#1a1a2e',
    'text':       '#ffffff',
    'text_dim':   '#888888',
    'grid':       '#333333',
    'grade_A':    '#00ff88',
    'grade_B':    '#88ff00',
    'grade_C':    '#ffaa00',
    'grade_D':    '#ff6600',
    'grade_F':    '#ff0000',
}

GRADE_COLORS = {
    'A+': COLORS['grade_A'],
    'A':  COLORS['grade_A'],
    'B+': COLORS['grade_B'],
    'B':  COLORS['grade_B'],
    'C':  COLORS['grade_C'],
    'D':  COLORS['grade_D'],
    'F':  COLORS['grade_F'],
}


def grade_color(grade: str) -> str:
    """Get color for a grade string."""
    return GRADE_COLORS.get(grade, COLORS['neutral'])


def delta_color(delta: float) -> str:
    """Green if faster (negative delta), red if slower."""
    return COLORS['faster'] if delta <= 0 else COLORS['slower']


# ============================================================
# Axis styling
# ============================================================

def style_axis(ax, ylabel=None, xlabel=None, title=None):
    """Apply consistent dark theme to an axis."""
    ax.set_facecolor(COLORS['bg'])
    ax.tick_params(colors='white', labelsize=8)
    ax.grid(alpha=0.08, color='white')
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
    if ylabel:
        ax.set_ylabel(ylabel, color='white', fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color='white', fontsize=9)
    if title:
        ax.set_title(title, color='white', fontsize=11,
                     fontweight='bold', loc='left', pad=8)


def style_card(ax):
    """Turn an axis into a card-style panel (no ticks, bordered)."""
    ax.set_facecolor(COLORS['card_bg'])
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rect = FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.02",
        facecolor=COLORS['card_bg'],
        edgecolor=COLORS['grid'], linewidth=1
    )
    ax.add_patch(rect)
    return ax


def add_corner_shading(ax, corners, track_length, alpha=0.06):
    """Add corner zones as background shading."""
    if corners is None:
        return
    for c in corners:
        entry = c.get('entry_m', c['apex_m'] - 50)
        exit_m = c.get('exit_m', c['apex_m'] + 50)
        if entry < track_length and exit_m > 0:
            ax.axvspan(entry, exit_m, alpha=alpha, color='#ffffff')


# ============================================================
# Figure helpers
# ============================================================

def create_figure(width=28, height=20, title=None):
    """Create a dark-themed figure."""
    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor(COLORS['bg'])
    if title:
        fig.suptitle(title, color='white', fontsize=16,
                     fontweight='bold', y=0.98)
    return fig


def save_figure(fig, path, dpi=150):
    """Save figure with error handling."""
    try:
        fig.savefig(path, dpi=dpi, bbox_inches='tight',
                    facecolor=COLORS['bg'])
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  ⚠ Failed to save {path}: {e}")
    finally:
        plt.close(fig)


def wrap_text(text, max_chars=60):
    """Word wrap text into lines."""
    words = text.split()
    lines = []
    line = ""
    for word in words:
        test = (line + " " + word).strip()
        if len(test) > max_chars:
            if line:
                lines.append(line)
            line = word
        else:
            line = test
    if line:
        lines.append(line)
    return lines