"""
F1 AI Driver Coach — FastAPI backend.

Thin web layer over the existing analysis pipeline in ``src/``.
Nothing in ``src/`` is modified; this module only *consumes* it.

Endpoints
---------
GET  /          -> serves the single-page premium UI (app/static/index.html)
POST /analyze   -> accepts a telemetry CSV upload + driver/year/session/track,
                   runs the full coaching pipeline, returns JSON (incl. the
                   interactive Track Explorer map data).
GET  /health    -> liveness probe.

Run
---
    pip install -e ".[web]"
    python -m app.main                # or: uvicorn app.main:app --reload
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

# --- Make the project root importable so `from src...` works regardless of CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Matplotlib must use a non-interactive backend BEFORE pyplot is imported,
#     otherwise it tries to open a GUI window from a server worker thread.
import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"

from fastapi import FastAPI, File, Form, UploadFile, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

# --- Existing pipeline (read-only consumption). ---
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402
from src.track_map import (  # noqa: E402
    _severity,
    _extract_track_xy,
    _find_corner_xy,
)
from src.loader import (  # noqa: E402
    detect_separator,
    get_lap_summary,
    load_and_prepare,
)
from src.corners import analyze_solo, summarize_corners  # noqa: E402
from src.track import load_track, auto_detect_track  # noqa: E402
from src.coaching import (  # noqa: E402  (read-only consumption)
    _generate_action_plan,
    _estimate_potential,
)

# Raw header names that mean "0-based lap index" in a Sim Racing Telemetry CSV.
LAP_INDEX_COLUMNS = ("lapIndex", "lap_index")

# Analysis modes exposed in the UI, mapped to the five scripts/ programs.
#   Solo modes run on the uploaded lap alone (no real F1 download) — scripts
#   01 (Telemetry Overview) and 02 (Lap Analysis).
#   Comparison modes run the full game-vs-real pipeline — scripts 03 (Lap
#   Comparison), 04 (Coaching Report) and 05 (Track Map).
SOLO_MODES = ("overview", "lap_analysis")
COMPARISON_MODES = ("comparison", "coaching", "track_map")
ALL_MODES = SOLO_MODES + COMPARISON_MODES
DEFAULT_MODE = "comparison"
DEFAULT_YEAR = 2025

# Supported tracks, in display order — the single source of truth served to the
# frontend via GET /tracks. Each "key" is the tracks/<key>.json stem understood
# by src.track.load_track(); the hardcoded <select> in index.html mirrors this
# list only as a static fallback for when /tracks can't be reached.
SUPPORTED_TRACKS = [
    {"key": "suzuka", "name": "Suzuka"},
    {"key": "monza", "name": "Monza"},
    {"key": "spa", "name": "Spa-Francorchamps"},
    {"key": "silverstone", "name": "Silverstone"},
    {"key": "singapore", "name": "Singapore"},
    {"key": "austin", "name": "Austin (COTA)"},
    {"key": "barcelona", "name": "Barcelona"},
    {"key": "interlagos", "name": "Interlagos"},
    {"key": "melbourne", "name": "Melbourne"},
    {"key": "miami", "name": "Miami"},
    {"key": "shanghai", "name": "Shanghai"},
]

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="F1 AI Driver Coach",
    description="Compare your sim-racing lap against a real F1 driver, corner by corner.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS — let the static frontend (GitHub Pages or local dev) call this API.
# ---------------------------------------------------------------------------
# The frontend may be served from a *different* origin than this backend (e.g.
# the UI on GitHub Pages, the API on Render), and browsers block cross-origin
# requests unless the server opts in. Allowed by default:
#   * any localhost / 127.0.0.1 port (local development) — matched by the
#     regex below, and
#   * the project's own GitHub Pages origin — an exact entry in allow_origins.
# Add extra exact origins (e.g. a custom domain) without touching code via the
# ``ALLOWED_ORIGINS`` env var (comma-separated).
# No cookies/credentials are used, so credentials stay off and file-upload /
# analysis POSTs work cross-origin (incl. their preflight OPTIONS).
_CORS_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
# The project's own GitHub Pages origin is always allowed (exact match).
_PROD_ORIGIN = "https://rippu-honwan.github.io"
_EXTRA_ORIGINS = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()
]
_ALLOWED_ORIGINS = [_PROD_ORIGIN, *_EXTRA_ORIGINS]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_origin_regex=_CORS_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _num(value: Any, digits: int = 2) -> Optional[float]:
    """Coerce a possibly-None / numpy scalar to a rounded JSON-safe float."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return round(f, digits)


def _parse_lap_index(value: Optional[str]) -> Optional[int]:
    """
    Map the optional ``lap_index`` form field to an int (0-based) or ``None``.

    ``None`` (and the sentinel strings "", "auto", "fastest") preserve the
    existing behaviour: let the pipeline auto-select the fastest lap.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "auto", "fastest", "none", "null"):
        return None
    try:
        return int(float(text))
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid lap selection: {value!r}.",
        )


def _read_lap_options(csv_path: str) -> "tuple[bool, list[dict]]":
    """
    Inspect an uploaded telemetry CSV and list its selectable laps.

    Returns ``(has_lap_index, options)``. ``has_lap_index`` is False when the
    file has no ``lapIndex`` column at all, so the UI can show a clear message.
    Each option is ``{lap_index, lap_number, lap_time, max_speed}``.
    """
    sep = detect_separator(csv_path)
    header = pd.read_csv(csv_path, sep=sep, nrows=0)
    header_names = {str(c).strip() for c in header.columns}
    has_lap_index = any(name in header_names for name in LAP_INDEX_COLUMNS)
    if not has_lap_index:
        return False, []

    # Preferred: the validated per-lap summary (gives lap times + top speed).
    options: "list[dict]" = []
    try:
        for s in get_lap_summary(csv_path):
            options.append({
                "lap_index": int(s["lap_index"]),
                "lap_number": int(s["lap_number"]),
                "lap_time": _num(s.get("lap_time"), 3),
                "max_speed": _num(s.get("max_speed"), 0),
            })
    except Exception:  # a few malformed rows shouldn't kill the whole listing
        traceback.print_exc()

    # Fallback: raw distinct lapIndex values (no timing) so the dropdown is
    # still usable even when no lap passes the summary's validity filters.
    if not options:
        col = pd.read_csv(
            csv_path, sep=sep,
            usecols=lambda c: str(c).strip() in LAP_INDEX_COLUMNS,
        )
        values = pd.to_numeric(col.iloc[:, 0], errors="coerce").dropna()
        for li in sorted({int(v) for v in values}):
            options.append({
                "lap_index": li,
                "lap_number": li + 1,
                "lap_time": None,
                "max_speed": None,
            })

    options.sort(key=lambda o: o["lap_index"])
    return True, options


def _corner_to_dict(ci: Any) -> dict:
    """Serialize a CornerInsight into the API shape the frontend expects."""
    return {
        "corner_id": int(ci.corner_id),
        "name": ci.name,
        "short": ci.short,
        "grade": ci.grade,
        "severity": _severity(ci),  # 'ok' | 'minor' | 'major'
        "time_delta": _num(ci.time_delta, 3),
        "issues": list(ci.issues or []),
        "tips": list(ci.tips or []),
        "brake_diff_m": _num(ci.brake_diff_m, 1),
        "apex_speed_diff": _num(ci.apex_speed_diff, 1),
        "exit_speed_diff": _num(ci.exit_speed_diff, 1),
    }


# ---------------------------------------------------------------------------
# Analysis runners (one per mode family)
# ---------------------------------------------------------------------------
def _resolve_track(track: str, track_length: float):
    """Best-effort ``Track`` object: by selected name, else auto-detect by length."""
    if track:
        try:
            return load_track(track)
        except FileNotFoundError:
            pass
    try:
        return auto_detect_track(track_length)
    except Exception:
        return None


def _detect_track_from_csv(csv_path: str) -> "tuple[Optional[str], Optional[str]]":
    """
    Read the SRT ``trackId`` column and map it to a supported track.

    Returns ``(track_key, track_name)`` when the CSV names a track we have a
    layout for, else ``(None, None)`` so the UI can fall back to manual choice.
    """
    try:
        sep = detect_separator(csv_path)
        raw = pd.read_csv(csv_path, sep=sep, nrows=1)
        if "trackId" not in raw.columns:
            return None, None
        tid = str(raw["trackId"].iloc[0]).strip().lower()
        if not tid or tid in ("nan", "none", ""):
            return None, None
        # Exact filename match first, then a loose contains-match.
        try:
            return tid, load_track(tid).name
        except FileNotFoundError:
            pass
        for p in (PROJECT_ROOT / "tracks").glob("*.json"):
            stem = p.stem.lower()
            if tid in stem or stem in tid:
                return stem, load_track(stem).name
        return None, None
    except Exception:
        traceback.print_exc()
        return None, None


def _run_solo(csv_path: str, mode: str, track: str,
              lap: Optional[int]) -> dict:
    """
    Single-lap analysis that needs no real F1 data (scripts 01 & 02).

    ``overview``     -> lap-time table for every lap + this lap's vitals.
    ``lap_analysis`` -> corner-by-corner breakdown of your own lap.
    """
    try:
        data, meta = load_and_prepare(csv_path, lap_index=lap)
    except SystemExit as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not read a lap from this CSV. ({exc})",
        )
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=422,
            detail=f"Could not read a lap from this CSV: {exc}",
        )

    track_obj = _resolve_track(track, float(meta.get("track_length") or 0.0))
    track_name = (
        track_obj.name if track_obj
        else (track.upper() if track else "Unknown Track")
    )

    payload = {
        "mode": mode,
        "track": track,
        "track_name": track_name,
        "lap_index": meta.get("lap_index"),
        "lap_number": meta.get("best_lap_number"),
        "lap_time": _num(meta.get("best_time"), 3),
        "track_length": _num(meta.get("track_length"), 0),
    }

    if mode == "overview":
        spd = data["speed_kmh"]
        payload["top_speed"] = _num(spd.max(), 0)
        payload["avg_speed"] = _num(spd.mean(), 0)
        if "throttle" in data.columns:
            payload["full_throttle_pct"] = _num(
                (data["throttle"] > 0.95).mean() * 100, 0)
        if "brake" in data.columns:
            payload["braking_pct"] = _num((data["brake"] > 0.1).mean() * 100, 0)
        if "gear" in data.columns:
            try:
                payload["max_gear"] = int(data["gear"].max())
            except (ValueError, TypeError):
                payload["max_gear"] = None

        sectors = (
            [{"start_m": s.start_m, "end_m": s.end_m} for s in track_obj.sectors]
            if track_obj and track_obj.sectors else None
        )
        laps: "list[dict]" = []
        try:
            for s in get_lap_summary(csv_path, sectors):
                laps.append({
                    "lap_index": int(s["lap_index"]),
                    "lap_number": int(s["lap_number"]),
                    "lap_time": _num(s.get("lap_time"), 3),
                    "sector_times": [_num(t, 3) for t in s.get("sector_times", [])],
                    "max_speed": _num(s.get("max_speed"), 0),
                })
        except Exception:  # a few malformed rows shouldn't kill the listing
            traceback.print_exc()
        payload["laps"] = laps
        payload["n_laps"] = len(laps)
        return payload

    # mode == "lap_analysis": corner-by-corner needs a known track layout.
    if track_obj is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Select a valid track to run Lap Analysis — corner "
                "definitions are required for a per-corner breakdown."
            ),
        )

    corners = analyze_solo(data, track_obj)
    solo = []
    for c in corners:
        solo.append({
            "id": int(c["id"]),
            "name": c.get("name", ""),
            "short": c.get("short", f"T{c['id']}"),
            "type": c.get("type", ""),
            "direction": c.get("direction", ""),
            "min_speed": _num(c.get("min_speed"), 0),
            "entry_speed": _num(c.get("entry_speed"), 0),
            "exit_speed": _num(c.get("exit_speed"), 0),
            "gear": c.get("gear"),
            "corner_time": _num(c.get("corner_time"), 3),
        })
    summary = summarize_corners(corners, mode="solo")

    def _corner_brief(c: Optional[dict]) -> Optional[dict]:
        if not c:
            return None
        return {
            "short": c.get("short"),
            "name": c.get("name"),
            "min_speed": _num(c.get("min_speed"), 0),
        }

    payload["corners_solo"] = solo
    payload["n_corners"] = len(solo)
    payload["slowest"] = _corner_brief(summary.get("slowest"))
    payload["fastest"] = _corner_brief(summary.get("fastest"))
    return payload


def _map_title(track: str, report) -> str:
    return (
        f"{track.upper()} - You vs {report.driver} "
        f"({report.year} {report.session})"
    )


# ---------------------------------------------------------------------------
# Comparison-chart helpers (Lap Comparison mode)
# ---------------------------------------------------------------------------
def _downsample(values, k: int = 36, digits: int = 3) -> "list":
    """Resample a 1-D sequence to ``k`` evenly spaced points (NaN-safe)."""
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return []
    idx = np.arange(a.size)
    good = np.isfinite(a)
    if not good.any():
        return [None] * k
    if not good.all():                      # bridge gaps so the line stays continuous
        a = np.interp(idx, idx[good], a[good])
    if a.size == 1:
        return [round(float(a[0]), digits)] * k
    xi = np.linspace(0, a.size - 1, k)
    return [round(float(v), digits) for v in np.interp(xi, idx, a)]


def _corner_charts(aligned, corner: dict, k: int = 36, pad: float = 40.0) -> dict:
    """You-vs-pro traces for a single corner window (brake/throttle/gear/speed)."""
    dist = aligned["lap_distance"].values
    apex = float(corner.get("apex_m", 0.0))
    entry = float(corner.get("entry_m", apex - 60)) - pad
    exit_m = float(corner.get("exit_m", apex + 60)) + pad
    mask = (dist >= entry) & (dist <= exit_m)
    if mask.sum() < 4:                       # fall back to a fixed window around the apex
        mask = (dist >= apex - 120) & (dist <= apex + 120)

    def chan(col, digits):
        if col not in aligned.columns:
            return []
        return _downsample(aligned[col].values[mask], k, digits)

    def gear(col):
        return [None if v is None else int(round(v)) for v in chan(col, 0)]

    return {
        "brake":    {"you": chan("game_brake", 3),    "pro": chan("real_brake", 3)},
        "throttle": {"you": chan("game_throttle", 3), "pro": chan("real_throttle", 3)},
        "gear":     {"you": gear("game_gear"),        "pro": gear("real_gear")},
        "speed":    {"you": chan("game_speed_kmh", 1), "pro": chan("real_speed_kmh", 1)},
    }


def _auto_explanation(ci) -> str:
    """Plain-language one-liner describing where the time goes in a corner."""
    bits = []
    if ci.apex_speed_diff is not None and ci.apex_speed_diff < -3:
        bits.append(f"{abs(ci.apex_speed_diff):.0f} km/h slower at the apex")
    if ci.brake_diff_m is not None and abs(ci.brake_diff_m) > 5:
        bits.append("braking %.0fm %s" % (
            abs(ci.brake_diff_m), "earlier" if ci.brake_diff_m < 0 else "later"))
    if ci.exit_speed_diff is not None and ci.exit_speed_diff < -3:
        bits.append(f"{abs(ci.exit_speed_diff):.0f} km/h slower onto the next straight")
    if not bits:
        return "Closely matched here — only small differences through the corner."
    return "You are " + ", ".join(bits) + "."


def _build_lap_chart(aligned, k: int = 110) -> dict:
    """Whole-lap you-vs-pro speed traces + cumulative time delta."""
    return {
        "you": _downsample(aligned["game_speed_kmh"].values, k, 1),
        "pro": _downsample(aligned["real_speed_kmh"].values, k, 1),
        "delta": (_downsample(aligned["time_delta"].values, k, 3)
                  if "time_delta" in aligned.columns else []),
    }


def _corner_markers(aligned, corners) -> "list[dict]":
    """Normalised lap positions (0..1) of each corner apex for the whole-lap chart.

    ``x`` is the fraction *along the sample sequence* (not raw metres) so a marker
    lands on the same horizontal position as the index-downsampled speed/delta
    traces produced by :func:`_build_lap_chart`.  Returns ``[]`` when lap distance
    is unavailable so the chart simply renders without markers.
    """
    if "lap_distance" not in aligned.columns:
        return []
    dist = np.asarray(aligned["lap_distance"].values, dtype=float)
    n = dist.size
    if n < 2 or not np.isfinite(dist).any():
        return []
    markers: "list[dict]" = []
    for i, c in enumerate(corners):
        cid = int(c.get("id", i + 1))
        apex = c.get("apex_m")
        if apex is None:                      # fall back to entry/exit midpoint
            entry, exit_m = c.get("entry_m"), c.get("exit_m")
            if entry is not None and exit_m is not None:
                apex = (float(entry) + float(exit_m)) / 2.0
        if apex is None or not np.isfinite(float(apex)):
            continue
        try:
            idx = int(np.nanargmin(np.abs(dist - float(apex))))
        except (ValueError, TypeError):
            continue
        markers.append({
            "corner_id": cid,
            "short": c.get("short", f"T{cid}"),
            "x": round(idx / (n - 1), 4),
        })
    return markers


def _sample_channel(d_target, d_full, values, digits) -> "list":
    """Interpolate a telemetry channel onto the explorer's sampled distances.

    NaN-safe and order-safe: drops non-finite pairs and sorts by distance so
    ``np.interp`` (which requires increasing x) behaves on any lap. Returns a
    list of rounded floats the same length as ``d_target`` (or all-None when the
    channel has too few valid points).
    """
    target = np.asarray(d_target, dtype=float)
    xp = np.asarray(d_full, dtype=float)
    fp = np.asarray(values, dtype=float)
    good = np.isfinite(xp) & np.isfinite(fp)
    if good.sum() < 2:
        return [None] * target.size
    xp, fp = xp[good], fp[good]
    order = np.argsort(xp)
    xp, fp = xp[order], fp[order]
    out = np.interp(target, xp, fp)
    return [round(float(v), digits) for v in out]


# --- Track Explorer path diagnostics ---------------------------------------
# Maintenance-only knob: set the env var F1_EXPLORER_DEBUG to a truthy value
# (1/true/yes/on) to log, on stderr, how each lap's polyline is classified
# (closed loop vs open path) plus the seam metrics behind that decision and the
# final point count. This is purely for future debugging — it is never surfaced
# in the /analyze payload or the UI.
_EXPLORER_DEBUG = os.environ.get("F1_EXPLORER_DEBUG", "").strip().lower() in (
    "1", "true", "yes", "on")


def _explorer_debug(msg: str) -> None:
    """Emit a Track Explorer path diagnostic when F1_EXPLORER_DEBUG is set."""
    if _EXPLORER_DEBUG:
        print(f"[track-explorer] {msg}", file=sys.stderr)


def _smooth_loop_xy(x, y, window: int = 12):
    """Endpoint-safe moving-average smoothing for the explorer's track polyline.

    ``src``'s ``_smooth_xy`` smooths with ``np.convolve(mode="same")``, which
    implicitly zero-pads the ends and so drags the first/last ~window/2 points
    toward the origin — the visible hook/spike right at the lap start-finish
    seam. This local copy (``app/`` only; ``src`` is never touched) pads the
    signal *before* convolving: wrap-around when the lap is a closed loop, so the
    start/finish junction stays continuous, otherwise edge-replicate so open
    paths don't curl inward. Returns ``(xs, ys, closed)``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n == 0:
        return x, y, False
    span = max(float(np.ptp(x)), float(np.ptp(y))) or 1.0
    closed = bool(np.hypot(x[0] - x[-1], y[0] - y[-1]) < 0.05 * span)
    if n < window * 2:
        return x, y, closed                  # too short to smooth meaningfully
    mode = "wrap" if closed else "edge"
    pad_l, pad_r = (window - 1) // 2, window // 2
    kernel = np.ones(window) / window
    xs = np.convolve(np.pad(x, (pad_l, pad_r), mode=mode), kernel, mode="valid")
    ys = np.convolve(np.pad(y, (pad_l, pad_r), mode=mode), kernel, mode="valid")
    return xs, ys, closed


def _build_track_explorer(aligned, corners, n_points: int = 300,
                          pad: float = 40.0) -> Optional[dict]:
    """Geometry + your-lap telemetry for the interactive Track Explorer.

    Builds an aspect-ratio-preserving SVG polyline of the circuit from the same
    GPS pipeline the static map used (``_extract_track_xy`` -> ``_smooth_xy``,
    both imported read-only from ``src``), downsampled to ``n_points``. Each
    polyline point carries the driver's own speed / throttle / brake / gear,
    interpolated onto that point's lap distance, so a single hovered index can
    drive both the marker on the map and every cursor in the chart below.

    Returns ``None`` when GPS isn't available (the UI then shows a fallback).
    """
    x, y, dist = _extract_track_xy(aligned)
    if x is None or len(x) < 10:
        return None
    # Raw start/end gap (pre-smoothing) — this is what drives the closed/open
    # decision in _smooth_loop_xy; kept here only for the debug log below.
    raw_gap = float(np.hypot(x[0] - x[-1], y[0] - y[-1]))
    # Endpoint-safe smoothing (app-local) so the start-finish seam stays clean;
    # ``closed`` says whether the lap is a loop the SVG should close itself.
    x, y, closed = _smooth_loop_xy(x, y, window=12)
    dist = np.asarray(dist, dtype=float)
    n = x.size
    if n < 2:
        return None

    # Bounding box -> uniform scale (longest side = 1000) so the SVG never
    # distorts. Y is flipped to SVG's top-left origin (so north stays up).
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    span = max(xmax - xmin, ymax - ymin) or 1.0
    scale = 1000.0 / span

    # Evenly spaced sample indices along the lap-ordered polyline.
    k = min(int(n_points), n)
    sel = np.unique(np.linspace(0, n - 1, k).round().astype(int))
    xs, ys, ds = x[sel], y[sel], dist[sel]

    # Drop consecutive near-duplicate points so the polyline is geometrically
    # clean, then drop a trailing point that coincides with the first. A closed
    # lap is closed by the SVG's own Z, so a coincident vertex would be a doubled
    # segment / seam spike; an OPEN path must likewise never duplicate its start.
    # Either way the coincident endpoint is removed here, so the path never
    # carries a duplicate closing point. Points and telemetry stay index-aligned.
    eps = 1e-3 * span
    keep = [0]
    for i in range(1, xs.size):
        if np.hypot(xs[i] - xs[keep[-1]], ys[i] - ys[keep[-1]]) > eps:
            keep.append(i)
    if (len(keep) > 3 and
            np.hypot(xs[keep[-1]] - xs[keep[0]], ys[keep[-1]] - ys[keep[0]]) <= eps):
        keep.pop()
    if len(keep) < 2:
        return None
    keep = np.asarray(keep, dtype=int)
    xs, ys, ds = xs[keep], ys[keep], ds[keep]

    def to_vb(px, py):
        return [round((px - xmin) * scale + pad, 1),
                round((ymax - py) * scale + pad, 1)]

    points = [to_vb(px, py) for px, py in zip(xs, ys)]
    vb_w = round((xmax - xmin) * scale + 2 * pad, 1)
    vb_h = round((ymax - ymin) * scale + 2 * pad, 1)

    # --- Path sanity (maintenance) ---------------------------------------
    # Invariant enforced by the dedup above: an OPEN path must not begin and end
    # at (nearly) the same point, and neither path carries a duplicate closing
    # vertex. A CLOSED lap intentionally ends near the start — the SVG adds the
    # closing Z itself (a small seam is expected there), so the check is scoped
    # to open paths. We only log a regression; we never alter the geometry here.
    seam_vb = (float(np.hypot(points[0][0] - points[-1][0],
                              points[0][1] - points[-1][1]))
               if len(points) >= 2 else 0.0)
    if not closed and seam_vb <= 0.5:
        _explorer_debug(
            f"WARNING: open path endpoints nearly identical (seam={seam_vb:.3f} "
            f"vb-units) — unexpected after dedup")
    _explorer_debug(
        f"lap classified {'CLOSED' if closed else 'OPEN'}: raw_gap={raw_gap:.2f} "
        f"span={span:.1f} thresh={0.05 * span:.1f} | points={len(points)} "
        f"seam_vb={seam_vb:.2f} closing={'Z' if closed else 'none'}")

    d_full = aligned["lap_distance"].values
    cols = aligned.columns

    def chan(name, digits):
        return (_sample_channel(ds, d_full, aligned[name].values, digits)
                if name in cols else [])

    gear_raw = chan("game_gear", 0)
    telemetry = {
        "dist": [round(float(v), 1) for v in ds],
        "speed": chan("game_speed_kmh", 1),
        "throttle": chan("game_throttle", 3),
        "brake": chan("game_brake", 3),
        "gear": [None if v is None else int(round(v)) for v in gear_raw],
    }

    markers = []
    for i, c in enumerate(corners or []):
        cid = int(c.get("id", i + 1))
        cx, cy = _find_corner_xy(c, x, y, dist)
        vx, vy = to_vb(cx, cy)
        markers.append({
            "corner_id": cid,
            "short": c.get("short", f"T{cid}"),
            "name": c.get("name", ""),
            "x": vx,
            "y": vy,
            "dist": _num(c.get("apex_m", c.get("apex_dist")), 1),
        })

    return {
        "track_path": {"viewbox_w": vb_w, "viewbox_h": vb_h,
                       "closed": closed, "points": points},
        "telemetry": telemetry,
        "corner_markers": markers,
    }


def _run_comparison(csv_path: str, mode: str, driver: str, year: int,
                    session: str, track: str, lap: Optional[int]) -> dict:
    """
    Game-vs-real pipeline (scripts 03, 04 & 05).

    The pipeline runs once, but each mode returns a *distinct* payload so the
    three views are genuinely different:

    * ``comparison`` -> you-vs-pro timing + per-corner delta cards + track map.
    * ``coaching``   -> grade, consistency, braking/throttle tendencies, an
      action plan and prioritised corner fixes (no map).
    * ``track_map``  -> the interactive Track Explorer: an SVG circuit polyline
      plus your-lap telemetry sampled along it, so hovering the map drives the
      linked telemetry chart (no static PNG, no coaching prose).
    """
    args = SimpleNamespace(
        csv=csv_path,
        lap=lap,                 # None -> auto fastest; int -> chosen lapIndex
        driver=driver or "VER",
        year=int(year),
        session=session or "Q",
        track=track,
        gp=None,
        no_corners=False,
    )

    try:
        aligned, corners, report, game_meta, real_meta = run_pipeline(args)
    except SystemExit as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not complete analysis. Check that the driver code, "
                "year, session and track are valid and that real F1 data "
                f"exists for them. ({exc})"
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:  # bad CSV, missing columns, FastF1 issues, ...
        traceback.print_exc()
        raise HTTPException(status_code=422, detail=f"Analysis failed: {exc}")

    ordered_ci = sorted(report.corner_insights, key=lambda c: c.corner_id)

    # Identity + headline timing — shared, lightweight, in every comparison view.
    payload = {
        "mode": mode,
        "driver": report.driver,
        "year": report.year,
        "session": report.session,
        "gp_name": report.gp_name,
        "track": track,
        "game_time": _num(report.game_time, 3),
        "real_time": _num(report.real_time, 3),
        "overall_delta": _num(report.overall_delta, 3),
        "overall_grade": report.overall_grade,
    }

    # ---- Lap Comparison: visual, you-vs-pro, no heatmap (that's Track Map). ----
    if mode == "comparison":
        payload["corner_insights"] = [_corner_to_dict(ci) for ci in ordered_ci]

        # Headline gained/lost split for the comparison summary strip.
        deltas = [ci.time_delta for ci in ordered_ci if ci.time_delta is not None]
        payload["corners_lost"] = sum(1 for d in deltas if d > 0.02)
        payload["corners_gained"] = sum(1 for d in deltas if d < -0.02)
        best = min(ordered_ci, key=lambda c: (c.time_delta if c.time_delta is not None else 0), default=None)
        worst = max(ordered_ci, key=lambda c: (c.time_delta if c.time_delta is not None else 0), default=None)
        payload["best_corner"] = (
            {"short": best.short, "time_delta": _num(best.time_delta, 3)}
            if best is not None else None)
        payload["worst_corner"] = (
            {"short": worst.short, "time_delta": _num(worst.time_delta, 3)}
            if worst is not None else None)

        # Whole-lap comparison view (bottom section) + corner markers so the
        # reader can connect each part of the lap to the corner cards above.
        payload["lap_chart"] = _build_lap_chart(aligned)
        markers = _corner_markers(aligned, corners)
        sev_by_id = {int(ci.corner_id): _severity(ci) for ci in ordered_ci}
        for m in markers:
            m["severity"] = sev_by_id.get(m["corner_id"], "ok")
        payload["corner_markers"] = markers

        # Per-corner comparison charts for EVERY corner, in track order
        # (ordered_ci is already sorted by corner_id). No time/priority filter.
        corner_by_id = {
            int(c.get("id", i + 1)): c for i, c in enumerate(corners)
        }
        key_corners = []
        for ci in ordered_ci:
            c = corner_by_id.get(int(ci.corner_id))
            if not c:
                continue
            key_corners.append({
                "corner_id": int(ci.corner_id),
                "short": ci.short,
                "name": ci.name,
                "grade": ci.grade,
                "severity": _severity(ci),
                "time_delta": _num(ci.time_delta, 3),
                "brake_diff_m": _num(ci.brake_diff_m, 1),
                "apex_speed_diff": _num(ci.apex_speed_diff, 1),
                "exit_speed_diff": _num(ci.exit_speed_diff, 1),
                "explanation": (ci.issues[0] if ci.issues else _auto_explanation(ci)),
                "tip": ci.tips[0] if ci.tips else None,
                "charts": _corner_charts(aligned, c),
            })
        payload["key_corners"] = key_corners
        return payload

    # ---- Coaching Report: actionable summaries + prioritised fixes (no map). --
    if mode == "coaching":
        payload["consistency_score"] = _num(report.consistency_score, 0)
        payload["braking_summary"] = report.braking_summary or ""
        payload["throttle_summary"] = report.throttle_summary or ""

        fixes = []
        for ci in sorted(report.corner_insights, key=lambda c: -(c.priority or 0)):
            if (ci.priority or 0) < 0.3:
                continue
            fixes.append({
                "corner_id": int(ci.corner_id),
                "short": ci.short,
                "name": ci.name,
                "grade": ci.grade,
                "severity": _severity(ci),
                "time_delta": _num(ci.time_delta, 3),
                "issue": ci.issues[0] if ci.issues else None,
                "tip": ci.tips[0] if ci.tips else None,
            })
            if len(fixes) >= 5:
                break
        payload["priority_fixes"] = fixes

        try:
            payload["action_plan"] = list(_generate_action_plan(report) or [])
        except Exception:
            traceback.print_exc()
            payload["action_plan"] = []
        try:
            potential = float(_estimate_potential(report))
        except Exception:
            traceback.print_exc()
            potential = None
        payload["potential_gain"] = _num(potential, 1)
        payload["target_time"] = (
            _num(report.game_time - potential, 3)
            if (potential is not None and report.game_time) else None)
        return payload

    # ---- Interactive Track Explorer: your-lap telemetry mapped onto the
    #      circuit. Replaces the static PNG track map (script 05). ----
    payload["track_explorer"] = _build_track_explorer(aligned, corners)
    payload["corner_severities"] = [
        {
            "corner_id": int(ci.corner_id),
            "short": ci.short,
            "name": ci.name,
            "grade": ci.grade,
            "severity": _severity(ci),
            "time_delta": _num(ci.time_delta, 3),
        }
        for ci in ordered_ci
    ]
    return payload


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/tracks")
def tracks() -> list:
    """Supported tracks for the frontend dropdown (single source of truth)."""
    return SUPPORTED_TRACKS


@app.post("/laps")
def laps(
    file: UploadFile = File(..., description="Sim Racing Telemetry CSV export"),
) -> JSONResponse:
    """
    List the laps contained in an uploaded telemetry CSV.

    Used by the UI to populate the "Lap to Analyze" dropdown right after upload,
    without committing to a full (slow) analysis run.
    """
    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")

    suffix = Path(file.filename or "lap.csv").suffix or ".csv"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(raw)
        tmp.close()
        try:
            has_lap_index, options = _read_lap_options(tmp.name)
        except HTTPException:
            raise
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(
                status_code=422,
                detail=f"Could not read laps from this CSV: {exc}",
            )
        detected_track, detected_track_name = _detect_track_from_csv(tmp.name)
    finally:
        if os.path.exists(tmp.name):
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    return JSONResponse({
        "has_lap_index": has_lap_index,
        "laps": options,
        "detected_track": detected_track,
        "detected_track_name": detected_track_name,
    })


@app.post("/analyze")
def analyze(
    file: UploadFile = File(..., description="Sim Racing Telemetry CSV export"),
    driver: str = Form("VER"),
    year: int = Form(DEFAULT_YEAR),
    session: str = Form("Q"),
    track: str = Form(...),
    lap_index: Optional[str] = Form(None),
    mode: str = Form(DEFAULT_MODE),
) -> JSONResponse:
    """
    Run the analysis program chosen by ``mode`` on an uploaded lap.

    Solo modes (``overview``, ``lap_analysis``) analyse the uploaded lap on
    its own; comparison modes (``comparison``, ``coaching``, ``track_map``)
    run the full game-vs-real pipeline.

    Defined as a *sync* function on purpose: the heavy modes are CPU- and
    network-bound (FastF1 download + matplotlib), so FastAPI runs them in a
    worker thread instead of blocking the event loop.
    """
    driver = (driver or "").strip().upper()
    track = (track or "").strip().lower()
    mode = (mode or DEFAULT_MODE).strip().lower()
    if mode not in ALL_MODES:
        mode = DEFAULT_MODE
    if not track:
        raise HTTPException(status_code=422, detail="A track must be selected.")

    # None -> auto-select fastest lap (unchanged behaviour); int -> that lapIndex.
    lap = _parse_lap_index(lap_index)

    # Persist the upload to a temp file — the pipeline reads from a path.
    suffix = Path(file.filename or "lap.csv").suffix or ".csv"
    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(raw)
        tmp.close()

        if mode in SOLO_MODES:
            payload = _run_solo(tmp.name, mode, track, lap)
        else:
            payload = _run_comparison(
                tmp.name, mode, driver, year, session, track, lap
            )
    finally:
        if os.path.exists(tmp.name):
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    return JSONResponse(payload)


# --- Static assets + SPA entrypoint (mounted last so /analyze etc. win). ---
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    index_html = STATIC_DIR / "index.html"
    if not index_html.exists():
        raise HTTPException(status_code=404, detail="Frontend not built.")
    return FileResponse(str(index_html))


if __name__ == "__main__":
    import uvicorn

    # Bind to $HOST / $PORT when provided. Hosted platforms (e.g. Render) set
    # $PORT and require binding to 0.0.0.0; locally we default to 127.0.0.1:8000.
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port, reload=False)
