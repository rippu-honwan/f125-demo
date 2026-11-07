"""Tests for the /best_driver fallback to a cached reference lap (app/main.py)."""
from app.main import _servable_driver


def test_pole_sitter_with_a_cached_lap_is_kept():
    assert _servable_driver(2025, "suzuka", "VER") == "VER"


def test_pole_sitter_without_a_cached_lap_falls_back_to_the_fastest_cached_lap():
    # 2025 Bahrain pole was PIA; cached are NOR 90.267s and VER 90.423s.
    assert _servable_driver(2025, "bahrain", "PIA") == "NOR"
    assert _servable_driver(2025, "bahrain", None) == "NOR"


def test_nothing_cached_leaves_the_driver_alone():
    assert _servable_driver(2019, "bahrain", "HAM") == "HAM"
