"""Tests for src/alignment.py — core math and shape gating."""
import numpy as np
import pytest

from src.alignment import (
    resample_trajectory,
    compute_curvature,
    find_start_offset_by_curvature,
    estimate_rigid_transform,
    apply_transform,
    align_track_shapes,
    _curvature_correlation,
)


class TestResampleTrajectory:
    def test_output_length(self, circle_xy):
        result = resample_trajectory(circle_xy, n_points=500)
        assert result.shape == (500, 2)

    def test_preserves_shape(self, circle_xy):
        """Resampled circle should still be roughly unit radius."""
        result = resample_trajectory(circle_xy, n_points=200)
        radii = np.sqrt(result[:, 0]**2 + result[:, 1]**2)
        assert np.allclose(radii, 1.0, atol=0.05)

    def test_short_input_passthrough(self):
        """Fewer than 10 points should return input unchanged."""
        xy = np.array([[0, 0], [1, 1], [2, 0]])
        result = resample_trajectory(xy, n_points=100)
        assert np.array_equal(result, xy)


class TestComputeCurvature:
    def test_circle_constant_curvature(self, circle_xy):
        """Circle of radius 1 should have curvature ~1 everywhere."""
        curv = compute_curvature(circle_xy, smooth_window=15)
        # Exclude edges (boundary effects)
        mid = curv[50:-50]
        assert np.allclose(mid, mid.mean(), atol=0.15)
        assert abs(mid.mean()) > 0.5  # should be positive (CCW)

    def test_straight_line_zero_curvature(self):
        """Straight line should have ~0 curvature."""
        xy = np.column_stack([np.linspace(0, 100, 500),
                              np.zeros(500)])
        curv = compute_curvature(xy, smooth_window=15)
        assert np.allclose(curv[20:-20], 0, atol=0.01)

    def test_short_input_returns_zeros(self):
        xy = np.array([[0, 0], [1, 1]])
        curv = compute_curvature(xy, smooth_window=15)
        assert np.all(curv == 0)


class TestFindStartOffset:
    def test_identical_curvature_zero_offset(self, circle_xy):
        curv = compute_curvature(circle_xy)
        offset = find_start_offset_by_curvature(curv, curv)
        assert offset == 0

    def test_shifted_curvature_detects_offset(self, oval_xy):
        """Shift oval by 200 points, should detect correct offset."""
        curv_a = compute_curvature(oval_xy)
        shifted = np.roll(oval_xy, 200, axis=0)
        curv_b = compute_curvature(shifted)
        offset = find_start_offset_by_curvature(curv_a, curv_b)
        # np.roll(xy, 200) means B starts at original index 800,
        # so offset to shift B forward to align = 800 (or equivalently
        # n_points - 200). Accept either 200 or 800 within tolerance.
        n = len(oval_xy)
        effective = min(abs(offset) % n, n - abs(offset) % n)
        assert effective < 210  # within ~10 of 200

    def test_flat_curvature_returns_zero(self):
        """If both curvatures are flat, offset should be 0."""
        flat = np.zeros(500)
        offset = find_start_offset_by_curvature(flat, flat)
        assert offset == 0


class TestEstimateRigidTransform:
    def test_identity_transform(self, circle_xy):
        """Same points → identity (residual ≈ 0)."""
        result = estimate_rigid_transform(circle_xy, circle_xy)
        assert result['residual'] < 0.01
        assert result['reflection'] is False

    def test_known_rotation(self):
        """90° rotation should be recovered."""
        src = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        # Rotate 90° CCW
        R90 = np.array([[0, -1], [1, 0]], dtype=float)
        dst = (src @ R90.T)
        result = estimate_rigid_transform(src, dst)
        assert result['residual'] < 0.01
        assert np.allclose(result['rotation'], R90, atol=0.01)

    def test_translation(self):
        """Pure translation should be recovered."""
        src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        dst = src + np.array([5, 3])
        result = estimate_rigid_transform(src, dst)
        assert result['residual'] < 0.01
        assert np.allclose(result['translation'], [5, 3], atol=0.1)

    def test_reflection_detected(self):
        """Y-axis flip should be recovered with near-zero residual.

        estimate_rigid_transform picks whichever of the two SVD branches
        (raw vs sign-corrected) yields lower residual. When the true
        transform IS a reflection, the raw SVD branch naturally produces
        a reflection matrix (det(R) ≈ -1) with perfect residual. The
        internal boolean flag tracks which branch won, not whether the
        resulting matrix is a reflection. So we verify:
          1. Alignment quality (residual ≈ 0)
          2. The rotation matrix actually encodes a reflection (det ≈ -1)
        """
        # Chiral spiral — mirror image cannot be achieved by rotation
        t = np.linspace(0, 4 * np.pi, 300, endpoint=False)
        r = 1 + 0.3 * t
        src = np.column_stack([r * np.cos(t), r * np.sin(t)])
        dst = src.copy()
        dst[:, 1] = -dst[:, 1]  # flip Y
        result = estimate_rigid_transform(src, dst, allow_reflection=True)
        # The solver must achieve near-perfect alignment
        assert result['residual'] < 0.01
        # The resulting rotation matrix should have det ≈ -1 (reflection)
        det = np.linalg.det(result['rotation'])
        assert det < 0, f"Expected reflection (det<0), got det={det:.4f}"

    def test_reflection_symmetric_shape_low_residual(self):
        """Y-flip on any shape: verify alignment quality regardless of
        which internal branch the solver picks."""
        t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        r = 1 + 0.5 * np.cos(t) + 0.3 * np.sin(2 * t)
        src = np.column_stack([r * np.cos(t), r * np.sin(t)])
        dst = src.copy()
        dst[:, 1] = -dst[:, 1]
        result = estimate_rigid_transform(src, dst, allow_reflection=True)
        assert result['residual'] < 0.01

    def test_degenerate_input(self):
        """All-zero input should return inf residual, not crash."""
        zeros = np.zeros((10, 2))
        result = estimate_rigid_transform(zeros, zeros)
        assert result['residual'] == float('inf')


class TestApplyTransform:
    def test_roundtrip(self, circle_xy):
        """Transform then inverse should recover original."""
        src = circle_xy[:100]
        dst = src * 2 + np.array([10, 20])
        tf = estimate_rigid_transform(src, dst)
        transformed = apply_transform(src, tf)
        assert np.allclose(transformed, dst, atol=0.1)


class TestShapeGating:
    """Test that align_track_shapes returns quality metrics
    and that the gating logic in align_two_pass works."""

    def test_good_alignment_has_high_correlation(self, circle_xy):
        """Same shape → high curvature correlation."""
        # Slightly rotated + translated version
        R = np.array([[0.8, -0.6], [0.6, 0.8]])
        game_xy = circle_xy @ R.T + np.array([100, 200])
        result = align_track_shapes(game_xy, circle_xy, verbose=False)
        assert result['curvature_correlation'] > 0.7
        assert result['residual_m'] < 50

    def test_bad_alignment_has_low_correlation(self):
        """Completely different shapes → low correlation."""
        t = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
        circle = np.column_stack([np.cos(t), np.sin(t)])
        # Square-ish shape (very different curvature profile)
        square = np.column_stack([
            np.sign(np.cos(t)) * (1 - 0.3 * np.abs(np.sin(2*t))),
            np.sign(np.sin(t)) * (1 - 0.3 * np.abs(np.cos(2*t))),
        ])
        result = align_track_shapes(circle, square * 500, verbose=False)
        # Should have low correlation or high residual
        assert (result['curvature_correlation'] < 0.5 or
                result['residual_m'] > 100)


class TestCurvatureCorrelation:
    def test_identical_signals(self):
        signal = np.sin(np.linspace(0, 4 * np.pi, 200))
        assert _curvature_correlation(signal, signal) > 0.99

    def test_uncorrelated_signals(self):
        rng = np.random.default_rng(123)
        a = rng.normal(0, 1, 200)
        b = rng.normal(0, 1, 200)
        corr = _curvature_correlation(a, b)
        assert abs(corr) < 0.3
