"""Unit tests for the temporal smoothing (pure numpy/scipy, no GPU/checkpoints).

Strategy: build a slow, smooth ground-truth signal, add high-frequency jitter,
smooth it, and assert the jitter shrinks while the underlying motion is
preserved (no bias, endpoints tracked, rotations stay valid). Also exercises the
hands-specific validity handling.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pipeline.algorithms.motion_smoothing import (
    smooth_rotation_sequence,
    smooth_translation_sequence,
)


def _jitter_energy(seq: np.ndarray) -> float:
    """Mean magnitude of frame-to-frame second differences -- a proxy for how
    jittery a sequence is (a smooth signal has near-zero curvature noise)."""
    return float(np.abs(np.diff(seq, n=2, axis=0)).mean())


def test_rotation_smoothing_reduces_jitter_and_preserves_motion():
    rng = np.random.default_rng(0)
    n = 60
    t = np.linspace(0, 1, n)
    # Smooth ground-truth: one joint sweeping about a fixed axis.
    clean = np.stack([1.2 * np.sin(2 * np.pi * t), np.zeros(n), np.zeros(n)], axis=1)
    noisy = clean + rng.normal(0, 0.05, clean.shape)

    smoothed = smooth_rotation_sequence(noisy, window=11)

    assert _jitter_energy(smoothed) < 0.3 * _jitter_energy(noisy)
    # Underlying motion preserved: closer to clean than the noisy input was.
    assert np.abs(smoothed - clean).mean() < np.abs(noisy - clean).mean()


def test_rotation_smoothing_returns_unit_rotations():
    rng = np.random.default_rng(1)
    noisy = rng.normal(0, 0.5, (40, 15, 3))  # 15 joints, like a MANO hand
    smoothed = smooth_rotation_sequence(noisy, window=9)
    assert smoothed.shape == noisy.shape
    mats = Rotation.from_rotvec(smoothed.reshape(-1, 3)).as_matrix()
    dets = np.linalg.det(mats)
    assert np.allclose(dets, 1.0, atol=1e-5)  # proper rotations, no scaling crept in


def test_validity_gap_does_not_pull_toward_placeholder():
    rng = np.random.default_rng(2)
    n = 50
    t = np.linspace(0, 1, n)
    clean = np.stack([0.8 * np.sin(2 * np.pi * t) + 1.0, np.zeros(n), np.zeros(n)], axis=1)
    seq = clean + rng.normal(0, 0.03, clean.shape)
    # Simulate an undetected-hand gap: those frames carry the zero placeholder.
    valid = np.ones(n, bool)
    valid[20:28] = False
    seq[~valid] = 0.0

    aware = smooth_rotation_sequence(seq, window=11, valid=valid)
    naive = smooth_rotation_sequence(seq, window=11)  # treats the zeros as real

    # On the valid frames bordering the gap, validity-aware smoothing stays near
    # the true motion instead of being dragged toward the zero placeholder.
    border = [18, 19, 28, 29]
    assert np.abs(aware[border] - clean[border]).mean() < np.abs(naive[border] - clean[border]).mean()


def test_interior_gap_interpolates_between_endpoints():
    """A hand that's occluded then reappears: the gap is bounded by a valid
    frame on both sides, so it should be filled by interpolating between them,
    not frozen at either one."""
    n = 40
    rest = np.zeros((n, 3))
    # A constant, easily-distinguished rotation before and after the gap.
    rest[:15] = [1.0, 0.0, 0.0]
    rest[25:] = [-1.0, 0.0, 0.0]
    valid = np.ones(n, bool)
    valid[15:25] = False  # interior occlusion, frames 15-24, recovers at 25

    smoothed = smooth_rotation_sequence(rest, window=5, valid=valid)

    mid = smoothed[20]  # well inside the gap
    # Interpolated, not frozen at either endpoint: strictly between the two
    # known values (with a comfortable margin, since savgol softens the corner).
    assert -0.9 < mid[0] < 0.9
    assert mid[0] != pytest.approx(1.0, abs=0.05)
    assert mid[0] != pytest.approx(-1.0, abs=0.05)


def test_trailing_gap_freezes_at_last_known_pose():
    """A hand that's occluded and never comes back (occlusion runs to the end
    of the clip): there's no second endpoint to interpolate toward, so it
    should hold the last real value instead of drifting or zeroing out."""
    n = 40
    rest = np.zeros((n, 3))
    rest[:20] = [1.0, 0.5, 0.0]
    valid = np.ones(n, bool)
    valid[20:] = False  # trailing occlusion, never recovers

    smoothed = smooth_rotation_sequence(rest, window=5, valid=valid)

    # Well past the last real frame, deep in the frozen tail.
    assert np.allclose(smoothed[35], [1.0, 0.5, 0.0], atol=0.05)


def test_leading_gap_freezes_at_first_known_pose():
    """Symmetric case: the hand isn't detected until partway through the clip
    (never seen before that). No 'before' endpoint exists either, so the fill
    should hold the first real value backward, not snap from zero."""
    n = 40
    rest = np.zeros((n, 3))
    rest[15:] = [0.0, -1.0, 0.3]
    valid = np.ones(n, bool)
    valid[:15] = False  # leading occlusion, not yet detected

    smoothed = smooth_rotation_sequence(rest, window=5, valid=valid)

    assert np.allclose(smoothed[2], [0.0, -1.0, 0.3], atol=0.05)


def test_translation_smoothing_reduces_jitter_and_keeps_mean():
    rng = np.random.default_rng(3)
    n = 80
    t = np.linspace(0, 1, n)
    clean = np.stack([t, 0.5 * np.sin(2 * np.pi * t), np.zeros(n)], axis=1)
    noisy = clean + rng.normal(0, 0.02, clean.shape)

    smoothed = smooth_translation_sequence(noisy, cutoff=0.15)

    assert _jitter_energy(smoothed) < 0.3 * _jitter_energy(noisy)
    # Zero-phase filter: no net drift/bias introduced.
    assert np.allclose(smoothed.mean(axis=0), noisy.mean(axis=0), atol=0.02)


def test_short_sequence_is_returned_unchanged():
    seq = np.random.default_rng(4).normal(0, 0.5, (2, 3))
    assert np.array_equal(smooth_rotation_sequence(seq, window=9), seq)
    assert np.array_equal(smooth_translation_sequence(seq, cutoff=0.15), seq)
