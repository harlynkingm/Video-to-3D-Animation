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
    decimate_rotation_sequence,
    decimate_translation_sequence,
    one_euro_filter_rotation_sequence,
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


def test_decimate_removes_jitter_outright():
    """Keyframe reduction fits a smooth curve through sparse knots, so residual
    jitter is gone by construction -- when the tolerance sits comfortably above
    the noise floor (as it does in the pipeline, where decimation runs after the
    one-euro pass has already knocked the residual down), the wiggle frames are
    dropped and the fit is far smoother than the input. (If the noise amplitude
    instead rivals the tolerance, RDP correctly keeps those frames as knots and
    smooths little -- that's the 'knots on noise' failure the one-euro pre-pass
    exists to prevent.)"""
    rng = np.random.default_rng(10)
    n = 200
    t = np.linspace(0, 1, n)
    clean = np.stack([1.0 * np.sin(2 * np.pi * t), np.zeros(n), np.zeros(n)], axis=1)
    noisy = clean + rng.normal(0, 0.008, clean.shape)  # ~0.5deg, well under the tolerance below

    decimated = decimate_rotation_sequence(noisy, tolerance_deg=3.0)

    assert _jitter_energy(decimated) < 0.15 * _jitter_energy(noisy)
    # The underlying slow motion is preserved, not flattened toward the mean: the
    # fit still swings out to the sine's peaks.
    assert decimated[:, 0].max() > 0.9
    assert decimated[:, 0].min() < -0.9


def test_decimate_stays_within_tolerance():
    """The fitted curve never departs the original by more than the tolerance
    (this is the guarantee the tolerance knob makes)."""
    rng = np.random.default_rng(11)
    n = 120
    t = np.linspace(0, 1, n)
    seq = np.stack([1.5 * t, 0.4 * np.sin(4 * np.pi * t), np.zeros(n)], axis=1)
    seq = seq + rng.normal(0, 0.02, seq.shape)

    tol_deg = 3.0
    decimated = decimate_rotation_sequence(seq, tolerance_deg=tol_deg)

    # Slerp reconstruction stays provably within the tolerance of the input at
    # every frame (a tiny epsilon covers float round-off). This is the guarantee
    # the tolerance knob makes, and why slerp was chosen over an overshooting spline.
    orig = Rotation.from_rotvec(seq)
    fit = Rotation.from_rotvec(decimated)
    dev_deg = np.degrees((fit * orig.inv()).magnitude())
    assert dev_deg.max() <= tol_deg + 1e-6


def test_decimate_larger_tolerance_is_smoother():
    """More slack -> fewer keyframes -> a flatter, smoother curve."""
    rng = np.random.default_rng(12)
    n = 150
    t = np.linspace(0, 1, n)
    seq = np.stack([np.sin(2 * np.pi * t), np.zeros(n), np.zeros(n)], axis=1)
    seq = seq + rng.normal(0, 0.04, seq.shape)

    loose = decimate_rotation_sequence(seq, tolerance_deg=5.0)
    tight = decimate_rotation_sequence(seq, tolerance_deg=1.0)
    assert _jitter_energy(loose) < _jitter_energy(tight)


def test_decimate_returns_proper_rotations():
    rng = np.random.default_rng(13)
    noisy = rng.normal(0, 0.5, (60, 15, 3))
    decimated = decimate_rotation_sequence(noisy, tolerance_deg=2.0)
    assert decimated.shape == noisy.shape
    mats = Rotation.from_rotvec(decimated.reshape(-1, 3)).as_matrix()
    assert np.allclose(np.linalg.det(mats), 1.0, atol=1e-5)


def test_decimate_constant_input_stays_constant():
    const = np.tile([0.5, 0.1, -0.2], (40, 1))
    decimated = decimate_rotation_sequence(const, tolerance_deg=2.0)
    assert np.allclose(decimated, const, atol=1e-4)


def test_decimate_trailing_gap_freezes():
    """Same occlusion contract as the filters: a trailing gap (never recovers)
    freezes at the last real pose rather than drifting."""
    n = 60
    rest = np.zeros((n, 3))
    rest[:30] = [1.0, 0.5, 0.0]
    valid = np.ones(n, bool)
    valid[30:] = False
    decimated = decimate_rotation_sequence(rest, tolerance_deg=2.0, valid=valid)
    assert np.allclose(decimated[50], [1.0, 0.5, 0.0], atol=0.05)


def test_short_sequence_is_returned_unchanged_by_decimate():
    seq = np.random.default_rng(14).normal(0, 0.5, (2, 3))
    assert np.array_equal(decimate_rotation_sequence(seq, tolerance_deg=2.0), seq)


def test_decimate_zero_valid_frames_returns_unchanged():
    """Regression guard: a hand whose wrist is rejected on every single frame
    (e.g. never once biomechanically plausible) previously crashed here --
    `_fill_invalid` has nothing to interpolate from with zero real anchors.
    Mirrors `smooth_rotation_sequence`/`one_euro_filter_rotation_sequence`'s
    own existing guard for the same case."""
    seq = np.random.default_rng(15).normal(0, 0.5, (20, 3))
    valid = np.zeros(20, dtype=bool)
    assert np.array_equal(decimate_rotation_sequence(seq, tolerance_deg=2.0, valid=valid), seq)


def test_translation_decimate_removes_jitter_outright():
    """Same RDP core as rotation decimation, Euclidean instead of geodesic --
    when the tolerance sits above the noise floor, jitter should collapse."""
    rng = np.random.default_rng(20)
    n = 200
    t = np.linspace(0, 1, n)
    clean = np.stack([t, 0.5 * np.sin(2 * np.pi * t), np.zeros(n)], axis=1)
    noisy = clean + rng.normal(0, 0.001, clean.shape)  # ~1mm noise

    decimated = decimate_translation_sequence(noisy, tolerance_m=0.01)  # 10mm tolerance

    assert _jitter_energy(decimated) < 0.15 * _jitter_energy(noisy)


def test_translation_decimate_stays_within_tolerance():
    """The fitted curve never departs the original by more than the tolerance --
    guaranteed by construction, since reconstruction (`_fill_invalid`, linear)
    matches the fit RDP selection used to check deviation."""
    rng = np.random.default_rng(21)
    n = 120
    t = np.linspace(0, 1, n)
    seq = np.stack([1.5 * t, 0.4 * np.sin(4 * np.pi * t), np.zeros(n)], axis=1)
    seq = seq + rng.normal(0, 0.002, seq.shape)

    tol_m = 0.01
    decimated = decimate_translation_sequence(seq, tolerance_m=tol_m)

    dev = np.linalg.norm(decimated - seq, axis=1)
    assert dev.max() <= tol_m + 1e-9


def test_translation_decimate_preserves_large_real_motion():
    """A genuinely large displacement must survive decimation near-intact, not
    get flattened the way jitter does -- decimation should distinguish real
    motion from noise by magnitude relative to tolerance, not erase both."""
    n = 60
    seq = np.zeros((n, 3))
    seq[:, 0] = np.linspace(0, 0.5, n)  # 500mm real displacement along X

    decimated = decimate_translation_sequence(seq, tolerance_m=0.01)

    assert decimated[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert decimated[-1, 0] == pytest.approx(0.5, abs=1e-6)


def test_translation_decimate_constant_input_stays_constant():
    const = np.tile([0.2, -0.1, 0.05], (40, 1))
    decimated = decimate_translation_sequence(const, tolerance_m=0.01)
    assert np.allclose(decimated, const, atol=1e-6)


def test_translation_decimate_larger_tolerance_is_smoother():
    rng = np.random.default_rng(22)
    n = 150
    t = np.linspace(0, 1, n)
    seq = np.stack([np.sin(2 * np.pi * t), np.zeros(n), np.zeros(n)], axis=1)
    seq = seq + rng.normal(0, 0.01, seq.shape)

    loose = decimate_translation_sequence(seq, tolerance_m=0.05)
    tight = decimate_translation_sequence(seq, tolerance_m=0.005)
    assert _jitter_energy(loose) < _jitter_energy(tight)


def test_short_sequence_is_returned_unchanged_by_translation_decimate():
    seq = np.random.default_rng(23).normal(0, 0.1, (2, 3))
    assert np.array_equal(decimate_translation_sequence(seq, tolerance_m=0.01), seq)


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


def test_one_euro_collapses_small_residual_wobble():
    """A joint circling in place with small amplitude (the axis-precession
    artifact this replaced) should be smoothed down close to a constant, not
    literally held (that discrete hold-then-snap is exactly what looked like
    stop-motion and got replaced) -- just continuously, substantially damped."""
    n = 60
    base = np.array([0.9, 0.0, 0.0])
    t = np.linspace(0, 4 * np.pi, n)
    wobble = 0.02 * np.stack([np.zeros(n), np.cos(t), np.sin(t)], axis=1)
    seq = base + wobble

    filtered = one_euro_filter_rotation_sequence(seq, fps=30.0, min_cutoff_hz=0.3, beta=0.3)

    assert _jitter_energy(filtered) < 0.3 * _jitter_energy(seq)
    # Continuous, not discretely held: consecutive frames keep changing by a
    # little, never bit-for-bit identical (that would be a hold, not a filter).
    frame_diff = np.linalg.norm(np.diff(filtered, axis=0), axis=1)
    assert (frame_diff < 1e-9).mean() == 0.0


def test_one_euro_speed_estimate_cancels_symmetric_noise():
    """Regression guard for a real bug: an earlier version estimated 'current
    speed' by low-pass filtering the already-rectified (always >= 0) geodesic
    distance between consecutive frames, which can never average toward zero
    -- rectified symmetric noise has a nonzero mean no matter how heavily it's
    smoothed. That let noise alone convince the filter a still joint was
    moving, loosening it and letting the noise leak through as persistent
    low-amplitude wobble (reported on real thumb data as "shaky, like a person
    with shaky hands"). The fix filters the *signed* angular-velocity vector
    before taking its norm, so symmetric noise properly cancels on averaging.
    A static true signal plus realistic-scale symmetric noise should be
    smoothed much harder than the old (buggy) approach could manage."""
    rng = np.random.default_rng(42)
    n = 200
    base = np.array([0.9, 0.1, -0.05])
    seq = base + rng.normal(0, 0.1, (n, 3))  # static true rotation, noisy per-frame estimate

    filtered = one_euro_filter_rotation_sequence(seq, fps=30.0, min_cutoff_hz=0.3, beta=0.3)

    # The old buggy approach only reached ~75-80% reduction at this noise
    # scale; the fixed approach clears 90%+.
    assert _jitter_energy(filtered) < 0.1 * _jitter_energy(seq)


def test_one_euro_tracks_real_motion_with_low_lag():
    """A genuinely large, sustained rotation change must be tracked closely,
    not smeared out or left lagging far behind."""
    n = 60
    seq = np.zeros((n, 3))
    seq[:20] = [0.1, 0.0, 0.0]
    seq[20:] = [1.5, 0.0, 0.0]  # a large, real, sustained change

    filtered = one_euro_filter_rotation_sequence(seq, fps=30.0, min_cutoff_hz=0.3, beta=0.3)

    assert np.allclose(filtered[0], [0.1, 0.0, 0.0], atol=1e-6)
    assert np.allclose(filtered[-1], [1.5, 0.0, 0.0], atol=0.05)  # fully caught up by the end
    # Low lag: well before the end of the sustained motion, it should already
    # be most of the way there, not still trailing behind.
    assert filtered[35][0] > 1.3


def test_one_euro_returns_proper_rotations():
    rng = np.random.default_rng(6)
    noisy = rng.normal(0, 0.5, (40, 15, 3))
    filtered = one_euro_filter_rotation_sequence(noisy, fps=30.0, min_cutoff_hz=0.3, beta=0.3)
    assert filtered.shape == noisy.shape
    mats = Rotation.from_rotvec(filtered.reshape(-1, 3)).as_matrix()
    assert np.allclose(np.linalg.det(mats), 1.0, atol=1e-5)


def test_one_euro_constant_input_stays_constant():
    """No artificial drift from the filter itself when there's no real motion."""
    const = np.tile([0.5, 0.1, -0.2], (40, 1))
    filtered = one_euro_filter_rotation_sequence(const, fps=30.0, min_cutoff_hz=0.3, beta=0.3)
    assert np.allclose(filtered, const, atol=1e-6)


def test_one_euro_occlusion_gap_uses_same_fill_contract():
    """Same interpolate-vs-freeze contract as smooth_rotation_sequence: an
    interior gap interpolates, a trailing gap freezes."""
    n = 40
    rest = np.zeros((n, 3))
    rest[:15] = [1.0, 0.0, 0.0]
    rest[25:] = [-1.0, 0.0, 0.0]
    valid = np.ones(n, bool)
    valid[15:25] = False

    filtered = one_euro_filter_rotation_sequence(rest, fps=30.0, min_cutoff_hz=0.3, beta=0.3, valid=valid)
    mid = filtered[20]
    assert -0.9 < mid[0] < 0.9  # interpolated, not frozen at either endpoint
