"""Unit tests for `face_pose_stabilization.py`. Pure numpy, no FLAME/torch
needed, these test the Kabsch orientation fit's own correctness, not its
integration into `fit_clip` (see test_face_landmark_fit.py's
`test_fit_clip_kabsch_anchor_recovers_occluded_orientation_spike` for that)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from pipeline.algorithms.face.face_pose_stabilization import (
    CHIN_CENTER, INNER_LIP_PAIRS, LEFT_BROW_LOWER, LEFT_EYE_RING, MOUTH_CORNER_LEFT, MOUTH_CORNER_RIGHT,
    NOSE_BRIDGE, RIGHT_BROW_LOWER, RIGHT_EYE_RING, RIGID_INDICES, stabilize_orientation,
)


def _synthetic_face_template(rng: np.random.Generator) -> np.ndarray:
    """A plausible-enough (478, 3) neutral-face point cloud: eyes/brows/nose/
    chin/lips at roughly correct relative positions, a mouth CLOSED (upper
    and lower inner-lip indices coincide) baseline, and every remaining
    RIGID_INDICES entry scattered in 3D so the Kabsch SVD fit isn't
    degenerate. Everything else stays zero (unused by this module)."""
    lm = np.zeros((478, 3), dtype=np.float64)
    for idx in RIGHT_EYE_RING:
        lm[idx] = [-30, 0, 0] + rng.normal(0, 1, size=3)
    for idx in LEFT_EYE_RING:
        lm[idx] = [30, 0, 0] + rng.normal(0, 1, size=3)
    lm[NOSE_BRIDGE] = [0, 10, 5]
    lm[CHIN_CENTER] = [0, -60, 0]
    for idx in LEFT_BROW_LOWER:
        lm[idx] = [20, 20, 0] + rng.normal(0, 3, size=3)
    for idx in RIGHT_BROW_LOWER:
        lm[idx] = [-20, 20, 0] + rng.normal(0, 3, size=3)
    lm[MOUTH_CORNER_LEFT] = [15, -30, 5]
    lm[MOUTH_CORNER_RIGHT] = [-15, -30, 5]
    for u, l in INNER_LIP_PAIRS:
        lm[u] = [rng.normal(0, 5), -30, 5]
        lm[l] = lm[u].copy()  # closed mouth baseline
    for idx in RIGID_INDICES:
        if not lm[idx].any():
            lm[idx] = rng.normal(0, 30, size=3)
    return lm


def _open_jaw(landmarks: np.ndarray, amount: float) -> np.ndarray:
    """Shifts only the lower inner-lip points down, simulates a mouth
    opening without touching anything in RIGID_INDICES."""
    out = landmarks.copy()
    for _u, l in INNER_LIP_PAIRS:
        out[l, 1] -= amount
    return out


def test_stabilize_orientation_recovers_relative_rotation_despite_jaw_motion():
    """`stabilize_orientation` returns each frame's rotation relative to an
    internal per-clip canonical frame, not world space directly, so
    R_out[i] == C @ R_true[i].T for some single, frame-independent C (the
    canonical frame's own unknown orientation), not R_out[i] == R_true[i]
    itself. Rearranged, R_out[i] @ R_true[i] should equal that same C on
    every frame, this is the actual invariant to check, and exactly the
    property calibrate_rotation_offset relies on (a single fixed offset
    reconciling this signal with FLAME's convention, valid for every frame)."""
    rng = np.random.default_rng(0)
    template = _synthetic_face_template(rng)
    n = 8
    r_true = Rotation.from_euler("xyz", rng.uniform(-45, 45, size=(n, 3)), degrees=True).as_matrix()
    # Every other frame has the jaw wide open, if the rotation fit leaked
    # any chin/lip geometry, its estimated C would drift from the
    # closed-mouth frames' C.
    jaw_open = np.arange(n) % 2 == 0

    landmarks = np.zeros((n, 478, 3), dtype=np.float64)
    for i in range(n):
        local = _open_jaw(template, amount=25.0) if jaw_open[i] else template
        translation = rng.normal(0, 50, size=3)  # similarity_fit must be translation-invariant
        landmarks[i] = (r_true[i] @ local.T).T + translation

    detected = np.ones(n, dtype=bool)
    r_out = stabilize_orientation(landmarks, detected)

    assert not np.isnan(r_out).any()
    c_est = r_out @ r_true
    for i in range(1, n):
        angle_error = Rotation.from_matrix(c_est[i] @ c_est[0].T).magnitude()
        assert np.degrees(angle_error) < 1.0, f"frame {i} (jaw_open={jaw_open[i]}): {np.degrees(angle_error):.2f} deg"


def test_stabilize_orientation_undetected_frames_are_nan():
    rng = np.random.default_rng(1)
    template = _synthetic_face_template(rng)
    n = 6
    detected = np.array([True, True, False, True, False, True])
    landmarks = np.tile(template, (n, 1, 1))

    r_out = stabilize_orientation(landmarks, detected)

    assert np.isnan(r_out[~detected]).all()
    assert not np.isnan(r_out[detected]).any()


def test_stabilize_orientation_no_detected_frames_returns_all_nan():
    n = 4
    landmarks = np.zeros((n, 478, 3), dtype=np.float64)
    detected = np.zeros(n, dtype=bool)

    r_out = stabilize_orientation(landmarks, detected)

    assert r_out.shape == (n, 3, 3)
    assert np.isnan(r_out).all()
