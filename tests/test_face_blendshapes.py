"""Unit tests for `face_blendshapes.py`'s FLAME-expression -> SMPL-X
`Exp###` map. Needs the real FLAME and SMPL-X model files plus their
vertex correspondence (`body_models/`, gitignored, see README's Setup
section), skipped without them.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.algorithms.face.face_blendshapes import (
    FACE_BASES_PATH, FLAME_EXPRESSION_DIM, FLAME_MODEL_PATH, FLAME_VERTEX_IDS_PATH, JAW_AXIS1_BOUND_RAD,
    JAW_OPEN_REFERENCE_RAD, MIN_CALIBRATION_SCALE, SMPLX_EXPRESSION_DIM, SMPLX_MODEL_PATH,
    _calibrate_group_s_channel, _flame_to_smplx_expression_matrix, _load_face_bases, direct_arkit_jaw_channels,
    flame_to_smplx_expression, solve_arkit_weights,
)

FACE_MODEL_ASSETS_PRESENT = (
    FLAME_MODEL_PATH.exists() and SMPLX_MODEL_PATH.exists() and FLAME_VERTEX_IDS_PATH.exists()
)
pytestmark = pytest.mark.skipif(
    not FACE_MODEL_ASSETS_PRESENT, reason="needs the FLAME/SMPL-X model files (see README's Setup section)"
)


def test_matrix_has_expected_shape_and_is_finite():
    matrix = _flame_to_smplx_expression_matrix()
    assert matrix.shape == (SMPLX_EXPRESSION_DIM, FLAME_EXPRESSION_DIM)
    assert np.isfinite(matrix).all()


def test_zero_expression_maps_to_zero():
    flame_expression = np.zeros((3, FLAME_EXPRESSION_DIM), dtype=np.float32)
    smplx_expression = flame_to_smplx_expression(flame_expression)
    assert np.allclose(smplx_expression, 0.0)


def test_output_shape_matches_frame_count():
    n = 7
    flame_expression = np.random.default_rng(0).normal(size=(n, FLAME_EXPRESSION_DIM)).astype(np.float32)
    smplx_expression = flame_to_smplx_expression(flame_expression)
    assert smplx_expression.shape == (n, SMPLX_EXPRESSION_DIM)


def test_mapping_is_linear():
    # flame_to_smplx_expression(a) + flame_to_smplx_expression(b) == flame_to_smplx_expression(a + b),
    # since it's a fixed matrix multiply, catches an accidental nonlinearity (e.g. a stray clamp) creeping in.
    rng = np.random.default_rng(1)
    a = rng.normal(size=(4, FLAME_EXPRESSION_DIM)).astype(np.float32)
    b = rng.normal(size=(4, FLAME_EXPRESSION_DIM)).astype(np.float32)
    assert np.allclose(
        flame_to_smplx_expression(a) + flame_to_smplx_expression(b),
        flame_to_smplx_expression(a + b),
        atol=1e-4,
    )


def test_matrix_is_cached_across_calls():
    assert _flame_to_smplx_expression_matrix() is _flame_to_smplx_expression_matrix()


# --- Group D: direct jaw channels ------------------------------------------

def test_direct_arkit_jaw_channels_pure_open():
    jaw_pose = np.array([[JAW_OPEN_REFERENCE_RAD * 0.5, 0.0, 0.0]], dtype=np.float32)
    out = direct_arkit_jaw_channels(jaw_pose)
    assert out["JawOpen"][0] == pytest.approx(0.5, abs=1e-5)
    assert out["JawLeft"][0] == 0.0
    assert out["JawRight"][0] == 0.0


def test_direct_arkit_jaw_channels_left_right_never_simultaneously_nonzero():
    jaw_pose = np.array([
        [0.0, JAW_AXIS1_BOUND_RAD * 0.5, 0.0],
        [0.0, -JAW_AXIS1_BOUND_RAD * 0.5, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)
    out = direct_arkit_jaw_channels(jaw_pose)
    assert out["JawLeft"][0] == pytest.approx(0.5, abs=1e-5) and out["JawRight"][0] == 0.0
    assert out["JawRight"][1] == pytest.approx(0.5, abs=1e-5) and out["JawLeft"][1] == 0.0
    assert out["JawLeft"][2] == 0.0 and out["JawRight"][2] == 0.0


def test_direct_arkit_jaw_channels_clips_to_unit_range():
    jaw_pose = np.array([[JAW_OPEN_REFERENCE_RAD * 5, JAW_AXIS1_BOUND_RAD * 5, 0.0]], dtype=np.float32)
    out = direct_arkit_jaw_channels(jaw_pose)
    assert out["JawOpen"][0] == 1.0
    assert out["JawLeft"][0] == 1.0


# --- Group S: the ARKit weight solve ---------------------------------------

FACE_BASES_PRESENT = FACE_BASES_PATH.exists()
face_bases_skip = pytest.mark.skipif(not FACE_BASES_PRESENT, reason="needs body_models/arkit/face_bases.npz")


@face_bases_skip
def test_solve_arkit_weights_recovers_a_known_shape():
    # valid=all-False deliberately disables the per-channel percentile
    # calibration (see _calibrate_group_s_channel's own empty-sample guard)
    #, with only 1 frame, a self-normalizing percentile would trivially
    # rescale any nonzero raw solve to 1.0, defeating this test's own
    # premise (checking the raw LSQ solve's own accuracy against a known
    # target, not the calibration layer).
    D, jaw_jacobian, names = _load_face_bases()
    k = names.index("NoseSneerLeft")
    target_weight = 0.6
    landmark_delta = (target_weight * D[:, k]).reshape(1, 51, 3)
    jaw_pose = np.zeros((1, 3), dtype=np.float32)

    out = solve_arkit_weights(
        landmark_delta, jaw_pose, temporal_lambda=0.0, ridge=1e-6, valid=np.zeros(1, dtype=bool),
    )
    assert out["NoseSneerLeft"][0] == pytest.approx(target_weight, abs=0.05)


@face_bases_skip
def test_solve_arkit_weights_pure_jaw_open_does_not_leak_into_mouth_channels():
    # A real double-counting regression class: a pure jaw-open landmark
    # delta (no real Group-S signal) must not force mouth channels to
    # compensate for jaw's own subtracted contribution. valid=all-False
    # disables calibration for the same reason as the test above, a
    # single-frame self-normalizing percentile would turn any nonzero leak,
    # however tiny, into a false 1.0.
    from pipeline.algorithms.face.face_landmark_fit import local_landmark_delta

    jaw_pose = np.array([[0.3, 0.0, 0.0]], dtype=np.float32)
    expression = np.zeros((1, 50), dtype=np.float32)
    landmark_delta = local_landmark_delta(jaw_pose, expression, device=__import__("torch").device("cpu"))

    out = solve_arkit_weights(
        landmark_delta, jaw_pose, temporal_lambda=0.0, ridge=0.01, valid=np.zeros(1, dtype=bool),
    )
    for name, values in out.items():
        assert values[0] < 0.3, f"{name} leaked jaw-open signal: {values[0]}"


def test_calibrate_group_s_channel_stretches_weak_signal_to_unit_range():
    weights = np.array([0.0, 0.02, 0.04, 0.05, 0.02, 0.0])  # p95 well above MIN_CALIBRATION_SCALE
    valid = np.ones(len(weights), dtype=bool)
    calibrated = _calibrate_group_s_channel(weights, valid)
    assert calibrated.max() == pytest.approx(1.0, abs=0.05)


def test_calibrate_group_s_channel_leaves_near_zero_channel_untouched():
    # Below MIN_CALIBRATION_SCALE, essentially noise, dividing by it would
    # amplify float noise into a false full-scale reading (see that
    # constant's own comment).
    weights = np.full(4, MIN_CALIBRATION_SCALE * 0.5)
    valid = np.ones(len(weights), dtype=bool)
    calibrated = _calibrate_group_s_channel(weights, valid)
    assert np.array_equal(calibrated, weights)


def test_calibrate_group_s_channel_only_considers_valid_frames():
    # A held/frozen frame at an old high value must not itself set the
    # calibration scale.
    weights = np.array([0.05, 0.05, 0.9, 0.9])  # last two are a held run, not real
    valid = np.array([True, True, False, False])
    calibrated = _calibrate_group_s_channel(weights, valid)
    assert calibrated[0] == pytest.approx(1.0, abs=0.05)


def test_calibrate_group_s_channel_empty_valid_returns_raw():
    weights = np.array([0.3, 0.4, 0.5])
    valid = np.zeros(3, dtype=bool)
    calibrated = _calibrate_group_s_channel(weights, valid)
    assert np.array_equal(calibrated, weights)


@face_bases_skip
def test_solve_arkit_weights_output_bounded_and_finite():
    D, jaw_jacobian, names = _load_face_bases()
    rng = np.random.default_rng(0)
    landmark_delta = (rng.normal(size=(5,)) [:, None, None] * D[:, 0].reshape(1, 51, 3)).astype(np.float32)
    jaw_pose = np.zeros((5, 3), dtype=np.float32)

    out = solve_arkit_weights(landmark_delta, jaw_pose)
    for values in out.values():
        assert np.isfinite(values).all()
        assert (values >= 0.0).all() and (values <= 1.0).all()
