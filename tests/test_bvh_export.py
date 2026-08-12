"""Unit tests for `write_bvh` itself (pure string/numpy logic, no GPU/checkpoints
or SMPL-X model file needed), in particular the root-position channel's two
modes: a static zero when `root_translation` is omitted (stage 4's hands-only
preview, which has no real root to show), and real per-frame values when it's
given (stage 5's retarget preview). Higher-level tests
(test_stage_4_estimate_hands.py, test_stage_5_retarget_hands.py) exercise both
call paths end to end but only check structure; these check the actual
position values written for each mode. Also covers `root_camera_to_upright`,
shared by every consumer that reorients GVHMR's own incam root into this same
upright frame (stage 10's export, stage 2's `--render-motion-preview`).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from pipeline.helpers.bvh_export import (
    CAMERA_TO_BVH_ROOT_ROTATION,
    camera_to_upright_rotation_matrix,
    camera_to_upright_translation,
    root_camera_to_upright,
    write_bvh,
)

# A trivial two-joint skeleton (root + one child), enough to exercise the
# MOTION block without needing a real body/hand hierarchy.
NAMES = ["Root", "Child"]
PARENTS = [-1, 0]
OFFSETS = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _motion_value_lines(text: str) -> list[str]:
    """The per-frame value lines, skipping the Frames/Frame Time header lines."""
    return text.split("MOTION\n")[1].splitlines()[2:]


def test_root_position_is_static_zero_when_translation_omitted(tmp_path):
    n = 4
    rotations = np.tile(np.eye(3), (n, len(NAMES), 1, 1))
    out = tmp_path / "test_zero.bvh"

    write_bvh(out, NAMES, PARENTS, OFFSETS, rotations, fps=30.0)
    lines = _motion_value_lines(out.read_text())

    assert len(lines) == n
    for line in lines:
        px, py, pz = (float(v) for v in line.split()[:3])
        assert (px, py, pz) == (0.0, 0.0, 0.0)


def test_root_position_carries_real_translation_when_given(tmp_path):
    n = 3
    rotations = np.tile(np.eye(3), (n, len(NAMES), 1, 1))
    root_translation = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [-4.0, 5.0, -6.0]])
    out = tmp_path / "test_real.bvh"

    write_bvh(out, NAMES, PARENTS, OFFSETS, rotations, fps=30.0, root_translation=root_translation)
    lines = _motion_value_lines(out.read_text())

    assert len(lines) == n
    for frame, expected in enumerate(root_translation):
        px, py, pz = (float(v) for v in lines[frame].split()[:3])
        assert np.allclose([px, py, pz], expected, atol=1e-5)


def test_camera_to_bvh_root_rotation_is_a_proper_rotation():
    """Sanity check on the shared constant both translation and rotation use:
    determinant +1 (a real rotation, not a mirror) and orthonormal."""
    m = CAMERA_TO_BVH_ROOT_ROTATION
    assert np.isclose(np.linalg.det(m), 1.0)
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-10)


def test_root_camera_to_upright_maps_camera_up_to_target_plus_y():
    """Camera space is Y-down (a point "up" from the origin has negative Y);
    the target frame is Y-up, matching the already-proven BVH convention
    this reuses, so a purely-vertical camera-space offset should land on
    positive Y, not Z or X, in the corrected frame."""
    global_orient = np.zeros((1, 3), dtype=np.float32)
    transl = np.array([[0.0, -1.0, 0.0]], dtype=np.float32)  # 1 unit "up" in camera space

    _, corrected_transl = root_camera_to_upright(global_orient, transl)

    assert np.allclose(corrected_transl, [[0.0, 1.0, 0.0]], atol=1e-5)


def test_root_camera_to_upright_round_trips_identity_orientation():
    """A zero rotation, once converted matrix->corrected->back to axis-angle,
    should land on the fixed correction's own rotation, not drift or explode
    (a real risk with axis-angle round trips near singular points)."""
    global_orient = np.zeros((1, 3), dtype=np.float32)
    transl = np.zeros((1, 3), dtype=np.float32)

    corrected_orient, corrected_transl = root_camera_to_upright(global_orient, transl)

    expected = Rotation.from_matrix(CAMERA_TO_BVH_ROOT_ROTATION).as_rotvec()
    assert np.allclose(corrected_orient[0], expected, atol=1e-5)
    assert np.allclose(corrected_transl, 0.0)


def test_camera_to_upright_rotation_matrix_left_multiplies_the_constant():
    rng = np.random.default_rng(0)
    m = Rotation.from_rotvec(rng.normal(size=3)).as_matrix()

    result = camera_to_upright_rotation_matrix(m)

    assert np.allclose(result, CAMERA_TO_BVH_ROOT_ROTATION @ m)


def test_camera_to_upright_rotation_matrix_broadcasts_over_frames():
    """The BVH preview passes a whole (F, 3, 3) per-frame batch at once, not
    one matrix at a time, confirms that shape works, not just a single
    (3, 3) matrix."""
    rng = np.random.default_rng(1)
    batch = Rotation.from_rotvec(rng.normal(size=(5, 3))).as_matrix()

    result = camera_to_upright_rotation_matrix(batch)

    assert result.shape == (5, 3, 3)
    for i in range(5):
        assert np.allclose(result[i], CAMERA_TO_BVH_ROOT_ROTATION @ batch[i])


def test_camera_to_upright_translation_applies_the_constant_transpose():
    transl = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])

    result = camera_to_upright_translation(transl)

    assert np.allclose(result, transl @ CAMERA_TO_BVH_ROOT_ROTATION.T)


def test_root_camera_to_upright_is_built_from_the_shared_primitives():
    """Regression guard against the two implementations drifting apart again,
    `root_camera_to_upright`'s own result must match composing the two
    matrix-level primitives directly, not just look similar."""
    rng = np.random.default_rng(2)
    global_orient = rng.normal(size=(4, 3)).astype(np.float32)
    transl = rng.normal(size=(4, 3)).astype(np.float32)

    orient, transl_out = root_camera_to_upright(global_orient, transl)

    expected_matrix = camera_to_upright_rotation_matrix(Rotation.from_rotvec(global_orient).as_matrix())
    expected_orient = Rotation.from_matrix(expected_matrix).as_rotvec()
    assert np.allclose(orient, expected_orient, atol=1e-5)
    assert np.allclose(transl_out, camera_to_upright_translation(transl), atol=1e-5)
