"""Unit tests for `write_bvh` itself (pure string/numpy logic, no GPU/checkpoints
or SMPL-X model file needed) -- in particular the root-position channel's two
modes: a static zero when `root_translation` is omitted (stage 4's hands-only
preview, which has no real root to show), and real per-frame values when it's
given (stage 5's retarget preview). Higher-level tests
(test_stage_4_estimate_hands.py, test_stage_5_retarget_hands.py) exercise both
call paths end to end but only check structure; these check the actual
position values written for each mode.
"""

from __future__ import annotations

import numpy as np

from pipeline.helpers.bvh_export import CAMERA_TO_BVH_ROOT_ROTATION, write_bvh

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
