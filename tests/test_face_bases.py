"""Validates `body_models/arkit/face_bases.npz` (built by
`scripts/build_face_bases.py`), skipped when the artifact is absent. These
are the tests that catch a botched ICT-FaceKit registration: a first attempt
at a full-mesh version of this basis had a ~20cm mean surface error that
these exact properties (mirror symmetry, finite values) would have caught
immediately, well before it ever reached a real clip.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.build_face_bases import GROUP_S_NAMES, NUM_FLAME_LANDMARKS, OUTPUT_PATH

pytestmark = pytest.mark.skipif(not OUTPUT_PATH.exists(), reason="needs body_models/arkit/face_bases.npz (see README's Setup section)")

# Left/right column pairs that should be near-mirror-images of each other in
# magnitude (not direction, a real scanned mesh isn't perfectly symmetric,
# but norms should match closely; a registration bug breaks this hard).
# Only the 6 channels GROUP_S_NAMES still covers, see that constant's own
# comment (face_blendshapes.py's docstring has the full story: most of
# Group S moved to MediaPipe's own native blendshapes).
_MIRROR_PAIRS = [
    ("EyeSquintLeft", "EyeSquintRight"),
    ("CheekSquintLeft", "CheekSquintRight"),
    ("NoseSneerLeft", "NoseSneerRight"),
]


@pytest.fixture(scope="module")
def bases():
    return np.load(OUTPUT_PATH)


def test_d_matrix_shape_and_finite(bases):
    D = bases["D"]
    assert D.shape == (NUM_FLAME_LANDMARKS * 3, len(GROUP_S_NAMES))
    assert np.isfinite(D).all()


def test_jaw_jacobian_shape_and_finite(bases):
    jac = bases["jaw_landmark_jacobian"]
    assert jac.shape == (NUM_FLAME_LANDMARKS * 3, 3)
    assert np.isfinite(jac).all()


def test_group_s_names_match_module_order(bases):
    assert list(bases["group_s_names"]) == GROUP_S_NAMES


def test_no_column_is_degenerate_zero(bases):
    # Every Group-S shape has *some* real ICT-FaceKit source, a zero column
    # would mean the landmark extraction silently found no displacement.
    norms = np.linalg.norm(bases["D"], axis=0)
    assert (norms > 1e-6).all()


@pytest.mark.parametrize("left,right", _MIRROR_PAIRS)
def test_left_right_column_norms_are_mirrored(bases, left, right):
    names = list(bases["group_s_names"])
    norm_left = np.linalg.norm(bases["D"][:, names.index(left)])
    norm_right = np.linalg.norm(bases["D"][:, names.index(right)])
    assert norm_left == pytest.approx(norm_right, rel=0.05)
