"""Stage 8 test: a real-data regression test (needs GPU/checkpoints/the
SMPL-X model file, skipped otherwise -- see conftest.py).
"""

from __future__ import annotations

import numpy as np
import torch


def test_object_pose_output_is_plausible(stage_8_result):
    data = torch.load(stage_8_result["object_pose"], weights_only=False)

    assert set(data.keys()) == {"translation", "rotation", "is_low_confidence"}
    translation = data["translation"]
    rotation = data["rotation"]
    is_low_confidence = data["is_low_confidence"]

    n_frames = translation.shape[0]
    assert translation.shape == (n_frames, 3)
    assert rotation.shape == (n_frames, 3, 3)
    assert is_low_confidence.shape == (n_frames,)

    assert torch.isfinite(translation).all()
    assert torch.isfinite(rotation).all()

    # Every rotation should be a proper (orthonormal, det=+1) rotation matrix.
    identity = torch.eye(3, dtype=rotation.dtype).expand(n_frames, 3, 3)
    assert torch.allclose(rotation @ rotation.transpose(-1, -2), identity, atol=1e-3)
    assert torch.allclose(torch.linalg.det(rotation), torch.ones(n_frames, dtype=rotation.dtype), atol=1e-2)

    # The test clip's own racket-holding grip spans (close to) the whole clip
    # (already confirmed by stage 7's own real-data test: one continuous,
    # full-confidence contact event) -- most frames should be a real
    # attachment, not a fallback/never-tracked flag.
    assert is_low_confidence.float().mean() < 0.5


def test_object_pose_npz_matches_the_pt_file(stage_8_result):
    pt_data = torch.load(stage_8_result["object_pose"], weights_only=False)
    with np.load(stage_8_result["object_pose_npz"]) as npz_data:
        assert np.allclose(npz_data["translation"], pt_data["translation"].numpy())
        assert np.allclose(npz_data["rotation"], pt_data["rotation"].numpy())
        assert np.array_equal(npz_data["is_low_confidence"], pt_data["is_low_confidence"].numpy())
