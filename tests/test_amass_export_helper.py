"""Unit tests for `amass_export_helper`: pure numpy, no GPU needed, always runs."""

from __future__ import annotations

import numpy as np

from pipeline.helpers.amass_export_helper import HAND_POSE_DIM, JAW_EYES_DIM, write_amass_npz

N_FRAMES = 4


def _body_arrays():
    return dict(
        global_orient=np.random.default_rng(0).normal(size=(N_FRAMES, 3)),
        body_pose=np.random.default_rng(1).normal(size=(N_FRAMES, 63)),
        betas=np.random.default_rng(2).normal(size=(10,)),
        transl=np.random.default_rng(3).normal(size=(N_FRAMES, 3)),
    )


def test_write_amass_npz_has_the_expected_keys_and_shapes(tmp_path):
    out_path = tmp_path / "preview.npz"
    write_amass_npz(**_body_arrays(), fps=29.97, out_path=out_path)

    with np.load(out_path) as data:
        for key in ("trans", "gender", "mocap_frame_rate", "betas", "poses"):
            assert key in data
        assert data["trans"].shape == (N_FRAMES, 3)
        assert data["poses"].shape == (N_FRAMES, 55 * 3)
        assert str(data["gender"]) == "neutral"
        assert int(data["mocap_frame_rate"]) == 30


def test_write_amass_npz_defaults_hand_pose_to_zero(tmp_path):
    out_path = tmp_path / "preview.npz"
    write_amass_npz(**_body_arrays(), fps=30.0, out_path=out_path)

    with np.load(out_path) as data:
        poses = data["poses"]
        hand_start = 3 + 63 + JAW_EYES_DIM
        hands = poses[:, hand_start:hand_start + 2 * HAND_POSE_DIM]
        assert np.all(hands == 0.0)


def test_write_amass_npz_embeds_real_hand_pose_when_given(tmp_path):
    out_path = tmp_path / "export.npz"
    left_hand_pose = np.full((N_FRAMES, HAND_POSE_DIM), 0.5)
    right_hand_pose = np.full((N_FRAMES, HAND_POSE_DIM), -0.5)

    write_amass_npz(
        **_body_arrays(), fps=30.0, out_path=out_path,
        left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose,
    )

    with np.load(out_path) as data:
        poses = data["poses"]
        hand_start = 3 + 63 + JAW_EYES_DIM
        left = poses[:, hand_start:hand_start + HAND_POSE_DIM]
        right = poses[:, hand_start + HAND_POSE_DIM:hand_start + 2 * HAND_POSE_DIM]
        assert np.allclose(left, 0.5)
        assert np.allclose(right, -0.5)


def test_write_amass_npz_defaults_jaw_and_eyes_to_zero(tmp_path):
    out_path = tmp_path / "export.npz"
    write_amass_npz(
        **_body_arrays(), fps=30.0, out_path=out_path,
        left_hand_pose=np.ones((N_FRAMES, HAND_POSE_DIM)),
        right_hand_pose=np.ones((N_FRAMES, HAND_POSE_DIM)),
    )

    with np.load(out_path) as data:
        jaw_eyes_start = 3 + 63
        jaw_eyes = data["poses"][:, jaw_eyes_start:jaw_eyes_start + JAW_EYES_DIM]
        assert np.all(jaw_eyes == 0.0)
