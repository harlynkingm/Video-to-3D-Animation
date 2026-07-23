"""Stage 2 regression test: runs real GVHMR on stage 1's tracked human mask
and checks the resulting SMPL-X body pose looks correct -- right shapes, no
NaN/Inf, physically plausible joint rotations and translation, and (since
`--dump-motion-preview` is on for these tests) a structurally valid AMASS
preview file. Needs the real GVHMR checkpoints and a CUDA GPU -- skipped
automatically otherwise (see conftest.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pipeline.adapters.gvhmr.gvhmr_adapter import (
    KEY_BETAS,
    KEY_BODY_POSE,
    KEY_GLOBAL_ORIENT,
    KEY_PRED_SMPL_PARAMS_GLOBAL,
    KEY_PRED_SMPL_PARAMS_INCAM,
    KEY_TRANSL,
)
from conftest import TEST_VIDEO_FRAME_COUNT

MAX_PLAUSIBLE_JOINT_ROTATION_RAD = 3.15  # any single axis-angle rotation maxes out at pi radians
MAX_PLAUSIBLE_FRAME_TO_FRAME_TRANSL_M = 1.0  # generous bound for a real person's root motion, per frame


def _load_motion(stage_2_result: dict[str, str]) -> dict:
    return torch.load(Path(stage_2_result["human_motion"]), weights_only=False)


def test_motion_output_has_correct_shapes_and_no_nan(stage_2_result):
    result = _load_motion(stage_2_result)

    for key in (KEY_PRED_SMPL_PARAMS_INCAM, KEY_PRED_SMPL_PARAMS_GLOBAL):
        params = result[key]
        assert params[KEY_BODY_POSE].shape == (TEST_VIDEO_FRAME_COUNT, 63)
        assert params[KEY_BETAS].shape == (TEST_VIDEO_FRAME_COUNT, 10)
        assert params[KEY_GLOBAL_ORIENT].shape == (TEST_VIDEO_FRAME_COUNT, 3)
        assert params[KEY_TRANSL].shape == (TEST_VIDEO_FRAME_COUNT, 3)

        for tensor in params.values():
            assert not torch.isnan(tensor).any()
            assert not torch.isinf(tensor).any()


def test_body_pose_rotations_are_physically_plausible(stage_2_result):
    result = _load_motion(stage_2_result)
    body_pose = result[KEY_PRED_SMPL_PARAMS_GLOBAL][KEY_BODY_POSE].reshape(TEST_VIDEO_FRAME_COUNT, 21, 3)
    magnitudes = body_pose.norm(dim=-1)
    assert magnitudes.max().item() < MAX_PLAUSIBLE_JOINT_ROTATION_RAD


def test_global_translation_does_not_explode_or_teleport(stage_2_result):
    result = _load_motion(stage_2_result)
    transl = result[KEY_PRED_SMPL_PARAMS_GLOBAL][KEY_TRANSL]
    frame_to_frame = (transl[1:] - transl[:-1]).norm(dim=-1)
    assert frame_to_frame.max().item() < MAX_PLAUSIBLE_FRAME_TO_FRAME_TRANSL_M


def test_motion_preview_npz_is_a_valid_amass_file(stage_2_result):
    with np.load(stage_2_result["motion_preview"]) as data:
        for key in ("trans", "gender", "mocap_frame_rate", "betas", "poses"):
            assert key in data
        assert data["trans"].shape == (TEST_VIDEO_FRAME_COUNT, 3)
        assert data["poses"].shape == (TEST_VIDEO_FRAME_COUNT, 165)
        assert str(data["gender"]) == "neutral"
        assert not np.isnan(data["poses"]).any()
