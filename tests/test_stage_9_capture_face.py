"""Stage 9 (capture_face) tests. The disabled-path test is pure and always
runs; other tests are gated on whatever real model files/assets they touch
(FLAME, SMPL-X, `body_models/arkit/face_bases.npz`), and
`test_run_produces_face_params_and_motion_files` also needs a CUDA GPU plus
the DECA/MICA/MediaPipe/ViTPose checkpoints (see conftest.py's
`stage_9_result` fixture).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import FLAME_MODEL_PATH, SMPLX_MODEL_PATH, make_run_input
from pipeline.adapters.gvhmr.gvhmr_adapter import (
    KEY_BODY_POSE, KEY_GLOBAL_ORIENT, KEY_PRED_SMPL_PARAMS_INCAM, KEY_ROOT_MOTION_UNRELIABLE,
)
from pipeline.algorithms.face.face_blendshapes import FACE_BASES_PATH, JAW_OPEN_REFERENCE_RAD
from pipeline.algorithms.face.face_landmark_fit import KEY_EXPRESSION, KEY_JAW_POSE, KEY_VALID
from pipeline.create_run import create_run
from pipeline.helpers.livelink_csv import ARKIT_BLENDSHAPE_NAMES, CSV_HEADER, write_livelink_csv
from pipeline.progress_tracker import StageName, StageStatus
from pipeline.stages import stage_2_estimate_human_motion, stage_9_capture_face


def test_run_returns_empty_dict_when_disabled(tmp_path):
    run_input = make_run_input(skip_face_capture=True)
    runRecord = create_run(tmp_path / "run", run_input, run_id="test")

    assert stage_9_capture_face.run(runRecord) == {}


@pytest.mark.skipif(not SMPLX_MODEL_PATH.exists(), reason="needs the SMPL-X model file (see README's Setup section)")
def test_body_head_rotation_extracts_head_joint_and_inverts_confidence(tmp_path):
    """Doesn't need a GPU or any checkpoint, a synthetic stage-2 output is
    enough to check the body-based-orientation-prior wiring itself: the right joint gets pulled
    out of `body_joint_global_rotations`' full per-joint output, and
    `head_confidence` is `root_motion_unreliable` inverted. The rotation math
    itself is already covered by `test_body_joint_global_rotations_matches_
    manual_fk` in test_stage_5_retarget_hands.py."""
    n = 3
    motion = {
        KEY_PRED_SMPL_PARAMS_INCAM: {KEY_GLOBAL_ORIENT: torch.zeros(n, 3), KEY_BODY_POSE: torch.zeros(n, 63)},
        KEY_ROOT_MOTION_UNRELIABLE: np.array([False, True, False]),
    }
    motion_path = tmp_path / "motion.pt"
    torch.save(motion, motion_path)

    runRecord = create_run(tmp_path / "run", make_run_input(), run_id="test")
    runRecord.mark_progress(
        StageName.STAGE_2_ESTIMATE_HUMAN_MOTION, StageStatus.COMPLETE,
        outputs={stage_2_estimate_human_motion.OUTPUT_HUMAN_MOTION: str(motion_path)},
    )

    head_rotmat, head_confidence = stage_9_capture_face._body_head_rotation(runRecord)

    assert head_rotmat.shape == (n, 3, 3)
    # At the identity pose (all-zero global_orient/body_pose), every joint's
    # global rotation, including Head, is the identity matrix.
    assert np.allclose(head_rotmat, np.eye(3, dtype=np.float32)[None].repeat(n, axis=0))
    assert np.array_equal(head_confidence, [True, False, True])


def test_run_produces_face_params_and_motion_files(stage_9_result):
    params_path = stage_9_result[stage_9_capture_face.OUTPUT_FACE_PARAMS]
    motion_path = stage_9_result[stage_9_capture_face.OUTPUT_FACE_MOTION]

    params = np.load(params_path)
    for key in (
        "deca_shape", "deca_exp", "deca_pose", "deca_valid", "mica_shape", "mica_valid",
        "mp_landmarks", "mp_valid", "mp_blendshapes",
    ):
        assert key in params

    n = len(params["deca_valid"])
    assert n > 0
    assert params["mp_landmarks"].shape == (n, 478, 3)
    assert params["mp_blendshapes"].shape == (n, len(ARKIT_BLENDSHAPE_NAMES))

    motion = np.load(motion_path)
    for key in ("flame_betas", "flame_expression", "flame_global_orient", "flame_jaw_pose", "flame_transl", "flame_valid"):
        assert key in motion
    assert motion["flame_expression"].shape == (n, 50)
    assert motion["flame_betas"].shape == (300,)
    assert np.isfinite(motion["flame_expression"]).all()
    assert np.isfinite(motion["flame_transl"]).all()


def test_run_produces_output_face_csv(stage_9_result, runRecord):
    csv_path = Path(stage_9_result[stage_9_capture_face.OUTPUT_FACE_CSV])
    assert csv_path.exists()
    assert csv_path.name == "output_face.csv"

    lines = csv_path.read_text().strip("\n").split("\n")
    assert lines[0] == ",".join(CSV_HEADER)

    params = np.load(stage_9_result[stage_9_capture_face.OUTPUT_FACE_PARAMS])
    n = len(params["deca_valid"])
    assert len(lines) == n + 1  # header + one row per frame

    for line in lines[1:]:
        values = [float(v) for v in line.split(",")[2:]]  # skip Timecode, BlendshapeCount
        assert all(np.isfinite(values))

    assert runRecord.outputs.final_face_csv == str(csv_path)


@pytest.mark.skipif(
    not (FLAME_MODEL_PATH.exists() and FACE_BASES_PATH.exists()),
    reason="needs the FLAME model file and body_models/arkit/face_bases.npz (see README's Setup section)",
)
def test_compute_arkit_channels_uses_mediapipe_for_jaw_open_and_wide(tmp_path):
    # Targets _compute_arkit_channels + write_livelink_csv directly, the
    # same two calls run() itself makes inline (see that function's own
    # body), rather than a bespoke wrapper function that duplicated them
    # for no caller but this test (removed as dead code, found by vulture).
    n = 4
    motion = {
        KEY_JAW_POSE: np.zeros((n, 3), dtype=np.float32), KEY_EXPRESSION: np.zeros((n, 50), dtype=np.float32),
        KEY_VALID: np.ones(n, dtype=bool),
    }
    motion[KEY_JAW_POSE][:, 0] = JAW_OPEN_REFERENCE_RAD * 0.5

    mp_landmarks = np.zeros((n, 478, 3), dtype=np.float32)
    from pipeline.algorithms.face.face_gaze import (
        LEFT_EYE_BOTTOM, LEFT_EYE_INNER, LEFT_EYE_OUTER, LEFT_EYE_TOP, LEFT_IRIS_CENTER, RIGHT_EYE_BOTTOM,
        RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_EYE_TOP, RIGHT_IRIS_CENTER,
    )
    for inner, outer, top, bottom, iris, y in (
        (RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_IRIS_CENTER, 0),
        (LEFT_EYE_INNER, LEFT_EYE_OUTER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_IRIS_CENTER, 10),
    ):
        mp_landmarks[:, inner] = [1 if inner in (LEFT_EYE_INNER,) else -1, y, 0]
        mp_landmarks[:, outer] = [-1 if inner in (LEFT_EYE_INNER,) else 1, y, 0]
        mp_landmarks[:, top] = [0, y - 1, 0]
        mp_landmarks[:, bottom] = [0, y + 1, 0]
        mp_landmarks[:, iris] = [0, y, 0]
    params = {
        "mp_landmarks": mp_landmarks, "mp_valid": np.ones(n, dtype=bool), "deca_valid": np.ones(n, dtype=bool),
        "mp_blendshapes": np.zeros((n, len(ARKIT_BLENDSHAPE_NAMES)), dtype=np.float32),
    }
    params["mp_blendshapes"][:, ARKIT_BLENDSHAPE_NAMES.index("EyeWideLeft")] = 0.42
    params["mp_blendshapes"][:, ARKIT_BLENDSHAPE_NAMES.index("EyeWideRight")] = 0.37
    params["mp_blendshapes"][:, ARKIT_BLENDSHAPE_NAMES.index("JawOpen")] = 0.61

    arkit_weights, head_eye_euler = stage_9_capture_face._compute_arkit_channels(
        motion, params, torch.device("cpu"), fps=30.0,
    )
    csv_path = tmp_path / "output_face.csv"
    write_livelink_csv(csv_path, arkit_weights, head_eye_euler, fps=30.0)
    lines = csv_path.read_text().strip("\n").split("\n")

    jaw_open_col = ARKIT_BLENDSHAPE_NAMES.index("JawOpen") + 2  # +2 for Timecode, BlendshapeCount
    values = lines[1].split(",")
    assert float(values[jaw_open_col]) == pytest.approx(0.61)
    assert np.allclose(arkit_weights[:, ARKIT_BLENDSHAPE_NAMES.index("JawOpen")], 0.61)
    assert np.allclose(arkit_weights[:, ARKIT_BLENDSHAPE_NAMES.index("EyeWideLeft")], 0.42)
    assert np.allclose(arkit_weights[:, ARKIT_BLENDSHAPE_NAMES.index("EyeWideRight")], 0.37)
