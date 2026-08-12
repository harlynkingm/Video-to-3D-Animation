"""Unit tests for `mp2dlib.py` and `face_landmarks_preprocess.py`, the pure
numpy landmark-correspondence and crop math. `test_face_landmarker_checkpoint`
covers the real MediaPipe model file separately, gated on it being present.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.adapters.face_landmarks.face_landmarks_preprocess import crop_for_landmarker, landmarks_to_full_frame
from pipeline.adapters.face_landmarks.mp2dlib import dlib68_to_arcface5, dlib68_to_flame51, mediapipe_to_dlib68

FACE_LANDMARKER_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "face_landmarker.task"


def test_mediapipe_to_dlib68_shape_and_averaging():
    # Two arbitrary points known to average two-Mediapipe-index Dlib point #4
    # (indices 132 and 58): put distinct values there and confirm the mean.
    landmarks = np.zeros((478, 3), dtype=np.float32)
    landmarks[132] = [10.0, 20.0, 0.0]
    landmarks[58] = [30.0, 40.0, 0.0]
    dlib68 = mediapipe_to_dlib68(landmarks)
    assert dlib68.shape == (68, 3)
    assert np.allclose(dlib68[3], [20.0, 30.0, 0.0])  # Dlib point 4 = index 3


def test_dlib68_to_flame51_drops_the_jawline():
    dlib68 = np.arange(68 * 2, dtype=np.float32).reshape(68, 2)
    flame51 = dlib68_to_flame51(dlib68)
    assert flame51.shape == (51, 2)
    assert np.allclose(flame51[0], dlib68[17])  # first FLAME landmark = Dlib point 18 (right brow start)


def test_dlib68_to_arcface5_eye_averaging_and_order():
    dlib68 = np.zeros((68, 2), dtype=np.float32)
    dlib68[36:42] = [10.0, 10.0]  # right eye contour, all identical -> mean is exact
    dlib68[42:48] = [50.0, 10.0]  # left eye contour
    dlib68[30] = [30.0, 30.0]  # nose tip
    dlib68[48] = [20.0, 50.0]  # left mouth corner
    dlib68[54] = [40.0, 50.0]  # right mouth corner
    arcface5 = dlib68_to_arcface5(dlib68)
    assert arcface5.shape == (5, 2)
    assert np.allclose(arcface5, [[10, 10], [50, 10], [30, 30], [20, 50], [40, 50]])


def test_crop_for_landmarker_offset_and_bounds():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    box = np.array([200, 150, 300, 250], dtype=np.float32)
    crop, offset = crop_for_landmarker(frame, box)
    assert crop.ndim == 3 and crop.shape[2] == 3
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    # crop is padded wider than the raw box (CROP_SCALE > 1)
    assert crop.shape[0] > 100 and crop.shape[1] > 100
    assert 0 <= offset[0] < box[0] and 0 <= offset[1] < box[1]


def test_crop_for_landmarker_clips_to_frame_bounds():
    # A box near the frame edge must not push the crop out of bounds.
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    box = np.array([0, 0, 50, 50], dtype=np.float32)
    crop, offset = crop_for_landmarker(frame, box)
    assert offset[0] >= 0 and offset[1] >= 0
    assert crop.shape[0] <= 200 and crop.shape[1] <= 200


def test_landmarks_to_full_frame_maps_corners_correctly():
    # A normalized landmark at (0, 0) should land exactly on the crop's offset;
    # one at (1, 1) should land at the crop's far corner in the full frame.
    landmarks_norm = np.array([[0.0, 0.0, 0.1], [1.0, 1.0, -0.1]], dtype=np.float32)
    offset = np.array([50.0, 20.0], dtype=np.float32)
    full = landmarks_to_full_frame(landmarks_norm, (100, 200), offset)  # (h=100, w=200)
    assert np.allclose(full[0], [50.0, 20.0, 0.1])
    assert np.allclose(full[1], [250.0, 120.0, -0.1])


def test_face_landmarker_checkpoint():
    # Catches a real download/format problem with the model file itself,
    # the pure-Python tests above only prove the surrounding math is correct.
    if not FACE_LANDMARKER_CHECKPOINT.exists():
        pytest.skip("needs the MediaPipe face_landmarker.task checkpoint (see README's Setup section)")

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    base_options = mp_python.BaseOptions(model_asset_path=str(FACE_LANDMARKER_CHECKPOINT))
    options = mp_vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    image = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image))
    assert isinstance(result.face_landmarks, list)  # no crash; random noise legitimately has 0 faces
