"""Unit tests for `deca_preprocess.py` and `deca_encoder.py`, the pure-Python
preprocessing and the network's architecture wiring. None of this needs the
real checkpoint or a GPU: `DecaEncoder()` with default (random) init is enough
to prove the resnet50 backbone and MLP head connect with the right shapes,
which is exactly the kind of wiring mistake (wrong slice indices, wrong
`OUTPUT_DIM`) that a real-checkpoint strict load would also catch, just
later and more confusingly. `test_deca_checkpoint.py` covers the real
checkpoint separately, gated on it being present.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from pipeline.adapters.deca.deca_encoder import N_CAM, N_EXP, N_POSE, N_SHAPE, DecaEncoder
from pipeline.adapters.deca.deca_preprocess import IMAGE_SIZE, crop_face, face_box_from_body_kpts

DECA_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "deca.safetensors"

# COCO-17 layout: index -> (x, y, confidence). Only 0-4 (nose, eyes, ears) matter here.
_CONF = 0.9


def _kpts(**face_points) -> np.ndarray:
    """Build a (17,3) keypoint array with everything at 0 confidence except
    the named face points, e.g. _kpts(nose=(100,50), leye=(90,40))."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    names = {"nose": 0, "leye": 1, "reye": 2, "lear": 3, "rear": 4}
    for name, (x, y) in face_points.items():
        kpts[names[name]] = [x, y, _CONF]
    return kpts


def test_face_box_needs_at_least_two_confident_points():
    assert face_box_from_body_kpts(_kpts(nose=(100, 100))) is None
    assert face_box_from_body_kpts(np.zeros((17, 3), dtype=np.float32)) is None


def test_face_box_brackets_the_confident_points():
    kpts = _kpts(nose=(100, 110), leye=(90, 90), reye=(110, 90))
    box = face_box_from_body_kpts(kpts)
    assert box is not None
    x1, y1, x2, y2 = box
    # The box must be a real square that contains every confident point (the
    # 0.12*size downward recenter can push the top edge past a point's own y,
    # so only check containment for x and the top-most/bottom-most extent).
    assert x1 < 90 and x2 > 110
    assert (x2 - x1) == (y2 - y1)  # square


def test_face_box_low_confidence_points_are_ignored():
    kpts = _kpts(nose=(100, 100), leye=(90, 90))
    kpts[2] = [500, 500, 0.05]  # reye, present but too low-confidence to count
    box_with_low_conf = face_box_from_body_kpts(kpts)
    box_without = face_box_from_body_kpts(_kpts(nose=(100, 100), leye=(90, 90)))
    assert np.allclose(box_with_low_conf, box_without)


def test_crop_face_shape_and_range():
    img = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
    box = np.array([200, 150, 400, 350], dtype=np.float32)
    crop = crop_face(img, box)
    assert crop.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert crop.dtype == np.float32
    # [0,1] scaling only, no ImageNet mean/std subtraction (unlike HaMeR's
    # hand crop), so every value must land inside the raw normalized range.
    assert crop.min() >= 0.0 and crop.max() <= 1.0


def test_crop_face_is_rgb_not_bgr():
    # The box sits well inside the source image (not touching its edges) so
    # the affine sample never spills outside into BORDER_CONSTANT-filled
    # (black) pixels, which would otherwise contaminate this color check.
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # pure blue in BGR
    box = np.array([100, 100, 200, 200], dtype=np.float32)
    crop = crop_face(img, box)
    # channel 0 of the CHW output should now be the red channel (all zero),
    # channel 2 should carry the original blue-channel data (all 255/255=1).
    assert np.allclose(crop[0], 0.0)
    assert np.allclose(crop[2], 1.0)


def test_deca_encoder_output_shapes():
    model = DecaEncoder().eval()
    image = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.inference_mode():
        out = model(image)
    assert out["shape"].shape == (2, N_SHAPE)
    assert out["exp"].shape == (2, N_EXP)
    assert out["pose"].shape == (2, N_POSE)
    assert out["cam"].shape == (2, N_CAM)


def test_deca_encoder_outputs_are_finite():
    model = DecaEncoder().eval()
    image = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.inference_mode():
        out = model(image)
    for v in out.values():
        assert torch.isfinite(v).all()


def test_real_checkpoint_strict_loads():
    # Catches a real architecture/naming drift between this port and the
    # actual released weights, the random-init tests above only prove the
    # wiring is shape-consistent with itself, not that it matches DECA's own
    # checkpoint.
    if not DECA_CHECKPOINT.exists():
        pytest.skip("needs the DECA checkpoint (see README's Setup section)")

    model = DecaEncoder()
    model.load_state_dict(load_file(str(DECA_CHECKPOINT)), strict=True)
