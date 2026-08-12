"""Unit tests for `mica_preprocess.py`'s alignment geometry and
`mica_encoder.py`'s architecture wiring. None of this needs the real
checkpoint or a GPU: random-init `IResNet100`/`MicaRegressor` prove the
backbone and regressor connect with the right shapes, exactly the kind of
mistake (wrong block counts, wrong stride placement, wrong activation) a real
checkpoint's strict load would also catch, just later and more confusingly.
`test_real_checkpoint_strict_loads` covers the real checkpoint separately,
gated on it being present.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from pipeline.adapters.mica.mica_encoder import EMBEDDING_DIM, N_SHAPE, IResNet100, MicaRegressor
from pipeline.adapters.mica.mica_preprocess import IMAGE_SIZE, REFERENCE_LANDMARKS, norm_crop, normalize_for_arcface

MICA_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "mica.safetensors"


def test_norm_crop_output_shape():
    img = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
    # A plausible 5-point layout, scaled/placed away from the image edges.
    pts = REFERENCE_LANDMARKS * 1.5 + [150, 100]
    aligned = norm_crop(img, pts)
    assert aligned.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert aligned.dtype == np.uint8


def test_norm_crop_recovers_a_known_similarity_transform():
    # If the 5 source points ARE the reference template scaled/rotated/
    # translated by a known transform, norm_crop should invert exactly that
    # transform, so sampling a uniquely-colored marker placed at the source
    # points' own location should land it back at the template's own pixel
    # position in the aligned output.
    theta = 0.15
    scale = 2.0
    offset = np.array([120.0, 90.0])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    src_pts = (scale * (REFERENCE_LANDMARKS @ rot.T)) + offset

    img = np.zeros((400, 400, 3), dtype=np.uint8)
    marker_src = tuple(src_pts[2].astype(int))  # the "nose" point
    cv2_marker_color = (0, 255, 0)
    img[marker_src[1] - 2:marker_src[1] + 2, marker_src[0] - 2:marker_src[0] + 2] = cv2_marker_color

    aligned = norm_crop(img, src_pts.astype(np.float32))
    template_dst = REFERENCE_LANDMARKS[2].astype(int)
    patch = aligned[template_dst[1] - 3:template_dst[1] + 3, template_dst[0] - 3:template_dst[0] + 3]
    assert (patch[:, :, 1] > 100).any()  # the green marker landed near the template's own nose position


def test_normalize_for_arcface_range_and_channel_order():
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # pure blue in BGR
    out = normalize_for_arcface(img)
    assert out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    # (255 - 127.5) / 127.5 == 1.0, (0 - 127.5) / 127.5 == -1.0
    assert np.allclose(out[2], 1.0)  # blue -> RGB channel 2, at the bright extreme
    assert np.allclose(out[0], -1.0)  # red channel, untouched, at the dark extreme


def test_mica_backbone_output_shape():
    model = IResNet100().eval()
    image = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.inference_mode():
        embedding = model(image)
    assert embedding.shape == (2, EMBEDDING_DIM)
    assert torch.isfinite(embedding).all()


def test_mica_regressor_output_shape():
    model = MicaRegressor().eval()
    embedding = torch.rand(3, EMBEDDING_DIM)
    with torch.inference_mode():
        shape = model(embedding)
    assert shape.shape == (3, N_SHAPE)
    assert torch.isfinite(shape).all()


def test_real_checkpoint_strict_loads():
    if not MICA_CHECKPOINT.exists():
        pytest.skip("needs the MICA checkpoint (see README's Setup section)")

    flat = load_file(str(MICA_CHECKPOINT))
    backbone_sd = {k[len("arcface."):]: v for k, v in flat.items() if k.startswith("arcface.")}
    regressor_sd = {k[len("regressor."):]: v for k, v in flat.items() if k.startswith("regressor.")}

    IResNet100().load_state_dict(backbone_sd, strict=True)
    MicaRegressor().load_state_dict(regressor_sd, strict=True)
