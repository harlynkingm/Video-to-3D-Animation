"""Unit tests for `face_preview.py`'s pure-numpy PC2/OBJ writing (no FLAME
model needed) plus a real-model-gated test of `write_flame_preview` itself.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from pipeline.algorithms.face.face_preview import (
    ANIMATION_PC2_FILENAME, ARKIT_PREVIEW_SHAPES_PATH, FLAME_TO_BLENDER_ROTATION, KEY_TEMPLATE_FACES,
    KEY_TEMPLATE_VERTICES, LANDMARK_PREVIEW_SEPARATION, TEMPLATE_NPZ_FILENAME, _landmark_knn_edges, _write_pc2,
    _write_template_npz, flame_to_blender, write_arkit_preview, write_flame_preview, write_landmark_preview,
)
from pipeline.helpers.livelink_csv import ARKIT_BLENDSHAPE_NAMES
from tests.conftest import FLAME_MODEL_PATH


def test_flame_to_blender_maps_flame_up_onto_blender_up():
    # FLAME canonical is +Y up; it must land on Blender's +Z.
    assert np.allclose(flame_to_blender(np.array([[0.0, 1.0, 0.0]])), [[0.0, 0.0, 1.0]])
    # ...and FLAME's +Z (out of the face) must land on Blender's +X, the
    # face points at the display axis a viewer reads as "facing right"
    # without orbiting first.
    assert np.allclose(flame_to_blender(np.array([[0.0, 0.0, 1.0]])), [[1.0, 0.0, 0.0]])


def test_flame_to_blender_is_a_rotation_not_a_mirror():
    """Determinant +1, not -1: a bare axis *swap* has determinant -1 and
    mirrors the mesh inside-out. Necessary but NOT sufficient, the
    upside-down (x, -z, y) variant also passes this, which is exactly why
    the anatomical test below exists."""
    assert np.linalg.det(FLAME_TO_BLENDER_ROTATION) == pytest.approx(1.0)
    assert np.allclose(FLAME_TO_BLENDER_ROTATION @ FLAME_TO_BLENDER_ROTATION.T, np.eye(3))


@pytest.mark.skipif(not FLAME_MODEL_PATH.exists(), reason="needs the FLAME model (see README's Setup section)")
def test_flame_to_blender_puts_brows_above_mouth():
    """The orientation check that actually bites. Both a determinant test and
    an axis-mapping test are satisfied by a rotation that renders the face
    upside down. Pinning a genuine anatomical invariant, eyebrows sit above
    the mouth, catches that; abstract matrix properties do not."""
    import torch

    from pipeline.algorithms.face.face_landmark_fit import (
        FLAME_NUM_BETAS, FLAME_NUM_EXPRESSION, NUM_FLAME_LANDMARKS, _build_flame_model,
    )

    model = _build_flame_model(torch.device("cpu"), batch_size=1)
    with torch.no_grad():
        out = model(
            betas=torch.zeros(1, FLAME_NUM_BETAS), expression=torch.zeros(1, FLAME_NUM_EXPRESSION),
            global_orient=torch.zeros(1, 3), neck_pose=torch.zeros(1, 3), jaw_pose=torch.zeros(1, 3),
            leye_pose=torch.zeros(1, 3), reye_pose=torch.zeros(1, 3), transl=torch.zeros(1, 3),
        )
    landmarks = out.joints[0, -NUM_FLAME_LANDMARKS:, :].numpy()

    brows = flame_to_blender(landmarks[[i - 17 for i in range(19, 25)]])  # dlib 19-24
    mouth = flame_to_blender(landmarks[[i - 17 for i in range(60, 68)]])  # dlib 60-67
    assert brows[:, 2].mean() > mouth[:, 2].mean()


def test_write_pc2_header_and_data_roundtrip(tmp_path):
    verts_per_frame = np.random.default_rng(0).normal(size=(3, 5, 3)).astype(np.float32)
    out_path = tmp_path / "anim.pc2"
    _write_pc2(out_path, verts_per_frame)

    with open(out_path, "rb") as f:
        signature, version, num_points, start_frame, sample_rate, num_samples = struct.unpack("<12s i i f f i", f.read(32))
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(num_samples, num_points, 3)

    assert signature == b"POINTCACHE2\x00"  # 12-byte C string, null-padded
    assert version == 1
    assert num_points == 5
    assert num_samples == 3
    assert np.allclose(data, verts_per_frame)


def test_write_template_npz_roundtrips_vertices_and_faces(tmp_path):
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    out_path = tmp_path / "template.npz"
    _write_template_npz(out_path, verts, faces)

    with np.load(out_path) as data:
        assert np.allclose(data[KEY_TEMPLATE_VERTICES], verts)
        # 0-indexed, matching bpy's own Mesh.from_pydata (OBJ's 1-indexing
        # was a source of off-by-one when the template was written as .obj).
        assert np.array_equal(data[KEY_TEMPLATE_FACES], faces)


@pytest.mark.skipif(not FLAME_MODEL_PATH.exists(), reason="needs the FLAME model (see README's Setup section)")
def test_write_flame_preview_produces_matching_frame_count(tmp_path):
    import torch

    n = 4
    motion = {
        "flame_betas": np.zeros(300, dtype=np.float32),
        "flame_expression": np.zeros((n, 50), dtype=np.float32),
        "flame_jaw_pose": np.zeros((n, 3), dtype=np.float32),
        "flame_global_orient": np.zeros((n, 3), dtype=np.float32),
        "flame_transl": np.tile([0.0, 0.0, 1.0], (n, 1)).astype(np.float32),
    }

    outputs = write_flame_preview(motion, tmp_path, device=torch.device("cpu"))

    assert outputs["face_preview_template"] == str(tmp_path / TEMPLATE_NPZ_FILENAME)
    assert outputs["face_preview_pc2"] == str(tmp_path / ANIMATION_PC2_FILENAME)
    assert (tmp_path / TEMPLATE_NPZ_FILENAME).exists()

    with open(tmp_path / ANIMATION_PC2_FILENAME, "rb") as f:
        _, _, num_points, _, _, num_samples = struct.unpack("<12s i i f f i", f.read(32))
    assert num_samples == n
    assert num_points > 0


def test_landmark_knn_edges_connects_every_point():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(30, 3))
    edges = _landmark_knn_edges(points, k=4)

    assert edges.ndim == 2 and edges.shape[1] == 2
    connected = set(edges[:, 0]) | set(edges[:, 1])
    assert connected == set(range(30))
    assert (edges[:, 0] < edges[:, 1]).all()  # deduplicated, canonical order


def test_write_landmark_preview_separates_raw_and_smoothed(tmp_path):
    rng = np.random.default_rng(1)
    n, v = 20, 15
    t = np.linspace(0, 1, n)
    clean = np.zeros((n, v, 3))
    clean[:, :, 0] = 500.0 * np.sin(2 * np.pi * t)[:, None]  # a real, slow sweep in pixel-scale x
    mp_landmarks = clean + rng.normal(0, 15.0, clean.shape)  # pixel-scale per-frame noise
    mp_valid = np.ones(n, dtype=bool)

    outputs = write_landmark_preview(mp_landmarks, mp_valid, smoothing_window=7, out_dir=tmp_path)

    assert set(outputs) == {
        "landmark_preview_raw_template", "landmark_preview_raw_pc2",
        "landmark_preview_smoothed_template", "landmark_preview_smoothed_pc2",
    }
    for path in outputs.values():
        assert Path(path).exists()

    with np.load(outputs["landmark_preview_raw_template"]) as raw_t, \
            np.load(outputs["landmark_preview_smoothed_template"]) as smoothed_t:
        raw_verts = raw_t[KEY_TEMPLATE_VERTICES]
        smoothed_verts = smoothed_t[KEY_TEMPLATE_VERTICES]
        # placed apart on X by 2x the separation constant (raw shifted -SEP, smoothed +SEP)
        assert np.isclose((smoothed_verts[:, 0] - raw_verts[:, 0]).mean(), 2 * LANDMARK_PREVIEW_SEPARATION, atol=0.05)
        assert raw_t[KEY_TEMPLATE_FACES].shape[1] == 2  # edges, not triangles

    with open(outputs["landmark_preview_raw_pc2"], "rb") as f:
        _, _, num_points, _, _, num_samples = struct.unpack("<12s i i f f i", f.read(32))
    assert num_samples == n
    assert num_points == v


ARKIT_PREVIEW_SHAPES_PRESENT = ARKIT_PREVIEW_SHAPES_PATH.exists()
arkit_preview_shapes_skip = pytest.mark.skipif(
    not ARKIT_PREVIEW_SHAPES_PATH.exists(), reason="needs body_models/arkit/face_preview_shapes.npz",
)


@arkit_preview_shapes_skip
def test_write_arkit_preview_produces_matching_frame_count(tmp_path):
    n = 5
    arkit_weights = np.zeros((n, len(ARKIT_BLENDSHAPE_NAMES)), dtype=np.float32)
    arkit_weights[:, ARKIT_BLENDSHAPE_NAMES.index("JawOpen")] = np.linspace(0, 1, n)
    head_eye_euler = np.zeros((n, 9), dtype=np.float32)

    outputs = write_arkit_preview(arkit_weights, head_eye_euler, tmp_path)

    assert set(outputs) == {"arkit_preview_template", "arkit_preview_pc2"}
    for path in outputs.values():
        assert Path(path).exists()

    with open(outputs["arkit_preview_pc2"], "rb") as f:
        _, _, num_points, _, _, num_samples = struct.unpack("<12s i i f f i", f.read(32))
    assert num_samples == n
    assert num_points == 5023  # FLAME's own native vertex count


@arkit_preview_shapes_skip
def test_write_arkit_preview_zero_weights_matches_neutral_template():
    # All-zero weights/eye-euler should deform nothing, frame 0's vertices
    # should equal the un-deformed neutral mesh written as the template.
    n = 3
    arkit_weights = np.zeros((n, len(ARKIT_BLENDSHAPE_NAMES)), dtype=np.float32)
    head_eye_euler = np.zeros((n, 9), dtype=np.float32)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        outputs = write_arkit_preview(arkit_weights, head_eye_euler, Path(tmp))
        with np.load(outputs["arkit_preview_template"]) as template:
            template_verts = template[KEY_TEMPLATE_VERTICES]
        with open(outputs["arkit_preview_pc2"], "rb") as f:
            f.read(32)
            frame0 = np.frombuffer(f.read(5023 * 3 * 4), dtype=np.float32).reshape(5023, 3)
    assert np.allclose(template_verts, frame0, atol=1e-5)


@arkit_preview_shapes_skip
def test_write_arkit_preview_returns_empty_when_shapes_missing(tmp_path, monkeypatch):
    from pipeline.algorithms.face import face_preview

    monkeypatch.setattr(face_preview, "ARKIT_PREVIEW_SHAPES_PATH", tmp_path / "does_not_exist.npz")
    n = 2
    arkit_weights = np.zeros((n, len(ARKIT_BLENDSHAPE_NAMES)), dtype=np.float32)
    head_eye_euler = np.zeros((n, 9), dtype=np.float32)

    assert write_arkit_preview(arkit_weights, head_eye_euler, tmp_path) == {}


@arkit_preview_shapes_skip
def test_write_arkit_preview_eyeball_rotates_with_gaze(tmp_path):
    # Regression guard for the real bug found on real footage: the eyeball
    # used to read as visually frozen (nearest-vertex ICT registration
    # structurally excludes iris/sclera source points, see
    # LEFT_EYEBALL_VERTS' own comment). A nonzero LeftEyeYaw should now
    # visibly move the left eyeball's own vertices away from the neutral
    # template, not leave them untouched.
    from pipeline.algorithms.face.face_preview import LEFT_EYEBALL_VERTS

    n = 2
    arkit_weights = np.zeros((n, len(ARKIT_BLENDSHAPE_NAMES)), dtype=np.float32)
    head_eye_euler = np.zeros((n, 9), dtype=np.float32)
    head_eye_euler[1, 3] = 25.0  # LeftEyeYaw, frame 1 only

    outputs = write_arkit_preview(arkit_weights, head_eye_euler, tmp_path)
    with open(outputs["arkit_preview_pc2"], "rb") as f:
        f.read(32)
        frame0 = np.frombuffer(f.read(5023 * 3 * 4), dtype=np.float32).reshape(5023, 3)
        frame1 = np.frombuffer(f.read(5023 * 3 * 4), dtype=np.float32).reshape(5023, 3)

    assert not np.allclose(frame0[LEFT_EYEBALL_VERTS], frame1[LEFT_EYEBALL_VERTS], atol=1e-4)
