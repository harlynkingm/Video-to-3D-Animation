"""Stage 6 regression tests.

Two layers:
  - A pure-numpy unit test of `fit_scene_scale` on a synthetic scene with a
    *known* scale, proving the math actually recovers it (no GPU/checkpoints
    needed -- always runs).
  - Real-data checks on the stage's output against the test clip (needs the
    full stage chain, so GPU + checkpoints + SMPL-X model; skipped otherwise
    via the stage_6_result fixture).
"""

from __future__ import annotations

import json

import numpy as np

from pipeline.algorithms.similarity_transform import fit_scene_scale


def _synthetic_scene(known_scale: float):
    """A frontal grid of 'SMPL-X' vertices at Z=2m, plus a depth map whose
    pixels read `known_scale x` deeper. Back-projecting the deeper depth gives a
    point cloud that is `known_scale` times larger in every dimension, so the
    fit should recover `known_scale` and a near-zero translation."""
    fx = fy = 500.0
    width = height = 200
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    # Well-separated grid, kept narrow enough that every point (at Z=2m with
    # fx=500) projects inside the 200px frame.
    xs = np.linspace(-0.35, 0.35, 20)
    ys = np.linspace(-0.35, 0.35, 20)
    gx, gy = np.meshgrid(xs, ys)
    z = 2.0
    verts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=-1)

    u = np.round(verts[:, 0] * fx / z + cx).astype(int)
    v = np.round(verts[:, 1] * fy / z + cy).astype(int)

    depth = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    depth[v, u] = z * known_scale
    mask[v, u] = True

    return verts, depth, K, mask


def test_fit_recovers_a_known_scale():
    known_scale = 1.5
    verts, depth, K, mask = _synthetic_scene(known_scale)
    # Only the grid pixels have depth; restrict correspondence to them.
    scale, translation, n = fit_scene_scale(verts, depth, K, mask)

    assert abs(scale - known_scale) / known_scale < 0.02
    assert np.linalg.norm(translation) < 0.05
    assert n >= 300


def test_fit_is_deterministic():
    verts, depth, K, mask = _synthetic_scene(1.5)
    a = fit_scene_scale(verts, depth, K, mask)
    b = fit_scene_scale(verts, depth, K, mask)
    assert a[0] == b[0]
    assert np.array_equal(a[1], b[1])
    assert a[2] == b[2]


def test_scene_scale_output_is_plausible(stage_6_result):
    data = json.loads(open(stage_6_result["scene_scale"]).read())

    scale = data["scale"]
    assert np.isfinite(scale)
    # DA3 metric depth vs GVHMR SMPL-X disagree by a modest factor on real data
    # (~1.26x measured on this clip); a wildly out-of-band value means the fit broke.
    assert 0.3 < scale < 4.0

    translation = np.array(data["translation"])
    assert translation.shape == (3,)
    assert np.isfinite(translation).all()

    assert data["n_correspondences"] >= 200


def _read_ply_colors(path):
    lines = path.read_text().splitlines()
    header_end = lines.index("end_header")
    rows = [line.split() for line in lines[header_end + 1:] if line.strip()]
    return np.array([[int(c) for c in row[3:6]] for row in rows])


def test_scene_preview_combines_all_three_elements(stage_6_result):
    from pathlib import Path

    from pipeline.stages.stage_6_align_scene_scale import HUMAN_COLOR, OBJECT_COLOR

    ply_path = Path(stage_6_result["scene_preview"])
    assert ply_path.exists()

    colors = _read_ply_colors(ply_path)
    assert len(colors) > 0

    # The human mesh (green) and the tracked object (red) must both be present,
    # alongside the RGB scene points -- proving all three elements landed in the
    # one combined, aligned point cloud.
    has_human = np.any(np.all(colors == HUMAN_COLOR, axis=1))
    has_object = np.any(np.all(colors == OBJECT_COLOR, axis=1))
    assert has_human, "no human-colored points in scene preview"
    assert has_object, "no object-colored points in scene preview (object was tracked on this clip)"
