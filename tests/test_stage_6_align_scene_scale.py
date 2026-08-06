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
import torch

from pipeline.adapters.sam31.sam31_tracker import KEY_PACKED_MASKS, pack_masks
from pipeline.algorithms.similarity_transform import _fit_anisotropic_scale, fit_scene_scale
from pipeline.progress_tracker import ObjectShapeHint
from pipeline.stages.stage_6_align_scene_scale import _resolve_auto_shape_hint, _select_shape_candidate_frames


def _synthetic_scene(known_scale: float):
    """A grid of 'SMPL-X' vertices with real Z variation (not a flat frontal
    plane -- a constant-Z grid would make the Z-axis spread degenerate, since
    the anisotropic fit needs real depth variation to measure a Z scale from),
    plus a depth map whose pixels read `known_scale x` deeper. Back-projecting
    a fixed pixel at a deeper depth scales X, Y, and Z by the same factor (a
    property of pinhole projection: `x = (u-cx)*depth/fx`), so this fixture
    can only exercise the *isotropic* case -- it proves recovering a shared
    known scale on both axis groups, not that they're fit independently (see
    `test_anisotropic_scale_recovers_independent_xy_and_z_ratios` for that)."""
    fx = fy = 500.0
    width = height = 200
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    # Well-separated grid, kept narrow enough that every point (at Z~2m with
    # fx=500) projects inside the 200px frame.
    xs = np.linspace(-0.35, 0.35, 20)
    ys = np.linspace(-0.35, 0.35, 20)
    gx, gy = np.meshgrid(xs, ys)
    z = 2.0 + 0.3 * gx  # real Z variation, still smooth/simple
    verts = np.stack([gx.ravel(), gy.ravel(), z.ravel()], axis=-1)

    u = np.round(verts[:, 0] * fx / verts[:, 2] + cx).astype(int)
    v = np.round(verts[:, 1] * fy / verts[:, 2] + cy).astype(int)

    depth = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    depth[v, u] = verts[:, 2] * known_scale
    mask[v, u] = True

    return verts, depth, K, mask


def test_fit_recovers_a_known_scale():
    known_scale = 1.5
    verts, depth, K, mask = _synthetic_scene(known_scale)
    # Only the grid pixels have depth; restrict correspondence to them.
    scale, translation, n = fit_scene_scale(verts, depth, K, mask)

    assert scale.shape == (3,)
    assert np.allclose(scale, known_scale, rtol=0.02)
    assert np.linalg.norm(translation) < 0.05
    assert n >= 300


def test_fit_is_deterministic():
    verts, depth, K, mask = _synthetic_scene(1.5)
    a = fit_scene_scale(verts, depth, K, mask)
    b = fit_scene_scale(verts, depth, K, mask)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert a[2] == b[2]


def test_anisotropic_scale_recovers_independent_xy_and_z_ratios():
    """Directly exercises `_fit_anisotropic_scale` on hand-built correspondence
    pairs (bypassing pixel projection entirely, which -- per `_synthetic_scene`'s
    own docstring -- can't produce a genuinely anisotropic scene on its own)."""
    rng = np.random.default_rng(0)
    body_pts = rng.uniform(-0.3, 0.3, (500, 3))
    scale_xy_true, scale_z_true = 1.5, 3.0
    scene_pts = body_pts * np.array([scale_xy_true, scale_xy_true, scale_z_true])

    scale = _fit_anisotropic_scale(body_pts, scene_pts)

    assert scale.shape == (3,)
    assert abs(scale[0] - scale_xy_true) < 0.05
    assert abs(scale[1] - scale_xy_true) < 0.05
    assert abs(scale[2] - scale_z_true) < 0.05


def test_scene_scale_output_is_plausible(stage_6_result):
    data = json.loads(open(stage_6_result["scene_scale"]).read())

    scale = np.array(data["scale"])
    assert scale.shape == (3,)
    assert np.isfinite(scale).all()
    assert scale[0] == scale[1]  # shared lateral X/Y scale
    # DA3 metric depth vs GVHMR SMPL-X disagree by a modest factor on real data;
    # a wildly out-of-band value on either axis group means the fit broke.
    assert (0.2 < scale).all() and (scale < 5.0).all()

    translation = np.array(data["translation"])
    assert translation.shape == (3,)
    assert np.isfinite(translation).all()

    assert data["n_correspondences"] >= 200

    # This test clip always has a tracked object (see conftest.py's
    # OBJECT_PROMPT), so stage 9's sub-phase B pivot needs this too.
    pelvis_rest = np.array(data["pelvis_rest_incam"])
    assert pelvis_rest.shape == (3,)
    assert np.isfinite(pelvis_rest).all()


def test_object_shape_output_is_plausible(stage_6_result):
    # This test clip always has a tracked object (see conftest.py's OBJECT_PROMPT),
    # so a shape fit should always be written alongside the scene scale.
    data = json.loads(open(stage_6_result["object_shape"]).read())

    assert data["kind"] in ("box", "ellipsoid", "cylinder")
    center = np.array(data["center"])
    assert center.shape == (3,)
    assert np.isfinite(center).all()

    if data["kind"] == "box":
        extents = np.array(data["half_extents"])
    elif data["kind"] == "ellipsoid":
        extents = np.array(data["semi_axes"])
    else:
        extents = np.array([data["radius"], data["radius"], data["half_height"]])
    assert extents.shape == (3,)
    assert np.isfinite(extents).all()
    assert (extents > 0).all()
    # A real handheld object shouldn't fit a multi-meter proxy -- a wildly large
    # value means the depth/mask correspondence broke.
    assert (extents < 2.0).all()


def _read_ply_colors(path):
    lines = path.read_text().splitlines()
    header_end = lines.index("end_header")
    rows = [line.split() for line in lines[header_end + 1:] if line.strip()]
    return np.array([[int(c) for c in row[3:6]] for row in rows])


def test_scene_preview_combines_every_element(stage_6_result):
    from pathlib import Path

    from pipeline.stages.stage_6_align_scene_scale import HUMAN_COLOR, OBJECT_COLOR, SHAPE_COLOR

    ply_path = Path(stage_6_result["scene_preview"])
    assert ply_path.exists()

    colors = _read_ply_colors(ply_path)
    assert len(colors) > 0

    # The human mesh (green), the tracked object (red), and the fitted proxy
    # primitive's wireframe (yellow) must all be present, alongside the RGB
    # scene points -- proving every element landed in the one combined,
    # aligned point cloud.
    has_human = np.any(np.all(colors == HUMAN_COLOR, axis=1))
    has_object = np.any(np.all(colors == OBJECT_COLOR, axis=1))
    has_shape = np.any(np.all(colors == SHAPE_COLOR, axis=1))
    assert has_human, "no human-colored points in scene preview"
    assert has_object, "no object-colored points in scene preview (object was tracked on this clip)"
    assert has_shape, "no fitted-shape wireframe points in scene preview (object was tracked on this clip)"


def _disk_mask(radius: int, size: int = 200) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    center = size // 2
    return ((xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2).astype(np.uint8)


def _box_mask(size: int = 200, w: int = 100, h: int = 60) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    cy, cx = size // 2, size // 2
    mask[cy - h // 2:cy + h // 2, cx - w // 2:cx + w // 2] = 1
    return mask


def _save_packed_masks(masks_by_frame: list[np.ndarray], path) -> None:
    stacked = torch.from_numpy(np.stack(masks_by_frame)).bool().unsqueeze(1)  # (N, 1, H, W)
    torch.save({KEY_PACKED_MASKS: pack_masks(stacked)}, path)


def test_select_shape_candidate_frames_ranks_a_notched_circle_below_the_clean_one(tmp_path):
    """Solidity is used for every shape kind, including a round object -- a
    notch (e.g. a gripping hand) breaks convexity the same way regardless of
    what the object's silhouette should otherwise look like."""
    clean_circle = _disk_mask(radius=40)
    notched_circle = clean_circle.copy()
    notched_circle[70:100, 70:100] = 0  # bites into the circle from an edge-adjacent area

    masks_path = tmp_path / "object.pt"
    _save_packed_masks([notched_circle, clean_circle], masks_path)

    candidates = _select_shape_candidate_frames(str(masks_path), n_candidates=2)
    assert candidates[0] == 1  # the clean circle (frame 1) ranks above the notched one (frame 0)


def test_select_shape_candidate_frames_ranks_a_notched_box_below_the_clean_one(tmp_path):
    clean_box = _box_mask()
    notched_box = clean_box.copy()
    notched_box[70:80, 90:110] = 0  # a real boundary concavity, see test_object_extent_fit.py

    masks_path = tmp_path / "object.pt"
    _save_packed_masks([notched_box, clean_box], masks_path)

    candidates = _select_shape_candidate_frames(str(masks_path), n_candidates=2)
    assert candidates[0] == 1  # the clean box (frame 1) ranks above the notched one (frame 0)


def test_select_shape_candidate_frames_excludes_empty_frames(tmp_path):
    empty = np.zeros((200, 200), dtype=np.uint8)
    real = _disk_mask(radius=40)

    masks_path = tmp_path / "object.pt"
    _save_packed_masks([empty, real, empty], masks_path)

    candidates = _select_shape_candidate_frames(str(masks_path), n_candidates=5)
    assert candidates == [1]


def test_select_shape_candidate_frames_respects_n_candidates_limit(tmp_path):
    masks = [_disk_mask(radius=30 + i) for i in range(5)]
    masks_path = tmp_path / "object.pt"
    _save_packed_masks(masks, masks_path)

    candidates = _select_shape_candidate_frames(str(masks_path), n_candidates=2)
    assert len(candidates) == 2


def test_resolve_auto_shape_hint_forces_ellipsoid_when_a_clean_circle_exists_anywhere(tmp_path):
    # Frame 0 (the would-be "anchor") is boxy; only a later frame is round --
    # mirrors a real case where the auto-picked anchor frame was itself
    # motion-blurred but a clean circle existed elsewhere in the clip.
    masks_path = tmp_path / "object.pt"
    _save_packed_masks([_box_mask(), _disk_mask(radius=40)], masks_path)

    assert _resolve_auto_shape_hint(str(masks_path)) == ObjectShapeHint.ELLIPSOID


def test_resolve_auto_shape_hint_falls_back_to_auto_when_nothing_reads_round(tmp_path):
    masks_path = tmp_path / "object.pt"
    _save_packed_masks([_box_mask(), _box_mask(w=80, h=80)], masks_path)

    assert _resolve_auto_shape_hint(str(masks_path)) == ObjectShapeHint.AUTO


def test_resolve_auto_shape_hint_falls_back_to_auto_when_only_an_occluded_circle_exists(tmp_path):
    occluded_circle = _disk_mask(radius=40)
    occluded_circle[70:100, 70:100] = 0  # same notch style as the candidate-ranking test above

    masks_path = tmp_path / "object.pt"
    _save_packed_masks([occluded_circle], masks_path)

    assert _resolve_auto_shape_hint(str(masks_path)) == ObjectShapeHint.AUTO
