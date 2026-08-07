"""align_scene_scale: recovers the metric relationship between the depth map
and the GVHMR SMPL-X human, at the anchor frame, and (if an object was
tracked) fits a proxy primitive to its shape in that same metric space.

DA3METRIC-LARGE's depth and GVHMR's SMPL-X are both nominally metric but
disagree by a systematic factor on real data (measured ~1.26x on the test
clip), so this stage fits the scale + translation that reconciles them
(`similarity_transform.fit_scene_scale`). The object's mask is back-projected
through the same depth map and the same fitted `(scale, translation)`, then
`object_extent_fit.fit_object_shape` picks a box, ellipsoid, or cylinder primitive (or the
user's `--object-shape-hint` override) -- see that module's own docstring for
why a primitive instead of a reconstructed mesh.

This stage uses the body-only SMPL-X from `estimate_human_motion` (hands
don't affect the body's overall scale), so it does not wait on
`retarget_hands`.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ..adapters.depth_anything3_adapter import DepthAnything3Adapter, KEY_DEPTH
from ..adapters.gvhmr.gvhmr_adapter import (
    KEY_BETAS,
    KEY_BODY_POSE,
    KEY_GLOBAL_ORIENT,
    KEY_PRED_SMPL_PARAMS_INCAM,
    KEY_TRANSL,
)
from ..adapters.sam31.sam31_tracker import KEY_PACKED_MASKS, unpack_masks
from ..algorithms.depth_unprojection import scale_intrinsics_to_resolution, unproject_depth_to_points
from ..algorithms.object_extent_fit import (
    correct_center_depth,
    correct_cylinder_extent_from_mask,
    correct_foreshortened_axis,
    correct_lateral_axes_from_mask,
    fit_object_shape,
    mask_circularity,
    mask_metric_extent,
    mask_solidity,
    sample_shape_surface,
)
from ..algorithms.similarity_transform import fit_scene_scale
from ..pipeline_stage_base import cli_entrypoint
from ..helpers.ply_export_helper import write_colored_ply
from ..helpers.progress_reporter import frame_progress, report_single_shot
from ..progress_tracker import ObjectShapeHint, RunRecord, StageName
from ..stages.stage_1_mask_and_track import OUTPUT_HUMAN_MASKS, OUTPUT_OBJECT_MASKS
from ..stages.stage_2_estimate_human_motion import OUTPUT_HUMAN_MOTION
from ..stages.stage_3_estimate_depth import OUTPUT_DEPTH

# Repo root is 2 levels up from this file (stages/ -> pipeline/ -> root). Same
# SMPL-X model file the GVHMR skeleton uses (see gvhmr_smplx_skeleton.py).
SMPLX_MODEL_PATH = Path(__file__).resolve().parents[2] / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"

# smplx.create config. The per-frame parameter names smplx.forward expects
# (global_orient/body_pose/betas/transl) are the same strings GVHMR names its
# own output keys after, so the KEY_* constants below double as those kwargs.
SMPLX_MODEL_TYPE = "smplx"
SMPLX_GENDER = "neutral"
SMPLX_NUM_BETAS = 10
SMPLX_BODY_POSE_DIM = 63  # 21 body joints x 3 axis-angle
POSE_AXIS_DIM = 3

# stage_0_ingest_video.py's own output key, consumed here (only for the optional preview).
FRAMES_DIR_OUTPUT_KEY = "frames_dir"

SCALE_DIRNAME = f"stage{StageName.STAGE_6_ALIGN_SCENE_SCALE.stage_number}_scale"
SCENE_SCALE_FILENAME = "scene_scale.json"
OBJECT_SHAPE_FILENAME = "object_shape.json"
SCENE_PREVIEW_FILENAME = "scene_preview.ply"

# This stage's own progress.json output keys.
OUTPUT_SCENE_SCALE = "scene_scale"
OUTPUT_OBJECT_SHAPE = "object_shape"
OUTPUT_SCENE_PREVIEW = "scene_preview"

# Keys inside scene_scale.json.
KEY_SCALE = "scale"
KEY_TRANSLATION = "translation"
KEY_N_CORRESPONDENCES = "n_correspondences"
# Only present when an object was tracked -- see _pelvis_rest_position's own
# docstring for why stage 10's object placement needs this.
KEY_PELVIS_REST = "pelvis_rest_incam"

# scene_preview.ply color coding, so the elements are visually separable.
HUMAN_COLOR = np.array([80, 220, 100], dtype=np.uint8)  # green: SMPL-X body mesh
OBJECT_COLOR = np.array([230, 60, 60], dtype=np.uint8)  # red: tracked object pixels
SHAPE_COLOR = np.array([255, 220, 0], dtype=np.uint8)  # yellow: fitted proxy primitive wireframe
# Drop scene points beyond this multiple of the human's own depth, so far
# background/sky doesn't dwarf the person and object (no sky mask is available
# in this stage, unlike stage 3's standalone preview).
SCENE_PREVIEW_DEPTH_CLIP_FACTOR = 3.0


def _object_points_in_body_space(
    depth: np.ndarray, K: np.ndarray, object_mask: np.ndarray, scale: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Back-projects the object's masked depth pixels and maps them into the
    same body-metric space `fit_scene_scale` already put the human in (see
    `_render_scene_preview`'s identical `scene_in_body` conversion)."""
    scene_cloud = unproject_depth_to_points(depth, K)
    object_points = scene_cloud[object_mask.reshape(-1)]
    return (object_points - translation) / scale


def _build_smplx_anchor_mesh(incam_params: dict, anchor_frame_index: int) -> np.ndarray:
    """Build the SMPL-X vertex mesh (camera-space, metric) at the anchor frame
    from GVHMR's incam body params. Hands/expression are left neutral -- they
    don't affect the body's overall scale, which is all this stage needs."""
    import smplx

    model = smplx.create(
        str(SMPLX_MODEL_PATH),
        model_type=SMPLX_MODEL_TYPE,
        gender=SMPLX_GENDER,
        num_betas=SMPLX_NUM_BETAS,
        use_pca=False,
        flat_hand_mean=True,
    )
    params = {
        KEY_GLOBAL_ORIENT: incam_params[KEY_GLOBAL_ORIENT][anchor_frame_index].reshape(1, POSE_AXIS_DIM),
        KEY_BODY_POSE: incam_params[KEY_BODY_POSE][anchor_frame_index].reshape(1, SMPLX_BODY_POSE_DIM),
        KEY_BETAS: incam_params[KEY_BETAS][anchor_frame_index].reshape(1, SMPLX_NUM_BETAS),
        KEY_TRANSL: incam_params[KEY_TRANSL][anchor_frame_index].reshape(1, POSE_AXIS_DIM),
    }
    output = model(**{key: value.float() for key, value in params.items()})
    return output.vertices.detach().numpy()[0]


def _pelvis_rest_position(betas: torch.Tensor) -> np.ndarray:
    """The pelvis joint's own position at zero pose/orientation/translation,
    for these betas -- SMPL-X's `global_orient` rotates the body *about this
    point*, not the world origin (confirmed empirically: with `transl=0`,
    changing `global_orient` leaves the reported pelvis joint position
    unchanged). A skeleton joint's own world position already accounts for
    this automatically via forward kinematics; a standalone point that isn't
    part of the kinematic chain (the fitted object's own `center`) needs this
    pivot explicitly to undergo the *same* rigid transform stage 10 applies to
    the body -- see that stage's `_object_shape_to_blender_world`.
    """
    import smplx

    model = smplx.create(
        str(SMPLX_MODEL_PATH),
        model_type=SMPLX_MODEL_TYPE,
        gender=SMPLX_GENDER,
        num_betas=SMPLX_NUM_BETAS,
        use_pca=False,
        flat_hand_mean=True,
    )
    output = model(betas=betas.float().reshape(1, SMPLX_NUM_BETAS))
    return output.joints.detach().numpy()[0, 0]


def _render_scene_preview(
    smplx_verts: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    scale: np.ndarray,
    translation: np.ndarray,
    human_mask: np.ndarray,
    object_mask: np.ndarray | None,
    object_shape: dict | None,
    anchor_frame_path: Path,
    out_path: Path,
) -> None:
    """One PLY that puts every element in the SMPL-X human's metric space,
    color-coded, so the fit can be eyeballed in Blender: the green SMPL-X body
    mesh should sit inside its own (RGB) depth points, the red object points
    should land at the hands, and (if the object was tracked) a yellow
    wireframe of the fitted proxy primitive should hug those same red points.
    The depth cloud is mapped into body space via the fitted
    `(scale, translation)`; the human mesh is already there.
    """
    scene_cloud = unproject_depth_to_points(depth, K)  # (H*W, 3), camera space
    scene_in_body = (scene_cloud - translation) / scale

    anchor_bgr = cv2.imread(str(anchor_frame_path))
    anchor_rgb = cv2.cvtColor(anchor_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(anchor_rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR).reshape(-1, 3)

    max_keep_depth = float(np.median(depth[human_mask])) * SCENE_PREVIEW_DEPTH_CLIP_FACTOR
    keep = (scene_cloud[:, 2] > 0) & (scene_cloud[:, 2] <= max_keep_depth)

    scene_points = scene_in_body[keep]
    scene_colors = rgb[keep].copy()
    if object_mask is not None:
        scene_colors[object_mask.reshape(-1)[keep]] = OBJECT_COLOR

    human_colors = np.tile(HUMAN_COLOR, (len(smplx_verts), 1))

    points = np.vstack([scene_points, smplx_verts])
    colors = np.vstack([scene_colors, human_colors])

    if object_shape is not None:
        shape_points = sample_shape_surface(object_shape)
        shape_colors = np.tile(SHAPE_COLOR, (len(shape_points), 1))
        points = np.vstack([points, shape_points])
        colors = np.vstack([colors, shape_colors])

    write_colored_ply(points, colors, out_path)


def _load_mask_at_depth_res(masks_path: str, anchor: int, depth_hw: tuple[int, int]) -> np.ndarray:
    packed = torch.load(masks_path, weights_only=False)[KEY_PACKED_MASKS]
    mask_anchor = unpack_masks(packed[anchor])[0].numpy().astype(np.uint8)
    return cv2.resize(mask_anchor, (depth_hw[1], depth_hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def _score_frames_by_mask(masks_path: str, score_fn) -> list[tuple[float, int]]:
    """Scores every frame with a real object mask via `score_fn`
    (`mask_circularity` or `mask_solidity`), descending. Works directly on
    the bit-packed masks, one frame at a time -- unpacking every frame of a
    long clip at once risks the same OOM `sam31_adapter._stitch_tracked_
    slots` was written to avoid. Shared by `_select_shape_candidate_frames`
    and `_resolve_auto_shape_hint` below -- both need the same per-frame
    scan, just a different slice of the result."""
    packed = torch.load(masks_path, weights_only=False)[KEY_PACKED_MASKS]
    n_frames = packed.shape[0]

    scored = []
    for frame in range(n_frames):
        if not packed[frame].any():
            continue
        mask = unpack_masks(packed[frame:frame + 1])[0, 0].numpy().astype(np.uint8)
        score = score_fn(mask)
        if score > 0:
            scored.append((score, frame))
    scored.sort(reverse=True)
    return scored


def _select_shape_candidate_frames(masks_path: str, n_candidates: int) -> list[int]:
    """Ranks every frame with a real object mask by how clean/unoccluded its
    2D silhouette looks, for `run()`'s own multi-frame proportions-
    aggregation pass below -- always via `mask_solidity`, which (per its own
    docstring) reads close to 1.0 for any clean, unoccluded convex silhouette
    regardless of shape kind. `mask_circularity` is deliberately not used
    here even for an ellipsoid: the final fitted `kind` can read "ellipsoid"
    either because `_resolve_auto_shape_hint` confirmed the object is
    genuinely round somewhere in the clip, or just because `fit_object_shape`
    AUTO's residual comparison happened to favor it for a non-round object at
    the anchor -- and for a non-round object, circularity isn't a "how clean
    is this frame" signal at all (confirmed on a real clip: it ranked a
    frame with a badly inflated mask measurement as the single cleanest one,
    while `mask_solidity` correctly excluded it)."""
    scored = _score_frames_by_mask(masks_path, mask_solidity)
    return [frame for _, frame in scored[:n_candidates]]


# A clean rasterized circle tops out around here under cv2's own perimeter
# measure (confirmed both synthetically and on a real clip's own cleanest
# frames, which read 0.88-0.89) -- see test_object_extent_fit.py's own
# circularity test for the same number.
_ROUND_MASK_CIRCULARITY_THRESHOLD = 0.85


def _resolve_auto_shape_hint(masks_path: str) -> ObjectShapeHint:
    """`fit_object_shape`'s own AUTO mode picks box/ellipsoid/cylinder by
    comparing mean point-to-surface residual on the anchor frame's own point
    cloud alone -- confirmed unreliable for a genuinely round object on a
    real clip: a single-view hemisphere sample of a sphere can score
    deceptively close to (sometimes better than) a tightly-bounding box, so
    tiny run-to-run measurement noise (DA3 depth inference isn't
    bit-deterministic) flipped the decision between two otherwise-identical
    reruns of the same clip, even though the recovered *size* stayed
    consistent either way.

    The object's own 2D silhouette is a much more direct signal for
    roundness than a point-cloud residual comparison, so before trusting
    that comparison, this scans every frame's own mask (not just the
    anchor's, which may itself be blurred/occluded) for the roundest
    silhouette found anywhere in the clip. If any frame clears
    `_ROUND_MASK_CIRCULARITY_THRESHOLD`, that's decisive: forces ELLIPSOID
    outright, skipping the unreliable residual comparison entirely.
    Otherwise falls back to AUTO unchanged -- box vs cylinder discrimination
    from a mask silhouette alone isn't attempted here, both can read as a
    similarly non-circular "stadium" shape, so there's no equally direct
    signal to prefer one over the residual comparison for that case.
    """
    scored = _score_frames_by_mask(masks_path, mask_circularity)
    if scored and scored[0][0] >= _ROUND_MASK_CIRCULARITY_THRESHOLD:
        return ObjectShapeHint.ELLIPSOID
    return ObjectShapeHint.AUTO


def _aggregate_object_shape_proportions(
    object_shape: dict,
    scale_xy: float,
    K_native: np.ndarray,
    native_hw: tuple[int, int],
    frame_paths: list[Path],
    stage_1_outputs: dict,
    n_candidates: int,
) -> dict:
    """Corrects `object_shape`'s own proportions (not its orientation or
    center, which stay from the anchor frame's fit -- those describe *where*
    the object sits relative to the body at one reference moment, not
    something a differently-chosen frame could stand in for) using the
    median of several candidate frames' own directly-measured 2D mask
    extents, ranked by how unoccluded they look
    (`_select_shape_candidate_frames`) -- the same per-frame correction
    `correct_lateral_axes_from_mask`/`correct_cylinder_extent_from_mask`
    already apply using just the anchor's own mask, just aggregated across
    several frames instead of trusting whichever one happens to be the
    anchor. Median specifically, not mean or max: a handheld object's mask
    is occluded to very different degrees frame to frame, so a robust
    central estimate is needed against errors in either direction, not just
    against overestimation.

    Falls back to `object_shape` unchanged if fewer than 2 candidate frames
    yield a usable mask measurement (e.g. the object is barely ever cleanly
    visible) -- not enough independent samples to trust a median over the
    anchor's own single measurement.
    """
    candidate_frames = _select_shape_candidate_frames(stage_1_outputs[OUTPUT_OBJECT_MASKS], n_candidates)
    focal_length_px = K_native[0, 0]

    majors, minors = [], []
    depth_adapter = DepthAnything3Adapter()
    depth_adapter.load()
    try:
        for frame in frame_progress(candidate_frames, total=len(candidate_frames),
                                     label=f"{StageName.STAGE_6B_SAMPLE_OBJECT_SHAPE.label}"):
            result = depth_adapter.infer(str(frame_paths[frame]), focal_length_px)
            candidate_depth = result[KEY_DEPTH]
            candidate_K = scale_intrinsics_to_resolution(K_native, native_hw, candidate_depth.shape)
            candidate_mask = _load_mask_at_depth_res(
                stage_1_outputs[OUTPUT_OBJECT_MASKS], frame, candidate_depth.shape
            )
            mask_extent = mask_metric_extent(candidate_mask, candidate_depth, candidate_K)
            if mask_extent is None:
                continue
            major, minor = sorted(mask_extent, reverse=True)
            majors.append(major)
            minors.append(minor)
    finally:
        depth_adapter.unload()

    if len(majors) < 2:
        return object_shape

    median_extent_m = (float(np.median(majors)), float(np.median(minors)))
    object_shape = correct_lateral_axes_from_mask(object_shape, median_extent_m, scale_xy)
    object_shape = correct_cylinder_extent_from_mask(object_shape, median_extent_m, scale_xy)
    return correct_foreshortened_axis(object_shape)


def run(runRecord: RunRecord) -> dict[str, str]:
    anchor = runRecord.scene.anchor_frame_index
    stage_1_outputs = runRecord.stages[StageName.STAGE_1_MASK_AND_TRACK].outputs

    motion = torch.load(
        runRecord.stages[StageName.STAGE_2_ESTIMATE_HUMAN_MOTION].outputs[OUTPUT_HUMAN_MOTION],
        weights_only=False,
    )
    smplx_verts = _build_smplx_anchor_mesh(motion[KEY_PRED_SMPL_PARAMS_INCAM], anchor)

    depth = np.load(runRecord.stages[StageName.STAGE_3_ESTIMATE_DEPTH].outputs[OUTPUT_DEPTH])
    depth_hw = depth.shape
    native_hw = (runRecord.scene.height, runRecord.scene.width)
    K = scale_intrinsics_to_resolution(np.array(runRecord.scene.intrinsics_K), native_hw, depth_hw)

    human_mask = _load_mask_at_depth_res(stage_1_outputs[OUTPUT_HUMAN_MASKS], anchor, depth_hw)
    object_mask = None
    if OUTPUT_OBJECT_MASKS in stage_1_outputs:
        object_mask = _load_mask_at_depth_res(stage_1_outputs[OUTPUT_OBJECT_MASKS], anchor, depth_hw)

    with report_single_shot(StageName.STAGE_6_ALIGN_SCENE_SCALE.label):
        scale, translation, n_correspondences = fit_scene_scale(smplx_verts, depth, K, human_mask)

        object_shape = None
        pelvis_rest = None
        if object_mask is not None:
            object_points = _object_points_in_body_space(depth, K, object_mask, scale, translation)
            shape_hint = runRecord.input.object_shape_hint
            if shape_hint == ObjectShapeHint.AUTO:
                shape_hint = _resolve_auto_shape_hint(stage_1_outputs[OUTPUT_OBJECT_MASKS])
            object_shape = fit_object_shape(object_points, shape_hint)
            mask_extent = mask_metric_extent(object_mask, depth, K)
            if mask_extent is not None:
                object_shape = correct_lateral_axes_from_mask(object_shape, mask_extent, scale[0])
                object_shape = correct_cylinder_extent_from_mask(object_shape, mask_extent, scale[0])
            object_shape = correct_foreshortened_axis(object_shape)
            pelvis_rest = _pelvis_rest_position(motion[KEY_PRED_SMPL_PARAMS_INCAM][KEY_BETAS][anchor])

    if object_shape is not None:
        frames_dir = Path(runRecord.stages[StageName.STAGE_0_INGEST_VIDEO].outputs[FRAMES_DIR_OUTPUT_KEY])
        frame_paths = sorted(frames_dir.glob("*.jpg"))
        K_native = np.array(runRecord.scene.intrinsics_K)
        object_shape = _aggregate_object_shape_proportions(
            object_shape, scale[0], K_native, native_hw, frame_paths, stage_1_outputs,
            runRecord.input.object_shape_candidate_frames,
        )
        object_shape = correct_center_depth(object_shape)

    scale_dir = Path(runRecord.progress_dir) / SCALE_DIRNAME
    scale_dir.mkdir(parents=True, exist_ok=True)
    scene_scale_path = scale_dir / SCENE_SCALE_FILENAME
    scene_scale_json = {
        KEY_SCALE: scale.tolist(),
        KEY_TRANSLATION: translation.tolist(),
        KEY_N_CORRESPONDENCES: n_correspondences,
    }
    if pelvis_rest is not None:
        scene_scale_json[KEY_PELVIS_REST] = pelvis_rest.tolist()
    scene_scale_path.write_text(json.dumps(scene_scale_json, indent=2))

    outputs = {OUTPUT_SCENE_SCALE: str(scene_scale_path)}

    if object_shape is not None:
        object_shape_path = scale_dir / OBJECT_SHAPE_FILENAME
        object_shape_path.write_text(json.dumps(object_shape, indent=2))
        outputs[OUTPUT_OBJECT_SHAPE] = str(object_shape_path)

    if runRecord.input.render_scene_preview:
        frames_dir = Path(runRecord.stages[StageName.STAGE_0_INGEST_VIDEO].outputs[FRAMES_DIR_OUTPUT_KEY])
        anchor_frame_path = sorted(frames_dir.glob("*.jpg"))[anchor]
        scene_preview_path = scale_dir / SCENE_PREVIEW_FILENAME
        _render_scene_preview(
            smplx_verts, depth, K, scale, translation, human_mask, object_mask, object_shape,
            anchor_frame_path, scene_preview_path,
        )
        outputs[OUTPUT_SCENE_PREVIEW] = str(scene_preview_path)

    return outputs


if __name__ == "__main__":
    cli_entrypoint(run, stage_name=StageName.STAGE_6_ALIGN_SCENE_SCALE)
