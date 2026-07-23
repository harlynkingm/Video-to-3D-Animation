"""align_scene_scale: recovers the metric relationship between the depth map
and the GVHMR SMPL-X human, at the anchor frame.

DA3METRIC-LARGE's depth and GVHMR's SMPL-X are both nominally metric but
disagree by a systematic factor on real data (measured ~1.26x on the test
clip), so this stage fits the scale + translation that reconciles them
(`similarity_transform.fit_scene_scale`). The result lets any depth-derived
geometry (later: the object point cloud) be placed in the SMPL-X human's metric space.

**Scope, as of when this was written**: this stage currently only fits the
scene scale. Fitting the object's proxy shape (box/sphere) from its mask +
depth -- the other half of the reference's `make_hoi.py` -- is deferred until
object placement is actually built. It uses the body-only SMPL-X from
`estimate_human_motion` (hands don't affect the body's overall scale), so it
does not wait on `retarget_hands`.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ..adapters.gvhmr.gvhmr_adapter import (
    KEY_BETAS,
    KEY_BODY_POSE,
    KEY_GLOBAL_ORIENT,
    KEY_PRED_SMPL_PARAMS_INCAM,
    KEY_TRANSL,
)
from ..adapters.sam31.sam31_tracker import KEY_PACKED_MASKS, unpack_masks
from ..algorithms.depth_unprojection import scale_intrinsics_to_resolution, unproject_depth_to_points
from ..algorithms.similarity_transform import fit_scene_scale
from ..pipeline_stage_base import cli_entrypoint
from ..helpers.ply_export_helper import write_colored_ply
from ..progress_tracker import ProgressRecord, StageName
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

SCALE_DIRNAME = "scale"
SCENE_SCALE_FILENAME = "scene_scale.json"
SCENE_PREVIEW_FILENAME = "scene_preview.ply"

# This stage's own progress.json output keys.
OUTPUT_SCENE_SCALE = "scene_scale"
OUTPUT_SCENE_PREVIEW = "scene_preview"

# Keys inside scene_scale.json.
KEY_SCALE = "scale"
KEY_TRANSLATION = "translation"
KEY_N_CORRESPONDENCES = "n_correspondences"

# scene_preview.ply color coding, so the three elements are visually separable.
HUMAN_COLOR = np.array([80, 220, 100], dtype=np.uint8)  # green: SMPL-X body mesh
OBJECT_COLOR = np.array([230, 60, 60], dtype=np.uint8)  # red: tracked object pixels
# Drop scene points beyond this multiple of the human's own depth, so far
# background/sky doesn't dwarf the person and object (no sky mask is available
# in this stage, unlike stage 3's standalone preview).
SCENE_PREVIEW_DEPTH_CLIP_FACTOR = 3.0


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


def _dump_scene_preview(
    smplx_verts: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    scale: float,
    translation: np.ndarray,
    human_mask: np.ndarray,
    object_mask: np.ndarray | None,
    anchor_frame_path: Path,
    out_path: Path,
) -> None:
    """One PLY that puts all three elements in the SMPL-X human's metric space,
    color-coded, so the fitted scale can be eyeballed in Blender: the green
    SMPL-X body mesh should sit inside its own (RGB) depth points, and the red
    object points should land at the hands. The depth cloud is mapped into body
    space via the fitted `(scale, translation)`; the human mesh is already there.
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
    write_colored_ply(points, colors, out_path)


def _load_mask_at_depth_res(masks_path: str, anchor: int, depth_hw: tuple[int, int]) -> np.ndarray:
    packed = torch.load(masks_path, weights_only=False)[KEY_PACKED_MASKS]
    mask_anchor = unpack_masks(packed[anchor])[0].numpy().astype(np.uint8)
    return cv2.resize(mask_anchor, (depth_hw[1], depth_hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def run(progress: ProgressRecord) -> dict[str, str]:
    anchor = progress.scene.anchor_frame_index
    stage_1_outputs = progress.stages[StageName.STAGE_1_MASK_AND_TRACK].outputs

    motion = torch.load(
        progress.stages[StageName.STAGE_2_ESTIMATE_HUMAN_MOTION].outputs[OUTPUT_HUMAN_MOTION],
        weights_only=False,
    )
    smplx_verts = _build_smplx_anchor_mesh(motion[KEY_PRED_SMPL_PARAMS_INCAM], anchor)

    depth = np.load(progress.stages[StageName.STAGE_3_ESTIMATE_DEPTH].outputs[OUTPUT_DEPTH])
    depth_hw = depth.shape
    native_hw = (progress.scene.height, progress.scene.width)
    K = scale_intrinsics_to_resolution(np.array(progress.scene.intrinsics_K), native_hw, depth_hw)

    human_mask = _load_mask_at_depth_res(stage_1_outputs[OUTPUT_HUMAN_MASKS], anchor, depth_hw)

    scale, translation, n_correspondences = fit_scene_scale(smplx_verts, depth, K, human_mask)

    scale_dir = Path(progress.progress_dir) / SCALE_DIRNAME
    scale_dir.mkdir(parents=True, exist_ok=True)
    scene_scale_path = scale_dir / SCENE_SCALE_FILENAME
    scene_scale_path.write_text(
        json.dumps(
            {
                KEY_SCALE: scale,
                KEY_TRANSLATION: translation.tolist(),
                KEY_N_CORRESPONDENCES: n_correspondences,
            },
            indent=2,
        )
    )

    outputs = {OUTPUT_SCENE_SCALE: str(scene_scale_path)}

    if progress.input.dump_scene_preview:
        object_mask = None
        if OUTPUT_OBJECT_MASKS in stage_1_outputs:
            object_mask = _load_mask_at_depth_res(stage_1_outputs[OUTPUT_OBJECT_MASKS], anchor, depth_hw)
        frames_dir = Path(progress.stages[StageName.STAGE_0_INGEST_VIDEO].outputs[FRAMES_DIR_OUTPUT_KEY])
        anchor_frame_path = sorted(frames_dir.glob("*.jpg"))[anchor]
        scene_preview_path = scale_dir / SCENE_PREVIEW_FILENAME
        _dump_scene_preview(
            smplx_verts, depth, K, scale, translation, human_mask, object_mask, anchor_frame_path, scene_preview_path
        )
        outputs[OUTPUT_SCENE_PREVIEW] = str(scene_preview_path)

    return outputs


if __name__ == "__main__":
    cli_entrypoint(run, stage_name=StageName.STAGE_6_ALIGN_SCENE_SCALE)
