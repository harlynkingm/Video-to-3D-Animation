"""estimate_human_motion: runs GVHMR on the tracked human to produce a
full-clip SMPL-X body pose, in both camera-space ("incam") and
world-grounded ("global") coordinates.

Depends on stage 1's human mask (for the per-frame bbox GVHMR needs), not
just stage 0's frames -- see `gvhmr_adapter.py`'s module docstring for why no
mask-to-video conversion is needed anywhere in this step.

If `RunInput.dump_motion_preview` is set, also writes an AMASS-format `.npz` of
the world-grounded ("global") motion -- importable into Blender via the
already-installed `jtesch/smplx_blender_addon`'s own "Add Animation" operator
(`anim_format="AMASS"`) for visual verification against a real, correctly-shaped
and -posed SMPL-X body, not just raw numbers. This is not a from-scratch export
format: it's the exact input format that operator (and eventually this
project's own FBX-export stage) already expects, so writing it here is a small
step, not new infrastructure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..adapters.gvhmr.gvhmr_adapter import (
    KEY_BETAS,
    KEY_BODY_POSE,
    KEY_GLOBAL_ORIENT,
    KEY_PRED_SMPL_PARAMS_GLOBAL,
    KEY_PRED_SMPL_PARAMS_INCAM,
    KEY_TRANSL,
    GVHMRAdapter,
)
from ..adapters.sam31.sam31_tracker import KEY_PACKED_MASKS, unpack_masks
from ..algorithms.motion_smoothing import smooth_rotation_sequence, smooth_translation_sequence
from ..pipeline_stage_base import cli_entrypoint
from ..progress_tracker import ProgressRecord, StageName
from .stage_1_mask_and_track import OUTPUT_HUMAN_MASKS

MOTION_DIRNAME = "motion"
HUMAN_MOTION_FILENAME = "human_motion.pt"
MOTION_PREVIEW_FILENAME = "blender_preview.npz"

# stage_0_ingest_video.py's own output key, consumed here.
FRAMES_DIR_OUTPUT_KEY = "frames_dir"

# This stage's own progress.json output keys.
OUTPUT_HUMAN_MOTION = "human_motion"
OUTPUT_MOTION_PREVIEW = "motion_preview"

# SMPL-X joint layout the AMASS `poses` array must match (see the addon's own
# `utils/model_spec.py`): 22 body joints (pelvis + 21, all GVHMR predicts) + jaw
# + 2 eyes + 15+15 hand joints = 55 joints * 3 axis-angle values = 165.
AMASS_POSE_DIM = 55 * 3
_GVHMR_POSE_DIM = 3 + 63  # global_orient + body_pose -- everything GVHMR itself predicts


def _dump_amass_npz(pred_smpl_params_global: dict, fps: float, out_path: Path) -> None:
    """Writes GVHMR's world-grounded pose as an AMASS `.npz`. Hand and face
    joints are left at zero (flat/neutral) -- GVHMR only ever predicts body
    pose; real hand pose is a later, not-yet-built stage (`retarget_hands`).
    Gender is always "neutral", matching the one SMPL-X body model
    (`SMPLX_NEUTRAL.npz`) this project's math already uses throughout.

    Uses the *global* (world-grounded) pose, not *incam* -- this artifact is
    for standing a character up in Blender's own world space, which is what
    GVHMR's "global" frame is for; `incam` is for this project's own later
    depth/scale alignment, a different consumer with a different need.
    """
    body_pose = pred_smpl_params_global[KEY_BODY_POSE].numpy()
    global_orient = pred_smpl_params_global[KEY_GLOBAL_ORIENT].numpy()
    transl = pred_smpl_params_global[KEY_TRANSL].numpy()
    betas = pred_smpl_params_global[KEY_BETAS][0].numpy()  # pooled across the clip, identical every frame

    num_frames = body_pose.shape[0]
    hand_and_face = np.zeros((num_frames, AMASS_POSE_DIM - _GVHMR_POSE_DIM), dtype=np.float32)
    poses = np.concatenate([global_orient, body_pose, hand_and_face], axis=-1).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        trans=transl.astype(np.float32),
        gender=np.array("neutral"),
        mocap_frame_rate=np.array(round(fps)),
        betas=betas.astype(np.float32),
        poses=poses,
    )


def _smooth_body_params(params: dict, window: int, cutoff: float) -> None:
    """In place: temporally smooth one GVHMR param sub-dict's rotation
    (global_orient + body_pose) and translation. betas is a shape parameter, not
    motion, so it is left untouched. Always on -- GVHMR's raw output still has
    residual jitter its temporal transformer doesn't fully remove."""
    for rotation_key in (KEY_GLOBAL_ORIENT, KEY_BODY_POSE):
        original = params[rotation_key]
        smoothed = smooth_rotation_sequence(original.detach().cpu().numpy(), window)
        params[rotation_key] = torch.from_numpy(smoothed).to(original.dtype)
    original = params[KEY_TRANSL]
    smoothed = smooth_translation_sequence(original.detach().cpu().numpy(), cutoff)
    params[KEY_TRANSL] = torch.from_numpy(smoothed).to(original.dtype)


def run(progress: ProgressRecord) -> dict[str, str]:
    frames_dir = Path(progress.stages[StageName.STAGE_0_INGEST_VIDEO].outputs[FRAMES_DIR_OUTPUT_KEY])
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")

    human_masks_path = Path(progress.stages[StageName.STAGE_1_MASK_AND_TRACK].outputs[OUTPUT_HUMAN_MASKS])
    human_result = torch.load(human_masks_path, weights_only=False)
    masks = unpack_masks(human_result[KEY_PACKED_MASKS]).squeeze(1)  # (N, H, W) bool

    K_fullimg = torch.tensor(progress.scene.intrinsics_K)

    adapter = GVHMRAdapter()
    adapter.load()
    try:
        result = adapter.infer(frame_paths, masks, K_fullimg)
    finally:
        adapter.unload()

    # Smooth both coordinate frames (incam feeds stage 5/6, global feeds the
    # eventual export + this stage's Blender preview) before anything reads them.
    window = progress.input.body_smoothing_window
    cutoff = progress.input.body_translation_cutoff
    _smooth_body_params(result[KEY_PRED_SMPL_PARAMS_INCAM], window, cutoff)
    _smooth_body_params(result[KEY_PRED_SMPL_PARAMS_GLOBAL], window, cutoff)

    motion_dir = Path(progress.progress_dir) / MOTION_DIRNAME
    motion_dir.mkdir(parents=True, exist_ok=True)
    motion_path = motion_dir / HUMAN_MOTION_FILENAME
    torch.save(result, motion_path)
    outputs = {OUTPUT_HUMAN_MOTION: str(motion_path)}

    if progress.input.dump_motion_preview:
        preview_path = motion_dir / MOTION_PREVIEW_FILENAME
        _dump_amass_npz(result[KEY_PRED_SMPL_PARAMS_GLOBAL], progress.scene.fps, preview_path)
        outputs[OUTPUT_MOTION_PREVIEW] = str(preview_path)

    return outputs


if __name__ == "__main__":
    cli_entrypoint(run, stage_name=StageName.STAGE_2_ESTIMATE_HUMAN_MOTION)
