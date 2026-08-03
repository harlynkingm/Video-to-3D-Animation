"""estimate_human_motion: runs GVHMR on the tracked human to produce a
full-clip SMPL-X body pose, in both camera-space ("incam") and
world-grounded ("global") coordinates.

Depends on stage 1's human mask (for the per-frame bbox GVHMR needs), not
just stage 0's frames -- see `gvhmr_adapter.py`'s module docstring for why no
mask-to-video conversion is needed anywhere in this step.

If `RunInput.render_motion_preview` is set, also writes an AMASS-format `.npz` of
the world-grounded ("global") motion -- importable into Blender via the
already-installed `jtesch/smplx_blender_addon`'s own "Add Animation" operator
(`anim_format="AMASS"`) for visual verification against a real, correctly-shaped
and -posed SMPL-X body, not just raw numbers. This is not a from-scratch export
format: it's the exact input format that operator (and eventually this
project's own export stage) already expects, so writing it here is a small
step, not new infrastructure.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..adapters.gvhmr.gvhmr_adapter import (
    KEY_BETAS,
    KEY_BODY_POSE,
    KEY_GLOBAL_ORIENT,
    KEY_PRED_SMPL_PARAMS_GLOBAL,
    KEY_PRED_SMPL_PARAMS_INCAM,
    KEY_TRANSL,
    KEY_TRANSL_INCAM_RAW,
    GVHMRAdapter,
)
from ..adapters.sam31.sam31_tracker import KEY_PACKED_MASKS, unpack_masks
from ..algorithms.motion_smoothing import smooth_rotation_sequence, smooth_translation_sequence
from ..helpers.amass_export_helper import write_amass_npz
from ..pipeline_stage_base import cli_entrypoint
from ..progress_tracker import RunRecord, StageName
from .stage_1_mask_and_track import OUTPUT_HUMAN_MASKS

MOTION_DIRNAME = f"stage{StageName.STAGE_2_ESTIMATE_HUMAN_MOTION.stage_number}_motion"
HUMAN_MOTION_FILENAME = "human_motion.pt"
MOTION_PREVIEW_FILENAME = "blender_preview.npz"

# stage_0_ingest_video.py's own output key, consumed here.
FRAMES_DIR_OUTPUT_KEY = "frames_dir"

# This stage's own progress.json output keys.
OUTPUT_HUMAN_MOTION = "human_motion"
OUTPUT_MOTION_PREVIEW = "motion_preview"


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


def run(runRecord: RunRecord) -> dict[str, str]:
    frames_dir = Path(runRecord.stages[StageName.STAGE_0_INGEST_VIDEO].outputs[FRAMES_DIR_OUTPUT_KEY])
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")

    human_masks_path = Path(runRecord.stages[StageName.STAGE_1_MASK_AND_TRACK].outputs[OUTPUT_HUMAN_MASKS])
    human_result = torch.load(human_masks_path, weights_only=False)
    masks = unpack_masks(human_result[KEY_PACKED_MASKS]).squeeze(1)  # (N, H, W) bool

    K_fullimg = torch.tensor(runRecord.scene.intrinsics_K)

    adapter = GVHMRAdapter()
    adapter.load()
    try:
        result = adapter.infer(frame_paths, masks, K_fullimg)
    finally:
        adapter.unload()

    # Smooth both coordinate frames (incam feeds stage 5/6, global feeds the
    # eventual export + this stage's Blender preview) before anything reads them.
    window = runRecord.input.body_smoothing_window
    cutoff = runRecord.input.body_translation_cutoff
    _smooth_body_params(result[KEY_PRED_SMPL_PARAMS_INCAM], window, cutoff)
    _smooth_body_params(result[KEY_PRED_SMPL_PARAMS_GLOBAL], window, cutoff)

    # Same translation smoothing as the corrected incam transl above (just not
    # the foot-lock drift correction, which this value is deliberately kept
    # free of -- see gvhmr_adapter.KEY_TRANSL_INCAM_RAW's own comment) so
    # stage 7 sees a signal that's smoothed the same way it always has been,
    # not a differently-conditioned one.
    raw_transl = result[KEY_TRANSL_INCAM_RAW]
    result[KEY_TRANSL_INCAM_RAW] = torch.from_numpy(
        smooth_translation_sequence(raw_transl.detach().cpu().numpy(), cutoff)
    ).to(raw_transl.dtype)

    motion_dir = Path(runRecord.progress_dir) / MOTION_DIRNAME
    motion_dir.mkdir(parents=True, exist_ok=True)
    motion_path = motion_dir / HUMAN_MOTION_FILENAME
    torch.save(result, motion_path)
    outputs = {OUTPUT_HUMAN_MOTION: str(motion_path)}

    if runRecord.input.render_motion_preview:
        preview_path = motion_dir / MOTION_PREVIEW_FILENAME
        # *global* (world-grounded) pose, not *incam* -- this artifact is for
        # standing a character up in Blender's own world space, which is what
        # GVHMR's "global" frame is for; `incam` is for this project's own
        # later depth/scale alignment, a different consumer with a different
        # need. No hand pose yet at this point in the pipeline (left/right
        # default to flat/neutral).
        global_params = result[KEY_PRED_SMPL_PARAMS_GLOBAL]
        write_amass_npz(
            global_orient=global_params[KEY_GLOBAL_ORIENT].numpy(),
            body_pose=global_params[KEY_BODY_POSE].numpy(),
            betas=global_params[KEY_BETAS][0].numpy(),
            transl=global_params[KEY_TRANSL].numpy(),
            fps=runRecord.scene.fps,
            out_path=preview_path,
        )
        outputs[OUTPUT_MOTION_PREVIEW] = str(preview_path)

    return outputs


if __name__ == "__main__":
    cli_entrypoint(run, stage_name=StageName.STAGE_2_ESTIMATE_HUMAN_MOTION)
