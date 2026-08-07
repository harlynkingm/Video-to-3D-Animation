"""retarget_hands: attaches HaMeR's per-hand MANO pose (stage 4) onto GVHMR's
SMPL-X body (stage 2), producing one merged full-body-plus-hands motion.

The body's wrist orientation from GVHMR is discarded in favor of HaMeR's, which
sees a zoomed-in crop and estimates the wrist far better than the body-only ViT
can. The reconciliation -- re-expressing HaMeR's global wrist orientation
relative to GVHMR's forearm, and copying the finger articulation across -- lives
in `hand_retarget`. This stage is the I/O around it: load both inputs, run the
retarget, write the merged SMPL-X params that the later HOI stages consume in
place of the body-only motion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..adapters.gvhmr.gvhmr_adapter import (
    KEY_BETAS,
    KEY_BODY_POSE,
    KEY_GLOBAL_ORIENT,
    KEY_PRED_SMPL_PARAMS_INCAM,
    KEY_ROOT_MOTION_UNRELIABLE,
    KEY_TRANSL,
)
from ..adapters.gvhmr.gvhmr_smplx_skeleton import SmplxSkeleton
from ..adapters.hamer.hamer_adapter import (
    KEY_LEFT_GLOBAL_ORIENT,
    KEY_LEFT_HAND_POSE,
    KEY_LEFT_VALID,
    KEY_LEFT_WRIST_VALID,
    KEY_RIGHT_GLOBAL_ORIENT,
    KEY_RIGHT_HAND_POSE,
    KEY_RIGHT_VALID,
    KEY_RIGHT_WRIST_VALID,
)
from ..algorithms.hand_retarget import retarget_hands
from ..pipeline_stage_base import cli_entrypoint
from ..progress_tracker import RunRecord, StageName
from ..stages.stage_2_estimate_human_motion import OUTPUT_HUMAN_MOTION
from ..stages.stage_4_estimate_hands import OUTPUT_HAND_POSE

from ..helpers.progress_reporter import report_single_shot

RETARGET_DIRNAME = f"stage{StageName.STAGE_5_RETARGET_HANDS.stage_number}_retarget"
RETARGET_MOTION_FILENAME = "retargeted_motion.pt"
RETARGET_MOTION_NPZ_FILENAME = "retargeted_motion.npz"
RETARGET_PREVIEW_FILENAME = "retarget_preview.bvh"

# This stage's own progress.json output keys.
OUTPUT_RETARGET_MOTION = "retarget_motion"
OUTPUT_RETARGET_MOTION_NPZ = "retarget_motion_npz"
OUTPUT_RETARGET_PREVIEW = "retarget_preview"

# The merged SMPL-X params are keyed by smplx.forward's own kwarg names (the same
# convention stage 6 uses) so a downstream SMPL-X build can splat them directly:
# the body keys are reused verbatim from GVHMR's output, the hand keys added.


def _as_f32_tensor(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x.detach().cpu().float()


def run(runRecord: RunRecord) -> dict[str, str]:
    motion = torch.load(
        runRecord.stages[StageName.STAGE_2_ESTIMATE_HUMAN_MOTION].outputs[OUTPUT_HUMAN_MOTION],
        weights_only=False,
    )
    incam = motion[KEY_PRED_SMPL_PARAMS_INCAM]
    global_orient = _as_f32_tensor(incam[KEY_GLOBAL_ORIENT])  # (F, 3)
    body_pose = _as_f32_tensor(incam[KEY_BODY_POSE])  # (F, 63)
    betas = _as_f32_tensor(incam[KEY_BETAS])
    transl = _as_f32_tensor(incam[KEY_TRANSL])
    # Pure passthrough -- unrelated to hand retargeting, but export (stage 10)
    # needs it and reads this stage's own npz output, not stage 2's directly.
    root_motion_unreliable = motion[KEY_ROOT_MOTION_UNRELIABLE]  # (F,) bool

    hands = np.load(runRecord.stages[StageName.STAGE_4_ESTIMATE_HANDS].outputs[OUTPUT_HAND_POSE])
    if hands[KEY_LEFT_VALID].shape[0] != global_orient.shape[0]:
        raise RuntimeError(
            f"frame count mismatch: body motion has {global_orient.shape[0]} frames, "
            f"hand pose has {hands[KEY_LEFT_VALID].shape[0]}"
        )

    with report_single_shot(StageName.STAGE_5_RETARGET_HANDS.label):
        merged_body_pose, left_hand_pose, right_hand_pose = retarget_hands(
            global_orient=global_orient,
            body_pose=body_pose,
            parents=SmplxSkeleton().parents,
            left_wrist_global=_as_f32_tensor(hands[KEY_LEFT_GLOBAL_ORIENT]),
            right_wrist_global=_as_f32_tensor(hands[KEY_RIGHT_GLOBAL_ORIENT]),
            left_hand_pose=_as_f32_tensor(hands[KEY_LEFT_HAND_POSE]),
            right_hand_pose=_as_f32_tensor(hands[KEY_RIGHT_HAND_POSE]),
            left_wrist_valid=torch.from_numpy(hands[KEY_LEFT_WRIST_VALID]),
            right_wrist_valid=torch.from_numpy(hands[KEY_RIGHT_WRIST_VALID]),
            left_finger_valid=torch.from_numpy(hands[KEY_LEFT_VALID]),
            right_finger_valid=torch.from_numpy(hands[KEY_RIGHT_VALID]),
        )

    merged = {
        KEY_GLOBAL_ORIENT: global_orient,
        KEY_BODY_POSE: merged_body_pose,
        KEY_BETAS: betas,
        KEY_TRANSL: transl,
        KEY_LEFT_HAND_POSE: left_hand_pose,
        KEY_RIGHT_HAND_POSE: right_hand_pose,
        KEY_ROOT_MOTION_UNRELIABLE: root_motion_unreliable,
    }

    retarget_dir = Path(runRecord.progress_dir) / RETARGET_DIRNAME
    retarget_dir.mkdir(parents=True, exist_ok=True)
    motion_path = retarget_dir / RETARGET_MOTION_FILENAME
    torch.save(merged, motion_path)

    # A plain-numpy companion to the .pt above -- not a preview, a real
    # interchange format for the one downstream consumer (export) that
    # runs in a separate environment with no torch installed at all.
    motion_npz_path = retarget_dir / RETARGET_MOTION_NPZ_FILENAME
    np.savez(motion_npz_path, **{key: value.numpy() for key, value in merged.items()})

    outputs = {
        OUTPUT_RETARGET_MOTION: str(motion_path),
        OUTPUT_RETARGET_MOTION_NPZ: str(motion_npz_path),
    }

    if runRecord.input.render_retarget_preview:
        from ..helpers.smplx_bvh_preview import render_body_hands_bvh

        preview_path = retarget_dir / RETARGET_PREVIEW_FILENAME
        render_body_hands_bvh(
            global_orient, merged_body_pose, left_hand_pose, right_hand_pose, transl, runRecord.scene.fps, preview_path
        )
        outputs[OUTPUT_RETARGET_PREVIEW] = str(preview_path)

    return outputs


if __name__ == "__main__":
    cli_entrypoint(run, stage_name=StageName.STAGE_5_RETARGET_HANDS)
