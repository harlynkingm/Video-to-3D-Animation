"""estimate_hands: runs HaMeR on every frame to recover per-frame MANO hand
pose for both hands, using the SAM 3.1 human mask (stage 1) to locate the
person and our COCO-17 ViTPose to locate each hand.

Output is the *raw* per-hand MANO pose (finger articulation + wrist
orientation, both in each hand crop's camera frame) plus a per-frame validity
flag (a hand may be off-screen or too occluded to detect). Grafting these onto
GVHMR's body -- the wrist/forearm reconciliation -- is stage 5's job
(retarget_hands), not this one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..adapters.hamer.hamer_adapter import (
    HamerAdapter,
    KEY_LEFT_GLOBAL_ORIENT,
    KEY_LEFT_HAND_POSE,
    KEY_LEFT_VALID,
    KEY_RIGHT_GLOBAL_ORIENT,
    KEY_RIGHT_HAND_POSE,
    KEY_RIGHT_VALID,
)
from ..pipeline_stage_base import cli_entrypoint
from ..progress_tracker import ProgressRecord, StageName
from ..stages.stage_1_mask_and_track import OUTPUT_HUMAN_MASKS

# stage_0_ingest_video.py's own output key, consumed here.
FRAMES_DIR_OUTPUT_KEY = "frames_dir"

HANDS_DIRNAME = "hands"
HAND_POSE_FILENAME = "hand_pose.npz"
HANDS_PREVIEW_FILENAME = "hands_preview.bvh"

# This stage's own progress.json output keys.
OUTPUT_HAND_POSE = "hand_pose"
OUTPUT_HANDS_PREVIEW = "hands_preview"


def run(progress: ProgressRecord) -> dict[str, str]:
    frames_dir = Path(progress.stages[StageName.STAGE_0_INGEST_VIDEO].outputs[FRAMES_DIR_OUTPUT_KEY])
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")

    human_masks = torch.load(
        progress.stages[StageName.STAGE_1_MASK_AND_TRACK].outputs[OUTPUT_HUMAN_MASKS],
        weights_only=False,
    )

    adapter = HamerAdapter()
    adapter.load()
    try:
        result = adapter.infer(frame_paths, human_masks)
    finally:
        adapter.unload()

    hands_dir = Path(progress.progress_dir) / HANDS_DIRNAME
    hands_dir.mkdir(parents=True, exist_ok=True)
    hand_pose_path = hands_dir / HAND_POSE_FILENAME
    np.savez(
        hand_pose_path,
        **{
            KEY_LEFT_HAND_POSE: result[KEY_LEFT_HAND_POSE],
            KEY_RIGHT_HAND_POSE: result[KEY_RIGHT_HAND_POSE],
            KEY_LEFT_GLOBAL_ORIENT: result[KEY_LEFT_GLOBAL_ORIENT],
            KEY_RIGHT_GLOBAL_ORIENT: result[KEY_RIGHT_GLOBAL_ORIENT],
            KEY_LEFT_VALID: result[KEY_LEFT_VALID],
            KEY_RIGHT_VALID: result[KEY_RIGHT_VALID],
        },
    )

    outputs = {OUTPUT_HAND_POSE: str(hand_pose_path)}

    if progress.input.dump_hands_preview:
        from ..adapters.hamer.hamer_bvh_preview import dump_hands_bvh

        preview_path = hands_dir / HANDS_PREVIEW_FILENAME
        dump_hands_bvh(result, progress.scene.fps, preview_path)
        outputs[OUTPUT_HANDS_PREVIEW] = str(preview_path)

    return outputs


if __name__ == "__main__":
    cli_entrypoint(run, stage_name=StageName.STAGE_4_ESTIMATE_HANDS)
