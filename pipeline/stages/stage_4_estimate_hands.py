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
from ..algorithms.motion_smoothing import smooth_rotation_sequence
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


def _smooth_hand_result(result: dict, window: int) -> None:
    """In place: temporally smooth each hand's finger articulation + wrist
    orientation. This is the stage that most needs it -- HaMeR runs per-frame
    with no temporal model, so its raw hands are far jitterier than GVHMR's body.

    Occlusion handling: `smooth_rotation_sequence`'s validity-aware gap fill
    already gives the right behavior for free, via `np.interp`'s own boundary
    semantics -- an *interior* occlusion (the hand is later detected again) is
    linearly interpolated between the last-seen and next-seen pose; a *trailing*
    (or leading) occlusion that never recovers has no second endpoint to
    interpolate toward, so it freezes at the nearest real detection instead. The
    saved `*_valid` flags are left untouched -- still the literal, honest
    per-frame HaMeR-detection record for any downstream consumer -- only the
    pose arrays themselves are filled in for invalid frames rather than left at
    a raw zero placeholder."""
    for pose_key, global_orient_key, valid_key in (
        (KEY_LEFT_HAND_POSE, KEY_LEFT_GLOBAL_ORIENT, KEY_LEFT_VALID),
        (KEY_RIGHT_HAND_POSE, KEY_RIGHT_GLOBAL_ORIENT, KEY_RIGHT_VALID),
    ):
        valid = result[valid_key]
        for rotation_key in (pose_key, global_orient_key):
            result[rotation_key] = smooth_rotation_sequence(result[rotation_key], window, valid=valid)


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

    _smooth_hand_result(result, progress.input.hand_smoothing_window)

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
