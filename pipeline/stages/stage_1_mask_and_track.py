"""mask_and_track: tracks the human (and optional object) across the whole clip
using SAM 3.1, text-prompted -- no manual clicks or boxes.

Also resolves `scene.anchor_frame_index`: the frame later stages use for
object-shape fitting and scale alignment. Uses the user's override if given,
otherwise the frame where the tracked object's mask has the largest area (a
simple, real heuristic -- a refined version that also penalizes occlusion/
fragmentation is proposed but not yet needed, see
docs/PROGRESS_SCHEMA.md/project_architecture_pending_review memory).
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..adapters.sam31.sam31_adapter import Sam31Adapter
from ..adapters.sam31.sam31_tracker import unpack_masks
from ..pipeline_stage_base import cli_entrypoint
from ..progress_tracker import ProgressRecord, StageName

MASKS_DIRNAME = "masks"


def _resolve_anchor_frame(object_result: dict | None) -> int:
    """Frame index with the largest tracked-object mask area, or 0 if there's no
    object (or it was never detected) -- see this module's docstring.
    """
    if object_result is None or object_result["packed_masks"] is None:
        return 0
    areas = unpack_masks(object_result["packed_masks"]).sum(dim=(-1, -2)).squeeze(1)
    return int(areas.argmax().item())


def run(progress: ProgressRecord) -> dict[str, str]:
    frames_dir = Path(progress.stages[StageName.STAGE_0_INGEST_VIDEO].outputs["frames_dir"])
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")

    adapter = Sam31Adapter()
    adapter.load()
    try:
        result = adapter.infer(
            frame_paths,
            human_prompt=progress.input.human_prompt,
            object_prompt=progress.input.object_prompt
        )
    finally:
        adapter.unload()

    if result["human"]["packed_masks"] is None:
        raise RuntimeError(f"human_prompt {progress.input.human_prompt!r} was never detected in this clip")

    masks_dir = Path(progress.progress_dir) / MASKS_DIRNAME
    masks_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    human_path = masks_dir / "human.pt"
    torch.save(result["human"], human_path)
    outputs["human_masks"] = str(human_path)

    if result["object"] is not None and result["object"]["packed_masks"] is not None:
        object_path = masks_dir / "object.pt"
        torch.save(result["object"], object_path)
        outputs["object_masks"] = str(object_path)

    if progress.input.anchor_frame_override is not None:
        progress.scene.anchor_frame_index = progress.input.anchor_frame_override
    else:
        progress.scene.anchor_frame_index = _resolve_anchor_frame(result["object"])

    return outputs


if __name__ == "__main__":
    cli_entrypoint(run, stage_name=StageName.STAGE_1_MASK_AND_TRACK)
