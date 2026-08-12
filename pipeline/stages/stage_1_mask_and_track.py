"""mask_and_track: tracks the human (and optional object) across the whole clip
using SAM 3.1, text-prompted, no manual clicks or boxes.

Also resolves `scene.anchor_frame_index`: the frame later stages use for
object-shape fitting and scale alignment. Uses the user's override if given,
otherwise the frame where the tracked object's mask has the largest area (a
simple, real heuristic, a refined version that also penalizes occlusion/
fragmentation would be a reasonable future improvement, but isn't needed yet).

If `RunInput.render_mask_previews` is set, also writes one black/white JPEG per
frame per tracked entity (white = masked region), a quick way to eyeball
whether SAM 3.1 actually tracked the right thing, without writing a separate
viewer. Off by default: it roughly doubles this stage's disk writes and isn't
needed once tracking quality on a given prompt/video is already trusted.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from ..adapters.sam31.sam31_adapter import KEY_HUMAN, KEY_OBJECT, Sam31Adapter
from ..adapters.sam31.sam31_tracker import KEY_PACKED_MASKS, unpack_masks
from ..pipeline_stage_base import cli_entrypoint
from ..progress_tracker import RunRecord, StageName

MASKS_DIRNAME = f"stage{StageName.STAGE_1_MASK_AND_TRACK.stage_number}_masks"
HUMAN_MASKS_FILENAME = "human.pt"
OBJECT_MASKS_FILENAME = "object.pt"
HUMAN_PREVIEW_DIRNAME = "preview_human"
OBJECT_PREVIEW_DIRNAME = "preview_object"

# stage_0_ingest_video.py's own output key, consumed here.
FRAMES_DIR_OUTPUT_KEY = "frames_dir"

# This stage's own progress.json output keys.
OUTPUT_HUMAN_MASKS = "human_masks"
OUTPUT_OBJECT_MASKS = "object_masks"
OUTPUT_HUMAN_MASKS_PREVIEW = "human_masks_preview"
OUTPUT_OBJECT_MASKS_PREVIEW = "object_masks_preview"


def _resolve_anchor_frame(object_result: dict | None) -> int:
    """Frame index with the largest tracked-object mask area, or 0 if there's no
    object (or it was never detected), see this module's docstring.
    """
    if object_result is None or object_result[KEY_PACKED_MASKS] is None:
        return 0
    areas = unpack_masks(object_result[KEY_PACKED_MASKS]).sum(dim=(-1, -2)).squeeze(1)
    return int(areas.argmax().item())


def _render_mask_previews(packed_masks: torch.Tensor, out_dir: Path, native_hw: tuple[int, int]) -> None:
    """One black/white JPEG per frame (white = the first tracked object's masked
    region, this project only ever tracks one instance per prompt, see
    `sam31_adapter.py`'s module docstring), resized from SAM 3.1's own fixed
    working resolution to the source video's actual resolution (`native_hw`) so
    previews aren't stretched to a square aspect ratio for widescreen video.
    Unpacks and resizes one frame at a time rather than the whole clip at once,
    so this stays cheap in RAM even on a long clip.

    Clears out `out_dir` first: a re-run (e.g. after changing the input video
    via `--force`) can produce a different frame count than a previous run,
    which would otherwise leave stale images past the new count sitting
    alongside the current ones.
    """
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    native_h, native_w = native_hw
    for i in range(packed_masks.shape[0]):
        mask = unpack_masks(packed_masks[i])[0]  # (H, W) bool, SAM 3.1's own working resolution
        image = mask.numpy().astype(np.uint8) * 255
        image = cv2.resize(image, (native_w, native_h), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(out_dir / f"{i:06d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 90])


def run(runRecord: RunRecord) -> dict[str, str]:
    frames_dir = Path(runRecord.stages[StageName.STAGE_0_INGEST_VIDEO].outputs[FRAMES_DIR_OUTPUT_KEY])
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")

    adapter = Sam31Adapter()
    adapter.load()
    try:
        result = adapter.infer(
            frame_paths,
            human_prompt=runRecord.input.human_prompt,
            object_prompt=runRecord.input.object_prompt,
            max_bridge_frames=runRecord.input.sam_track_max_bridge_frames,
        )
    finally:
        adapter.unload()

    if result[KEY_HUMAN][KEY_PACKED_MASKS] is None:
        raise RuntimeError(f"human_prompt {runRecord.input.human_prompt!r} was never detected in this clip")

    masks_dir = Path(runRecord.progress_dir) / MASKS_DIRNAME
    masks_dir.mkdir(parents=True, exist_ok=True)

    native_hw = (runRecord.scene.height, runRecord.scene.width)

    outputs = {}
    human_path = masks_dir / HUMAN_MASKS_FILENAME
    torch.save(result[KEY_HUMAN], human_path)
    outputs[OUTPUT_HUMAN_MASKS] = str(human_path)
    if runRecord.input.render_mask_previews:
        _render_mask_previews(result[KEY_HUMAN][KEY_PACKED_MASKS], masks_dir / HUMAN_PREVIEW_DIRNAME, native_hw)
        outputs[OUTPUT_HUMAN_MASKS_PREVIEW] = str(masks_dir / HUMAN_PREVIEW_DIRNAME)

    if result[KEY_OBJECT] is not None and result[KEY_OBJECT][KEY_PACKED_MASKS] is not None:
        object_path = masks_dir / OBJECT_MASKS_FILENAME
        torch.save(result[KEY_OBJECT], object_path)
        outputs[OUTPUT_OBJECT_MASKS] = str(object_path)
        if runRecord.input.render_mask_previews:
            _render_mask_previews(result[KEY_OBJECT][KEY_PACKED_MASKS], masks_dir / OBJECT_PREVIEW_DIRNAME, native_hw)
            outputs[OUTPUT_OBJECT_MASKS_PREVIEW] = str(masks_dir / OBJECT_PREVIEW_DIRNAME)

    if runRecord.input.anchor_frame_override is not None:
        runRecord.scene.anchor_frame_index = runRecord.input.anchor_frame_override
    else:
        runRecord.scene.anchor_frame_index = _resolve_anchor_frame(result[KEY_OBJECT])

    return outputs


if __name__ == "__main__":
    cli_entrypoint(run, stage_name=StageName.STAGE_1_MASK_AND_TRACK)
