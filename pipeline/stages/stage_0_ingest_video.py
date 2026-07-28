"""ingest: reads the input video, extracts frames to disk, and computes camera intrinsics.

The only stage with no dependencies -- everything else in the pipeline builds
on the frames and scene info this produces.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from ..helpers.camera_info_helpers import compute_intrinsics_matrix
from ..helpers.progress_reporter import frame_progress
from ..pipeline_stage_base import cli_entrypoint
from ..progress_tracker import StageName, RunRecord

INPUT_FRAMES_DIRNAME = "input_frames"

# High-quality JPEG rather than lossless PNG: downstream models (SAM 3.1, GVHMR,
# depth estimation) are themselves trained on JPEG-compressed web imagery, so the
# accuracy cost is negligible, and it's a fraction of the disk space/write time.
JPEG_QUALITY = 95

# Phones tag "vertical" recordings with rotation metadata rather than
# re-encoding pixels -- the sensor itself always captures landscape.
# cv2.VideoCapture.read() returns that raw sensor-orientation frame
# regardless of the metadata (CAP_PROP_ORIENTATION_AUTO defaults to off, and
# has a real history of being unreliable across backends/platforms even when
# explicitly enabled), so this stage reads CAP_PROP_ORIENTATION_META itself
# and rotates each frame before writing it out. The exact direction was
# confirmed against a real vertical phone clip by visual inspection, not
# assumed -- metadata value 90 requires ROTATE_90_CLOCKWISE to produce a
# correctly upright frame; getting this backwards would silently produce
# upside-down/mirrored frames that still "look" fixed (right dimensions).
_ROTATION_TO_CV2_CODE: dict[int, int] = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _resolve_rotation(orientation_meta: float, raw_width: int, raw_height: int) -> tuple[int | None, int, int]:
    """`(rotate_code, final_width, final_height)` -- `rotate_code` is a
    `cv2.rotate()` code, or `None` if no rotation is needed. The final
    dimensions already account for the width/height swap a 90/270 rotation
    causes; `compute_intrinsics_matrix` still needs `raw_width` separately
    (the sensor's own native width, not this stage's rotated output) for its
    own focal-length ratio.
    """
    rotation = int(orientation_meta) % 360
    rotate_code = _ROTATION_TO_CV2_CODE.get(rotation)
    if rotation in (90, 270):
        return rotate_code, raw_height, raw_width
    return rotate_code, raw_width, raw_height


def run(runRecord: RunRecord) -> dict[str, str]:
    video_path = runRecord.input.video_path
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    raw_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    rotate_code, width, height = _resolve_rotation(
        capture.get(cv2.CAP_PROP_ORIENTATION_META), raw_width, raw_height
    )

    frames_dir = Path(runRecord.progress_dir) / INPUT_FRAMES_DIRNAME
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    with frame_progress(None, total=total_frames, label=StageName.STAGE_0_INGEST_VIDEO.label) as progress_update:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)
            if frame_count >= progress_update.total:
                progress_update.total = frame_count + 1
                progress_update.refresh()
            frame_path = frames_dir / f"{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            frame_count += 1
            progress_update.update(1)
    capture.release()

    if frame_count == 0:
        raise RuntimeError(f"No frames could be read from video: {video_path}")

    runRecord.scene.fps = fps
    runRecord.scene.width = width
    runRecord.scene.height = height
    runRecord.scene.frame_count = frame_count
    runRecord.scene.intrinsics_K = compute_intrinsics_matrix(
        focal_length_mm=runRecord.input.focal_length_mm,
        sensor_width_mm=runRecord.input.sensor_width_mm,
        sensor_width_px=raw_width,
        image_width_px=width,
        image_height_px=height,
    )

    return {"frames_dir": str(frames_dir)}


if __name__ == "__main__":
    cli_entrypoint(run, stage_name=StageName.STAGE_0_INGEST_VIDEO)
