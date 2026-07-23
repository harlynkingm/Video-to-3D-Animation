"""ingest: reads the input video, extracts frames to disk, and computes camera intrinsics.

The only stage with no dependencies -- everything else in the pipeline builds
on the frames and scene info this produces.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from ..helpers.camera_info_helpers import compute_intrinsics_matrix
from ..pipeline_stage_base import cli_entrypoint
from ..progress_tracker import StageName, ProgressRecord

# High-quality JPEG rather than lossless PNG: downstream models (SAM 3.1, GVHMR,
# depth estimation) are themselves trained on JPEG-compressed web imagery, so the
# accuracy cost is negligible, and it's a fraction of the disk space/write time.
JPEG_QUALITY = 95


def run(progress: ProgressRecord) -> dict[str, str]:
    video_path = progress.input.video_path
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_dir = Path(progress.progress_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        frame_path = frames_dir / f"{frame_count:06d}.jpg"
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        frame_count += 1
    capture.release()

    if frame_count == 0:
        raise RuntimeError(f"No frames could be read from video: {video_path}")

    progress.scene.fps = fps
    progress.scene.width = width
    progress.scene.height = height
    progress.scene.frame_count = frame_count
    progress.scene.intrinsics_K = compute_intrinsics_matrix(
        focal_length_mm=progress.input.focal_length_mm,
        sensor_width_mm=progress.input.sensor_width_mm,
        image_width_px=width,
        image_height_px=height,
    )

    return {"frames_dir": str(frames_dir)}


if __name__ == "__main__":
    cli_entrypoint(run, stage_name=StageName.STAGE_0_INGEST_VIDEO)
