"""ingest: reads the input video (or a pre-extracted image-sequence folder),
extracts frames to disk, and computes camera intrinsics.

The only stage with no dependencies, everything else in the pipeline builds
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

# `--input-video` accepts a directory of these instead of a video file, see
# `_ingest_image_folder`. Case-insensitive (`Path.suffix` preserves case).
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


def _ingest_video(video_path: str, frames_dir: Path) -> tuple[float, int, int, int, int]:
    """`(fps, raw_width, width, height, frame_count)`.

    `raw_width` is the encoded sensor-frame width, captured while OpenCV's
    display-orientation transform is disabled. It is deliberately kept
    separate from `width`: a vertical phone recording may decode to an
    upright portrait frame whose horizontal axis was the sensor's vertical
    axis, while focal length in pixels is still measured against the encoded
    sensor width.
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # FFmpeg/OpenCV can apply a video's display-orientation metadata itself.
    # Read the encoded dimensions with that transform explicitly off, then
    # explicitly enable it for decoding. Relying on the backend default
    # caused a manual rotation here to be applied twice for portrait iOS MOVs.
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    raw_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    width = height = 0
    frame_count = 0
    with frame_progress(None, total=total_frames, label=StageName.STAGE_0_INGEST_VIDEO.label) as progress_update:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_count == 0:
                # The decoded frame, rather than a backend property, is the
                # authoritative output resolution. It is what every
                # downstream stage will actually consume from input_frames.
                height, width = frame.shape[:2]
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
    return fps, raw_width, width, height, frame_count


def _ingest_image_folder(folder: Path, fps: float, frames_dir: Path) -> tuple[float, int, int, int, int]:
    """`(fps, raw_width, width, height, frame_count)`, `fps` is just
    `RunInput.source_fps` passed back through (an image sequence has no
    embedded frame rate the way a video container does; `validate_video_input`
    enforces it's set before this stage ever runs). No rotation handling:
    unlike a phone video, these are already-extracted or individually
    authored images with no comparable sensor-orientation metadata to correct
    for. `raw_width` equals `width` here, there's no separate pre-rotation
    sensor dimension to track."""
    image_paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise RuntimeError(f"No JPEG/PNG images found in {folder}")

    width = height = 0
    frame_count = 0
    for frame_count, image_path in enumerate(
        frame_progress(image_paths, total=len(image_paths), label=StageName.STAGE_0_INGEST_VIDEO.label)
    ):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        if frame_count == 0:
            height, width = frame.shape[:2]
        frame_path = frames_dir / f"{frame_count:06d}.jpg"
        if image_path.resolve() == frame_path.resolve():
            continue
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return fps, width, width, height, frame_count + 1


def run(runRecord: RunRecord) -> dict[str, str]:
    video_path = runRecord.input.video_path
    frames_dir = Path(runRecord.progress_dir) / INPUT_FRAMES_DIRNAME
    frames_dir.mkdir(parents=True, exist_ok=True)

    if Path(video_path).is_dir():
        if runRecord.input.source_fps is None or runRecord.input.source_fps <= 0:
            # Normally caught earlier by validate_video_input (create_run.py/
            # pipeline.run.py's own main()s), this is a defensive backstop
            # for direct run() calls (tests, a resumed run whose progress.json
            # predates that validator) that skip the CLI validation path.
            raise RuntimeError("--source-fps is required when --input-video is a directory of images")
        fps, raw_width, width, height, frame_count = _ingest_image_folder(
            Path(video_path), runRecord.input.source_fps, frames_dir
        )
    else:
        fps, raw_width, width, height, frame_count = _ingest_video(video_path, frames_dir)

    runRecord.scene.fps = fps
    runRecord.scene.width = width
    runRecord.scene.height = height
    runRecord.scene.frame_count = frame_count
    if runRecord.input.intrinsics_k is not None:
        # Real calibration data, given as-is, see RunInput.intrinsics_k's
        # own comment for why this is preferred over the lens-spec path when
        # both are available (a real K's principal point isn't necessarily
        # centered the way compute_intrinsics_matrix has to assume).
        runRecord.scene.intrinsics_K = runRecord.input.intrinsics_k
    else:
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
