"""Stage 0 regression test: extracts frames from the small test clip and
computes camera intrinsics from real (tiny) input data. Runs anywhere, no
GPU or checkpoints needed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import FOCAL_LENGTH_MM, SENSOR_WIDTH_MM, TEST_VIDEO_FRAME_COUNT, TEST_VIDEO_FPS, TEST_VIDEO_HEIGHT, TEST_VIDEO_WIDTH, make_run_input
from pipeline.create_run import create_run
from pipeline.stages import stage_0_ingest_video


def test_extracts_every_frame_as_a_valid_jpeg(stage_0_result):
    frames_dir = Path(stage_0_result["frames_dir"])
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    assert len(frame_paths) == TEST_VIDEO_FRAME_COUNT

    for path in frame_paths:
        image = cv2.imread(str(path))
        assert image is not None, f"{path} did not decode as a valid image"
        assert image.shape[:2] == (TEST_VIDEO_HEIGHT, TEST_VIDEO_WIDTH)


def test_scene_info_matches_the_real_video(runRecord, stage_0_result):
    assert runRecord.scene.frame_count == TEST_VIDEO_FRAME_COUNT
    assert runRecord.scene.width == TEST_VIDEO_WIDTH
    assert runRecord.scene.height == TEST_VIDEO_HEIGHT
    assert runRecord.scene.fps == pytest.approx(TEST_VIDEO_FPS, abs=0.1)


def test_intrinsics_matrix_is_built_from_the_given_lens_info(runRecord, stage_0_result):
    K = runRecord.scene.intrinsics_K
    expected_focal_px = FOCAL_LENGTH_MM * (TEST_VIDEO_WIDTH / SENSOR_WIDTH_MM)
    assert K[0][0] == pytest.approx(expected_focal_px)
    assert K[1][1] == pytest.approx(expected_focal_px)
    assert K[0][2] == pytest.approx(TEST_VIDEO_WIDTH / 2)
    assert K[1][2] == pytest.approx(TEST_VIDEO_HEIGHT / 2)
    assert K[2][2] == 1.0


def test_intrinsics_k_bypasses_the_computed_matrix_when_given(tmp_path):
    """A one-off RunRecord (not the shared session-scoped `runRecord` fixture,
    since this needs its own distinct RunInput) with a raw K given should use
    it as-is, ignoring focal_length_mm/sensor_width_mm entirely."""
    raw_k = [[123.0, 0.0, 45.0], [0.0, 123.0, 67.0], [0.0, 0.0, 1.0]]
    run_input = make_run_input(focal_length_mm=0.0, sensor_width_mm=0.0, intrinsics_k=raw_k)
    runRecord = create_run(tmp_path, run_input)

    stage_0_ingest_video.run(runRecord)

    assert runRecord.scene.intrinsics_K == raw_k


def test_metadata_rotated_video_is_portrait_and_preserves_sensor_width_for_intrinsics(tmp_path):
    """Runs Stage 0 against a real 64x32 MP4 with a 90-degree display matrix.

    With OpenCV auto-orientation disabled, the fixture exposes its 64x32
    encoded frame. Stage 0 must decode the same content exactly once into its
    32x64 portrait display orientation, while retaining the 64px sensor width
    for the focal-length conversion.
    """
    portrait_clip = Path(__file__).parent / "assets" / "portrait_clip.mp4"
    raw_capture = cv2.VideoCapture(str(portrait_clip))
    raw_capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    raw_width = int(raw_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_height = int(raw_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    success, raw_frame = raw_capture.read()
    raw_capture.release()
    assert success
    assert (raw_width, raw_height) == (64, 32)

    run_input = make_run_input(video_path=str(portrait_clip))
    runRecord = create_run(tmp_path / "run", run_input)

    outputs = stage_0_ingest_video.run(runRecord)

    assert (runRecord.scene.width, runRecord.scene.height) == (32, 64)
    expected_focal_px = FOCAL_LENGTH_MM * (64 / SENSOR_WIDTH_MM)
    assert runRecord.scene.intrinsics_K[0][0] == pytest.approx(expected_focal_px)
    assert runRecord.scene.intrinsics_K[1][1] == pytest.approx(expected_focal_px)
    assert runRecord.scene.intrinsics_K[0][2] == pytest.approx(32 / 2)
    assert runRecord.scene.intrinsics_K[1][2] == pytest.approx(64 / 2)

    written = cv2.imread(str(Path(outputs["frames_dir"]) / "000000.jpg"))
    expected = cv2.rotate(raw_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    assert written.shape == expected.shape
    # The frame is written as JPEG at quality 95. Check each asymmetric color
    # region well away from its boundary so JPEG chroma bleed cannot mask an
    # incorrect 90-degree transform.
    for x, y in ((8, 8), (24, 8), (16, 48)):
        assert np.abs(written[y, x].astype(np.int16) - expected[y, x].astype(np.int16)).max() <= 8


def _write_solid_frame(path: Path, value: int, size: tuple[int, int] = (12, 16)) -> None:
    """A tiny single-color image, distinct per `value`, lets a test verify
    which source image ended up at which output frame index."""
    height, width = size
    frame = np.full((height, width, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), frame)


def test_ingests_an_image_folder_instead_of_a_video(tmp_path):
    image_dir = tmp_path / "frames"
    image_dir.mkdir()
    for i in range(3):
        _write_solid_frame(image_dir / f"{i:06d}.jpg", value=i * 50)

    run_input = make_run_input(video_path=str(image_dir), source_fps=24.0)
    runRecord = create_run(tmp_path / "run", run_input)

    outputs = stage_0_ingest_video.run(runRecord)

    frames_dir = Path(outputs["frames_dir"])
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    assert len(frame_paths) == 3
    assert runRecord.scene.frame_count == 3
    assert runRecord.scene.fps == 24.0
    assert runRecord.scene.width == 16
    assert runRecord.scene.height == 12


def test_image_folder_frames_are_written_in_filename_sorted_order(tmp_path):
    """Regression against accidental directory-iteration order (not
    guaranteed by `Path.iterdir()`), output frame index must follow the
    source filenames' own sort order, not creation/insertion order."""
    image_dir = tmp_path / "frames"
    image_dir.mkdir()
    # Written out of order on purpose.
    _write_solid_frame(image_dir / "000002.jpg", value=200)
    _write_solid_frame(image_dir / "000000.jpg", value=0)
    _write_solid_frame(image_dir / "000001.jpg", value=100)

    run_input = make_run_input(video_path=str(image_dir), source_fps=24.0)
    runRecord = create_run(tmp_path / "run", run_input)
    outputs = stage_0_ingest_video.run(runRecord)

    frame_paths = sorted(Path(outputs["frames_dir"]).glob("*.jpg"))
    values = [int(cv2.imread(str(p))[0, 0, 0]) for p in frame_paths]
    assert values == [0, 100, 200]


def test_image_folder_skips_rewriting_a_frame_when_the_source_is_frames_dir_itself(tmp_path, monkeypatch):
    """Re-ingesting a run's own already-extracted input_frames/ folder in
    place (video_path pointed directly at frames_dir, already using the exact
    zero-padded naming) should read but not rewrite each frame, confirmed
    here via a patched cv2.imwrite that fails the test if called at all."""
    run_input = make_run_input(video_path="placeholder", source_fps=24.0)
    runRecord = create_run(tmp_path / "run", run_input)
    frames_dir = tmp_path / "run" / stage_0_ingest_video.INPUT_FRAMES_DIRNAME
    frames_dir.mkdir(parents=True)
    _write_solid_frame(frames_dir / "000000.jpg", value=10)
    _write_solid_frame(frames_dir / "000001.jpg", value=20)
    runRecord.input.video_path = str(frames_dir)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("cv2.imwrite should not be called when the source image IS the target frame path")

    monkeypatch.setattr(stage_0_ingest_video.cv2, "imwrite", _fail_if_called)

    outputs = stage_0_ingest_video.run(runRecord)

    assert runRecord.scene.frame_count == 2
    assert Path(outputs["frames_dir"]) == frames_dir


def test_image_folder_overwrites_stale_frames_from_a_different_source(tmp_path):
    """A --force-all-style rerun pointed at a genuinely different source
    folder must actually regenerate frames_dir's own contents, even when a
    same-named stale frame is already sitting there from a prior run,
    silently keeping it would defeat --force-all's whole point."""
    run_input = make_run_input(video_path="placeholder", source_fps=24.0)
    runRecord = create_run(tmp_path / "run", run_input)
    frames_dir = tmp_path / "run" / stage_0_ingest_video.INPUT_FRAMES_DIRNAME
    frames_dir.mkdir(parents=True)
    _write_solid_frame(frames_dir / "000000.jpg", value=99)  # stale, from a prior ingest

    new_source_dir = tmp_path / "new_frames"
    new_source_dir.mkdir()
    _write_solid_frame(new_source_dir / "000000.jpg", value=42)
    runRecord.input.video_path = str(new_source_dir)

    stage_0_ingest_video.run(runRecord)

    written = cv2.imread(str(frames_dir / "000000.jpg"))
    assert int(written[0, 0, 0]) == 42


def test_image_folder_accepts_a_mix_of_jpeg_and_png(tmp_path):
    image_dir = tmp_path / "frames"
    image_dir.mkdir()
    _write_solid_frame(image_dir / "000000.jpg", value=10)
    _write_solid_frame(image_dir / "000001.png", value=20)

    run_input = make_run_input(video_path=str(image_dir), source_fps=24.0)
    runRecord = create_run(tmp_path / "run", run_input)

    outputs = stage_0_ingest_video.run(runRecord)

    assert runRecord.scene.frame_count == 2
    assert len(list(Path(outputs["frames_dir"]).glob("*.jpg"))) == 2


def test_image_folder_raises_without_source_fps(tmp_path):
    image_dir = tmp_path / "frames"
    image_dir.mkdir()
    _write_solid_frame(image_dir / "000000.jpg", value=10)

    run_input = make_run_input(video_path=str(image_dir), source_fps=None)
    runRecord = create_run(tmp_path / "run", run_input)

    with pytest.raises(RuntimeError, match="--source-fps"):
        stage_0_ingest_video.run(runRecord)


def test_image_folder_raises_when_empty(tmp_path):
    image_dir = tmp_path / "frames"
    image_dir.mkdir()

    run_input = make_run_input(video_path=str(image_dir), source_fps=24.0)
    runRecord = create_run(tmp_path / "run", run_input)

    with pytest.raises(RuntimeError, match="No JPEG/PNG images found"):
        stage_0_ingest_video.run(runRecord)


def test_image_folder_intrinsics_use_output_width_as_the_sensor_width(tmp_path):
    """No separate pre-rotation sensor dimension exists for a plain image folder, raw_width and
    width are the same value, so the focal length in px should come out as
    a simple focal_length_mm * (width / sensor_width_mm), no rotation-aware
    adjustment involved."""
    image_dir = tmp_path / "frames"
    image_dir.mkdir()
    _write_solid_frame(image_dir / "000000.jpg", value=10, size=(12, 16))

    run_input = make_run_input(video_path=str(image_dir), source_fps=24.0)
    runRecord = create_run(tmp_path / "run", run_input)

    stage_0_ingest_video.run(runRecord)

    K = runRecord.scene.intrinsics_K
    expected_focal_px = FOCAL_LENGTH_MM * (16 / SENSOR_WIDTH_MM)
    assert K[0][0] == pytest.approx(expected_focal_px)
    assert K[0][2] == pytest.approx(16 / 2)
    assert K[1][2] == pytest.approx(12 / 2)


def test_scene_info_records_a_camera_up_measurement_attempt(runRecord, stage_0_result):
    """Stage 0 always leaves `camera_up` in one of its two valid states: a
    measured unit direction, or empty for "nothing measurable, assume level".
    The test clip is tiny and synthetic, so which one it lands on is not the
    point; storing a well-formed answer either way is."""
    camera_up = runRecord.scene.camera_up

    assert isinstance(camera_up, list)
    if camera_up:
        assert len(camera_up) == 3
        assert np.isclose(np.linalg.norm(camera_up), 1.0)
        # Up is up: the measurement is sign-resolved toward camera -Y, so a
        # result that points downward would be a convention bug.
        assert camera_up[1] < 0.0
