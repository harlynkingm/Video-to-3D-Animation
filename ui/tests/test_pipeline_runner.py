import sys

from ui.pipeline_runner import RunFormState, build_run_argv


def _base_state(**overrides) -> RunFormState:
    defaults = dict(
        video_path="C:/clips/video.mp4",
        destination_folder="C:/runs/my_clip",
        human_prompt="a person",
        focal_length_mm=26.0,
        sensor_width_mm=36.0,
    )
    defaults.update(overrides)
    return RunFormState(**defaults)


def test_build_run_argv_basic_video_run():
    argv = build_run_argv(_base_state())

    assert argv[:3] == [sys.executable, "-m", "pipeline.run"]
    assert "--input-video" in argv and argv[argv.index("--input-video") + 1] == "C:/clips/video.mp4"
    assert "--output-dir" in argv and argv[argv.index("--output-dir") + 1] == "C:/runs/my_clip"
    assert "--human-prompt" in argv and argv[argv.index("--human-prompt") + 1] == "a person"
    assert "--focal-length-mm" in argv and argv[argv.index("--focal-length-mm") + 1] == "26.0"
    assert "--sensor-width-mm" in argv and argv[argv.index("--sensor-width-mm") + 1] == "36.0"
    assert "--start-on-stage" in argv and argv[argv.index("--start-on-stage") + 1] == "0"
    assert "--stop-after-stage" in argv and argv[argv.index("--stop-after-stage") + 1] == "9"
    assert "--object-shape-hint" in argv and argv[argv.index("--object-shape-hint") + 1] == "auto"
    assert "--object-prompt" not in argv
    assert "--source-fps" not in argv
    assert "--force-all" not in argv
    assert "--render-previews" not in argv


def test_build_run_argv_includes_object_prompt_when_set():
    argv = build_run_argv(_base_state(object_prompt="a teddy bear"))

    assert "--object-prompt" in argv
    assert argv[argv.index("--object-prompt") + 1] == "a teddy bear"


def test_build_run_argv_image_sequence_includes_source_fps():
    argv = build_run_argv(_base_state(
        video_path="C:/clips/frames", is_image_sequence=True, source_fps=29.97,
    ))

    assert "--source-fps" in argv
    assert argv[argv.index("--source-fps") + 1] == "29.97"


def test_build_run_argv_omits_source_fps_when_not_image_sequence():
    # A stray source_fps value should be ignored unless is_image_sequence is set.
    argv = build_run_argv(_base_state(is_image_sequence=False, source_fps=29.97))

    assert "--source-fps" not in argv


def test_build_run_argv_force_all_and_render_previews_flags():
    argv = build_run_argv(_base_state(force_all=True, render_previews=True))

    assert "--force-all" in argv
    assert "--render-previews" in argv


def test_build_run_argv_stage_range_and_object_shape():
    argv = build_run_argv(_base_state(start_stage=4, stop_stage=6, object_shape="box"))

    assert argv[argv.index("--start-on-stage") + 1] == "4"
    assert argv[argv.index("--stop-after-stage") + 1] == "6"
    assert argv[argv.index("--object-shape-hint") + 1] == "box"
