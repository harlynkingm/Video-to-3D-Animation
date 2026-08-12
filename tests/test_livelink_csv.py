"""Unit tests for `livelink_csv.py`. Pure numpy/stdlib, no model assets
needed, always runs.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.helpers.livelink_csv import (
    ARKIT_BLENDSHAPE_NAMES, BLENDSHAPE_COUNT, CSV_HEADER, HEAD_EYE_COLUMN_NAMES, format_timecode, write_livelink_csv,
)

# Byte-for-byte from a real LiveLinkFace capture's own header row.
REAL_CAPTURE_HEADER = (
    "Timecode,BlendshapeCount,EyeBlinkLeft,EyeLookDownLeft,EyeLookInLeft,EyeLookOutLeft,EyeLookUpLeft,"
    "EyeSquintLeft,EyeWideLeft,EyeBlinkRight,EyeLookDownRight,EyeLookInRight,EyeLookOutRight,EyeLookUpRight,"
    "EyeSquintRight,EyeWideRight,JawForward,JawRight,JawLeft,JawOpen,MouthClose,MouthFunnel,MouthPucker,"
    "MouthRight,MouthLeft,MouthSmileLeft,MouthSmileRight,MouthFrownLeft,MouthFrownRight,MouthDimpleLeft,"
    "MouthDimpleRight,MouthStretchLeft,MouthStretchRight,MouthRollLower,MouthRollUpper,MouthShrugLower,"
    "MouthShrugUpper,MouthPressLeft,MouthPressRight,MouthLowerDownLeft,MouthLowerDownRight,MouthUpperUpLeft,"
    "MouthUpperUpRight,BrowDownLeft,BrowDownRight,BrowInnerUp,BrowOuterUpLeft,BrowOuterUpRight,CheekPuff,"
    "CheekSquintLeft,CheekSquintRight,NoseSneerLeft,NoseSneerRight,TongueOut,HeadYaw,HeadPitch,HeadRoll,"
    "LeftEyeYaw,LeftEyePitch,LeftEyeRoll,RightEyeYaw,RightEyePitch,RightEyeRoll"
)


def test_header_matches_real_capture_byte_for_byte():
    assert ",".join(CSV_HEADER) == REAL_CAPTURE_HEADER


def test_header_has_63_columns():
    assert len(CSV_HEADER) == 63


def test_blendshape_count_is_61():
    assert BLENDSHAPE_COUNT == 61
    assert BLENDSHAPE_COUNT == len(ARKIT_BLENDSHAPE_NAMES) + len(HEAD_EYE_COLUMN_NAMES)


@pytest.mark.parametrize("fps", [30.0, 29.97, 60.0])
def test_format_timecode_starts_at_zero(fps):
    assert format_timecode(0, fps) == "00:00:00:00.000"


def test_format_timecode_one_second_boundary_at_30fps():
    # Frame 30 at 30fps lands exactly on the 1-second mark.
    assert format_timecode(30, 30.0) == "00:00:01:00.000"


def test_format_timecode_one_minute_boundary_at_30fps():
    assert format_timecode(30 * 60, 30.0) == "00:01:00:00.000"


def test_format_timecode_one_hour_boundary_at_30fps():
    assert format_timecode(30 * 3600, 30.0) == "01:00:00:00.000"


def test_format_timecode_60fps_frame_within_second():
    tc = format_timecode(45, 60.0)
    assert tc == "00:00:00:45.750"


def test_format_timecode_2997fps_does_not_produce_invalid_frame_or_ms():
    # 29.97fps: frame_index=29 lands just under 1 second, FF/mmm must stay
    # within [0, round(fps)-1] / [0, 999], never overflow into an invalid string.
    for i in range(60):
        tc = format_timecode(i, 29.97)
        hh, mm, ss, rest = tc.split(":")
        ff, mmm = rest.split(".")
        assert 0 <= int(ff) < 30
        assert 0 <= int(mmm) <= 999


def test_format_timecode_mmm_rounding_carries_into_seconds_not_left_at_1000():
    # Construct fps/frame_index so the fractional-second component rounds to
    # 1.000, the carry path (mmm reset to 0, whole_seconds bumped) must fire
    # instead of emitting "SS.1000".
    # At fps=1000.0, frame 999 -> total_seconds=0.999 -> frac=0.999 -> mmm=999 (no carry).
    # frame 1000 -> total_seconds=1.0 exactly -> whole_seconds=1, frac=0.0 -> "00:00:01:00.000".
    # The actual carry case: a frac that rounds up to 1.0 while whole_seconds
    # doesn't already reflect it, e.g. fps where total_seconds has a
    # floating remainder just under 1 that rounds to 1000ms.
    tc = format_timecode(2999, 3000.0)  # total_seconds = 0.9996666... -> mmm rounds to 1000
    assert ".1000" not in tc
    hh, mm, ss, rest = tc.split(":")
    ff, mmm = rest.split(".")
    assert int(mmm) < 1000


def test_write_livelink_csv_row_count_matches_frame_count():
    import tempfile
    from pathlib import Path

    n = 5
    arkit_weights = np.zeros((n, len(ARKIT_BLENDSHAPE_NAMES)), dtype=np.float32)
    head_eye_euler = np.zeros((n, len(HEAD_EYE_COLUMN_NAMES)), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "output_face.csv"
        write_livelink_csv(path, arkit_weights, head_eye_euler, fps=30.0)
        lines = path.read_text().strip("\n").split("\n")

    assert len(lines) == n + 1  # header + n rows
    assert lines[0] == REAL_CAPTURE_HEADER


def test_write_livelink_csv_every_row_has_63_columns():
    import tempfile
    from pathlib import Path

    n = 3
    rng = np.random.default_rng(0)
    arkit_weights = rng.uniform(0, 1, size=(n, len(ARKIT_BLENDSHAPE_NAMES))).astype(np.float32)
    head_eye_euler = rng.uniform(-30, 30, size=(n, len(HEAD_EYE_COLUMN_NAMES))).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "output_face.csv"
        write_livelink_csv(path, arkit_weights, head_eye_euler, fps=30.0)
        lines = path.read_text().strip("\n").split("\n")

    for line in lines:
        assert len(line.split(",")) == 63
