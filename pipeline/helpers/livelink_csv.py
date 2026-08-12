"""Writes `output_face.csv` in LiveLinkFace's own CSV format: `Timecode`,
`BlendshapeCount`, the 52 ARKit blendshape channels, then 9 head/eye Euler
columns.

Column names and order verified byte-for-byte against a real LiveLinkFace
capture, not reconstructed from documentation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ARKIT_BLENDSHAPE_NAMES = [
    "EyeBlinkLeft", "EyeLookDownLeft", "EyeLookInLeft", "EyeLookOutLeft", "EyeLookUpLeft", "EyeSquintLeft", "EyeWideLeft",
    "EyeBlinkRight", "EyeLookDownRight", "EyeLookInRight", "EyeLookOutRight", "EyeLookUpRight", "EyeSquintRight", "EyeWideRight",
    "JawForward", "JawRight", "JawLeft", "JawOpen",
    "MouthClose", "MouthFunnel", "MouthPucker", "MouthRight", "MouthLeft", "MouthSmileLeft", "MouthSmileRight",
    "MouthFrownLeft", "MouthFrownRight", "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft", "MouthStretchRight",
    "MouthRollLower", "MouthRollUpper", "MouthShrugLower", "MouthShrugUpper", "MouthPressLeft", "MouthPressRight",
    "MouthLowerDownLeft", "MouthLowerDownRight", "MouthUpperUpLeft", "MouthUpperUpRight",
    "BrowDownLeft", "BrowDownRight", "BrowInnerUp", "BrowOuterUpLeft", "BrowOuterUpRight",
    "CheekPuff", "CheekSquintLeft", "CheekSquintRight",
    "NoseSneerLeft", "NoseSneerRight",
    "TongueOut",
]
assert len(ARKIT_BLENDSHAPE_NAMES) == 52

HEAD_EYE_COLUMN_NAMES = [
    "HeadYaw", "HeadPitch", "HeadRoll",
    "LeftEyeYaw", "LeftEyePitch", "LeftEyeRoll",
    "RightEyeYaw", "RightEyePitch", "RightEyeRoll",
]
assert len(HEAD_EYE_COLUMN_NAMES) == 9

# The real capture's own `BlendshapeCount` value: 52 ARKit channels + 9
# head/eye Euler columns, i.e. every data column after `Timecode`, not the
# same number as `len(ARKIT_BLENDSHAPE_NAMES)` alone, and not the total
# column count (which also includes `Timecode`/`BlendshapeCount` themselves).
BLENDSHAPE_COUNT = len(ARKIT_BLENDSHAPE_NAMES) + len(HEAD_EYE_COLUMN_NAMES)

CSV_HEADER = ["Timecode", "BlendshapeCount"] + ARKIT_BLENDSHAPE_NAMES + HEAD_EYE_COLUMN_NAMES


def format_timecode(frame_index: int, fps: float) -> str:
    """`HH:MM:SS:FF.mmm`, zero-based from `frame_index`/`fps`. `FF` is the
    frame number within the current second (0 to round(fps)-1, SMPTE
    non-drop-frame style); `mmm` is milliseconds within the current second,
    computed independently from the same fractional-second value, rounding
    can push it to 1000, which must carry into the second (not just clamp),
    or the string would show an invalid "SS.1000".
    """
    nominal_fps = round(fps)
    total_seconds = frame_index / fps
    whole_seconds = int(total_seconds)
    frac = total_seconds - whole_seconds

    mmm = round(frac * 1000)
    if mmm >= 1000:
        mmm = 0
        whole_seconds += 1
        frac = 0.0

    ff = round(frac * nominal_fps)
    if ff >= nominal_fps:
        ff = nominal_fps - 1

    hh, remainder = divmod(whole_seconds, 3600)
    mm, ss = divmod(remainder, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}.{mmm:03d}"


def write_livelink_csv(path: Path, arkit_weights: np.ndarray, head_eye_euler: np.ndarray, fps: float) -> None:
    """`arkit_weights`: (F, 52) in `ARKIT_BLENDSHAPE_NAMES` order, values in
    [0, 1]. `head_eye_euler`: (F, 9) in `HEAD_EYE_COLUMN_NAMES` order,
    degrees (the `Head*` columns are expected to already be zero, see the
    plan's own reasoning for why, this function just writes whatever it's
    given).
    """
    n = arkit_weights.shape[0]
    assert arkit_weights.shape == (n, len(ARKIT_BLENDSHAPE_NAMES))
    assert head_eye_euler.shape == (n, len(HEAD_EYE_COLUMN_NAMES))

    lines = [",".join(CSV_HEADER)]
    for i in range(n):
        row = [format_timecode(i, fps), str(BLENDSHAPE_COUNT)]
        row += [f"{v:.10f}" for v in arkit_weights[i]]
        row += [f"{v:.10f}" for v in head_eye_euler[i]]
        lines.append(",".join(row))

    Path(path).write_text("\n".join(lines) + "\n")
