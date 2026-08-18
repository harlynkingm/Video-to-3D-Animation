"""Camera intrinsics math: real-world lens/sensor specs -> a 3x3 K matrix.

A bare focal length isn't enough to build this, the same "35mm" lens
produces a very different field of view on a phone sensor than on a
full-frame camera. Sensor width disambiguates that.
"""

from __future__ import annotations

def compute_intrinsics_matrix(
    focal_length_mm: float,
    sensor_width_mm: float,
    sensor_width_px: int,
    image_width_px: int,
    image_height_px: int,
) -> list[list[float]]:
    """Build the pinhole camera intrinsics matrix K.

    `sensor_width_px` is the encoded sensor-frame width, what the
    focal-length ratio is measured against. It is NOT always the same as
    `image_width_px`/`image_height_px` (the decoded output frame's dimensions,
    used only to center the principal point). For a phone's vertical video,
    display-orientation metadata can turn a landscape sensor frame into a
    portrait decoded frame without changing the physical sensor width.

    Assumes square pixels and a centered principal point, which holds for
    essentially all consumer camera/phone footage.
    """
    focal_length_px = focal_length_mm * (sensor_width_px / sensor_width_mm)
    cx = image_width_px / 2.0
    cy = image_height_px / 2.0
    return [
        [focal_length_px, 0.0, cx],
        [0.0, focal_length_px, cy],
        [0.0, 0.0, 1.0],
    ]
