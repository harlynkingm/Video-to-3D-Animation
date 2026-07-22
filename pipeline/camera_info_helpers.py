"""Camera intrinsics math: real-world lens/sensor specs -> a 3x3 K matrix.

A bare focal length isn't enough to build this -- the same "35mm" lens
produces a very different field of view on a phone sensor than on a
full-frame camera. Sensor width disambiguates that.
"""


def compute_intrinsics_matrix(
    focal_length_mm: float,
    sensor_width_mm: float,
    image_width_px: int,
    image_height_px: int,
) -> list[list[float]]:
    """Build the pinhole camera intrinsics matrix K.

    Assumes square pixels and a centered principal point, which holds for
    essentially all consumer camera/phone footage.
    """
    focal_length_px = focal_length_mm * (image_width_px / sensor_width_mm)
    cx = image_width_px / 2.0
    cy = image_height_px / 2.0
    return [
        [focal_length_px, 0.0, cx],
        [0.0, focal_length_px, cy],
        [0.0, 0.0, 1.0],
    ]
