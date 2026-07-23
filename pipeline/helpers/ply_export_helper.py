"""Write colored point clouds to PLY for Blender visual verification.

Centralizes the camera-space -> Blender Z-up rotation (learned the hard way --
see docs/ARCHITECTURE.md's stage 3 notes) so every preview this pipeline emits
imports upright with no manual rotation, and so that knowledge lives in exactly
one place.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_PLY_HEADER = (
    "ply\nformat ascii 1.0\n"
    "element vertex {n}\n"
    "property float x\nproperty float y\nproperty float z\n"
    "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    "end_header\n"
)


def camera_space_to_blender(points_xyz: np.ndarray) -> np.ndarray:
    """Computer-vision camera convention (X right, Y down, Z forward) -> Blender
    Z-up world (X right, Y forward, Z up): a fixed -90 degree rotation about X."""
    return np.stack([points_xyz[:, 0], points_xyz[:, 2], -points_xyz[:, 1]], axis=-1)


def write_colored_ply(
    points_xyz: np.ndarray, colors_rgb: np.ndarray, out_path: Path, *, to_blender: bool = True
) -> None:
    """Write an ASCII PLY point cloud.

    Args:
        points_xyz: (N, 3) float positions in meters.
        colors_rgb: (N, 3) uint8 per-point RGB.
        out_path: destination file.
        to_blender: rotate camera-space points into Blender's Z-up convention
            first (the default, since these files exist to be opened in Blender).
    """
    points = camera_space_to_blender(points_xyz) if to_blender else points_xyz
    rows = np.hstack([points, colors_rgb.astype(np.float32)])
    with open(out_path, "w") as f:
        f.write(_PLY_HEADER.format(n=points.shape[0]))
        np.savetxt(f, rows, fmt="%.4f %.4f %.4f %d %d %d")
