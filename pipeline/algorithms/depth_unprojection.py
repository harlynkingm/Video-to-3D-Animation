"""Depth map -> 3D point cloud, pure math (numpy only, no model/GPU). Shared by
stage_3_estimate_depth.py's optional preview output and, later,
align_scene_scale's depth-cloud<->SMPL-X similarity fit (see
docs/ARCHITECTURE.md's Stage DAG) -- this project's port of open4dhoi's
make_hoi.py scale-alignment step needs exactly this same unprojection.
"""

from __future__ import annotations

import numpy as np


def scale_intrinsics_to_resolution(
    K: np.ndarray, native_hw: tuple[int, int], target_hw: tuple[int, int]
) -> np.ndarray:
    """Rescale a camera intrinsics matrix `K` (built for `native_hw`) to match
    a different resolution `target_hw` -- needed because Depth-Anything-3
    (like SAM 3.1) resizes internally to its own working resolution rather
    than the source frame's native size (see ARCHITECTURE.md's
    Depth-Anything-3 notes).
    """
    native_h, native_w = native_hw
    target_h, target_w = target_hw
    scale_x = target_w / native_w
    scale_y = target_h / native_h
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 2] *= scale_y  # cy
    return K_scaled


def unproject_depth_to_points(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Unproject a (H, W) metric depth map into an (H*W, 3) camera-space point
    cloud using the pinhole camera model. `K` must already match `depth`'s own
    resolution (see `scale_intrinsics_to_resolution` if it doesn't).
    """
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    return np.stack([x, y, depth], axis=-1).reshape(-1, 3)
