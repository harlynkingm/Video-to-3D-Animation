"""estimate_depth: runs Depth-Anything-3 (DA3METRIC-LARGE) once on the anchor
frame `mask_and_track` already resolved (`scene.anchor_frame_index`),
producing a metric depth map (meters). Consumed by `align_scene_scale` (not
yet implemented) to recover the scene's real-world scale.

No confidence map: DA3METRIC-LARGE's forward pass never populates a
"depth_conf" output (confirmed via real inference, see
depth_anything3_adapter.py), unlike the Any-view/Nested checkpoints this
project doesn't use.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..adapters.depth_anything3_adapter import DepthAnything3Adapter, KEY_DEPTH, KEY_SKY
from ..algorithms.depth_unprojection import scale_intrinsics_to_resolution, unproject_depth_to_points
from ..pipeline_stage_base import cli_entrypoint
from ..progress_tracker import ProgressRecord, StageName

# stage_0_ingest_video.py's own output key, consumed here.
FRAMES_DIR_OUTPUT_KEY = "frames_dir"

DEPTH_DIRNAME = "depth"
DEPTH_FILENAME = "anchor_depth.npy"
POINTCLOUD_FILENAME = "anchor_pointcloud.ply"

# This stage's own progress.json output keys.
OUTPUT_DEPTH = "anchor_depth"
OUTPUT_DEPTH_PREVIEW = "anchor_pointcloud_preview"


def _dump_pointcloud_preview(
    depth: np.ndarray,
    sky: np.ndarray | None,
    anchor_frame_path: Path,
    K: np.ndarray,
    native_hw: tuple[int, int],
    out_path: Path,
) -> None:
    """Colored point cloud for visual spot-checking in Blender (File > Import
    > Stanford (.ply) -- built in, no addon needed, unlike stage 2's SMPL-X
    preview). Excludes sky pixels (set to max depth by the model) so they
    don't dominate the point cloud as a distracting dome.

    Points are unprojected in the standard computer-vision camera convention
    (X right, Y down, Z forward) but written out rotated into Blender's Z-up
    world convention (X right, Y forward, Z up -- a fixed -90 degree rotation
    about X) so the cloud imports upright with no manual rotation needed.
    This rotation is specific to this preview file; `anchor_depth.npy` (this
    stage's real output) stays in plain camera-space coordinates for
    `align_scene_scale`'s future math.
    """
    depth_hw = depth.shape
    K_scaled = scale_intrinsics_to_resolution(K, native_hw, depth_hw)
    points = unproject_depth_to_points(depth, K_scaled)
    points = np.stack([points[:, 0], points[:, 2], -points[:, 1]], axis=-1)

    anchor_bgr = cv2.imread(str(anchor_frame_path))
    anchor_rgb = cv2.cvtColor(anchor_bgr, cv2.COLOR_BGR2RGB)
    colors = cv2.resize(anchor_rgb, (depth_hw[1], depth_hw[0]), interpolation=cv2.INTER_LINEAR)

    valid = ~sky.reshape(-1) if sky is not None else np.ones(points.shape[0], dtype=bool)
    points = points[valid]
    colors = colors.reshape(-1, 3)[valid]

    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    rows = np.hstack([points, colors.astype(np.float32)])
    with open(out_path, "w") as f:
        f.write(header)
        np.savetxt(f, rows, fmt="%.4f %.4f %.4f %d %d %d")


def run(progress: ProgressRecord) -> dict[str, str]:
    frames_dir = Path(progress.stages[StageName.STAGE_0_INGEST_VIDEO].outputs[FRAMES_DIR_OUTPUT_KEY])
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    anchor_frame_path = frame_paths[progress.scene.anchor_frame_index]

    K = np.array(progress.scene.intrinsics_K)
    # fx == fy by construction (camera_info_helpers.compute_intrinsics_matrix assumes square pixels).
    focal_length_px = K[0, 0]

    adapter = DepthAnything3Adapter()
    adapter.load()
    try:
        result = adapter.infer(str(anchor_frame_path), focal_length_px)
    finally:
        adapter.unload()

    depth_dir = Path(progress.progress_dir) / DEPTH_DIRNAME
    depth_dir.mkdir(parents=True, exist_ok=True)

    depth_path = depth_dir / DEPTH_FILENAME
    np.save(depth_path, result[KEY_DEPTH])

    outputs = {OUTPUT_DEPTH: str(depth_path)}

    if progress.input.dump_depth_preview:
        pointcloud_path = depth_dir / POINTCLOUD_FILENAME
        native_hw = (progress.scene.height, progress.scene.width)
        _dump_pointcloud_preview(
            result[KEY_DEPTH], result.get(KEY_SKY), anchor_frame_path, K, native_hw, pointcloud_path
        )
        outputs[OUTPUT_DEPTH_PREVIEW] = str(pointcloud_path)

    return outputs


if __name__ == "__main__":
    cli_entrypoint(run, stage_name=StageName.STAGE_3_ESTIMATE_DEPTH)
