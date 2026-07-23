"""Scene-scale alignment: fit the scale + translation that brings the depth
map's back-projected point cloud into the same metric space as the GVHMR
SMPL-X human mesh.

Both point sets live in the *same camera coordinate frame* already (GVHMR's
"incam" SMPL-X and the depth cloud are both camera-space), so there is no
rotation to solve -- only a scale `s` and translation `b`. This is the
static-camera, metric-depth specialization of `open4dhoi`'s
`preprocessing/scripts/hoi_utils.py::align` (which had to first normalize a
*relative*-depth map and use an orthographic back-projection; DA3METRIC-LARGE
gives real metric depth, so we back-project with the real intrinsics `K` instead).

Why a spread-ratio scale rather than a per-point least-squares fit: the
correspondence between a SMPL-X surface vertex and the depth value at its
projected pixel is only approximate (the depth surface and the SMPL-X surface
are not the same physical points), so we trust only aggregate statistics --
the ratio of spatial spreads for scale, the centroid offset for translation --
not individual point matches. This mirrors the reference's own deliberate
robustness choice.
"""

from __future__ import annotations

import numpy as np

from .depth_unprojection import unproject_depth_to_points

# Cap on how many correspondence points feed the O(n^2) mean-pairwise-distance
# spread estimate. Subsampled deterministically (even stride, no RNG) so the
# fit is reproducible.
MAX_CORRESPONDENCE_POINTS = 2000

# Below this many human/depth correspondences the scale estimate isn't
# trustworthy (e.g. the person is almost entirely out of frame at the anchor).
MIN_CORRESPONDENCE_POINTS = 50


def _mean_pairwise_distance(points: np.ndarray) -> float:
    """Mean Euclidean distance between all point pairs -- a spread measure that,
    unlike an axis-aligned bounding box, is rotation-invariant and robust to a
    few outliers."""
    from scipy.spatial.distance import pdist

    return float(pdist(points).mean())


def _subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    """Deterministic even-stride subsample (no RNG, for reproducibility)."""
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).astype(int)
    return points[idx]


def _project_to_pixels(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(N, 3) camera-space points -> (N, 2) pixel coordinates via K."""
    projected = points @ K.T
    return projected[:, :2] / projected[:, 2:3]


def _correspond_human_to_depth(
    smplx_verts: np.ndarray, depth: np.ndarray, K: np.ndarray, human_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Match each visible (front-facing) SMPL-X vertex to the depth-cloud point
    at the pixel it projects to, restricted to the human mask. Returns
    `(body_pts, scene_pts)`, aligned row-for-row.

    "Front-facing" = when several vertices project to the same pixel, keep the
    nearest-Z one (the visible surface), matching the reference's
    depth-sort-then-unique-by-pixel logic.
    """
    height, width = depth.shape
    uv = _project_to_pixels(smplx_verts, K)
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)

    in_frame = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (smplx_verts[:, 2] > 0)
    in_frame_idx = np.where(in_frame)[0]
    on_mask = human_mask[v[in_frame_idx], u[in_frame_idx]]
    candidate_idx = in_frame_idx[on_mask]

    # Among vertices sharing a pixel, keep the nearest (smallest Z) -- the front surface.
    pixel_key = v[candidate_idx] * width + u[candidate_idx]
    nearest_first = candidate_idx[np.argsort(smplx_verts[candidate_idx, 2])]
    pixel_key_sorted = pixel_key[np.argsort(smplx_verts[candidate_idx, 2])]
    _, first_occurrence = np.unique(pixel_key_sorted, return_index=True)
    final_idx = nearest_first[first_occurrence]

    scene_cloud = unproject_depth_to_points(depth, K)  # (H*W, 3), row-major (v*W + u)
    body_pts = smplx_verts[final_idx]
    scene_pts = scene_cloud[v[final_idx] * width + u[final_idx]]
    return body_pts, scene_pts


def fit_scene_scale(
    smplx_verts: np.ndarray, depth: np.ndarray, K: np.ndarray, human_mask: np.ndarray
) -> tuple[float, np.ndarray, int]:
    """Fit `(scale, translation)` mapping SMPL-X metric space onto the depth
    cloud's space: a depth-space point `p` maps back into SMPL-X space via
    `(p - translation) / scale`, and a SMPL-X point `q` maps into depth space
    via `q * scale + translation`.

    Args:
        smplx_verts: (V, 3) SMPL-X vertices, camera-space metric (GVHMR incam).
        depth: (H, W) metric depth map.
        K: (3, 3) intrinsics matching `depth`'s resolution (rescale first if the
            depth map isn't at native resolution -- see
            `depth_unprojection.scale_intrinsics_to_resolution`).
        human_mask: (H, W) bool, True on the person, at `depth`'s resolution.

    Returns:
        `(scale, translation, n_correspondences)`.
    """
    body_pts, scene_pts = _correspond_human_to_depth(smplx_verts, depth, K, human_mask)
    if len(body_pts) < MIN_CORRESPONDENCE_POINTS:
        raise RuntimeError(
            f"only {len(body_pts)} human/depth correspondences (need >= "
            f"{MIN_CORRESPONDENCE_POINTS}); the person may be mostly out of frame at the anchor"
        )

    body_spread = _mean_pairwise_distance(_subsample(body_pts, MAX_CORRESPONDENCE_POINTS))
    scene_spread = _mean_pairwise_distance(_subsample(scene_pts, MAX_CORRESPONDENCE_POINTS))
    scale = scene_spread / body_spread
    translation = scene_pts.mean(axis=0) - scale * body_pts.mean(axis=0)
    return float(scale), translation, len(body_pts)
