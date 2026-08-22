"""Camera-space gravity ("up") direction, measured from the scene's own
vertical lines.

Every consumer of GVHMR's camera-space motion downstream of stage 2 needs to
know which way is up, and until this existed the whole pipeline simply assumed
the camera was perfectly level: `bvh_export.CAMERA_TO_BVH_ROOT_ROTATION` maps
camera -Y to world up unconditionally. That assumption is a real error source
on handheld footage.

The estimate is one direction for the whole clip, not per frame: it corrects
the frame the motion is expressed in, and the pipeline deliberately exports
camera-relative motion (see `gvhmr_adapter`'s module docstring on incam vs the
vestigial "global" frame), so a camera that rotates mid-clip is out of scope
here and its rotation stays baked into the motion either way.

Method: a line-segment detector over sampled frames; each near-vertical
segment back-projects to a plane through the camera centre whose normal is
perpendicular to the segment's own 3D direction, so the scene vertical is the
direction most nearly orthogonal to every such normal (the classic vertical
vanishing point, solved as the smallest singular vector). Robustness matters
more than precision here, because the subject's own body produces long,
convincingly vertical segments that are not scene verticals at all: hence the
per-frame solve plus a geodesic median across frames, rather than pooling
every segment in the clip into one solve.
"""

from __future__ import annotations

import numpy as np

# cv2 is imported inside the functions that need it, not at module scope: the
# `export` environment has no cv2 at all (it exists only to pin bpy's own
# Python), and it reaches this module through `bvh_export`, which needs
# `LEVEL_CAMERA_UP` and the rotation math but never runs a detector.

# Camera space is X right, Y down, Z forward (see `bvh_export`'s own module
# comment), so a level camera's up direction is -Y. This is what the pipeline
# assumed unconditionally before this module existed, and it stays the
# fallback whenever the scene gives nothing measurable.
LEVEL_CAMERA_UP = np.array([0.0, -1.0, 0.0])

# The clip is sampled as evenly-spaced windows of consecutive frames, each
# window solved independently and the windows combined with a geodesic median.
# Windows rather than single frames because a small, sparsely-built scene can
# hold too few long segments to solve one frame on its own. The window stays
# short so that consecutive frames within it share a camera pose.
SAMPLE_WINDOW_COUNT = 12
SAMPLE_WINDOW_FRAMES = 5

# A segment must be at least this fraction of the frame height to vote. Short
# segments carry almost no angular information and are dominated by texture
# noise (grass, tile grout speckle, fabric folds).
MIN_SEGMENT_LENGTH_FRACTION = 0.04

# First-pass selection: how far from image-vertical a segment may lie and
# still be treated as a candidate scene vertical. Generous, because camera
# roll and perspective both tilt true verticals in the image; the second pass
# re-selects against the actual estimate instead of against the image axis.
FIRST_PASS_TOLERANCE_DEGREES = 30.0

# Second-pass selection: a segment votes if its own image direction points at
# the estimated vertical vanishing point to within this angle.
VANISHING_POINT_TOLERANCE_DEGREES = 20.0

# Reweighting passes for the per-frame solve. Three is where the estimate
# stopped moving on test clips; more only costs time.
REWEIGHT_ITERATIONS = 3

# A window needs this many voting segments to produce an estimate at all, and
# the clip needs this many such windows. Below either, the scene simply has no
# usable vertical structure and the level-camera assumption is the honest
# answer rather than a fabricated tilt.
MIN_SEGMENTS_PER_WINDOW = 15
MIN_USABLE_WINDOWS = 3

# If the per-window estimates disagree by more than this about their own
# median, they are not measuring one consistent scene vertical (a camera that
# rotates through the take, or a scene whose only long lines belong to the
# moving subject), and the result is discarded rather than trusted.
CAMERA_UP_MAX_DISPERSION_DEGREES = 15.0


def _frame_segments(gray: np.ndarray, K_inv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(directions, midpoints, normals)` for one frame's long line segments.

    `directions` are unit image-space segment directions and `midpoints` their
    image-space centres (both used for vanishing-point reselection); `normals`
    are the normals of the plane each segment spans with the camera centre,
    length-weighted so a long, well-localized architectural edge outvotes a
    short one.
    """
    import cv2

    segments = cv2.createLineSegmentDetector().detect(gray)[0]
    if segments is None:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 3))

    endpoints = segments.reshape(-1, 4)
    deltas = endpoints[:, 2:] - endpoints[:, :2]
    lengths = np.linalg.norm(deltas, axis=1)
    long_enough = lengths >= MIN_SEGMENT_LENGTH_FRACTION * gray.shape[0]
    endpoints, deltas, lengths = endpoints[long_enough], deltas[long_enough], lengths[long_enough]
    if len(endpoints) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 3))

    corners = endpoints.reshape(-1, 2, 2)
    homogeneous = np.concatenate([corners, np.ones((len(endpoints), 2, 1))], axis=2)
    rays = homogeneous @ K_inv.T
    normals = _unit(np.cross(rays[:, 0], rays[:, 1]))
    return deltas / lengths[:, None], corners.mean(axis=1), normals * lengths[:, None]


def _solve_vertical(normals: np.ndarray) -> np.ndarray:
    """The unit direction most nearly orthogonal to every plane normal, i.e.
    the shared 3D direction of the segments those planes came from. Solved as
    the smallest singular vector, then reweighted so a minority of segments
    belonging to some other direction cannot drag the answer."""
    vertical = np.linalg.svd(normals)[2][-1]
    for _ in range(REWEIGHT_ITERATIONS):
        residual = np.abs(normals @ vertical) / np.linalg.norm(normals, axis=1)
        weight = 1.0 / (residual + np.percentile(residual, 40) + 1e-6)
        vertical = np.linalg.svd(normals * weight[:, None])[2][-1]
    # Camera Y points down, so the up-pointing sign is the negative-Y one.
    if vertical[1] > 0:
        vertical = -vertical
    return vertical / np.linalg.norm(vertical)


def _window_vertical(
    directions: np.ndarray, midpoints: np.ndarray, normals: np.ndarray, K: np.ndarray,
) -> np.ndarray | None:
    """One window's camera-space up estimate, or None if it has too few usable
    segments. Two passes: the first selects candidates by how vertical they
    look in the image, the second re-selects by whether they actually point at
    the resulting vanishing point, which recovers true verticals that camera
    roll or strong perspective tilted out of the first pass's window."""
    if len(normals) < MIN_SEGMENTS_PER_WINDOW:
        return None

    # atan2(dx, dy) measures away from the image's own vertical axis; a
    # segment and its reverse describe the same line, hence the fold at 90.
    from_image_vertical = np.abs(np.degrees(np.arctan2(directions[:, 0], directions[:, 1])))
    candidate = np.minimum(from_image_vertical, 180.0 - from_image_vertical) <= FIRST_PASS_TOLERANCE_DEGREES
    if candidate.sum() < MIN_SEGMENTS_PER_WINDOW:
        return None
    vertical = _solve_vertical(normals[candidate])

    # A 3D direction's vanishing point is where its rays meet in the image.
    # It escapes to infinity as the direction becomes parallel to the image
    # plane, in which case every true vertical is already parallel in-image
    # and the first pass's selection was the right one to keep.
    vanishing_point = K @ vertical
    if abs(vanishing_point[2]) > 1e-6:
        to_vanishing_point = vanishing_point[:2] / vanishing_point[2] - midpoints
        cosine = np.abs(np.sum(_unit(to_vanishing_point) * directions, axis=1))
        candidate = cosine >= np.cos(np.radians(VANISHING_POINT_TOLERANCE_DEGREES))
        if candidate.sum() >= MIN_SEGMENTS_PER_WINDOW:
            vertical = _solve_vertical(normals[candidate])
    return vertical


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)


def _geodesic_median(directions: np.ndarray) -> np.ndarray:
    """Weiszfeld iteration on the unit sphere. A plain mean is pulled off
    course by the handful of windows where the subject's own limbs dominate
    the segment set; the median is not."""
    median = _unit(directions.mean(axis=0))
    for _ in range(REWEIGHT_ITERATIONS * 3):
        distance = np.linalg.norm(directions - median, axis=1)
        weight = 1.0 / np.maximum(distance, 1e-6)
        median = _unit((directions * weight[:, None]).sum(axis=0))
    return median


def angle_between_degrees(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(_unit(np.asarray(a)), _unit(np.asarray(b))), -1.0, 1.0))))


def estimate_camera_up(frame_paths: list, K: np.ndarray) -> list[float] | None:
    """Camera-space up direction for a clip, or None if it isn't measurable.

    `None` is a real answer, not a failure: a scene with no vertical structure
    (or a camera that rotates enough that no single direction describes the
    clip) gives nothing to measure, and callers should keep assuming a level
    camera rather than acting on a fabricated tilt.
    """
    import cv2

    K = np.asarray(K, dtype=np.float64)
    K_inv = np.linalg.inv(K)
    stride = max(SAMPLE_WINDOW_FRAMES, len(frame_paths) // SAMPLE_WINDOW_COUNT)

    estimates = []
    for start in range(0, len(frame_paths), stride)[:SAMPLE_WINDOW_COUNT]:
        window = [], [], []
        for path in frame_paths[start:start + SAMPLE_WINDOW_FRAMES]:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            for collected, part in zip(window, _frame_segments(gray, K_inv)):
                collected.append(part)
        if not window[0]:
            continue
        vertical = _window_vertical(*(np.concatenate(part) for part in window), K)
        if vertical is not None:
            estimates.append(vertical)

    if len(estimates) < MIN_USABLE_WINDOWS:
        return None
    estimates = np.array(estimates)
    median = _geodesic_median(estimates)
    dispersion = np.median([angle_between_degrees(e, median) for e in estimates])
    if dispersion > CAMERA_UP_MAX_DISPERSION_DEGREES:
        return None
    return [float(v) for v in median]
