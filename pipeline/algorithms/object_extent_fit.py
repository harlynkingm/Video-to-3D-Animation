"""Object shape/extent fit: given the tracked object's back-projected point
cloud at the anchor frame (already in the same metric space as the SMPL-X
human body -- see `similarity_transform.fit_scene_scale`), fits a simple
proxy primitive (oriented box, ellipsoid, or cylinder) that approximates its
shape. Ports the "object shape" half of open4dhoi's `make_hoi.py`, but fits a
primitive instead of resizing a reconstructed mesh (no SAM-3D-Objects on
this hardware).

Box and ellipsoid both get independent extents per axis (not a cube / a
single-radius sphere) -- a real "rectangular object" or "circular object" has
different depth/height/width, and PCA already gives an oriented frame to
measure them in. Cylinder covers a real gap between the two: bottles, cans,
and cups have a flat-ended, constant-radius profile that a box wastes residual
on (rounded sides) and an ellipsoid can't represent at all (no flat ends).

A depth back-projection only ever sees the object's front-facing surface from
one viewpoint (never a full 360-degree scan), so "goodness of fit" is scored
the same way for every candidate: the mean distance from each observed point
to the *surface* of the fitted primitive (not just whether it's contained
inside), so a shape that doesn't match the object's real profile scores worse
regardless of which one it is. `auto` picks whichever residual is lower; the
box is fit to exactly span the point cloud along its own principal axes, so
its residual is always an interior (inside-to-nearest-face) distance -- the
"outside" branch of `_point_to_box_surface_distance` exists for reuse (e.g. a
future stage 8 contact-distance cost term), not because this fit itself ever
produces exterior points.

Before any of that, `_reject_depth_outliers` trims the raw point cloud: DA3
(like any monocular depth model) blurs/bleeds its estimate across a real
depth discontinuity, so a handful of pixels right on the object mask's own
silhouette edge can read meters deeper than the object's real surface. A
plain min/max (box) or std (ellipsoid) fit -- and even the PCA orientation
every fitter starts from -- is not robust to that tail, so it gets dropped
before fitting rather than corrected after.
"""

from __future__ import annotations

import numpy as np

from ..progress_tracker import ObjectShapeHint

# Below this many object points, a shape fit isn't trustworthy (e.g. the
# object is barely visible/mostly occluded at the anchor frame).
MIN_OBJECT_POINTS = 20

# Beyond this many scaled MADs (median absolute deviations) from the point
# cloud's own median, a point is rejected before fitting. 1.4826x the MAD
# approximates a Gaussian std; ~3.5 of those is generous enough to keep a
# real object's own shape spread while dropping a depth-bleed tail.
_OUTLIER_REJECTION_MAD_MULTIPLIER = 3.5

KEY_KIND = "kind"
KIND_BOX = "box"
KIND_ELLIPSOID = "ellipsoid"
KIND_CYLINDER = "cylinder"


def _pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centroid + principal-axis rotation shared by both candidate fits.
    Returns `(centroid, rotation, local)`, where `local` is `points` re-expressed
    in the centered, PCA-rotated frame (object-local -> body-space is `rotation`).
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    rotation = vt.T
    return centroid, rotation, centered @ rotation


def _point_to_box_surface_distance(points_local: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    """Distance from each point (already in the box's own centered, axis-aligned
    local frame) to the box's surface -- positive whether the point is outside
    (distance to the nearest corner/edge/face) or inside (distance to the
    nearest face)."""
    q = np.abs(points_local)
    outside = np.maximum(q - half_extents, 0)
    outside_dist = np.linalg.norm(outside, axis=1)
    inside_dist = np.clip((half_extents - q).min(axis=1), 0, None)
    return np.where(outside_dist > 0, outside_dist, inside_dist)


def _point_to_ellipsoid_surface_distance(points_local: np.ndarray, semi_axes: np.ndarray) -> np.ndarray:
    """Approximate distance from each point to the ellipsoid's surface: the
    difference between the point's own radius and the radius of the ellipsoid
    surface along that *same direction* from the center (exact for a sphere;
    a standard, cheap approximation of the true nearest-surface-point distance
    for a general ellipsoid -- good enough for comparing candidate shapes)."""
    r = np.linalg.norm(points_local, axis=1)
    r_safe = np.maximum(r, 1e-9)
    directions = points_local / r_safe[:, None]
    denom = np.linalg.norm(directions / semi_axes, axis=1)
    r_surface = 1.0 / np.maximum(denom, 1e-9)
    return np.abs(r - r_surface)


def _point_to_cylinder_surface_distance(points_local: np.ndarray, radius: float, half_height: float) -> np.ndarray:
    """Approximate distance from each point (in the cylinder's own centered
    frame, local axis 0 = the cylinder's length) to its surface -- the curved
    side plus two flat end caps. `axial`/`radial` decompose each point into its
    position along the length vs. its distance from the central axis; overhang
    past either bound combines via Pythagoras (same style as the box's
    corner-distance), matching the true nearest-point distance for a point
    outside near an edge, and a cheap lower-bound-ish approximation elsewhere
    -- consistent with the other two shapes' "good enough to compare" bar."""
    axial = points_local[:, 0]
    radial = np.linalg.norm(points_local[:, 1:3], axis=1)

    axial_overhang = np.maximum(np.abs(axial) - half_height, 0)
    radial_excess = np.maximum(radial - radius, 0)
    outside_dist = np.sqrt(axial_overhang**2 + radial_excess**2)

    inside_dist = np.clip(np.minimum(radius - radial, half_height - np.abs(axial)), 0, None)
    return np.where(outside_dist > 0, outside_dist, inside_dist)


def _fit_cylinder(points: np.ndarray) -> tuple[dict, float]:
    """PCA-oriented cylinder: local axis 0 (the principal axis with the largest
    spread, since `_pca_frame`'s SVD orders them descending) is the cylinder's
    length; radius is the median cross-section distance from that axis (robust
    to a few outlier depth points, same choice `_fit_ellipsoid` makes)."""
    centroid, rotation, local = _pca_frame(points)
    axial = local[:, 0]
    half_height = (axial.max() - axial.min()) / 2
    axial_center_offset = (axial.max() + axial.min()) / 2
    center = centroid + rotation @ np.array([axial_center_offset, 0.0, 0.0])

    local_centered = local.copy()
    local_centered[:, 0] -= axial_center_offset
    radius = float(np.median(np.linalg.norm(local_centered[:, 1:3], axis=1)))

    residual = float(np.mean(_point_to_cylinder_surface_distance(local_centered, radius, half_height)))
    descriptor = {
        KEY_KIND: KIND_CYLINDER,
        "center": center.tolist(),
        "radius": radius,
        "half_height": half_height,
        "rotation": rotation.tolist(),  # (3, 3), object-local -> body-space; local axis 0 = length
    }
    return descriptor, residual


def _fit_box(points: np.ndarray) -> tuple[dict, float]:
    """PCA-oriented bounding box: spans exactly to the projected extremes along
    each principal axis, so width/height/depth can all differ."""
    centroid, rotation, local = _pca_frame(points)
    mins, maxs = local.min(axis=0), local.max(axis=0)
    half_extents = (maxs - mins) / 2
    local_center_offset = (maxs + mins) / 2
    center = centroid + rotation @ local_center_offset

    residual = float(np.mean(_point_to_box_surface_distance(local - local_center_offset, half_extents)))
    descriptor = {
        KEY_KIND: KIND_BOX,
        "center": center.tolist(),
        "half_extents": half_extents.tolist(),
        "rotation": rotation.tolist(),  # (3, 3), object-local -> body-space
    }
    return descriptor, residual


def _fit_ellipsoid(points: np.ndarray) -> tuple[dict, float]:
    """PCA-oriented ellipsoid: each semi-axis is estimated from the point
    cloud's own spread along that principal axis (`std * sqrt(3)`, exact for
    points uniform on a sphere's surface, a standard approximation otherwise),
    so a genuinely elongated/flattened object doesn't get forced into a
    perfect sphere."""
    centroid, rotation, local = _pca_frame(points)
    n = len(points)
    singular_values = np.sqrt(np.sum(local**2, axis=0))
    std = singular_values / np.sqrt(max(n - 1, 1))
    semi_axes = std * np.sqrt(3.0)

    residual = float(np.mean(_point_to_ellipsoid_surface_distance(local, semi_axes)))
    descriptor = {
        KEY_KIND: KIND_ELLIPSOID,
        "center": centroid.tolist(),
        "semi_axes": semi_axes.tolist(),
        "rotation": rotation.tolist(),  # (3, 3), object-local -> body-space
    }
    return descriptor, residual


def _reject_depth_outliers(points: np.ndarray) -> np.ndarray:
    """Drops points far (on any axis) from the cloud's own median, using a
    robust spread estimate (MAD) instead of std/min/max so the rejection
    itself isn't skewed by the same outliers it's trying to catch. Falls back
    to the untrimmed cloud if rejection would leave too few points to fit
    (e.g. a genuinely tight object where the MAD is near zero)."""
    median = np.median(points, axis=0)
    scaled_mad = np.median(np.abs(points - median), axis=0) * 1.4826
    scaled_mad = np.maximum(scaled_mad, 1e-6)  # a flat/degenerate axis shouldn't reject everything
    keep = np.all(np.abs(points - median) <= _OUTLIER_REJECTION_MAD_MULTIPLIER * scaled_mad, axis=1)
    if np.count_nonzero(keep) < MIN_OBJECT_POINTS:
        return points
    return points[keep]


def fit_position_and_orientation(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Position + orientation only, no shape/dimension fit -- reuses the same
    outlier rejection and PCA machinery `fit_object_shape` itself builds on,
    for a caller that already knows the object's own fixed dimensions (e.g.
    re-deriving *where* the object is at some other frame, not what size it
    is) and just needs a fresh `(center, rotation)`.

    Returns `None` if there aren't enough points to trust a fit (mirrors
    `fit_object_shape`'s own `MIN_OBJECT_POINTS` gate, but returns a sentinel
    instead of raising -- a caller re-deriving pose at an arbitrary frame
    expects this to sometimes fail, e.g. a heavily hand-occluded grip, not a
    hard error).
    """
    if len(points) < MIN_OBJECT_POINTS:
        return None
    points = _reject_depth_outliers(points)
    centroid, rotation, _ = _pca_frame(points)
    return centroid, rotation


_FITTERS = {
    ObjectShapeHint.BOX: _fit_box,
    ObjectShapeHint.ELLIPSOID: _fit_ellipsoid,
    ObjectShapeHint.CYLINDER: _fit_cylinder,
}


def fit_object_shape(points: np.ndarray, shape_hint: ObjectShapeHint = ObjectShapeHint.AUTO) -> dict:
    """Fits a proxy primitive to the object's point cloud.

    Args:
        points: (N, 3) object point cloud, in the same metric space as the
            SMPL-X body (already scale/translation-aligned).
        shape_hint: `AUTO` picks whichever of box/ellipsoid/cylinder has the
            lower mean surface residual; forcing one of the three returns that
            shape regardless of fit quality.

    Returns a JSON-serializable tagged-union dict, one of:
    `{"kind": "box", "center": [...], "half_extents": [...], "rotation": [[...], ...]}`,
    `{"kind": "ellipsoid", "center": [...], "semi_axes": [...], "rotation": [[...], ...]}`,
    `{"kind": "cylinder", "center": [...], "radius": ..., "half_height": ..., "rotation": [[...], ...]}`.
    """
    if len(points) < MIN_OBJECT_POINTS:
        raise RuntimeError(
            f"only {len(points)} object points (need >= {MIN_OBJECT_POINTS}); "
            "the object may be mostly occluded/out of frame at the anchor"
        )

    points = _reject_depth_outliers(points)

    if shape_hint in _FITTERS:
        descriptor, _ = _FITTERS[shape_hint](points)
        return descriptor

    candidates = [fitter(points) for fitter in _FITTERS.values()]
    return min(candidates, key=lambda candidate: candidate[1])[0]


def sample_shape_surface(descriptor: dict, points_per_line: int = 20) -> np.ndarray:
    """Samples a wireframe outline of a fitted shape descriptor, for visual
    verification in a point-cloud preview (box: 12 edges; ellipsoid: a handful
    of latitude/longitude rings; cylinder: two end-cap rims + side meridians)
    -- not a filled surface, so it reads as an outline overlaid on the real
    point cloud rather than another solid blob.
    """
    if descriptor[KEY_KIND] == KIND_BOX:
        return _sample_box_wireframe(descriptor, points_per_line)
    if descriptor[KEY_KIND] == KIND_CYLINDER:
        return _sample_cylinder_wireframe(descriptor, points_per_line)
    return _sample_ellipsoid_wireframe(descriptor, points_per_line)


def _sample_box_wireframe(descriptor: dict, points_per_edge: int) -> np.ndarray:
    half_extents = np.array(descriptor["half_extents"])
    rotation = np.array(descriptor["rotation"])
    center = np.array(descriptor["center"])

    signs = np.array([-1.0, 1.0])
    corners_local = np.array(
        [[sx, sy, sz] for sx in signs for sy in signs for sz in signs]
    ) * half_extents
    # 12 edges: pairs of corners differing along exactly one axis.
    edges = [
        (i, j)
        for i in range(8)
        for j in range(i + 1, 8)
        if np.count_nonzero(corners_local[i] != corners_local[j]) == 1
    ]

    t = np.linspace(0.0, 1.0, points_per_edge)[:, None]
    lines_local = np.concatenate(
        [corners_local[i] * (1 - t) + corners_local[j] * t for i, j in edges], axis=0
    )
    return lines_local @ rotation.T + center


def _sample_ellipsoid_wireframe(descriptor: dict, points_per_ring: int, n_rings: int = 6) -> np.ndarray:
    semi_axes = np.array(descriptor["semi_axes"])
    rotation = np.array(descriptor["rotation"])
    center = np.array(descriptor["center"])

    theta = np.linspace(0, 2 * np.pi, points_per_ring, endpoint=False)
    unit_rings = []
    # Longitude rings (through the poles), evenly spaced around the "vertical" local axis.
    for phi in np.linspace(0, np.pi, n_rings, endpoint=False):
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        unit_rings.append(np.stack([x, y, z], axis=-1))
    # A few latitude rings for a recognizable "globe" outline.
    for elevation in np.linspace(-np.pi / 3, np.pi / 3, 3):
        r = np.cos(elevation)
        z = np.full(points_per_ring, np.sin(elevation))
        unit_rings.append(np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=-1))

    local = np.concatenate(unit_rings, axis=0) * semi_axes  # unit sphere -> ellipsoid, per-axis scale
    return local @ rotation.T + center


def _sample_cylinder_wireframe(descriptor: dict, points_per_ring: int, n_meridians: int = 8) -> np.ndarray:
    radius = descriptor["radius"]
    half_height = descriptor["half_height"]
    rotation = np.array(descriptor["rotation"])
    center = np.array(descriptor["center"])

    theta = np.linspace(0, 2 * np.pi, points_per_ring, endpoint=False)
    circle = np.stack([np.zeros_like(theta), radius * np.cos(theta), radius * np.sin(theta)], axis=-1)

    top = circle.copy(); top[:, 0] = half_height
    bottom = circle.copy(); bottom[:, 0] = -half_height

    meridian_theta = np.linspace(0, 2 * np.pi, n_meridians, endpoint=False)
    t = np.linspace(-1.0, 1.0, points_per_ring)[:, None]
    meridians = np.concatenate(
        [
            np.stack([t[:, 0] * half_height, np.full_like(t[:, 0], radius * np.cos(mt)),
                      np.full_like(t[:, 0], radius * np.sin(mt))], axis=-1)
            for mt in meridian_theta
        ],
        axis=0,
    )

    local = np.concatenate([top, bottom, meridians], axis=0)
    return local @ rotation.T + center
