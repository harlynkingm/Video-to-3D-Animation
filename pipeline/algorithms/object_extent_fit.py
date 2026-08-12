"""Object shape/extent fit: given the tracked object's back-projected point
cloud at the anchor frame (already in the same metric space as the SMPL-X
human body, see `similarity_transform.fit_scene_scale`), fits a simple
proxy primitive (oriented box, ellipsoid, or cylinder) that approximates its
shape. Ports the "object shape" half of open4dhoi's `make_hoi.py`, but fits a
primitive instead of resizing a reconstructed mesh (no SAM-3D-Objects on
this hardware).

Box and ellipsoid both get independent extents per axis (not a cube / a
single-radius sphere), a real "rectangular object" or "circular object" has
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
its residual is always an interior (inside-to-nearest-face) distance, the
"outside" branch of `_point_to_box_surface_distance` exists for reuse (e.g. a
future stage 8 contact-distance cost term), not because this fit itself ever
produces exterior points.

Before any of that, `_reject_depth_outliers` trims the raw point cloud: DA3
(like any monocular depth model) blurs/bleeds its estimate across a real
depth discontinuity, so a handful of pixels right on the object mask's own
silhouette edge can read meters deeper than the object's real surface. A
plain min/max (box) or std (ellipsoid) fit, and even the PCA orientation
every fitter starts from, is not robust to that tail, so it gets dropped
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
    local frame) to the box's surface, positive whether the point is outside
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
    for a general ellipsoid, good enough for comparing candidate shapes)."""
    r = np.linalg.norm(points_local, axis=1)
    r_safe = np.maximum(r, 1e-9)
    directions = points_local / r_safe[:, None]
    denom = np.linalg.norm(directions / semi_axes, axis=1)
    r_surface = 1.0 / np.maximum(denom, 1e-9)
    return np.abs(r - r_surface)


def _point_to_cylinder_surface_distance(points_local: np.ndarray, radius: float, half_height: float) -> np.ndarray:
    """Approximate distance from each point (in the cylinder's own centered
    frame, local axis 0 = the cylinder's length) to its surface, the curved
    side plus two flat end caps. `axial`/`radial` decompose each point into its
    position along the length vs. its distance from the central axis; overhang
    past either bound combines via Pythagoras (same style as the box's
    corner-distance), matching the true nearest-point distance for a point
    outside near an edge, and a cheap lower-bound-ish approximation elsewhere,
    consistent with the other two shapes' "good enough to compare" bar."""
    axial = points_local[:, 0]
    radial = np.linalg.norm(points_local[:, 1:3], axis=1)

    axial_overhang = np.maximum(np.abs(axial) - half_height, 0)
    radial_excess = np.maximum(radial - radius, 0)
    outside_dist = np.sqrt(axial_overhang**2 + radial_excess**2)

    inside_dist = np.clip(np.minimum(radius - radial, half_height - np.abs(axial)), 0, None)
    return np.where(outside_dist > 0, outside_dist, inside_dist)


def _fit_cylinder(points: np.ndarray) -> tuple[dict, float]:
    """PCA-oriented cylinder. Local axis 0 (the cylinder's length) is picked
    from the two axes *not* aligned with the camera's own viewing direction
    (`_depth_axis_index`), whichever of those two has the larger point-cloud
    spread, comparing only the two lateral axes, never the depth axis
    itself, since depth noise can otherwise spread the cloud further along
    the depth axis than the object's real length, getting mistaken for it.
    Radius is the median cross-section distance from the length axis
    (robust to a few outlier depth points, same choice `_fit_ellipsoid`
    makes). Both are typically overwritten by `correct_cylinder_extent_
    from_mask` right after, see that function's own docstring for why the
    raw point-cloud spread isn't trusted for either dimension, only for the
    length axis's *direction*."""
    centroid, rotation, local = _pca_frame(points)
    depth_idx = _depth_axis_index(rotation)
    length_idx, radial_idx = sorted(
        (i for i in range(3) if i != depth_idx),
        key=lambda i: local[:, i].max() - local[:, i].min(),
        reverse=True,
    )

    axial = local[:, length_idx]
    half_height = (axial.max() - axial.min()) / 2
    axial_center_offset = (axial.max() + axial.min()) / 2
    center = centroid + rotation[:, length_idx] * axial_center_offset

    radius = float(np.median(np.sqrt(local[:, radial_idx] ** 2 + local[:, depth_idx] ** 2)))

    reordered_rotation = rotation[:, [length_idx, radial_idx, depth_idx]]
    local_centered = local[:, [length_idx, radial_idx, depth_idx]].copy()
    local_centered[:, 0] -= axial_center_offset
    residual = float(np.mean(_point_to_cylinder_surface_distance(local_centered, radius, half_height)))
    descriptor = {
        KEY_KIND: KIND_CYLINDER,
        "center": center.tolist(),
        "radius": radius,
        "half_height": half_height,
        "rotation": reordered_rotation.tolist(),  # (3, 3), object-local -> body-space; local axis 0 = length
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
    """Position + orientation only, no shape/dimension fit, reuses the same
    outlier rejection and PCA machinery `fit_object_shape` itself builds on,
    for a caller that already knows the object's own fixed dimensions (e.g.
    re-deriving *where* the object is at some other frame, not what size it
    is) and just needs a fresh `(center, rotation)`.

    Returns `None` if there aren't enough points to trust a fit (mirrors
    `fit_object_shape`'s own `MIN_OBJECT_POINTS` gate, but returns a sentinel
    instead of raising, a caller re-deriving pose at an arbitrary frame
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


def _depth_axis_index(rotation: np.ndarray) -> int:
    """Which of a fitted shape's own 3 local axes (rotation's columns,
    object-local -> body-space, see `_pca_frame`) is most aligned with the
    camera's own viewing direction. `align_scene_scale`'s scale+translation
    fit is a pure per-axis scale between camera space and body space (no
    rotation involved), so the camera's forward direction is exactly the
    body-space Z axis regardless of the fitted anisotropic scale, no
    extra information beyond the shape's own already-fitted rotation is
    needed to find it."""
    rotation = np.asarray(rotation)
    return int(np.argmax(np.abs(rotation[2, :])))


def mask_metric_extent(mask: np.ndarray, depth: np.ndarray, K: np.ndarray) -> tuple[float, float] | None:
    """The object's own real-world lateral width/height (meters, camera
    space, not yet scaled into body space, see `correct_lateral_axes_from_
    mask`), measured directly from the mask's own oriented 2D bounding box
    and a representative depth, via the pinhole relation `real_size =
    pixel_size * depth / focal_length_px`.

    This avoids a bias in `_fit_ellipsoid`/`_fit_box`'s own std-based
    sizing: those infer spread statistically from the back-projected point
    cloud, which only samples a convex object's near-facing hemisphere, and
    that hemisphere's points are pixel-uniform, not surface-uniform (denser
    near the center, sparser near the silhouette edge), understating spread
    even along axes that aren't otherwise foreshortened (see `correct_
    foreshortened_axis` for the depth-axis half of this problem). The
    mask's own pixel extent has no such bias.

    `mask`, `depth` must already be at the same resolution; `K` must already
    be scaled to match (see `depth_unprojection.scale_intrinsics_to_
    resolution`). Returns `None` if the mask is empty or too small to fit a
    rectangle to.
    """
    import cv2  # local: see mask_circularity's own comment on why

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_OBJECT_POINTS:
        return None

    (_, _), (w_px, h_px), _ = cv2.minAreaRect(contour)
    object_depth = float(np.median(depth[mask]))
    focal_length_px = float(K[0, 0])
    if focal_length_px == 0:
        return None
    return w_px * object_depth / focal_length_px, h_px * object_depth / focal_length_px


def correct_lateral_axes_from_mask(descriptor: dict, mask_extent_m: tuple[float, float], scale_xy: float) -> dict:
    """Replaces a fitted shape's two non-depth-axis size fields with the
    mask's own directly-measured width/height (`mask_metric_extent`,
    converted from camera-space meters into body-space via `scale_xy`,
    the shared X/Y component of `similarity_transform.fit_scene_scale`'s own
    anisotropic scale), matched to the fitted axes by magnitude rank (the
    larger mask dimension replaces the larger fitted lateral axis) rather
    than exact 3D correspondence, simple, and good enough for a roughly
    front-facing view of a handheld object, without needing to project each
    fitted axis into image space to find its own exact 2D counterpart.

    Run this *before* `correct_foreshortened_axis`, not after: that
    function derives its own depth-axis estimate from whichever two axes
    are currently stored as "lateral", so it should see these better,
    directly-measured values, not the original std-based ones.

    Cylinder is returned unchanged here, it has its own sibling function,
    `correct_cylinder_extent_from_mask`, since it only exposes two size
    fields (`radius`/`half_height`) rather than three, so the "which fields
    are lateral" bookkeeping this function does doesn't apply as-is.
    """
    kind = descriptor[KEY_KIND]
    if kind == KIND_CYLINDER:
        return descriptor

    rotation = np.array(descriptor["rotation"])
    depth_idx = _depth_axis_index(rotation)
    size_key = "half_extents" if kind == KIND_BOX else "semi_axes"
    sizes = list(descriptor[size_key])
    lateral_idxs = [i for i in range(3) if i != depth_idx]

    w_m, h_m = mask_extent_m
    mask_half_sorted = sorted([w_m / (2 * scale_xy), h_m / (2 * scale_xy)], reverse=True)
    fitted_order = sorted(lateral_idxs, key=lambda i: sizes[i], reverse=True)
    for rank, axis_idx in enumerate(fitted_order):
        sizes[axis_idx] = mask_half_sorted[rank]

    corrected = dict(descriptor)
    corrected[size_key] = sizes
    return corrected


def correct_cylinder_extent_from_mask(descriptor: dict, mask_extent_m: tuple[float, float], scale_xy: float) -> dict:
    """Cylinder's own sibling to `correct_lateral_axes_from_mask`, but
    unlike box/ellipsoid (which keep their depth-axis size and only replace
    it via `correct_foreshortened_axis`'s harmonic-mean formula), this
    replaces *both* of the cylinder's size fields, `half_height` and
    `radius`, directly from the mask. A cylinder only ever exposes two size
    numbers, and `_fit_cylinder` already excludes the depth axis entirely
    when picking which two axes they come from (see that function's own
    docstring), so there's no third, depth-derived value worth preserving
    here the way there is for box/ellipsoid.

    Checked against real reference dimensions: the point-cloud-derived
    radius overestimates a small, glossy object's diameter by ~14-20%, most
    likely a depth-estimation bias on small, low-texture, reflective
    surfaces (ruled out mask-pixel precision, the error got slightly
    worse, not better, at a higher DA3 `process_res`). The same bias can
    inflate `half_height` far worse, up to badly mistaking the depth axis
    itself for the length axis (see `_fit_cylinder`). The mask's own pixel
    extent carries no such bias, assumed here to be the object's true
    width/length as long as the view is roughly front-on or side-on (a
    reasonable default for a handheld object, not a geometric guarantee),
    the larger of the two mask dimensions becoming `half_height` and the
    smaller becoming `radius`.
    """
    if descriptor[KEY_KIND] != KIND_CYLINDER:
        return descriptor

    w_m, h_m = mask_extent_m
    half_height, radius = sorted((w_m / (2 * scale_xy), h_m / (2 * scale_xy)), reverse=True)

    corrected = dict(descriptor)
    corrected["half_height"] = half_height
    corrected["radius"] = radius
    return corrected


def correct_foreshortened_axis(descriptor: dict) -> dict:
    """A single camera view only ever sees a convex object's near-facing
    surface, so whichever fitted axis is most aligned with the camera's own
    viewing direction (`_depth_axis_index`) is foreshortened toward roughly
    half the object's true extent along that axis, a systematic bias, not
    noise, so multi-frame averaging alone can't fix it.

    Corrects that axis using the harmonic mean of the other two (real,
    laterally-observed) dimensions: equal to their shared value when
    similar (a round or square cross-section), leaning toward the smaller
    of the two the more they diverge (an elongated/flattened object's own
    depth tends to track its narrower visible dimension, not its wider one,
    a book's thickness follows its short edge, not its long one). Center
    and rotation are untouched, only the one corrected axis's size changes.

    Cylinder is returned unchanged: it only ever exposes two size fields
    (`radius`/`half_height`), both of which `_fit_cylinder` already assigns
    from the two axes *not* aligned with the camera (see that function's own
    docstring), there's no third, depth-axis-aligned field left over here
    to foreshorten-correct the way box/ellipsoid's third axis is.
    """
    kind = descriptor[KEY_KIND]
    if kind == KIND_CYLINDER:
        return descriptor

    rotation = np.array(descriptor["rotation"])
    depth_idx = _depth_axis_index(rotation)
    size_key = "half_extents" if kind == KIND_BOX else "semi_axes"
    sizes = list(descriptor[size_key])
    lateral = [s for i, s in enumerate(sizes) if i != depth_idx]
    a, b = lateral
    sizes[depth_idx] = 2 * a * b / (a + b) if (a + b) > 0 else 0.0

    corrected = dict(descriptor)
    corrected[size_key] = sizes
    return corrected


# Verified analytically and numerically (a Monte Carlo check against the
# closed form matched to 4 decimal places): the mean depth of a sphere's
# near-facing hemisphere, sampled image-plane-uniformly (as a real depth
# back-projection does, see mask_metric_extent's own docstring for why
# that sampling isn't surface-uniform), is exactly -2/3 of the radius from
# the sphere's own center. See correct_center_depth's own docstring.
_HEMISPHERE_CENTROID_DEPTH_FACTOR = 2 / 3


def correct_center_depth(descriptor: dict) -> dict:
    """The fitted `center` is the mean position of one camera view's own
    back-projected points, which, for the same reason `correct_
    foreshortened_axis`'s depth-axis size is biased, only ever samples a
    convex object's near-facing surface. That surface's mean depth sits
    closer to the camera than the object's true center: for a sphere viewed
    head-on with image-plane-uniform sampling, the mean visible-surface
    depth works out to exactly 2/3 of the object's own depth-axis half-
    extent short of the true center (`_HEMISPHERE_CENTROID_DEPTH_FACTOR`,
    derived in closed form and confirmed with a Monte Carlo check). Shifts
    `center` by that amount, away from the camera, along the object's own
    already-identified depth axis.

    Exact for a sphere/ellipsoid under this sampling model; an approximation
    for box (whose near-facing surface is flatter than a dome, so the real
    correction is somewhat larger for a face-on view), applied anyway as a
    reasonable first pass. Skipped for cylinder: `_fit_cylinder` does now
    identify its own depth axis (see that function's own docstring), but the
    2/3-of-radius derivation above is specific to a sphere's hemisphere, not
    yet verified for a cylinder's curved surface, left unimplemented
    rather than reusing an unverified formula.

    Run this *after* `correct_foreshortened_axis` and after any multi-frame
    size aggregation, the shift should use the most trustworthy size
    estimate available.
    """
    kind = descriptor[KEY_KIND]
    if kind == KIND_CYLINDER:
        return descriptor

    rotation = np.array(descriptor["rotation"])
    depth_idx = _depth_axis_index(rotation)
    size_key = "half_extents" if kind == KIND_BOX else "semi_axes"
    depth_size = descriptor[size_key][depth_idx]

    depth_axis = rotation[:, depth_idx].copy()
    if depth_axis[2] < 0:  # keep pointing away from the camera (body-space +Z, see _depth_axis_index)
        depth_axis = -depth_axis

    center = np.array(descriptor["center"]) + _HEMISPHERE_CENTROID_DEPTH_FACTOR * depth_size * depth_axis
    corrected = dict(descriptor)
    corrected["center"] = center.tolist()
    return corrected


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

    This fits whatever point cloud it's given, full stop, it does not
    assume single-camera-view geometry (a synthetic full-surface test cloud
    is exactly as valid an input as a real one-sided depth back-projection).
    A caller whose points genuinely come from one camera view (every real
    pipeline caller) should apply `correct_foreshortened_axis` to the
    result, kept as an explicit, separate step rather than folded in here,
    since "fit a shape to points" and "correct for single-view foreshortening"
    are different concerns with different validity conditions.
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


def mask_circularity(mask: np.ndarray) -> float:
    """4*pi*area/perimeter^2 of a 2D binary mask's largest contour, 1.0 for
    a perfect circle, lower for anything elongated, notched, or fragmented.
    Used to rank candidate frames for `align_scene_scale`'s multi-frame size
    aggregation: a real round object's silhouette should read close to
    circular when it's cleanly, fully visible, and reads lower when it's
    motion-blurred (smeared, elongated) or occluded (a bite taken out of the
    silhouette by a gripping hand), either failure mode breaks circularity,
    even though neither reliably breaks SAM's own per-frame tracking
    confidence (which answers "is the object still here", not "is this
    frame's silhouette clean"). Not a guarantee on its own (a partial,
    occluded sliver can still look roughly round), just a free, cheap,
    real filter used ahead of a robust multi-frame combine, not in place
    of one. Returns 0.0 for an empty or degenerate mask."""
    import cv2  # local: cv2 isn't installed in stage 10's own (bpy-only) env,
    # which imports this module for its KEY_KIND/KIND_* constants alone

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return float(4 * np.pi * area / (perimeter ** 2))


def mask_solidity(mask: np.ndarray) -> float:
    """contour_area / convex_hull_area of a 2D binary mask's largest contour,
    1.0 for a fully convex silhouette, lower once something bites a
    concave notch out of it. The shape-agnostic sibling of `mask_circularity`,
    a box, cylinder, *and* ellipsoid are all convex solids, so any of
    their clean, unoccluded silhouettes should read close to 1.0 regardless
    of which one it is, while a gripping hand's occlusion breaks convexity
    the same way for all three. What it does NOT catch, unlike circularity:
    motion blur, a linear blur smears a silhouette into an elongated but
    still-convex "stadium" shape, so a blurred frame can still read fully
    solid. Use `mask_circularity` instead when the object is already known
    (or expected) to be round, it catches both failure modes there, at
    the cost of not generalizing to box/cylinder shapes. Returns 0.0 for an
    empty or degenerate mask."""
    import cv2  # local: see mask_circularity's own comment on why

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area == 0:
        return 0.0
    return float(area / hull_area)


def equivalent_radius(descriptor: dict) -> float:
    """A single rotation-invariant "effective size" for any fitted shape kind,
    the geometric mean of its own size dimensions, useful for a caller
    that just needs one overall scale without resolving which axis in one
    frame's fit corresponds to which axis in another's (PCA's own axis
    assignment has no stable answer frame to frame for a near-symmetric
    object, the same problem this project already hit once for
    orientation, in the free-tracking rotation-instability fix)."""
    kind = descriptor[KEY_KIND]
    if kind == KIND_BOX:
        hx, hy, hz = descriptor["half_extents"]
        return float((hx * hy * hz) ** (1 / 3))
    if kind == KIND_ELLIPSOID:
        a, b, c = descriptor["semi_axes"]
        return float((a * b * c) ** (1 / 3))
    if kind == KIND_CYLINDER:
        radius, half_height = descriptor["radius"], descriptor["half_height"]
        return float((radius * radius * half_height) ** (1 / 3))
    raise ValueError(f"unknown shape kind: {kind!r}")


def sample_shape_surface(descriptor: dict, points_per_line: int = 20) -> np.ndarray:
    """Samples a wireframe outline of a fitted shape descriptor, for visual
    verification in a point-cloud preview (box: 12 edges; ellipsoid: a handful
    of latitude/longitude rings; cylinder: two end-cap rims + side meridians),
    not a filled surface, so it reads as an outline overlaid on the real
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
