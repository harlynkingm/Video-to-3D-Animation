"""Unit tests for `object_extent_fit`: pure-numpy fits on synthetic point
clouds with a *known* shape, no GPU/checkpoints needed -- always runs.
"""

from __future__ import annotations

import numpy as np

from pipeline.algorithms.object_extent_fit import (
    KIND_BOX,
    KIND_CYLINDER,
    KIND_ELLIPSOID,
    MIN_OBJECT_POINTS,
    _depth_axis_index,
    _reject_depth_outliers,
    correct_center_depth,
    correct_cylinder_extent_from_mask,
    correct_foreshortened_axis,
    correct_lateral_axes_from_mask,
    equivalent_radius,
    fit_object_shape,
    fit_position_and_orientation,
    mask_circularity,
    mask_metric_extent,
    mask_solidity,
    sample_shape_surface,
)
from pipeline.progress_tracker import ObjectShapeHint


def _box_point_cloud(center, half_extents, rotation, n=4000, rng=None):
    """A dense random surface sample of a real oriented box (points ON the six
    faces, not filled interior) -- the kind of front-facing-surface cloud a
    real depth back-projection would actually produce."""
    rng = rng or np.random.default_rng(0)
    axis = rng.integers(0, 3, n)
    sign = rng.choice([-1.0, 1.0], n)
    local = rng.uniform(-1.0, 1.0, (n, 3)) * half_extents
    local[np.arange(n), axis] = sign * half_extents[axis]
    return local @ rotation.T + center


def _ellipsoid_point_cloud(center, semi_axes, n=6000, rng=None):
    """Uniform directions on the unit sphere, scaled per-axis -- a valid
    surface sample of the ellipsoid `(x/a)^2+(y/b)^2+(z/c)^2=1` (not
    perfectly uniform by surface area, but a standard, good-enough synthetic
    fixture)."""
    rng = rng or np.random.default_rng(0)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions * semi_axes + center


def _cylinder_point_cloud(center, rotation, radius, half_height, n=6000, rng=None):
    """A surface sample of a real cylinder: mostly the curved side, plus the
    two flat end caps -- local axis 0 is the length, matching
    `_fit_cylinder`'s own convention."""
    rng = rng or np.random.default_rng(0)
    n_side = int(n * 0.7)
    n_caps = n - n_side

    theta = rng.uniform(0, 2 * np.pi, n_side)
    axial = rng.uniform(-half_height, half_height, n_side)
    side = np.stack([axial, radius * np.cos(theta), radius * np.sin(theta)], axis=-1)

    cap_theta = rng.uniform(0, 2 * np.pi, n_caps)
    cap_r = radius * np.sqrt(rng.uniform(0, 1, n_caps))  # uniform over disk area
    cap_sign = rng.choice([-1.0, 1.0], n_caps)
    caps = np.stack([cap_sign * half_height, cap_r * np.cos(cap_theta), cap_r * np.sin(cap_theta)], axis=-1)

    local = np.concatenate([side, caps], axis=0)
    return local @ rotation.T + center


def test_box_fit_recovers_known_dimensions():
    center = np.array([0.1, -0.2, 2.0])
    half_extents = np.array([0.3, 0.15, 0.1])
    rotation = np.eye(3)  # axis-aligned, so recovered half_extents map directly
    points = _box_point_cloud(center, half_extents, rotation)

    descriptor = fit_object_shape(points, shape_hint=ObjectShapeHint.BOX)

    assert descriptor["kind"] == KIND_BOX
    assert np.allclose(descriptor["center"], center, atol=0.02)
    assert np.allclose(sorted(descriptor["half_extents"]), sorted(half_extents), atol=0.02)


def test_ellipsoid_fit_recovers_independent_semi_axes():
    """The real point of an ellipsoid over a plain sphere: width/height/depth
    can all differ. Uses a clearly elongated shape (not a perfect sphere) so a
    bug that collapses all three axes to one value would fail this test."""
    center = np.array([0.0, 0.0, 1.5])
    semi_axes = np.array([0.25, 0.12, 0.08])
    points = _ellipsoid_point_cloud(center, semi_axes)

    descriptor = fit_object_shape(points, shape_hint=ObjectShapeHint.ELLIPSOID)

    assert descriptor["kind"] == KIND_ELLIPSOID
    assert np.allclose(descriptor["center"], center, atol=0.02)
    assert np.allclose(sorted(descriptor["semi_axes"]), sorted(semi_axes), atol=0.03)
    # Not degenerated into a sphere -- the three recovered axes must actually differ.
    recovered = sorted(descriptor["semi_axes"])
    assert recovered[2] - recovered[0] > 0.05


def test_cylinder_fit_recovers_known_dimensions():
    center = np.array([0.1, -0.1, 2.0])
    radius = 0.05
    half_height = 0.2
    rotation = np.eye(3)
    points = _cylinder_point_cloud(center, rotation, radius, half_height)

    descriptor = fit_object_shape(points, shape_hint=ObjectShapeHint.CYLINDER)

    assert descriptor["kind"] == KIND_CYLINDER
    assert np.allclose(descriptor["center"], center, atol=0.02)
    assert abs(descriptor["radius"] - radius) < 0.01
    assert abs(descriptor["half_height"] - half_height) < 0.02


def test_cylinder_fit_ignores_depth_axis_noise_when_picking_the_length_axis():
    """The real bug this guards against: a small/distant/reflective object's
    DA3 depth can be noisier than the object's own real length, spreading
    the point cloud further along the camera's own viewing direction (world
    Z here, identity rotation) than along the object's true length axis. The
    old "largest spread wins, no matter which axis" rule would have picked
    that noisy depth axis as the cylinder's length; excluding the depth axis
    from the comparison (see `_fit_cylinder`'s own docstring) should keep
    the length axis on the real one (world X) regardless."""
    center = np.array([0.0, 0.0, 2.0])
    radius = 0.04
    half_height = 0.08
    points = _cylinder_point_cloud(center, np.eye(3), radius, half_height)

    rng = np.random.default_rng(4)
    points = points.copy()
    points[:, 2] += rng.normal(scale=0.15, size=len(points))  # depth noise bigger than the real length

    descriptor = fit_object_shape(points, shape_hint=ObjectShapeHint.CYLINDER)

    length_axis = np.array(descriptor["rotation"])[:, 0]
    assert abs(length_axis[0]) > 0.9  # still ~world X, not swallowed by the Z noise


def test_auto_picks_cylinder_for_a_bottle_shaped_cloud():
    points = _cylinder_point_cloud(np.array([0.0, 0.0, 2.0]), np.eye(3), radius=0.04, half_height=0.15)
    descriptor = fit_object_shape(points, shape_hint=ObjectShapeHint.AUTO)
    assert descriptor["kind"] == KIND_CYLINDER


def test_auto_picks_box_for_a_boxy_cloud():
    points = _box_point_cloud(np.array([0.0, 0.0, 2.0]), np.array([0.3, 0.1, 0.2]), np.eye(3))
    descriptor = fit_object_shape(points, shape_hint=ObjectShapeHint.AUTO)
    assert descriptor["kind"] == KIND_BOX


def test_auto_picks_ellipsoid_for_an_elongated_cloud():
    points = _ellipsoid_point_cloud(np.array([0.0, 0.0, 2.0]), np.array([0.25, 0.12, 0.08]))
    descriptor = fit_object_shape(points, shape_hint=ObjectShapeHint.AUTO)
    assert descriptor["kind"] == KIND_ELLIPSOID


def test_shape_hint_forces_the_requested_kind_even_against_a_worse_fit():
    # A near-perfect ellipsoid: forcing "box" should still return a box descriptor.
    points = _ellipsoid_point_cloud(np.array([0.0, 0.0, 2.0]), np.array([0.15, 0.15, 0.15]))
    descriptor = fit_object_shape(points, shape_hint=ObjectShapeHint.BOX)
    assert descriptor["kind"] == KIND_BOX


def test_too_few_points_raises():
    points = np.zeros((3, 3))
    try:
        fit_object_shape(points)
        assert False, "expected a RuntimeError"
    except RuntimeError:
        pass


def test_box_fit_ignores_a_depth_bleed_outlier_tail():
    """Simulates the real bug: a small percentage of pixels on the object
    mask's own silhouette edge get a DA3 depth reading meters deeper than the
    object's real surface (monocular depth bleeding across the discontinuity
    with whatever is behind the object). Without outlier rejection this drags
    the box's depth extent backward with it."""
    center = np.array([0.1, -0.2, 2.0])
    half_extents = np.array([0.3, 0.15, 0.1])
    rotation = np.eye(3)
    points = _box_point_cloud(center, half_extents, rotation)

    rng = np.random.default_rng(1)
    n_outliers = int(len(points) * 0.03)
    outliers = points[:n_outliers].copy()
    outliers[:, 2] += rng.uniform(0.5, 1.5, n_outliers)  # bled meters deeper
    points_with_outliers = np.concatenate([points, outliers], axis=0)

    descriptor = fit_object_shape(points_with_outliers, shape_hint=ObjectShapeHint.BOX)

    assert np.allclose(descriptor["center"], center, atol=0.05)
    assert np.allclose(sorted(descriptor["half_extents"]), sorted(half_extents), atol=0.05)


def test_reject_depth_outliers_drops_a_far_tail_keeps_the_dense_core():
    rng = np.random.default_rng(2)
    core = rng.normal(loc=[0.0, 0.0, 2.0], scale=0.02, size=(500, 3))
    tail = rng.normal(loc=[0.0, 0.0, 3.5], scale=0.02, size=(20, 3))  # far outliers on Z
    points = np.concatenate([core, tail], axis=0)

    kept = _reject_depth_outliers(points)

    assert len(kept) == len(core)
    assert kept[:, 2].max() < 2.5


def test_reject_depth_outliers_falls_back_when_trim_would_starve_the_fit():
    """A genuinely sparse/uniform cloud (no real dense core) shouldn't lose
    most of its points to rejection -- better to fit the untrimmed cloud than
    fail outright."""
    rng = np.random.default_rng(3)
    points = rng.uniform(-1.0, 1.0, (MIN_OBJECT_POINTS + 5, 3))

    kept = _reject_depth_outliers(points)

    assert len(kept) == len(points)


def test_fit_position_and_orientation_recovers_center_and_rotation():
    center = np.array([0.1, -0.2, 2.0])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90deg about Z
    points = _box_point_cloud(center, np.array([0.3, 0.1, 0.2]), rotation)

    result = fit_position_and_orientation(points)

    assert result is not None
    fitted_center, fitted_rotation = result
    assert np.allclose(fitted_center, center, atol=0.02)
    # PCA's own axis order/sign is ambiguous (see hoi_object_pose's own
    # axis-disambiguation for why) -- checking the *plane spanned* by the
    # fitted rotation matches the real one is what's actually guaranteed here,
    # not that the columns land in the same order/sign as the input.
    assert np.allclose(fitted_rotation @ fitted_rotation.T, np.eye(3), atol=1e-6)
    assert abs(np.linalg.det(fitted_rotation) - 1.0) < 1e-6


def test_fit_position_and_orientation_returns_none_for_too_few_points():
    points = np.zeros((MIN_OBJECT_POINTS - 1, 3))
    assert fit_position_and_orientation(points) is None


def test_sample_shape_surface_box_returns_points_on_the_surface():
    center = np.array([0.1, -0.2, 2.0])
    half_extents = np.array([0.3, 0.15, 0.1])
    descriptor = {
        "kind": KIND_BOX,
        "center": center.tolist(),
        "half_extents": half_extents.tolist(),
        "rotation": np.eye(3).tolist(),
    }
    wireframe = sample_shape_surface(descriptor, points_per_line=10)

    local = wireframe - center
    # Every wireframe point should lie on at least one face (within numerical tolerance).
    on_a_face = np.any(np.isclose(np.abs(local), half_extents, atol=1e-6), axis=1)
    assert on_a_face.all()
    assert np.all(np.abs(local) <= half_extents[None, :] + 1e-6)


def test_sample_shape_surface_ellipsoid_returns_points_on_the_surface():
    center = np.array([0.0, 0.0, 2.0])
    semi_axes = np.array([0.25, 0.12, 0.08])
    descriptor = {
        "kind": KIND_ELLIPSOID,
        "center": center.tolist(),
        "semi_axes": semi_axes.tolist(),
        "rotation": np.eye(3).tolist(),
    }
    wireframe = sample_shape_surface(descriptor, points_per_line=16)

    local = wireframe - center
    normalized_radius = np.linalg.norm(local / semi_axes, axis=1)
    assert np.allclose(normalized_radius, 1.0, atol=1e-6)


def test_sample_shape_surface_cylinder_returns_points_on_the_surface():
    center = np.array([0.1, -0.1, 2.0])
    radius = 0.05
    half_height = 0.2
    descriptor = {
        "kind": KIND_CYLINDER,
        "center": center.tolist(),
        "radius": radius,
        "half_height": half_height,
        "rotation": np.eye(3).tolist(),
    }
    wireframe = sample_shape_surface(descriptor, points_per_line=12)

    local = wireframe - center
    axial, radial = local[:, 0], np.linalg.norm(local[:, 1:3], axis=1)
    assert np.allclose(radial, radius, atol=1e-6)
    assert np.all(np.abs(axial) <= half_height + 1e-6)


def _disk_mask(radius: int, size: int = 200) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    center = size // 2
    return ((xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2).astype(np.uint8)


def test_mask_circularity_of_a_real_circle_is_close_to_one():
    # cv2's own perimeter measure overestimates a rasterized (stair-stepped)
    # circle's true circumference, so even a perfect disk caps out around
    # ~0.88-0.89 here -- matches the real range measured on the basketball
    # clip's own cleanest frames, not a bug in the metric.
    mask = _disk_mask(radius=40)
    assert mask_circularity(mask) > 0.85


def test_mask_circularity_of_an_elongated_shape_is_lower():
    size = 200
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[90:110, 20:180] = 1  # a thin horizontal bar -- far from circular
    assert mask_circularity(mask) < 0.5


def test_mask_circularity_of_a_notched_circle_is_lower_than_the_full_circle():
    # A disk with a bite taken out of it (like a hand gripping part of a
    # ball's silhouette) should read less circular than the full disk.
    full = _disk_mask(radius=40)
    notched = full.copy()
    notched[100:140, 60:140] = 0
    assert mask_circularity(notched) < mask_circularity(full)


def test_mask_circularity_of_an_empty_mask_is_zero():
    assert mask_circularity(np.zeros((50, 50), dtype=np.uint8)) == 0.0


def test_equivalent_radius_of_a_sphere_matches_its_own_radius():
    descriptor = {"kind": KIND_ELLIPSOID, "semi_axes": [0.12, 0.12, 0.12]}
    assert np.isclose(equivalent_radius(descriptor), 0.12)


def test_equivalent_radius_is_the_geometric_mean_for_an_anisotropic_ellipsoid():
    descriptor = {"kind": KIND_ELLIPSOID, "semi_axes": [0.6, 0.2, 0.15]}
    expected = (0.6 * 0.2 * 0.15) ** (1 / 3)
    assert np.isclose(equivalent_radius(descriptor), expected)


def test_equivalent_radius_covers_box_and_cylinder_kinds():
    box = {"kind": KIND_BOX, "half_extents": [0.1, 0.2, 0.3]}
    assert np.isclose(equivalent_radius(box), (0.1 * 0.2 * 0.3) ** (1 / 3))

    cylinder = {"kind": KIND_CYLINDER, "radius": 0.05, "half_height": 0.2}
    assert np.isclose(equivalent_radius(cylinder), (0.05 * 0.05 * 0.2) ** (1 / 3))


def _box_mask(size: int = 200, w: int = 100, h: int = 60) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    cy, cx = size // 2, size // 2
    mask[cy - h // 2:cy + h // 2, cx - w // 2:cx + w // 2] = 1
    return mask


def test_mask_solidity_of_a_clean_box_is_close_to_one():
    assert mask_solidity(_box_mask()) > 0.99


def test_mask_solidity_of_a_notched_box_is_lower_than_the_clean_box():
    # A hand gripping part of a box's silhouette bites a concave notch out
    # of it -- the exact case mask_solidity is meant to catch, for any
    # convex shape kind (box, cylinder, or ellipsoid alike).
    clean = _box_mask()  # spans y in [70, 130), x in [50, 150)
    notched = clean.copy()
    notched[70:80, 90:110] = 0  # cuts into the top edge -- a real boundary concavity,
    # unlike a fully interior hole (which findContours' RETR_EXTERNAL wouldn't see at all)
    assert mask_solidity(notched) < mask_solidity(clean)


def test_mask_solidity_does_not_catch_motion_blur_elongation():
    """Documents a real limitation, not just a passing case: a linear-blur
    smear of a box is still convex (a "stadium" shape), so solidity alone
    can't tell a blurred frame from a clean one the way mask_circularity
    can for a round object -- see this function's own docstring."""
    clean = _box_mask(w=100, h=60)
    blurred = _box_mask(w=180, h=60)  # smeared wider, but still a solid rectangle
    assert mask_solidity(blurred) > 0.99  # still reads as fully solid


def test_mask_solidity_of_an_empty_mask_is_zero():
    assert mask_solidity(np.zeros((50, 50), dtype=np.uint8)) == 0.0


def test_depth_axis_index_picks_whichever_column_is_camera_aligned():
    # Camera-forward is body-space Z (see _depth_axis_index's own docstring).
    # Local axis 1 (column 1) is the one pointing along Z here.
    rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])
    assert _depth_axis_index(rotation) == 1


def test_depth_axis_index_with_identity_rotation_is_the_last_axis():
    assert _depth_axis_index(np.eye(3)) == 2


def test_correct_foreshortened_axis_replaces_the_camera_axis_with_the_harmonic_mean():
    # Identity rotation -> axis 2 (Z) is the depth axis (see above). Real
    # lateral axes are 0.6 and 0.2; a fabricated foreshortened depth
    # reading of 0.05 should be replaced with their harmonic mean, not left
    # alone or replaced with their arithmetic mean.
    descriptor = {
        "kind": KIND_ELLIPSOID,
        "center": [1.0, 2.0, 3.0],
        "semi_axes": [0.6, 0.2, 0.05],
        "rotation": np.eye(3).tolist(),
    }
    corrected = correct_foreshortened_axis(descriptor)
    expected_depth = 2 * 0.6 * 0.2 / (0.6 + 0.2)
    assert np.isclose(corrected["semi_axes"][2], expected_depth)
    assert corrected["semi_axes"][0] == 0.6  # lateral axes untouched
    assert corrected["semi_axes"][1] == 0.2
    assert corrected["center"] == descriptor["center"]
    assert corrected["rotation"] == descriptor["rotation"]


def test_correct_foreshortened_axis_leaves_a_genuinely_matching_depth_axis_unchanged():
    # If the "foreshortened" axis already happens to equal the harmonic mean
    # of the other two, correction is a no-op -- not a guarantee anything was
    # wrong, just confirms the formula doesn't introduce spurious drift.
    descriptor = {
        "kind": KIND_ELLIPSOID,
        "center": [0.0, 0.0, 0.0],
        "semi_axes": [0.1, 0.1, 0.1],
        "rotation": np.eye(3).tolist(),
    }
    corrected = correct_foreshortened_axis(descriptor)
    assert np.allclose(corrected["semi_axes"], descriptor["semi_axes"])


def test_correct_foreshortened_axis_works_on_box_half_extents_too():
    descriptor = {
        "kind": KIND_BOX,
        "center": [0.0, 0.0, 0.0],
        "half_extents": [0.5, 0.3, 0.02],
        "rotation": np.eye(3).tolist(),
    }
    corrected = correct_foreshortened_axis(descriptor)
    expected_depth = 2 * 0.5 * 0.3 / (0.5 + 0.3)
    assert np.isclose(corrected["half_extents"][2], expected_depth)
    assert corrected["half_extents"][:2] == [0.5, 0.3]


def test_correct_foreshortened_axis_leaves_cylinder_unchanged():
    """Cylinder's own `radius` already blends both radial directions into
    one median distance -- there's no separate lateral measurement left to
    correct from, so it's returned untouched (a documented, open
    approximation, not a silent guess)."""
    descriptor = {
        "kind": KIND_CYLINDER,
        "center": [0.0, 0.0, 0.0],
        "radius": 0.05,
        "half_height": 0.2,
        "rotation": np.eye(3).tolist(),
    }
    corrected = correct_foreshortened_axis(descriptor)
    assert corrected == descriptor


def test_mask_metric_extent_recovers_a_known_real_size():
    # Pinhole: pixel_size = real_size * focal_length_px / depth. Pick real
    # width=0.24m, height=0.16m, depth=2.0m, focal_length_px=500 -> 60x40px.
    focal_length_px = 500.0
    depth_value = 2.0
    real_w, real_h = 0.24, 0.16
    pixel_w = round(real_w * focal_length_px / depth_value)  # 60
    pixel_h = round(real_h * focal_length_px / depth_value)  # 40

    size = 200
    mask = np.zeros((size, size), dtype=np.uint8)
    cy, cx = size // 2, size // 2
    mask[cy - pixel_h // 2:cy + pixel_h // 2, cx - pixel_w // 2:cx + pixel_w // 2] = 1
    depth = np.full((size, size), depth_value)
    K = np.array([[focal_length_px, 0, size / 2], [0, focal_length_px, size / 2], [0, 0, 1.0]])

    result = mask_metric_extent(mask, depth, K)
    assert result is not None
    w_m, h_m = sorted(result, reverse=True)
    assert np.isclose(w_m, real_w, atol=0.01)
    assert np.isclose(h_m, real_h, atol=0.01)


def test_mask_metric_extent_returns_none_for_an_empty_mask():
    mask = np.zeros((100, 100), dtype=np.uint8)
    depth = np.full((100, 100), 2.0)
    K = np.array([[500.0, 0, 50], [0, 500.0, 50], [0, 0, 1.0]])
    assert mask_metric_extent(mask, depth, K) is None


def test_correct_lateral_axes_from_mask_matches_by_magnitude_rank():
    # Identity rotation -> axis 2 is the depth axis (see _depth_axis_index
    # tests above); axes 0 and 1 are lateral. Fitted axis 0 (0.05) is
    # currently the *smaller* lateral value, axis 1 (0.3) the larger --
    # the mask's own larger measured dimension should replace axis 1, not
    # axis 0, regardless of index order.
    descriptor = {
        "kind": KIND_ELLIPSOID,
        "center": [0.0, 0.0, 0.0],
        "semi_axes": [0.05, 0.3, 0.2],  # depth axis (idx 2) = 0.2, arbitrary/wrong for now
        "rotation": np.eye(3).tolist(),
    }
    mask_extent_m = (0.5, 0.08)  # full width/height in meters, order doesn't matter
    corrected = correct_lateral_axes_from_mask(descriptor, mask_extent_m, scale_xy=1.0)

    assert np.isclose(corrected["semi_axes"][1], 0.25)  # larger fitted axis <- larger mask dim / 2
    assert np.isclose(corrected["semi_axes"][0], 0.04)  # smaller fitted axis <- smaller mask dim / 2
    assert corrected["semi_axes"][2] == 0.2  # depth axis untouched here
    assert corrected["center"] == descriptor["center"]
    assert corrected["rotation"] == descriptor["rotation"]


def test_correct_lateral_axes_from_mask_divides_by_scale_xy():
    descriptor = {
        "kind": KIND_ELLIPSOID,
        "center": [0.0, 0.0, 0.0],
        "semi_axes": [0.05, 0.3, 0.2],
        "rotation": np.eye(3).tolist(),
    }
    corrected = correct_lateral_axes_from_mask(descriptor, (0.5, 0.08), scale_xy=2.0)
    assert np.isclose(corrected["semi_axes"][1], 0.125)  # 0.5 / 2 / 2.0


def test_correct_lateral_axes_from_mask_leaves_cylinder_unchanged():
    descriptor = {
        "kind": KIND_CYLINDER,
        "center": [0.0, 0.0, 0.0],
        "radius": 0.05,
        "half_height": 0.2,
        "rotation": np.eye(3).tolist(),
    }
    corrected = correct_lateral_axes_from_mask(descriptor, (0.5, 0.08), scale_xy=1.0)
    assert corrected == descriptor


def test_correct_center_depth_shifts_away_from_camera_by_two_thirds_depth_size():
    descriptor = {
        "kind": KIND_ELLIPSOID,
        "center": [1.0, 2.0, 3.0],
        "semi_axes": [0.1, 0.1, 0.12],  # identity rotation -> axis 2 (Z) is the depth axis
        "rotation": np.eye(3).tolist(),
    }
    corrected = correct_center_depth(descriptor)
    assert np.isclose(corrected["center"][2], 3.0 + (2 / 3) * 0.12)
    assert corrected["center"][0] == 1.0  # lateral coordinates untouched
    assert corrected["center"][1] == 2.0
    assert corrected["semi_axes"] == descriptor["semi_axes"]
    assert corrected["rotation"] == descriptor["rotation"]


def test_correct_center_depth_flips_a_negative_facing_depth_axis():
    # The depth-axis column (index 2) points toward the camera (-Z) here --
    # the shift must still move the center *away* from the camera (+Z),
    # not blindly follow the raw PCA axis direction.
    rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])
    descriptor = {
        "kind": KIND_ELLIPSOID,
        "center": [0.0, 0.0, 5.0],
        "semi_axes": [0.1, 0.1, 0.09],
        "rotation": rotation.tolist(),
    }
    corrected = correct_center_depth(descriptor)
    assert corrected["center"][2] > descriptor["center"][2]  # moved deeper, not shallower
    assert np.isclose(corrected["center"][2], 5.0 + (2 / 3) * 0.09)


def test_correct_center_depth_works_for_box_too():
    descriptor = {
        "kind": KIND_BOX,
        "center": [0.0, 0.0, 2.0],
        "half_extents": [0.2, 0.15, 0.1],
        "rotation": np.eye(3).tolist(),
    }
    corrected = correct_center_depth(descriptor)
    assert np.isclose(corrected["center"][2], 2.0 + (2 / 3) * 0.1)


def test_correct_center_depth_leaves_cylinder_unchanged():
    descriptor = {
        "kind": KIND_CYLINDER,
        "center": [0.0, 0.0, 2.0],
        "radius": 0.05,
        "half_height": 0.2,
        "rotation": np.eye(3).tolist(),
    }
    corrected = correct_center_depth(descriptor)
    assert corrected == descriptor


def test_correct_cylinder_extent_from_mask_uses_larger_dim_for_length_smaller_for_radius():
    descriptor = {
        "kind": KIND_CYLINDER,
        "center": [0.0, 0.0, 0.0],
        "radius": 0.02,  # stand-ins for a badly-biased point-cloud fit
        "half_height": 5.0,
        "rotation": np.eye(3).tolist(),
    }
    # 0.10m = diameter (radius 0.05), 0.5m = length (half_height 0.25) -- order shouldn't matter.
    corrected = correct_cylinder_extent_from_mask(descriptor, (0.5, 0.10), scale_xy=1.0)
    assert np.isclose(corrected["radius"], 0.05)
    assert np.isclose(corrected["half_height"], 0.25)
    assert corrected["center"] == descriptor["center"]
    assert corrected["rotation"] == descriptor["rotation"]


def test_correct_cylinder_extent_from_mask_divides_by_scale_xy():
    descriptor = {
        "kind": KIND_CYLINDER, "center": [0.0, 0.0, 0.0], "radius": 0.02,
        "half_height": 5.0, "rotation": np.eye(3).tolist(),
    }
    corrected = correct_cylinder_extent_from_mask(descriptor, (0.5, 0.10), scale_xy=2.0)
    assert np.isclose(corrected["radius"], 0.025)  # 0.10 / 2 / 2.0
    assert np.isclose(corrected["half_height"], 0.125)  # 0.5 / 2 / 2.0


def test_correct_cylinder_extent_from_mask_ignores_non_cylinder_shapes():
    descriptor = {
        "kind": KIND_ELLIPSOID, "center": [0.0, 0.0, 0.0],
        "semi_axes": [0.1, 0.1, 0.1], "rotation": np.eye(3).tolist(),
    }
    corrected = correct_cylinder_extent_from_mask(descriptor, (0.5, 0.10), scale_xy=1.0)
    assert corrected == descriptor
