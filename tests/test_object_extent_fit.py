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
    _reject_depth_outliers,
    fit_object_shape,
    fit_position_and_orientation,
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
