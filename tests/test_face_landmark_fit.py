"""Unit tests for `face_landmark_fit.py`. The pure-math helpers
(`_project_points`, `_init_shared_betas`, `_init_translation`) need no
FLAME model and always run; `test_fit_clip_recovers_known_pose` needs the
real FLAME model (`body_models/flame/`, gitignored, see README's Setup
section) and is skipped without it.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from pipeline.algorithms.face.face_landmark_fit import (
    EXPRESSION_BOUND, FLAME_MODEL_DIR,
    FLAME_NUM_BETAS, FLAME_NUM_EXPRESSION, JAW_AXIS0_BOUND_RAD, JAW_AXIS1_BOUND_RAD, JAW_AXIS1_DECA_DEVIATION_THRESHOLD,
    JAW_AXIS2_BOUND_RAD, MAX_BRIDGE_FRAMES, NUM_FLAME_LANDMARKS,
    FitInputs, KEY_EXPRESSION, KEY_GLOBAL_ORIENT, KEY_JAW_POSE, KEY_TRANSL, KEY_VALID,
    _bridge_keep_mask, _bridge_short_gaps, _build_flame_model, _demote_short_valid_runs, _detect_outlier_frames,
    _init_shared_betas, _init_translation, _lead_from_neutral, _project_points,
    _snap_jaw_to_deca_on_axis_deviation, calibrate_rotation_offset, fit_clip,
)

FLAME_ASSETS_PRESENT = (FLAME_MODEL_DIR / "flame" / "FLAME_NEUTRAL.npz").exists()
TEST_FPS = 30.0  # fit_clip's own fps is always sourced from real clip data (RunRecord.scene.fps); no default to fall back to here either


def _random_rotations(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return Rotation.from_euler("xyz", rng.uniform(-180, 180, size=(n, 3)), degrees=True).as_matrix()


def test_calibrate_rotation_offset_recovers_known_offset():
    n = 30
    r_offset_true = Rotation.from_euler("xyz", [15.0, -25.0, 40.0], degrees=True).as_matrix()
    r_head = _random_rotations(n, seed=1)
    r_deca = r_offset_true[None] @ r_head  # exact, noiseless agreement
    deca_global_orient = Rotation.from_matrix(r_deca).as_rotvec().astype(np.float32)

    r_offset_est = calibrate_rotation_offset(deca_global_orient, r_head.astype(np.float32), np.ones(n, dtype=bool))

    angle_error = Rotation.from_matrix(r_offset_est @ r_offset_true.T).magnitude()
    assert np.degrees(angle_error) < 1.0


def test_calibrate_rotation_offset_robust_to_a_few_outlier_frames():
    n = 20
    r_offset_true = Rotation.from_euler("xyz", [15.0, -25.0, 40.0], degrees=True).as_matrix()
    r_head = _random_rotations(n, seed=2)
    r_deca = r_offset_true[None] @ r_head
    deca_global_orient = Rotation.from_matrix(r_deca).as_rotvec().astype(np.float32)

    # Corrupt 3 of the 20 frames with an unrelated DECA estimate, as if DECA
    # badly misread those frames, exactly the noisy-but-mostly-good case this
    # function is meant to tolerate via averaging rather than a single frame.
    rng = np.random.default_rng(3)
    outlier_idx = rng.choice(n, size=3, replace=False)
    deca_global_orient[outlier_idx] = Rotation.from_euler(
        "xyz", rng.uniform(-180, 180, size=(3, 3)), degrees=True
    ).as_rotvec().astype(np.float32)

    r_offset_est = calibrate_rotation_offset(deca_global_orient, r_head.astype(np.float32), np.ones(n, dtype=bool))

    angle_error = Rotation.from_matrix(r_offset_est @ r_offset_true.T).magnitude()
    assert np.degrees(angle_error) < 15.0  # degraded by the outliers, but not blown up


def test_calibrate_rotation_offset_no_valid_frames_returns_identity():
    n = 4
    result = calibrate_rotation_offset(
        np.zeros((n, 3), dtype=np.float32), np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)), np.zeros(n, dtype=bool),
    )
    assert np.allclose(result, np.eye(3))


def test_project_points_identity_intrinsics_at_unit_depth():
    K = torch.eye(3, dtype=torch.float32)
    points = torch.tensor([[3.0, 4.0, 1.0], [0.0, 0.0, 2.0]])
    pixels = _project_points(points, K)
    assert torch.allclose(pixels, torch.tensor([[3.0, 4.0], [0.0, 0.0]]))


def test_project_points_scales_with_focal_length():
    K = torch.tensor([[100.0, 0, 0], [0, 100.0, 0], [0, 0, 1.0]])
    # A point straight ahead at depth 10, offset 1 unit laterally -> pixel = f * (x/z)
    points = torch.tensor([[1.0, 0.0, 10.0]])
    pixels = _project_points(points, K)
    assert torch.allclose(pixels, torch.tensor([[10.0, 0.0]]))


def test_init_shared_betas_prefers_mica():
    mica_shape = np.zeros((3, FLAME_NUM_BETAS), dtype=np.float32)
    mica_shape[1] = 5.0
    mica_valid = np.array([False, True, False])
    deca_shape = np.ones((3, 100), dtype=np.float32)
    deca_valid = np.array([True, True, True])
    betas = _init_shared_betas(mica_shape, mica_valid, deca_shape, deca_valid)
    assert np.allclose(betas, 5.0)


def test_init_shared_betas_falls_back_to_deca_when_mica_all_invalid():
    mica_shape = np.zeros((2, FLAME_NUM_BETAS), dtype=np.float32)
    mica_valid = np.array([False, False])
    deca_shape = np.full((2, 100), 2.0, dtype=np.float32)
    deca_valid = np.array([True, True])
    betas = _init_shared_betas(mica_shape, mica_valid, deca_shape, deca_valid)
    assert np.allclose(betas[:100], 2.0)
    assert np.allclose(betas[100:], 0.0)  # zero-padded, not garbage


def test_init_translation_invalid_frames_borrow_nearest_valid():
    landmarks = np.zeros((3, NUM_FLAME_LANDMARKS, 2), dtype=np.float32)
    landmarks[:, :, :] = [100.0, 100.0]
    # Give frame 1 (the only valid one) a distinct, resolvable eye separation.
    landmarks[1, 19 - 17] = [90.0, 100.0]
    landmarks[1, 28 - 17] = [110.0, 100.0]
    valid = np.array([False, True, False])
    template = np.zeros((NUM_FLAME_LANDMARKS, 3), dtype=np.float32)
    template[19 - 17] = [-0.03, 0, 0]
    template[28 - 17] = [0.03, 0, 0]
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]], dtype=np.float32)

    transl = _init_translation(landmarks, valid, template, K)
    assert np.allclose(transl[0], transl[1])  # frame 0 borrows frame 1's estimate
    assert np.allclose(transl[2], transl[1])  # frame 2 borrows frame 1's estimate
    assert transl[1, 2] > 0  # positive depth


def test_bridge_keep_mask_marks_short_interior_run_as_not_kept():
    # frames 3-4 invalid (length 2), bounded by valid frames, a "blip" bridgeable at max_bridge_frames=8.
    valid = np.array([True, True, True, False, False, True, True, True])
    keep = _bridge_keep_mask(valid, max_bridge_frames=8)
    assert not keep[3] and not keep[4]
    assert keep[[0, 1, 2, 5, 6, 7]].all()


def test_bridge_keep_mask_leaves_long_interior_run_kept():
    # frames 2-9 invalid (length 8), just over max_bridge_frames=7, real occlusion, left untouched.
    valid = np.array([True, True] + [False] * 8 + [True, True])
    keep = _bridge_keep_mask(valid, max_bridge_frames=7)
    assert keep[2:10].all()  # long run, kept as-is (still anchor-driven), not bridged


def test_bridge_keep_mask_leaves_leading_and_trailing_runs_kept():
    # No valid frame on one side to interpolate toward, fill_invalid's own freeze behavior applies, not bridging.
    valid = np.array([False, False, True, True, True, False, False])
    keep = _bridge_keep_mask(valid, max_bridge_frames=8)
    assert keep[0] and keep[1] and keep[5] and keep[6]


def test_bridge_short_gaps_interpolates_short_run_between_valid_neighbors():
    """Mirrors the real bug this was built for: a short MediaPipe
    dropout sandwiched between two well-fit valid frames, where the
    anchor-driven fit (here simulated directly, not via a real optimization)
    lands somewhere bad, bridging should replace it with an interpolation
    between the flanking valid frames' own values, ignoring what the
    anchor-driven fit produced there."""
    n = 5
    rotvec = np.zeros((n, 3), dtype=np.float32)
    rotvec[0] = [0.0, 0.0, 0.0]
    rotvec[1] = [0.9, 0.0, 0.0]  # bad anchor-driven value on the invalid gap frame
    rotvec[2] = [0.0, 0.0, 0.0]
    rotvec[3] = [0.5, 0.0, 0.0]  # bad anchor-driven value on the invalid gap frame
    rotvec[4] = [1.0, 0.0, 0.0]
    valid = np.array([True, False, False, False, True])  # frames 1-3 invalid, a 3-frame blip
    linear = {"expr": np.array([[0.0], [9.0], [9.0], [9.0], [2.0]], dtype=np.float32)}

    bridged_rotvec, bridged_linear = _bridge_short_gaps(rotvec, linear, valid, max_bridge_frames=8)

    # Interpolated strictly between the flanking valid values (0.0 and 1.0 on
    # axis 0), monotonically increasing, not equal to the bad anchor-driven
    # values that were there before bridging.
    assert 0.0 < bridged_rotvec[1, 0] < bridged_rotvec[2, 0] < bridged_rotvec[3, 0] < 1.0
    assert 0.0 < bridged_linear["expr"][1, 0] < 2.0
    assert bridged_linear["expr"][2, 0] < 9.0  # not the bad anchor-driven value
    # Valid frames themselves are untouched.
    assert np.allclose(bridged_rotvec[0], rotvec[0])
    assert np.allclose(bridged_rotvec[4], rotvec[4])


def test_bridge_short_gaps_leaves_long_run_untouched():
    n = 12
    rotvec = np.zeros((n, 3), dtype=np.float32)
    rotvec[2:10, 0] = np.linspace(0.3, 0.7, 8)  # the "anchor-driven" values during a real 8-frame occlusion
    valid = np.array([True, True] + [False] * 8 + [True, True])
    linear = {"transl": np.zeros((n, 1), dtype=np.float32)}

    bridged_rotvec, _ = _bridge_short_gaps(rotvec, linear, valid, max_bridge_frames=7)

    # Run length 8 > max_bridge_frames=7, real occlusion, left exactly as the anchor-driven fit produced it.
    assert np.allclose(bridged_rotvec[2:10, 0], rotvec[2:10, 0])


def test_detect_outlier_frames_flags_a_single_frame_spike():
    """Mirrors the real bug this was built for: all
    frames are landmark-valid, but one frame's own fitted value swings far
    from both neighbors while they agree closely with each other, a real
    per-frame Adam optimization outlier, not a landmark-dropout artifact
    (which _bridge_keep_mask's validity-driven pass already handles).

    Values are deliberately chosen so a spike of size S produces deviation S
    at the spike itself but only S/2 at each immediate neighbor (the spike
    necessarily pulls their own midpoint-comparison partway toward it too),
    threshold sits between S/2 and S so only the spike itself is flagged."""
    values = np.array([[0.10], [0.10], [0.25], [0.10], [0.10]], dtype=np.float32)  # idx2 is a S=0.15 spike
    valid = np.ones(5, dtype=bool)
    outlier = _detect_outlier_frames(values, valid, threshold=0.1)
    assert outlier.tolist() == [False, False, True, False, False]


def test_detect_outlier_frames_does_not_flag_normal_variation():
    # A smooth, gradually-rising trend, every frame sits exactly on its neighbors' own midpoint.
    values = np.array([[0.10], [0.13], [0.16], [0.19], [0.22]], dtype=np.float32)
    valid = np.ones(5, dtype=bool)
    outlier = _detect_outlier_frames(values, valid, threshold=0.1)
    assert not outlier.any()


def test_detect_outlier_frames_ignores_invalid_frames():
    # Same spike as the first test, but the spike frame is landmark-invalid,
    # not this function's job (bridging on validity already covers it), so it must not be flagged here.
    values = np.array([[0.10], [0.10], [0.25], [0.10], [0.10]], dtype=np.float32)
    valid = np.array([True, True, False, True, True])
    outlier = _detect_outlier_frames(values, valid, threshold=0.1)
    assert not outlier.any()


def test_detect_outlier_frames_window_1_misses_a_paired_spike():
    # Regression guard documenting the real gap JAW_OUTLIER_WINDOW fixes:
    # two ADJACENT frames spiking together pollute each other's own
    # 2-point-midpoint reference (idx2's neighbor idx3 is itself a spike),
    # so the default window=1 check can't see either.
    values = np.array([[0.10], [0.10], [0.25], [0.25], [0.10], [0.10]], dtype=np.float32)
    valid = np.ones(6, dtype=bool)
    outlier = _detect_outlier_frames(values, valid, threshold=0.1)
    assert not outlier.any()


def test_detect_outlier_frames_wider_window_catches_a_paired_spike():
    # Same paired spike as above, a window=2 median reference tolerates
    # the one corrupted neighbor on each side and correctly flags both
    # spike frames.
    values = np.array([[0.10], [0.10], [0.25], [0.25], [0.10], [0.10]], dtype=np.float32)
    valid = np.ones(6, dtype=bool)
    outlier = _detect_outlier_frames(values, valid, threshold=0.1, window=2)
    assert outlier.tolist() == [False, False, True, True, False, False]


def test_detect_outlier_frames_wider_window_still_ignores_real_trend():
    # A smooth ramp must not get flagged just because the window widened,
    # the median reference of a monotonic trend still sits close to each
    # frame's own value.
    values = np.array([[0.10], [0.13], [0.16], [0.19], [0.22], [0.25], [0.28]], dtype=np.float32)
    valid = np.ones(7, dtype=bool)
    outlier = _detect_outlier_frames(values, valid, threshold=0.1, window=3)
    assert not outlier.any()


def test_detect_outlier_frames_rotation_flags_a_single_frame_spike():
    # Same S/(spike)-vs-S/2/(neighbor) shape as the linear test above (confirmed
    # numerically: single-axis rotation composition halves the same way here).
    rotvec = np.zeros((5, 3), dtype=np.float32)
    rotvec[:, 0] = [0.0, 0.0, 0.5, 0.0, 0.0]  # idx2 is a real rotation spike, flanked by a flat baseline
    valid = np.ones(5, dtype=bool)
    outlier = _detect_outlier_frames(rotvec, valid, threshold=0.35, is_rotation=True)
    assert outlier[2]
    assert not outlier[1] and not outlier[3]


def test_snap_jaw_to_deca_on_axis_deviation_replaces_flagged_frames():
    """Mirrors the real bug this was built for: a sustained multi-frame drift
    on axis1 (yaw) that agrees with its own immediate neighbors throughout
    (so `_detect_outlier_frames` can't see it) while sitting far from DECA's
    own stable estimate for the whole stretch, see
    JAW_AXIS1_DECA_DEVIATION_THRESHOLD's own comment."""
    jaw_pose = np.array([[0.0, 0.01, 0.0], [0.0, 0.10, 0.0], [0.0, 0.11, 0.0], [0.0, 0.02, 0.0]], dtype=np.float32)
    deca_jaw = np.array([[0.0, 0.01, 0.0], [0.0, 0.02, 0.0], [0.0, 0.02, 0.0], [0.0, 0.02, 0.0]], dtype=np.float32)
    valid = np.ones(4, dtype=bool)

    snapped, flagged = _snap_jaw_to_deca_on_axis_deviation(
        jaw_pose, deca_jaw, valid, axis=1, threshold=JAW_AXIS1_DECA_DEVIATION_THRESHOLD,
    )

    assert flagged.tolist() == [False, True, True, False]
    assert np.array_equal(snapped[1], deca_jaw[1])
    assert np.array_equal(snapped[2], deca_jaw[2])
    assert np.array_equal(snapped[0], jaw_pose[0])
    assert np.array_equal(snapped[3], jaw_pose[3])


def test_snap_jaw_to_deca_on_axis_deviation_leaves_small_deviation_untouched():
    jaw_pose = np.array([[0.0, 0.03, 0.0], [0.0, 0.04, 0.0]], dtype=np.float32)
    deca_jaw = np.array([[0.0, 0.01, 0.0], [0.0, 0.02, 0.0]], dtype=np.float32)  # both deviations under threshold
    valid = np.ones(2, dtype=bool)

    snapped, flagged = _snap_jaw_to_deca_on_axis_deviation(
        jaw_pose, deca_jaw, valid, axis=1, threshold=JAW_AXIS1_DECA_DEVIATION_THRESHOLD,
    )

    assert not flagged.any()
    assert np.array_equal(snapped, jaw_pose)


def test_snap_jaw_to_deca_on_axis_deviation_ignores_invalid_frames():
    # Same large deviation as the first test, but the frame is landmark-invalid,
    # not this function's job (the fit has zero gradient there anyway), so it must not be flagged.
    jaw_pose = np.array([[0.0, 0.10, 0.0]], dtype=np.float32)
    deca_jaw = np.array([[0.0, 0.01, 0.0]], dtype=np.float32)
    valid = np.array([False])

    snapped, flagged = _snap_jaw_to_deca_on_axis_deviation(
        jaw_pose, deca_jaw, valid, axis=1, threshold=JAW_AXIS1_DECA_DEVIATION_THRESHOLD,
    )

    assert not flagged.any()
    assert np.array_equal(snapped, jaw_pose)


def test_demote_short_valid_runs_demotes_a_brief_interior_island():
    """Mirrors the real bug this was built for: a real 111-frame occlusion
    had exactly two single-frame "detections" in the middle of it, ~110
    frames apart,
    each individually too isolated to cross-check against anything, and
    each one's own standalone fit turned out to be unreliable. A brief
    interior run (bounded by invalid frames on both sides) is demoted."""
    valid = np.array([True, True, True, False, False, True, False, False, True, True, True])
    demoted = _demote_short_valid_runs(valid, min_valid_run_frames=6)
    assert not demoted[5]  # the isolated single-frame island is gone
    assert demoted[[0, 1, 2, 8, 9, 10]].all()  # the two real (>=6-frame-eligible-length... here just longer) runs stay


def test_demote_short_valid_runs_leaves_leading_and_trailing_runs_alone():
    # Short runs, but at the very edges, no invalid neighbor on both sides to be suspicious relative to.
    valid = np.array([True, True, False, False, False, False, False, False, True, True])
    demoted = _demote_short_valid_runs(valid, min_valid_run_frames=6)
    assert np.array_equal(demoted, valid)


def test_demote_short_valid_runs_leaves_whole_clip_valid_run_alone():
    # A run spanning the entire array (e.g. a short synthetic test clip) is neither leading, trailing, nor interior,
    # demoting it would silently zero out an otherwise-fine short clip's only data.
    valid = np.ones(5, dtype=bool)
    demoted = _demote_short_valid_runs(valid, min_valid_run_frames=6)
    assert demoted.all()


def test_lead_from_neutral_holds_at_zero_then_glides_into_recovery():
    n = 10
    values = np.zeros((n, 3), dtype=np.float32)
    values[8] = [0.4, 0.0, 0.0]  # the first real detection's own value
    values[9] = [0.5, 0.0, 0.0]
    valid = np.array([False] * 8 + [True, True])

    out_values, out_valid = _lead_from_neutral(values, valid, max_bridge_frames=3)

    assert out_valid[:9].all()  # the whole leading run (0-7) plus the first real frame (8) now trustworthy
    # Held flat at neutral for the bulk (frames 0-4, past the final 3-frame glide window [5,6,7]).
    assert np.allclose(out_values[:5], 0.0)
    # Frames 5-7 (the final max_bridge_frames=3 before recovery) glide toward the real value,
    # strictly increasing and never overshooting it.
    assert 0.0 < out_values[5, 0] < out_values[6, 0] < out_values[7, 0] < values[8, 0]
    # The real detection itself (frame 8) is untouched.
    assert out_values[8, 0] == values[8, 0]


def test_lead_from_neutral_glides_across_a_short_run_entirely():
    # A leading run shorter than max_bridge_frames has no separate "hold" phase, the whole thing glides.
    n = 5
    values = np.zeros((n, 3), dtype=np.float32)
    values[2] = [0.6, 0.0, 0.0]
    valid = np.array([False, False, True, True, True])

    out_values, out_valid = _lead_from_neutral(values, valid, max_bridge_frames=8)

    assert out_valid[:3].all()
    assert 0.0 < out_values[0, 0] < out_values[1, 0] < values[2, 0]


def test_lead_from_neutral_is_a_no_op_without_a_leading_gap():
    n = 5
    values = np.array([[0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0], [0.0, 0, 0], [0.0, 0, 0]], dtype=np.float32)
    valid = np.array([True, True, True, False, False])

    out_values, out_valid = _lead_from_neutral(values, valid, max_bridge_frames=3)

    assert np.array_equal(out_values, values)
    assert np.array_equal(out_valid, valid)


def _synthetic_inputs(n: int, K: np.ndarray, true_transl_z: float) -> tuple[FitInputs, np.ndarray]:
    """Build a FitInputs with landmark targets rendered from a *known* FLAME
    configuration (small nonzero jaw/expression/rotation), so `fit_clip` has
    a ground truth to recover. Returns `(inputs, true_global_orient)`."""
    device = torch.device("cpu")
    model = _build_flame_model(device, batch_size=n)
    rng = np.random.default_rng(0)

    true_betas = torch.zeros(1, FLAME_NUM_BETAS)
    true_expression = torch.zeros(n, FLAME_NUM_EXPRESSION)
    true_expression[:, 0] = torch.linspace(0.0, 1.5, n)
    true_jaw = torch.zeros(n, 3)
    true_jaw[:, 0] = torch.linspace(0.0, 0.2, n)
    true_global_orient = torch.zeros(n, 3)
    true_global_orient[:, 1] = torch.linspace(-0.1, 0.1, n)
    true_transl = torch.zeros(n, 3)
    true_transl[:, 2] = true_transl_z

    with torch.no_grad():
        out = model(
            betas=true_betas.expand(n, -1), expression=true_expression, global_orient=true_global_orient,
            neck_pose=torch.zeros(n, 3), jaw_pose=true_jaw,
            leye_pose=torch.zeros(n, 3), reye_pose=torch.zeros(n, 3), transl=true_transl,
        )
        landmarks_cam = out.joints[:, -NUM_FLAME_LANDMARKS:, :]
        landmarks_px = _project_points(landmarks_cam, torch.tensor(K, dtype=torch.float32)).numpy()

    inputs = FitInputs(
        landmarks_51=landmarks_px.astype(np.float32),
        landmarks_valid=np.ones(n, dtype=bool),
        deca_exp=np.zeros((n, FLAME_NUM_EXPRESSION), dtype=np.float32),  # deliberately wrong init
        deca_pose=np.zeros((n, 6), dtype=np.float32),
        deca_shape=np.zeros((n, 100), dtype=np.float32),
        deca_valid=np.ones(n, dtype=bool),
        mica_shape=np.zeros((n, FLAME_NUM_BETAS), dtype=np.float32),
        mica_valid=np.ones(n, dtype=bool),
        intrinsics_k=K,
    )
    return inputs, true_global_orient.numpy()


@pytest.mark.skipif(not FLAME_ASSETS_PRESENT, reason="needs the FLAME model (see README's Setup section)")
def test_fit_clip_recovers_known_pose():
    K = np.array([[1000.0, 0, 512], [0, 1000.0, 384], [0, 0, 1]], dtype=np.float32)
    # A ~1.3s jaw ramp so this test measures the optimizer's own recovery
    # accuracy, not the one-euro post-filter's cold-start lag on a short clip.
    n = 40
    inputs, true_global_orient = _synthetic_inputs(n, K, true_transl_z=1.0)

    result = fit_clip(
        inputs, fps=TEST_FPS, device=torch.device("cpu"), stage1_iters=400, stage1_lr=0.05, stage2_iters=400, stage2_lr=0.05,
        temporal_weight=1.0,
        # This test recovers from a deliberately wrong DECA init via
        # reprojection alone, so the DECA-anchor ridge is disabled here,
        # left on, it would pull the fit back toward that wrong init.
        jaw_deca_anchor_weight=0.0, expr_deca_anchor_weight=0.0,
    )

    assert result[KEY_VALID].all()
    # Starting from a deliberately wrong (all-zero) DECA init, the optimizer
    # should still land close to the true rotation/jaw/translation that
    # generated the target landmarks.
    assert np.allclose(result[KEY_GLOBAL_ORIENT], true_global_orient, atol=0.05)
    # Stage 2 frees jaw_pose and expression together, so some of jaw's true
    # displacement is genuinely explainable by expression instead, a wider
    # tolerance here reflects that real ambiguity, not slack in the test.
    assert np.abs(result[KEY_JAW_POSE][:, 0] - np.linspace(0.0, 0.2, n)).max() < 0.12
    # Depth (Z) has its own inherent monocular ambiguity (a bigger face
    # further away reprojects the same as a smaller face closer up), a
    # wider tolerance than the other params for the same reason.
    assert np.abs(result[KEY_TRANSL][:, 2] - 1.0).max() < 0.1


@pytest.mark.skipif(not FLAME_ASSETS_PRESENT, reason="needs the FLAME model (see README's Setup section)")
def test_fit_clip_stage2_respects_hard_bounds_under_pathological_target():
    """Stage 2's tanh reparameterization is a hard bound by construction
    (JAW_AXIS0/1/2_BOUND_RAD, EXPRESSION_BOUND), not a soft penalty,
    unlike every anchor tried before this redesign (up to a 200x-boosted
    weight), no amount of reprojection pull should
    ever push jaw_pose or expression past their bounds. Verified
    adversarially: the landmark target is shifted far outside anything a
    real face could produce, so reprojection loss pulls jaw_raw/expr_raw as
    hard as possible toward the bound for the whole optimization."""
    K = np.array([[1000.0, 0, 512], [0, 1000.0, 384], [0, 0, 1]], dtype=np.float32)
    n = 3
    inputs, _ = _synthetic_inputs(n, K, true_transl_z=1.0)
    pathological_landmarks = inputs.landmarks_51.copy()
    pathological_landmarks[:, :, 1] += 5000.0  # unreachable by any valid FLAME configuration
    inputs = dataclasses.replace(inputs, landmarks_51=pathological_landmarks)

    result = fit_clip(
        inputs, fps=TEST_FPS, device=torch.device("cpu"), stage1_iters=50, stage1_lr=0.03, stage2_iters=500, stage2_lr=0.05,
    )

    jaw_bounds = np.array([JAW_AXIS0_BOUND_RAD, JAW_AXIS1_BOUND_RAD, JAW_AXIS2_BOUND_RAD])
    assert (np.abs(result[KEY_JAW_POSE]) <= jaw_bounds[None, :] + 1e-4).all()
    assert (np.abs(result[KEY_EXPRESSION]) <= EXPRESSION_BOUND + 1e-4).all()


@pytest.mark.skipif(not FLAME_ASSETS_PRESENT, reason="needs the FLAME model (see README's Setup section)")
def test_fit_clip_deca_anchor_resists_pull_toward_pathological_target():
    """DEFAULT_JAW_DECA_ANCHOR_WEIGHT/DEFAULT_EXPR_DECA_ANCHOR_WEIGHT's own
    real-data justification (731-757/172-218, see that constant's comment)
    was catching the fit swinging far from a *stable, reasonable* DECA
    estimate with no support from the actual landmarks, this reproduces
    that shape synthetically: an unreachable landmark target pulls jaw/expr
    hard toward their bounds (same fixture as the hard-bounds test above),
    and a nonzero DECA anchor should measurably resist that pull, landing
    closer to the (zero) DECA init than an unanchored fit does."""
    K = np.array([[1000.0, 0, 512], [0, 1000.0, 384], [0, 0, 1]], dtype=np.float32)
    n = 3
    inputs, _ = _synthetic_inputs(n, K, true_transl_z=1.0)
    pathological_landmarks = inputs.landmarks_51.copy()
    pathological_landmarks[:, :, 1] += 5000.0
    inputs = dataclasses.replace(inputs, landmarks_51=pathological_landmarks)

    kwargs = dict(fps=TEST_FPS, device=torch.device("cpu"), stage1_iters=50, stage1_lr=0.03, stage2_iters=500, stage2_lr=0.05)
    with_anchor = fit_clip(inputs, jaw_deca_anchor_weight=50000.0, expr_deca_anchor_weight=50000.0, **kwargs)
    without_anchor = fit_clip(inputs, jaw_deca_anchor_weight=0.0, expr_deca_anchor_weight=0.0, **kwargs)

    # DECA's own jaw/expression init is all-zero in this fixture, a strong
    # anchor should keep the fit much closer to it than the unanchored fit,
    # which rushes all the way to the hard bound instead (confirmed by the
    # sibling test above).
    assert np.abs(with_anchor[KEY_JAW_POSE]).max() < np.abs(without_anchor[KEY_JAW_POSE]).max()
    assert np.abs(with_anchor[KEY_EXPRESSION]).max() < np.abs(without_anchor[KEY_EXPRESSION]).max()


def _synthetic_inputs_with_occlusion_and_head_prior(K: np.ndarray) -> tuple[FitInputs, np.ndarray, np.ndarray]:
    """7 frames, a large global_orient spike on frames 3-4 only (a fast turn
    during an occlusion), with landmarks_valid False there, so reprojection
    loss can't see it, and temporal smoothing alone would pull those frames
    back toward the flat (0) neighboring baseline instead of the true spike.
    Recoverable only via a known FLAME<->SMPL-X rest-orientation offset
    applied to a synthetic body track (`head_rotmat`) that stays valid straight
    through the occlusion, exactly what the body-based orientation prior
    is meant to exploit. `deca_pose` is left all-zero (matches ground truth everywhere
    *except* the spike, so calibration, which only sees the non-occluded
    frames, succeeds cleanly, isolating the "does the prior reach the
    occluded frames" claim from calibration noise, already covered separately
    by `test_calibrate_rotation_offset_*`). Returns (inputs, true_global_orient,
    occluded_frame_mask)."""
    device = torch.device("cpu")
    n = 7
    model = _build_flame_model(device, batch_size=n)

    true_global_orient = torch.zeros(n, 3)
    true_global_orient[:, 1] = torch.tensor([0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0])
    true_transl = torch.zeros(n, 3)
    true_transl[:, 2] = 1.0

    with torch.no_grad():
        out = model(
            betas=torch.zeros(1, FLAME_NUM_BETAS).expand(n, -1), expression=torch.zeros(n, FLAME_NUM_EXPRESSION),
            global_orient=true_global_orient, neck_pose=torch.zeros(n, 3), jaw_pose=torch.zeros(n, 3),
            leye_pose=torch.zeros(n, 3), reye_pose=torch.zeros(n, 3), transl=true_transl,
        )
        landmarks_px = _project_points(out.joints[:, -NUM_FLAME_LANDMARKS:, :], torch.tensor(K, dtype=torch.float32)).numpy()

    occluded = np.array([False, False, False, True, True, False, False])

    r_offset_true = Rotation.from_euler("xyz", [12.0, -7.0, 20.0], degrees=True).as_matrix()
    r_global = Rotation.from_rotvec(true_global_orient.numpy()).as_matrix()
    head_rotmat = (np.transpose(r_offset_true)[None] @ r_global).astype(np.float32)  # R_head = R_offset^T @ R_flame

    inputs = FitInputs(
        landmarks_51=landmarks_px.astype(np.float32),
        landmarks_valid=~occluded,
        deca_exp=np.zeros((n, FLAME_NUM_EXPRESSION), dtype=np.float32),
        deca_pose=np.zeros((n, 6), dtype=np.float32),
        deca_shape=np.zeros((n, 100), dtype=np.float32),
        deca_valid=np.ones(n, dtype=bool),
        mica_shape=np.zeros((n, FLAME_NUM_BETAS), dtype=np.float32),
        mica_valid=np.ones(n, dtype=bool),
        intrinsics_k=K,
        head_rotmat=head_rotmat,
        head_confidence=np.ones(n, dtype=bool),
    )
    return inputs, true_global_orient.numpy(), occluded


@pytest.mark.skipif(not FLAME_ASSETS_PRESENT, reason="needs the FLAME model (see README's Setup section)")
def test_fit_clip_head_prior_recovers_occluded_rotation_spike():
    K = np.array([[1000.0, 0, 512], [0, 1000.0, 384], [0, 0, 1]], dtype=np.float32)
    inputs, true_global_orient, occluded = _synthetic_inputs_with_occlusion_and_head_prior(K)

    # global_orient_anchor_weight=0 isolates this test's own claim (does the
    # GVHMR head-prior recover an occluded rotation spike) from the separate
    # DECA-global_orient anchor added afterward, this synthetic clip's own
    # `deca_pose` is deliberately all-zero (see the fixture's docstring), so
    # without isolating it, that anchor competes with the very recovery this
    # test is checking for, rather than testing it.
    kwargs = dict(fps=TEST_FPS, device=torch.device("cpu"), stage1_iters=800, stage1_lr=0.03, stage2_iters=10, temporal_weight=1.0, global_orient_anchor_weight=0.0)
    with_prior = fit_clip(inputs, **kwargs)
    without_prior_inputs = dataclasses.replace(inputs, head_rotmat=None, head_confidence=None)
    without_prior = fit_clip(without_prior_inputs, **kwargs)

    with_error = np.abs(with_prior[KEY_GLOBAL_ORIENT][occluded, 1] - true_global_orient[occluded, 1])
    without_error = np.abs(without_prior[KEY_GLOBAL_ORIENT][occluded, 1] - true_global_orient[occluded, 1])

    # With the prior, the occluded frames land close to the true 0.5 rad spike.
    assert with_error.max() < 0.15
    # Without it, the only pull on those frames is temporal smoothing toward
    # the flat neighboring baseline, meaningfully worse, not just noisier.
    assert without_error.min() > with_error.max()


@pytest.mark.skipif(not FLAME_ASSETS_PRESENT, reason="needs the FLAME model (see README's Setup section)")
def test_fit_clip_gates_off_near_singular_prior():
    """A near-180 deg offset composed with near-identity head rotations puts
    the composed prior right at the axis-angle singularity, structurally
    identical to a real clip (GVHMR's incam convention isn't camera-relative,
    and some real clips' camera/person geometry lands global_orient near 180
    deg) that measurably destabilized the fit before this gate existed. The
    gate should suppress the prior entirely on this clip, converging to
    exactly the same result as if head_rotmat/head_confidence were never
    passed at all, not just "closer", since nothing about the optimization
    should differ once the gate fires."""
    K = np.array([[1000.0, 0, 512], [0, 1000.0, 384], [0, 0, 1]], dtype=np.float32)
    n = 5
    device = torch.device("cpu")
    model = _build_flame_model(device, batch_size=n)

    true_global_orient = torch.zeros(n, 3)
    true_global_orient[:, 1] = torch.linspace(-0.05, 0.05, n)
    true_transl = torch.zeros(n, 3)
    true_transl[:, 2] = 1.0
    with torch.no_grad():
        out = model(
            betas=torch.zeros(1, FLAME_NUM_BETAS).expand(n, -1), expression=torch.zeros(n, FLAME_NUM_EXPRESSION),
            global_orient=true_global_orient, neck_pose=torch.zeros(n, 3), jaw_pose=torch.zeros(n, 3),
            leye_pose=torch.zeros(n, 3), reye_pose=torch.zeros(n, 3), transl=true_transl,
        )
        landmarks_px = _project_points(out.joints[:, -NUM_FLAME_LANDMARKS:, :], torch.tensor(K, dtype=torch.float32)).numpy()

    r_offset_true = Rotation.from_euler("xyz", [170.0, 0.0, 0.0], degrees=True).as_matrix()
    rng = np.random.default_rng(4)
    head_rotmat = Rotation.from_rotvec(rng.normal(0, 0.05, size=(n, 3))).as_matrix().astype(np.float32)
    deca_global_orient = Rotation.from_matrix(r_offset_true[None] @ head_rotmat).as_rotvec().astype(np.float32)

    base_kwargs = dict(
        landmarks_51=landmarks_px.astype(np.float32), landmarks_valid=np.ones(n, dtype=bool),
        deca_exp=np.zeros((n, FLAME_NUM_EXPRESSION), dtype=np.float32),
        deca_pose=np.concatenate([deca_global_orient, np.zeros((n, 3), dtype=np.float32)], axis=1),
        deca_shape=np.zeros((n, 100), dtype=np.float32), deca_valid=np.ones(n, dtype=bool),
        mica_shape=np.zeros((n, FLAME_NUM_BETAS), dtype=np.float32), mica_valid=np.ones(n, dtype=bool),
        intrinsics_k=K,
    )

    with_gated_prior = fit_clip(
        FitInputs(**base_kwargs, head_rotmat=head_rotmat, head_confidence=np.ones(n, dtype=bool)),
        fps=TEST_FPS, device=device, stage1_iters=200, stage1_lr=0.03, stage2_iters=10,
    )
    without_prior = fit_clip(
        FitInputs(**base_kwargs, head_rotmat=None, head_confidence=None),
        fps=TEST_FPS, device=device, stage1_iters=200, stage1_lr=0.03, stage2_iters=10,
    )

    assert np.allclose(with_gated_prior[KEY_GLOBAL_ORIENT], without_prior[KEY_GLOBAL_ORIENT], atol=1e-5)


@pytest.mark.skipif(not FLAME_ASSETS_PRESENT, reason="needs the FLAME model (see README's Setup section)")
def test_fit_clip_global_orient_anchor_recovers_occluded_orientation_spike():
    """Same structure as the jaw-anchor test above, for global_orient:
    global_orient can be poorly constrained on its own, with jaw_pose partly
    compensating for an under-constrained head pitch. `head_rotmat`/
    `head_confidence` stay None here, this isolates the DECA-global_orient
    anchor from the separate GVHMR-based body-orientation prior anchor.

    The occluded run here is deliberately longer than MAX_BRIDGE_FRAMES: a
    short occluded run gets interpolated by `_bridge_short_gaps` instead of
    relying on the DECA anchor at all (DECA is excluded from
    `_bridge_keep_mask`'s `trustworthy_anchor`, since it has no confidence
    signal distinguishing a good per-frame estimate from a bad one). This
    test isolates the DECA anchor's own still-needed job: a long occlusion,
    past bridging's reach, where DECA is the only per-frame signal
    available."""
    K = np.array([[1000.0, 0, 512], [0, 1000.0, 384], [0, 0, 1]], dtype=np.float32)
    occlusion_len = MAX_BRIDGE_FRAMES + 1  # past bridging's reach, see docstring above
    n = 5 + occlusion_len
    device = torch.device("cpu")
    model = _build_flame_model(device, batch_size=n)

    true_global_orient = torch.zeros(n, 3)
    true_global_orient[:, 1] = torch.tensor([0.0, 0.0, 0.0] + [0.4] * occlusion_len + [0.0, 0.0])
    true_transl = torch.zeros(n, 3)
    true_transl[:, 2] = 1.0
    with torch.no_grad():
        out = model(
            betas=torch.zeros(1, FLAME_NUM_BETAS).expand(n, -1), expression=torch.zeros(n, FLAME_NUM_EXPRESSION),
            global_orient=true_global_orient, neck_pose=torch.zeros(n, 3), jaw_pose=torch.zeros(n, 3),
            leye_pose=torch.zeros(n, 3), reye_pose=torch.zeros(n, 3), transl=true_transl,
        )
        landmarks_px = _project_points(out.joints[:, -NUM_FLAME_LANDMARKS:, :], torch.tensor(K, dtype=torch.float32)).numpy()

    occluded = np.array([False, False, False] + [True] * occlusion_len + [False, False])
    deca_pose = np.zeros((n, 6), dtype=np.float32)
    deca_pose[:, :3] = true_global_orient.numpy()  # DECA matches ground truth everywhere, including occluded frames

    base_kwargs = dict(
        landmarks_51=landmarks_px.astype(np.float32), landmarks_valid=~occluded,
        deca_exp=np.zeros((n, FLAME_NUM_EXPRESSION), dtype=np.float32), deca_pose=deca_pose,
        deca_shape=np.zeros((n, 100), dtype=np.float32), deca_valid=np.ones(n, dtype=bool),
        mica_shape=np.zeros((n, FLAME_NUM_BETAS), dtype=np.float32), mica_valid=np.ones(n, dtype=bool),
        intrinsics_k=K,
    )

    with_anchor = fit_clip(
        FitInputs(**base_kwargs), fps=TEST_FPS, device=device, stage1_iters=800, stage1_lr=0.03, stage2_iters=10,
        temporal_weight=1.0, global_orient_anchor_weight=10.0,
    )
    without_anchor = fit_clip(
        FitInputs(**base_kwargs), fps=TEST_FPS, device=device, stage1_iters=800, stage1_lr=0.03, stage2_iters=10,
        temporal_weight=1.0, global_orient_anchor_weight=0.0,
    )

    with_error = np.abs(with_anchor[KEY_GLOBAL_ORIENT][occluded, 1] - true_global_orient[occluded, 1].numpy())
    without_error = np.abs(without_anchor[KEY_GLOBAL_ORIENT][occluded, 1] - true_global_orient[occluded, 1].numpy())

    assert with_error.max() < 0.1
    assert without_error.min() > with_error.max()


@pytest.mark.skipif(not FLAME_ASSETS_PRESENT, reason="needs the FLAME model (see README's Setup section)")
def test_fit_clip_kabsch_anchor_recovers_occluded_orientation_spike():
    """Same structure as the DECA-global_orient-anchor test above, for the
    third orientation signal: face_pose_stabilization's rigid-landmark
    Kabsch rotation. `deca_pose` is left all-zero (wrong everywhere except
    by coincidence on the non-occluded frames, which sit at the true
    identity rotation) and `global_orient_anchor_weight=0` disables the DECA
    anchor entirely, isolating this anchor's own claim: on frames the
    reprojection loss can't see, kabsch_rotmat alone (calibrated against
    DECA on the *non*-occluded frames only, per calibrate_rotation_offset)
    should recover the true occluded spike."""
    K = np.array([[1000.0, 0, 512], [0, 1000.0, 384], [0, 0, 1]], dtype=np.float32)
    n = 7
    device = torch.device("cpu")
    model = _build_flame_model(device, batch_size=n)

    true_global_orient = torch.zeros(n, 3)
    true_global_orient[:, 1] = torch.tensor([0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0])
    true_transl = torch.zeros(n, 3)
    true_transl[:, 2] = 1.0
    with torch.no_grad():
        out = model(
            betas=torch.zeros(1, FLAME_NUM_BETAS).expand(n, -1), expression=torch.zeros(n, FLAME_NUM_EXPRESSION),
            global_orient=true_global_orient, neck_pose=torch.zeros(n, 3), jaw_pose=torch.zeros(n, 3),
            leye_pose=torch.zeros(n, 3), reye_pose=torch.zeros(n, 3), transl=true_transl,
        )
        landmarks_px = _project_points(out.joints[:, -NUM_FLAME_LANDMARKS:, :], torch.tensor(K, dtype=torch.float32)).numpy()

    occluded = np.array([False, False, False, True, True, False, False])

    r_offset_true = Rotation.from_euler("xyz", [8.0, -12.0, 5.0], degrees=True).as_matrix()
    r_global = Rotation.from_rotvec(true_global_orient.numpy()).as_matrix()
    kabsch_rotmat = (np.transpose(r_offset_true)[None] @ r_global).astype(np.float32)

    base_kwargs = dict(
        landmarks_51=landmarks_px.astype(np.float32), landmarks_valid=~occluded,
        deca_exp=np.zeros((n, FLAME_NUM_EXPRESSION), dtype=np.float32),
        deca_pose=np.zeros((n, 6), dtype=np.float32),
        deca_shape=np.zeros((n, 100), dtype=np.float32), deca_valid=np.ones(n, dtype=bool),
        mica_shape=np.zeros((n, FLAME_NUM_BETAS), dtype=np.float32), mica_valid=np.ones(n, dtype=bool),
        intrinsics_k=K,
        kabsch_confidence=np.ones(n, dtype=bool),
    )

    with_anchor = fit_clip(
        FitInputs(**base_kwargs, kabsch_rotmat=kabsch_rotmat),
        fps=TEST_FPS, device=device, stage1_iters=800, stage1_lr=0.03, stage2_iters=10,
        temporal_weight=1.0, global_orient_anchor_weight=0.0, kabsch_anchor_weight=10.0,
    )
    without_anchor = fit_clip(
        FitInputs(**{**base_kwargs, "kabsch_rotmat": None, "kabsch_confidence": None}),
        fps=TEST_FPS, device=device, stage1_iters=800, stage1_lr=0.03, stage2_iters=10,
        temporal_weight=1.0, global_orient_anchor_weight=0.0, kabsch_anchor_weight=10.0,
    )

    with_error = np.abs(with_anchor[KEY_GLOBAL_ORIENT][occluded, 1] - true_global_orient[occluded, 1].numpy())
    without_error = np.abs(without_anchor[KEY_GLOBAL_ORIENT][occluded, 1] - true_global_orient[occluded, 1].numpy())

    assert with_error.max() < 0.15
    assert without_error.min() > with_error.max()

