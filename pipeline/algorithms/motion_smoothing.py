"""Temporal smoothing to remove per-frame jitter from the pose estimators'
output, applied inside stage 2 (body) and stage 4 (hands) so their saved output
is already higher-quality than the raw model prediction.

Why the two stages need very different strengths: GVHMR (stage 2) runs a
temporal transformer over the whole clip, so the body arrives with frame-to-frame
consistency and only needs light polish. HaMeR (stage 4) infers every frame
independently from an isolated crop with no temporal model at all, so the hands
are far jitterier and need heavier smoothing, plus per-frame validity handling,
since a hand that wasn't detected on some frames must not bleed its placeholder
identity pose into its detected neighbours.

Rotations are smoothed in quaternion space (Savitzky-Golay per joint, with
sign-continuity enforced) rather than directly on axis-angle: naively filtering
across an axis-angle wraparound produces spikes instead of removing them.
Translation is smoothed with a zero-phase (filtfilt) Butterworth low-pass, which
adds no lag. Ported from this project's own ComfyUI `SmoothSMPLMotion` node
(savgol-in-quaternion + butterworth-filtfilt), extended here with validity-aware
gap handling for the hands, including `cap_long_gaps_with_hold`, which keeps
a long invalid run's straight-line interpolation from sweeping through a large,
visibly implausible rotation by holding most of the gap instead.

Finger joints go through `one_euro_filter_rotation_sequence` instead: an
adaptive filter that stays heavy (kills small residual wobble) while a joint is
nearly still and loosens automatically (tracks with low lag) once it starts
moving, all as one continuous blend, no separate smoothing-then-hysteresis
passes, and no discrete hold/release states to produce a stop-motion look. It
replaced an earlier two-stage design (savgol, then a hard hold-until-threshold
deadzone) that fixed a circular-precession artifact on still joints but visibly
snapped on release, which read as more stop-motion than the jitter it fixed.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from .contact_detection import contiguous_true_runs

POSE_AXIS_DIM = 3

# Internal defaults, the exposed knobs are the savgol *window* (per stage),
# the butterworth *cutoff* (body only), and the one-euro filter's *min_cutoff*/
# *beta* (fingers only); polynomial/filter order and the one-euro filter's own
# derivative cutoff are left fixed at values that rarely need touching.
DEFAULT_POLYORDER = 3
DEFAULT_BUTTER_ORDER = 2
DEFAULT_ONE_EURO_DCUTOFF_HZ = 1.0  # the "1€" in One Euro Filter, the paper's own standard default

# A per-frame hand estimator can briefly jump to a different pose, then jump
# straight back on the next frame. A real fast finger reversal can have that
# shape too, so this is deliberately only a *soft* ambiguity signal: Stage 4
# reduces the adaptive bandwidth rather than invalidating the measurement.
TRANSIENT_REVERSAL_MIN_DEGREES = 6.0
TRANSIENT_REVERSAL_BETA_SCALE = 0.15


def _odd_window(window: int, n_frames: int) -> int | None:
    """Clamp a requested savgol window to an odd value in [3, n_frames], or None
    if the clip is too short (< 3 frames) to filter meaningfully."""
    if n_frames < 3:
        return None
    w = min(window, n_frames)
    if w % 2 == 0:
        w -= 1
    return max(w, 3)


def fill_invalid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid rows of `values` (T, C) per channel from the valid ones, so
    the filter downstream never sees the placeholder poses on undetected frames.
    `np.interp`'s own default boundary behavior gives exactly the two occlusion
    cases their right treatment for free: an *interior* run of invalid frames
    (bounded by a valid frame on both sides, the tracked thing reappears) is
    linearly interpolated between those two real values; a *leading* or
    *trailing* run (invalid all the way to one end of the clip, it never
    reappears, or wasn't yet detected at the start) has no second real endpoint
    to interpolate toward, so `np.interp` holds it constant at the one real
    value it does have (`left`/`right` default to `fp[0]`/`fp[-1]`), a freeze,
    not a fabricated guess. This is the caller's actual occlusion contract, not
    just an internal filtering detail, so downstream code depends on it.

    Degenerate case, the same contract taken to its limit: if `valid` is False
    for the *entire* clip, there is no single real value left to hold at
    either, so `values` is returned unchanged rather than raising (`np.interp`
    itself errors on an empty sample-point array) or fabricating a default."""
    if not valid.any():
        return values.copy()
    frame_idx = np.arange(values.shape[0])
    valid_idx = frame_idx[valid]
    filled = values.copy()
    for channel in range(values.shape[1]):
        filled[:, channel] = np.interp(frame_idx, valid_idx, values[valid, channel])
    return filled


def cap_long_gaps_with_hold(
    values: np.ndarray, valid: np.ndarray, max_bridge_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """For any *interior* invalid run longer than `max_bridge_frames`, freeze
    its early portion at the last known-good (entry) value and mark it valid,
    `fill_invalid` then only has the final `max_bridge_frames` left to
    interpolate across, blending into the recovery instead of sweeping across
    the whole gap.

    Straight-line interpolation across a long gap (real clips commonly have
    40-90+ frame invalid runs) can visibly swing through a large rotation,
    reading as physically impossible even though each frame is individually
    "between" two real values. There's no way to know what really happened
    during a multi-second occlusion, so holding the last confirmed pose is
    the same conservative default already used for leading/trailing gaps
    (`fill_invalid`), just extended to the bulk of a long interior one too.

    Runs at or under `max_bridge_frames` are left untouched (interpolating
    already looks fine at that length), as are leading/trailing runs
    (`fill_invalid`'s own boundary behavior already freezes those, and a
    leading run has no entry value to hold from anyway).

    values: (T, C). valid: (T,) bool. Returns (values, valid), both copies.
    """
    valid = np.asarray(valid, dtype=bool)
    held_values = values.copy()
    held_valid = valid.copy()

    for start, end in contiguous_true_runs(~valid):  # inclusive
        if start == 0 or end == len(valid) - 1:
            continue  # leading/trailing, fill_invalid already freezes these correctly
        run_length = end - start + 1
        if run_length <= max_bridge_frames:
            continue  # short enough to interpolate as-is
        held_end = end - max_bridge_frames + 1  # exclusive, [start, held_end) gets held
        held_values[start:held_end] = values[start - 1]
        held_valid[start:held_end] = True

    return held_values, held_valid


def hemisphere_aligned_quats(joint_axis_angle: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    """One joint's (T, 3) axis-angle sequence -> (T, 4) xyzw quaternions, sign
    -continuity enforced across *detected* frames (a unit quaternion and its
    negation are the same rotation, so each is flipped to share a hemisphere
    with the previous real one, otherwise interpolation/filtering treats a
    sign flip as a huge jump), then gap-filled per `fill_invalid` if `valid` is
    given. Shared by both rotation-smoothing functions below (and by
    `gvhmr_postprocess.pp_bridge_low_confidence_root_motion`, which needs the
    identical gap-bridging behavior for the pelvis's own root orientation) so
    the fiddly continuity/gap-fill logic exists in exactly one place."""
    quats = Rotation.from_rotvec(joint_axis_angle).as_quat()  # (T, 4) xyzw
    n_frames = quats.shape[0]
    last = None
    for t in range(n_frames):
        if valid is None or valid[t]:
            if last is not None and float(np.dot(quats[t], last)) < 0:
                quats[t] = -quats[t]
            last = quats[t]
    if valid is not None:
        quats = fill_invalid(quats, valid)
    return quats


def smooth_rotation_sequence(
    axis_angle: np.ndarray, window: int, polyorder: int = DEFAULT_POLYORDER, valid: np.ndarray | None = None
) -> np.ndarray:
    """Smooth a per-frame axis-angle rotation sequence in quaternion space.

    Args:
        axis_angle: (T, ...) where the trailing dims flatten to a multiple of 3
            (e.g. (T, 3), (T, 63), (T, 15, 3)), each 3-vector a joint rotation.
        window: Savitzky-Golay window in frames (clamped to odd, <= T, >= 3).
        polyorder: polynomial order fit within each window (< window).
        valid: optional (T,) bool. Where given, invalid frames are filled before
            filtering: interpolated if bounded by valid frames on both sides
            (the tracked thing reappears), held constant at the nearest valid
            frame if not (occlusion runs to either end of the clip), see
            `fill_invalid`. The filter then runs over the filled series.

    Returns the same shape/dtype, smoothed. Returned unchanged when the clip is
    too short, or when `valid` marks too few real frames to smooth meaningfully.
    """
    axis_angle = np.asarray(axis_angle)
    original_shape = axis_angle.shape
    n_frames = original_shape[0]
    flat = axis_angle.reshape(n_frames, -1)
    if flat.shape[1] % POSE_AXIS_DIM != 0:
        raise ValueError(f"rotation sequence trailing dims not divisible by 3: {original_shape}")
    n_joints = flat.shape[1] // POSE_AXIS_DIM
    joints = flat.reshape(n_frames, n_joints, POSE_AXIS_DIM)

    w = _odd_window(window, n_frames)
    if w is None:
        return axis_angle
    poly = min(polyorder, w - 1)

    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if int(valid.sum()) < max(3, poly + 1):
            return axis_angle  # too few detected frames to smooth meaningfully

    out = np.zeros_like(joints)
    for j in range(n_joints):
        quats = hemisphere_aligned_quats(joints[:, j, :], valid)
        smoothed = savgol_filter(quats, w, poly, axis=0)
        norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        smoothed /= norms
        out[:, j, :] = Rotation.from_quat(smoothed).as_rotvec()

    return out.reshape(original_shape).astype(axis_angle.dtype, copy=False)


def _one_euro_alpha(cutoff_hz: float, dt: float) -> float:
    """Exponential-smoothing weight for a first-order low-pass at `cutoff_hz`,
    sampled every `dt` seconds (the One Euro Filter's own formula: higher
    cutoff -> alpha closer to 1 -> more weight on the new sample, less lag)."""
    tau = 1.0 / (2.0 * np.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


def transient_rotation_reversal_mask(
    axis_angle: np.ndarray, fps: float, valid: np.ndarray | None = None,
) -> np.ndarray:
    """Flag each joint's large one-frame pose pop before any smoothing.

    A flagged frame sits between two large opposing raw rotations. The check
    intentionally runs on the original sequence: a centered pre-pass would
    spread a one-frame pop across its neighbours and erase the very reversal
    that identifies it. A short held keypress has a zero/same-direction next
    rotation and is therefore not flagged. A true rapid reversal can be
    flagged too, but downstream treats the flag as soft smoothing evidence,
    never as a rejected measurement.
    """
    axis_angle = np.asarray(axis_angle)
    n_frames = axis_angle.shape[0]
    flat = axis_angle.reshape(n_frames, -1)
    if flat.shape[1] % POSE_AXIS_DIM != 0:
        raise ValueError(f"rotation sequence trailing dims not divisible by {POSE_AXIS_DIM}: {axis_angle.shape}")
    n_joints = flat.shape[1] // POSE_AXIS_DIM
    joints = flat.reshape(n_frames, n_joints, POSE_AXIS_DIM)
    out = np.zeros((n_frames, n_joints), dtype=bool)
    if n_frames < 3:
        return out

    dt = 1.0 / fps
    min_speed = np.radians(TRANSIENT_REVERSAL_MIN_DEGREES) / dt
    for joint in range(n_joints):
        quats = hemisphere_aligned_quats(joints[:, joint, :], valid)
        rotations = Rotation.from_quat(quats)
        omega = (rotations[1:] * rotations[:-1].inv()).as_rotvec() / dt
        current = omega[:-1]
        following = omega[1:]
        out[1:-1, joint] = (
            (np.linalg.norm(current, axis=1) >= min_speed)
            & (np.linalg.norm(following, axis=1) >= min_speed)
            & (np.einsum("ij,ij->i", current, following) < 0.0)
        )
    return out


def slerp(q0: np.ndarray, q1: np.ndarray, frac: float) -> np.ndarray:
    """Spherical interpolation from unit quaternion q0 toward q1 by `frac`
    (0 -> q0, 1 -> q1), taking the shorter path across the double-cover."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:  # nearly identical, linear blend avoids dividing by sin(angle)~0
        result = q0 + frac * (q1 - q0)
        return result / np.linalg.norm(result)
    angle = np.arccos(dot)
    sin_angle = np.sin(angle)
    return (np.sin((1.0 - frac) * angle) / sin_angle) * q0 + (np.sin(frac * angle) / sin_angle) * q1


def one_euro_filter_rotation_sequence(
    axis_angle: np.ndarray,
    fps: float,
    min_cutoff_hz: float,
    beta: float,
    dcutoff_hz: float = DEFAULT_ONE_EURO_DCUTOFF_HZ,
    valid: np.ndarray | None = None,
    transient_reversal_mask: np.ndarray | None = None,
    beta_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Adaptively smooth a per-frame rotation sequence: heavy smoothing while
    a joint is nearly still, loosening automatically (low lag) once it
    starts moving, the One Euro Filter (Casiez, Roussel & Vogel 2012),
    adapted for rotations.

    Filtering each quaternion component independently (as the paper does
    for scalars) would let each pick its own adaptive cutoff per frame,
    reintroducing a subtler version of the per-component distortion this
    replaced. Instead one shared speed estimate drives the whole joint: a
    *signed* angular-velocity vector between consecutive raw samples
    (relative rotation's axis-angle / dt), smoothed component-wise
    (`dcutoff_hz`) *before* taking its norm. Filtering the signed vector
    first matters, opposite-direction noise cancels on averaging; an
    earlier version filtered the magnitude instead, which can't cancel
    (never negative), leaving a persistent nonzero "speed" floor at rest
    even under heavy smoothing, confirmed on real data. That scalar speed
    sets one adaptive cutoff per frame, blending the previous filtered
    orientation toward the new raw one via proper quaternion slerp, one
    rotation-aware decision per frame, not four independent scalar ones.

    Args:
        axis_angle: (T, ...), trailing dims a multiple of 3 (same
            convention as `smooth_rotation_sequence`).
        fps: sample rate (frames/sec). min_cutoff_hz: cutoff at zero speed
            (lower = smoother at rest). beta: how fast the cutoff rises
            with speed (higher = faster response, more jitter passed
            through while moving). dcutoff_hz: fixed cutoff for the speed
            estimate itself.
        valid: optional (T,) bool, gap-filled like `smooth_rotation_
            sequence` (`fill_invalid`).
        transient_reversal_mask: optional ``(T, J)`` flags produced from the
            original, pre-smoothed sequence. A flagged joint/frame keeps the
            adaptive bandwidth low to remove a brief estimator pose pop.
        beta_scale: optional ``(T,)`` positive multiplier for the adaptive
            bandwidth. Callers can lower it while a measurement is known to
            be unreliable, then restore it gradually without changing the
            ordinary motion profile elsewhere in the clip.

    Returns the same shape/dtype, unchanged if fewer than 2 (valid) frames.
    """
    axis_angle = np.asarray(axis_angle)
    original_shape = axis_angle.shape
    n_frames = original_shape[0]
    if n_frames < 2:
        return axis_angle
    flat = axis_angle.reshape(n_frames, -1)
    if flat.shape[1] % POSE_AXIS_DIM != 0:
        raise ValueError(f"rotation sequence trailing dims not divisible by 3: {original_shape}")
    n_joints = flat.shape[1] // POSE_AXIS_DIM
    joints = flat.reshape(n_frames, n_joints, POSE_AXIS_DIM)

    if transient_reversal_mask is not None:
        transient_reversal_mask = np.asarray(transient_reversal_mask, dtype=bool)
        if transient_reversal_mask.shape != (n_frames, n_joints):
            raise ValueError(
                "transient_reversal_mask must have shape "
                f"{(n_frames, n_joints)}, got {transient_reversal_mask.shape}"
            )

    if beta_scale is not None:
        beta_scale = np.asarray(beta_scale, dtype=float)
        if beta_scale.shape != (n_frames,):
            raise ValueError(f"beta_scale must have shape {(n_frames,)}, got {beta_scale.shape}")
        if np.any(beta_scale <= 0.0):
            raise ValueError("beta_scale must be strictly positive")

    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if int(valid.sum()) < 2:
            return axis_angle  # too few real frames to estimate speed from

    dt = 1.0 / fps
    speed_alpha = _one_euro_alpha(dcutoff_hz, dt)  # fixed given dcutoff/dt, same every frame
    out = np.zeros_like(joints)
    for j in range(n_joints):
        quats = hemisphere_aligned_quats(joints[:, j, :], valid)

        # Signed angular-velocity vector between every consecutive pair of raw
        # samples, computed once for the whole sequence (not recursive, so no
        # need for a per-frame loop here), the relative rotation's own
        # axis-angle, scaled by 1/dt.
        r = Rotation.from_quat(quats)
        omega_seq = (r[1:] * r[:-1].inv()).as_rotvec() / dt  # (T-1, 3)

        filtered = np.empty_like(quats)
        filtered[0] = quats[0]
        smoothed_omega = np.zeros(3)
        for t in range(1, n_frames):
            smoothed_omega = speed_alpha * omega_seq[t - 1] + (1.0 - speed_alpha) * smoothed_omega
            speed = float(np.linalg.norm(smoothed_omega))

            effective_beta = beta
            if beta_scale is not None:
                effective_beta *= beta_scale[t]
            if transient_reversal_mask is not None and transient_reversal_mask[t, j]:
                effective_beta *= TRANSIENT_REVERSAL_BETA_SCALE

            value_alpha = _one_euro_alpha(min_cutoff_hz + effective_beta * speed, dt)
            filtered[t] = slerp(filtered[t - 1], quats[t], value_alpha)

        out[:, j, :] = Rotation.from_quat(filtered).as_rotvec()

    return out.reshape(original_shape).astype(axis_angle.dtype, copy=False)


def one_euro_filter_sequence(
    values: np.ndarray,
    fps: float,
    min_cutoff_hz: float,
    beta: float,
    dcutoff_hz: float = DEFAULT_ONE_EURO_DCUTOFF_HZ,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Linear-vector counterpart to `one_euro_filter_rotation_sequence`, for
    per-frame quantities that aren't rotations (e.g. FLAME's PCA `expression`
    coefficients), same adaptive heavy-at-rest/loose-while-moving behavior,
    exponential blending in place of slerp.

    Shares the parent filter's own reasoning for using one scalar speed to
    drive every channel at once (`||smoothed velocity vector||`, not a
    per-channel adaptive cutoff): filtering each channel independently would
    let e.g. one PCA component's own noise floor pick a different cutoff than
    its neighbors on the same frame, reintroducing the per-component
    distortion the shared-speed design exists to avoid.

    Args:
        values: (T, C).
        fps: sample rate (frames/sec). min_cutoff_hz: cutoff at zero speed
            (lower = smoother at rest). beta: how fast the cutoff rises with
            speed. dcutoff_hz: fixed cutoff for the speed estimate itself.
        valid: optional (T,) bool, gap-filled first (`fill_invalid`).

    Returns the same shape/dtype, unchanged if fewer than 2 (valid) frames.
    """
    values = np.asarray(values)
    n_frames = values.shape[0]
    if n_frames < 2:
        return values

    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if int(valid.sum()) < 2:
            return values
        values = fill_invalid(values, valid)

    dt = 1.0 / fps
    speed_alpha = _one_euro_alpha(dcutoff_hz, dt)

    velocity = (values[1:] - values[:-1]) / dt  # (T-1, C)

    filtered = np.empty_like(values)
    filtered[0] = values[0]
    smoothed_velocity = np.zeros(values.shape[1])
    for t in range(1, n_frames):
        smoothed_velocity = speed_alpha * velocity[t - 1] + (1.0 - speed_alpha) * smoothed_velocity
        speed = float(np.linalg.norm(smoothed_velocity))

        value_alpha = _one_euro_alpha(min_cutoff_hz + beta * speed, dt)
        filtered[t] = value_alpha * values[t] + (1.0 - value_alpha) * filtered[t - 1]

    return filtered.astype(values.dtype, copy=False)


def _rdp_knot_indices(n_frames: int, tol: float, fit, error) -> np.ndarray:
    """Ramer-Douglas-Peucker knot selection, generalized over what "straight-line
    fit" and "deviation" mean, the same recursive tree shared by both rotation
    (quaternion slerp, geodesic error) and translation (linear interpolation,
    Euclidean error) decimation below; only `fit`/`error` differ per domain.

    Returns the sorted frame indices (always including both endpoints) of a
    sparse keyframe set such that `fit`'s interpolation between consecutive kept
    knots stays within `tol` of every original in-between frame, per `error`.
    This is the general form of what Blender's Decimate (Allowed Change) does
    per F-curve: recursively keep the single worst-fit frame as a new knot until
    the fit is everywhere within tolerance.

    Args:
        n_frames: length of the sequence being decimated.
        tol: max allowed deviation, in `error`'s own units.
        fit(a, b, idx): interpolates anchors at frames a, b, evaluated at the
            frame indices in `idx` (an array of interior frames between a, b).
        error(fitted, idx): per-frame scalar deviation of `fitted` from the
            true samples at `idx`.
    """
    keep = np.zeros(n_frames, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n_frames - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue  # no interior frames between these two knots
        idx = np.arange(a + 1, b)
        err = error(fit(a, b, idx), idx)
        worst = int(err.argmax())
        if err[worst] > tol:
            mid = a + 1 + worst
            keep[mid] = True
            stack.append((a, mid))
            stack.append((mid, b))
    return np.flatnonzero(keep)


def decimate_rotation_sequence(axis_angle: np.ndarray, tolerance_deg: float, valid: np.ndarray | None = None) -> np.ndarray:
    """Replace a per-frame rotation sequence with a mathematically smooth
    curve fit through a sparse set of keyframes, keyframe reduction, the
    way an animator cleans up mocap by hand (Blender's Decimate), done in
    quaternion space instead of per Euler channel. Unlike a filter (a
    moving average that always leaves some residual jitter), the fitted
    curve has no degrees of freedom to wiggle between keyframes, so
    high-frequency content is gone by construction. `tolerance_deg` is the
    one knob, max angular deviation (degrees) the fit is allowed from the
    original; larger means fewer keyframes and a flatter curve, smaller
    hugs the input more tightly (and past a point starts preserving jitter
    again).

    Knots are chosen by geodesic-error RDP (`_rdp_knot_indices`), rebuilt by
    slerping between consecutive knots. A smooth C2 spline was tried (to
    soften the velocity "snap" at a knot) but rejected: it overshoots
    between sparse knots, breaking the tolerance guarantee. Slerp keeps the
    output provably within `tolerance_deg`, and at the knot densities this
    runs at, segments are short enough that the residual snap isn't visible.

    Args:
        axis_angle: (T, ...), trailing dims a multiple of 3 (same
            convention as the smoothing functions above).
        tolerance_deg: max allowed angular deviation of the fit, degrees.
        valid: optional (T,) bool, gap-filled before fitting (`fill_
            invalid`), the occlusion contract survives decimation: a
            frozen trailing gap needs no interior knots, an interpolated
            interior gap is already representable by its two bounding knots.

    Returns the same shape/dtype, unchanged if the clip is too short
    (< 3 frames) or `valid` marks too few real frames to fit through.
    """
    axis_angle = np.asarray(axis_angle)
    original_shape = axis_angle.shape
    n_frames = original_shape[0]
    if n_frames < 3:
        return axis_angle
    flat = axis_angle.reshape(n_frames, -1)
    if flat.shape[1] % POSE_AXIS_DIM != 0:
        raise ValueError(f"rotation sequence trailing dims not divisible by 3: {original_shape}")
    n_joints = flat.shape[1] // POSE_AXIS_DIM
    joints = flat.reshape(n_frames, n_joints, POSE_AXIS_DIM)

    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if int(valid.sum()) < 2:
            return axis_angle  # too few real frames to fit through

    tol_rad = np.radians(tolerance_deg)
    all_frames = np.arange(n_frames, dtype=float)
    out = np.zeros_like(joints)
    for j in range(n_joints):
        quats = hemisphere_aligned_quats(joints[:, j, :], valid)

        def fit(a, b, idx, quats=quats):
            return Slerp([a, b], Rotation.from_quat(quats[[a, b]]))(idx)

        def error(fitted, idx, quats=quats):
            return (fitted * Rotation.from_quat(quats[idx]).inv()).magnitude()

        knots = _rdp_knot_indices(n_frames, tol_rad, fit, error)
        if knots.shape[0] >= n_frames:  # every frame kept, nothing to fit
            out[:, j, :] = joints[:, j, :]
            continue
        arc = Slerp(knots.astype(float), Rotation.from_quat(quats[knots]))
        out[:, j, :] = arc(all_frames).as_rotvec()

    return out.reshape(original_shape).astype(axis_angle.dtype, copy=False)


def decimate_translation_sequence(transl: np.ndarray, tolerance_m: float) -> np.ndarray:
    """Translation counterpart to `decimate_rotation_sequence`, sharing its RDP
    core (`_rdp_knot_indices`) with a Euclidean fit/error instead of a
    geodesic/slerp one, straight-line RDP is in fact the classic Douglas-
    Peucker algorithm's original domain (cartographic point simplification);
    quaternion decimation above is the generalization, not the other way round.

    `tolerance_m` is the max allowed straight-line deviation, in meters, exactly
    analogous to `tolerance_deg` for rotations: larger means fewer keyframes and
    a flatter curve: enough to hold a near-still root motionless instead of
    letting it float, without needing a separate lock/release/hysteresis
    mechanism to get there.

    Reconstruction reuses `fill_invalid` rather than a bespoke linear
    interpolant: from that function's point of view, the frames that AREN'T
    kept as knots are exactly the "invalid" gap it already knows how to
    linearly interpolate across, with the knots standing in as the "valid" frames.

    Args:
        transl: (T, 3) root translation.
        tolerance_m: max allowed deviation of the fit, in meters.

    Returns the same shape/dtype. Returned unchanged when the clip is too short
    (< 3 frames) to have any interior frame to drop.
    """
    transl = np.asarray(transl)
    n_frames = transl.shape[0]
    if n_frames < 3:
        return transl

    def fit(a, b, idx):
        t = (idx - a) / (b - a)
        return transl[a] + t[:, None] * (transl[b] - transl[a])

    def error(fitted, idx):
        return np.linalg.norm(fitted - transl[idx], axis=1)

    knots = _rdp_knot_indices(n_frames, tolerance_m, fit, error)
    if knots.shape[0] >= n_frames:  # every frame kept, nothing to fit
        return transl

    knot_mask = np.zeros(n_frames, dtype=bool)
    knot_mask[knots] = True
    return fill_invalid(transl, knot_mask).astype(transl.dtype, copy=False)


def smooth_position_sequence(
    positions: np.ndarray, window: int, polyorder: int = DEFAULT_POLYORDER, valid: np.ndarray | None = None
) -> np.ndarray:
    """Smooth a per-frame Euclidean/pixel-position sequence
    with a centered Savitzky-Golay filter, no quaternion
    reprojection needed afterward, unlike `smooth_rotation_sequence`, so this
    is a single vectorized `savgol_filter` call over all flattened trailing
    dims at once. Shares this module's gap-handling (`fill_invalid`) and
    window-clamping (`_odd_window`) with the rotation/translation smoothers.

    positions: (T, ...), trailing dims flattened for filtering and restored
    on return. valid: optional (T,) bool, gap-filled before filtering exactly
    like `smooth_rotation_sequence` (interior gaps interpolated, leading/
    trailing gaps held).

    Returns the same shape/dtype, unchanged if the clip is too short or
    `valid` marks too few real frames to filter meaningfully.
    """
    positions = np.asarray(positions)
    original_shape = positions.shape
    n_frames = original_shape[0]
    flat = positions.reshape(n_frames, -1)

    w = _odd_window(window, n_frames)
    if w is None:
        return positions
    poly = min(polyorder, w - 1)

    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if int(valid.sum()) < max(3, poly + 1):
            return positions
        flat = fill_invalid(flat, valid)

    smoothed = savgol_filter(flat, w, poly, axis=0)
    return smoothed.reshape(original_shape).astype(positions.dtype, copy=False)


def smooth_translation_sequence(transl: np.ndarray, cutoff: float, order: int = DEFAULT_BUTTER_ORDER) -> np.ndarray:
    """Zero-phase Butterworth low-pass on a (T, 3) root-translation sequence.
    `cutoff` is a fraction of Nyquist (0-0.5). Returned unchanged when the clip is
    too short for `filtfilt`'s edge padding."""
    transl = np.asarray(transl)
    b, a = butter(order, cutoff)
    padlen = 3 * max(len(a), len(b))
    if transl.shape[0] <= padlen:
        return transl
    return filtfilt(b, a, transl, axis=0).astype(transl.dtype, copy=False)
