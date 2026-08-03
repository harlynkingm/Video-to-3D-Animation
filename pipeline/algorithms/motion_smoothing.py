"""Temporal smoothing to remove per-frame jitter from the pose estimators'
output, applied inside stage 2 (body) and stage 4 (hands) so their saved output
is already higher-quality than the raw model prediction.

Why the two stages need very different strengths: GVHMR (stage 2) runs a
temporal transformer over the whole clip, so the body arrives with frame-to-frame
consistency and only needs light polish. HaMeR (stage 4) infers every frame
independently from an isolated crop with no temporal model at all, so the hands
are far jitterier and need heavier smoothing -- plus per-frame validity handling,
since a hand that wasn't detected on some frames must not bleed its placeholder
identity pose into its detected neighbours.

Rotations are smoothed in quaternion space (Savitzky-Golay per joint, with
sign-continuity enforced) rather than directly on axis-angle: naively filtering
across an axis-angle wraparound produces spikes instead of removing them.
Translation is smoothed with a zero-phase (filtfilt) Butterworth low-pass, which
adds no lag. Ported from this project's own ComfyUI `SmoothSMPLMotion` node
(savgol-in-quaternion + butterworth-filtfilt), extended here with validity-aware
gap handling for the hands.

Finger joints go through `one_euro_filter_rotation_sequence` instead: an
adaptive filter that stays heavy (kills small residual wobble) while a joint is
nearly still and loosens automatically (tracks with low lag) once it starts
moving, all as one continuous blend -- no separate smoothing-then-hysteresis
passes, and no discrete hold/release states to produce a stop-motion look. It
replaced an earlier two-stage design (savgol, then a hard hold-until-threshold
deadzone) that fixed a circular-precession artifact on still joints but visibly
snapped on release, which read as more stop-motion than the jitter it fixed.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.spatial.transform import Rotation, Slerp

POSE_AXIS_DIM = 3

# Internal defaults -- the exposed knobs are the savgol *window* (per stage),
# the butterworth *cutoff* (body only), and the one-euro filter's *min_cutoff*/
# *beta* (fingers only); polynomial/filter order and the one-euro filter's own
# derivative cutoff are left fixed at values that rarely need touching.
DEFAULT_POLYORDER = 3
DEFAULT_BUTTER_ORDER = 2
DEFAULT_ONE_EURO_DCUTOFF_HZ = 1.0  # the "1€" in One Euro Filter -- the paper's own standard default


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
    (bounded by a valid frame on both sides -- the tracked thing reappears) is
    linearly interpolated between those two real values; a *leading* or
    *trailing* run (invalid all the way to one end of the clip -- it never
    reappears, or wasn't yet detected at the start) has no second real endpoint
    to interpolate toward, so `np.interp` holds it constant at the one real
    value it does have (`left`/`right` default to `fp[0]`/`fp[-1]`) -- a freeze,
    not a fabricated guess. This is the caller's actual occlusion contract, not
    just an internal filtering detail, so downstream code depends on it."""
    frame_idx = np.arange(values.shape[0])
    valid_idx = frame_idx[valid]
    filled = values.copy()
    for channel in range(values.shape[1]):
        filled[:, channel] = np.interp(frame_idx, valid_idx, values[valid, channel])
    return filled


def hemisphere_aligned_quats(joint_axis_angle: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    """One joint's (T, 3) axis-angle sequence -> (T, 4) xyzw quaternions, sign
    -continuity enforced across *detected* frames (a unit quaternion and its
    negation are the same rotation, so each is flipped to share a hemisphere
    with the previous real one -- otherwise interpolation/filtering treats a
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
            (e.g. (T, 3), (T, 63), (T, 15, 3)) -- each 3-vector a joint rotation.
        window: Savitzky-Golay window in frames (clamped to odd, <= T, >= 3).
        polyorder: polynomial order fit within each window (< window).
        valid: optional (T,) bool. Where given, invalid frames are filled before
            filtering: interpolated if bounded by valid frames on both sides
            (the tracked thing reappears), held constant at the nearest valid
            frame if not (occlusion runs to either end of the clip) -- see
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


def slerp(q0: np.ndarray, q1: np.ndarray, frac: float) -> np.ndarray:
    """Spherical interpolation from unit quaternion q0 toward q1 by `frac`
    (0 -> q0, 1 -> q1), taking the shorter path across the double-cover."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:  # nearly identical -- linear blend avoids dividing by sin(angle)~0
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
) -> np.ndarray:
    """Adaptively smooth a per-frame rotation sequence: heavy smoothing while a
    joint is nearly still, loosening automatically (low lag) once it starts
    moving -- the One Euro Filter (Casiez, Roussel & Vogel 2012), adapted for
    rotations.

    The paper's own filter operates on independent scalars (e.g. a cursor's x
    and y separately). Doing that naively per quaternion component here would
    let each of the 4 components pick its own adaptive cutoff per frame based
    on its own local behaviour -- reintroducing a subtler version of the exact
    artifact this replaced (per-component quaternion filtering + renormalizing
    can distort the true rotation axis). Instead, one shared speed estimate
    drives the whole joint, computed the way the paper's own derivative is:
    a *signed* angular-velocity vector between consecutive raw samples
    (the relative rotation's axis-angle, scaled by 1/dt) -- not its magnitude.
    That vector is smoothed component-wise by its own fixed-cutoff low-pass
    (`dcutoff_hz`) *before* taking its norm to get a scalar speed. Filtering the
    signed vector first matters: opposite-direction noise cancels on averaging,
    the way the paper's own signed scalar derivative does. Filtering the
    magnitude instead (as an earlier version of this function did) cannot --
    a magnitude is never negative, so rectified per-frame noise has a nonzero
    mean no matter how heavily it's smoothed, confirmed on real data: even a
    very heavy low-pass left a persistent, non-shrinking "speed" floor at rest,
    which kept nudging the cutoff up and let noise leak through as a persistent
    low-amplitude wobble rather than the intended calm hold. The resulting
    scalar speed sets one adaptive cutoff per frame, which blends the previous
    filtered orientation toward the new raw one via proper quaternion slerp --
    a single rotation-aware decision per frame, not four independent scalar ones.

    Args:
        axis_angle: (T, ...) where the trailing dims flatten to a multiple of 3,
            same convention as `smooth_rotation_sequence`.
        fps: sample rate (frames/sec) -- the filter's notion of real time.
        min_cutoff_hz: cutoff at zero speed. Lower = smoother/more lag at rest.
        beta: how fast the cutoff rises with speed. Higher = faster response to
            real motion, at the cost of passing more raw jitter through while moving.
        dcutoff_hz: fixed cutoff for smoothing the speed estimate itself.
        valid: optional (T,) bool, gap-filled the same way as
            `smooth_rotation_sequence` -- see `fill_invalid`.

    Returns the same shape/dtype. Returned unchanged when there are fewer than
    2 frames, or fewer than 2 valid frames to filter from.
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

    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if int(valid.sum()) < 2:
            return axis_angle  # too few real frames to estimate speed from

    dt = 1.0 / fps
    speed_alpha = _one_euro_alpha(dcutoff_hz, dt)  # fixed given dcutoff/dt -- same every frame
    out = np.zeros_like(joints)
    for j in range(n_joints):
        quats = hemisphere_aligned_quats(joints[:, j, :], valid)

        # Signed angular-velocity vector between every consecutive pair of raw
        # samples, computed once for the whole sequence (not recursive, so no
        # need for a per-frame loop here) -- the relative rotation's own
        # axis-angle, scaled by 1/dt.
        r = Rotation.from_quat(quats)
        omega_seq = (r[1:] * r[:-1].inv()).as_rotvec() / dt  # (T-1, 3)

        filtered = np.empty_like(quats)
        filtered[0] = quats[0]
        smoothed_omega = np.zeros(3)
        for t in range(1, n_frames):
            smoothed_omega = speed_alpha * omega_seq[t - 1] + (1.0 - speed_alpha) * smoothed_omega
            speed = float(np.linalg.norm(smoothed_omega))

            value_alpha = _one_euro_alpha(min_cutoff_hz + beta * speed, dt)
            filtered[t] = slerp(filtered[t - 1], quats[t], value_alpha)

        out[:, j, :] = Rotation.from_quat(filtered).as_rotvec()

    return out.reshape(original_shape).astype(axis_angle.dtype, copy=False)


def _rdp_knot_indices(n_frames: int, tol: float, fit, error) -> np.ndarray:
    """Ramer-Douglas-Peucker knot selection, generalized over what "straight-line
    fit" and "deviation" mean -- the same recursive tree shared by both rotation
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
    """Replace a per-frame rotation sequence with a mathematically smooth curve
    fitted through a sparse set of keyframes -- keyframe reduction, the way an
    animator would clean up mocap by hand (Blender's Decimate), but done in
    quaternion space instead of per Euler channel.

    Unlike a filter, which is a moving average that always leaves *some* residual
    of the input jitter in its passband, this removes jitter outright: the output
    between keyframes is a fitted curve with no degrees of freedom to wiggle, so
    high-frequency content is gone by construction. `tolerance_deg` is the one
    knob -- the maximum angular deviation (degrees) the fitted curve is allowed
    from the original: larger means fewer keyframes and a smoother, flatter curve
    that departs further from the raw motion; smaller hugs the input more tightly
    (and, past a point, starts preserving its jitter again).

    Knots are chosen by geodesic-error RDP (`_rdp_knot_indices`); the full
    per-frame sequence is then rebuilt by slerping between consecutive knots. A
    smooth C2 spline was tried for the reconstruction (to soften the velocity
    "snap" at a hard direction-change knot) but rejected: it overshoots between
    sparse knots, swinging the fit several times `tolerance_deg` past the real
    pose -- both a new artifact of its own and a broken tolerance guarantee.
    Slerp instead keeps the output provably within `tolerance_deg` of the input,
    and at the knot densities this runs at (the flat regions decimate hard, the
    fast regions keep most frames as knots) the segments are short enough that
    the residual snap is not visible.

    Args:
        axis_angle: (T, ...) trailing dims flatten to a multiple of 3, same
            convention as the smoothing functions above.
        tolerance_deg: max allowed angular deviation of the fit, in degrees.
        valid: optional (T,) bool, gap-filled the same way as
            `smooth_rotation_sequence` before fitting -- see `fill_invalid`. The
            occlusion contract survives decimation: a frozen (constant) trailing
            gap keeps needing no interior knots, and a linearly-interpolated
            interior gap is already representable by its two bounding knots.

    Returns the same shape/dtype. Returned unchanged when the clip is too short
    (< 3 frames), or when `valid` marks too few real frames to fit through
    (mirrors `smooth_rotation_sequence`/`one_euro_filter_rotation_sequence`'s
    own guard -- with zero real frames there's nothing for `fill_invalid` to
    interpolate from).
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
        if knots.shape[0] >= n_frames:  # every frame kept -- nothing to fit
            out[:, j, :] = joints[:, j, :]
            continue
        arc = Slerp(knots.astype(float), Rotation.from_quat(quats[knots]))
        out[:, j, :] = arc(all_frames).as_rotvec()

    return out.reshape(original_shape).astype(axis_angle.dtype, copy=False)


def decimate_translation_sequence(transl: np.ndarray, tolerance_m: float) -> np.ndarray:
    """Translation counterpart to `decimate_rotation_sequence`, sharing its RDP
    core (`_rdp_knot_indices`) with a Euclidean fit/error instead of a
    geodesic/slerp one -- straight-line RDP is in fact the classic Douglas-
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
    if knots.shape[0] >= n_frames:  # every frame kept -- nothing to fit
        return transl

    knot_mask = np.zeros(n_frames, dtype=bool)
    knot_mask[knots] = True
    return fill_invalid(transl, knot_mask).astype(transl.dtype, copy=False)


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
