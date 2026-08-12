"""Two independent uses of FLAME's tracked expression state.

`flame_to_smplx_expression` maps FLAME's 50 tracked expression coefficients
onto SMPL-X's own `Exp000-099` basis via a fitted least-squares matrix, to
drive `output.blend`'s face, a coefficient-space approximation, not an
exact correspondence (see `_flame_to_smplx_expression_matrix`'s own
docstring for why, and its scope).

`direct_arkit_jaw_channels` (Group D) and `solve_arkit_weights` (a small
Group-S solve covering `NoseSneerLeft/Right`, `CheekSquintLeft/Right`,
`EyeSquintLeft/Right`) feed `output_face.csv` instead. Every other ARKit
channel comes from MediaPipe's own native blendshape output
(`stage_9_capture_face.py`'s channel merge); see `scripts/build_face_bases.py`
for why Group S is solved in landmark space, not mesh space, and why it's
scoped to just these six channels.

Pure numpy, deliberately: stage 9 (tracking, `main` env, torch) and stage 10
(export, `bpy`-only `export` env) both need this, and neither should import
the other's heavy stack just to convert an expression vector.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

BODY_MODELS_DIR = Path(__file__).resolve().parents[3] / "body_models"
FLAME_MODEL_PATH = BODY_MODELS_DIR / "flame" / "FLAME_NEUTRAL.npz"
SMPLX_MODEL_PATH = BODY_MODELS_DIR / "smplx" / "SMPLX_NEUTRAL.npz"
FLAME_VERTEX_IDS_PATH = BODY_MODELS_DIR / "correspondences" / "SMPL-X__FLAME_vertex_ids.npy"

# Matches smplx.body_models.{FLAME,SMPLX}.SHAPE_SPACE_DIM: both models' own
# `shapedirs` arrays pack 300 shape/identity components first, then
# expression components, expression starts at this column in both.
SHAPE_SPACE_DIM = 300
FLAME_EXPRESSION_DIM = 50  # matches face_landmark_fit.FLAME_NUM_EXPRESSION, what's actually tracked
SMPLX_EXPRESSION_DIM = 100  # matches smplx.body_models.SMPLX.EXPRESSION_SPACE_DIM, Exp000-099


@lru_cache(maxsize=1)
def _flame_to_smplx_expression_matrix() -> np.ndarray:
    """(100, 50) least-squares map, fit from the real FLAME and SMPL-X
    expression bases restricted to their shared vertex subset (via
    `SMPL-X__FLAME_vertex_ids.npy`, already validated elsewhere at
    vertex-position cosine similarity 0.9993). Cheap enough (well under a
    second) to compute once per process rather than shipping a separate
    offline-build step and a derived artifact that could go stale against
    the model files it was built from.
    """
    flame = np.load(FLAME_MODEL_PATH)
    smplx_model = np.load(SMPLX_MODEL_PATH, allow_pickle=True)
    flame_vertex_ids = np.load(FLAME_VERTEX_IDS_PATH)

    flame_expr = flame["shapedirs"][:, :, SHAPE_SPACE_DIM:SHAPE_SPACE_DIM + FLAME_EXPRESSION_DIM]
    smplx_expr = smplx_model["shapedirs"][:, :, SHAPE_SPACE_DIM:SHAPE_SPACE_DIM + SMPLX_EXPRESSION_DIM]
    smplx_expr_shared = smplx_expr[flame_vertex_ids]  # (5023, 3, 100), reindexed into FLAME's own vertex order

    a_flame = flame_expr.reshape(-1, FLAME_EXPRESSION_DIM)
    a_smplx = smplx_expr_shared.reshape(-1, SMPLX_EXPRESSION_DIM)
    # Solve a_smplx @ M ~= a_flame: for any FLAME coefficient vector psi,
    # a_smplx @ (M @ psi) is the closest reproduction of FLAME's own
    # a_flame @ psi displacement that SMPL-X's basis can represent.
    matrix, *_ = np.linalg.lstsq(a_smplx, a_flame, rcond=None)
    return matrix


def flame_to_smplx_expression(flame_expression: np.ndarray) -> np.ndarray:
    """`flame_expression`: (F, 50), as tracked by `face_landmark_fit.fit_clip`.
    Returns (F, 100) for direct assignment to `Exp000-099` shape-key values.
    """
    return flame_expression @ _flame_to_smplx_expression_matrix().T


# --- ARKit CSV export: jaw (Group D) + the 6-channel Group-S remnant ------

FACE_BASES_PATH = BODY_MODELS_DIR / "arkit" / "face_bases.npz"

# Matches face_landmark_fit.py's own JAW_AXIS0_BOUND_RAD/JAW_AXIS1_BOUND_RAD
# (not re-derived here, these are the real, cross-clip-validated bounds
# that module's own comments explain at length).
JAW_AXIS0_BOUND_RAD = 0.8  # opening/closing
JAW_AXIS1_BOUND_RAD = 0.08  # yaw (side-to-side)

# Jaw-open axis angle (radians) treated as ARKit's fully-open reference, for
# normalizing `direct_arkit_jaw_channels`'s `JawOpen` into [0, 1].
# Deliberately separate from `JAW_AXIS0_BOUND_RAD`: that constant is the
# fitter's own optimizer safety margin, not a claim about how far a real
# open mouth rotates, and using it as the reference undershoots real
# magnitude. Calibrated against paired ground truth instead. `JawLeft`/
# `JawRight` have no equivalent calibrated reference and still normalize
# against `JAW_AXIS1_BOUND_RAD` directly.
JAW_OPEN_REFERENCE_RAD = 0.25

# `ridge` keeps the over-complete Group-S basis from letting near-collinear
# columns co-activate on noise; `temporal_lambda` softly couples a frame's
# weights to the previous frame's, the same role DEFAULT_TEMPORAL_WEIGHT
# plays in face_landmark_fit.py.
DEFAULT_ARKIT_TEMPORAL_LAMBDA = 0.05
DEFAULT_ARKIT_RIDGE = 0.01

# Per-clip percentile calibration on `solve_arkit_weights`' own output,
# same shape of fix as `face_gaze._calibrate_signed_ratio`/`face_eyelid`'s
# blink calibration: the ridge solve's own structural shrinkage-toward-zero
# undershoots real magnitude even when correlation is already good.
# `MIN_CALIBRATION_SCALE` is a real-data-placed floor: a channel whose raw
# 95th-percentile weight never clears it is treated as inactive rather than
# calibrated (dividing by near-zero would amplify float noise into a false
# full-scale reading, the same failure `face_eyelid.MIN_CALIBRATION_RANGE`
# guards against).
GROUP_S_CALIBRATION_PERCENTILE = 95.0
MIN_CALIBRATION_SCALE = 0.035


@lru_cache(maxsize=1)
def _load_face_bases() -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = np.load(FACE_BASES_PATH)
    return data["D"].astype(np.float64), data["jaw_landmark_jacobian"].astype(np.float64), list(data["group_s_names"])


def direct_arkit_jaw_channels(jaw_pose: np.ndarray) -> dict[str, np.ndarray]:
    """`jaw_pose`: (F, 3) axis-angle, `face_landmark_fit`'s own convention
    (axis0=open/close, axis1=yaw/side-to-side). Returns ARKit's 3 jaw
    channels recoverable from a pure rotation DOF, `JawOpen` (axis0,
    clipped to its positive half: the jaw doesn't open backwards) and
    `JawLeft`/`JawRight` (axis1, sign-split into non-negative halves,
    ARKit's own convention for a bidirectional motion never simultaneously
    nonzero).

    Sign verified directly against FLAME's own forward kinematics, not
    assumed (this project's own established rigor: see
    `feedback_naming_and_rigor`, always verify against real data):
    perturbing axis1 positively shifts *both* mouth-corner landmarks
    together in the same direction (a rigid lower-face shift, as expected
    for a jaw-yaw rotation about the TMJ, not an asymmetric stretch), and
    FLAME's own +X axis is confirmed (via the right/left eye landmark
    split, which sit at -X/+X respectively) to be the subject's own left,
    so axis1 > 0 is `JawLeft`.

    `JawForward` isn't included: SMPL-X's jaw is a pure rotation joint with
    no translation DOF to represent it, always zero (Group W).
    """
    open_axis, yaw_axis = jaw_pose[:, 0], jaw_pose[:, 1]
    return {
        "JawOpen": np.clip(open_axis / JAW_OPEN_REFERENCE_RAD, 0.0, 1.0),
        "JawLeft": np.clip(yaw_axis / JAW_AXIS1_BOUND_RAD, 0.0, 1.0),
        "JawRight": np.clip(-yaw_axis / JAW_AXIS1_BOUND_RAD, 0.0, 1.0),
    }


def _calibrate_group_s_channel(weights: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    """Per-channel percentile stretch, see `MIN_CALIBRATION_SCALE`'s own
    comment for why the floor is where it is and what it deliberately
    leaves uncalibrated."""
    sample = weights[valid] if valid is not None else weights
    if sample.size == 0:
        return weights
    scale = np.percentile(sample, GROUP_S_CALIBRATION_PERCENTILE)
    if scale < MIN_CALIBRATION_SCALE:
        return weights
    return np.clip(weights / scale, 0.0, 1.0)


def solve_arkit_weights(
    landmark_delta: np.ndarray, jaw_pose: np.ndarray,
    temporal_lambda: float = DEFAULT_ARKIT_TEMPORAL_LAMBDA, ridge: float = DEFAULT_ARKIT_RIDGE,
    valid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """`landmark_delta`: (F, 51, 3), the tracked FLAME landmarks in their
    own local (no head rotation/translation) frame, minus that same clip's
    own canonical zero-pose/zero-expression neutral (see
    `face_landmark_fit.local_landmark_delta`, the intended source of this
    argument). `jaw_pose`: (F, 3), same convention as
    `direct_arkit_jaw_channels`. `valid`: optional (F,) bool, restricts which
    frames inform each channel's own percentile calibration (see
    `MIN_CALIBRATION_SCALE`'s own comment) to real fitted frames, not
    held/frozen ones. Returns this module's own (now 6-channel) Group-S
    subset, each (F,) in [0, 1], see this module's own docstring for why
    it's not the original 34.

    Ordering constraint, load-bearing: jaw's own predicted landmark
    contribution (`jaw_landmark_jacobian @ jaw_pose`) is subtracted from the
    observed delta *before* solving, or `JawOpen`'s own landmark motion gets
    double-counted into these channels (the same double-counting regression
    `test_face_blendshapes.py` checks for directly).
    """
    from scipy.optimize import lsq_linear

    D, jaw_jacobian, names = _load_face_bases()
    n_channels = D.shape[1]
    n_frames = landmark_delta.shape[0]

    flat = landmark_delta.reshape(n_frames, -1).astype(np.float64)
    residual = flat - jaw_pose.astype(np.float64) @ jaw_jacobian.T

    identity = np.eye(n_channels)
    weights = np.zeros((n_frames, n_channels), dtype=np.float64)
    prev = np.zeros(n_channels)
    for t in range(n_frames):
        A = np.vstack([D, np.sqrt(temporal_lambda) * identity, np.sqrt(ridge) * identity])
        b = np.concatenate([residual[t], np.sqrt(temporal_lambda) * prev, np.zeros(n_channels)])
        weights[t] = lsq_linear(A, b, bounds=(0.0, 1.0)).x
        prev = weights[t]

    return {
        name: _calibrate_group_s_channel(weights[:, i], valid).astype(np.float32) for i, name in enumerate(names)
    }
