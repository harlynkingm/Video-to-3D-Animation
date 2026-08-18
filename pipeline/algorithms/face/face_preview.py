"""Raw FLAME preview: writes the tracked FLAME mesh, betas (shared) plus
per-frame expression/jaw_pose/global_orient/transl, exactly as `fit_clip`
produced them, as a PC2 point cache plus a static template mesh, played
back in Blender via the native Mesh Cache modifier (no addon needed).

Deliberately *not* mapped through the SMPL-X expression-basis conversion and
*not* the SMPL-X export path: this exists to isolate whether a visual defect
originates in the fitting loop itself or downstream of it (the SMPL-X
mapping, `_keyframe_face_expression`). Added as a debugging necessity after
real-clip verification showed the tracked identity coming out grotesquely
distorted, a quality gap that needed isolating before it could be fixed.

PC2 format (`POINTCACHE2`, a long-standing Maya/Softimage/Blender interchange
format): a 32-byte header, then `numSamples * numPoints` (x, y, z) float32
triples in order. Blender's own Mesh Cache modifier reads it natively,
no addon, no Python-side keyframe bookkeeping needed. Its axis conversion is
left at identity and the change of basis is applied here instead, in numpy
(see `FLAME_TO_BLENDER_ROTATION`), so the coordinates written are exactly
the coordinates Blender shows.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

# Mesh Cache's own axis conversion is left at identity (confirmed empirically
# against a real Blender session: `up_axis="POS_Z"`/`forward_axis="POS_Y"`
# maps file (x, y, z) straight to Blender (x, y, z)), the FLAME -> Blender
# change of basis is applied in numpy instead, via FLAME_TO_BLENDER_ROTATION
# below, so the conversion is visible and testable here rather than split
# across two representations. The modifier itself is set in
# `stage_10_export.py` (bpy-only, can't live in this torch-importing module)
#, see that file's own `_FACE_PREVIEW_PC2_UP_AXIS`/`_FACE_PREVIEW_PC2_
# FORWARD_AXIS`, duplicated rather than imported for the same reason.

# The mesh is built in FLAME's own canonical space (+Y up, +Z out of the
# face), see `write_flame_preview` for why head pose is deliberately not
# baked in. Blender is +Z up; the face is pointed at Blender's +X (a display
# preference, Blender's default view axes make a face pointing +X read
# immediately as "facing right" without orbiting first) via
# (x, y, z) -> (z, x, y): FLAME's up becomes Blender's up, and FLAME's
# forward (+Z, out of the face) becomes Blender's +X.
#
# A bare axis-swap permutation can satisfy both a determinant check and an
# axis-mapping check while still landing the face upside down, so
# `test_flame_to_blender_puts_brows_above_mouth` also pins the result
# against a real anatomical invariant.
FLAME_TO_BLENDER_ROTATION = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])

# A plain .npz of vertices+faces rather than an .obj: Blender's own OBJ
# importer applies its *own* Y-up -> Z-up conversion (as a 90-degree object
# rotation, not baked into the vertex data), which compounds with the
# conversion already applied here and leaves the mesh mis-oriented in world
# space. numpy is available in the `export` env, so the template is read
# directly and the mesh built by hand, no importer axis semantics involved.
TEMPLATE_NPZ_FILENAME = "face_preview_template.npz"
ANIMATION_PC2_FILENAME = "face_preview_animation.pc2"
KEY_TEMPLATE_VERTICES = "vertices"
KEY_TEMPLATE_FACES = "faces"

# ARKit-52 CSV visualization: the *exact* weights `output_face.csv` gets,
# applied to a full-mesh ICT-FaceKit blendshape basis (built offline by
# `scripts/build_arkit_preview_shapes.py`) instead of FLAME's own tracked
# expression/jaw, lets the CSV's own accuracy be judged visually, not just
# numerically. Same PC2/Mesh-Cache-modifier mechanism as the FLAME preview
# above, just a different source mesh basis.
ARKIT_PREVIEW_SHAPES_PATH = Path(__file__).resolve().parents[3] / "body_models" / "arkit" / "face_preview_shapes.npz"
ARKIT_PREVIEW_TEMPLATE_FILENAME = "arkit_preview_template.npz"
ARKIT_PREVIEW_PC2_FILENAME = "arkit_preview_animation.pc2"

# FLAME's own eyeball geometry (5023-vertex native mesh), found empirically
# rather than from a masks file this project doesn't have: perturbing
# `leye_pose`/`reye_pose` from zero and checking which vertices actually
# move under LBS skinning gives exactly 546 vertices per eye, in two disjoint
# contiguous ranges. Real-footage review of `ARKit_face_preview.blend`
# flagged the eyeball as visually frozen/clipping through the lid on real
# footage, root cause: `build_arkit_preview_shapes.py`'s own nearest-vertex
# ICT-FaceKit registration only draws from `idx_to_fitting_verts`, which
# structurally excludes iris/sclera source points (see that script's own
# docstring), so these vertices land on whatever ICT *skin* point is
# nearest, never real gaze motion. Overridden below with FLAME's own
# working eyeball-rotation mechanism instead, driven by the same gaze angle
# already in the CSV, rather than trying to source gaze motion through the
# ICT delta basis at all.
LEFT_EYEBALL_VERTS = slice(3931, 4477)
RIGHT_EYEBALL_VERTS = slice(4477, 5023)

# Axis-angle component meaning for `leye_pose`/`reye_pose`, also found
# empirically (rotating each axis alone and checking which world direction
# the eyeball's own front-center vertex moved in): index 0 = pitch, index 1
# = yaw, index 2 = roll (always 0, `eye_euler_degrees` never predicts
# roll). Sign directly untested against real footage yet, if a rendered
# preview shows gaze reading backwards, flip here, not in `face_gaze.py`
# (whose own In/Out sign is already verified against real ground truth
# LiveLinkFace data, see that module's docstring).
_EYE_PITCH_AXIS, _EYE_YAW_AXIS = 0, 1

# Raw-vs-smoothed dense-landmark preview: two side-by-side wireframes of
# MediaPipe's own 478-point detection, upstream of DECA/MICA/FLAME entirely,
# so a bad clip's landmark noise is visible before any of that machinery
# runs, lets MediaPipe's raw signal quality be checked independent of
# anything this project's own fitting loop might add or fix.
LANDMARK_PREVIEW_RAW_TEMPLATE_FILENAME = "landmark_preview_raw_template.npz"
LANDMARK_PREVIEW_RAW_PC2_FILENAME = "landmark_preview_raw_animation.pc2"
LANDMARK_PREVIEW_SMOOTHED_TEMPLATE_FILENAME = "landmark_preview_smoothed_template.npz"
LANDMARK_PREVIEW_SMOOTHED_PC2_FILENAME = "landmark_preview_smoothed_animation.pc2"
# `KEY_TEMPLATE_FACES` holds EDGE pairs here, not triangles, MediaPipe's
# real 478-point face topology isn't available in this project's install,
# and the point cloud reads as a face well enough with a cheap k-nearest-
# neighbor wireframe (see `_landmark_knn_edges`). Purely cosmetic connectivity,
# not a claim about real anatomy.
LANDMARK_PREVIEW_KNN_NEIGHBORS = 4

# MediaPipe's landmarks are full-frame pixel x/y (order ~1000s) plus a much
# smaller relative z, rendered as-is those units are both far too large for
# Blender's default view framing and badly flattened in depth. LANDMARK_
# PREVIEW_Z_SCALE brings z to a comparable order of magnitude to x/y;
# LANDMARK_PREVIEW_XY_SCALE (interocular distance is typically 150-250px on
# a face-visible clip) brings a real face down to roughly 1-2 Blender units
# across, a comfortable size at Blender's default view distance.
LANDMARK_PREVIEW_Z_SCALE = 12.0
LANDMARK_PREVIEW_XY_SCALE = 1.0 / 400.0
LANDMARK_PREVIEW_SEPARATION = 2.0  # Blender units between the raw/smoothed pair


def flame_to_blender(verts: np.ndarray) -> np.ndarray:
    """`verts`: (..., 3) in FLAME's own canonical space. Returns the same
    points in Blender's world space, via `FLAME_TO_BLENDER_ROTATION`."""
    return verts @ FLAME_TO_BLENDER_ROTATION.T


def _write_pc2(out_path: Path, verts_per_frame: np.ndarray) -> None:
    """`verts_per_frame`: (F, V, 3), already in PC2 file-axis order."""
    num_samples, num_points, _ = verts_per_frame.shape
    with open(out_path, "wb") as f:
        f.write(struct.pack("<12s i i f f i", b"POINTCACHE2", 1, num_points, 0.0, 1.0, num_samples))
        f.write(verts_per_frame.astype(np.float32).tobytes())


def _write_template_npz(out_path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    """`verts`: (V, 3) already in Blender's own axes (see module docstring).
    `faces`: (F, 3) 0-indexed vertex indices, the same indexing
    `bpy.types.Mesh.from_pydata` expects."""
    np.savez(out_path, **{KEY_TEMPLATE_VERTICES: verts, KEY_TEMPLATE_FACES: faces})


def write_flame_preview(motion: dict, out_dir: Path, device: torch.device | None = None) -> dict[str, str]:
    """`motion`: `fit_clip`'s own output dict (KEY_BETAS/KEY_EXPRESSION/
    KEY_JAW_POSE/KEY_GLOBAL_ORIENT/KEY_TRANSL from `face_landmark_fit.py`).
    Writes `TEMPLATE_NPZ_FILENAME` and `ANIMATION_PC2_FILENAME` into
    `out_dir` and returns their paths, keyed for `stage_9_capture_face.py`'s
    own outputs dict.
    """
    import torch
    from pipeline.helpers.torch_helpers import sys_torch_device

    from .face_landmark_fit import KEY_BETAS, KEY_EXPRESSION, KEY_JAW_POSE, _build_flame_model

    device = device or sys_torch_device()
    n = motion[KEY_EXPRESSION].shape[0]
    model = _build_flame_model(device, batch_size=n)

    betas = torch.tensor(motion[KEY_BETAS], device=device, dtype=torch.float32).unsqueeze(0).expand(n, -1)
    expression = torch.tensor(motion[KEY_EXPRESSION], device=device, dtype=torch.float32)
    jaw_pose = torch.tensor(motion[KEY_JAW_POSE], device=device, dtype=torch.float32)
    # Head pose (`KEY_GLOBAL_ORIENT`/`KEY_TRANSL`) is deliberately NOT applied:
    # the fit solves those in camera space, so baking them in renders the head
    # at whatever angle and distance the real camera happened to have, for
    # a handheld clip that is an arbitrary, per-clip tilt the viewer then has
    # to rotate back out by hand before they can judge anything. Since this
    # preview exists to inspect identity and expression, the head is left
    # canonical: origin-centred, front-facing, stable across frames, so the
    # only thing that moves is the face itself.
    with torch.no_grad():
        out = model(
            betas=betas, expression=expression, global_orient=torch.zeros(n, 3, device=device),
            neck_pose=torch.zeros(n, 3, device=device), jaw_pose=jaw_pose,
            leye_pose=torch.zeros(n, 3, device=device), reye_pose=torch.zeros(n, 3, device=device),
            transl=torch.zeros(n, 3, device=device),
        )
    verts = out.vertices.cpu().numpy()  # (F, V, 3), FLAME canonical space
    faces = model.faces  # (F, 3), topology is frame-independent

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blender_verts = flame_to_blender(verts)

    template_path = out_dir / TEMPLATE_NPZ_FILENAME
    _write_template_npz(template_path, blender_verts[0], faces)

    pc2_path = out_dir / ANIMATION_PC2_FILENAME
    _write_pc2(pc2_path, blender_verts)

    return {"face_preview_template": str(template_path), "face_preview_pc2": str(pc2_path)}


def _landmark_knn_edges(points: np.ndarray, k: int = LANDMARK_PREVIEW_KNN_NEIGHBORS) -> np.ndarray:
    """Cosmetic-only wireframe: each point connects to its `k` nearest
    neighbors (deduplicated). Not real face topology, see this module's
    `LANDMARK_PREVIEW_KNN_NEIGHBORS` comment for why."""
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    _, neighbor_idx = tree.query(points, k=k)
    edges = {(min(i, int(j)), max(i, int(j))) for i, neighbors in enumerate(neighbor_idx) for j in neighbors[1:]}
    return np.array(sorted(edges), dtype=np.int64)


def write_landmark_preview(mp_landmarks: np.ndarray, mp_valid: np.ndarray, smoothing_window: int, out_dir: Path) -> dict[str, str]:
    """`mp_landmarks`: (F, 478, 3) full-frame pixel x/y + MediaPipe's own
    relative z, exactly as `FaceLandmarksAdapter.infer` produced them,
    upstream of DECA/MICA/FLAME entirely. `smoothing_window`: the same
    window (frames) `stage_9_capture_face.py` uses on the sparse 51-point
    set for the real fit (`FineTuningOptions.face_smoothing_window`), reused here so
    this preview shows the production smoothing setting, not a separate one.

    Writes two template+PC2 pairs (raw, smoothed) into `out_dir`, offset
    apart on X so both display side by side in the same scene, and returns
    their paths keyed for `stage_9_capture_face.py`'s own outputs dict.
    """
    from ..motion_smoothing import smooth_position_sequence

    display = mp_landmarks.astype(np.float32).copy()
    display[:, :, 1] *= -1  # image y-down -> Blender y-up
    display[:, :, 2] *= LANDMARK_PREVIEW_Z_SCALE
    center = display[mp_valid][0].mean(axis=0)
    display = (display - center) * LANDMARK_PREVIEW_XY_SCALE

    smoothed = smooth_position_sequence(display, window=smoothing_window, valid=mp_valid)
    edges = _landmark_knn_edges(display[mp_valid][0])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_shifted = display.copy()
    raw_shifted[:, :, 0] -= LANDMARK_PREVIEW_SEPARATION
    smoothed_shifted = smoothed.copy()
    smoothed_shifted[:, :, 0] += LANDMARK_PREVIEW_SEPARATION

    raw_template_path = out_dir / LANDMARK_PREVIEW_RAW_TEMPLATE_FILENAME
    raw_pc2_path = out_dir / LANDMARK_PREVIEW_RAW_PC2_FILENAME
    smoothed_template_path = out_dir / LANDMARK_PREVIEW_SMOOTHED_TEMPLATE_FILENAME
    smoothed_pc2_path = out_dir / LANDMARK_PREVIEW_SMOOTHED_PC2_FILENAME

    _write_template_npz(raw_template_path, raw_shifted[mp_valid][0], edges)
    _write_pc2(raw_pc2_path, raw_shifted)
    _write_template_npz(smoothed_template_path, smoothed_shifted[mp_valid][0], edges)
    _write_pc2(smoothed_pc2_path, smoothed_shifted)

    return {
        "landmark_preview_raw_template": str(raw_template_path),
        "landmark_preview_raw_pc2": str(raw_pc2_path),
        "landmark_preview_smoothed_template": str(smoothed_template_path),
        "landmark_preview_smoothed_pc2": str(smoothed_pc2_path),
    }


def _rotate_eyeball(neutral_eye_verts: np.ndarray, yaw_deg: np.ndarray, pitch_deg: np.ndarray) -> np.ndarray:
    """`neutral_eye_verts`: (V, 3), one eye's own neutral vertices. `yaw_deg`/
    `pitch_deg`: (F,). Returns (F, V, 3): each frame's eyeball rigidly
    rotated about its own neutral centroid, see `LEFT_EYEBALL_VERTS`'s own
    comment for why this replaces the ICT delta basis for these vertices."""
    from scipy.spatial.transform import Rotation

    center = neutral_eye_verts.mean(axis=0)
    local = neutral_eye_verts - center  # (V, 3)

    n = len(yaw_deg)
    axis_angle = np.zeros((n, 3), dtype=np.float32)
    axis_angle[:, _EYE_PITCH_AXIS] = np.radians(pitch_deg)
    axis_angle[:, _EYE_YAW_AXIS] = np.radians(yaw_deg)
    rotmat = Rotation.from_rotvec(axis_angle).as_matrix().astype(np.float32)  # (F, 3, 3)

    return center[None, None, :] + np.einsum("fij,vj->fvi", rotmat, local)


def write_arkit_preview(arkit_weights: np.ndarray, head_eye_euler: np.ndarray, out_dir: Path) -> dict[str, str]:
    """`arkit_weights`: (F, 52), in `livelink_csv.ARKIT_BLENDSHAPE_NAMES`
    order, the *exact* array `output_face.csv` is written from (same
    values, same column order). Deforms `scripts/build_arkit_preview_
    shapes.py`'s offline full-mesh ICT-FaceKit basis by those weights, per
    frame, and writes the same template+PC2 pair `write_flame_preview` does.
    A real 3D face performance driven only by the CSV's own numbers, for
    judging its accuracy visually rather than just per-channel correlation.
    `head_eye_euler`: (F, 9) in `livelink_csv.HEAD_EYE_COLUMN_NAMES` order,
    the same array the CSV's own eye-Euler columns come from, drives the
    eyeball vertex override below (see `LEFT_EYEBALL_VERTS`'s own comment).

    Absent when `body_models/arkit/face_preview_shapes.npz` hasn't been
    built (see that script's own docstring), returns `{}` rather than
    raising, matching this stage's existing tolerance for missing optional
    assets (`--render-face-preview` should degrade, not crash, if the
    preview basis was never generated locally).
    """
    if not ARKIT_PREVIEW_SHAPES_PATH.exists():
        return {}

    with np.load(ARKIT_PREVIEW_SHAPES_PATH) as data:
        D_mesh = data["D_mesh"]  # (5023, 3, 52)
        neutral = data["neutral_verts"]  # (5023, 3)
        faces = data["faces"]

    delta = np.einsum("fk,vdk->fvd", arkit_weights.astype(np.float32), D_mesh)
    verts = neutral[None] + delta  # (F, 5023, 3), FLAME canonical space

    verts[:, LEFT_EYEBALL_VERTS] = _rotate_eyeball(
        neutral[LEFT_EYEBALL_VERTS], head_eye_euler[:, 3], head_eye_euler[:, 4],
    )
    verts[:, RIGHT_EYEBALL_VERTS] = _rotate_eyeball(
        neutral[RIGHT_EYEBALL_VERTS], head_eye_euler[:, 6], head_eye_euler[:, 7],
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blender_verts = flame_to_blender(verts)

    template_path = out_dir / ARKIT_PREVIEW_TEMPLATE_FILENAME
    _write_template_npz(template_path, blender_verts[0], faces)
    pc2_path = out_dir / ARKIT_PREVIEW_PC2_FILENAME
    _write_pc2(pc2_path, blender_verts)

    return {"arkit_preview_template": str(template_path), "arkit_preview_pc2": str(pc2_path)}
