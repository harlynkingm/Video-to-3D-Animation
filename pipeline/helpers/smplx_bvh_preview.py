"""Full-body-plus-hands BVH skeleton for the optional stage 5 preview: the whole
SMPL-X body with the reconciled hands attached at the wrists, animated, so the
wrist/finger grafting can be eyeballed in Blender (File > Import > Motion Capture
.bvh). The point of the preview is to confirm the hands sit correctly on the arms
-- if a palm comes out visibly rolled, HaMeR's wrist frame needs an alignment
correction before it matches GVHMR's.

Reuses `write_bvh` and SMPL-X's own rest skeleton (which loads with plain numpy,
no chumpy). Body joints are driven by the merged `global_orient`/`body_pose`,
fingers by `left_hand_pose`/`right_hand_pose`. The root stays at the origin --
orientation is what this previews, so the body's global translation is omitted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .bvh_export import write_bvh

# Repo root is 2 levels up (helpers/ -> pipeline/ -> root).
SMPLX_MODEL_PATH = Path(__file__).resolve().parents[2] / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"

# The 22 body joints (root + 21 body-pose joints), in SMPL-X order.
BODY_JOINT_NAMES = [
    "Pelvis", "LeftHip", "RightHip", "Spine1", "LeftKnee", "RightKnee", "Spine2",
    "LeftAnkle", "RightAnkle", "Spine3", "LeftFoot", "RightFoot", "Neck",
    "LeftCollar", "RightCollar", "Head", "LeftShoulder", "RightShoulder",
    "LeftElbow", "RightElbow", "LeftWrist", "RightWrist",
]
# MANO/SMPL-X hand-joint order (matches HaMeR's 15-joint hand_pose).
FINGER_NAMES = [
    "Index1", "Index2", "Index3", "Middle1", "Middle2", "Middle3", "Pinky1", "Pinky2", "Pinky3",
    "Ring1", "Ring2", "Ring3", "Thumb1", "Thumb2", "Thumb3",
]

# SMPL-X joint layout: 22 body joints, then left fingers 25-39, right fingers
# 40-54 (jaw/eyes at 22-24 carry no pose here and are skipped).
LEFT_FINGERS = range(25, 40)
RIGHT_FINGERS = range(40, 55)
INCLUDED_JOINTS = list(range(22)) + list(LEFT_FINGERS) + list(RIGHT_FINGERS)


def _smplx_rest_joints_and_parents() -> tuple[np.ndarray, np.ndarray]:
    data = np.load(SMPLX_MODEL_PATH, allow_pickle=True)
    rest_joints = np.asarray(data["J_regressor"]) @ np.asarray(data["v_template"])  # (55, 3)
    parents = np.asarray(data["kintree_table"][0]).astype(int)
    parents[0] = -1
    return rest_joints, parents


def _joint_name(smplx_idx: int) -> str:
    if smplx_idx < 22:
        return BODY_JOINT_NAMES[smplx_idx]
    if smplx_idx < 40:
        return "Left" + FINGER_NAMES[smplx_idx - 25]
    return "Right" + FINGER_NAMES[smplx_idx - 40]


def _np(x: np.ndarray | torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def dump_body_hands_bvh(
    global_orient: np.ndarray | torch.Tensor,
    body_pose: np.ndarray | torch.Tensor,
    left_hand_pose: np.ndarray | torch.Tensor,
    right_hand_pose: np.ndarray | torch.Tensor,
    fps: float,
    out_path: Path,
) -> None:
    """Write a full-body-plus-hands BVH. All pose args are per-frame axis-angle:
    global_orient (F, 3), body_pose (F, 63), left/right_hand_pose (F, 45)."""
    global_orient, body_pose = _np(global_orient), _np(body_pose)
    left_hand_pose, right_hand_pose = _np(left_hand_pose), _np(right_hand_pose)
    n_frames = global_orient.shape[0]

    rest, smplx_parents = _smplx_rest_joints_and_parents()
    remap = {smplx_j: out_i for out_i, smplx_j in enumerate(INCLUDED_JOINTS)}
    names = [_joint_name(j) for j in INCLUDED_JOINTS]
    parents = [-1 if smplx_parents[j] == -1 else remap[smplx_parents[j]] for j in INCLUDED_JOINTS]
    offsets = np.stack(
        [np.zeros(3) if smplx_parents[j] == -1 else rest[j] - rest[smplx_parents[j]] for j in INCLUDED_JOINTS]
    )

    def axis_angle_for(smplx_idx: int, frame: int) -> np.ndarray:
        if smplx_idx == 0:
            return global_orient[frame]
        if smplx_idx < 22:
            base = (smplx_idx - 1) * 3
            return body_pose[frame, base : base + 3]
        if smplx_idx < 40:
            base = (smplx_idx - 25) * 3
            return left_hand_pose[frame, base : base + 3]
        base = (smplx_idx - 40) * 3
        return right_hand_pose[frame, base : base + 3]

    rotations = np.tile(np.eye(3), (n_frames, len(INCLUDED_JOINTS), 1, 1))
    for frame in range(n_frames):
        for out_i, smplx_idx in enumerate(INCLUDED_JOINTS):
            rotations[frame, out_i] = Rotation.from_rotvec(axis_angle_for(smplx_idx, frame)).as_matrix()

    write_bvh(out_path, names, parents, offsets, rotations, fps)
