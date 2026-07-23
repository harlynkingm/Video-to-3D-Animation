"""Build a two-hands BVH skeleton from HaMeR's per-frame MANO pose, for the
optional stage 4 preview. Uses the hand joints' rest offsets and hierarchy from
SMPL-X's own model file (`SMPLX_NEUTRAL.npz`, which loads with plain numpy) --
so no MANO mesh, and no chumpy. HaMeR's MANO finger order already matches
SMPL-X's hand-joint order, so the predicted `hand_pose` drives the SMPL-X
fingers directly.

Each hand is re-parented under a synthetic root and offset sideways so the two
hands sit apart. Wrist orientation is the predicted `global_orient` (in the
hand crop's camera frame -- this previews the raw stage 4 output, before any
body reconciliation, which is stage 5's job). Frames where a hand wasn't
detected fall back to the rest pose (identity).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ...helpers.bvh_export import write_bvh
from .hamer_adapter import (
    KEY_LEFT_GLOBAL_ORIENT,
    KEY_LEFT_HAND_POSE,
    KEY_LEFT_VALID,
    KEY_RIGHT_GLOBAL_ORIENT,
    KEY_RIGHT_HAND_POSE,
    KEY_RIGHT_VALID,
)

# Repo root is 3 levels up (hamer/ -> adapters/ -> pipeline/ -> root).
SMPLX_MODEL_PATH = Path(__file__).resolve().parents[3] / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"

# SMPL-X joint layout: wrists 20/21, left fingers 25-39, right fingers 40-54.
LEFT_WRIST, RIGHT_WRIST = 20, 21
LEFT_FINGERS = list(range(25, 40))
RIGHT_FINGERS = list(range(40, 55))
# MANO/SMPL-X hand-joint order (matches HaMeR's 15-joint hand_pose).
FINGER_NAMES = [
    "Index1", "Index2", "Index3", "Middle1", "Middle2", "Middle3", "Pinky1", "Pinky2", "Pinky3",
    "Ring1", "Ring2", "Ring3", "Thumb1", "Thumb2", "Thumb3",
]
HAND_SEPARATION = 0.30  # meters each hand sits from the synthetic root (spread apart for a readable preview)


def _smplx_rest_joints_and_parents() -> tuple[np.ndarray, np.ndarray]:
    data = np.load(SMPLX_MODEL_PATH, allow_pickle=True)
    rest_joints = np.asarray(data["J_regressor"]) @ np.asarray(data["v_template"])  # (55, 3)
    parents = np.asarray(data["kintree_table"][0]).astype(int)
    parents[0] = -1
    return rest_joints, parents


def _build_skeleton():
    """Returns (names, parents, offsets, rot_sources) for the two-hand rig.
    rot_sources[j] says where joint j's per-frame rotation comes from."""
    rest, smplx_parents = _smplx_rest_joints_and_parents()

    names: list[str] = ["Root"]
    parents: list[int] = [-1]
    offsets: list[np.ndarray] = [np.zeros(3)]
    rot_sources: list = ["root"]

    def add_hand(side: str, wrist_idx: int, finger_idxs: list[int], wrist_offset: np.ndarray) -> None:
        smplx_to_out = {wrist_idx: len(names)}
        names.append(f"{side}Wrist")
        parents.append(0)  # under the synthetic root
        offsets.append(wrist_offset)
        rot_sources.append(f"{side.lower()}_wrist")
        for k, smplx_j in enumerate(finger_idxs):
            smplx_to_out[smplx_j] = len(names)
            names.append(f"{side}{FINGER_NAMES[k]}")
            parents.append(smplx_to_out[smplx_parents[smplx_j]])
            offsets.append(rest[smplx_j] - rest[smplx_parents[smplx_j]])
            rot_sources.append((f"{side.lower()}_finger", k))

    add_hand("Left", LEFT_WRIST, LEFT_FINGERS, np.array([-HAND_SEPARATION, 0.0, 0.0]))
    add_hand("Right", RIGHT_WRIST, RIGHT_FINGERS, np.array([HAND_SEPARATION, 0.0, 0.0]))
    return names, parents, np.stack(offsets), rot_sources


def dump_hands_bvh(hand_data: dict, fps: float, out_path: Path) -> None:
    names, parents, offsets, rot_sources = _build_skeleton()
    n_frames = len(hand_data[KEY_LEFT_VALID])
    n_joints = len(names)

    pose = {
        "left_wrist": hand_data[KEY_LEFT_GLOBAL_ORIENT],
        "right_wrist": hand_data[KEY_RIGHT_GLOBAL_ORIENT],
        "left_finger": hand_data[KEY_LEFT_HAND_POSE].reshape(n_frames, 15, 3),
        "right_finger": hand_data[KEY_RIGHT_HAND_POSE].reshape(n_frames, 15, 3),
    }
    valid = {"left": hand_data[KEY_LEFT_VALID], "right": hand_data[KEY_RIGHT_VALID]}

    rotations = np.tile(np.eye(3), (n_frames, n_joints, 1, 1))
    for f in range(n_frames):
        for j, src in enumerate(rot_sources):
            if src == "root":
                continue
            side = src[0].split("_")[0] if isinstance(src, tuple) else src.split("_")[0]
            if not valid[side][f]:
                continue
            aa = pose[src][f] if isinstance(src, str) else pose[f"{side}_finger"][f, src[1]]
            rotations[f, j] = Rotation.from_rotvec(aa).as_matrix()

    write_bvh(out_path, names, parents, offsets, rotations, fps)
