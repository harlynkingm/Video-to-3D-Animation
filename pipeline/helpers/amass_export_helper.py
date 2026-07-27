"""Writes a standard AMASS-format `.npz` -- the exact input format the
already-installed `jtesch/smplx_blender_addon`'s "Add Animation" operator
expects, whether the caller is a preview (stage 2, body only) or a real
export (stage 9, body + hands).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# SMPL-X joint layout the AMASS `poses` array must match (see the addon's own
# `utils/model_spec.py`): pelvis + 21 body joints, jaw, 2 eyes, 15+15 hand
# joints, 3 axis-angle values each -- 55 * 3 = 165.
JAW_EYES_DIM = 9  # jaw (3) + 2 eyes (3 each) -- never estimated by this pipeline
HAND_POSE_DIM = 45  # 15 joints * 3 axis-angle values, per hand


def write_amass_npz(
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    betas: np.ndarray,
    transl: np.ndarray,
    fps: float,
    out_path: Path,
    left_hand_pose: np.ndarray | None = None,
    right_hand_pose: np.ndarray | None = None,
) -> None:
    """`global_orient`: (F, 3), `body_pose`: (F, 63), `betas`: (10,) (pooled
    across the clip, identical every frame), `transl`: (F, 3). Hand poses
    default to zero (flat/neutral) for a caller with no real hand estimate
    yet; jaw/eyes are always zero -- this pipeline never estimates facial
    pose. Gender is always "neutral", matching the one SMPL-X body model
    (`SMPLX_NEUTRAL.npz`) used throughout this project.
    """
    num_frames = body_pose.shape[0]
    if left_hand_pose is None:
        left_hand_pose = np.zeros((num_frames, HAND_POSE_DIM), dtype=np.float32)
    if right_hand_pose is None:
        right_hand_pose = np.zeros((num_frames, HAND_POSE_DIM), dtype=np.float32)
    jaw_and_eyes = np.zeros((num_frames, JAW_EYES_DIM), dtype=np.float32)

    poses = np.concatenate(
        [global_orient, body_pose, jaw_and_eyes, left_hand_pose, right_hand_pose], axis=-1
    ).astype(np.float32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        trans=np.asarray(transl, dtype=np.float32),
        gender=np.array("neutral"),
        mocap_frame_rate=np.array(round(fps)),
        betas=np.asarray(betas, dtype=np.float32),
        poses=poses,
    )
