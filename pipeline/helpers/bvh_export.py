"""Write a skeleton + animation to BVH (Biovision Hierarchy), which Blender
imports natively (File > Import > Motion Capture (.bvh)) as an armature with
keyframed bone rotations. Used for the stage 4 hands-only preview: bones, not a
mesh, so it needs no MANO mesh (and thus no chumpy).

Generic writer: give it a joint hierarchy (names, parents, rest offsets) and
per-frame local rotation matrices, and it emits the HIERARCHY + MOTION blocks.
Rotations are written as ZXY-order Euler angles, with matching `CHANNELS`
declarations, so a reader composes them back to the same matrix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_ROT_CHANNELS = "Zrotation Xrotation Yrotation"
_EULER_ORDER = "ZXY"  # matches _ROT_CHANNELS; scipy intrinsic convention


def _children_of(parents: list[int]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {i: [] for i in range(len(parents))}
    for j, p in enumerate(parents):
        if p >= 0:
            children[p].append(j)
    return children


def _write_hierarchy(
    lines: list[str], order: list[int], joint: int, names: list[str], offsets: np.ndarray,
    children: dict[int, list[int]], depth: int, is_root: bool,
) -> None:
    pad = "  " * depth
    ox, oy, oz = offsets[joint]
    if is_root:
        lines.append(f"ROOT {names[joint]}")
        lines.append("{")
        lines.append(f"  OFFSET {ox:.6f} {oy:.6f} {oz:.6f}")
        lines.append(f"  CHANNELS 6 Xposition Yposition Zposition {_ROT_CHANNELS}")
    else:
        lines.append(f"{pad}JOINT {names[joint]}")
        lines.append(f"{pad}{{")
        lines.append(f"{pad}  OFFSET {ox:.6f} {oy:.6f} {oz:.6f}")
        lines.append(f"{pad}  CHANNELS 3 {_ROT_CHANNELS}")
    order.append(joint)

    kids = children[joint]
    if kids:
        for k in kids:
            _write_hierarchy(lines, order, k, names, offsets, children, depth + 1, is_root=False)
    else:
        # End Site gives the leaf bone a length; continue the last bone direction.
        tip = offsets[joint]
        lines.append(f"{pad}  End Site")
        lines.append(f"{pad}  {{")
        lines.append(f"{pad}    OFFSET {tip[0]:.6f} {tip[1]:.6f} {tip[2]:.6f}")
        lines.append(f"{pad}  }}")
    lines.append(("  " * depth) + "}")


def write_bvh(
    path: Path, joint_names: list[str], parents: list[int], offsets: np.ndarray,
    rotations: np.ndarray, fps: float,
) -> None:
    """rotations: (F, J, 3, 3) local rotation matrices, one per joint per frame.
    `parents[root] == -1` (exactly one root, listed before its children)."""
    root = parents.index(-1)
    children = _children_of(parents)

    lines: list[str] = ["HIERARCHY"]
    order: list[int] = []
    _write_hierarchy(lines, order, root, joint_names, offsets, children, depth=0, is_root=True)

    n_frames = rotations.shape[0]
    lines.append("MOTION")
    lines.append(f"Frames: {n_frames}")
    lines.append(f"Frame Time: {1.0 / fps:.6f}")

    euler = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_euler(_EULER_ORDER, degrees=True)
    euler = euler.reshape(n_frames, len(joint_names), 3)  # (F, J, [z, x, y])
    for f in range(n_frames):
        values: list[str] = []
        for j in order:
            if j == root:
                values += ["0.000000", "0.000000", "0.000000"]  # static root position
            z, x, y = euler[f, j]
            values += [f"{z:.6f}", f"{x:.6f}", f"{y:.6f}"]
        lines.append(" ".join(values))

    Path(path).write_text("\n".join(lines) + "\n")
