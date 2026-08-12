"""Keyframes the mesh's own `Exp000-099` shape keys from the mapped SMPL-X
expression basis (see `algorithms.face.face_blendshapes.
flame_to_smplx_expression`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .blender_constants import _FIRST_MOTION_BLENDER_FRAME
from .blender_scene import _iter_action_fcurves

if TYPE_CHECKING:
    import bpy

# Mirrors the addon's own utils/shapekeys.py SHAPEKEY_VALUE_RANGE, not
# imported (addon-internal, not a public API), just the same value: default
# shape-key slider ranges are too narrow for a real expression coefficient,
# and ShapeKey.value is hard-clamped to its own slider_min/slider_max.
_EXPRESSION_SHAPEKEY_RANGE = 10.0


def _keyframe_face_expression(body_mesh: bpy.types.Object, expression: np.ndarray) -> None:
    """Keyframes the `Exp000-099` shape keys from `expression` (F, 100,
    SMPL-X's own basis, see `face_blendshapes.flame_to_smplx_expression`).
    One frame index per real motion frame, offset by
    `_FIRST_MOTION_BLENDER_FRAME` like every other per-frame keyframe in
    this package.

    A component that's zero on every frame is skipped entirely. A moving
    component gets a neutral (0.0) keyframe at the prepended rest-pose frame
    (frame 1, the same jump-not-blend treatment the body pose gets across
    that boundary), then one keyframe per frame where its value actually
    changes from both neighbors, skipping the redundant middle of a flat
    stretch like `blender_object_attachment._keyframe_held_object_pose`
    does for the object.

    Default shape-key slider ranges are too narrow for a real expression
    coefficient, so every touched key's range is widened first.
    """
    key_blocks = body_mesh.data.shape_keys.key_blocks
    n_frames, n_components = expression.shape

    for component in range(n_components):
        values = expression[:, component]
        if np.allclose(values, 0.0):
            continue

        key_block = key_blocks[f"Exp{component:03d}"]
        key_block.slider_min = -_EXPRESSION_SHAPEKEY_RANGE
        key_block.slider_max = _EXPRESSION_SHAPEKEY_RANGE

        key_block.value = 0.0
        key_block.keyframe_insert(data_path="value", frame=_FIRST_MOTION_BLENDER_FRAME - 1)

        prev_value = 0.0
        for i in range(n_frames):
            value = float(values[i])
            prev_same = np.isclose(value, prev_value)
            next_same = i < n_frames - 1 and np.isclose(value, values[i + 1])
            if prev_same and next_same:
                prev_value = value
                continue  # strictly redundant, both neighbors already pin this exact value

            key_block.value = value
            key_block.keyframe_insert(data_path="value", frame=i + _FIRST_MOTION_BLENDER_FRAME)
            prev_value = value

    shape_keys = body_mesh.data.shape_keys
    if shape_keys.animation_data and shape_keys.animation_data.action:
        for fcurve in _iter_action_fcurves(shape_keys.animation_data.action):
            for keyframe_point in fcurve.keyframe_points:
                keyframe_point.interpolation = "LINEAR"
