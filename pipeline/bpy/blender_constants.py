"""Constants shared across more than one `pipeline/bpy/` module, kept here
instead of duplicated locally in each.
"""

from __future__ import annotations

# Stage 8's object_pose.npz index 0 (the source video's own first frame)
# lands on this Blender frame, one later than the index alone would suggest,
# Blender frame 1 is always the body's own prepended rest-pose frame
# (`stage_10_export._prepend_rest_pose_frame`), which has no equivalent
# concept for the object/face/preview data these frames offset against.
_FIRST_MOTION_BLENDER_FRAME = 2
