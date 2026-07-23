"""Camera/bbox conditioning math the GVHMR transformer and its I/O need:
turning a per-frame bbox + camera intrinsics into the `f_cliffcam` conditioning
signal, turning the network's predicted (scale, tx, ty) camera params into an
actual 3D translation, and normalizing raw 2D keypoints into bbox-relative
coordinates for the network's `obs` input.

Ported from `comfyui-motioncapture/nodes/motion_utils/hmr_cam.py`, restricted
to the three functions this project's call path actually reaches (confirmed
against `gvhmr/model.py`'s `Pipeline.forward` and `DemoPL.predict`): the
source's `estimate_K`/`create_camera_sensor` fallbacks are for when no real
lens info is available, which never applies here (this project always builds
K from the user's real focal length + sensor width, see
`camera_info_helpers.py`); `convert_xys_to_cliff_cam_wham` is an alternate,
unused cliffcam formula (BEDLAM's is the one GVHMR's own pipeline calls);
`get_a_pred_cam`/`project_to_bi01`/`perspective_projection` are training-only,
never called during inference.
"""

from __future__ import annotations

import torch


def compute_bbox_info_bedlam(bbx_xys: torch.Tensor, K_fullimg: torch.Tensor) -> torch.Tensor:
    """(B, L, 3) bbox [center_x, center_y, size] + (B, L, 3, 3) intrinsics ->
    (B, L, 3) `f_cliffcam`: bbox center offset from the principal point and
    bbox size, both scaled by focal length so the network sees a
    resolution/focal-length-independent signal (BEDLAM's formulation)."""
    fl = K_fullimg[..., 0, 0].unsqueeze(-1)
    icx = K_fullimg[..., 0, 2]
    icy = K_fullimg[..., 1, 2]
    cx, cy, size = bbx_xys[..., 0], bbx_xys[..., 1], bbx_xys[..., 2]
    bbox_info = torch.stack([cx - icx, cy - icy, size], dim=-1)
    return bbox_info / fl


def compute_transl_full_cam(pred_cam: torch.Tensor, bbx_xys: torch.Tensor, K_fullimg: torch.Tensor) -> torch.Tensor:
    """The network predicts a crop-relative (scale, tx, ty); this recovers the
    actual full-image camera-space 3D translation from it, using the same bbox
    and intrinsics the crop was taken with."""
    s, tx, ty = pred_cam[..., 0], pred_cam[..., 1], pred_cam[..., 2]
    focal_length = K_fullimg[..., 0, 0]
    icx = K_fullimg[..., 0, 2]
    icy = K_fullimg[..., 1, 2]
    sb = s * bbx_xys[..., 2]
    cx = 2 * (bbx_xys[..., 0] - icx) / (sb + 1e-9)
    cy = 2 * (bbx_xys[..., 1] - icy) / (sb + 1e-9)
    tz = 2 * focal_length / (sb + 1e-9)
    return torch.stack([tx + cx, ty + cy, tz], dim=-1)


def normalize_kp2d(obs_kp2d: torch.Tensor, bbx_xys: torch.Tensor, clamp_scale_min: bool = False) -> torch.Tensor:
    """(B, L, J, 3) [x, y, confidence] image-space keypoints + (B, L, 3) bbox ->
    (B, L, J, 3) bbox-relative keypoints in roughly [-1, 1], with confidence
    zeroed for any keypoint that falls outside the bbox."""
    obs_xy = obs_kp2d[..., :2]
    obs_conf = obs_kp2d[..., 2]
    center = bbx_xys[..., :2]
    scale = bbx_xys[..., [2]]

    xy_max = center + scale / 2
    xy_min = center - scale / 2
    invisible = (
        (obs_xy[..., 0] < xy_min[..., None, 0]) | (obs_xy[..., 0] > xy_max[..., None, 0])
        | (obs_xy[..., 1] < xy_min[..., None, 1]) | (obs_xy[..., 1] > xy_max[..., None, 1])
    )
    obs_conf = obs_conf * ~invisible
    if clamp_scale_min:
        scale = scale.clamp(min=1e-5)
    normalized_xy = 2 * (obs_xy - center.unsqueeze(-2)) / scale.unsqueeze(-2)
    return torch.cat([normalized_xy, obs_conf[..., None]], dim=-1)
