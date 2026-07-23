"""Thin load()/infer()/unload() wrapper producing per-frame MANO hand pose from
HaMeR, for both hands. Clean-room port (the reference is a research repo, MIT
licensed) that reuses this project's existing pieces heavily:

  - the shared ViT-H backbone (`VitHugeBackbone`, same weights family, loaded
    here from `hamer.safetensors`),
  - our COCO-17 ViTPose (`gvhmr_vitpose`) for the wrist/elbow keypoints that
    locate each hand -- HaMeR's demo uses a whole-body ViTPose for tight hand
    boxes, which this project doesn't have; see `hamer_preprocess`,
  - the SAM 3.1 human mask (stage 1) for the person box, rescaled from SAM's
    1008x1008 working resolution to native like `gvhmr_adapter` does,
  - the rotation math (`matrix_to_axis_angle`).

The head outputs rotation matrices in the hand crop's camera frame; this adapter
converts them to axis-angle and, for left hands (which are flipped to look like
right hands before inference), applies HaMeR's `fliplr` correction (negate the
y/z axis-angle components). Reconciling the wrist orientation with GVHMR's
forearm is deliberately left to stage 5 (retarget_hands) -- this stage only
produces the raw per-hand MANO pose.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors import safe_open

from ...helpers.vit_huge_backbone import VitHugeBackbone
from ..gvhmr.gvhmr_adapter import (
    VITPOSE_CHECKPOINT,
    _load_direct_state,
    _rescale_bbox_xywh,
    extract_bbox_from_numpy_mask,
)
from ..gvhmr.gvhmr_rotation_math import matrix_to_axis_angle
from ..gvhmr.gvhmr_vitpose import GVHMRViTPoseModel, estimate_keypoints
from ..sam31.sam31_tracker import KEY_PACKED_MASKS, unpack_masks
from .hamer_mano_head import MANOTransformerDecoderHead
from .hamer_preprocess import crop_hand, hand_box_from_body_kpts

CHECKPOINT_DIR = Path(__file__).resolve().parents[3] / "checkpoints"
HAMER_CHECKPOINT = CHECKPOINT_DIR / "hamer.safetensors"

_BACKBONE_PREFIX = "backbone."
_MANO_HEAD_PREFIX = "mano_head."
# The backbone is trained on 256x192; the crop is 256x256, so trim 32px off each
# side of the width -- matching HaMeR's `self.backbone(x[:,:,:,32:-32])`.
_WIDTH_CROP = 32

HAND_POSE_DIM = 45  # 15 MANO joints x 3 axis-angle

# infer() output keys (per-frame arrays).
KEY_LEFT_HAND_POSE = "left_hand_pose"
KEY_RIGHT_HAND_POSE = "right_hand_pose"
KEY_LEFT_GLOBAL_ORIENT = "left_global_orient"
KEY_RIGHT_GLOBAL_ORIENT = "right_global_orient"
KEY_LEFT_VALID = "left_valid"
KEY_RIGHT_VALID = "right_valid"


def _fliplr_axis_angle(aa: np.ndarray) -> np.ndarray:
    """Mirror an axis-angle pose (..., 3*K) for a horizontal image flip: negate
    the y and z components of every 3-vector (HaMeR's `fliplr_params`)."""
    flipped = aa.reshape(-1, 3).copy()
    flipped[:, 1] *= -1
    flipped[:, 2] *= -1
    return flipped.reshape(aa.shape)


class HamerAdapter:
    def __init__(self, device: torch.device | None = None, dtype: torch.dtype = torch.float16):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._backbone: VitHugeBackbone | None = None
        self._head: MANOTransformerDecoderHead | None = None
        self._vitpose: GVHMRViTPoseModel | None = None

    def load(self, hamer_checkpoint: Path = HAMER_CHECKPOINT, vitpose_checkpoint: Path = VITPOSE_CHECKPOINT) -> None:
        backbone_sd, head_sd = {}, {}
        with safe_open(str(hamer_checkpoint), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(_BACKBONE_PREFIX):
                    backbone_sd[key[len(_BACKBONE_PREFIX):]] = f.get_tensor(key)
                elif key.startswith(_MANO_HEAD_PREFIX):
                    head_sd[key[len(_MANO_HEAD_PREFIX):]] = f.get_tensor(key)

        self._backbone = VitHugeBackbone()
        self._backbone.load_state_dict(backbone_sd, strict=True)
        self._head = MANOTransformerDecoderHead()
        self._head.load_state_dict(head_sd, strict=True)
        self._vitpose = GVHMRViTPoseModel()
        self._vitpose.load_state_dict(_load_direct_state(vitpose_checkpoint), strict=True)

        for module in (self._backbone, self._head, self._vitpose):
            module.to(device=self.device, dtype=self.dtype).eval()

    @torch.inference_mode()
    def _infer_one_hand(self, frame_bgr: np.ndarray, keypoints: np.ndarray, is_right: bool):
        box = hand_box_from_body_kpts(keypoints, is_right)
        if box is None:
            return None
        crop = crop_hand(frame_bgr, box, is_right)
        crop_t = torch.from_numpy(crop).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        feats = self._backbone(crop_t[:, :, :, _WIDTH_CROP:-_WIDTH_CROP])
        out = self._head(feats)

        global_orient = matrix_to_axis_angle(out["global_orient"].float().reshape(1, 3, 3)).reshape(3).cpu().numpy()
        hand_pose = matrix_to_axis_angle(out["hand_pose"].float().reshape(-1, 3, 3)).reshape(-1).cpu().numpy()
        if not is_right:
            global_orient = _fliplr_axis_angle(global_orient)
            hand_pose = _fliplr_axis_angle(hand_pose)
        return global_orient.astype(np.float32), hand_pose.astype(np.float32)

    def infer(self, frame_paths: list[Path], human_masks: dict) -> dict[str, np.ndarray]:
        packed = human_masks[KEY_PACKED_MASKS]
        n = len(frame_paths)
        out = {
            KEY_LEFT_HAND_POSE: np.zeros((n, HAND_POSE_DIM), np.float32),
            KEY_RIGHT_HAND_POSE: np.zeros((n, HAND_POSE_DIM), np.float32),
            KEY_LEFT_GLOBAL_ORIENT: np.zeros((n, 3), np.float32),
            KEY_RIGHT_GLOBAL_ORIENT: np.zeros((n, 3), np.float32),
            KEY_LEFT_VALID: np.zeros(n, bool),
            KEY_RIGHT_VALID: np.zeros(n, bool),
        }

        for i, frame_path in enumerate(frame_paths):
            frame_bgr = cv2.imread(str(frame_path))
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            mask = unpack_masks(packed[i])[0].numpy()
            bbox = extract_bbox_from_numpy_mask(mask)
            if bbox is None:
                continue
            bbox = _rescale_bbox_xywh(bbox, from_hw=mask.shape, to_hw=frame_bgr.shape[:2])
            keypoints = estimate_keypoints(self._vitpose, frame_rgb, bbox, self.device, self.dtype)

            for is_right, pose_key, go_key, valid_key in (
                (True, KEY_RIGHT_HAND_POSE, KEY_RIGHT_GLOBAL_ORIENT, KEY_RIGHT_VALID),
                (False, KEY_LEFT_HAND_POSE, KEY_LEFT_GLOBAL_ORIENT, KEY_LEFT_VALID),
            ):
                result = self._infer_one_hand(frame_bgr, keypoints, is_right)
                if result is not None:
                    out[go_key][i], out[pose_key][i] = result
                    out[valid_key][i] = True

        return out

    def unload(self) -> None:
        del self._backbone, self._head, self._vitpose
        self._backbone = self._head = self._vitpose = None
        torch.cuda.empty_cache()
