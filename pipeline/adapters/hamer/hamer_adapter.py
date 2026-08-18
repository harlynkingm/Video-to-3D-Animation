"""Thin load()/infer()/unload() wrapper producing per-frame MANO hand pose from
HaMeR, for both hands. Clean-room port (the reference is a research repo, MIT
licensed) that reuses this project's existing pieces heavily:

  - the shared ViT-H backbone (`VitHugeBackbone`, same weights family, loaded
    here from `hamer.safetensors`),
  - our COCO-17 ViTPose (`gvhmr_vitpose`) for the wrist/elbow keypoints that
    locate each hand, HaMeR's demo uses a whole-body ViTPose for tight hand
    boxes, which this project doesn't have; see `hamer_preprocess`,
  - the SAM 3.1 human mask (stage 1) for the person box, rescaled from SAM's
    1008x1008 working resolution to native like `gvhmr_adapter` does,
  - the rotation math (`matrix_to_axis_angle`).

The head outputs rotation matrices in the hand crop's camera frame; this adapter
converts them to axis-angle and, for left hands (which are flipped to look like
right hands before inference), applies HaMeR's `fliplr` correction (negate the
y/z axis-angle components). Reconciling the wrist orientation with GVHMR's
forearm is deliberately left to stage 5 (retarget_hands), this stage only
produces the raw per-hand MANO pose.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors import safe_open
from scipy.ndimage import maximum_filter1d, minimum_filter1d

from pipeline.progress_tracker import StageName

from ...helpers.progress_reporter import frame_progress
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
from .hamer_preprocess import COCO_L_WRIST, COCO_R_WRIST, crop_hand, hand_box_from_body_kpts

CHECKPOINT_DIR = Path(__file__).resolve().parents[3] / "checkpoints"
HAMER_CHECKPOINT = CHECKPOINT_DIR / "hamer.safetensors"

_BACKBONE_PREFIX = "backbone."
_MANO_HEAD_PREFIX = "mano_head."
# The backbone is trained on 256x192; the crop is 256x256, so trim 32px off each
# side of the width, matching HaMeR's `self.backbone(x[:,:,:,32:-32])`.
_WIDTH_CROP = 32

HAND_POSE_DIM = 45  # 15 MANO joints x 3 axis-angle

# HaMeR has no visibility/occlusion signal of its own, it regresses *a* MANO
# pose from whatever's in the crop regardless of whether the hand is actually
# there, and `hand_box_from_body_kpts`'s own keypoint-confidence gate (0.3)
# doesn't catch this: a body-occluded wrist (e.g. gripping something out of view
# behind the torso) can still get a confidently-placed box from ViTPose's pose
# prior, since that confidence reflects "a plausible keypoint location," not
# "the hand is visible here." The result is a `valid=True` frame with a
# physically implausible pose.
#
# A kinematic (angular-velocity) outlier check was tried first and rejected: on
# a real clip mixing walking and reaching, legitimate motion is itself fast and
# noisy enough that there's no velocity threshold separating it from an occluded
# stretch's per-frame noise, both cover the same range. Wrist keypoint
# confidence is a much cleaner signal because it's a direct estimate of
# visibility rather than an inference from motion. On the same clip, confidence
# alone (thresholded on a rolling window, not a single frame, confidence can
# blip back up mid-occlusion for a stray frame or two) isolated exactly the
# visually-confirmed occluded stretch with zero false positives anywhere else in
# 700 frames across both hands.
#
# Frames that fail this are left `valid=False`, letting the existing gap-fill
# contract (`motion_smoothing.fill_invalid`, called from stage 4) interpolate
# or freeze through them exactly like any other occlusion.
MIN_ROLLING_WRIST_CONFIDENCE = 0.6
WRIST_CONFIDENCE_WINDOW_FRAMES = 7  # centered rolling window

# HaMeR regresses each hand from an independent crop. Nearby overlapping hands
# can obscure fingers, whereas an object in the crop often means an intentional,
# visible grip. Both are softer confidence cues, never missing-data evidence,
# so real grip adjustments remain continuous. The small temporal dilation keeps
# one-frame hand-crop-boundary flicker from changing that confidence abruptly.
MIN_HAND_OBJECT_CROP_COVERAGE = 0.02
MIN_HAND_OVERLAP_CROP_AREA = 0.10
MIN_HAND_OVERLAP_WRIST_DISTANCE_CROP_SCALE = 0.35
MIN_HAND_AMBIGUITY_WINDOW_FRAMES = 5


def _reject_low_confidence_stretches(valid: np.ndarray, wrist_conf: np.ndarray) -> np.ndarray:
    """Demote to invalid any frame whose wrist-keypoint confidence, over a
    centered rolling window, dips below `MIN_ROLLING_WRIST_CONFIDENCE`, the
    rolling min (rather than the frame's own raw confidence) is what catches a
    frame that happens to look momentarily confident in the middle of an
    otherwise-occluded stretch."""
    half = WRIST_CONFIDENCE_WINDOW_FRAMES // 2
    rolling_min_conf = minimum_filter1d(wrist_conf, size=2 * half + 1, mode="nearest")
    return valid & (rolling_min_conf >= MIN_ROLLING_WRIST_CONFIDENCE)


# infer() output keys (per-frame arrays). KEY_LEFT_VALID/KEY_RIGHT_VALID are
# this stage's detection-based validity (did HaMeR find something here at all,
# confidently), gates finger smoothing directly, and is the base that
# stage_4_estimate_hands's own KEY_LEFT_WRIST_VALID/KEY_RIGHT_WRIST_VALID
# (computed downstream, not produced by infer() itself, since it needs GVHMR's
# elbow) narrows further for the wrist specifically. Kept as two separate
# arrays rather than one shared one: a biomechanically implausible wrist
# estimate doesn't mean the finger articulation from the same frame is
# untrustworthy too (see hand_retarget.py's module docstring).
KEY_LEFT_HAND_POSE = "left_hand_pose"
KEY_RIGHT_HAND_POSE = "right_hand_pose"
KEY_LEFT_GLOBAL_ORIENT = "left_global_orient"
KEY_RIGHT_GLOBAL_ORIENT = "right_global_orient"
KEY_LEFT_VALID = "left_valid"
KEY_RIGHT_VALID = "right_valid"
KEY_LEFT_WRIST_VALID = "left_wrist_valid"
KEY_RIGHT_WRIST_VALID = "right_wrist_valid"
KEY_LEFT_FINGER_AMBIGUOUS = "left_finger_ambiguous"
KEY_RIGHT_FINGER_AMBIGUOUS = "right_finger_ambiguous"
KEY_LEFT_FINGER_OBJECT_CONTACT = "left_finger_object_contact"
KEY_RIGHT_FINGER_OBJECT_CONTACT = "right_finger_object_contact"


def _box_overlapping_area(a: np.ndarray, b: np.ndarray) -> float:
    left, top = np.maximum(a[:2], b[:2])
    right, bottom = np.minimum(a[2:], b[2:])
    overlap = np.maximum(right - left, 0.0) * np.maximum(bottom - top, 0.0)
    area_a = np.prod(np.maximum(a[2:] - a[:2], 0.0))
    area_b = np.prod(np.maximum(b[2:] - b[:2], 0.0))
    union = area_a + area_b - overlap
    return float(overlap / union) if union > 0 else 0.0


def _hand_crop_has_object_contact(hand_box: np.ndarray, object_mask: np.ndarray | None) -> bool:
    """Whether a tracked object occupies a meaningful part of the hand crop."""
    if object_mask is None:
        return False
    height, width = object_mask.shape
    x1, y1 = np.maximum(np.floor(hand_box[:2]).astype(int), 0)
    x2, y2 = np.minimum(np.ceil(hand_box[2:]).astype(int), [width, height])
    if x2 <= x1 or y2 <= y1:
        return False
    return float(object_mask[y1:y2, x1:x2].mean()) >= MIN_HAND_OBJECT_CROP_COVERAGE


def _hand_crop_is_ambiguous(
    hand_box: np.ndarray,
    other_hand_box: np.ndarray | None,
    wrist: np.ndarray,
    other_wrist: np.ndarray | None,
) -> bool:
    """Whether nearby overlapping hands make a crop less reliable.

    Crop overlap alone is insufficient because forearm-derived boxes can
    overlap for two well-separated hands. Require the wrists to be nearby too.
    Face proximity is deliberately not used: a nearby face does not establish
    finger occlusion. Object contact is separately retained as a soft cue.
    """
    if other_hand_box is not None and other_wrist is not None:
        crop_scale = float(min(np.max(hand_box[2:] - hand_box[:2]), np.max(other_hand_box[2:] - other_hand_box[:2])))
        if (
            _box_overlapping_area(hand_box, other_hand_box) >= MIN_HAND_OVERLAP_CROP_AREA
            and np.linalg.norm(wrist[:2] - other_wrist[:2]) <= MIN_HAND_OVERLAP_WRIST_DISTANCE_CROP_SCALE * crop_scale
        ):
            return True

    return False


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
    def _infer_one_hand(self, frame_bgr: np.ndarray, box: np.ndarray, is_right: bool):
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

    def infer(self, frame_paths: list[Path], human_masks: dict, object_masks: dict | None = None) -> dict[str, np.ndarray]:
        packed = human_masks[KEY_PACKED_MASKS]
        n = len(frame_paths)
        out = {
            KEY_LEFT_HAND_POSE: np.zeros((n, HAND_POSE_DIM), np.float32),
            KEY_RIGHT_HAND_POSE: np.zeros((n, HAND_POSE_DIM), np.float32),
            KEY_LEFT_GLOBAL_ORIENT: np.zeros((n, 3), np.float32),
            KEY_RIGHT_GLOBAL_ORIENT: np.zeros((n, 3), np.float32),
            KEY_LEFT_VALID: np.zeros(n, bool),
            KEY_RIGHT_VALID: np.zeros(n, bool),
            KEY_LEFT_FINGER_AMBIGUOUS: np.zeros(n, bool),
            KEY_RIGHT_FINGER_AMBIGUOUS: np.zeros(n, bool),
            KEY_LEFT_FINGER_OBJECT_CONTACT: np.zeros(n, bool),
            KEY_RIGHT_FINGER_OBJECT_CONTACT: np.zeros(n, bool),
        }
        object_packed = None if object_masks is None else object_masks.get(KEY_PACKED_MASKS)
        # Wrist keypoint confidence per frame, per hand, collected for the
        # rolling-confidence outlier pass below, which needs the whole sequence
        # (including frames after each one) so it can't run inline in this loop.
        wrist_conf = {True: np.zeros(n, np.float32), False: np.zeros(n, np.float32)}

        for i, frame_path in frame_progress(enumerate(frame_paths), total=n, label=StageName.STAGE_4_ESTIMATE_HANDS.label):
            frame_bgr = cv2.imread(str(frame_path))
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            mask = unpack_masks(packed[i])[0].numpy()
            bbox = extract_bbox_from_numpy_mask(mask)
            if bbox is None:
                continue
            bbox = _rescale_bbox_xywh(bbox, from_hw=mask.shape, to_hw=frame_bgr.shape[:2])
            keypoints = estimate_keypoints(self._vitpose, frame_rgb, bbox, self.device, self.dtype)
            wrist_conf[True][i] = keypoints[COCO_R_WRIST][2]
            wrist_conf[False][i] = keypoints[COCO_L_WRIST][2]

            object_mask = None
            if object_packed is not None:
                object_mask = unpack_masks(object_packed[i])[0].numpy().astype(np.uint8)
                object_mask = cv2.resize(
                    object_mask, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            hand_boxes = {
                True: hand_box_from_body_kpts(keypoints, is_right=True),
                False: hand_box_from_body_kpts(keypoints, is_right=False),
            }
            wrists = {
                True: keypoints[COCO_R_WRIST],
                False: keypoints[COCO_L_WRIST],
            }

            for is_right, pose_key, go_key, valid_key in (
                (True, KEY_RIGHT_HAND_POSE, KEY_RIGHT_GLOBAL_ORIENT, KEY_RIGHT_VALID),
                (False, KEY_LEFT_HAND_POSE, KEY_LEFT_GLOBAL_ORIENT, KEY_LEFT_VALID),
            ):
                result = self._infer_one_hand(frame_bgr, hand_boxes[is_right], is_right)
                if result is None:
                    continue
                out[go_key][i], out[pose_key][i] = result
                out[valid_key][i] = True
                ambiguous_key = KEY_RIGHT_FINGER_AMBIGUOUS if is_right else KEY_LEFT_FINGER_AMBIGUOUS
                object_contact_key = KEY_RIGHT_FINGER_OBJECT_CONTACT if is_right else KEY_LEFT_FINGER_OBJECT_CONTACT
                out[ambiguous_key][i] = _hand_crop_is_ambiguous(
                    hand_boxes[is_right], hand_boxes[not is_right], wrists[is_right], wrists[not is_right],
                )
                out[object_contact_key][i] = _hand_crop_has_object_contact(hand_boxes[is_right], object_mask)

        for is_right, valid_key in ((True, KEY_RIGHT_VALID), (False, KEY_LEFT_VALID)):
            out[valid_key] = _reject_low_confidence_stretches(out[valid_key], wrist_conf[is_right])
        for ambiguous_key in (KEY_LEFT_FINGER_AMBIGUOUS, KEY_RIGHT_FINGER_AMBIGUOUS):
            out[ambiguous_key] = maximum_filter1d(
                out[ambiguous_key].astype(np.uint8), size=MIN_HAND_AMBIGUITY_WINDOW_FRAMES, mode="nearest",
            ).astype(bool)

        return out

    def unload(self) -> None:
        del self._backbone, self._head, self._vitpose
        self._backbone = self._head = self._vitpose = None
        torch.cuda.empty_cache()
