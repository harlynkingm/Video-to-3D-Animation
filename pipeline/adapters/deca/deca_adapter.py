"""Thin load()/infer()/unload() wrapper producing per-frame DECA-predicted
FLAME parameters (shape/expression/pose/camera), for use as the fitting
loop's initial guess, DECA's own iterative refinement and photometric
rendering are not reproduced here (see `deca_encoder.py`'s module docstring).

Reuses this project's existing pieces: our COCO-17 ViTPose (`gvhmr_vitpose`)
for the face keypoints that locate the crop (see `deca_preprocess.py`), and
the SAM 3.1 human mask (stage 1) for the person box, rescaled from SAM's
working resolution to native like `gvhmr_adapter`/`hamer_adapter` do.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

from pipeline.progress_tracker import StageName

from ...helpers.progress_reporter import frame_progress
from ..gvhmr.gvhmr_adapter import VITPOSE_CHECKPOINT, _load_direct_state, _rescale_bbox_xywh, extract_bbox_from_numpy_mask
from ..gvhmr.gvhmr_vitpose import GVHMRViTPoseModel, estimate_keypoints
from ..sam31.sam31_tracker import KEY_PACKED_MASKS, unpack_masks
from .deca_encoder import DecaEncoder
from .deca_preprocess import crop_face, face_box_from_body_kpts

CHECKPOINT_DIR = Path(__file__).resolve().parents[3] / "checkpoints"
DECA_CHECKPOINT = CHECKPOINT_DIR / "deca.safetensors"

N_SHAPE = 100
N_EXP = 50

# Plain string, not a StageName reference: the face-capture stage this adapter
# feeds hasn't been wired into progress_tracker.py yet (that happens together
# with the export-stage renumber, as its own dedicated step).
PROGRESS_LABEL = f"{StageName.STAGE_9_CAPTURE_FACE.label} 1/5 (DECA)"

# infer() output keys (per-frame arrays).
KEY_SHAPE = "deca_shape"
KEY_EXP = "deca_exp"
KEY_POSE = "deca_pose"
KEY_CAM = "deca_cam"
KEY_VALID = "deca_valid"


class DecaAdapter:
    def __init__(self, device: torch.device | None = None, dtype: torch.dtype = torch.float16):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._encoder: DecaEncoder | None = None
        self._vitpose: GVHMRViTPoseModel | None = None

    def load(self, deca_checkpoint: Path = DECA_CHECKPOINT, vitpose_checkpoint: Path = VITPOSE_CHECKPOINT) -> None:
        self._encoder = DecaEncoder()
        self._encoder.load_state_dict(load_file(str(deca_checkpoint)), strict=True)
        self._vitpose = GVHMRViTPoseModel()
        self._vitpose.load_state_dict(_load_direct_state(vitpose_checkpoint), strict=True)

        for module in (self._encoder, self._vitpose):
            module.to(device=self.device, dtype=self.dtype).eval()

    @torch.inference_mode()
    def _infer_one_frame(self, frame_bgr: np.ndarray, keypoints: np.ndarray):
        box = face_box_from_body_kpts(keypoints)
        if box is None:
            return None
        crop = crop_face(frame_bgr, box)
        crop_t = torch.from_numpy(crop).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        out = self._encoder(crop_t)
        return {k: v.float().cpu().numpy()[0] for k, v in out.items()}

    def infer(self, frame_paths: list[Path], human_masks: dict) -> dict[str, np.ndarray]:
        packed = human_masks[KEY_PACKED_MASKS]
        n = len(frame_paths)
        out = {
            KEY_SHAPE: np.zeros((n, N_SHAPE), np.float32),
            KEY_EXP: np.zeros((n, N_EXP), np.float32),
            KEY_POSE: np.zeros((n, 6), np.float32),
            KEY_CAM: np.zeros((n, 3), np.float32),
            KEY_VALID: np.zeros(n, bool),
        }

        for i, frame_path in frame_progress(enumerate(frame_paths), total=n, label=PROGRESS_LABEL):
            frame_bgr = cv2.imread(str(frame_path))
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            mask = unpack_masks(packed[i])[0].numpy()
            bbox = extract_bbox_from_numpy_mask(mask)
            if bbox is None:
                continue
            bbox = _rescale_bbox_xywh(bbox, from_hw=mask.shape, to_hw=frame_bgr.shape[:2])
            keypoints = estimate_keypoints(self._vitpose, frame_rgb, bbox, self.device, self.dtype)

            result = self._infer_one_frame(frame_bgr, keypoints)
            if result is None:
                continue
            out[KEY_SHAPE][i] = result["shape"]
            out[KEY_EXP][i] = result["exp"]
            out[KEY_POSE][i] = result["pose"]
            out[KEY_CAM][i] = result["cam"]
            out[KEY_VALID][i] = True

        return out

    def unload(self) -> None:
        del self._encoder, self._vitpose
        self._encoder = self._vitpose = None
        torch.cuda.empty_cache()
