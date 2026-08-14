"""Thin load()/infer()/unload() wrapper producing per-frame MICA-predicted
FLAME shape coefficients, for use as the fitting loop's identity estimate,
MICA specializes in shape/identity alone (no expression, pose, or camera).

Unlike `DecaAdapter`, this does not derive its own face crop from body
keypoints: ArcFace-family networks need a precise 5-point similarity-transform
alignment (see `mica_preprocess.py`), not a tolerant bbox crop, so `infer()`
takes already-detected 5-point landmarks per frame rather than computing them
itself. Wiring in a real landmark source (this project's own face-landmark
detector) is the caller's job.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

from pipeline.progress_tracker import StageName

from ...helpers.progress_reporter import frame_progress
from .mica_encoder import N_SHAPE, IResNet100, MicaRegressor
from .mica_preprocess import norm_crop, normalize_for_arcface

CHECKPOINT_DIR = Path(__file__).resolve().parents[3] / "checkpoints"
MICA_CHECKPOINT = CHECKPOINT_DIR / "mica.safetensors"

_ARCFACE_PREFIX = "arcface."
_REGRESSOR_PREFIX = "regressor."

PROGRESS_LABEL = f"{StageName.STAGE_9_CAPTURE_FACE.label} 3/5 (MICA)"

# infer() output keys (per-frame arrays).
KEY_SHAPE = "mica_shape"
KEY_VALID = "mica_valid"


class MicaAdapter:
    def __init__(self, device: torch.device | None = None, dtype: torch.dtype = torch.float16):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._backbone: IResNet100 | None = None
        self._regressor: MicaRegressor | None = None

    def load(self, mica_checkpoint: Path = MICA_CHECKPOINT) -> None:
        backbone_sd, regressor_sd = {}, {}
        for key, value in load_file(str(mica_checkpoint)).items():
            if key.startswith(_ARCFACE_PREFIX):
                backbone_sd[key[len(_ARCFACE_PREFIX):]] = value
            elif key.startswith(_REGRESSOR_PREFIX):
                regressor_sd[key[len(_REGRESSOR_PREFIX):]] = value

        self._backbone = IResNet100()
        self._backbone.load_state_dict(backbone_sd, strict=True)
        self._regressor = MicaRegressor()
        self._regressor.load_state_dict(regressor_sd, strict=True)

        for module in (self._backbone, self._regressor):
            module.to(device=self.device, dtype=self.dtype).eval()

    @torch.inference_mode()
    def _infer_one_frame(self, frame_bgr: np.ndarray, landmarks_5pt: np.ndarray) -> np.ndarray:
        aligned = norm_crop(frame_bgr, landmarks_5pt)
        crop = normalize_for_arcface(aligned)
        crop_t = torch.from_numpy(crop).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        embedding = self._backbone(crop_t)
        # L2-normalize to unit length before the regressor, matching MICA's own
        # `encode()` (`F.normalize(self.arcface(arcface_imgs))`). Load-bearing,
        # not cosmetic: the backbone's raw BatchNorm output has a norm around
        # 20, so skipping this inflates every regressed shape coefficient by
        # roughly that factor, FLAME betas are PCA units where a real face
        # sits within about |2-3|, and unnormalized they land above |11|,
        # deforming the mesh into a caricature.
        embedding = torch.nn.functional.normalize(embedding)
        shape = self._regressor(embedding)
        return shape.float().cpu().numpy()[0]

    def infer(
        self, frame_paths: list[Path], landmarks_5pt: np.ndarray, landmarks_valid: np.ndarray
    ) -> dict[str, np.ndarray]:
        """landmarks_5pt: (F, 5, 2) per-frame left eye/right eye/nose/left
        mouth/right mouth pixel coordinates. landmarks_valid: (F,) bool, from
        whatever detected them, frames without usable landmarks are skipped."""
        n = len(frame_paths)
        out = {
            KEY_SHAPE: np.zeros((n, N_SHAPE), np.float32),
            KEY_VALID: np.zeros(n, bool),
        }

        for i, frame_path in frame_progress(enumerate(frame_paths), total=n, label=PROGRESS_LABEL):
            if not landmarks_valid[i]:
                continue
            frame_bgr = cv2.imread(str(frame_path))
            out[KEY_SHAPE][i] = self._infer_one_frame(frame_bgr, landmarks_5pt[i])
            out[KEY_VALID][i] = True

        return out

    def unload(self) -> None:
        del self._backbone, self._regressor
        self._backbone = self._regressor = None
        torch.cuda.empty_cache()
