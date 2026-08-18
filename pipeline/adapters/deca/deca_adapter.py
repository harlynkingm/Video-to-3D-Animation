"""Thin load()/infer()/unload() wrapper producing per-frame DECA-predicted
FLAME parameters (shape/expression/pose/camera), for use as the fitting
loop's initial guess. Stage 9's shared face pre-pass supplies the exact
per-frame face boxes, avoiding a redundant ViTPose invocation here.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

from pipeline.progress_tracker import StageName

from ...helpers.progress_reporter import frame_progress
from .deca_encoder import DecaEncoder
from .deca_preprocess import crop_face

CHECKPOINT_DIR = Path(__file__).resolve().parents[3] / "checkpoints"
DECA_CHECKPOINT = CHECKPOINT_DIR / "deca.safetensors"

N_SHAPE = 100
N_EXP = 50

PROGRESS_LABEL = f"{StageName.STAGE_9_CAPTURE_FACE.label} 2/5 (DECA)"
DEFAULT_INFERENCE_BATCH_SIZE = 16

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

    def load(self, deca_checkpoint: Path = DECA_CHECKPOINT) -> None:
        self._encoder = DecaEncoder()
        self._encoder.load_state_dict(load_file(str(deca_checkpoint)), strict=True)
        self._encoder.to(device=self.device, dtype=self.dtype).eval()

    @torch.inference_mode()
    def _infer_crops(self, crops: list[np.ndarray]) -> dict[str, np.ndarray]:
        assert self._encoder is not None, "Call load() before infer()."
        crop_t = torch.from_numpy(np.stack(crops)).to(device=self.device, dtype=self.dtype)
        out = self._encoder(crop_t)
        return {k: v.float().cpu().numpy() for k, v in out.items()}

    def infer(
        self, frame_paths: list[Path], face_boxes: list[np.ndarray | None], batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    ) -> dict[str, np.ndarray]:
        """Runs batched DECA inference over boxes from ``FacePrepassAdapter``."""
        if len(face_boxes) != len(frame_paths):
            raise ValueError("face_boxes must contain exactly one entry per frame")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        n = len(frame_paths)
        out = {
            KEY_SHAPE: np.zeros((n, N_SHAPE), np.float32),
            KEY_EXP: np.zeros((n, N_EXP), np.float32),
            KEY_POSE: np.zeros((n, 6), np.float32),
            KEY_CAM: np.zeros((n, 3), np.float32),
            KEY_VALID: np.zeros(n, bool),
        }

        batch_indices: list[int] = []
        batch_crops: list[np.ndarray] = []

        def flush_batch() -> None:
            if not batch_indices:
                return
            results = self._infer_crops(batch_crops)
            for batch_index, frame_index in enumerate(batch_indices):
                out[KEY_SHAPE][frame_index] = results["shape"][batch_index]
                out[KEY_EXP][frame_index] = results["exp"][batch_index]
                out[KEY_POSE][frame_index] = results["pose"][batch_index]
                out[KEY_CAM][frame_index] = results["cam"][batch_index]
                out[KEY_VALID][frame_index] = True
            batch_indices.clear()
            batch_crops.clear()

        for i, frame_path in frame_progress(enumerate(frame_paths), total=n, label=PROGRESS_LABEL):
            frame_bgr = cv2.imread(str(frame_path))
            face_box = face_boxes[i]
            if face_box is None:
                continue
            batch_indices.append(i)
            batch_crops.append(crop_face(frame_bgr, face_box))
            if len(batch_indices) == batch_size:
                flush_batch()

        flush_batch()

        return out

    def unload(self) -> None:
        del self._encoder
        self._encoder = None
        torch.cuda.empty_cache()
