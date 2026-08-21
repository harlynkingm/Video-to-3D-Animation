"""Shared Stage 9 face-localization pre-pass.

DECA and MediaPipe used to independently decode every input frame, unpack the
same stage-1 human mask, run the same ViTPose checkpoint, and derive the same
face box. This adapter performs that deterministic work once and retains only
the resulting ``[x1, y1, x2, y2]`` boxes, not full-resolution frames.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch

from pipeline.helpers.torch_helpers import empty_cuda_cache, sys_torch_device
from pipeline.progress_tracker import StageName

from ..helpers.progress_reporter import frame_progress
from .deca.deca_preprocess import face_box_from_body_kpts
from .gvhmr.gvhmr_adapter import VITPOSE_CHECKPOINT, _load_direct_state, _rescale_bbox_xywh, extract_bbox_from_numpy_mask
from .gvhmr.gvhmr_vitpose import GVHMRViTPoseModel, estimate_keypoints_batch
from .sam31.sam31_tracker import KEY_PACKED_MASKS, unpack_masks


PROGRESS_LABEL = f"{StageName.STAGE_9_CAPTURE_FACE.label} 1/5 (MediaPipe + ViTPose)"
DEFAULT_INFERENCE_BATCH_SIZE = 16


class FacePrepassAdapter:
    """Loads ViTPose once and produces the shared per-frame face boxes."""

    def __init__(self, device: torch.device | None = None, dtype: torch.dtype = torch.float16):
        self.device = device or sys_torch_device()
        self.dtype = dtype
        self._vitpose: GVHMRViTPoseModel | None = None

    def load(self, vitpose_checkpoint: Path = VITPOSE_CHECKPOINT) -> None:
        self._vitpose = GVHMRViTPoseModel()
        self._vitpose.load_state_dict(_load_direct_state(vitpose_checkpoint), strict=True)
        self._vitpose.to(device=self.device, dtype=self.dtype).eval()

    @torch.inference_mode()
    def infer(
        self,
        frame_paths: list[Path],
        human_masks: dict,
        batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
        on_face_boxes: Callable[[list[int], list[np.ndarray], list[np.ndarray]], None] | None = None,
    ) -> list[np.ndarray | None]:
        """Returns one face ``xyxy`` box per source frame, or ``None``.

        The mask rescaling, RGB conversion, ViTPose call, and face-box
        calculation intentionally match the former DECA/MediaPipe code paths.
        When ``on_face_boxes`` is given, it receives each completed ViTPose
        batch's source indices, already-decoded BGR frames, and non-null face
        boxes. Stage 9 uses this to overlap the unchanged CPU MediaPipe image
        inference with the next GPU ViTPose batch.
        """
        assert self._vitpose is not None, "Call load() before infer()."
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        packed = human_masks[KEY_PACKED_MASKS]
        face_boxes: list[np.ndarray | None] = [None] * len(frame_paths)

        batch_indices: list[int] = []
        batch_frames_bgr: list[np.ndarray] = []
        batch_frames_rgb: list[np.ndarray] = []
        batch_bboxes: list[tuple[int, int, int, int]] = []

        def flush_batch() -> None:
            if not batch_indices:
                return
            keypoints_batch = estimate_keypoints_batch(
                self._vitpose, batch_frames_rgb, batch_bboxes, self.device, self.dtype,
            )
            detected_indices: list[int] = []
            detected_frames_bgr: list[np.ndarray] = []
            detected_face_boxes: list[np.ndarray] = []
            for batch_index, frame_index in enumerate(batch_indices):
                face_box = face_box_from_body_kpts(keypoints_batch[batch_index])
                face_boxes[frame_index] = face_box
                if face_box is not None:
                    detected_indices.append(frame_index)
                    detected_frames_bgr.append(batch_frames_bgr[batch_index])
                    detected_face_boxes.append(face_box)
            if on_face_boxes is not None and detected_indices:
                on_face_boxes(detected_indices, detected_frames_bgr, detected_face_boxes)
            batch_indices.clear()
            batch_frames_bgr.clear()
            batch_frames_rgb.clear()
            batch_bboxes.clear()

        for i, frame_path in frame_progress(enumerate(frame_paths), total=len(frame_paths), label=PROGRESS_LABEL):
            frame_bgr = cv2.imread(str(frame_path))
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mask = unpack_masks(packed[i])[0].numpy()
            bbox = extract_bbox_from_numpy_mask(mask)
            if bbox is None:
                continue
            bbox = _rescale_bbox_xywh(bbox, from_hw=mask.shape, to_hw=frame_bgr.shape[:2])
            batch_indices.append(i)
            batch_frames_bgr.append(frame_bgr)
            batch_frames_rgb.append(frame_rgb)
            batch_bboxes.append(bbox)
            if len(batch_indices) == batch_size:
                flush_batch()

        flush_batch()

        return face_boxes

    def unload(self) -> None:
        del self._vitpose
        self._vitpose = None
        empty_cuda_cache(self.device)
