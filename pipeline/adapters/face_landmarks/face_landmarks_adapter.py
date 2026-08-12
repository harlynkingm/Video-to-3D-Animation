"""Thin load()/infer()/unload() wrapper around MediaPipe's FaceLandmarker
Task, producing per-frame dense (478-point) face landmarks in full-frame
pixel coordinates, the 2D observations `face_landmark_fit.py`'s per-frame
optimization fits FLAME against, and the source for the 68-point/51-point/
5-point subsets `mp2dlib.py` derives for FLAME's landmark embedding and
MICA's ArcFace alignment respectively, plus MediaPipe's own native ARKit-52
blendshape scores from the same detection call (`output_face_blendshapes=
True`), which now source most of `output_face.csv` directly (see
`face_blendshapes.py`'s own docstring for the full story: measured across
three real paired-capture clips to beat this project's own hand-derived
FLAME->ARKit translation on every channel except jaw, horizontal gaze,
blink/wide, and `NoseSneer`/`CheekSquint`).

Reuses this project's existing pieces exactly like `DecaAdapter` does: COCO-17
ViTPose for the face keypoints that locate the crop, and the SAM 3.1 human
mask for the person box. Unlike DECA/MICA, MediaPipe's Task API is not a
torch module, it runs its own (CPU, XNNPACK-accelerated) TFLite graph, so
there is no `device`/`dtype` to place it on.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from pipeline.progress_tracker import StageName

from ...helpers.progress_reporter import frame_progress
from ...helpers.livelink_csv import ARKIT_BLENDSHAPE_NAMES
from ..deca.deca_preprocess import face_box_from_body_kpts
from ..gvhmr.gvhmr_adapter import VITPOSE_CHECKPOINT, _load_direct_state, _rescale_bbox_xywh, extract_bbox_from_numpy_mask
from ..gvhmr.gvhmr_vitpose import GVHMRViTPoseModel, estimate_keypoints
from ..sam31.sam31_tracker import KEY_PACKED_MASKS, unpack_masks
from .face_landmarks_preprocess import crop_for_landmarker, landmarks_to_full_frame

CHECKPOINT_DIR = Path(__file__).resolve().parents[3] / "checkpoints"
FACE_LANDMARKER_CHECKPOINT = CHECKPOINT_DIR / "face_landmarker.task"

NUM_LANDMARKS = 478
NUM_BLENDSHAPES = 52
PROGRESS_LABEL = f"{StageName.STAGE_9_CAPTURE_FACE.label} 2/3 (MediaPipe)"

# infer() output keys (per-frame arrays).
KEY_LANDMARKS = "mp_landmarks"  # (F, 478, 3): full-frame pixel x/y, relative z
KEY_BLENDSHAPES = "mp_blendshapes"  # (F, 52), ordered to match livelink_csv.ARKIT_BLENDSHAPE_NAMES exactly
KEY_VALID = "mp_valid"

# MediaPipe's own 52 ARKit-named categories (its own `output_face_
# blendshapes` result, a 53rd `_neutral` category dropped) use lowerCamelCase
# ("browInnerUp"); this project's own convention throughout (`ARKIT_
# BLENDSHAPE_NAMES`, the real LiveLinkFace capture's own CSV header) is
# PascalCase ("BrowInnerUp"), verified directly (not assumed) against a
# real capture: `_ARKIT_NAME_TO_MP_NAME` built once here so every other
# caller can work in this project's own name order without knowing
# MediaPipe's own convention exists. One real, confirmed exception: `Tongue
# Out` isn't among MediaPipe's own predicted categories at all (its model
# never scores tongue state), `_infer_one_frame`'s own lookup below
# defaults it to 0.0 rather than KeyError-ing, which also matches this
# project's own already-established policy of treating `TongueOut` as hard
# zero regardless of source (see `stage_9_capture_face.HARD_ZERO_CHANNELS`).
_ARKIT_NAME_TO_MP_NAME = {name: name[0].lower() + name[1:] for name in ARKIT_BLENDSHAPE_NAMES}


class FaceLandmarksAdapter:
    def __init__(self, device: torch.device | None = None, dtype: torch.dtype = torch.float16):
        # device/dtype are for the ViTPose sub-model only; the landmarker itself is CPU.
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._landmarker: mp_vision.FaceLandmarker | None = None
        self._vitpose: GVHMRViTPoseModel | None = None

    def load(
        self, landmarker_checkpoint: Path = FACE_LANDMARKER_CHECKPOINT, vitpose_checkpoint: Path = VITPOSE_CHECKPOINT
    ) -> None:
        base_options = mp_python.BaseOptions(model_asset_path=str(landmarker_checkpoint))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options, num_faces=1, output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        self._vitpose = GVHMRViTPoseModel()
        self._vitpose.load_state_dict(_load_direct_state(vitpose_checkpoint), strict=True)
        self._vitpose.to(device=self.device, dtype=self.dtype).eval()

    def _infer_one_frame(self, frame_bgr: np.ndarray, keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        box = face_box_from_body_kpts(keypoints)
        if box is None:
            return None
        crop_rgb, offset_xy = crop_for_landmarker(frame_bgr, box)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks or not result.face_blendshapes:
            return None
        landmarks_norm = np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]], dtype=np.float32)
        landmarks = landmarks_to_full_frame(landmarks_norm, crop_rgb.shape, offset_xy)

        scores_by_name = {c.category_name: c.score for c in result.face_blendshapes[0]}
        blendshapes = np.array(
            [scores_by_name.get(_ARKIT_NAME_TO_MP_NAME[name], 0.0) for name in ARKIT_BLENDSHAPE_NAMES], dtype=np.float32,
        )
        return landmarks, blendshapes

    def infer(self, frame_paths: list[Path], human_masks: dict) -> dict[str, np.ndarray]:
        packed = human_masks[KEY_PACKED_MASKS]
        n = len(frame_paths)
        out = {
            KEY_LANDMARKS: np.zeros((n, NUM_LANDMARKS, 3), np.float32),
            KEY_BLENDSHAPES: np.zeros((n, NUM_BLENDSHAPES), np.float32),
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
            out[KEY_LANDMARKS][i], out[KEY_BLENDSHAPES][i] = result
            out[KEY_VALID][i] = True

        return out

    def unload(self) -> None:
        del self._landmarker, self._vitpose
        self._landmarker = self._vitpose = None
        torch.cuda.empty_cache()
