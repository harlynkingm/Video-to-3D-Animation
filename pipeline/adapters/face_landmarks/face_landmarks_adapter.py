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
FLAME->ARKit translation on every channel except lateral jaw, horizontal
gaze, blink, and `NoseSneer`/`CheekSquint`; the later six-capture comparison
also selects MediaPipe for `JawOpen` and `EyeWide`).

Receives face boxes from Stage 9's shared ViTPose/SAM pre-pass. Unlike
DECA/MICA, MediaPipe's Task API is not a torch module: it runs its own CPU,
XNNPACK-accelerated TFLite graph, so there is no ``device``/``dtype`` to
place the landmarker itself on.
"""

from __future__ import annotations

from queue import Full, Queue
from pathlib import Path
from threading import Thread

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from pipeline.progress_tracker import StageName

from ...helpers.progress_reporter import frame_progress
from ...helpers.livelink_csv import ARKIT_BLENDSHAPE_NAMES
from .face_landmarks_preprocess import crop_for_landmarker, landmarks_to_full_frame

CHECKPOINT_DIR = Path(__file__).resolve().parents[3] / "checkpoints"
FACE_LANDMARKER_CHECKPOINT = CHECKPOINT_DIR / "face_landmarker.task"

NUM_LANDMARKS = 478
NUM_BLENDSHAPES = 52
PROGRESS_LABEL = f"{StageName.STAGE_9_CAPTURE_FACE.label} 1/5 (MediaPipe + ViTPose)"

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


def _empty_output(n: int) -> dict[str, np.ndarray]:
    return {
        KEY_LANDMARKS: np.zeros((n, NUM_LANDMARKS, 3), np.float32),
        KEY_BLENDSHAPES: np.zeros((n, NUM_BLENDSHAPES), np.float32),
        KEY_VALID: np.zeros(n, bool),
    }


class FaceLandmarksAdapter:
    def __init__(self, device: torch.device | None = None, dtype: torch.dtype = torch.float16):
        # Retained for API consistency with the other Stage 9 adapters.
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._landmarker: mp_vision.FaceLandmarker | None = None

    def load(self, landmarker_checkpoint: Path = FACE_LANDMARKER_CHECKPOINT) -> None:
        base_options = mp_python.BaseOptions(model_asset_path=str(landmarker_checkpoint))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options, num_faces=1, output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def _infer_one_frame(self, frame_bgr: np.ndarray, face_box: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        crop_rgb, offset_xy = crop_for_landmarker(frame_bgr, face_box)
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

    def infer(self, frame_paths: list[Path], face_boxes: list[np.ndarray | None]) -> dict[str, np.ndarray]:
        """Runs MediaPipe over boxes from ``FacePrepassAdapter``."""
        if len(face_boxes) != len(frame_paths):
            raise ValueError("face_boxes must contain exactly one entry per frame")
        n = len(frame_paths)
        out = _empty_output(n)

        for i, frame_path in frame_progress(enumerate(frame_paths), total=n, label=PROGRESS_LABEL):
            frame_bgr = cv2.imread(str(frame_path))
            face_box = face_boxes[i]
            if face_box is None:
                continue
            result = self._infer_one_frame(frame_bgr, face_box)
            if result is None:
                continue
            out[KEY_LANDMARKS][i], out[KEY_BLENDSHAPES][i] = result
            out[KEY_VALID][i] = True

        return out

    def unload(self) -> None:
        del self._landmarker
        self._landmarker = None
        # This adapter no longer owns a torch/CUDA model. In the pipelined
        # path unload() runs on the CPU worker while ViTPose may still be
        # executing on the main thread, so it must not touch CUDA state.


class FaceLandmarksWorker:
    """One image-mode MediaPipe task running beside GPU ViTPose batches.

    The task is constructed, used, and destroyed entirely in this one worker
    thread. Its input queue is deliberately bounded to one batch: this keeps
    memory bounded and applies back-pressure instead of retaining a clip's
    worth of decoded frames. No two threads ever call ``detect`` on the same
    task instance.
    """

    def __init__(self, n_frames: int, max_queued_batches: int = 1):
        if max_queued_batches < 1:
            raise ValueError("max_queued_batches must be positive")
        self._queue: Queue[tuple[list[int], list[np.ndarray], list[np.ndarray]] | None] = Queue(maxsize=max_queued_batches)
        self._out = _empty_output(n_frames)
        self._error: BaseException | None = None
        self._thread = Thread(target=self._run, name="stage9-mediapipe", daemon=True)
        self._started = False
        self._finished = False

    def start(self) -> None:
        self._thread.start()
        self._started = True

    def submit(self, indices: list[int], frames_bgr: list[np.ndarray], face_boxes: list[np.ndarray]) -> None:
        if len(indices) != len(frames_bgr) or len(indices) != len(face_boxes):
            raise ValueError("MediaPipe batch indices, frames, and face boxes must have the same length")
        if not indices:
            return
        item = (indices, frames_bgr, face_boxes)
        while True:
            self._raise_if_failed()
            try:
                self._queue.put(item, timeout=0.1)
                return
            except Full:
                continue

    def finish(self) -> dict[str, np.ndarray]:
        if not self._started:
            raise RuntimeError("Call start() before finish().")
        if self._finished:
            raise RuntimeError("finish() may only be called once")
        self._finished = True
        while self._thread.is_alive():
            self._raise_if_failed()
            try:
                self._queue.put(None, timeout=0.1)
                break
            except Full:
                continue
        self._thread.join()
        self._raise_if_failed()
        return self._out

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("MediaPipe worker failed") from self._error

    def _run(self) -> None:
        adapter = FaceLandmarksAdapter()
        try:
            adapter.load()
            while True:
                item = self._queue.get()
                if item is None:
                    return
                indices, frames_bgr, face_boxes = item
                for frame_index, frame_bgr, face_box in zip(indices, frames_bgr, face_boxes):
                    result = adapter._infer_one_frame(frame_bgr, face_box)
                    if result is None:
                        continue
                    self._out[KEY_LANDMARKS][frame_index], self._out[KEY_BLENDSHAPES][frame_index] = result
                    self._out[KEY_VALID][frame_index] = True
        except BaseException as exc:
            self._error = exc
        finally:
            if adapter._landmarker is not None:
                adapter.unload()
