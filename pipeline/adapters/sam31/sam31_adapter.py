"""Thin wrapper tying the four sam31 port files together into the one call
`stage_1_mask_and_track.py` needs: given a directory of extracted frames plus a
human prompt and an optional object prompt, track each entity across the whole
clip and return its per-frame masks.

Follows a `load()`/`infer()`/`unload()` convention informally, without a shared
`ModelAdapter` base class: with only one adapter written so far, that base
would be structure before there's a second implementation to generalize from --
add it once a GVHMR/depth adapter actually exists too.

**Human and object are tracked via two independent `track_video_with_detection`
calls, not one joint call with both prompts.** SAM 3.1's multi-entity detection
concatenates every prompt's candidate queries into one pool before running NMS
and assigning multiplex slots -- there is no query-to-prompt identity preserved
across that step, so a single joint call can't reliably say which tracked
object is "the human" versus "the object" (in the tracker's own verification
test, the human happened to score higher and land first only by coincidence --
not something to rely on). Two separate calls make each entity's identity
unambiguous by construction, at the cost of running the ViTDet backbone twice
per frame instead of once -- a real but bounded runtime cost, not a VRAM one
(frames are still processed one at a time either way). This still avoids
loading the checkpoint into VRAM twice, which is what this project's original
"merge human+object into one stage" scope decision was actually about.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import torch
from safetensors import safe_open

from .sam31_clip_text import Sam31TextTower, encode_prompt, load_tokenizer
from .sam31_detector import Sam31Detector
from .sam31_tracker import KEY_MASKS, KEY_N_FRAMES, KEY_PACKED_MASKS, KEY_SCORES, Sam31Tracker
from .sam31_vitdet_backbone import Sam31VisionBackbone, TrackerMode

# Repo root is 3 levels up from this file (sam31/ -> adapters/ -> pipeline/ -> root).
CHECKPOINT_PATH = Path(__file__).resolve().parents[3] / "checkpoints" / "sam3.1_multiplex_fp16.safetensors"

# Checkpoint tensor-key prefixes, used to split the flat checkpoint into each module's
# own state dict (see _load_checkpoint_state).
_VISION_BACKBONE_PREFIX = "detector.backbone.vision_backbone."
_TEXT_TOWER_PREFIX = "detector.backbone.language_backbone.encoder."
_TEXT_RESIZER_PREFIX = "detector.backbone.language_backbone.resizer"
_TEXT_RESIZER_KEY_PREFIX = "text_resizer"  # this project's own renamed key prefix for resizer weights
_DETECTOR_PREFIX = "detector."
_DETECTOR_BACKBONE_PREFIX = "detector.backbone."
_TRACKER_PREFIX = "tracker.model."

# _load_checkpoint_state's returned dict keys (module name -> its own state dict).
KEY_VISION_BACKBONE = "vision_backbone"
KEY_TEXT_TOWER = "text_tower"
KEY_DETECTOR = "detector"
KEY_TRACKER = "tracker"

# This adapter's own per-entity result dict keys.
KEY_SCORE = "score"  # singular -- first-detection confidence, distinct from the tracker's own per-frame KEY_SCORES list

# infer()'s top-level output dict keys.
KEY_HUMAN = "human"
KEY_OBJECT = "object"


def _load_checkpoint_state(checkpoint_path: Path) -> dict[str, dict]:
    """Split the checkpoint's flat tensor keys into each module's own state dict,
    stripping its key prefix -- the same slicing verified in each module's own
    test harness while writing it.
    """
    vision_backbone, text_tower, detector, resizer, tracker = {}, {}, {}, {}, {}
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(_VISION_BACKBONE_PREFIX):
                vision_backbone[key[len(_VISION_BACKBONE_PREFIX):]] = f.get_tensor(key)
            elif key.startswith(_TEXT_TOWER_PREFIX):
                text_tower[key[len(_TEXT_TOWER_PREFIX):]] = f.get_tensor(key)
            elif key.startswith(_TEXT_RESIZER_PREFIX):
                resizer[_TEXT_RESIZER_KEY_PREFIX + key[len(_TEXT_RESIZER_PREFIX):]] = f.get_tensor(key)
            elif key.startswith(_DETECTOR_PREFIX) and not key.startswith(_DETECTOR_BACKBONE_PREFIX):
                detector[key[len(_DETECTOR_PREFIX):]] = f.get_tensor(key)
            elif key.startswith(_TRACKER_PREFIX):
                tracker[key[len(_TRACKER_PREFIX):]] = f.get_tensor(key)
    return {
        KEY_VISION_BACKBONE: vision_backbone,
        KEY_TEXT_TOWER: text_tower,
        KEY_DETECTOR: {**detector, **resizer},
        KEY_TRACKER: tracker,
    }


class _LazyFrameLoader:
    """Presents disk-backed frames as the `[N, 3, H, W]`-shaped, sliceable object
    `Sam31Tracker.track_video_with_detection` expects for its `images` argument,
    without ever materializing the whole clip in memory at once.

    The tracker only ever reads one frame at a time (`images[frame_idx:frame_idx+1]`,
    via `_prep_frame` in `sam31_tracker.py` -- its only call site), so this decodes
    just that one frame from disk on each access instead of preloading every frame
    as a float32 CPU tensor up front. That preload was the real memory bottleneck
    for long videos (~11MB/frame at 720p, scaling linearly with clip length) --
    this makes stage 1's memory footprint independent of how long the clip is.
    """

    def __init__(self, frame_paths: list[Path]):
        self._frame_paths = frame_paths
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    @property
    def shape(self) -> tuple[int]:
        return (len(self._frame_paths),)

    def __getitem__(self, index: slice) -> torch.Tensor:
        frames = []
        for path in self._frame_paths[index]:
            frame_bgr = cv2.imread(str(path))
            if frame_bgr is None:
                raise RuntimeError(f"Could not read frame: {path}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0)
        return torch.stack(frames)


class Sam31Adapter:
    """`load()` once per stage run, `infer()` for the whole clip, `unload()` before
    the process exits -- brackets the checkpoint's time on the GPU to match this
    project's one-model-at-a-time VRAM budget.
    """

    def __init__(self, device: torch.device | None = None, dtype: torch.dtype = torch.float16):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._loaded = False

    def load(self, checkpoint_path: Path = CHECKPOINT_PATH) -> None:
        state = _load_checkpoint_state(checkpoint_path)

        self.vision_backbone = Sam31VisionBackbone().to(dtype=self.dtype)
        self.vision_backbone.load_state_dict(state[KEY_VISION_BACKBONE], strict=True)

        self.text_tower = Sam31TextTower().to(dtype=self.dtype)
        self.text_tower.load_state_dict(state[KEY_TEXT_TOWER], strict=True)

        self.detector = Sam31Detector().to(dtype=self.dtype)
        self.detector.load_state_dict(state[KEY_DETECTOR], strict=True)

        self.tracker = Sam31Tracker().to(dtype=self.dtype)
        self.tracker.load_state_dict(state[KEY_TRACKER], strict=True)

        for module in (self.vision_backbone, self.text_tower, self.detector, self.tracker):
            module.to(self.device).eval()

        self.tokenizer = load_tokenizer()
        self._loaded = True

    def unload(self) -> None:
        if not self._loaded:
            return
        del self.vision_backbone, self.text_tower, self.detector, self.tracker, self.tokenizer
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._loaded = False

    def _track_one_prompt(self, images: _LazyFrameLoader, prompt: str) -> dict:
        """Run one full-clip tracking pass for a single prompt. Returns a single-object
        result -- see this module's docstring for why one prompt is tracked at a time.
        """
        with torch.inference_mode():
            text_emb, text_mask = encode_prompt(prompt, self.text_tower, self.tokenizer, self.device, self.dtype)

            def backbone_fn(frame, frame_idx=None):
                trunk_out = self.vision_backbone.trunk(frame)
                _, _, tf, tp = self.vision_backbone(
                    frame, tracker_mode=TrackerMode.PROPAGATION, cached_trunk=trunk_out, tracker_only=True)
                return tf, tp, trunk_out

            def detect_fn(trunk_out):
                features = [conv(trunk_out) for conv in self.vision_backbone.convs]
                positions = [self.vision_backbone.position_encoding(f) for f in features]
                det = self.detector.forward_from_trunk(features, positions, text_emb, text_mask)
                return {KEY_SCORES: det[KEY_SCORES], KEY_MASKS: det[KEY_MASKS]}

            result = self.tracker.track_video_with_detection(
                backbone_fn, images, initial_masks=None, detect_fn=detect_fn,
                new_det_thresh=0.5, max_objects=0, detect_interval=1,
                backbone_obj=self.vision_backbone, target_device=self.device, target_dtype=self.dtype,
            )

        if result[KEY_PACKED_MASKS] is None or not result[KEY_SCORES]:
            return {KEY_PACKED_MASKS: None, KEY_N_FRAMES: result[KEY_N_FRAMES], KEY_SCORE: None}

        # A singular prompt ("a tennis player") should track exactly one object; if the
        # detector's NMS pool ever yields more than one candidate for it (e.g. a second
        # person briefly matching "a tennis player"), keep only the highest-confidence
        # one rather than returning an ambiguous multi-object result for what this
        # pipeline treats as a single entity (see the locked "rigid, single-instance
        # object" scope decision).
        best_idx = max(range(len(result[KEY_SCORES])), key=lambda i: result[KEY_SCORES][i])
        return {
            KEY_PACKED_MASKS: result[KEY_PACKED_MASKS][:, best_idx:best_idx + 1],
            KEY_N_FRAMES: result[KEY_N_FRAMES],
            KEY_SCORE: result[KEY_SCORES][best_idx],
        }

    def infer(self, frame_paths: list[Path], human_prompt: str, object_prompt: str | None) -> dict:
        """Track `human_prompt` (always) and `object_prompt` (if given) across the
        whole clip.

        Returns {"human": <result>, "object": <result> | None}, each result being
        {"packed_masks": [N_frames, 1, H, W//8] bit-packed uint8 (or None if that
        entity was never detected), "n_frames": N, "score": first-detection
        confidence (or None)}. See `sam31_tracker.pack_masks`/`unpack_masks` for the
        packed mask format.
        """
        images = _LazyFrameLoader(frame_paths)
        human_result = self._track_one_prompt(images, human_prompt)
        object_result = self._track_one_prompt(images, object_prompt) if object_prompt else None
        return {KEY_HUMAN: human_result, KEY_OBJECT: object_result}
