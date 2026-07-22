"""Thin wrapper tying the four sam31 port files together into the one call
`stage_1_mask_and_track.py` needs: given a directory of extracted frames plus a
human prompt and an optional object prompt, track each entity across the whole
clip and return its per-frame masks.

Follows this project's adapter convention informally (`load()`/`infer()`/
`unload()` -- see docs/ARCHITECTURE.md's repository structure) without a shared
`ModelAdapter` base class: with only one adapter written so far, that base
would be structure before there's a second implementation to generalize from
(see [[feedback-minimal-codebase]]) -- add it once a GVHMR/depth adapter
actually exists too.

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
from .sam31_tracker import Sam31Tracker
from .sam31_vitdet_backbone import Sam31VisionBackbone

# Repo root is 3 levels up from this file (sam31/ -> adapters/ -> pipeline/ -> root).
CHECKPOINT_PATH = Path(__file__).resolve().parents[3] / "checkpoints" / "sam3.1_multiplex_fp16.safetensors"


def _load_checkpoint_state(checkpoint_path: Path) -> dict[str, dict]:
    """Split the checkpoint's flat tensor keys into each module's own state dict,
    stripping its key prefix -- the same slicing verified in each module's own
    test harness while writing it (see docs/ARCHITECTURE.md's port-scope section).
    """
    vision_backbone, text_tower, detector, resizer, tracker = {}, {}, {}, {}, {}
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith("detector.backbone.vision_backbone."):
                vision_backbone[key[len("detector.backbone.vision_backbone."):]] = f.get_tensor(key)
            elif key.startswith("detector.backbone.language_backbone.encoder."):
                text_tower[key[len("detector.backbone.language_backbone.encoder."):]] = f.get_tensor(key)
            elif key.startswith("detector.backbone.language_backbone.resizer"):
                resizer["text_resizer" + key[len("detector.backbone.language_backbone.resizer"):]] = f.get_tensor(key)
            elif key.startswith("detector.") and not key.startswith("detector.backbone."):
                detector[key[len("detector."):]] = f.get_tensor(key)
            elif key.startswith("tracker.model."):
                tracker[key[len("tracker.model."):]] = f.get_tensor(key)
    return {
        "vision_backbone": vision_backbone,
        "text_tower": text_tower,
        "detector": {**detector, **resizer},
        "tracker": tracker,
    }


def _load_frames(frame_paths: list[Path]) -> torch.Tensor:
    """Read frames back from disk (as extracted by stage 0) into one [N, 3, H, W]
    CPU float tensor in [0, 1] range -- the `images` shape
    `Sam31Tracker.track_video_with_detection` expects (it resizes each frame to
    the backbone's working size itself).
    """
    frames = []
    for path in frame_paths:
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
        self.vision_backbone.load_state_dict(state["vision_backbone"], strict=True)

        self.text_tower = Sam31TextTower().to(dtype=self.dtype)
        self.text_tower.load_state_dict(state["text_tower"], strict=True)

        self.detector = Sam31Detector().to(dtype=self.dtype)
        self.detector.load_state_dict(state["detector"], strict=True)

        self.tracker = Sam31Tracker().to(dtype=self.dtype)
        self.tracker.load_state_dict(state["tracker"], strict=True)

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

    def _track_one_prompt(self, images: torch.Tensor, prompt: str) -> dict:
        """Run one full-clip tracking pass for a single prompt. Returns a single-object
        result -- see this module's docstring for why one prompt is tracked at a time.
        """
        with torch.inference_mode():
            text_emb, text_mask = encode_prompt(prompt, self.text_tower, self.tokenizer, self.device, self.dtype)

            def backbone_fn(frame, frame_idx=None):
                trunk_out = self.vision_backbone.trunk(frame)
                _, _, tf, tp = self.vision_backbone(
                    frame, tracker_mode="propagation", cached_trunk=trunk_out, tracker_only=True)
                return tf, tp, trunk_out

            def detect_fn(trunk_out):
                features = [conv(trunk_out) for conv in self.vision_backbone.convs]
                positions = [self.vision_backbone.position_encoding(f) for f in features]
                det = self.detector.forward_from_trunk(features, positions, text_emb, text_mask)
                return {"scores": det["scores"], "masks": det["masks"]}

            result = self.tracker.track_video_with_detection(
                backbone_fn, images, initial_masks=None, detect_fn=detect_fn,
                new_det_thresh=0.5, max_objects=0, detect_interval=1,
                backbone_obj=self.vision_backbone, target_device=self.device, target_dtype=self.dtype,
            )

        if result["packed_masks"] is None or not result["scores"]:
            return {"packed_masks": None, "n_frames": result["n_frames"], "score": None}

        # A singular prompt ("a tennis player") should track exactly one object; if the
        # detector's NMS pool ever yields more than one candidate for it (e.g. a second
        # person briefly matching "a tennis player"), keep only the highest-confidence
        # one rather than returning an ambiguous multi-object result for what this
        # pipeline treats as a single entity (see the locked "rigid, single-instance
        # object" scope decision).
        best_idx = max(range(len(result["scores"])), key=lambda i: result["scores"][i])
        return {
            "packed_masks": result["packed_masks"][:, best_idx:best_idx + 1],
            "n_frames": result["n_frames"],
            "score": result["scores"][best_idx],
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
        images = _load_frames(frame_paths)
        human_result = self._track_one_prompt(images, human_prompt)
        object_result = self._track_one_prompt(images, object_prompt) if object_prompt else None
        return {"human": human_result, "object": object_result}
