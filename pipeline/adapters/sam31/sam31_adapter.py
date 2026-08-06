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

import math
from pathlib import Path

import cv2
import torch
from safetensors import safe_open

from pipeline.progress_tracker import StageName

from ...algorithms.contact_detection import contiguous_true_runs
from .sam31_clip_text import Sam31TextTower, encode_prompt, load_tokenizer
from .sam31_detector import Sam31Detector
from .sam31_tracker import KEY_MASKS, KEY_N_FRAMES, KEY_PACKED_MASKS, KEY_SCORES, Sam31Tracker, unpack_masks
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
KEY_SCORE = "score"  # singular -- highest first-detection confidence among the slots that
                     # ended up contributing a frame (see _stitch_tracked_slots), distinct
                     # from the tracker's own per-slot KEY_SCORES list

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


# How far (pixels) a chosen mask's own centroid may move between two
# consecutive resolved frames before being treated as a different, wrongly-
# matched object rather than the same one continuing to move -- confirmed
# on real footage that a genuine identity switch jumps 900+ pixels in a
# single frame, while even the fastest real motion seen (an object mid-
# throw) stayed under 35px/frame. A wide margin either direction, not
# independently calibrated per clip/resolution -- a fast enough real throw
# could in principle still exceed it and get wrongly treated as a gap
# (left to `max_bridge_frames` to paper over), same tradeoff any
# continuity-based check has.
MAX_PLAUSIBLE_JUMP_PX = 200.0


def _mask_centroid(packed_frame_slot_mask: torch.Tensor) -> tuple[float, float]:
    """(x, y) pixel centroid of a single already-known-nonempty packed (H,
    W//8) mask."""
    ys, xs = torch.nonzero(unpack_masks(packed_frame_slot_mask), as_tuple=True)
    return float(xs.float().mean()), float(ys.float().mean())


def _stitch_tracked_slots(packed_masks: torch.Tensor, scores: list[float], max_bridge_frames: int) -> tuple[torch.Tensor, float]:
    """Merge `track_video_with_detection`'s multiplex result for a singular
    prompt ("a basketball") into one object's per-frame mask sequence.

    The tracker can lose and re-detect the same physical object several
    times in one clip, each re-detection landing as its own multiplex slot;
    a second, unrelated real-world object can also independently match the
    same prompt and get its own slot.

    Per frame: among slots with a nonempty mask, pick whichever centroid is
    closest to the previously chosen frame's centroid -- not whichever has
    the higher fixed (first-detection) score, which can favor a completely
    different, unmoving object over the genuinely-tracked one. Even a lone
    active slot is checked this way: if it's farther than `MAX_PLAUSIBLE_
    JUMP_PX` from the last chosen position, the frame is left as a gap
    rather than accepted, since a lone wrong slot is exactly how one would
    otherwise permanently take over. Falls back to the higher score only at
    the very first active frame, with no prior position to compare against.
    Gaps of at most `max_bridge_frames` are then bridged by holding the
    previous frame's mask forward, same convention as `motion_smoothing.
    cap_long_gaps_with_hold`.

    Returns (merged (N, 1, H, W//8) packed masks, the highest score among
    slots that actually contributed a frame).
    """
    n_frames, n_obj = packed_masks.shape[0], packed_masks.shape[1]
    scores_t = torch.tensor(scores, dtype=torch.float32)

    active = torch.stack([packed_masks[:, s].reshape(n_frames, -1).any(dim=1).bool() for s in range(n_obj)])  # (n_obj, N)

    winner = torch.full((n_frames,), -1, dtype=torch.long)
    prev_centroid: tuple[float, float] | None = None
    for f in range(n_frames):
        candidates = torch.nonzero(active[:, f], as_tuple=True)[0].tolist()
        if not candidates:
            continue

        if prev_centroid is None:
            chosen = candidates[int(scores_t[candidates].argmax())]
            winner[f] = chosen
            prev_centroid = _mask_centroid(packed_masks[f, chosen])
            continue

        centroids = {s: _mask_centroid(packed_masks[f, s]) for s in candidates}
        chosen = min(candidates, key=lambda s: math.dist(centroids[s], prev_centroid))
        if math.dist(centroids[chosen], prev_centroid) > MAX_PLAUSIBLE_JUMP_PX:
            continue  # every candidate implausibly far -- leave this frame a gap for bridging to fill
        winner[f] = chosen
        prev_centroid = centroids[chosen]

    merged = torch.zeros(n_frames, 1, *packed_masks.shape[2:], dtype=packed_masks.dtype)
    any_active = winner >= 0
    frame_idx = torch.nonzero(any_active, as_tuple=True)[0]
    merged[frame_idx, 0] = packed_masks[frame_idx, winner[frame_idx]]

    # Bridging holds the previous frame's already-resolved MASK CONTENT forward
    # across the gap -- not the winning slot index re-looked-up at the gap frame
    # itself, which is empty there by definition (that's why it's a gap).
    for start, end in contiguous_true_runs((winner == -1).numpy()):
        if start == 0 or end == n_frames - 1:
            continue  # leading/trailing -- no prior mask to hold from / never recovers
        if end - start + 1 <= max_bridge_frames:
            merged[start:end + 1] = merged[start - 1]

    contributing = set(winner[winner >= 0].tolist())
    best_score = max(scores[i] for i in contributing)
    return merged, best_score


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

    def _track_one_prompt(
        self, images: _LazyFrameLoader, prompt: str, progress_label: str, max_bridge_frames: int
    ) -> dict:
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
                progress_label=progress_label,
            )

        if result[KEY_PACKED_MASKS] is None or not result[KEY_SCORES]:
            return {KEY_PACKED_MASKS: None, KEY_N_FRAMES: result[KEY_N_FRAMES], KEY_SCORE: None}

        merged, best_score = _stitch_tracked_slots(result[KEY_PACKED_MASKS], result[KEY_SCORES], max_bridge_frames)
        return {KEY_PACKED_MASKS: merged, KEY_N_FRAMES: result[KEY_N_FRAMES], KEY_SCORE: best_score}

    def infer(
        self, frame_paths: list[Path], human_prompt: str, object_prompt: str | None, max_bridge_frames: int
    ) -> dict:
        """Track `human_prompt` (always) and `object_prompt` (if given) across the
        whole clip.

        `max_bridge_frames`: see `_stitch_tracked_slots` -- how many consecutive
        frames with no active detection get bridged by holding the last tracked
        mask forward, before a gap is left genuinely empty.

        Returns {"human": <result>, "object": <result> | None}, each result being
        {"packed_masks": [N_frames, 1, H, W//8] bit-packed uint8 (or None if that
        entity was never detected), "n_frames": N, "score": the highest
        first-detection confidence among slots that actually contributed a frame
        (or None)}. See `sam31_tracker.pack_masks`/`unpack_masks` for the packed
        mask format.
        """
        images = _LazyFrameLoader(frame_paths)
        human_result = self._track_one_prompt(
            images, human_prompt, progress_label=StageName.STAGE_1A_HUMAN_MASK.label,
            max_bridge_frames=max_bridge_frames,
        )
        object_result = (
            self._track_one_prompt(
                images, object_prompt, progress_label=StageName.STAGE_1B_OBJECT_MASK.label,
                max_bridge_frames=max_bridge_frames,
            )
            if object_prompt else None
        )
        return {KEY_HUMAN: human_result, KEY_OBJECT: object_result}
