"""Thin `load()`/`infer()`/`unload()` wrapper around the `depth-anything-3` pip
package's own high-level API, producing metric depth (meters) and confidence
for a single frame. Unlike SAM 3.1/GVHMR, this isn't a clean-room port --
Depth-Anything-3's official repo is a standalone, Apache 2.0-licensed Python
package with no ComfyUI/GPL dependency, so it's used directly as a normal pip
dependency (see ARCHITECTURE.md's Depth-Anything-3 notes).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from depth_anything_3.api import DepthAnything3

# Apache 2.0 checkpoint, single-image *metric* depth (real meters, not just
# relative-scale) -- see ARCHITECTURE.md for why this checkpoint specifically,
# out of DA3's several licensing tiers.
MODEL_NAME = "depth-anything/DA3METRIC-LARGE"

# Repo root is 2 levels up from this file (adapters/ -> pipeline/ -> root).
# Routes the auto-downloaded HF checkpoint into this project's usual
# checkpoints/ folder instead of the user's global ~/.cache/huggingface, so
# every model this pipeline uses lives in one place on disk.
CHECKPOINT_CACHE_DIR = Path(__file__).resolve().parents[2] / "checkpoints" / "depth_anything_3"

# DA3METRIC-LARGE's raw network output is not already in meters -- this is the
# model's own documented conversion (its repo README's FAQ section), confirmed
# against the real Hugging Face checkpoint card before relying on it.
METRIC_DEPTH_DIVISOR = 300.0

KEY_DEPTH = "depth"
KEY_SKY = "sky"


class DepthAnything3Adapter:
    def __init__(self) -> None:
        self._model: DepthAnything3 | None = None

    def load(self) -> None:
        self._model = DepthAnything3.from_pretrained(MODEL_NAME, cache_dir=CHECKPOINT_CACHE_DIR)
        self._model = self._model.to(device="cuda").eval()

    def infer(self, frame_path: str, focal_length_px: float) -> dict[str, np.ndarray]:
        # DA3METRIC-LARGE's forward pass never populates "depth_conf" -- confirmed
        # by real inference, prediction.conf is always None for this checkpoint
        # (unlike the Any-view/Nested checkpoints this project doesn't use). No
        # confidence output exists to return here. It does populate a sky mask
        # (also confirmed via real inference), useful for excluding sky pixels
        # (set to max depth by the model) from any point-cloud visualization.
        prediction = self._model.inference([frame_path])
        raw_depth = prediction.depth[0]  # (H, W) float32 -- net output, not yet in meters
        metric_depth = raw_depth * (focal_length_px / METRIC_DEPTH_DIVISOR)
        result = {KEY_DEPTH: metric_depth}
        if prediction.sky is not None:
            result[KEY_SKY] = prediction.sky[0]
        return result

    def unload(self) -> None:
        del self._model
        self._model = None
        torch.cuda.empty_cache()
