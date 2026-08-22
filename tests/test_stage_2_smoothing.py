"""Focused Stage 2 wiring tests that do not need GVHMR checkpoints or a GPU."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from pipeline.adapters.gvhmr.gvhmr_adapter import (
    KEY_BODY_POSE,
    KEY_BETAS,
    KEY_GLOBAL_ORIENT,
    KEY_PRED_SMPL_PARAMS_GLOBAL,
    KEY_PRED_SMPL_PARAMS_INCAM,
    KEY_STATIC_CONF_LOGITS,
    KEY_TRANSL,
    KEY_TRANSL_INCAM_RAW,
)
from pipeline.progress_tracker import StageName
from pipeline.stages import stage_2_estimate_human_motion as stage_2


def _params(n_frames: int) -> dict[str, torch.Tensor]:
    return {
        KEY_BODY_POSE: torch.zeros(n_frames, 63),
        KEY_BETAS: torch.zeros(n_frames, 10),
        KEY_GLOBAL_ORIENT: torch.zeros(n_frames, 3),
        KEY_TRANSL: torch.zeros(n_frames, 3),
    }


def test_stage_2_relocks_smoothed_incam_motion_and_drops_working_confidences(tmp_path, monkeypatch):
    """The final lock must run after smoothing, but static confidences must
    remain private working data rather than changing human_motion.pt's schema."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for frame in range(3):
        (frames_dir / f"{frame:06d}.jpg").touch()
    masks_path = tmp_path / "human.pt"
    torch.save({stage_2.KEY_PACKED_MASKS: None}, masks_path)

    fake_incam = _params(3)
    fake_global = _params(3)
    fake_static_logits = torch.ones(3, 6)

    class FakeAdapter:
        relock_inputs: tuple[dict[str, torch.Tensor], torch.Tensor] | None = None
        grounding_inputs: tuple[dict[str, torch.Tensor], torch.Tensor, float, list[float] | None] | None = None
        stance_inputs: tuple[dict[str, torch.Tensor], torch.Tensor, float, list[float] | None] | None = None

        def load(self) -> None:
            pass

        def unload(self) -> None:
            pass

        def infer(self, *_args) -> dict:
            return {
                KEY_PRED_SMPL_PARAMS_INCAM: {key: value.clone() for key, value in fake_incam.items()},
                KEY_PRED_SMPL_PARAMS_GLOBAL: {key: value.clone() for key, value in fake_global.items()},
                KEY_TRANSL_INCAM_RAW: torch.zeros(3, 3),
                KEY_STATIC_CONF_LOGITS: fake_static_logits.clone(),
            }

        def relock_smoothed_incam_feet(self, params: dict[str, torch.Tensor], logits: torch.Tensor) -> torch.Tensor:
            FakeAdapter.relock_inputs = (params, logits)
            return torch.full_like(params[KEY_TRANSL], 7.0)

        def ground_smoothed_incam_vertical(
            self, params: dict[str, torch.Tensor], logits: torch.Tensor, fps: float,
            camera_up: list[float] | None = None,
        ) -> torch.Tensor:
            FakeAdapter.grounding_inputs = (
                {key: value.clone() for key, value in params.items()}, logits, fps, camera_up,
            )
            return torch.full_like(params[KEY_TRANSL], 11.0)

        def relock_stance_feet(
            self, params: dict[str, torch.Tensor], logits: torch.Tensor, fps: float,
            camera_up: list[float] | None = None,
        ) -> torch.Tensor:
            FakeAdapter.stance_inputs = (
                {key: value.clone() for key, value in params.items()}, logits, fps, camera_up,
            )
            return torch.full_like(params[KEY_BODY_POSE], 13.0)

    monkeypatch.setattr(stage_2, "GVHMRAdapter", FakeAdapter)
    monkeypatch.setattr(stage_2, "unpack_masks", lambda _packed: torch.zeros(3, 1, 1, 1, dtype=torch.bool))

    run_record = SimpleNamespace(
        stages={
            StageName.STAGE_0_INGEST_VIDEO: SimpleNamespace(outputs={stage_2.FRAMES_DIR_OUTPUT_KEY: str(frames_dir)}),
            StageName.STAGE_1_MASK_AND_TRACK: SimpleNamespace(outputs={stage_2.OUTPUT_HUMAN_MASKS: str(masks_path)}),
        },
        scene=SimpleNamespace(
            intrinsics_K=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], fps=30.0, camera_up=[],
        ),
        fine_tuning=SimpleNamespace(body_smoothing_window=9, body_translation_cutoff=0.15),
        progress_dir=str(tmp_path),
        input=SimpleNamespace(render_motion_preview=False),
    )

    output = stage_2.run(run_record)
    saved = torch.load(Path(output[stage_2.OUTPUT_HUMAN_MOTION]), weights_only=False)

    assert FakeAdapter.relock_inputs is not None
    assert torch.equal(FakeAdapter.relock_inputs[1], fake_static_logits)
    assert FakeAdapter.grounding_inputs is not None
    assert torch.equal(FakeAdapter.grounding_inputs[0][KEY_TRANSL], torch.full((3, 3), 7.0))
    assert torch.equal(FakeAdapter.grounding_inputs[1], fake_static_logits)
    assert FakeAdapter.grounding_inputs[2] == 30.0
    assert torch.equal(saved[KEY_PRED_SMPL_PARAMS_INCAM][KEY_TRANSL], torch.full((3, 3), 11.0))
    # The per-foot IK relock runs last, and must see the root-corrected transl
    # rather than the pre-correction one: it takes only the residual the whole-
    # body move could not fix.
    assert FakeAdapter.stance_inputs is not None
    assert torch.equal(FakeAdapter.stance_inputs[0][KEY_TRANSL], torch.full((3, 3), 11.0))
    assert torch.equal(FakeAdapter.stance_inputs[1], fake_static_logits)
    assert torch.equal(
        saved[KEY_PRED_SMPL_PARAMS_INCAM][KEY_BODY_POSE],
        torch.full_like(saved[KEY_PRED_SMPL_PARAMS_INCAM][KEY_BODY_POSE], 13.0),
    )
    assert KEY_STATIC_CONF_LOGITS not in saved
