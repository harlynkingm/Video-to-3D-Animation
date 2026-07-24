"""Tracks a pipeline run's progress in a single JSON file (progress.json).

This is what makes a run resumable: each stage checks the progress record
before running to see whether its dependencies are already complete, and
updates it when it finishes. Killing the process and rerunning the same
command picks up where it left off instead of starting over.
"""

from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1

FIELD_INPUT = "input"
FIELD_SCENE = "scene"
FIELD_STAGES = "stages"
FIELD_STATUS = "status"
FIELD_OUTPUTS = "outputs"
FIELD_OBJECT_SHAPE_HINT = "object_shape_hint"

PROGRESS_JSON_NAME = "progress.json"


class StageName(enum.StrEnum):
    def __new__(cls, value, title) -> StageName:
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.title = title
        return obj

    STAGE_0_INGEST_VIDEO = "ingest_video", "initial stage: processing video"
    STAGE_1_MASK_AND_TRACK = "mask_and_track", "stage 1: generate masks"
    STAGE_1A_HUMAN_MASK = "generate_human_mask", "stage 1: generate human mask"
    STAGE_1B_OBJECT_MASK = "generate_object_mask", "stage 1: generate object mask"
    STAGE_2_ESTIMATE_HUMAN_MOTION = "estimate_human_motion", "stage 2: estimate human motion"
    STAGE_3_ESTIMATE_DEPTH = "estimate_depth", "stage 3: estimate scene depth"
    STAGE_4_ESTIMATE_HANDS = "estimate_hands", "stage 4: estimate hands motion"
    STAGE_5_RETARGET_HANDS = "retarget_hands","stage 5: fix hand tracking"
    STAGE_6_ALIGN_SCENE_SCALE = "align_scene_scale", "stage 6: fix scene scale"
    STAGE_7_ANNOTATE_CONTACTS = "annotate_contacts", "stage 7: align human-object contact points"
    STAGE_8_OPTIMIZE_HOI = "optimize_hoi", "stage 8: optimize animation"
    STAGE_9_EXPORT_FBX = "export_fbx", "stage 9: exporting animation"


class StageStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


class ObjectShapeHint(enum.StrEnum):
    AUTO = "auto"
    BOX = "box"
    SPHERE = "sphere"


@dataclass
class RunInput:
    video_path: str
    human_prompt: str
    object_prompt: str | None = None
    object_shape_hint: ObjectShapeHint = ObjectShapeHint.AUTO
    focal_length_mm: float = 0.0
    sensor_width_mm: float = 0.0
    anchor_frame_override: int | None = None
    dump_mask_previews: bool = False
    dump_motion_preview: bool = False
    dump_depth_preview: bool = False
    dump_scene_preview: bool = False
    dump_hands_preview: bool = False
    dump_retarget_preview: bool = False

    # Temporal-smoothing knobs. Not exposed as create_run CLI flags on purpose --
    # the defaults are tuned to need no adjustment; a power user can override them
    # by hand-editing these fields in a run's progress.json before running stage
    # 2/4. Body needs only light polish (GVHMR already runs a temporal model);
    # hands need a heavier window (HaMeR infers each frame independently). See
    # pipeline/algorithms/motion_smoothing.py.
    body_smoothing_window: int = 9
    body_translation_cutoff: float = 0.15
    hand_smoothing_window: int = 15


@dataclass
class SceneInfo:
    fps: float = 0.0
    width: int = 0
    height: int = 0
    frame_count: int = 0
    intrinsics_K: list[list[float]] = field(default_factory=list)
    anchor_frame_index: int = 0


@dataclass
class StageRecord:
    status: StageStatus = StageStatus.PENDING
    depends_on: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)  # output name -> path
    error: str | None = None


@dataclass
class RunOutputs:
    final_fbx: str | None = None


@dataclass
class ProgressRecord:
    run_id: str
    progress_dir: str
    input: RunInput
    scene: SceneInfo = field(default_factory=SceneInfo)
    stages: dict[str, StageRecord] = field(default_factory=dict)
    outputs: RunOutputs = field(default_factory=RunOutputs)
    schema_version: int = SCHEMA_VERSION

    @property
    def path(self) -> Path:
        return Path(self.progress_dir) / PROGRESS_JSON_NAME

    def save(self) -> None:
        data = asdict(self)
        tmp_path = self.path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(data, indent=2))
        tmp_path.replace(self.path)  # atomic rename on the same filesystem

    @classmethod
    def load(cls, progress_dir: str | Path) -> ProgressRecord:
        data = json.loads((Path(progress_dir) / PROGRESS_JSON_NAME).read_text())
        data[FIELD_INPUT] = RunInput(
            **{
                **data[FIELD_INPUT],
                FIELD_OBJECT_SHAPE_HINT: ObjectShapeHint(data[FIELD_INPUT][FIELD_OBJECT_SHAPE_HINT]),
            }
        )
        data[FIELD_SCENE] = SceneInfo(**data[FIELD_SCENE])
        data[FIELD_STAGES] = {
            name: StageRecord(**{**rec, FIELD_STATUS: StageStatus(rec[FIELD_STATUS])})
            for name, rec in data[FIELD_STAGES].items()
        }
        data[FIELD_OUTPUTS] = RunOutputs(**data[FIELD_OUTPUTS])
        return cls(**data)

    def is_complete(self, stage_name: StageName) -> bool:
        record = self.stages.get(stage_name)
        return record is not None and record.status == StageStatus.COMPLETE

    def dependencies_met(self, stage_name: StageName) -> bool:
        return all(self.is_complete(dep) for dep in self.stages[stage_name].depends_on)

    def mark_progress(
        self,
        stage_name: StageName,
        status: StageStatus,
        outputs: dict[str, str] | None = None,
        error: str | None = None,
    ) -> None:
        record = self.stages[stage_name]
        record.status = status
        record.error = error
        if outputs is not None:
            record.outputs = outputs
        self.save()
