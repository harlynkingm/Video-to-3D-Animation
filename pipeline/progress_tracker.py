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
    STAGE_0_INGEST_VIDEO = "ingest_video"
    STAGE_1_MASK_AND_TRACK = "mask_and_track"
    STAGE_2_ESTIMATE_HUMAN_MOTION = "estimate_human_motion"
    STAGE_3_ESTIMATE_DEPTH = "estimate_depth"
    STAGE_4_ESTIMATE_HANDS = "estimate_hands"
    STAGE_5_RETARGET_HANDS = "retarget_hands"
    STAGE_6_ALIGN_SCENE_SCALE = "align_scene_scale"
    STAGE_7_ANNOTATE_CONTACTS = "annotate_contacts"
    STAGE_8_OPTIMIZE_HOI = "optimize_hoi"
    STAGE_9_EXPORT_FBX = "export_fbx"


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
