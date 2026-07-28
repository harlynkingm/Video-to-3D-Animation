"""create_run: bootstraps a new pipeline run -- a fresh progress directory with
a `progress.json` seeded from the user's input (video path, prompts, camera
info). Every stage script expects this file to already exist (see
`pipeline_stage_base.cli_entrypoint`), so this runs first, once per clip.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .progress_tracker import (
    NewRunID,
    RunRecord,
    RunInput,
    StageName,
    StageRecord,
    add_dataclass_cli_arguments,
    add_run_input_arguments,
    run_input_from_args,
)

# The pipeline's dependency DAG. Stages not yet implemented are included here too
# (as pending, never-run records) so the full chain is visible in `progress.json`
# from the start, and so a later stage's `depends_on` doesn't need editing in once
# its own file is finally written.
STAGE_DEPENDS_ON: dict[StageName, list[StageName]] = {
    StageName.STAGE_0_INGEST_VIDEO: [],
    StageName.STAGE_1_MASK_AND_TRACK: [StageName.STAGE_0_INGEST_VIDEO],
    StageName.STAGE_2_ESTIMATE_HUMAN_MOTION: [StageName.STAGE_0_INGEST_VIDEO, StageName.STAGE_1_MASK_AND_TRACK],
    StageName.STAGE_3_ESTIMATE_DEPTH: [StageName.STAGE_0_INGEST_VIDEO, StageName.STAGE_1_MASK_AND_TRACK],
    # estimate_hands needs the person mask (stage 1) to locate the person, its
    # own ViTPose pass to locate the hands, AND the body motion (stage 2): a
    # hand is an extension of the arm, not an independent tracked object, and
    # stage 4 checks every raw wrist estimate against GVHMR's own elbow
    # orientation for biomechanical plausibility before its own smoothing runs
    # (see stage_4_estimate_hands.py's module docstring). This is a real
    # kinematic dependency, not just an implementation convenience, so hands and
    # body motion no longer run in parallel.
    StageName.STAGE_4_ESTIMATE_HANDS: [
        StageName.STAGE_0_INGEST_VIDEO,
        StageName.STAGE_1_MASK_AND_TRACK,
        StageName.STAGE_2_ESTIMATE_HUMAN_MOTION,
    ],
    # retarget_hands attaches the stage-4 hands onto the stage-2 body -- it needs
    # both, and nothing else.
    StageName.STAGE_5_RETARGET_HANDS: [
        StageName.STAGE_2_ESTIMATE_HUMAN_MOTION,
        StageName.STAGE_4_ESTIMATE_HANDS,
    ],
    # align_scene_scale's eventual DAG position routes its SMPL-X input through
    # retarget_hands, but scene *scale* only needs the body's overall size,
    # which the body-only estimate_human_motion already gives -- so it depends
    # on that directly and does not wait on the (not-yet-built) hand stages.
    StageName.STAGE_6_ALIGN_SCENE_SCALE: [
        StageName.STAGE_1_MASK_AND_TRACK,
        StageName.STAGE_2_ESTIMATE_HUMAN_MOTION,
        StageName.STAGE_3_ESTIMATE_DEPTH,
    ],
    # annotate_contacts detects hand-to-object proximity in 2D image space,
    # using the retargeted hand joints (stage 5) and the object's own 2D mask
    # (stage 1) -- deliberately NOT align_scene_scale's 3D object primitive or
    # real-world depth, since GVHMR's own Z estimate is measurably unreliable
    # for a reaching/foreshortened arm and an animation only needs the hand and
    # object to agree with each other in their own shared space, not with
    # absolute ground truth (see contact_detection.py's module docstring).
    StageName.STAGE_7_ANNOTATE_CONTACTS: [
        StageName.STAGE_1_MASK_AND_TRACK,
        StageName.STAGE_5_RETARGET_HANDS,
    ],
    # optimize_hoi solves the tracked object's own per-frame pose: rigidly
    # attached to a body joint during a qualifying contact event (needs
    # annotate_contacts), against the object's fitted shape and the scale/
    # translation that reconciles depth with the body (needs
    # align_scene_scale). It never touches retarget_hands directly -- the
    # body/hand motion is trusted as-is, not refined.
    StageName.STAGE_8_OPTIMIZE_HOI: [
        StageName.STAGE_6_ALIGN_SCENE_SCALE,
        StageName.STAGE_7_ANNOTATE_CONTACTS,
    ],
    # export also reads retarget_hands' body motion (stage 5) and
    # align_scene_scale's object shape (stage 6) directly, but doesn't list
    # either here -- optimize_hoi (its only real dependency now, for the
    # object's real per-frame pose) already transitively guarantees both are
    # complete by the time it finishes: directly via align_scene_scale, and
    # via annotate_contacts -> retarget_hands.
    StageName.STAGE_9_EXPORT: [
        StageName.STAGE_8_OPTIMIZE_HOI,
    ],
}


def create_run(progress_dir: Path, run_input: RunInput, run_id: str | None = None) -> RunRecord:
    """`run_id` is just a human-readable label stored alongside the run's data --
    `progress_dir` is what actually identifies a run on disk -- so it defaults to
    the directory's own name rather than requiring the caller to repeat it.
    """
    progress_dir.mkdir(parents=True, exist_ok=True)
    runRecord = RunRecord(
        run_id=run_id or progress_dir.name,
        progress_dir=str(progress_dir),
        input=run_input,
        stages={
            stage.value: StageRecord(depends_on=[dep.value for dep in deps])
            for stage, deps in STAGE_DEPENDS_ON.items()
        },
    )
    runRecord.save()
    return runRecord


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a new pipeline run")
    add_dataclass_cli_arguments(parser, NewRunID)
    add_run_input_arguments(parser)
    args = parser.parse_args()

    run_input = run_input_from_args(args)
    runRecord = create_run(args.progress_dir, run_input, run_id=args.run_id)
    print(f"Created run {runRecord.run_id!r} at {args.progress_dir}")


if __name__ == "__main__":
    main()
