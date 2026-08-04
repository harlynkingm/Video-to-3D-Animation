"""Standalone diagnostic, not part of the pipeline -- no stage imports this,
and it isn't wired into any DAG. Safe to run by hand against any already-
exported run to spot-check wrist plausibility; nothing else depends on it.

Measures the real angle between the forearm bone (`{side}_elbow`, which spans
elbow->wrist) and the hand bone (`{side}_wrist`, which spans wrist->hand) in
an exported `output.blend`'s own rig -- the same ground-truth check that
caught and confirmed the 2026-08-03 fix for wrists rendering anatomically
impossible (folded back into the forearm) despite every pre-export validity
gate reading the pose as plausible. A real wrist rarely exceeds ~90 degrees
here; this project's own SMPL-X rest pose reads under 10.

Run inside the `export` pixi environment, pointing at a run directory that
already has `output.blend` (i.e. has completed stage 9):

    pixi run -e export python scripts/measure_wrist_forearm_angle.py runs/my_clip

Prints per-side summary stats and any contiguous frame ranges over the
threshold -- read the printed output rather than the process exit code:
like every other bpy CLI invocation in this project, the interpreter's own
teardown can report a nonzero exit even after this script's own logic
finished and printed cleanly (a known bpy quirk, not a bug here).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_BLEND_FILENAME = "output.blend"

SIDES = ("left", "right")
DEFAULT_THRESHOLD_DEG = 90.0  # roughly the real limit of human wrist flexion/extension


def _bone_direction(pose_bone) -> np.ndarray:
    d = np.array(pose_bone.tail - pose_bone.head, dtype=float)
    return d / np.linalg.norm(d)


def _rest_angle_deg(armature, side: str) -> float:
    forearm = armature.data.bones[f"{side}_elbow"]
    hand = armature.data.bones[f"{side}_wrist"]
    f_dir = np.array(forearm.tail_local - forearm.head_local, dtype=float)
    h_dir = np.array(hand.tail_local - hand.head_local, dtype=float)
    f_dir /= np.linalg.norm(f_dir)
    h_dir /= np.linalg.norm(h_dir)
    return float(np.degrees(np.arccos(np.clip(f_dir @ h_dir, -1.0, 1.0))))


def _contiguous_runs(over: np.ndarray, frames: list[int]) -> list[tuple[int, int]]:
    runs = []
    i = 0
    while i < len(over):
        if not over[i]:
            i += 1
            continue
        j = i
        while j < len(over) and over[j]:
            j += 1
        runs.append((frames[i], frames[j - 1]))
        i = j
    return runs


def measure(blend_path: Path, threshold_deg: float) -> bool:
    """Returns True if every frame stayed under `threshold_deg` on both sides."""
    import bpy

    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    scene = bpy.context.scene

    frames = list(range(scene.frame_start, scene.frame_end + 1))
    all_ok = True

    for side in SIDES:
        rest_deg = _rest_angle_deg(armature, side)
        print(f"\n{side}: rest-pose forearm-to-hand angle = {rest_deg:.1f} deg")

        forearm_bone = armature.pose.bones[f"{side}_elbow"]
        hand_bone = armature.pose.bones[f"{side}_wrist"]
        angles = np.empty(len(frames))
        for k, frame in enumerate(frames):
            scene.frame_set(frame)
            f_dir = _bone_direction(forearm_bone)
            h_dir = _bone_direction(hand_bone)
            angles[k] = np.degrees(np.arccos(np.clip(f_dir @ h_dir, -1.0, 1.0)))

        print(f"  min={angles.min():.1f}  median={np.median(angles):.1f}  max={angles.max():.1f}")
        over = angles > threshold_deg
        runs = _contiguous_runs(over, frames)
        if runs:
            all_ok = False
            print(f"  {len(runs)} run(s) over {threshold_deg:.0f} deg:")
            for start, end in runs:
                print(f"    frames {start}-{end} ({end - start + 1} frames)")
        else:
            print(f"  no frames over {threshold_deg:.0f} deg")

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the real wrist-vs-forearm bone angle in an exported output.blend"
    )
    parser.add_argument("run_dir", type=Path, help="Run directory containing output.blend (i.e. stage 9 has run)")
    parser.add_argument(
        "--threshold-deg",
        type=float,
        default=DEFAULT_THRESHOLD_DEG,
        help=f"Flag any frame over this angle (default {DEFAULT_THRESHOLD_DEG:.0f}, roughly real wrist range)",
    )
    args = parser.parse_args()

    blend_path = args.run_dir / OUTPUT_BLEND_FILENAME
    if not blend_path.exists():
        raise SystemExit(f"{blend_path} not found -- has stage 9 (export) run for this directory?")

    ok = measure(blend_path, args.threshold_deg)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
