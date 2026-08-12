"""Quantitative ARKit-52 CSV export validation: compares this pipeline's own
`output_face.csv` against a real LiveLinkFace ground-truth capture,
per-channel Pearson correlation and RMS error, ranked worst-channel first.
The natural place to tune the still-untuned placeholders: `face_gaze.
ANGLE_SCALE_DEG`, `face_blendshapes.DEFAULT_ARKIT_TEMPORAL_LAMBDA`/
`DEFAULT_ARKIT_RIDGE`, `face_eyelid`'s calibration/hysteresis constants, and
`face_landmark_fit.DEFAULT_ITERS`/`lr`.

Two alignment details, both load-bearing (get either wrong and every number
this script reports is meaningless):

1. The ground-truth CSV's own row index does NOT correspond 1:1 with video
   frame index, `frame_log.csv` interleaves "B" (blendshape sample) and
   "V" (video frame) records, each carrying a shared nanosecond timestamp,
   and the two counters drift apart over the clip. The real mapping is
   B-row -> nearest-timestamp V-row -> that V-row's own frame index.
2. The ground-truth capture's own neutral pose is not zero (`_neutral.csv`)
  , comparing our zero-neutral output against it directly would show a
   constant per-channel offset that is a calibration artifact, not a real
   error, so `_neutral.csv`'s single row is subtracted from every raw row
   first.

Usage: `pixi run -e main python scripts/diagnostic/measure_arkit_error.py <run_dir>`
where `<run_dir>` is a completed pipeline run (through stage 9) that also
holds its own paired LiveLinkFace capture, `<run_dir>/*_raw.csv`,
`*_neutral.csv`, and `frame_log.csv` (the same layout the iPhone LiveLinkFace
app + `create_run.py` already produce; every real ground-truth clip so far
has followed it unmodified).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np


def _find_ground_truth_csv(run_dir: Path, suffix: str) -> Path:
    """`run_dir/*{suffix}`, e.g. `*_raw.csv`, found by glob rather than a
    hardcoded clip name, so this script works on any paired-capture run
    directory, not just the one it was originally built against. Raises
    with the actual glob and directory on a miss/ambiguity, since a wrong
    guess here would make every number this script prints meaningless (see
    this module's own docstring)."""
    matches = sorted(run_dir.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one *{suffix} in {run_dir}, found {matches}")
    return matches[0]

# The 52 ARKit blendshape column names, in the ground-truth CSV's own order
#, identical to livelink_csv.ARKIT_BLENDSHAPE_NAMES (both verified against
# this same real capture), imported directly so the two can never drift apart.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.helpers.livelink_csv import ARKIT_BLENDSHAPE_NAMES, HEAD_EYE_COLUMN_NAMES  # noqa: E402

ALL_COMPARABLE_COLUMNS = ARKIT_BLENDSHAPE_NAMES + HEAD_EYE_COLUMN_NAMES

# The real capture's own Head/Eye Euler columns are in radians, not the
# degrees `livelink_csv.write_livelink_csv`'s own contract documents for
# this pipeline's output, so they need converting before comparison. A
# validation-script-only issue (this file's own ground-truth loading, not
# `livelink_csv`'s output contract), correlation alone can't catch a
# unit mismatch, since it's scale-invariant; only a magnitude check does.
_HEAD_EYE_RADIANS_TO_DEGREES = HEAD_EYE_COLUMN_NAMES


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def _channel_matrix(rows: list[dict[str, str]], columns: list[str] = ALL_COMPARABLE_COLUMNS) -> np.ndarray:
    """(N, len(columns)), in `columns` order."""
    return np.array([[float(row[name]) for name in columns] for row in rows], dtype=np.float64)


def _blendshape_row_to_video_frame(frame_log_path: Path) -> np.ndarray:
    """Returns `mapping` (num_b_rows,): `mapping[i]` is the video frame index
    the i-th blendshape ("B") row corresponds to, via nearest-timestamp join
    against the "V" (video) records, see this module's own docstring."""
    b_index, b_ts, v_index, v_ts = [], [], [], []
    with open(frame_log_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if not parts or parts[0] not in ("B", "V"):
                continue
            kind, idx, timestamp = parts[0], int(parts[1]), int(parts[2])
            if kind == "B":
                b_index.append(idx)
                b_ts.append(timestamp)
            else:
                v_index.append(idx)
                v_ts.append(timestamp)

    b_ts_arr, v_ts_arr, v_index_arr = np.array(b_ts), np.array(v_ts), np.array(v_index)
    order = np.argsort(b_index)
    b_ts_sorted = b_ts_arr[order]

    nearest_v = np.searchsorted(v_ts_arr, b_ts_sorted)
    nearest_v = np.clip(nearest_v, 0, len(v_ts_arr) - 1)
    # searchsorted gives the insertion point, check the neighbor too, keep whichever timestamp is closer.
    left = np.clip(nearest_v - 1, 0, len(v_ts_arr) - 1)
    use_left = np.abs(v_ts_arr[left] - b_ts_sorted) < np.abs(v_ts_arr[nearest_v] - b_ts_sorted)
    nearest_v = np.where(use_left, left, nearest_v)

    mapping = np.zeros(len(b_index), dtype=np.int64)
    mapping[order] = v_index_arr[nearest_v]
    return mapping


def aligned_channel_data(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """`(ours_aligned, gt_aligned)`, both `(F, len(ALL_COMPARABLE_COLUMNS))`,
    the shared alignment/loading logic behind both this script's own console
    report and `build_ground_truth_report.py`'s HTML report, so the two can
    never compute inconsistent numbers for the same run. Raises if
    `run_dir/output_face.csv` doesn't exist, the paired capture's own
    `*_raw.csv`/`*_neutral.csv` can't be found unambiguously (see
    `_find_ground_truth_csv`), or `frame_log.csv`'s own B-record count
    doesn't match the raw ground-truth CSV's row count."""
    output_csv = run_dir / "output_face.csv"
    if not output_csv.exists():
        raise FileNotFoundError(f"{output_csv} not found, run the pipeline through stage 9 first")

    raw_csv = _find_ground_truth_csv(run_dir, "_raw.csv")
    neutral_csv = _find_ground_truth_csv(run_dir, "_neutral.csv")
    frame_log_csv = run_dir / "frame_log.csv"

    _, gt_rows = _read_csv_rows(raw_csv)
    _, neutral_rows = _read_csv_rows(neutral_csv)
    _, ours_rows = _read_csv_rows(output_csv)

    gt = _channel_matrix(gt_rows)
    neutral = _channel_matrix(neutral_rows)[0]
    head_eye_idx = [ALL_COMPARABLE_COLUMNS.index(name) for name in _HEAD_EYE_RADIANS_TO_DEGREES]
    gt[:, head_eye_idx] = np.degrees(gt[:, head_eye_idx])
    neutral[head_eye_idx] = np.degrees(neutral[head_eye_idx])
    gt = gt - neutral[None, :]

    ours = _channel_matrix(ours_rows)

    frame_mapping = _blendshape_row_to_video_frame(frame_log_csv)
    if len(frame_mapping) != len(gt):
        raise RuntimeError(f"frame_log.csv has {len(frame_mapping)} B-records, raw.csv has {len(gt)} rows, mismatch")

    valid = frame_mapping < len(ours)
    return ours[frame_mapping[valid]], gt[valid]


def measure(run_dir: Path) -> None:
    ours_aligned, gt_aligned = aligned_channel_data(run_dir)
    print(f"{len(gt_aligned)} aligned frames\n")

    results = []
    for i, name in enumerate(ALL_COMPARABLE_COLUMNS):
        gt_col, ours_col = gt_aligned[:, i], ours_aligned[:, i]
        rms = float(np.sqrt(np.mean((gt_col - ours_col) ** 2)))
        if gt_col.std() < 1e-6 or ours_col.std() < 1e-6:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(gt_col, ours_col)[0, 1])
        results.append((name, corr, rms))

    results.sort(key=lambda r: (np.nan_to_num(r[1], nan=1.0), -r[2]))
    print(f"{'channel':<22} {'corr':>8} {'rms':>8}")
    for name, corr, rms in results:
        print(f"{name:<22} {corr:>8.3f} {rms:>8.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <run_dir>")
        sys.exit(1)
    measure(Path(sys.argv[1]))
